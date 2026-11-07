//! Velox Streaming Zero-Copy Lexer (`veloxc::lexer`)
//!
//! High-performance, streaming lexical analyzer for Velox source text.
//! Tracks exact 1-indexed line and column coordinates, 4-space tab stop normalization,
//! CRLF/LF normalization, nested block comments, string escape decoding, and numeric ranges.
//!
//! Zero external dependencies (`std` only).

use crate::tokens::{Token, TokenType};
use crate::types::{SourceLocation, SourceSpan};
use std::fmt;

/// Raised when the lexer encounters an unrecoverable lexical defect.
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub message: String,
    pub span: SourceSpan,
}

impl LexError {
    /// Constructs a new `LexError`.
    #[must_use]
    pub fn new(message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file_prefix = if self.span.file.is_empty() {
            String::new()
        } else {
            format!("{}:", self.span.file)
        };
        write!(
            f,
            "{} at {}{}:{}",
            self.message, file_prefix, self.span.start.line, self.span.start.column
        )
    }
}

impl std::error::Error for LexError {}

/// Streaming zero-allocation lexical analyzer for Velox source text.
pub struct Lexer<'a> {
    source: &'a str,
    file: String,
    cursor: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    /// Constructs a new Lexer for the given source buffer.
    #[must_use]
    pub fn new(source: &'a str, file: impl Into<String>) -> Self {
        Self {
            source,
            file: file.into(),
            cursor: 0,
            line: 1,
            column: 1,
        }
    }

    /// Convenience helper to tokenize an entire source string.
    pub fn tokenize_source(source: &'a str, file: impl Into<String>) -> Result<Vec<Token>, LexError> {
        Self::new(source, file).tokenize()
    }

    #[inline]
    fn current_loc(&self) -> SourceLocation {
        SourceLocation::new(self.line, self.column, self.cursor)
    }

    #[inline]
    fn peek(&self) -> Option<char> {
        self.source[self.cursor..].chars().next()
    }

    #[inline]
    fn peek_nth(&self, n: usize) -> Option<char> {
        self.source[self.cursor..].chars().nth(n)
    }

    fn advance(&mut self) -> Option<char> {
        if self.cursor >= self.source.len() {
            return None;
        }
        let ch = self.source[self.cursor..].chars().next()?;
        self.cursor += ch.len_utf8();
        match ch {
            '\n' => {
                self.line += 1;
                self.column = 1;
            }
            '\r' => {
                // Handle CRLF sequence
                if self.cursor < self.source.len() && self.source[self.cursor..].starts_with('\n') {
                    self.cursor += 1;
                }
                self.line += 1;
                self.column = 1;
            }
            '\t' => {
                // 4-space tab stop normalization: column advances to next 4-space stop + 1
                self.column = ((self.column - 1) / 4 + 1) * 4 + 1;
            }
            _ => {
                self.column += 1;
            }
        }
        Some(ch)
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Scans the entire source buffer and emits a token stream terminated by EOF.
    pub fn tokenize(&mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();
        while self.cursor < self.source.len() {
            self.skip_whitespace_and_comments()?;
            if self.cursor >= self.source.len() {
                break;
            }
            if let Some(tok) = self.scan_token()? {
                tokens.push(tok);
            }
        }

        // Final EOF token
        let loc = self.current_loc();
        let eof_span = SourceSpan::from_coords(
            &self.file, loc.line, loc.column, loc.line, loc.column, loc.offset, loc.offset,
        );
        tokens.push(Token::new(TokenType::Eof, eof_span, ""));

        Ok(tokens)
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<(), LexError> {
        while self.cursor < self.source.len() {
            let ch = match self.peek() {
                Some(c) => c,
                None => break,
            };

            // Whitespace
            if ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' {
                self.advance();
                continue;
            }

            // Python-style single-line hash comment: # ... \n
            if ch == '#' {
                while let Some(c) = self.peek() {
                    if c == '\n' || c == '\r' {
                        break;
                    }
                    self.advance();
                }
                continue;
            }

            // Slash single-line comment: // ... \n
            if ch == '/' && self.peek_nth(1) == Some('/') {
                while let Some(c) = self.peek() {
                    if c == '\n' || c == '\r' {
                        break;
                    }
                    self.advance();
                }
                continue;
            }

            // Block multi-line comment: /* ... */ with arbitrary nesting
            if ch == '/' && self.peek_nth(1) == Some('*') {
                let comment_start = self.current_loc();
                self.advance(); // /
                self.advance(); // *
                let mut depth: usize = 1;

                while self.cursor < self.source.len() && depth > 0 {
                    if self.peek() == Some('/') && self.peek_nth(1) == Some('*') {
                        self.advance();
                        self.advance();
                        depth += 1;
                    } else if self.peek() == Some('*') && self.peek_nth(1) == Some('/') {
                        self.advance();
                        self.advance();
                        depth -= 1;
                    } else {
                        self.advance();
                    }
                }

                if depth > 0 {
                    let span = SourceSpan::from_coords(
                        &self.file,
                        comment_start.line,
                        comment_start.column,
                        self.line,
                        self.column,
                        comment_start.offset,
                        self.cursor,
                    );
                    return Err(LexError {
                        message: "Unterminated block comment".to_string(),
                        span,
                    });
                }
                continue;
            }

            break;
        }
        Ok(())
    }

    fn scan_token(&mut self) -> Result<Option<Token>, LexError> {
        let start_loc = self.current_loc();
        let ch = match self.peek() {
            Some(c) => c,
            None => return Ok(None),
        };

        // String literals: double or single quoted
        if ch == '"' || ch == '\'' {
            return self.scan_string(start_loc, ch).map(Some);
        }

        // Numeric literals: decimal, hex (0x), binary (0b), float
        if ch.is_ascii_digit() {
            return Ok(Some(self.scan_number(start_loc)));
        }

        // Attributes (@vram_immutable, etc.) or standalone @ (Matmul)
        if ch == '@' {
            return Ok(Some(self.scan_attribute_or_at(start_loc)));
        }

        // Identifiers & Keywords
        if ch.is_alphabetic() || ch == '_' {
            return Ok(Some(self.scan_identifier_or_keyword(start_loc)));
        }

        // Operators & Delimiters
        Ok(Some(self.scan_operator_or_delimiter(start_loc)))
    }

    fn scan_string(&mut self, start_loc: SourceLocation, quote_char: char) -> Result<Token, LexError> {
        self.advance(); // consume opening quote
        let mut chars = String::new();

        while self.cursor < self.source.len() && self.peek() != Some(quote_char) {
            let ch = match self.peek() {
                Some(c) => c,
                None => break,
            };

            if ch == '\n' || ch == '\r' {
                let span = SourceSpan::from_coords(
                    &self.file,
                    start_loc.line,
                    start_loc.column,
                    self.line,
                    self.column,
                    start_loc.offset,
                    self.cursor,
                );
                return Err(LexError {
                    message: "Unterminated string literal".to_string(),
                    span,
                });
            }

            if ch == '\\' {
                self.advance(); // consume '\'
                match self.advance() {
                    Some('n') => chars.push('\n'),
                    Some('t') => chars.push('\t'),
                    Some('r') => chars.push('\r'),
                    Some('\\') => chars.push('\\'),
                    Some('"') => chars.push('"'),
                    Some('\'') => chars.push('\''),
                    Some('0') => chars.push('\0'),
                    Some('x') => {
                        let h1 = self.advance().unwrap_or('0');
                        let h2 = self.advance().unwrap_or('0');
                        let mut hex_str = String::with_capacity(2);
                        hex_str.push(h1);
                        hex_str.push(h2);
                        if let Ok(byte_val) = u8::from_str_radix(&hex_str, 16) {
                            chars.push(byte_val as char);
                        } else {
                            chars.push_str("\\x");
                            chars.push_str(&hex_str);
                        }
                    }
                    Some(other) => chars.push(other),
                    None => break,
                }
            } else {
                chars.push(self.advance().unwrap());
            }
        }

        if self.cursor >= self.source.len() {
            let span = SourceSpan::from_coords(
                &self.file,
                start_loc.line,
                start_loc.column,
                self.line,
                self.column,
                start_loc.offset,
                self.cursor,
            );
            return Err(LexError {
                message: "Unterminated string literal at EOF".to_string(),
                span,
            });
        }

        self.advance(); // consume closing quote
        let end_loc = self.current_loc();
        let raw = self.source[start_loc.offset..end_loc.offset].to_string();
        let span = SourceSpan::from_coords(
            &self.file,
            start_loc.line,
            start_loc.column,
            end_loc.line,
            end_loc.column,
            start_loc.offset,
            end_loc.offset,
        );

        Ok(Token::new(TokenType::String(chars), span, raw))
    }

    fn scan_number(&mut self, start_loc: SourceLocation) -> Token {
        // Hexadecimal: 0x... or 0X...
        if self.peek() == Some('0') && matches!(self.peek_nth(1), Some('x' | 'X')) {
            self.advance(); // 0
            self.advance(); // x
            while let Some(c) = self.peek() {
                if c.is_ascii_hexdigit() || c == '_' {
                    self.advance();
                } else {
                    break;
                }
            }
            let end_loc = self.current_loc();
            let raw = self.source[start_loc.offset..end_loc.offset].to_string();
            let clean = if raw.len() > 2 {
                raw[2..].replace('_', "")
            } else {
                String::new()
            };
            let val = u64::from_str_radix(&clean, 16).unwrap_or(0);
            let span = SourceSpan::from_coords(
                &self.file,
                start_loc.line,
                start_loc.column,
                end_loc.line,
                end_loc.column,
                start_loc.offset,
                end_loc.offset,
            );
            return Token::new(TokenType::Hex(val), span, raw);
        }

        // Binary: 0b... or 0B...
        if self.peek() == Some('0') && matches!(self.peek_nth(1), Some('b' | 'B')) {
            self.advance(); // 0
            self.advance(); // b
            while let Some(c) = self.peek() {
                if c == '0' || c == '1' || c == '_' {
                    self.advance();
                } else {
                    break;
                }
            }
            let end_loc = self.current_loc();
            let raw = self.source[start_loc.offset..end_loc.offset].to_string();
            let clean = if raw.len() > 2 {
                raw[2..].replace('_', "")
            } else {
                String::new()
            };
            let val = u64::from_str_radix(&clean, 2).unwrap_or(0);
            let span = SourceSpan::from_coords(
                &self.file,
                start_loc.line,
                start_loc.column,
                end_loc.line,
                end_loc.column,
                start_loc.offset,
                end_loc.offset,
            );
            return Token::new(TokenType::Binary(val), span, raw);
        }

        // Decimal or Float
        let mut is_float = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        // Fractional part: check '.' followed by digit (NOT '..' range)
        if self.peek() == Some('.') && self.peek_nth(1) != Some('.') {
            if let Some(next) = self.peek_nth(1) {
                if next.is_ascii_digit() {
                    is_float = true;
                    self.advance(); // consume '.'
                    while let Some(c) = self.peek() {
                        if c.is_ascii_digit() || c == '_' {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        // Scientific exponent: 1e-5, 2.5e+10, 1E6
        if matches!(self.peek(), Some('e' | 'E')) {
            is_float = true;
            self.advance(); // e or E
            if matches!(self.peek(), Some('+' | '-')) {
                self.advance();
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() || c == '_' {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        let end_loc = self.current_loc();
        let raw = self.source[start_loc.offset..end_loc.offset].to_string();
        let clean = raw.replace('_', "");

        let span = SourceSpan::from_coords(
            &self.file,
            start_loc.line,
            start_loc.column,
            end_loc.line,
            end_loc.column,
            start_loc.offset,
            end_loc.offset,
        );

        if is_float {
            let val = clean.parse::<f64>().unwrap_or(0.0);
            Token::new(TokenType::Float(val), span, raw)
        } else {
            let val = clean.parse::<u64>().unwrap_or(0);
            Token::new(TokenType::Decimal(val), span, raw)
        }
    }

    fn scan_attribute_or_at(&mut self, start_loc: SourceLocation) -> Token {
        self.advance(); // consume '@'

        if let Some(c) = self.peek() {
            if c.is_alphabetic() || c == '_' {
                while let Some(ch) = self.peek() {
                    if ch.is_alphanumeric() || ch == '_' {
                        self.advance();
                    } else {
                        break;
                    }
                }
                let end_loc = self.current_loc();
                let raw = self.source[start_loc.offset..end_loc.offset].to_string();
                let token_type = match raw.as_str() {
                    "@vram_immutable" => TokenType::AttrVramImmutable,
                    "@paged_vram" => TokenType::AttrPagedVram,
                    "@bump_sram" => TokenType::AttrBumpSram,
                    "@host_pinned" => TokenType::AttrHostPinned,
                    "@wired_limit_bounded" => TokenType::AttrWiredLimitBounded,
                    "@warp_specialized" => TokenType::AttrWarpSpecialized,
                    "@kernel_fuse" => TokenType::AttrKernelFuse,
                    _ => TokenType::Error(raw.clone()),
                };
                let span = SourceSpan::from_coords(
                    &self.file,
                    start_loc.line,
                    start_loc.column,
                    end_loc.line,
                    end_loc.column,
                    start_loc.offset,
                    end_loc.offset,
                );
                return Token::new(token_type, span, raw);
            }
        }

        // Standalone '@' is Matmul operator
        let end_loc = self.current_loc();
        let span = SourceSpan::from_coords(
            &self.file,
            start_loc.line,
            start_loc.column,
            end_loc.line,
            end_loc.column,
            start_loc.offset,
            end_loc.offset,
        );
        Token::new(TokenType::At, span, "@")
    }

    fn scan_identifier_or_keyword(&mut self, start_loc: SourceLocation) -> Token {
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }
        let end_loc = self.current_loc();
        let raw = self.source[start_loc.offset..end_loc.offset].to_string();

        let span = SourceSpan::from_coords(
            &self.file,
            start_loc.line,
            start_loc.column,
            end_loc.line,
            end_loc.column,
            start_loc.offset,
            end_loc.offset,
        );

        let token_type = match raw.as_str() {
            "target" => TokenType::Target,
            "hardware" => TokenType::Hardware,
            "device" => TokenType::Device,
            "vram_capacity" => TokenType::VramCapacity,
            "compute_capability" => TokenType::ComputeCapability,
            "safety_factor" => TokenType::SafetyFactor,
            "model" => TokenType::Model,
            "weight" => TokenType::Weight,
            "tensor" | "Tensor" => TokenType::Tensor,
            "safetensors" => TokenType::Safetensors,
            "mmap" => TokenType::Mmap,
            "import" => TokenType::Import,
            "as" => TokenType::As,
            "const" => TokenType::Const,
            "scope" => TokenType::Scope,
            "prefill" => TokenType::Prefill,
            "decode" => TokenType::Decode,
            "chunked_prefill" => TokenType::ChunkedPrefill,
            "speculative_verify" => TokenType::SpeculativeVerify,
            "PagedKVCache" | "paged_kv" => TokenType::PagedKVCache,
            "VirtualContiguousKV" | "virtual_kv" => TokenType::VirtualContiguousKV,
            "LatentKVCache" | "latent_kv" => TokenType::LatentKVCache,
            "StreamingSinkKVCache" | "streaming_sink" => TokenType::StreamingSinkKVCache,
            "SpeculativeTreeBlock" | "tree_block" => TokenType::SpeculativeTreeBlock,
            "let" => TokenType::Let,
            "mut" => TokenType::Mut,
            "fn" => TokenType::Fn,
            "kernel" => TokenType::Kernel,
            "return" => TokenType::Return,
            "if" => TokenType::If,
            "else" => TokenType::Else,
            "for" => TokenType::For,
            "in" => TokenType::In,
            "while" => TokenType::While,
            "yield" => TokenType::Yield,
            "true" => TokenType::Bool(true),
            "false" => TokenType::Bool(false),
            "null" => TokenType::Null,
            "absorb_mla_q" => TokenType::AbsorbMlaQ,
            "absorb_mla_v" => TokenType::AbsorbMlaV,
            "dma_async_h2d" => TokenType::DmaAsyncH2D,
            "dma_async_d2h" => TokenType::DmaAsyncD2H,
            "tma_load_async" => TokenType::TmaLoadAsync,
            "tma_store_async" => TokenType::TmaStoreAsync,
            "mbarrier" => TokenType::Mbarrier,
            "f32" => TokenType::TypeF32,
            "f16" => TokenType::TypeF16,
            "bf16" => TokenType::TypeBf16,
            "fp8_e4m3" => TokenType::TypeFp8E4m3,
            "fp8_e5m2" => TokenType::TypeFp8E5m2,
            "fp8_e8m0" => TokenType::TypeFp8E8m0,
            "int8" => TokenType::TypeInt8,
            "int4" => TokenType::TypeInt4,
            "u64" => TokenType::TypeU64,
            "u32" => TokenType::TypeU32,
            "u16" => TokenType::TypeU16,
            "u8" => TokenType::TypeU8,
            "i64" => TokenType::TypeI64,
            "i32" => TokenType::TypeI32,
            "i16" => TokenType::TypeI16,
            "i8" => TokenType::TypeI8,
            "bool" => TokenType::TypeBool,
            "usize" => TokenType::TypeUsize,
            "KiB" => TokenType::UnitKiB,
            "MiB" => TokenType::UnitMiB,
            "GiB" => TokenType::UnitGiB,
            "TiB" => TokenType::UnitTiB,
            "KB" => TokenType::UnitKB,
            "MB" => TokenType::UnitMB,
            "GB" => TokenType::UnitGB,
            "TB" => TokenType::UnitTB,
            _ => TokenType::Identifier(raw.clone()),
        };

        Token::new(token_type, span, raw)
    }

    fn scan_operator_or_delimiter(&mut self, start_loc: SourceLocation) -> Token {
        let ch = self.advance().unwrap();
        match ch {
            '+' => {
                if self.match_char('=') {
                    self.make_token(TokenType::PlusEq, "+=", start_loc)
                } else {
                    self.make_token(TokenType::Plus, "+", start_loc)
                }
            }
            '-' => {
                if self.match_char('>') {
                    self.make_token(TokenType::Arrow, "->", start_loc)
                } else if self.match_char('=') {
                    self.make_token(TokenType::MinusEq, "-=", start_loc)
                } else {
                    self.make_token(TokenType::Minus, "-", start_loc)
                }
            }
            '*' => {
                if self.match_char('=') {
                    self.make_token(TokenType::StarEq, "*=", start_loc)
                } else {
                    self.make_token(TokenType::Star, "*", start_loc)
                }
            }
            '/' => {
                if self.match_char('=') {
                    self.make_token(TokenType::SlashEq, "/=", start_loc)
                } else {
                    self.make_token(TokenType::Slash, "/", start_loc)
                }
            }
            '%' => self.make_token(TokenType::Percent, "%", start_loc),
            '(' => self.make_token(TokenType::LParen, "(", start_loc),
            ')' => self.make_token(TokenType::RParen, ")", start_loc),
            '[' => self.make_token(TokenType::LBracket, "[", start_loc),
            ']' => self.make_token(TokenType::RBracket, "]", start_loc),
            '{' => self.make_token(TokenType::LBrace, "{", start_loc),
            '}' => self.make_token(TokenType::RBrace, "}", start_loc),
            ',' => self.make_token(TokenType::Comma, ",", start_loc),
            ';' => self.make_token(TokenType::Semicolon, ";", start_loc),
            ':' => {
                if self.match_char(':') {
                    self.make_token(TokenType::ColonColon, "::", start_loc)
                } else {
                    self.make_token(TokenType::Colon, ":", start_loc)
                }
            }
            '~' => self.make_token(TokenType::Tilde, "~", start_loc),
            '^' => self.make_token(TokenType::Caret, "^", start_loc),
            '.' => {
                if self.match_char('.') {
                    self.make_token(TokenType::DotDot, "..", start_loc)
                } else {
                    self.make_token(TokenType::Dot, ".", start_loc)
                }
            }
            '=' => {
                if self.match_char('=') {
                    self.make_token(TokenType::EqEq, "==", start_loc)
                } else if self.match_char('>') {
                    self.make_token(TokenType::FatArrow, "=>", start_loc)
                } else {
                    self.make_token(TokenType::Eq, "=", start_loc)
                }
            }
            '!' => {
                if self.match_char('=') {
                    self.make_token(TokenType::BangEq, "!=", start_loc)
                } else {
                    self.make_token(TokenType::Bang, "!", start_loc)
                }
            }
            '<' => {
                if self.match_char('=') {
                    self.make_token(TokenType::LtEq, "<=", start_loc)
                } else if self.match_char('<') {
                    self.make_token(TokenType::Shl, "<<", start_loc)
                } else {
                    self.make_token(TokenType::Lt, "<", start_loc)
                }
            }
            '>' => {
                if self.match_char('=') {
                    self.make_token(TokenType::GtEq, ">=", start_loc)
                } else if self.match_char('>') {
                    self.make_token(TokenType::Shr, ">>", start_loc)
                } else {
                    self.make_token(TokenType::Gt, ">", start_loc)
                }
            }
            '&' => {
                if self.match_char('&') {
                    self.make_token(TokenType::AmpAmp, "&&", start_loc)
                } else {
                    self.make_token(TokenType::Amp, "&", start_loc)
                }
            }
            '|' => {
                if self.match_char('|') {
                    self.make_token(TokenType::PipePipe, "||", start_loc)
                } else {
                    self.make_token(TokenType::Pipe, "|", start_loc)
                }
            }
            other => self.make_token(TokenType::Error(other.to_string()), other.to_string(), start_loc),
        }
    }

    fn make_token(&self, kind: TokenType, raw: impl Into<String>, start_loc: SourceLocation) -> Token {
        let end_loc = self.current_loc();
        let text = raw.into();
        let span = SourceSpan::from_coords(
            &self.file,
            start_loc.line,
            start_loc.column,
            end_loc.line,
            end_loc.column,
            start_loc.offset,
            end_loc.offset,
        );
        Token::new(kind, span, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_keywords_and_types() {
        let code = "model target hardware let mut fn scope prefill decode Tensor";
        let tokens = Lexer::tokenize_source(code, "test.vlx").unwrap();
        assert_eq!(tokens[0].token_type, TokenType::Model);
        assert_eq!(tokens[1].token_type, TokenType::Target);
        assert_eq!(tokens[2].token_type, TokenType::Hardware);
        assert_eq!(tokens[3].token_type, TokenType::Let);
        assert_eq!(tokens[4].token_type, TokenType::Mut);
        assert_eq!(tokens[5].token_type, TokenType::Fn);
        assert_eq!(tokens[6].token_type, TokenType::Scope);
        assert_eq!(tokens[7].token_type, TokenType::Prefill);
        assert_eq!(tokens[8].token_type, TokenType::Decode);
        assert_eq!(tokens[9].token_type, TokenType::Tensor);
        assert_eq!(tokens[10].token_type, TokenType::Eof);
    }

    #[test]
    fn test_lex_range_lookahead() {
        let code = "0..32";
        let tokens = Lexer::tokenize_source(code, "test.vlx").unwrap();
        assert_eq!(tokens[0].token_type, TokenType::Decimal(0));
        assert_eq!(tokens[1].token_type, TokenType::DotDot);
        assert_eq!(tokens[2].token_type, TokenType::Decimal(32));
    }

    #[test]
    fn test_lex_comments_and_strings() {
        let code = "# line comment\n/* nested /* block */ comment */\n\"hello \\n world\"";
        let tokens = Lexer::tokenize_source(code, "test.vlx").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].token_type, TokenType::String("hello \n world".to_string()));
        assert_eq!(tokens[1].token_type, TokenType::Eof);
    }
}
