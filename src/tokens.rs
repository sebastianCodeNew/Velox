//! Velox Compiler Tokens (`veloxc::tokens`)
//!
//! Lexical token classifications, keyword catalogs, operator enumerations,
//! and strongly-typed token representation.
//!
//! Zero external dependencies (`std` only).

use crate::types::SourceSpan;
use std::fmt;

/// Exhaustive lexical token classification for the Velox programming language.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // -------------------------------------------------------------------------
    // 1. Structural & Control Tokens
    // -------------------------------------------------------------------------
    Eof,
    Error(String),

    // -------------------------------------------------------------------------
    // 2. Literals & Identifiers
    // -------------------------------------------------------------------------
    Identifier(String),
    Decimal(u64),
    Hex(u64),
    Binary(u64),
    Float(f64),
    String(String),
    Bool(bool),

    // -------------------------------------------------------------------------
    // 3. Keywords: System & Architecture
    // -------------------------------------------------------------------------
    Target,
    Hardware,
    Device,
    VramCapacity,
    ComputeCapability,
    SafetyFactor,
    Model,
    Weight,
    Tensor,
    Safetensors,
    Mmap,
    Import,
    As,
    Const,

    // -------------------------------------------------------------------------
    // 4. Keywords: Execution Scopes
    // -------------------------------------------------------------------------
    Scope,
    Prefill,
    Decode,
    ChunkedPrefill,
    SpeculativeVerify,

    // -------------------------------------------------------------------------
    // 5. Keywords: Key-Value Cache Architectures
    // -------------------------------------------------------------------------
    PagedKVCache,
    VirtualContiguousKV,
    LatentKVCache,
    StreamingSinkKVCache,
    SpeculativeTreeBlock,

    // -------------------------------------------------------------------------
    // 6. Keywords: Declarations & Control Flow
    // -------------------------------------------------------------------------
    Let,
    Mut,
    Fn,
    Kernel,
    Return,
    If,
    Else,
    For,
    In,
    While,
    Yield,
    Null,

    // -------------------------------------------------------------------------
    // 7. Keywords: Serving & Async Hardware Primitives
    // -------------------------------------------------------------------------
    AbsorbMlaQ,
    AbsorbMlaV,
    DmaAsyncH2D,
    DmaAsyncD2H,
    TmaLoadAsync,
    TmaStoreAsync,
    Mbarrier,

    // -------------------------------------------------------------------------
    // 8. Memory Tier Qualifiers & Scope Attributes (Leading '@')
    // -------------------------------------------------------------------------
    AttrVramImmutable,
    AttrPagedVram,
    AttrBumpSram,
    AttrHostPinned,
    AttrWiredLimitBounded,
    AttrWarpSpecialized,
    AttrKernelFuse,

    // -------------------------------------------------------------------------
    // 9. Scalar & Integer Type Identifiers
    // -------------------------------------------------------------------------
    TypeF32,
    TypeF16,
    TypeBf16,
    TypeFp8E4m3,
    TypeFp8E5m2,
    TypeFp8E8m0,
    TypeInt8,
    TypeInt4,
    TypeU64,
    TypeU32,
    TypeU16,
    TypeU8,
    TypeI64,
    TypeI32,
    TypeI16,
    TypeI8,
    TypeBool,
    TypeUsize,

    // -------------------------------------------------------------------------
    // 10. Memory Capacity Multipliers
    // -------------------------------------------------------------------------
    UnitKiB,
    UnitMiB,
    UnitGiB,
    UnitTiB,
    UnitKB,
    UnitMB,
    UnitGB,
    UnitTB,

    // -------------------------------------------------------------------------
    // 11. Operators (14-Level Precedence Hierarchy)
    // -------------------------------------------------------------------------
    At,         // @  (Matrix Multiplication, Level 3)
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Shl,        // <<
    Shr,        // >>
    Amp,        // &
    Caret,      // ^
    Pipe,       // |
    Tilde,      // ~
    Bang,       // !
    AmpAmp,     // &&
    PipePipe,   // ||
    Eq,         // =
    EqEq,       // ==
    BangEq,     // !=
    Lt,         // <
    LtEq,       // <=
    Gt,         // >
    GtEq,       // >=
    PlusEq,     // +=
    MinusEq,    // -=
    StarEq,     // *=
    SlashEq,    // /=
    Arrow,      // ->
    FatArrow,   // =>
    ColonColon, // ::

    // -------------------------------------------------------------------------
    // 12. Delimiters & Punctuation
    // -------------------------------------------------------------------------
    Dot,       // .
    DotDot,    // ..
    Comma,     // ,
    Colon,     // :
    Semicolon, // ;
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    LBrace,    // {
    RBrace,    // }
}

impl TokenType {
    /// Returns a human-readable display name for the token type.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Eof => "EOF",
            Self::Error(_) => "ERROR",
            Self::Identifier(_) => "IDENTIFIER",
            Self::Decimal(_) => "DECIMAL_LITERAL",
            Self::Hex(_) => "HEX_LITERAL",
            Self::Binary(_) => "BIN_LITERAL",
            Self::Float(_) => "FLOAT_LITERAL",
            Self::String(_) => "STRING_LITERAL",
            Self::Bool(_) => "BOOLEAN_LITERAL",

            Self::Target => "target",
            Self::Hardware => "hardware",
            Self::Device => "device",
            Self::VramCapacity => "vram_capacity",
            Self::ComputeCapability => "compute_capability",
            Self::SafetyFactor => "safety_factor",
            Self::Model => "model",
            Self::Weight => "weight",
            Self::Tensor => "Tensor",
            Self::Safetensors => "safetensors",
            Self::Mmap => "mmap",
            Self::Import => "import",
            Self::As => "as",
            Self::Const => "const",

            Self::Scope => "scope",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::ChunkedPrefill => "chunked_prefill",
            Self::SpeculativeVerify => "speculative_verify",

            Self::PagedKVCache => "PagedKVCache",
            Self::VirtualContiguousKV => "VirtualContiguousKV",
            Self::LatentKVCache => "LatentKVCache",
            Self::StreamingSinkKVCache => "StreamingSinkKVCache",
            Self::SpeculativeTreeBlock => "SpeculativeTreeBlock",

            Self::Let => "let",
            Self::Mut => "mut",
            Self::Fn => "fn",
            Self::Kernel => "kernel",
            Self::Return => "return",
            Self::If => "if",
            Self::Else => "else",
            Self::For => "for",
            Self::In => "in",
            Self::While => "while",
            Self::Yield => "yield",
            Self::Null => "null",

            Self::AbsorbMlaQ => "absorb_mla_q",
            Self::AbsorbMlaV => "absorb_mla_v",
            Self::DmaAsyncH2D => "dma_async_h2d",
            Self::DmaAsyncD2H => "dma_async_d2h",
            Self::TmaLoadAsync => "tma_load_async",
            Self::TmaStoreAsync => "tma_store_async",
            Self::Mbarrier => "mbarrier",

            Self::AttrVramImmutable => "@vram_immutable",
            Self::AttrPagedVram => "@paged_vram",
            Self::AttrBumpSram => "@bump_sram",
            Self::AttrHostPinned => "@host_pinned",
            Self::AttrWiredLimitBounded => "@wired_limit_bounded",
            Self::AttrWarpSpecialized => "@warp_specialized",
            Self::AttrKernelFuse => "@kernel_fuse",

            Self::TypeF32 => "f32",
            Self::TypeF16 => "f16",
            Self::TypeBf16 => "bf16",
            Self::TypeFp8E4m3 => "fp8_e4m3",
            Self::TypeFp8E5m2 => "fp8_e5m2",
            Self::TypeFp8E8m0 => "fp8_e8m0",
            Self::TypeInt8 => "int8",
            Self::TypeInt4 => "int4",
            Self::TypeU64 => "u64",
            Self::TypeU32 => "u32",
            Self::TypeU16 => "u16",
            Self::TypeU8 => "u8",
            Self::TypeI64 => "i64",
            Self::TypeI32 => "i32",
            Self::TypeI16 => "i16",
            Self::TypeI8 => "i8",
            Self::TypeBool => "bool",
            Self::TypeUsize => "usize",

            Self::UnitKiB => "KiB",
            Self::UnitMiB => "MiB",
            Self::UnitGiB => "GiB",
            Self::UnitTiB => "TiB",
            Self::UnitKB => "KB",
            Self::UnitMB => "MB",
            Self::UnitGB => "GB",
            Self::UnitTB => "TB",

            Self::At => "@",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Star => "*",
            Self::Slash => "/",
            Self::Percent => "%",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::Amp => "&",
            Self::Caret => "^",
            Self::Pipe => "|",
            Self::Tilde => "~",
            Self::Bang => "!",
            Self::AmpAmp => "&&",
            Self::PipePipe => "||",
            Self::Eq => "=",
            Self::EqEq => "==",
            Self::BangEq => "!=",
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
            Self::PlusEq => "+=",
            Self::MinusEq => "-=",
            Self::StarEq => "*=",
            Self::SlashEq => "/=",
            Self::Arrow => "->",
            Self::FatArrow => "=>",
            Self::ColonColon => "::",

            Self::Dot => ".",
            Self::DotDot => "..",
            Self::Comma => ",",
            Self::Colon => ":",
            Self::Semicolon => ";",
            Self::LParen => "(",
            Self::RParen => ")",
            Self::LBracket => "[",
            Self::RBracket => "]",
            Self::LBrace => "{",
            Self::RBrace => "}",
        }
    }

    /// Returns `true` if this token represents an identifier.
    #[inline]
    #[must_use]
    pub const fn is_identifier(&self) -> bool {
        matches!(self, Self::Identifier(_))
    }

    /// Returns `true` if this token represents a literal constant.
    #[inline]
    #[must_use]
    pub const fn is_literal(&self) -> bool {
        matches!(
            self,
            Self::Decimal(_)
                | Self::Hex(_)
                | Self::Binary(_)
                | Self::Float(_)
                | Self::String(_)
                | Self::Bool(_)
        )
    }

    /// Returns `true` if this token is a keyword.
    #[must_use]
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            Self::Target
                | Self::Hardware
                | Self::Device
                | Self::VramCapacity
                | Self::ComputeCapability
                | Self::SafetyFactor
                | Self::Model
                | Self::Weight
                | Self::Tensor
                | Self::Safetensors
                | Self::Mmap
                | Self::Import
                | Self::As
                | Self::Const
                | Self::Scope
                | Self::Prefill
                | Self::Decode
                | Self::ChunkedPrefill
                | Self::SpeculativeVerify
                | Self::PagedKVCache
                | Self::VirtualContiguousKV
                | Self::LatentKVCache
                | Self::StreamingSinkKVCache
                | Self::SpeculativeTreeBlock
                | Self::Let
                | Self::Mut
                | Self::Fn
                | Self::Kernel
                | Self::Return
                | Self::If
                | Self::Else
                | Self::For
                | Self::In
                | Self::While
                | Self::Yield
                | Self::Null
                | Self::AbsorbMlaQ
                | Self::AbsorbMlaV
                | Self::DmaAsyncH2D
                | Self::DmaAsyncD2H
                | Self::TmaLoadAsync
                | Self::TmaStoreAsync
                | Self::Mbarrier
        )
    }

    /// Returns `true` if this token is an operator.
    #[must_use]
    pub fn is_operator(&self) -> bool {
        matches!(
            self,
            Self::At
                | Self::Plus
                | Self::Minus
                | Self::Star
                | Self::Slash
                | Self::Percent
                | Self::Shl
                | Self::Shr
                | Self::Amp
                | Self::Caret
                | Self::Pipe
                | Self::Tilde
                | Self::Bang
                | Self::AmpAmp
                | Self::PipePipe
                | Self::Eq
                | Self::EqEq
                | Self::BangEq
                | Self::Lt
                | Self::LtEq
                | Self::Gt
                | Self::GtEq
                | Self::PlusEq
                | Self::MinusEq
                | Self::StarEq
                | Self::SlashEq
                | Self::Arrow
                | Self::FatArrow
                | Self::ColonColon
        )
    }
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identifier(s) => write!(f, "Identifier({s})"),
            Self::Decimal(n) => write!(f, "Decimal({n})"),
            Self::Hex(n) => write!(f, "Hex(0x{n:X})"),
            Self::Binary(n) => write!(f, "Binary(0b{n:b})"),
            Self::Float(v) => write!(f, "Float({v})"),
            Self::String(s) => write!(f, "String({s:?})"),
            Self::Bool(b) => write!(f, "Bool({b})"),
            Self::Error(e) => write!(f, "Error({e})"),
            other => write!(f, "{}", other.name()),
        }
    }
}

/// Strongly-typed lexical token representation with source span and raw text.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// Categorical classification and semantic value payload.
    pub token_type: TokenType,
    /// Bounding coordinate span in the source translation unit.
    pub span: SourceSpan,
    /// Verbatim text slice from source buffer.
    pub text: String,
}

impl Token {
    /// Constructs a new `Token`.
    #[must_use]
    pub fn new(token_type: TokenType, span: SourceSpan, text: impl Into<String>) -> Self {
        Self {
            token_type,
            span,
            text: text.into(),
        }
    }

    /// Returns `true` if this token matches the specified token type.
    #[inline]
    #[must_use]
    pub fn is(&self, expected: &TokenType) -> bool {
        &self.token_type == expected
    }

    /// Returns `true` if this token is of kind `EOF`.
    #[inline]
    #[must_use]
    pub fn is_eof(&self) -> bool {
        matches!(self.token_type, TokenType::Eof)
    }

    /// Returns `true` if this token is of kind `Error`.
    #[inline]
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self.token_type, TokenType::Error(_))
    }

    /// Returns `true` if this token is an identifier.
    #[inline]
    #[must_use]
    pub fn is_identifier(&self) -> bool {
        self.token_type.is_identifier()
    }

    /// Returns `true` if this token is a literal.
    #[inline]
    #[must_use]
    pub fn is_literal(&self) -> bool {
        self.token_type.is_literal()
    }

    /// Returns `true` if this token is a keyword.
    #[inline]
    #[must_use]
    pub fn is_keyword(&self) -> bool {
        self.token_type.is_keyword()
    }

    /// Returns `true` if this token is an operator.
    #[inline]
    #[must_use]
    pub fn is_operator(&self) -> bool {
        self.token_type.is_operator()
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Token({:?}, {:?}, {}:{})",
            self.token_type, self.text, self.span.start.line, self.span.start.column
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SourceLocation;

    #[test]
    fn test_token_creation() {
        let span = SourceSpan::new(
            "test.vlx",
            SourceLocation::new(1, 1, 0),
            SourceLocation::new(1, 4, 3),
        );
        let tok = Token::new(TokenType::Let, span, "let");
        assert!(tok.is(&TokenType::Let));
        assert!(tok.is_keyword());
        assert!(!tok.is_eof());
        assert_eq!(tok.text, "let");
    }

    #[test]
    fn test_token_type_predicates() {
        assert!(TokenType::Identifier("foo".to_string()).is_identifier());
        assert!(TokenType::Decimal(42).is_literal());
        assert!(TokenType::Plus.is_operator());
        assert!(TokenType::Model.is_keyword());
    }
}
