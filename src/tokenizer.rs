//! Velox Zero-Dependency Native Rust BPE Tokenizer (`veloxc::tokenizer`)
//!
//! A 100% pure standard library Rust BPE tokenizer implementation with:
//! - Streaming zero-dependency JSON parser for `tokenizer.json` (~2.1 MB) without serde or serde_json.
//! - Regex-free pre-tokenization Finite State Machine (FSM) matching GPT-2 / Llama style:
//!   - Contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`)
//!   - Alphabetic word runs (` ?\p{L}+`)
//!   - Single digits with individual_digits=true (` ?\p{N}`)
//!   - Punctuation / non-word/non-number runs (` ?[^\s\p{L}\p{N}]+`)
//!   - Whitespace runs (`\s+`)
//! - 256-byte bijective table (`bytes_to_unicode()` and `unicode_to_bytes()`) ensuring zero out-of-vocabulary bytes.
//! - BPE merge engine: fast rank minimization using pair merge rules.
//! - Special tokens handling: `<|im_start|>` (1), `<|im_end|>` (2), `<|endoftext|>` (0).
//! - `StreamingDecoder` with pending byte buffer for multi-byte UTF-8 partial byte sequences.
//!
//! Zero external dependencies (`std` only).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ============================================================================
// Byte-Level Bijective Mapping Table (GPT-2 / HuggingFace ByteLevel Standard)
// ============================================================================

/// Constructs the standard 256-element bijective mapping table from raw bytes to unicode characters,
/// and the inverse mapping from unicode characters back to raw bytes.
///
/// Ensures every possible byte value `0..=255` is mapped to a distinct Unicode character,
/// eliminating any out-of-vocabulary byte errors.
#[must_use]
pub fn build_byte_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::with_capacity(256);
    // ! to ~ (33..=126)
    for b in 33u8..=126u8 {
        bs.push(b);
    }
    // ¡ to ¬ (161..=172)
    for b in 161u8..=172u8 {
        bs.push(b);
    }
    // ® to ÿ (174..=255)
    for b in 174u8..=255u8 {
        bs.push(b);
    }

    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    let mut n = 0u32;
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(char::from_u32(256 + n).expect("valid unicode scalar"));
            n += 1;
        }
    }

    let mut b2u = ['\0'; 256];
    let mut u2b = HashMap::with_capacity(256);
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        b2u[b as usize] = c;
        u2b.insert(c, b);
    }

    (b2u, u2b)
}

// ============================================================================
// Zero-Dependency JSON Scanner for tokenizer.json
// ============================================================================

struct JsonScanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonScanner<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    #[inline]
    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    #[inline]
    fn peek(&mut self) -> Option<u8> {
        self.skip_whitespace();
        if self.pos < self.bytes.len() {
            Some(self.bytes[self.pos])
        } else {
            None
        }
    }

    #[inline]
    fn expect_byte(&mut self, expected: u8) -> Result<(), String> {
        self.skip_whitespace();
        if self.pos >= self.bytes.len() {
            return Err(format!(
                "Unexpected EOF: expected '{}' at offset {}",
                expected as char, self.pos
            ));
        }
        if self.bytes[self.pos] == expected {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!(
                "Expected '{}' at offset {}, found '{}'",
                expected as char, self.pos, self.bytes[self.pos] as char
            ))
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        if self.pos + 4 > self.bytes.len() {
            return Err("Unexpected EOF reading \\u hex escape".to_string());
        }
        let hex_slice = &self.bytes[self.pos..self.pos + 4];
        self.pos += 4;
        let hex_str = std::str::from_utf8(hex_slice)
            .map_err(|e| format!("Invalid hex sequence: {e}"))?;
        u16::from_str_radix(hex_str, 16)
            .map_err(|_| format!("Invalid hex escape '\\u{}'", hex_str))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.skip_whitespace();
        if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'"' {
            return Err(format!("Expected '\"' string start at offset {}", self.pos));
        }
        self.pos += 1;

        let mut s = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(s);
            } else if b == b'\\' {
                self.pos += 1;
                if self.pos >= self.bytes.len() {
                    return Err("Unterminated escape sequence in JSON string".to_string());
                }
                match self.bytes[self.pos] {
                    b'"' => {
                        s.push('"');
                        self.pos += 1;
                    }
                    b'\\' => {
                        s.push('\\');
                        self.pos += 1;
                    }
                    b'/' => {
                        s.push('/');
                        self.pos += 1;
                    }
                    b'b' => {
                        s.push('\x08');
                        self.pos += 1;
                    }
                    b'f' => {
                        s.push('\x0C');
                        self.pos += 1;
                    }
                    b'n' => {
                        s.push('\n');
                        self.pos += 1;
                    }
                    b'r' => {
                        s.push('\r');
                        self.pos += 1;
                    }
                    b't' => {
                        s.push('\t');
                        self.pos += 1;
                    }
                    b'u' => {
                        self.pos += 1;
                        let code = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            // High surrogate, check for low surrogate
                            if self.pos + 2 < self.bytes.len()
                                && self.bytes[self.pos] == b'\\'
                                && self.bytes[self.pos + 1] == b'u'
                            {
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                if (0xDC00..=0xDFFF).contains(&low) {
                                    let scalar = 0x10000
                                        + (((code - 0xD800) as u32) << 10)
                                        + ((low - 0xDC00) as u32);
                                    if let Some(ch) = char::from_u32(scalar) {
                                        s.push(ch);
                                    }
                                } else {
                                    s.push('\u{FFFD}');
                                }
                            } else {
                                s.push('\u{FFFD}');
                            }
                        } else if let Some(ch) = char::from_u32(code as u32) {
                            s.push(ch);
                        } else {
                            s.push('\u{FFFD}');
                        }
                    }
                    other => {
                        s.push(other as char);
                        self.pos += 1;
                    }
                }
            } else {
                // Fast path: consume contiguous chunk of normal characters
                let start = self.pos;
                while self.pos < self.bytes.len()
                    && self.bytes[self.pos] != b'"'
                    && self.bytes[self.pos] != b'\\'
                {
                    self.pos += 1;
                }
                let chunk = std::str::from_utf8(&self.bytes[start..self.pos])
                    .map_err(|e| format!("Invalid UTF-8 in JSON string: {e}"))?;
                s.push_str(chunk);
            }
        }

        Err("Unterminated string literal in JSON".to_string())
    }

    fn parse_u32(&mut self) -> Result<u32, String> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(format!("Expected integer at offset {}", self.pos));
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|e| format!("UTF-8 error parsing integer: {e}"))?;
        s.parse::<u32>()
            .map_err(|_| format!("Invalid u32 literal '{}'", s))
    }

    fn skip_value(&mut self) -> Result<(), String> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'"') => {
                self.parse_string()?;
                Ok(())
            }
            Some(b'{') => {
                self.pos += 1;
                let mut depth = 1usize;
                while self.pos < self.bytes.len() && depth > 0 {
                    match self.bytes[self.pos] {
                        b'"' => {
                            self.parse_string()?;
                        }
                        b'{' => {
                            depth += 1;
                            self.pos += 1;
                        }
                        b'}' => {
                            depth -= 1;
                            self.pos += 1;
                        }
                        _ => {
                            self.pos += 1;
                        }
                    }
                }
                if depth == 0 {
                    Ok(())
                } else {
                    Err("Unterminated object in JSON".to_string())
                }
            }
            Some(b'[') => {
                self.pos += 1;
                let mut depth = 1usize;
                while self.pos < self.bytes.len() && depth > 0 {
                    match self.bytes[self.pos] {
                        b'"' => {
                            self.parse_string()?;
                        }
                        b'[' => {
                            depth += 1;
                            self.pos += 1;
                        }
                        b']' => {
                            depth -= 1;
                            self.pos += 1;
                        }
                        _ => {
                            self.pos += 1;
                        }
                    }
                }
                if depth == 0 {
                    Ok(())
                } else {
                    Err("Unterminated array in JSON".to_string())
                }
            }
            _ => {
                // Primitive literal (number, boolean, null)
                while self.pos < self.bytes.len() {
                    let b = self.bytes[self.pos];
                    if b == b',' || b == b'}' || b == b']' || b.is_ascii_whitespace() {
                        break;
                    }
                    self.pos += 1;
                }
                Ok(())
            }
        }
    }
}

// ============================================================================
// Tokenizer Data Structures & FSM Pre-Tokenizer
// ============================================================================

/// Complete native Rust BPE Tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub inv_vocab: Vec<Option<String>>,
    pub merges: HashMap<(String, String), usize>,
    pub special_tokens: Vec<(String, u32)>,
    pub byte_to_unicode: [char; 256],
    pub unicode_to_byte: HashMap<char, u8>,
}

impl Tokenizer {
    /// Loads and parses a `tokenizer.json` file from disk using a pure standard library JSON parser.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let text = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read tokenizer file '{:?}': {e}", path.as_ref()))?;
        Self::from_json_str(&text)
    }

    /// Parses tokenizer configurations directly from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let (byte_to_unicode, unicode_to_byte) = build_byte_to_unicode();
        let mut scanner = JsonScanner::new(json.as_bytes());

        scanner.expect_byte(b'{')?;

        let mut vocab = HashMap::new();
        let mut merges = HashMap::new();
        let mut special_tokens = Vec::new();

        while let Some(b) = scanner.peek() {
            if b == b'}' {
                break;
            }

            let key = scanner.parse_string()?;
            scanner.expect_byte(b':')?;

            if key == "added_tokens" {
                scanner.expect_byte(b'[')?;
                while let Some(tb) = scanner.peek() {
                    if tb == b']' {
                        scanner.pos += 1;
                        break;
                    }
                    scanner.expect_byte(b'{')?;
                    let mut tok_id_opt = None;
                    let mut content_opt = None;

                    while let Some(ob) = scanner.peek() {
                        if ob == b'}' {
                            scanner.pos += 1;
                            break;
                        }
                        let prop = scanner.parse_string()?;
                        scanner.expect_byte(b':')?;
                        if prop == "id" {
                            tok_id_opt = Some(scanner.parse_u32()?);
                        } else if prop == "content" {
                            content_opt = Some(scanner.parse_string()?);
                        } else {
                            scanner.skip_value()?;
                        }
                        if scanner.peek() == Some(b',') {
                            scanner.pos += 1;
                        }
                    }

                    if let (Some(id), Some(content)) = (tok_id_opt, content_opt) {
                        special_tokens.push((content, id));
                    }

                    if scanner.peek() == Some(b',') {
                        scanner.pos += 1;
                    }
                }
            } else if key == "model" {
                scanner.expect_byte(b'{')?;
                while let Some(mb) = scanner.peek() {
                    if mb == b'}' {
                        scanner.pos += 1;
                        break;
                    }
                    let model_prop = scanner.parse_string()?;
                    scanner.expect_byte(b':')?;
                    if model_prop == "vocab" {
                        scanner.expect_byte(b'{')?;
                        while let Some(vb) = scanner.peek() {
                            if vb == b'}' {
                                scanner.pos += 1;
                                break;
                            }
                            let token_str = scanner.parse_string()?;
                            scanner.expect_byte(b':')?;
                            let token_id = scanner.parse_u32()?;
                            vocab.insert(token_str, token_id);
                            if scanner.peek() == Some(b',') {
                                scanner.pos += 1;
                            }
                        }
                    } else if model_prop == "merges" {
                        scanner.expect_byte(b'[')?;
                        let mut rank = 0usize;
                        while let Some(ab) = scanner.peek() {
                            if ab == b']' {
                                scanner.pos += 1;
                                break;
                            }
                            let merge_str = scanner.parse_string()?;
                            if let Some(space_idx) = merge_str.find(' ') {
                                let left = merge_str[..space_idx].to_string();
                                let right = merge_str[space_idx + 1..].to_string();
                                merges.insert((left, right), rank);
                                rank += 1;
                            }
                            if scanner.peek() == Some(b',') {
                                scanner.pos += 1;
                            }
                        }
                    } else {
                        scanner.skip_value()?;
                    }
                    if scanner.peek() == Some(b',') {
                        scanner.pos += 1;
                    }
                }
            } else {
                scanner.skip_value()?;
            }

            if scanner.peek() == Some(b',') {
                scanner.pos += 1;
            }
        }

        // Add special tokens to vocab if missing
        for (content, id) in &special_tokens {
            vocab.entry(content.clone()).or_insert(*id);
        }

        // Sort special tokens by descending length for greedy matching
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Build inverse vocab table indexed directly by token ID
        let max_id = vocab.values().copied().max().unwrap_or(0);
        let mut inv_vocab: Vec<Option<String>> = vec![None; (max_id + 1) as usize];
        for (tok_str, &tok_id) in &vocab {
            if (tok_id as usize) < inv_vocab.len() {
                inv_vocab[tok_id as usize] = Some(tok_str.clone());
            }
        }

        Ok(Self {
            vocab,
            inv_vocab,
            merges,
            special_tokens,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    /// Pre-tokenizes a clean text chunk using the regex-free Finite State Machine (FSM)
    /// adhering to GPT-2 / Llama byte-level conventions.
    pub fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut chunks = Vec::new();
        let mut offset = 0;

        while offset < text.len() {
            let remaining = &text[offset..];

            // 1. Contractions: 's, 't, 're, 've, 'm, 'll, 'd (and unicode smart quote \u2019)
            let quote_len = if remaining.starts_with('\'') {
                1
            } else if remaining.starts_with('\u{2019}') {
                '\u{2019}'.len_utf8()
            } else {
                0
            };

            if quote_len > 0 {
                let after_quote = &remaining[quote_len..];
                let matched_len = if starts_with_ignore_case(after_quote, "s") {
                    Some(quote_len + 1)
                } else if starts_with_ignore_case(after_quote, "t") {
                    Some(quote_len + 1)
                } else if starts_with_ignore_case(after_quote, "m") {
                    Some(quote_len + 1)
                } else if starts_with_ignore_case(after_quote, "d") {
                    Some(quote_len + 1)
                } else if starts_with_ignore_case(after_quote, "re") {
                    Some(quote_len + 2)
                } else if starts_with_ignore_case(after_quote, "ve") {
                    Some(quote_len + 2)
                } else if starts_with_ignore_case(after_quote, "ll") {
                    Some(quote_len + 2)
                } else {
                    None
                };

                if let Some(len) = matched_len {
                    chunks.push(&text[offset..offset + len]);
                    offset += len;
                    continue;
                }
            }

            // 2. Space followed by word, single digit, or punctuation
            if remaining.starts_with(' ') {
                let after_space = &remaining[1..];
                if let Some(next_ch) = after_space.chars().next() {
                    if next_ch.is_alphabetic() {
                        // ' ?\p{L}+'
                        let mut len = 1;
                        for ch in after_space.chars() {
                            if ch.is_alphabetic() {
                                len += ch.len_utf8();
                            } else {
                                break;
                            }
                        }
                        chunks.push(&text[offset..offset + len]);
                        offset += len;
                        continue;
                    } else if next_ch.is_numeric() {
                        // ' ?\p{N}' (single digit with individual_digits=true)
                        let len = 1 + next_ch.len_utf8();
                        chunks.push(&text[offset..offset + len]);
                        offset += len;
                        continue;
                    } else if !next_ch.is_whitespace() {
                        // ' ?[^\s\p{L}\p{N}]+'
                        let mut len = 1;
                        for ch in after_space.chars() {
                            if !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric() {
                                len += ch.len_utf8();
                            } else {
                                break;
                            }
                        }
                        chunks.push(&text[offset..offset + len]);
                        offset += len;
                        continue;
                    }
                }
            }

            // 3. Runs without leading space
            let first_ch = remaining.chars().next().expect("remaining not empty");
            if first_ch.is_alphabetic() {
                let mut len = 0;
                for ch in remaining.chars() {
                    if ch.is_alphabetic() {
                        len += ch.len_utf8();
                    } else {
                        break;
                    }
                }
                chunks.push(&text[offset..offset + len]);
                offset += len;
            } else if first_ch.is_numeric() {
                // Single digit run
                let len = first_ch.len_utf8();
                chunks.push(&text[offset..offset + len]);
                offset += len;
            } else if first_ch.is_whitespace() {
                // Whitespace run
                let mut len = 0;
                for ch in remaining.chars() {
                    if ch.is_whitespace() {
                        len += ch.len_utf8();
                    } else {
                        break;
                    }
                }
                chunks.push(&text[offset..offset + len]);
                offset += len;
            } else {
                // Non-whitespace, non-alphabetic, non-numeric run
                let mut len = 0;
                for ch in remaining.chars() {
                    if !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric() {
                        len += ch.len_utf8();
                    } else {
                        break;
                    }
                }
                chunks.push(&text[offset..offset + len]);
                offset += len;
            }
        }

        chunks
    }

    /// Performs BPE rank minimization merge on a list of byte-encoded symbol strings.
    fn bpe_merge(&self, mut symbols: Vec<String>) -> Vec<String> {
        while symbols.len() > 1 {
            let mut min_rank = None;
            let mut best_pair = None;

            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if min_rank.is_none() || rank < min_rank.unwrap() {
                        min_rank = Some(rank);
                        best_pair = Some(pair);
                    }
                }
            }

            let Some((left, right)) = best_pair else {
                break;
            };

            let mut new_symbols = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i < symbols.len() - 1 && symbols[i] == left && symbols[i + 1] == right {
                    let mut merged = String::with_capacity(left.len() + right.len());
                    merged.push_str(&left);
                    merged.push_str(&right);
                    new_symbols.push(merged);
                    i += 2;
                } else {
                    new_symbols.push(symbols[i].clone());
                    i += 1;
                }
            }
            symbols = new_symbols;
        }

        symbols
    }

    /// Encodes a single pre-tokenized word/chunk into token IDs using byte fallback and BPE merges.
    pub fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        if chunk.is_empty() {
            return Vec::new();
        }

        // Fast path: exact match in vocab
        if let Some(&id) = self.vocab.get(chunk) {
            return vec![id];
        }

        // Convert raw UTF-8 bytes to bijective Unicode characters
        let symbols: Vec<String> = chunk
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_unicode[b as usize].to_string())
            .collect();

        // Perform BPE rank minimization
        let merged = self.bpe_merge(symbols);

        let mut out = Vec::with_capacity(merged.len());
        for sym in merged {
            if let Some(&id) = self.vocab.get(&sym) {
                out.push(id);
            } else {
                // Fallback to individual characters
                for ch in sym.chars() {
                    let ch_str = ch.to_string();
                    if let Some(&cid) = self.vocab.get(&ch_str) {
                        out.push(cid);
                    }
                }
            }
        }
        out
    }

    /// Full BPE encoding of an input text string into token IDs, correctly extracting special tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut cursor = 0;

        while cursor < text.len() {
            let remaining = &text[cursor..];

            // Check if current position matches any special token
            let mut matched_special = None;
            for (spec_text, spec_id) in &self.special_tokens {
                if remaining.starts_with(spec_text) {
                    matched_special = Some((spec_text.as_str(), *spec_id));
                    break;
                }
            }

            if let Some((spec_str, spec_id)) = matched_special {
                tokens.push(spec_id);
                cursor += spec_str.len();
                continue;
            }

            // Find distance to next special token (or end of string)
            let mut next_special_idx = text.len();
            for (spec_text, _) in &self.special_tokens {
                if let Some(idx) = remaining.find(spec_text) {
                    let absolute_idx = cursor + idx;
                    if absolute_idx < next_special_idx {
                        next_special_idx = absolute_idx;
                    }
                }
            }

            let slice = &text[cursor..next_special_idx];
            let pre_chunks = self.pre_tokenize(slice);
            for chunk in pre_chunks {
                tokens.extend(self.encode_chunk(chunk));
            }

            cursor = next_special_idx;
        }

        tokens
    }

    /// Full decoding of a slice of token IDs back into a UTF-8 string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut raw_bytes = Vec::new();
        for &tok_id in tokens {
            if let Some(Some(tok_str)) = self.inv_vocab.get(tok_id as usize) {
                for ch in tok_str.chars() {
                    if let Some(&b) = self.unicode_to_byte.get(&ch) {
                        raw_bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        let encoded = ch.encode_utf8(&mut buf);
                        raw_bytes.extend_from_slice(encoded.as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&raw_bytes).into_owned()
    }

    /// Decodes a single token into its raw UTF-8 byte representation.
    pub fn decode_token(&self, token_id: u32) -> Option<Vec<u8>> {
        if let Some(Some(tok_str)) = self.inv_vocab.get(token_id as usize) {
            let mut raw = Vec::new();
            for ch in tok_str.chars() {
                if let Some(&b) = self.unicode_to_byte.get(&ch) {
                    raw.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    raw.extend_from_slice(encoded.as_bytes());
                }
            }
            Some(raw)
        } else {
            None
        }
    }

    /// Creates a new `StreamingDecoder` instance using this tokenizer's vocabularies.
    pub fn streaming_decoder(&self) -> StreamingDecoder {
        StreamingDecoder::new(self.inv_vocab.clone(), self.unicode_to_byte.clone())
    }
}

// ============================================================================
// Streaming Decoder with Multi-Byte UTF-8 Partial Buffering
// ============================================================================

/// Real-time streaming decoder with pending byte buffer for multi-byte UTF-8 partial byte sequences.
#[derive(Debug, Clone)]
pub struct StreamingDecoder {
    inv_vocab: Vec<Option<String>>,
    unicode_to_byte: HashMap<char, u8>,
    pending_bytes: Vec<u8>,
}

impl StreamingDecoder {
    /// Constructs a new `StreamingDecoder`.
    pub fn new(
        inv_vocab: Vec<Option<String>>,
        unicode_to_byte: HashMap<char, u8>,
    ) -> Self {
        Self {
            inv_vocab,
            unicode_to_byte,
            pending_bytes: Vec::new(),
        }
    }

    /// Decodes a single token and returns newly completed, valid UTF-8 characters.
    /// Trailing incomplete multi-byte UTF-8 bytes are retained in `pending_bytes`.
    pub fn decode_token(&mut self, token_id: u32) -> String {
        let token_str_opt = if (token_id as usize) < self.inv_vocab.len() {
            self.inv_vocab[token_id as usize].as_deref()
        } else {
            None
        };

        if let Some(tok_str) = token_str_opt {
            for ch in tok_str.chars() {
                if let Some(&b) = self.unicode_to_byte.get(&ch) {
                    self.pending_bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    self.pending_bytes.extend_from_slice(encoded.as_bytes());
                }
            }
        }

        self.drain_valid_utf8()
    }

    fn drain_valid_utf8(&mut self) -> String {
        if self.pending_bytes.is_empty() {
            return String::new();
        }

        match std::str::from_utf8(&self.pending_bytes) {
            Ok(s) => {
                let out = s.to_string();
                self.pending_bytes.clear();
                out
            }
            Err(err) => {
                let valid_up_to = err.valid_up_to();
                if valid_up_to > 0 {
                    let valid_str = std::str::from_utf8(&self.pending_bytes[..valid_up_to])
                        .expect("valid utf-8 prefix")
                        .to_string();
                    self.pending_bytes.drain(..valid_up_to);
                    valid_str
                } else if let Some(err_len) = err.error_len() {
                    // Invalid sequence encountered: emit replacement character and drop bad bytes
                    self.pending_bytes.drain(..err_len);
                    "\u{FFFD}".to_string()
                } else {
                    // Valid prefix is 0 and error_len is None: incomplete UTF-8 sequence waiting for more bytes
                    String::new()
                }
            }
        }
    }

    /// Flushes any remaining bytes in the buffer, converting them to UTF-8 lossy.
    pub fn flush(&mut self) -> String {
        if self.pending_bytes.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending_bytes).into_owned();
        self.pending_bytes.clear();
        out
    }

    /// Resets the decoder buffer.
    pub fn reset(&mut self) {
        self.pending_bytes.clear();
    }
}

// ============================================================================
// Internal Helpers
// ============================================================================

#[inline]
fn starts_with_ignore_case(s: &str, prefix: &str) -> bool {
    if s.len() < prefix.len() {
        return false;
    }
    s[..prefix.len()].eq_ignore_ascii_case(prefix)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_unicode_bijection_256() {
        let (b2u, u2b) = build_byte_to_unicode();
        let mut seen = std::collections::HashSet::new();

        for b in 0u8..=255u8 {
            let ch = b2u[b as usize];
            assert!(seen.insert(ch), "Duplicate char mapped for byte {b}");
            assert_eq!(u2b.get(&ch), Some(&b), "Inverse mapping failed for byte {b}");
        }
        assert_eq!(seen.len(), 256);
        assert_eq!(b2u[0x20 as usize], '\u{0120}'); // Space -> Ġ
        assert_eq!(b2u[0x0a as usize], '\u{010a}'); // Newline -> Ċ
    }

    #[test]
    fn test_json_scanner_strings_and_numbers() {
        let json = r#"{"vocab": {"Once": 6403, "Ġupon": 1980, "special\n": 1}, "merges": ["Ġ u", "Ġu p"]}"#;
        let mut scanner = JsonScanner::new(json.as_bytes());
        scanner.expect_byte(b'{').unwrap();

        let key1 = scanner.parse_string().unwrap();
        assert_eq!(key1, "vocab");
        scanner.expect_byte(b':').unwrap();
        scanner.expect_byte(b'{').unwrap();

        let tok1 = scanner.parse_string().unwrap();
        assert_eq!(tok1, "Once");
        scanner.expect_byte(b':').unwrap();
        let id1 = scanner.parse_u32().unwrap();
        assert_eq!(id1, 6403);
    }

    #[test]
    fn test_fsm_pre_tokenization() {
        let (b2u, u2b) = build_byte_to_unicode();
        let tok = Tokenizer {
            vocab: HashMap::new(),
            inv_vocab: Vec::new(),
            merges: HashMap::new(),
            special_tokens: Vec::new(),
            byte_to_unicode: b2u,
            unicode_to_byte: u2b,
        };

        let chunks = tok.pre_tokenize("Once upon a time, don't 123 foo!");
        assert_eq!(
            chunks,
            vec!["Once", " upon", " a", " time", ",", " don", "'t", " 1", "2", "3", " foo", "!"]
        );
    }

    #[test]
    fn test_streaming_decoder_multibyte_partial_buffering() {
        let (_, u2b) = build_byte_to_unicode();

        // Create custom single-byte mappings for testing
        let mut test_u2b = u2b.clone();
        test_u2b.insert('A', 0xE4);
        test_u2b.insert('B', 0xBD);
        test_u2b.insert('C', 0xA0);

        let test_inv = vec![
            Some("A".to_string()), // 0xE4
            Some("B".to_string()), // 0xBD
            Some("C".to_string()), // 0xA0 -> completes '你'
            Some("D".to_string()), // standard 'D'
        ];
        test_u2b.insert('D', b'D');

        let mut stream = StreamingDecoder::new(test_inv, test_u2b);

        // Step 1: byte 1 of 3 (0xE4)
        let s1 = stream.decode_token(0);
        assert_eq!(s1, "", "Partial 1-byte sequence must buffer");

        // Step 2: byte 2 of 3 (0xBD)
        let s2 = stream.decode_token(1);
        assert_eq!(s2, "", "Partial 2-byte sequence must buffer");

        // Step 3: byte 3 of 3 (0xA0) -> completes '你'
        let s3 = stream.decode_token(2);
        assert_eq!(s3, "你", "Completed 3-byte sequence must emit decoded character");

        // Step 4: next token 'D'
        let s4 = stream.decode_token(3);
        assert_eq!(s4, "D", "ASCII token must emit immediately");
    }

    #[test]
    fn test_tokenizer_ground_truth_if_file_exists() {
        let tok_path = Path::new("tokenizer.json");
        if !tok_path.exists() {
            return;
        }

        let tokenizer = Tokenizer::from_file(tok_path).expect("Failed to load tokenizer.json");

        // 1. Text prompt encoding ground truth
        let encoded = tokenizer.encode("Once upon a time");
        assert_eq!(encoded, vec![6403, 1980, 253, 655]);

        // 2. Special token + text prompt encoding ground truth
        let encoded_chat = tokenizer.encode("<|im_start|>The you");
        assert_eq!(encoded_chat, vec![1, 504, 346]);

        // 3. Token decoding ground truth
        let decoded = tokenizer.decode(&[359, 253, 4028, 282]);
        assert_eq!(decoded, " are a member of");

        // 4. Streaming decoder ground truth
        let mut stream = tokenizer.streaming_decoder();
        let mut stream_out = String::new();
        for &tok_id in &[359, 253, 4028, 282] {
            stream_out.push_str(&stream.decode_token(tok_id));
        }
        stream_out.push_str(&stream.flush());
        assert_eq!(stream_out, " are a member of");
    }
}
