//! Integration Tests for Velox Streaming Lexer (`tests/test_lexer.rs`)
//!
//! Exhaustive test suite covering:
//! - Keyword classification (system, hardware, scopes, KV caches, statements, async primitives)
//! - Memory tier attribute classification (@vram_immutable, @paged_vram, etc.)
//! - Scalar precision types and memory capacity units
//! - Operators across all 14 precedence levels and delimiters
//! - Decimal, hex, binary, and scientific float numeric literals
//! - Decimal range lookahead (`0..32` produces Decimal, DotDot, Decimal without float consumption)
//! - String literals and escape decoding (\n, \t, \r, \\, \", \0, \xHH)
//! - Unterminated strings and unterminated block comments error reporting
//! - 1-indexed line and column tracking, 4-space tab stop normalization, CRLF & LF line endings
//! - Single-line hash (`#`) and slash (`//`) comments, and arbitrarily nested block comments (`/* ... /* ... */ ... */`)
//! - Error tokens for unrecognized characters and unrecognized attributes

use veloxc::lexer::Lexer;
use veloxc::tokens::{Token, TokenType};

// ============================================================================
// 1. Keywords, Attributes, Types, and Units Classification
// ============================================================================

#[test]
fn test_system_and_model_keywords() {
    let code = "target hardware device vram_capacity compute_capability safety_factor model weight tensor Tensor safetensors mmap import as const";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::Target,
        TokenType::Hardware,
        TokenType::Device,
        TokenType::VramCapacity,
        TokenType::ComputeCapability,
        TokenType::SafetyFactor,
        TokenType::Model,
        TokenType::Weight,
        TokenType::Tensor,
        TokenType::Tensor,
        TokenType::Safetensors,
        TokenType::Mmap,
        TokenType::Import,
        TokenType::As,
        TokenType::Const,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_execution_scope_keywords() {
    let code = "scope prefill decode chunked_prefill speculative_verify";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::Scope,
        TokenType::Prefill,
        TokenType::Decode,
        TokenType::ChunkedPrefill,
        TokenType::SpeculativeVerify,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_kv_cache_type_keywords() {
    let code = "PagedKVCache paged_kv VirtualContiguousKV virtual_kv LatentKVCache latent_kv StreamingSinkKVCache streaming_sink SpeculativeTreeBlock tree_block";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::PagedKVCache,
        TokenType::PagedKVCache,
        TokenType::VirtualContiguousKV,
        TokenType::VirtualContiguousKV,
        TokenType::LatentKVCache,
        TokenType::LatentKVCache,
        TokenType::StreamingSinkKVCache,
        TokenType::StreamingSinkKVCache,
        TokenType::SpeculativeTreeBlock,
        TokenType::SpeculativeTreeBlock,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_control_flow_and_statement_keywords() {
    let code = "let mut fn kernel return if else for in while yield true false null";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::Let,
        TokenType::Mut,
        TokenType::Fn,
        TokenType::Kernel,
        TokenType::Return,
        TokenType::If,
        TokenType::Else,
        TokenType::For,
        TokenType::In,
        TokenType::While,
        TokenType::Yield,
        TokenType::Bool(true),
        TokenType::Bool(false),
        TokenType::Null,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_async_and_serving_primitives_keywords() {
    let code = "absorb_mla_q absorb_mla_v dma_async_h2d dma_async_d2h tma_load_async tma_store_async mbarrier";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::AbsorbMlaQ,
        TokenType::AbsorbMlaV,
        TokenType::DmaAsyncH2D,
        TokenType::DmaAsyncD2H,
        TokenType::TmaLoadAsync,
        TokenType::TmaStoreAsync,
        TokenType::Mbarrier,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_memory_tier_attributes() {
    let code = "@vram_immutable @paged_vram @bump_sram @host_pinned @wired_limit_bounded @warp_specialized @kernel_fuse";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::AttrVramImmutable,
        TokenType::AttrPagedVram,
        TokenType::AttrBumpSram,
        TokenType::AttrHostPinned,
        TokenType::AttrWiredLimitBounded,
        TokenType::AttrWarpSpecialized,
        TokenType::AttrKernelFuse,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_scalar_type_keywords() {
    let code = "f32 f16 bf16 fp8_e4m3 fp8_e5m2 fp8_e8m0 int8 int4 u64 u32 u16 u8 i64 i32 i16 i8 bool usize";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::TypeF32,
        TokenType::TypeF16,
        TokenType::TypeBf16,
        TokenType::TypeFp8E4m3,
        TokenType::TypeFp8E5m2,
        TokenType::TypeFp8E8m0,
        TokenType::TypeInt8,
        TokenType::TypeInt4,
        TokenType::TypeU64,
        TokenType::TypeU32,
        TokenType::TypeU16,
        TokenType::TypeU8,
        TokenType::TypeI64,
        TokenType::TypeI32,
        TokenType::TypeI16,
        TokenType::TypeI8,
        TokenType::TypeBool,
        TokenType::TypeUsize,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_memory_units() {
    let code = "KiB MiB GiB TiB KB MB GB TB";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::UnitKiB,
        TokenType::UnitMiB,
        TokenType::UnitGiB,
        TokenType::UnitTiB,
        TokenType::UnitKB,
        TokenType::UnitMB,
        TokenType::UnitGB,
        TokenType::UnitTB,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

// ============================================================================
// 2. Operators & Delimiters Classification
// ============================================================================

#[test]
fn test_operators_all_precedence_levels() {
    let code = "@ + - * / % & | ^ ~ ! && || << >> = == != < <= > >= += -= *= /= -> => ::";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::At,
        TokenType::Plus,
        TokenType::Minus,
        TokenType::Star,
        TokenType::Slash,
        TokenType::Percent,
        TokenType::Amp,
        TokenType::Pipe,
        TokenType::Caret,
        TokenType::Tilde,
        TokenType::Bang,
        TokenType::AmpAmp,
        TokenType::PipePipe,
        TokenType::Shl,
        TokenType::Shr,
        TokenType::Eq,
        TokenType::EqEq,
        TokenType::BangEq,
        TokenType::Lt,
        TokenType::LtEq,
        TokenType::Gt,
        TokenType::GtEq,
        TokenType::PlusEq,
        TokenType::MinusEq,
        TokenType::StarEq,
        TokenType::SlashEq,
        TokenType::Arrow,
        TokenType::FatArrow,
        TokenType::ColonColon,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_delimiters_and_punctuation() {
    let code = "( ) [ ] { } , : ; . .. -> =>";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = vec![
        TokenType::LParen,
        TokenType::RParen,
        TokenType::LBracket,
        TokenType::RBracket,
        TokenType::LBrace,
        TokenType::RBrace,
        TokenType::Comma,
        TokenType::Colon,
        TokenType::Semicolon,
        TokenType::Dot,
        TokenType::DotDot,
        TokenType::Arrow,
        TokenType::FatArrow,
        TokenType::Eof,
    ];
    let actual: Vec<TokenType> = tokens.into_iter().map(|t| t.token_type).collect();
    assert_eq!(actual, expected);
}

// ============================================================================
// 3. Numeric Literals Decoding
// ============================================================================

#[test]
fn test_decimal_integers() {
    let code = "0 42 1048576 1_000_000";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    assert_eq!(tokens[0].token_type, TokenType::Decimal(0));
    assert_eq!(tokens[1].token_type, TokenType::Decimal(42));
    assert_eq!(tokens[2].token_type, TokenType::Decimal(1048576));
    assert_eq!(tokens[3].token_type, TokenType::Decimal(1000000));
}

#[test]
fn test_hexadecimal_integers() {
    let code = "0x0 0x2A 0xDEAD_BEEF 0Xffff";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    assert_eq!(tokens[0].token_type, TokenType::Hex(0));
    assert_eq!(tokens[1].token_type, TokenType::Hex(42));
    assert_eq!(tokens[2].token_type, TokenType::Hex(0xDEADBEEF));
    assert_eq!(tokens[3].token_type, TokenType::Hex(0xFFFF));
}

#[test]
fn test_binary_integers() {
    let code = "0b0 0b101010 0b1111_0000 0B101";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    assert_eq!(tokens[0].token_type, TokenType::Binary(0));
    assert_eq!(tokens[1].token_type, TokenType::Binary(42));
    assert_eq!(tokens[2].token_type, TokenType::Binary(240));
    assert_eq!(tokens[3].token_type, TokenType::Binary(5));
}

#[test]
#[allow(clippy::approx_constant)]
fn test_floating_point_numbers() {
    let code = "0.0 3.14159 0.95 1e-4 2.5e+10 1E6";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");

    match tokens[0].token_type {
        TokenType::Float(f) => assert_eq!(f, 0.0),
        _ => panic!("Expected Float"),
    }
    match tokens[1].token_type {
        TokenType::Float(f) => assert!((f - 3.14159).abs() < 1e-6),
        _ => panic!("Expected Float"),
    }
    match tokens[2].token_type {
        TokenType::Float(f) => assert!((f - 0.95).abs() < 1e-6),
        _ => panic!("Expected Float"),
    }
    match tokens[3].token_type {
        TokenType::Float(f) => assert!((f - 1e-4).abs() < 1e-9),
        _ => panic!("Expected Float"),
    }
    match tokens[4].token_type {
        TokenType::Float(f) => assert!((f - 2.5e10).abs() < 1e-2),
        _ => panic!("Expected Float"),
    }
    match tokens[5].token_type {
        TokenType::Float(f) => assert!((f - 1e6).abs() < 1e-2),
        _ => panic!("Expected Float"),
    }
}

#[test]
fn test_decimal_before_range_operator() {
    // In '0..10', '0' is decimal literal followed by '..' operator (not float '0.')
    let code = "0..10";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    assert_eq!(tokens[0].token_type, TokenType::Decimal(0));
    assert_eq!(tokens[1].token_type, TokenType::DotDot);
    assert_eq!(tokens[2].token_type, TokenType::Decimal(10));
}

// ============================================================================
// 4. String Literals & Escape Decoding
// ============================================================================

#[test]
fn test_string_literals_basic() {
    let code = r#""hello world" "models/llama3.safetensors" """#;
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    assert_eq!(tokens[0].token_type, TokenType::String("hello world".to_string()));
    assert_eq!(tokens[1].token_type, TokenType::String("models/llama3.safetensors".to_string()));
    assert_eq!(tokens[2].token_type, TokenType::String("".to_string()));
}

#[test]
fn test_string_literals_escapes() {
    let code = r#""line1\nline2\ttab\"quoted\"\\backslash\0null""#;
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let expected = "line1\nline2\ttab\"quoted\"\\backslash\0null";
    assert_eq!(tokens[0].token_type, TokenType::String(expected.to_string()));
}

#[test]
fn test_string_hex_byte_escapes() {
    let code = r#""Hello \x41\x42\x43!""#;
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    assert_eq!(tokens[0].token_type, TokenType::String("Hello ABC!".to_string()));
}

#[test]
fn test_unterminated_string_raises_lexer_error() {
    let code = "\"unterminated string";
    let err = Lexer::tokenize_source(code, "test.vlx").expect_err("should fail on unclosed string");
    assert!(err.message.contains("Unterminated string literal"));
    assert_eq!(err.span.start.line, 1);
    assert_eq!(err.span.start.column, 1);
}

#[test]
fn test_unterminated_string_across_newline() {
    let code = "\"line 1\nline 2\"";
    let err = Lexer::tokenize_source(code, "test.vlx").expect_err("should fail across newline");
    assert!(err.message.contains("Unterminated string literal"));
}

// ============================================================================
// 5. Position & Coordinate Tracking, Tabs, CRLF
// ============================================================================

#[test]
fn test_multiline_line_and_column_tracking() {
    let code = "let a = 10;\nlet b = 20;\nlet c = 30;";
    let tokens = Lexer::tokenize_source(code, "multi.vlx").expect("tokenization should succeed");
    let let_tokens: Vec<&Token> = tokens.iter().filter(|t| t.token_type == TokenType::Let).collect();
    assert_eq!(let_tokens.len(), 3);
    assert_eq!((let_tokens[0].span.start.line, let_tokens[0].span.start.column), (1, 1));
    assert_eq!((let_tokens[1].span.start.line, let_tokens[1].span.start.column), (2, 1));
    assert_eq!((let_tokens[2].span.start.line, let_tokens[2].span.start.column), (3, 1));
}

#[test]
fn test_tab_stop_normalization_4_spaces() {
    // Col 1 + tab -> Col 5
    // Col 2 ('a') + tab -> Col 5
    // Col 5 + tab -> Col 9
    let code = "\tlet a = 1;\n  \tlet b = 2;";
    let tokens = Lexer::tokenize_source(code, "tab.vlx").expect("tokenization should succeed");
    let let_a = &tokens[0];
    let let_b = tokens
        .iter()
        .find(|t| t.text == "let" && t.span.start.line == 2)
        .expect("let b should exist");

    assert_eq!(let_a.span.start.line, 1);
    assert_eq!(let_a.span.start.column, 5);

    assert_eq!(let_b.span.start.line, 2);
    assert_eq!(let_b.span.start.column, 5);
}

#[test]
fn test_crlf_and_lf_newline_handling() {
    let code = "let x = 1;\r\nlet y = 2;\nlet z = 3;\r\n";
    let tokens = Lexer::tokenize_source(code, "crlf.vlx").expect("tokenization should succeed");
    let lets: Vec<&Token> = tokens.iter().filter(|t| t.token_type == TokenType::Let).collect();
    assert_eq!(lets.len(), 3);
    assert_eq!(lets[0].span.start.line, 1);
    assert_eq!(lets[1].span.start.line, 2);
    assert_eq!(lets[2].span.start.line, 3);
}

#[test]
fn test_token_span_extent() {
    let code = "let model_name = 123;";
    let tokens = Lexer::tokenize_source(code, "span.vlx").expect("tokenization should succeed");
    let id_tok = &tokens[1];
    assert_eq!(id_tok.text, "model_name");
    assert_eq!(id_tok.span.start.column, 5);
    assert_eq!(id_tok.span.end.column, 15); // [5, 15) has length 10
    assert_eq!(id_tok.span.end.offset - id_tok.span.start.offset, 10);
}

// ============================================================================
// 6. Comments & Trivia Filtering
// ============================================================================

#[test]
fn test_single_line_comments_python_and_slash() {
    let code = r#"
    # Python-style hash comment on line 1
    let a = 1; // C-style slash comment
    // Full line slash comment
    # Another hash comment
    let b = 2;
    "#;
    let tokens = Lexer::tokenize_source(code, "comments.vlx").expect("tokenization should succeed");
    let names: Vec<&str> = tokens
        .iter()
        .filter_map(|t| match &t.token_type {
            TokenType::Identifier(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn test_nested_block_comments() {
    let code = r#"
    let x = 1;
    /* Level 1
       /* Level 2
          /* Level 3 */
       */
       Back to Level 1
    */
    let y = 2;
    "#;
    let tokens = Lexer::tokenize_source(code, "nested.vlx").expect("tokenization should succeed");
    let names: Vec<&str> = tokens
        .iter()
        .filter_map(|t| match &t.token_type {
            TokenType::Identifier(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["x", "y"]);
}

#[test]
fn test_unterminated_block_comment_raises_lexer_error() {
    let code = "let x = 1; /* Unclosed block comment";
    let err = Lexer::tokenize_source(code, "unclosed.vlx").expect_err("should fail on unclosed comment");
    assert!(err.message.contains("Unterminated block comment"));
}

// ============================================================================
// 7. Unrecognized Tokens & Lexical Errors
// ============================================================================

#[test]
fn test_unrecognized_character_emits_error_token() {
    let code = "let $x = 10;";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let errors: Vec<&Token> = tokens
        .iter()
        .filter(|t| matches!(t.token_type, TokenType::Error(_)))
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].text, "$");
    assert_eq!(errors[0].span.start.column, 5);
}

#[test]
fn test_unrecognized_attribute_emits_error_token() {
    let code = "@unrecognized_attribute weight w: f16;";
    let tokens = Lexer::tokenize_source(code, "test.vlx").expect("tokenization should succeed");
    let errors: Vec<&Token> = tokens
        .iter()
        .filter(|t| matches!(t.token_type, TokenType::Error(_)))
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].text, "@unrecognized_attribute");
}

// ============================================================================
// 8. Positive Tokenization of All 5 Reference Model Scripts
// ============================================================================

#[test]
fn test_tokenize_all_5_reference_model_scripts() {
    let example_files = [
        "examples/tinystories_smollm.vlx",
        "examples/llama3_8b.vlx",
        "examples/qwen2_5_7b.vlx",
        "examples/deepseek_v3_mla.vlx",
        "examples/nanbeige_33b.vlx",
    ];

    for file_path in &example_files {
        let content = std::fs::read_to_string(file_path)
            .unwrap_or_else(|err| panic!("Failed to read {file_path}: {err}"));
        let tokens = Lexer::tokenize_source(&content, *file_path)
            .unwrap_or_else(|err| panic!("Failed to tokenize {file_path}: {err}"));

        assert!(!tokens.is_empty(), "Tokens should not be empty for {file_path}");
        assert!(
            tokens.last().unwrap().is_eof(),
            "Last token must be EOF for {file_path}"
        );

        // Check that there are zero lexical error tokens in valid reference models
        let error_tokens: Vec<&Token> = tokens
            .iter()
            .filter(|t| matches!(t.token_type, TokenType::Error(_)))
            .collect();
        assert!(
            error_tokens.is_empty(),
            "Expected 0 error tokens in {file_path}, but found: {:?}",
            error_tokens
        );
    }
}

