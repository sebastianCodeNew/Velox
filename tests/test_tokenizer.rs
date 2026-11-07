//! Integration test suite for the Velox Native Zero-Dependency BPE Tokenizer (`tests/test_tokenizer.rs`)
//!
//! Verifies:
//! - 256-byte bijective table mapping (`bytes_to_unicode` & `unicode_to_bytes`).
//! - Pure standard library streaming JSON parser on synthetic and live `tokenizer.json`.
//! - Regex-free pre-tokenization Finite State Machine (FSM) matching GPT-2 / Llama style.
//! - BPE merge engine: pair merge rank minimization.
//! - Special tokens handling: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`.
//! - StreamingDecoder with partial multi-byte UTF-8 buffering (2-byte, 3-byte CJK, 4-byte emoji).
//! - Verified ground-truth mappings against SmolLM-135M reference tokenizer.

use std::collections::HashMap;
use std::path::Path;
use veloxc::tokenizer::{build_byte_to_unicode, StreamingDecoder, Tokenizer};

// ============================================================================
// 1. Byte-to-Unicode Bijective Mapping Table Tests
// ============================================================================

#[test]
fn test_bytes_to_unicode_full_bijection_256() {
    let (b2u, u2b) = build_byte_to_unicode();
    let mut seen_chars = std::collections::HashSet::new();

    for b in 0u8..=255u8 {
        let c = b2u[b as usize];
        assert!(seen_chars.insert(c), "Duplicate char mapped for byte {b}");
        assert_eq!(u2b.get(&c), Some(&b), "Inverse mapping mismatch for byte {b}");
    }

    assert_eq!(seen_chars.len(), 256, "Must contain exactly 256 unique mapped characters");

    // Standard GPT-2 byte mappings
    assert_eq!(b2u[0x20 as usize], '\u{0120}'); // Space -> Ġ
    assert_eq!(b2u[0x0a as usize], '\u{010a}'); // Newline -> Ċ
    assert_eq!(b2u[0x09 as usize], '\u{0109}'); // Tab -> č
    assert_eq!(b2u[b'A' as usize], 'A');        // Printable ASCII -> identity
    assert_eq!(b2u[b'z' as usize], 'z');
}

// ============================================================================
// 2. Synthetic Zero-Dependency JSON Parser Tests
// ============================================================================

#[test]
fn test_synthetic_json_tokenizer_parser() {
    let synthetic_json = r#"{
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "special": true},
            {"id": 1, "content": "<|im_start|>", "special": true},
            {"id": 2, "content": "<|im_end|>", "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "<|endoftext|>": 0,
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "H": 10,
                "e": 11,
                "l": 12,
                "o": 13,
                "He": 14,
                "ll": 15,
                "Hell": 16,
                "Hello": 17,
                "Ġ": 18,
                "w": 19,
                "r": 20,
                "d": 21,
                "Ġw": 22,
                "or": 23,
                "ld": 24,
                "Ġworld": 25
            },
            "merges": [
                "H e",
                "l l",
                "He ll",
                "Hell o",
                "Ġ w",
                "o r",
                "l d",
                "Ġw orld"
            ]
        }
    }"#;

    let tokenizer = Tokenizer::from_json_str(synthetic_json)
        .expect("Failed to parse synthetic tokenizer JSON");

    assert_eq!(tokenizer.vocab.get("<|im_start|>"), Some(&1));
    assert_eq!(tokenizer.vocab.get("Hello"), Some(&17));
    assert_eq!(tokenizer.special_tokens.len(), 3);

    // Test encode with BPE merges
    let encoded = tokenizer.encode("Hello");
    assert_eq!(encoded, vec![17]);

    // Test decode
    let decoded = tokenizer.decode(&[17]);
    assert_eq!(decoded, "Hello");
}

// ============================================================================
// 3. Regex-Free Pre-Tokenization FSM Tests
// ============================================================================

#[test]
fn test_fsm_pre_tokenizer_comprehensive_cases() {
    let (b2u, u2b) = build_byte_to_unicode();
    let tok = Tokenizer {
        vocab: HashMap::new(),
        inv_vocab: Vec::new(),
        merges: HashMap::new(),
        special_tokens: Vec::new(),
        byte_to_unicode: b2u,
        unicode_to_byte: u2b,
    };

    // 1. Contractions
    let chunks_cont = tok.pre_tokenize("I'm you've they'll he'd don't it's we're");
    assert_eq!(
        chunks_cont,
        vec![
            "I", "'m", " you", "'ve", " they", "'ll", " he", "'d", " don", "'t", " it", "'s",
            " we", "'re"
        ]
    );

    // 2. Individual digits (individual_digits=true)
    let chunks_digits = tok.pre_tokenize("Count: 12345");
    assert_eq!(
        chunks_digits,
        vec!["Count", ":", " 1", "2", "3", "4", "5"]
    );

    // 3. Punctuation and symbols
    let chunks_punct = tok.pre_tokenize("Hello, world! -- Ready?");
    assert_eq!(
        chunks_punct,
        vec!["Hello", ",", " world", "!", " --", " Ready", "?"]
    );

    // 4. Whitespace sequences
    let chunks_ws = tok.pre_tokenize("Line1\n\nLine2   Line3");
    assert_eq!(
        chunks_ws,
        vec!["Line", "1", "\n\n", "Line", "2", "   ", "Line", "3"]
    );
}

// ============================================================================
// 4. Special Tokens Handling Tests
// ============================================================================

#[test]
fn test_special_tokens_preservation() {
    let synthetic_json = r#"{
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>", "special": true},
            {"id": 1, "content": "<|im_start|>", "special": true},
            {"id": 2, "content": "<|im_end|>", "special": true}
        ],
        "model": {
            "type": "BPE",
            "vocab": {
                "<|endoftext|>": 0,
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "T": 10,
                "h": 11,
                "e": 12,
                "The": 504,
                "Ġ": 18,
                "y": 19,
                "o": 20,
                "u": 21,
                "Ġy": 22,
                "ou": 23,
                "Ġyou": 346
            },
            "merges": [
                "T h",
                "Th e",
                "Ġ y",
                "o u",
                "Ġy ou"
            ]
        }
    }"#;

    let tokenizer = Tokenizer::from_json_str(synthetic_json)
        .expect("Failed to parse synthetic tokenizer");

    let encoded = tokenizer.encode("<|im_start|>The you<|im_end|>");
    assert_eq!(encoded, vec![1, 504, 346, 2]);

    let decoded = tokenizer.decode(&[1, 504, 346, 2]);
    assert_eq!(decoded, "<|im_start|>The you<|im_end|>");
}

// ============================================================================
// 5. Streaming Decoder Multi-Byte UTF-8 Buffering Tests
// ============================================================================

#[test]
fn test_streaming_decoder_multibyte_utf8_buffering() {
    let (_, u2b) = build_byte_to_unicode();

    // 2-byte UTF-8 character: 'é' = [0xC3, 0xA9]
    // 3-byte UTF-8 character: '你' = [0xE4, 0xBD, 0xA0]
    // 4-byte UTF-8 character: '🚀' = [0xF0, 0x9F, 0x9A, 0x80]

    let mut test_u2b = u2b.clone();
    test_u2b.insert('1', 0xC3);
    test_u2b.insert('2', 0xA9);

    test_u2b.insert('3', 0xE4);
    test_u2b.insert('4', 0xBD);
    test_u2b.insert('5', 0xA0);

    test_u2b.insert('6', 0xF0);
    test_u2b.insert('7', 0x9F);
    test_u2b.insert('8', 0x9A);
    test_u2b.insert('9', 0x80);

    let test_inv = vec![
        Some("1".to_string()), // token 0: byte 0xC3
        Some("2".to_string()), // token 1: byte 0xA9 -> completes 'é'
        Some("3".to_string()), // token 2: byte 0xE4
        Some("4".to_string()), // token 3: byte 0xBD
        Some("5".to_string()), // token 4: byte 0xA0 -> completes '你'
        Some("6".to_string()), // token 5: byte 0xF0
        Some("7".to_string()), // token 6: byte 0x9F
        Some("8".to_string()), // token 7: byte 0x9A
        Some("9".to_string()), // token 8: byte 0x80 -> completes '🚀'
    ];

    let mut stream = StreamingDecoder::new(test_inv, test_u2b);

    // 1. 2-byte sequence 'é'
    assert_eq!(stream.decode_token(0), "");
    assert_eq!(stream.decode_token(1), "é");

    // 2. 3-byte sequence '你'
    assert_eq!(stream.decode_token(2), "");
    assert_eq!(stream.decode_token(3), "");
    assert_eq!(stream.decode_token(4), "你");

    // 3. 4-byte emoji sequence '🚀'
    assert_eq!(stream.decode_token(5), "");
    assert_eq!(stream.decode_token(6), "");
    assert_eq!(stream.decode_token(7), "");
    assert_eq!(stream.decode_token(8), "🚀");
}

// ============================================================================
// 6. Ground-Truth Verification Against SmolLM-135M Tokenizer
// ============================================================================

#[test]
fn test_smollm_tokenizer_ground_truth_parity() {
    let tok_path = Path::new("tokenizer.json");
    if !tok_path.exists() {
        return;
    }

    let tokenizer = Tokenizer::from_file(tok_path)
        .expect("Failed to load tokenizer.json from workspace root");

    // Check vocabulary size
    assert!(tokenizer.vocab.len() >= 49152);

    // Check merges count
    assert!(tokenizer.merges.len() >= 48900);

    // 1. Ground truth mapping: "Once upon a time" -> [6403, 1980, 253, 655]
    let enc1 = tokenizer.encode("Once upon a time");
    assert_eq!(enc1, vec![6403, 1980, 253, 655]);

    // 2. Ground truth mapping: "<|im_start|>The you" -> [1, 504, 346]
    let enc2 = tokenizer.encode("<|im_start|>The you");
    assert_eq!(enc2, vec![1, 504, 346]);

    // 3. Ground truth mapping: [359, 253, 4028, 282] -> " are a member of"
    let dec1 = tokenizer.decode(&[359, 253, 4028, 282]);
    assert_eq!(dec1, " are a member of");

    // 4. Full 32-token sequence decode check
    let expected_32 = [
        359, 253, 4028, 282, 260, 1773, 30, 198, 26501, 368, 281, 260, 1773, 198, 20255, 3726,
        288, 9517, 253, 12301, 198, 19161, 354, 12301, 198, 19161, 354, 12301, 198, 2683, 416, 722,
    ];
    let dec32 = tokenizer.decode(&expected_32);
    assert_eq!(
        dec32,
        " are a member of the community.\nParticipate in the community\nJoin Here to Write a Essay\nWrite an Essay\nWrite an Essay\nYou can use"
    );

    // 5. Streaming decoder step-by-step equivalence
    let mut stream = tokenizer.streaming_decoder();
    let mut stream_output = String::new();
    for &tok in &expected_32 {
        stream_output.push_str(&stream.decode_token(tok));
    }
    stream_output.push_str(&stream.flush());
    assert_eq!(stream_output, dec32);
}
