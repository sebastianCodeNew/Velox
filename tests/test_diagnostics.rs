//! Negative tests and diagnostic subsystem verification (`tests/test_diagnostics.rs`)
//!
//! Verifies:
//! - Complete diagnostic catalog coverage for all 25 codes (E001–E025)
//! - Exact negative test triggers and source coordinate pointers for E001, E003, E007, E008, E009, E015, E016, E022
//! - Rustc-grade visual boxed report rendering (80-char border, carets, explanations, remediation)
//! - ANSI color formatting and color suppression
//! - Secondary span reporting
//! - DiagnosticEngine emission, filtering, and batch rendering

use veloxc::diagnostics::{
    Diagnostic, DiagnosticCode, DiagnosticEngine, DiagnosticFormatter, Severity, ALL_CODES,
};
use veloxc::parser::Parser;
use veloxc::types::SourceSpan;

fn parse_for_diagnostics(code: &str) -> Vec<Diagnostic> {
    let mut parser = Parser::from_source(code, "test_diag.vlx").expect("Lexer failed");
    parser.parse_program_safe();
    parser.diagnostics
}

// ============================================================================
// 1. Negative Parser Tests for Architectural Invariants
// ============================================================================

#[test]
fn test_diagnostic_e001_malloc_in_scope() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model ProhibitedMalloc {
        scope decode(token: u32) -> u32 {
            let buffer = malloc(1024);
            return token;
        }
        scope prefill() -> u32 { return 0; }
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e001_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E001).collect();
    assert!(!e001_diags.is_empty(), "E001 diagnostic not triggered for malloc");
    let d = e001_diags[0];
    assert!(d.message.contains("malloc"));
    assert_eq!(d.span.start.line, 5);
    assert_eq!(d.severity, Severity::Error);
}

#[test]
fn test_diagnostic_e001_cuda_malloc_in_prefill() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model ProhibitedCudaMalloc {
        scope prefill(tokens: Tensor<[1, 10], u32>) -> u32 {
            let ptr = cudaMalloc(1024);
            return 0;
        }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e001_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E001).collect();
    assert!(!e001_diags.is_empty(), "E001 diagnostic not triggered for cudaMalloc");
    assert!(e001_diags[0].message.contains("cudaMalloc"));
}

#[test]
fn test_diagnostic_e003_blocking_host_sync_item_in_decode() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model ProhibitedSync {
        scope decode(next_tok: Tensor<[1, 1], u32>) -> u32 {
            let val = next_tok.item();
            return val;
        }
        scope prefill() -> u32 { return 0; }
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e003_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E003).collect();
    assert!(!e003_diags.is_empty(), "E003 diagnostic not triggered for item()");
    assert!(e003_diags[0].message.contains("item"));
    assert_eq!(e003_diags[0].span.start.line, 5);
}

#[test]
fn test_diagnostic_e003_blocking_host_sync_cuda_sync_in_decode() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model ProhibitedCudaSync {
        scope decode(token: u32) -> u32 {
            cudaStreamSynchronize();
            return token;
        }
        scope prefill() -> u32 { return 0; }
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e003_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E003).collect();
    assert!(!e003_diags.is_empty(), "E003 diagnostic not triggered for cudaStreamSynchronize");
    assert!(e003_diags[0].message.contains("cudaStreamSynchronize"));
}

#[test]
fn test_diagnostic_e007_unsupported_hardware_sm_capability() {
    let code = r#"
    target hardware {
        device: "kepler_k40";
        vram_capacity: 12288 * MiB;
        compute_capability: 3.5;
        safety_factor: 0.95;
    }
    model OldGPU {
        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e007_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E007).collect();
    assert!(!e007_diags.is_empty(), "E007 diagnostic not triggered for SM 3.5");
    assert!(e007_diags[0].message.to_lowercase().contains("compute capability"));
    assert_eq!(e007_diags[0].span.start.line, 2);
}

#[test]
fn test_diagnostic_e008_missing_mandatory_decode_scope() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model MissingDecode {
        scope prefill(tokens: Tensor<[1, 10], u32>) -> u32 {
            return 0;
        }
        // decode scope missing!
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e008_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E008).collect();
    assert!(!e008_diags.is_empty(), "E008 diagnostic not triggered for missing decode");
    assert!(e008_diags[0].message.contains("MissingDecode"));
}

#[test]
fn test_diagnostic_e008_missing_mandatory_prefill_scope() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model MissingPrefill {
        scope decode(token: u32) -> u32 {
            return token;
        }
        // prefill scope missing!
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e008_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E008).collect();
    assert!(!e008_diags.is_empty(), "E008 diagnostic not triggered for missing prefill");
    assert!(e008_diags[0].message.contains("MissingPrefill"));
}

#[test]
fn test_diagnostic_e009_negative_const_generic_parameter() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model NegativeParam<const B: usize = -32> {
        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e009_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E009).collect();
    assert!(!e009_diags.is_empty(), "E009 diagnostic not triggered for negative const generic");
    assert!(e009_diags[0].message.to_lowercase().contains("negative"));
}

#[test]
fn test_diagnostic_e015_invalid_paged_kv_block_size() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model BadKVBlock {
        let mut kv: PagedKVCache<32, 8, 128, 25, f16>; // 25 is not in {16, 32, 64}
        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e015_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E015).collect();
    assert!(!e015_diags.is_empty(), "E015 diagnostic not triggered for block size 25");
    assert!(e015_diags[0].message.contains("25"));
}

#[test]
fn test_diagnostic_e016_unbounded_streaming_sink_window() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model BadStreaming {
        let mut kv: StreamingSinkKVCache<32, 8, 128, 0, 0, f16>; // 0 is invalid
        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e016_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E016).collect();
    assert!(!e016_diags.is_empty(), "E016 diagnostic not triggered for window 0");
    assert!(e016_diags[0].message.to_lowercase().contains("positive"));
}

#[test]
fn test_diagnostic_e022_duplicate_model_member_name() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model DuplicateWeight {
        @vram_immutable weight w_q: Tensor<[4096, 4096], f16> = safetensors("m.safetensors", "q1");
        @vram_immutable weight w_q: Tensor<[4096, 4096], f16> = safetensors("m.safetensors", "q2");
        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let diags = parse_for_diagnostics(code);
    let e022_diags: Vec<&Diagnostic> = diags.iter().filter(|d| d.code == DiagnosticCode::E022).collect();
    assert!(!e022_diags.is_empty(), "E022 diagnostic not triggered for duplicate member");
    assert!(e022_diags[0].message.contains("w_q"));
    assert!(!e022_diags[0].secondary_spans.is_empty(), "Missing secondary span pointer to earlier definition");
}

// ============================================================================
// 2. Complete Catalog Coverage (E001 - E025)
// ============================================================================

#[test]
fn test_all_25_codes_exist_in_catalog() {
    assert_eq!(ALL_CODES.len(), 25);
    for code in &ALL_CODES {
        let code_str = code.as_str();
        assert!(!code_str.is_empty());
        assert!(!code.name().is_empty(), "Empty name for {code_str}");
        assert!(!code.category().is_empty(), "Empty category for {code_str}");
        assert!(!code.explanation().is_empty(), "Empty explanation for {code_str}");
        assert!(!code.remediation().is_empty(), "Empty remediation for {code_str}");
        assert_eq!(DiagnosticCode::from_str(code_str), Some(*code));
    }
}

// ============================================================================
// 3. Visual Formatting Verification for All 25 Codes
// ============================================================================

#[test]
fn test_visual_formatting_all_25_codes() {
    let sample_source = "target hardware { device: \"sm_86\"; }\nmodel Example {\n    let problematic_expression = 42;\n}\n";
    let span = SourceSpan::from_coords("example.vlx", 3, 9, 3, 31, 44, 66);
    let formatter = DiagnosticFormatter::new(false);

    for code in &ALL_CODES {
        let diag = Diagnostic::error(
            *code,
            format!("Synthetic negative test verification for {}", code.as_str()),
            span.clone(),
        );

        let rendered = formatter.format(&diag, sample_source);

        // 1. Box separator: 80 '=' characters
        assert!(rendered.contains(&"=".repeat(80)));

        // 2. Header contains [Exxx] and the official catalog name
        let expected_header = format!("[{}]: {}", code.as_str(), code.name());
        assert!(rendered.contains(&expected_header), "Rendered missing header: {expected_header}");

        // 3. Location line coordinate
        assert!(rendered.contains("Line 3, Column 9"));

        // 4. Source line context is visible
        assert!(rendered.contains("let problematic_expression = 42;"));

        // 5. Visual caret pointer (^) is present
        assert!(rendered.contains('^'));

        // 6. Explanatory block is present
        assert!(rendered.contains("Explanation:"));
        let first_expl_line = code.explanation().lines().next().unwrap();
        assert!(rendered.contains(first_expl_line));

        // 7. Remediation block is present
        assert!(rendered.contains("Remediation:"));
    }
}

#[test]
fn test_visual_formatting_with_ansi_color() {
    let sample = "model M { let bad = 1; }\n";
    let span = SourceSpan::from_coords("m.vlx", 1, 15, 1, 18, 14, 17);
    let diag = Diagnostic::error(DiagnosticCode::E001, "Error message", span);

    let formatter = DiagnosticFormatter::new(true);
    let rendered = formatter.format(&diag, sample);

    // Contains ANSI bold red and reset sequences
    assert!(rendered.contains("\x1b[1;31m"));
    assert!(rendered.contains("\x1b[0m"));
}

#[test]
fn test_visual_formatting_secondary_spans() {
    let sample = "model M {\n    weight w_q: f16;\n    weight w_q: f16;\n}\n";
    let primary_span = SourceSpan::from_coords("dup.vlx", 3, 12, 3, 15, 36, 39);
    let secondary_span = SourceSpan::from_coords("dup.vlx", 2, 12, 2, 15, 15, 18);

    let diag = Diagnostic::error(DiagnosticCode::E022, "Duplicate member 'w_q'", primary_span)
        .with_secondary_span(secondary_span, "Previous definition of 'w_q' was here");

    let formatter = DiagnosticFormatter::new(false);
    let rendered = formatter.format(&diag, sample);
    assert!(rendered.contains("Also noted at Line 2, Column 12: Previous definition of 'w_q' was here"));
}

// ============================================================================
// 4. DiagnosticEngine Mechanics
// ============================================================================

#[test]
fn test_diagnostic_engine_emission_and_filtering() {
    let mut engine = DiagnosticEngine::new("sample", "sample.vlx", false);
    let span = SourceSpan::from_coords("sample.vlx", 1, 1, 1, 5, 0, 4);

    assert!(!engine.has_errors());
    assert_eq!(engine.error_count(), 0);

    // Add warning
    engine.emit_warning(DiagnosticCode::E024, "Warning precision loss", span.clone());
    assert!(!engine.has_errors());
    assert_eq!(engine.error_count(), 0);
    assert_eq!(engine.diagnostics.len(), 1);

    // Add error
    engine.emit_error(DiagnosticCode::E001, "Prohibited allocation", span);
    assert!(engine.has_errors());
    assert_eq!(engine.error_count(), 1);
    assert_eq!(engine.diagnostics.len(), 2);

    // Batch formatting
    let rendered_all = engine.format_all();
    assert!(rendered_all.contains("[E024]"));
    assert!(rendered_all.contains("[E001]"));
}
