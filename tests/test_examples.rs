//! Integration test suite for all 5 reference model scripts in `examples/` (`tests/test_examples.rs`)
//!
//! Production Reference Models:
//! 1. examples/tinystories_smollm.vlx (SmolLM-135M, Llama-style GQA 3:1, D=576, L=30, PagedKVCache)
//! 2. examples/llama3_8b.vlx (Meta Llama-3-8B, GQA 4:1, D=4096, L=32, PagedKVCache, SwiGLU)
//! 3. examples/qwen2_5_7b.vlx (Qwen2.5-7B, GQA 7:1, D=3584, L=28, Tied word embeddings, QKV biases)
//! 4. examples/deepseek_v3_mla.vlx (DeepSeek-V3, D=7168, L=61, MLA dc=512, dR=64, matrix absorption, MoE routing)
//! 5. examples/nanbeige_33b.vlx (Nanbeige 33B, D=6144, L=60, VirtualContiguousKV VMM, Sarathi chunked prefill, SpecInfer tree verification)
//!
//! Verifies:
//! - 0 syntax or validation errors across all 5 models.
//! - Non-empty ProgramNode AST with valid TargetHardwareNode and ModelDeclarationNode.
//! - Correct model names, minimum weights count, and expected scopes count.
//! - ASTPrettyPrinter text rendering completeness.
//! - Pure-Rust JSON schema exportability via to_json().
//! - Model-specific architectural invariants and memory tiers.

use std::fs;
use std::path::Path;
use veloxc::ast::*;
use veloxc::cli::ASTPrettyPrinter;
use veloxc::lexer::Lexer;
use veloxc::parser::Parser;
use veloxc::tokens::TokenType;
use veloxc::types::{ScalarType, ScopeKind};

const MODEL_SPECS: [(&str, &str, usize, usize); 5] = [
    ("tinystories_smollm.vlx", "SmolLM_135M", 12, 2),
    ("llama3_8b.vlx", "Llama3_8B", 12, 2),
    ("qwen2_5_7b.vlx", "Qwen2_5_7B", 14, 2),
    ("deepseek_v3_mla.vlx", "DeepSeek_V3_MLA", 18, 2),
    ("nanbeige_33b.vlx", "Nanbeige_33B", 12, 4),
];

fn parse_reference_file(filename: &str) -> (String, ProgramNode) {
    let path = Path::new("examples").join(filename);
    assert!(path.exists(), "Reference model file does not exist: {path:?}");

    let source = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {filename}: {e}"));
    assert!(source.len() > 100, "Source file {filename} unexpectedly short");

    let mut parser = Parser::from_source(&source, path.to_str().unwrap())
        .unwrap_or_else(|e| panic!("Lexer error in {filename}: {e}"));

    let program = parser
        .parse()
        .unwrap_or_else(|diags| {
            let rendered: Vec<String> = diags.iter().map(|d| format!("[{}]: {}", d.code_str(), d.message)).collect();
            panic!("Parser error in {filename}:\n{}", rendered.join("\n"));
        });

    assert!(
        parser.diagnostics.is_empty(),
        "Expected 0 diagnostics for {filename}, got: {:?}",
        parser.diagnostics
    );

    (source, program)
}

// ============================================================================
// 1. General Tokenization & Parsing Tests Across All 5 Reference Models
// ============================================================================

#[test]
fn test_all_5_reference_models_tokenization_and_parsing() {
    for (filename, expected_model_name, min_weights, min_scopes) in MODEL_SPECS {
        let path = Path::new("examples").join(filename);
        let source = fs::read_to_string(&path).expect("Failed to read file");

        // 1. Tokenize
        let tokens = Lexer::tokenize_source(&source, path.to_str().unwrap())
            .unwrap_or_else(|e| panic!("Lexer failure for {filename}: {e}"));
        assert!(
            tokens.len() > 50,
            "Token stream unexpectedly small for {filename}: {}",
            tokens.len()
        );
        assert_eq!(
            tokens.last().unwrap().token_type,
            TokenType::Eof,
            "Last token must be EOF for {filename}"
        );

        // 2. Parse & Invariant Validation
        let mut parser = Parser::new(tokens, path.to_str().unwrap(), &source);
        let program = parser
            .parse()
            .unwrap_or_else(|diags| panic!("Diagnostics emitted for {filename}: {diags:?}"));

        assert_eq!(
            parser.diagnostics.len(),
            0,
            "Diagnostics emitted for {filename}: {:?}",
            parser.diagnostics
        );

        // 3. Structural checks
        assert!(
            program.target_hardware.is_some(),
            "Target hardware must be declared in {filename}"
        );
        assert_eq!(
            program.model_declarations.len(),
            1,
            "Expected exactly 1 model in {filename}"
        );

        let model = &program.model_declarations[0];
        assert_eq!(
            model.name, expected_model_name,
            "Model name mismatch in {filename}"
        );
        assert!(
            model.weights.len() >= min_weights,
            "Expected at least {min_weights} weights in {filename}, got {}",
            model.weights.len()
        );
        assert!(
            model.scopes.len() >= min_scopes,
            "Expected at least {min_scopes} scopes in {filename}, got {}",
            model.scopes.len()
        );
    }
}

// ============================================================================
// 2. AST Pretty Printer Verification Across All 5 Reference Models
// ============================================================================

#[test]
fn test_all_5_reference_models_pretty_printer() {
    for (filename, expected_model_name, _, _) in MODEL_SPECS {
        let (_, program) = parse_reference_file(filename);

        let printer = ASTPrettyPrinter::new();
        let rendered = printer.print(&program);

        assert!(
            rendered.len() > 500,
            "Pretty-printed output for {filename} unexpectedly short: {}",
            rendered.len()
        );
        assert!(
            rendered.contains("ProgramNode"),
            "Pretty-printed output for {filename} must contain 'ProgramNode'"
        );
        assert!(
            rendered.contains("TargetHardwareNode"),
            "Pretty-printed output for {filename} must contain 'TargetHardwareNode'"
        );
        assert!(
            rendered.contains(expected_model_name),
            "Pretty-printed output for {filename} must contain model name '{expected_model_name}'"
        );
    }
}

// ============================================================================
// 3. Pure-Rust JSON Serialization Across All 5 Reference Models
// ============================================================================

#[test]
fn test_all_5_reference_models_json_serialization() {
    for (filename, expected_model_name, _, _) in MODEL_SPECS {
        let (_, program) = parse_reference_file(filename);

        let json_str = program.to_json();
        assert!(
            json_str.len() > 1000,
            "JSON output for {filename} unexpectedly short: {}",
            json_str.len()
        );

        assert!(
            json_str.contains("\"node_type\": \"ProgramNode\""),
            "JSON must contain ProgramNode node_type for {filename}"
        );
        assert!(
            json_str.contains("\"target_hardware\""),
            "JSON must contain target_hardware key for {filename}"
        );
        assert!(
            json_str.contains("\"model_declarations\""),
            "JSON must contain model_declarations key for {filename}"
        );
        assert!(
            json_str.contains(expected_model_name),
            "JSON must contain model name '{expected_model_name}' for {filename}"
        );

        // Milestone 5 Verification: weights, @vram_immutable, kv_caches, and scopes
        assert!(
            json_str.contains("\"weights\""),
            "JSON must contain weights array for {filename}"
        );
        assert!(
            json_str.contains("vram_immutable"),
            "JSON must contain 'vram_immutable' memory tier for {filename}"
        );
        assert!(
            json_str.contains("\"kv_caches\""),
            "JSON must contain kv_caches array for {filename}"
        );
        assert!(
            json_str.contains("\"scopes\""),
            "JSON must contain scopes array for {filename}"
        );
        assert!(
            json_str.contains("\"prefill\""),
            "JSON must contain 'prefill' scope for {filename}"
        );
        assert!(
            json_str.contains("\"decode\""),
            "JSON must contain 'decode' scope for {filename}"
        );

        // Architecture-specific KV cache identifier assertions
        match expected_model_name {
            "SmolLM_135M" | "Llama3_8B" | "Qwen2_5_7B" => {
                assert!(
                    json_str.contains("PagedKVCache"),
                    "JSON must contain PagedKVCache for {filename}"
                );
            }
            "DeepSeek_V3_MLA" => {
                assert!(
                    json_str.contains("LatentKVCache"),
                    "JSON must contain LatentKVCache for {filename}"
                );
            }
            "Nanbeige_33B" => {
                assert!(
                    json_str.contains("VirtualContiguousKV"),
                    "JSON must contain VirtualContiguousKV for {filename}"
                );
                assert!(
                    json_str.contains("SpeculativeTreeBlock"),
                    "JSON must contain SpeculativeTreeBlock for {filename}"
                );
            }
            _ => {}
        }
    }
}

// ============================================================================
// 4. Model-Specific Architectural Verification
// ============================================================================

#[test]
fn test_tinystories_smollm_specifics() {
    let (_, program) = parse_reference_file("tinystories_smollm.vlx");
    let model = &program.model_declarations[0];
    assert_eq!(model.name, "SmolLM_135M");

    // Target Hardware
    let hw = program.target_hardware.as_ref().unwrap();
    assert_eq!(hw.device_name, "nvidia_rtx_3050_laptop");
    assert_eq!(hw.vram_capacity_bytes, 4096 * 1024 * 1024);
    assert_eq!(hw.compute_capability, (8, 6));
    assert!((hw.safety_factor - 0.95).abs() < 1e-4);

    // KV Cache must be PagedKVTypeNode with block_size=16, dtype=F16
    assert_eq!(model.kv_caches.len(), 1);
    let kv = &model.kv_caches[0];
    let spec = kv.cache_type.as_ref().expect("KV cache must have type");
    if let TypeKind::PagedKV(ref paged) = spec.kind {
        assert_eq!(paged.dtype, ScalarType::F16);
        if let ExprNode::Literal(ref lit) = *paged.block_size {
            assert_eq!(lit.raw_text, "16");
        } else {
            panic!("Expected literal block_size");
        }
    } else {
        panic!("Expected PagedKV cache type for SmolLM");
    }

    // Both prefill and decode scopes present
    let scope_kinds: Vec<ScopeKind> = model.scopes.iter().map(|s| s.kind()).collect();
    assert!(scope_kinds.contains(&ScopeKind::Prefill));
    assert!(scope_kinds.contains(&ScopeKind::Decode));
}

#[test]
fn test_llama3_8b_specifics() {
    let (_, program) = parse_reference_file("llama3_8b.vlx");
    let model = &program.model_declarations[0];
    assert_eq!(model.name, "Llama3_8B");

    // Target Hardware
    let hw = program.target_hardware.as_ref().unwrap();
    assert_eq!(hw.device_name, "nvidia_a100_80gb");
    assert_eq!(hw.vram_capacity_bytes, 80 * 1024 * 1024 * 1024);
    assert_eq!(hw.compute_capability, (8, 0));

    // KV Cache: PagedKVCache with block_size=16
    assert_eq!(model.kv_caches.len(), 1);
    let kv = &model.kv_caches[0];
    let spec = kv.cache_type.as_ref().expect("KV cache must have type");
    if let TypeKind::PagedKV(ref paged) = spec.kind {
        if let ExprNode::Identifier(ref id) = *paged.heads {
            assert_eq!(id.name, "NUM_KV_HEADS");
        }
        if let ExprNode::Identifier(ref id) = *paged.head_dim {
            assert_eq!(id.name, "HEAD_DIM");
        }
        if let ExprNode::Literal(ref lit) = *paged.block_size {
            assert_eq!(lit.raw_text, "16");
        }
    } else {
        panic!("Expected PagedKV cache type for Llama-3");
    }

    let scope_kinds: Vec<ScopeKind> = model.scopes.iter().map(|s| s.kind()).collect();
    assert!(scope_kinds.contains(&ScopeKind::Prefill));
    assert!(scope_kinds.contains(&ScopeKind::Decode));
}

#[test]
fn test_qwen2_5_7b_specifics() {
    let (_, program) = parse_reference_file("qwen2_5_7b.vlx");
    let model = &program.model_declarations[0];
    assert_eq!(model.name, "Qwen2_5_7B");

    // Target Hardware
    let hw = program.target_hardware.as_ref().unwrap();
    assert_eq!(hw.device_name, "nvidia_rtx_4090");
    assert_eq!(hw.vram_capacity_bytes, 24 * 1024 * 1024 * 1024);
    assert_eq!(hw.compute_capability, (8, 9));

    // QKV bias weights must be present
    let weight_names: Vec<&str> = model.weights.iter().map(|w| w.name.as_str()).collect();
    assert!(weight_names.contains(&"b_q"), "Missing b_q bias weight");
    assert!(weight_names.contains(&"b_k"), "Missing b_k bias weight");
    assert!(weight_names.contains(&"b_v"), "Missing b_v bias weight");

    let scope_kinds: Vec<ScopeKind> = model.scopes.iter().map(|s| s.kind()).collect();
    assert!(scope_kinds.contains(&ScopeKind::Prefill));
    assert!(scope_kinds.contains(&ScopeKind::Decode));
}

#[test]
fn test_deepseek_v3_mla_specifics() {
    let (_, program) = parse_reference_file("deepseek_v3_mla.vlx");
    let model = &program.model_declarations[0];
    assert_eq!(model.name, "DeepSeek_V3_MLA");

    // Target Hardware
    let hw = program.target_hardware.as_ref().unwrap();
    assert_eq!(hw.device_name, "nvidia_h100_sxm5_cluster");
    assert_eq!(hw.vram_capacity_bytes, 1280 * 1024 * 1024 * 1024);
    assert_eq!(hw.compute_capability, (9, 0));

    // KV Cache must be LatentKVCache
    assert_eq!(model.kv_caches.len(), 1);
    let kv = &model.kv_caches[0];
    let spec = kv.cache_type.as_ref().expect("KV cache must have type");
    if let TypeKind::LatentKV(ref latent) = spec.kind {
        if let ExprNode::Identifier(ref id) = *latent.latent_dim {
            assert_eq!(id.name, "LATENT_KV_DIM");
        }
        if let ExprNode::Identifier(ref id) = *latent.rope_dim {
            assert_eq!(id.name, "ROPE_DIM");
        }
    } else {
        panic!("Expected LatentKVCache type for DeepSeek-V3");
    }

    // Absorbed MLA weights must be present
    let absorbed_weights: Vec<&WeightDeclarationNode> = model
        .weights
        .iter()
        .filter(|w| matches!(&w.binding, Some(WeightBinding::AbsorbMla(_))))
        .collect();
    assert!(
        absorbed_weights.len() >= 2,
        "Expected at least 2 absorbed MLA weights, found {}",
        absorbed_weights.len()
    );
    let absorbed_names: Vec<&str> = absorbed_weights.iter().map(|w| w.name.as_str()).collect();
    assert!(absorbed_names.contains(&"w_absorbed_q"));
    assert!(absorbed_names.contains(&"w_absorbed_o"));
}

#[test]
fn test_nanbeige_33b_specifics() {
    let (_, program) = parse_reference_file("nanbeige_33b.vlx");
    let model = &program.model_declarations[0];
    assert_eq!(model.name, "Nanbeige_33B");

    // Target Hardware
    let hw = program.target_hardware.as_ref().unwrap();
    assert_eq!(hw.device_name, "nvidia_h100_node");
    assert_eq!(hw.vram_capacity_bytes, 320 * 1024 * 1024 * 1024);
    assert_eq!(hw.compute_capability, (9, 0));

    // All 4 phase-disjoint execution scopes must be present
    let scope_kinds: Vec<ScopeKind> = model.scopes.iter().map(|s| s.kind()).collect();
    assert!(scope_kinds.contains(&ScopeKind::Prefill), "Missing Prefill scope");
    assert!(scope_kinds.contains(&ScopeKind::Decode), "Missing Decode scope");
    assert!(
        scope_kinds.contains(&ScopeKind::ChunkedPrefill),
        "Missing ChunkedPrefill scope"
    );
    assert!(
        scope_kinds.contains(&ScopeKind::SpeculativeVerify),
        "Missing SpeculativeVerify scope"
    );

    // KV Caches: VirtualContiguousKV and SpeculativeTreeBlock
    assert_eq!(model.kv_caches.len(), 2);
    let mut has_virtual_kv = false;
    let mut has_speculative_tree = false;
    for kv in &model.kv_caches {
        let spec = kv.cache_type.as_ref().expect("KV cache must have type");
        match &spec.kind {
            TypeKind::VirtualKV(_) => has_virtual_kv = true,
            TypeKind::SpeculativeTree(_) => has_speculative_tree = true,
            _ => {}
        }
    }
    assert!(has_virtual_kv, "Missing VirtualContiguousKV cache in Nanbeige");
    assert!(
        has_speculative_tree,
        "Missing SpeculativeTreeBlock cache in Nanbeige"
    );
}
