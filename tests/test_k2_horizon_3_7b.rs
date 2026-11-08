//! Comprehensive Unit and Integration Test Suite for K2-Horizon-3.7B Foundation Model
//! File: tests/test_k2_horizon_3_7b.rs
//!
//! Verifies:
//! 1. Lexing and tokenization of `examples/k2_horizon_3_7b.vlx`.
//! 2. Safe AST parsing, const-generic model parameters, and weight bindings.
//! 3. AST JSON serialization and pretty-printing.
//! 4. Semantic analysis, type checking, shape inference, and tier validation.
//! 5. Static memory planning, interval-graph coloring, VRAM proof certificate on RTX 3050 (4GB),
//!    and Host RAM calculation (~45.0 GiB for 512K context).
//! 6. Velox IR (VIR) lowering, kernel fusion, and DAG monomorphization.
//! 7. Zero-allocation C99 and CUDA emission, candidate-stride two-stage tree argmax kernel ($V = 250,624$).

use std::fs;
use std::path::Path;
use veloxc::cli::ASTPrettyPrinter;
use veloxc::codegen::CodeGenerator;
use veloxc::ir::{lower_program_to_vir, VirInst};
use veloxc::lexer::Lexer;
use veloxc::memory_planner::StaticMemoryPlanner;
use veloxc::optimizer::optimize_vir_module;
use veloxc::parser::Parser;
use veloxc::sema::SemanticAnalyzer;
use veloxc::tokens::TokenType;
use veloxc::types::{MemoryTier, ScalarType, ScopeKind};

const K2_PATH: &str = "examples/k2_horizon_3_7b.vlx";

fn load_k2_source() -> (String, &'static str) {
    let path = Path::new(K2_PATH);
    assert!(path.exists(), "K2-Horizon-3.7B example file must exist: {path:?}");
    let source = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {K2_PATH}: {e}"));
    (source, K2_PATH)
}

#[test]
fn test_k2_horizon_3_7b_tokenization_and_parsing() {
    let (source, file_path) = load_k2_source();

    // 1. Tokenize source
    let tokens = Lexer::tokenize_source(&source, file_path)
        .expect("Lexing K2-Horizon-3.7B must succeed with 0 errors");
    assert!(
        tokens.len() > 100,
        "Token stream unexpectedly small: {}",
        tokens.len()
    );
    assert_eq!(
        tokens.last().unwrap().token_type,
        TokenType::Eof,
        "Last token must be EOF"
    );

    // 2. Parse into AST
    let mut parser = Parser::new(tokens, file_path, &source);
    let program = parser
        .parse_program_safe()
        .expect("Parsing K2-Horizon-3.7B must succeed with 0 diagnostics");

    assert!(
        parser.diagnostics.is_empty(),
        "Expected 0 parser diagnostics, got: {:?}",
        parser.diagnostics
    );

    // 3. Verify TargetHardwareNode invariants
    let hw = program
        .target_hardware
        .as_ref()
        .expect("Program must define target hardware");
    assert_eq!(hw.device_name, "nvidia_rtx_3050_laptop");
    assert_eq!(
        hw.vram_capacity_bytes,
        4096 * 1024 * 1024,
        "Expected 4096 MiB VRAM capacity"
    );
    assert_eq!(hw.compute_capability, (8, 6));
    assert!((hw.safety_factor - 0.95).abs() < 1e-6);

    // 4. Verify ModelDeclarationNode invariants
    assert_eq!(program.model_declarations.len(), 1);
    let model = &program.model_declarations[0];
    assert_eq!(model.name, "K2_Horizon_3_7B");

    // Const-generic parameters
    let const_names: Vec<&str> = model.generic_params.iter().map(|p| p.name.as_str()).collect();
    for expected in [
        "B_MAX",
        "S_MAX",
        "VOCAB",
        "HIDDEN_DIM",
        "INTERMEDIATE_DIM",
        "NUM_LAYERS",
        "NUM_HEADS",
        "NUM_KV_HEADS",
        "HEAD_DIM",
        "ROPE_THETA",
    ] {
        assert!(
            const_names.contains(&expected),
            "Missing const generic param '{expected}'"
        );
    }

    // Weight count and names
    assert_eq!(model.weights.len(), 12);
    let weight_names: Vec<&str> = model.weights.iter().map(|w| w.name.as_str()).collect();
    for expected_weight in [
        "w_embed",
        "w_q",
        "w_k",
        "w_v",
        "w_o",
        "w_gate",
        "w_up",
        "w_down",
        "w_input_norm",
        "w_post_norm",
        "w_norm",
        "w_lm_head",
    ] {
        assert!(
            weight_names.contains(&expected_weight),
            "Missing weight '{expected_weight}'"
        );
    }

    // Check INT4 dtypes on weights
    for w in &model.weights {
        assert_eq!(
            w.tier,
            MemoryTier::VramImmutable,
            "Weight {} must reside in @vram_immutable",
            w.name
        );
    }

    // Check KV cache declaration with @host_pinned tier
    assert_eq!(model.kv_caches.len(), 1);
    let kv = &model.kv_caches[0];
    assert_eq!(kv.name, "kv_cache");
    assert!(kv.is_mut);
    let cache_spec = kv.cache_type.as_ref().expect("KV cache must have type spec");
    assert_eq!(
        cache_spec.tier,
        Some(MemoryTier::HostPinned),
        "K2-Horizon-3.7B KV cache must be @host_pinned for 512K context window"
    );

    // Check Scopes
    assert_eq!(model.scopes.len(), 2);
    assert!(model.scopes.iter().any(|s| s.kind() == ScopeKind::Prefill));
    assert!(model.scopes.iter().any(|s| s.kind() == ScopeKind::Decode));
}

#[test]
fn test_k2_horizon_3_7b_ast_json_and_pretty_printer() {
    let (source, file_path) = load_k2_source();
    let tokens = Lexer::tokenize_source(&source, file_path).unwrap();
    let mut parser = Parser::new(tokens, file_path, &source);
    let program = parser.parse_program_safe().unwrap();

    // AST Pretty Printer
    let printer = ASTPrettyPrinter::new();
    let printed = printer.print(&program);
    assert!(printed.contains("K2_Horizon_3_7B"));
    assert!(printed.contains("nvidia_rtx_3050_laptop"));
    assert!(printed.contains("w_embed"));
    assert!(printed.contains("kv_cache"));
    assert!(printed.contains("prefill"));
    assert!(printed.contains("decode"));

    // Pure Rust JSON Serialization
    let json_str = program.to_json();
    assert!(json_str.contains("\"node_type\": \"ProgramNode\""));
    assert!(json_str.contains("\"name\": \"K2_Horizon_3_7B\""));
    assert!(json_str.contains("\"safety_factor\": 0.95"));
}

#[test]
fn test_k2_horizon_3_7b_semantic_analysis_and_shape_inference() {
    let (source, file_path) = load_k2_source();
    let tokens = Lexer::tokenize_source(&source, file_path).unwrap();
    let mut parser = Parser::new(tokens, file_path, &source);
    let program = parser.parse_program_safe().unwrap();

    let typed = SemanticAnalyzer::new()
        .analyze(&program)
        .expect("Semantic analysis must succeed for K2-Horizon-3.7B");

    let typed_model = &typed.models[0];
    assert_eq!(typed_model.name, "K2_Horizon_3_7B");
    assert_eq!(typed_model.constants.get("HIDDEN_DIM"), Some(&2560));
    assert_eq!(typed_model.constants.get("INTERMEDIATE_DIM"), Some(&10240));
    assert_eq!(typed_model.constants.get("NUM_LAYERS"), Some(&36));
    assert_eq!(typed_model.constants.get("NUM_HEADS"), Some(&32));
    assert_eq!(typed_model.constants.get("NUM_KV_HEADS"), Some(&8));
    assert_eq!(typed_model.constants.get("HEAD_DIM"), Some(&80));
    assert_eq!(typed_model.constants.get("VOCAB"), Some(&250624));
    assert_eq!(typed_model.constants.get("S_MAX"), Some(&524288));
    assert_eq!(typed_model.constants.get("ROPE_THETA"), Some(&500000));

    // Verify concrete weight shapes
    let w_embed = typed_model.weights.iter().find(|w| w.name == "w_embed").unwrap();
    assert_eq!(w_embed.shape, vec![250624, 2560]);
    assert_eq!(w_embed.dtype, ScalarType::Int4);

    let w_q = typed_model.weights.iter().find(|w| w.name == "w_q").unwrap();
    assert_eq!(w_q.shape, vec![36, 2560, 2560]);
    assert_eq!(w_q.dtype, ScalarType::Int4);

    let w_k = typed_model.weights.iter().find(|w| w.name == "w_k").unwrap();
    assert_eq!(w_k.shape, vec![36, 2560, 640]);
    assert_eq!(w_k.dtype, ScalarType::Int4);

    let w_v = typed_model.weights.iter().find(|w| w.name == "w_v").unwrap();
    assert_eq!(w_v.shape, vec![36, 2560, 640]);
    assert_eq!(w_v.dtype, ScalarType::Int4);

    let w_gate = typed_model.weights.iter().find(|w| w.name == "w_gate").unwrap();
    assert_eq!(w_gate.shape, vec![36, 2560, 10240]);
    assert_eq!(w_gate.dtype, ScalarType::Int4);

    let w_norm = typed_model.weights.iter().find(|w| w.name == "w_norm").unwrap();
    assert_eq!(w_norm.shape, vec![2560]);
    assert_eq!(w_norm.dtype, ScalarType::F16);

    let w_lm_head = typed_model.weights.iter().find(|w| w.name == "w_lm_head").unwrap();
    assert_eq!(w_lm_head.shape, vec![2560, 250624]);
    assert_eq!(w_lm_head.dtype, ScalarType::Int4);

    // Verify KV cache is HostPinned tier
    let kv = &typed_model.kv_caches[0];
    assert_eq!(kv.name, "kv_cache");
    assert_eq!(kv.tier, MemoryTier::HostPinned);
    assert_eq!(kv.layers, 36);
    assert_eq!(kv.heads, 8);
    assert_eq!(kv.head_dim, 80);
    assert_eq!(kv.max_seq_len, 524288);
}

#[test]
fn test_k2_horizon_3_7b_static_memory_planning_and_vram_feasibility() {
    let (source, file_path) = load_k2_source();
    let tokens = Lexer::tokenize_source(&source, file_path).unwrap();
    let mut parser = Parser::new(tokens, file_path, &source);
    let program = parser.parse_program_safe().unwrap();
    let typed = SemanticAnalyzer::new().analyze(&program).unwrap();

    let planner = StaticMemoryPlanner::new();
    let plans = planner
        .plan_program(&typed)
        .expect("Static memory planner must succeed for K2-Horizon-3.7B");

    assert_eq!(plans.len(), 1);
    let plan = &plans[0];

    // RTX 3050 4GB VRAM Proof Invariants
    assert!(
        plan.is_feasible,
        "K2-Horizon-3.7B MUST be verified feasible on RTX 3050 (4GB)"
    );
    assert_eq!(plan.vram_capacity_bytes, 4096 * 1024 * 1024);
    assert_eq!(plan.safe_budget_bytes, (4096.0 * 1024.0 * 1024.0 * 0.95) as u64);
    assert!(
        plan.total_required_vram_bytes <= plan.safe_budget_bytes,
        "Total required VRAM ({} B) must not exceed safe budget ({} B)",
        plan.total_required_vram_bytes,
        plan.safe_budget_bytes
    );

    // Check VRAM components
    assert!(plan.weight_arena_bytes > 0);
    assert_eq!(
        plan.kv_cache_pool_bytes, 0,
        "KV cache pool in VRAM must be 0 bytes when offloaded to @host_pinned"
    );
    assert!(plan.activation_arena_bytes > 0);
    assert_eq!(plan.system_reserve_bytes, 256 * 1024 * 1024); // 256 MiB reserve

    // Check Host RAM requirement for 512K tokens (36 layers * 2 * 524288 * 8 * 80 * 2 B = ~45.0 GiB)
    assert!(
        plan.host_ram_bytes > 40 * 1024 * 1024 * 1024,
        "Host RAM requirement ({} B) must be ~45.0 GiB for 512K context",
        plan.host_ram_bytes
    );

    // Check Proof Certificate
    let proof = plan.format_proof_certificate();
    assert!(proof.contains("VERIFIED FEASIBLE (Q.E.D.)"));
    assert!(proof.contains("VELOX_HOST_RAM_REQUIRED_BYTES"));
    assert!(proof.contains("VELOX_HARDWARE_VRAM_CAPACITY_BYTES   4294967296ULL"));
    assert!(proof.contains("VELOX_SAFE_VRAM_BUDGET_BYTES         4080218931ULL"));
    assert!(proof.contains("VELOX_TIER2_KV_CACHE_POOL_BYTES      0ULL"));
    assert!(proof.contains("VELOX_TIER4_SYSTEM_RESERVE_BYTES     268435456ULL"));

    // Verify 256-byte alignment across all weights and activations
    for w in &plan.weight_offsets {
        assert_eq!(
            w.offset_bytes % 256,
            0,
            "Weight {} offset must be 256-byte aligned",
            w.name
        );
        assert_eq!(
            w.aligned_size_bytes % 256,
            0,
            "Weight {} aligned size must be a multiple of 256",
            w.name
        );
    }
}

#[test]
fn test_k2_horizon_3_7b_vir_lowering_and_optimization() {
    let (source, file_path) = load_k2_source();
    let tokens = Lexer::tokenize_source(&source, file_path).unwrap();
    let mut parser = Parser::new(tokens, file_path, &source);
    let program = parser.parse_program_safe().unwrap();
    let typed = SemanticAnalyzer::new().analyze(&program).unwrap();
    let plans = StaticMemoryPlanner::new().plan_program(&typed).unwrap();

    let mut vir_modules = lower_program_to_vir(&typed, &plans);
    assert_eq!(vir_modules.len(), 1);
    let vir = &mut vir_modules[0];

    assert_eq!(vir.model_name, "K2_Horizon_3_7B");
    assert_eq!(vir.device_name, "nvidia_rtx_3050_laptop");
    assert_eq!(vir.functions.len(), 2);

    let report = optimize_vir_module(vir);
    assert!(vir.kernel_fusion_applied);
    assert!(vir.dag_monomorphized);
    assert!(report.fused_instruction_count <= report.unfused_instruction_count);

    // Verify synthesized fused attention and SwiGLU residual kernels
    let has_fused_attn = vir.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|i| matches!(i, VirInst::FusedNormQkvRopeAttn { .. }))
        })
    });
    assert!(
        has_fused_attn,
        "Expected FusedNormQkvRopeAttn instruction in optimized VIR"
    );

    let has_fused_swiglu = vir.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|i| matches!(i, VirInst::FusedOutProjResSwiGluRes { .. }))
        })
    });
    assert!(
        has_fused_swiglu,
        "Expected FusedOutProjResSwiGluRes instruction in optimized VIR"
    );
}

#[test]
fn test_k2_horizon_3_7b_codegen_c99_and_cuda_zero_heap_alloc() {
    let (source, file_path) = load_k2_source();
    let tokens = Lexer::tokenize_source(&source, file_path).unwrap();
    let mut parser = Parser::new(tokens, file_path, &source);
    let program = parser.parse_program_safe().unwrap();
    let typed = SemanticAnalyzer::new().analyze(&program).unwrap();
    let plans = StaticMemoryPlanner::new().plan_program(&typed).unwrap();

    let mut vir_modules = lower_program_to_vir(&typed, &plans);
    let vir = &mut vir_modules[0];
    let _ = optimize_vir_module(vir);

    // 1. C99 Codegen
    let c99 = CodeGenerator::emit_c99(vir);
    assert!(c99.contains("#define VELOX_MODEL_HIDDEN_DIM               2560ULL"));
    assert!(c99.contains("#define VELOX_MODEL_INTERMEDIATE_DIM         10240ULL"));
    assert!(c99.contains("#define VELOX_MODEL_NUM_LAYERS               36ULL"));
    assert!(c99.contains("#define VELOX_MODEL_NUM_HEADS                32ULL"));
    assert!(c99.contains("#define VELOX_MODEL_NUM_KV_HEADS             8ULL"));
    assert!(c99.contains("#define VELOX_MODEL_HEAD_DIM                 80ULL"));
    assert!(c99.contains("#define VELOX_MODEL_VOCAB_SIZE               250624ULL"));
    assert!(c99.contains("#define VELOX_MODEL_MAX_SEQ_LEN              524288ULL"));
    assert!(c99.contains("#define VELOX_MODEL_ROPE_THETA               500000.0f"));
    assert!(c99.contains("velox_K2_Horizon_3_7B_prefill"));
    assert!(c99.contains("velox_K2_Horizon_3_7B_decode"));

    // 2. CUDA Codegen
    let cuda = CodeGenerator::emit_cuda(vir);
    assert!(cuda.contains("#define VELOX_HIDDEN_DIM                     2560ULL"));
    assert!(cuda.contains("#define VELOX_INTERMEDIATE_DIM               10240ULL"));
    assert!(cuda.contains("#define VELOX_NUM_LAYERS                     36ULL"));
    assert!(cuda.contains("#define VELOX_VOCAB_SIZE                     250624ULL"));
    assert!(cuda.contains("#define VELOX_MODEL_ROPE_THETA               500000.0f"));

    // Verify Candidate-Stride Two-Stage Tree Argmax Kernel for V = 250,624
    assert!(cuda.contains("velox_cuda_lm_head_argmax_stage1"));
    assert!(cuda.contains("velox_cuda_lm_head_argmax_stage2"));
    assert!(cuda.contains("dim3((VELOX_VOCAB_SIZE + 255ULL) / 256ULL)"));
    assert!(cuda.contains("for (size_t c = tid; c < num_candidates; c += blockDim.x)"));

    // 3. Strict 0 Dynamic Heap Allocations Invariant
    for forbidden in [
        "malloc(",
        "free(",
        "calloc(",
        "realloc(",
        "cudaMalloc(",
        "cudaFree(",
    ] {
        assert!(
            !c99.contains(forbidden),
            "Forbidden dynamic memory allocation '{forbidden}' found in generated C99 code"
        );
        assert!(
            !cuda.contains(forbidden),
            "Forbidden dynamic memory allocation '{forbidden}' found in generated CUDA code"
        );
    }
}
