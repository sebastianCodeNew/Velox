//! Integration Tests for Step 6: Semantic Analysis, Shape Inference, Symbolic Unification,
//! Interval Graph Coloring, and Compile-Time VRAM Feasibility Proofs (`veloxc::sema` & `veloxc::memory_planner`).

use veloxc::cli::{run_cli_with_io, EXIT_COMPILATION_ERROR, EXIT_SUCCESS};
use veloxc::diagnostics::DiagnosticCode;
use veloxc::lexer::Lexer;
use veloxc::memory_planner::StaticMemoryPlanner;
use veloxc::parser::Parser;
use veloxc::sema::SemanticAnalyzer;

fn analyze_source(src: &str) -> Result<veloxc::sema::TypedProgram, Vec<veloxc::diagnostics::Diagnostic>> {
    let tokens = Lexer::tokenize_source(src, "test.vlx").expect("Lexing must succeed");
    let mut parser = Parser::new(tokens, "test.vlx", src);
    let program = parser.parse_program_safe().expect("Parsing must succeed");
    assert!(
        parser.diagnostics.is_empty(),
        "Unexpected parser diagnostics: {:?}",
        parser.diagnostics
    );
    let sema = SemanticAnalyzer::new();
    sema.analyze(&program)
}

#[test]
fn test_step6_all_5_reference_models_sema_and_vram_proofs() {
    let models = [
        ("examples/tinystories_smollm.vlx", 4096 * 1024 * 1024u64),
        ("examples/llama3_8b.vlx", 80 * 1024 * 1024 * 1024u64),
        ("examples/qwen2_5_7b.vlx", 24 * 1024 * 1024 * 1024u64),
        ("examples/deepseek_v3_mla.vlx", 1280 * 1024 * 1024 * 1024u64),
        ("examples/nanbeige_33b.vlx", 320 * 1024 * 1024 * 1024u64),
    ];

    for (path, expected_vram) in models {
        let src = std::fs::read_to_string(path).expect("Reference model file must exist");
        let tokens = Lexer::tokenize_source(&src, path).expect("Tokenization must succeed");
        let mut parser = Parser::new(tokens, path, &src);
        let program = parser.parse_program_safe().expect("Parsing must succeed");
        assert!(parser.diagnostics.is_empty(), "Parser diagnostics in {path}");

        let typed = SemanticAnalyzer::new()
            .analyze(&program)
            .unwrap_or_else(|e| panic!("Sema failed on {path}: {e:?}"));
        let plans = StaticMemoryPlanner::new()
            .plan_program(&typed)
            .unwrap_or_else(|e| panic!("Memory planner failed on {path}: {e:?}"));

        assert_eq!(plans.len(), 1);
        let plan = &plans[0];
        assert_eq!(plan.vram_capacity_bytes, expected_vram);
        assert!(plan.is_feasible, "Model {path} must be VRAM-feasible");
        assert!(plan.total_required_vram_bytes <= plan.safe_budget_bytes);

        // Verify 256-byte alignment and non-overlapping interval coloring in every scope
        for scope_plan in &plan.scope_plans {
            for act in &scope_plan.allocations {
                assert_eq!(
                    act.offset_bytes % 256,
                    0,
                    "Activation {} in {}:{} must be 256-byte aligned",
                    act.name,
                    path,
                    scope_plan.scope_name
                );
                assert_eq!(
                    act.aligned_size_bytes % 256,
                    0,
                    "Activation size {} in {}:{} must be padded to 256-byte boundary",
                    act.name,
                    path,
                    scope_plan.scope_name
                );
            }

            // Pairwise verify that any two activations with overlapping lifetime intervals
            // have strictly disjoint byte ranges in the Velox Activation Arena
            for i in 0..scope_plan.allocations.len() {
                for j in (i + 1)..scope_plan.allocations.len() {
                    let a = &scope_plan.allocations[i];
                    let b = &scope_plan.allocations[j];
                    if a.aligned_size_bytes == 0 || b.aligned_size_bytes == 0 {
                        continue;
                    }
                    let intervals_overlap =
                        a.birth_step <= b.death_step && b.birth_step <= a.death_step;
                    if intervals_overlap {
                        let a_end = a.offset_bytes + a.aligned_size_bytes;
                        let b_end = b.offset_bytes + b.aligned_size_bytes;
                        let memory_disjoint = a_end <= b.offset_bytes || b_end <= a.offset_bytes;
                        assert!(
                            memory_disjoint,
                            "Overlapping live activations '{}' [{}, {}] and '{}' [{}, {}] share memory [{}, {}) vs [{}, {}) in {}:{}",
                            a.name,
                            a.birth_step,
                            a.death_step,
                            b.name,
                            b.birth_step,
                            b.death_step,
                            a.offset_bytes,
                            a_end,
                            b.offset_bytes,
                            b_end,
                            path,
                            scope_plan.scope_name
                        );
                    }
                }
            }
        }

        let cert = plan.format_proof_certificate();
        assert!(cert.contains("VELOX COMPILE-TIME VRAM FEASIBILITY PROOF CERTIFICATE"));
        assert!(cert.contains("VERIFIED FEASIBLE (Q.E.D.)"));
    }
}

#[test]
fn test_step6_diagnostic_e004_mutation_of_vram_immutable_weight() {
    let src = r#"
target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
model MutateWeight {
    @vram_immutable weight w: Tensor<[16, 16], f16> = safetensors("m.safetensors", "w");
    scope prefill(x: Tensor<[16, 16], f16>) -> Tensor<[16, 16], f16> {
        w = x;
        return x;
    }
    scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
}
"#;
    let err = analyze_source(src).expect_err("Should fail with E004");
    assert!(err.iter().any(|d| d.code == DiagnosticCode::E004));
}

#[test]
fn test_step6_diagnostic_e005_invalid_dma_transfer_path() {
    let src = r#"
target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
model BadDma {
    scope prefill(
        src_sram: @bump_sram Tensor<[16, 16], f16>,
        dst_host: @host_pinned Tensor<[16, 16], f16>
    ) -> Tensor<[16, 16], f16> {
        dma_async_h2d(src_sram, dst_host);
        return src_sram;
    }
    scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
}
"#;
    let err = analyze_source(src).expect_err("Should fail with E005");
    assert!(err.iter().any(|d| d.code == DiagnosticCode::E005));
}

#[test]
fn test_step6_diagnostic_e006_sram_reference_escape() {
    let src = r#"
target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
model SramEscape {
    scope prefill(sram_tile: @bump_sram Tensor<[16, 16], f16>) -> &Tensor<[16, 16], f16> {
        return &sram_tile;
    }
    scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
}
"#;
    let err = analyze_source(src).expect_err("Should fail with E006");
    assert!(err.iter().any(|d| d.code == DiagnosticCode::E006));
}

#[test]
fn test_step6_diagnostic_e011_broadcast_shape_mismatch() {
    let src = r#"
target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
model BroadcastFail {
    scope prefill(a: Tensor<[4, 8], f16>, b: Tensor<[3, 7], f16>) -> Tensor<[4, 8], f16> {
        let c: Tensor<[4, 8], f16> = a + b;
        return c;
    }
    scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
}
"#;
    let err = analyze_source(src).expect_err("Should fail with E011");
    assert!(err.iter().any(|d| d.code == DiagnosticCode::E011));
}

#[test]
fn test_step6_diagnostic_e012_kv_cache_borrow_aliasing() {
    let src = r#"
target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
model KVAlias {
    let mut kv: PagedKVCache<4, 4, 64, 16, f16>;
    scope prefill(k: Tensor<[1, 8, 4, 64], f16>, v: Tensor<[1, 8, 4, 64], f16>) -> Tensor<[1, 1], u32> {
        let slot1 = kv.borrow_mut_slot(0, 0);
        let slot2 = kv.borrow_mut_slot(0, 0);
        let out: Tensor<[1, 1], u32> = argmax(k, -1);
        return out;
    }
    scope decode(x: Tensor<[1, 1], u32>) -> Tensor<[1, 1], u32> { return x; }
}
"#;
    let err = analyze_source(src).expect_err("Should fail with E012");
    assert!(err.iter().any(|d| d.code == DiagnosticCode::E012));
}

#[test]
fn test_step6_diagnostic_e020_missing_tma_barrier_wait() {
    let src = r#"
target hardware { device: "sm_90"; vram_capacity: 8192 * MiB; compute_capability: 9.0; }
model MissingBarrier {
    @vram_immutable weight w: Tensor<[64, 64], f16> = safetensors("m.safetensors", "w");
    scope prefill(x: Tensor<[1, 64], f16>, bar: mbarrier) -> Tensor<[1, 64], f16> {
        let mut tile: @bump_sram Tensor<[64, 64], f16>;
        tma_load_async(tile, w, bar);
        return x;
    }
    scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
}
"#;
    let err = analyze_source(src).expect_err("Should fail with E020");
    assert!(err.iter().any(|d| d.code == DiagnosticCode::E020));
}

#[test]
fn test_step6_diagnostic_e024_implicit_narrowing_cast() {
    let src = r#"
target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
model NarrowingCast {
    scope prefill(x: Tensor<[4, 64], f32>) -> Tensor<[4, 64], f16> {
        let y: Tensor<[4, 64], f16> = x;
        return y;
    }
    scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
}
"#;
    let err = analyze_source(src).expect_err("Should fail with E024");
    assert!(err.iter().any(|d| d.code == DiagnosticCode::E024));
}

#[test]
fn test_step6_cli_prove_vram_flag() {
    let mut out = Vec::new();
    let mut err = Vec::new();
    let args = vec![
        "--prove-vram".to_string(),
        "examples/tinystories_smollm.vlx".to_string(),
    ];
    let code = run_cli_with_io(&args, &mut out, &mut err);
    assert_eq!(code, EXIT_SUCCESS);
    let stdout_str = String::from_utf8(out).unwrap();
    assert!(stdout_str.contains("#define VELOX_VRAM_PROOF_SMOLLM_135M_H"));
    assert!(stdout_str.contains("VERIFIED FEASIBLE (Q.E.D.)"));
    assert!(stdout_str.contains("VELOX COMPILE-TIME VRAM FEASIBILITY PROOF CERTIFICATE"));

    // Verify E002 CLI exit code when VRAM budget is overflowed
    let tmp_dir = std::env::temp_dir();
    let bad_vram_path = tmp_dir.join("velox_test_e002_cli.vlx");
    std::fs::write(
        &bad_vram_path,
        r#"
target hardware { device: "tiny"; vram_capacity: 1 * MiB; compute_capability: 8.6; }
model TooBig {
    @vram_immutable weight w: Tensor<[4096, 4096], f32> = safetensors("m.safetensors", "w");
    scope prefill(x: Tensor<[1, 4096], f32>) -> Tensor<[1, 4096], f32> { return x @ w; }
    scope decode(x: Tensor<[1, 4096], f32>) -> Tensor<[1, 4096], f32> { return x @ w; }
}
"#,
    )
    .unwrap();
    let mut out2 = Vec::new();
    let mut err2 = Vec::new();
    let args2 = vec![
        "--prove-vram".to_string(),
        bad_vram_path.to_string_lossy().to_string(),
    ];
    let code2 = run_cli_with_io(&args2, &mut out2, &mut err2);
    let _ = std::fs::remove_file(&bad_vram_path);
    assert_eq!(code2, EXIT_COMPILATION_ERROR);
    let stderr_str = String::from_utf8(err2).unwrap();
    assert!(stderr_str.contains("E002"));
}
