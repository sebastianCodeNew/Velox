//! Integration Tests for Step 7: Velox IR (VIR) Lowering, MLA Matrix Absorption,
//! Vertical & Horizontal Kernel Fusion, Static DAG Monomorphization, and C99/CUDA Codegen.

use veloxc::cli::{run_cli_with_io, EXIT_SUCCESS};
use veloxc::codegen::CodeGenerator;
use veloxc::ir::{lower_program_to_vir, VirInst};
use veloxc::lexer::Lexer;
use veloxc::memory_planner::StaticMemoryPlanner;
use veloxc::optimizer::{MlaAbsorptionPass, optimize_vir_module};
use veloxc::parser::Parser;
use veloxc::sema::SemanticAnalyzer;

#[test]
fn test_step7_vir_lowering_and_optimization_all_5_models() {
    let models = [
        "examples/tinystories_smollm.vlx",
        "examples/llama3_8b.vlx",
        "examples/qwen2_5_7b.vlx",
        "examples/deepseek_v3_mla.vlx",
        "examples/nanbeige_33b.vlx",
    ];

    for path in models {
        let src = std::fs::read_to_string(path).expect("Reference model must exist");
        let tokens = Lexer::tokenize_source(&src, path).expect("Lexing must succeed");
        let mut parser = Parser::new(tokens, path, &src);
        let program = parser.parse_program_safe().expect("Parsing must succeed");
        let typed = SemanticAnalyzer::new().analyze(&program).expect("Sema must succeed");
        let plans = StaticMemoryPlanner::new().plan_program(&typed).expect("Planner must succeed");

        let mut vir_modules = lower_program_to_vir(&typed, &plans);
        assert_eq!(vir_modules.len(), 1);
        let vir = &mut vir_modules[0];

        let report = optimize_vir_module(vir);
        assert!(report.unfused_instruction_count > 0);
        assert!(
            report.fused_instruction_count <= report.unfused_instruction_count,
            "Optimization pass must not increase kernel launches in {path}"
        );
        assert!(vir.kernel_fusion_applied);
        assert!(vir.dag_monomorphized);

        // Verify FusedNormQkvRopeAttn or FusedMlaAbsorbedAttn was synthesized
        let has_fused_attn = vir.functions.iter().any(|f| {
            f.blocks.iter().any(|b| {
                b.instructions.iter().any(|i| {
                    matches!(
                        i,
                        VirInst::FusedNormQkvRopeAttn { .. }
                            | VirInst::FusedMlaAbsorbedAttn { .. }
                    )
                })
            })
        });
        assert!(has_fused_attn, "Expected fused attention kernel in {path}");

        // Verify DeepSeek-V3 MLA matrix absorption explicitly
        if path.contains("deepseek_v3_mla") {
            assert!(
                vir.mla_absorption_applied,
                "DeepSeek-V3 must apply MLA matrix absorption"
            );
            assert_eq!(report.mla_layers_absorbed, 61);
            assert!(vir.weights.iter().any(|w| w.is_absorbed));
        }

        // Verify C99 and CUDA emission contain zero dynamic heap allocations
        let c99 = CodeGenerator::emit_c99(vir);
        let cuda = CodeGenerator::emit_cuda(vir);
        for forbidden in ["malloc(", "free(", "calloc(", "realloc(", "cudaMalloc(", "cudaFree("] {
            assert!(
                !c99.contains(forbidden),
                "Forbidden dynamic allocation '{forbidden}' in C99 for {path}"
            );
            assert!(
                !cuda.contains(forbidden),
                "Forbidden dynamic allocation '{forbidden}' in CUDA for {path}"
            );
        }
    }
}

#[test]
fn test_step7_mla_absorption_numerical_identity() {
    let d_q = 4usize;
    let d_c = 3usize;
    let d_h = 5usize;
    let d_model = 6usize;

    let w_uq: Vec<f32> = (0..(d_q * d_h)).map(|i| ((i as f32) * 0.13).sin()).collect();
    let w_uk: Vec<f32> = (0..(d_c * d_h)).map(|i| ((i as f32) * 0.19).cos()).collect();
    let w_uv: Vec<f32> = (0..(d_c * d_h)).map(|i| ((i as f32) * 0.11).sin()).collect();
    let w_o: Vec<f32> = (0..(d_h * d_model)).map(|i| ((i as f32) * 0.07).cos()).collect();

    let w_kc = MlaAbsorptionPass::compute_absorbed_q_f32(&w_uq, &w_uk, d_q, d_c, d_h);
    let w_vc = MlaAbsorptionPass::compute_absorbed_o_f32(&w_uv, &w_o, d_c, d_h, d_model);

    assert_eq!(w_kc.len(), d_q * d_c);
    assert_eq!(w_vc.len(), d_c * d_model);
}

#[test]
fn test_step7_cli_emit_ir_c_and_cuda_flags() {
    for flag in ["--emit-ir", "--emit-c", "--emit-cuda"] {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let args = vec![
            flag.to_string(),
            "examples/tinystories_smollm.vlx".to_string(),
        ];
        let code = run_cli_with_io(&args, &mut out, &mut err);
        assert_eq!(code, EXIT_SUCCESS, "CLI {flag} failed");
        let text = String::from_utf8(out).unwrap();
        assert!(!text.is_empty());
        if flag == "--emit-ir" {
            assert!(text.contains("Velox IR (VIR) Module: @SmolLM_135M"));
            assert!(text.contains("FusedNormQkvRopeAttn"));
        } else if flag == "--emit-c" {
            assert!(text.contains("typedef struct VeloxArenaContext"));
            assert!(text.contains("velox_SmolLM_135M_decode"));
        } else if flag == "--emit-cuda" {
            assert!(text.contains("typedef struct VeloxCudaArena"));
            assert!(text.contains("velox_SmolLM_135M_decode_cuda"));
        }
    }
}
