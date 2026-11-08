//! Adversarial Stress-Test Suite for Step 7: Velox IR Lowering & SOTA Optimization Passes
//!
//! Focuses on adversarial verification of:
//! 1. MLA matrix absorption numerical equivalence ($< 10^{-5}$ error across random matrices)
//! 2. Kernel fusion launch reduction (14 ops -> 2 launches across small, medium, and huge models)
//! 3. DAG monomorphization (prefill & decode disjoint non-branching streams with active_mask predication)
//! 4. 256-byte register alignment invariant (offset % 256 == 0 across all virtual registers)

use veloxc::ir::{lower_program_to_vir, VirInst, VirModule, VirReg, VirTerminator};
use veloxc::lexer::Lexer;
use veloxc::memory_planner::StaticMemoryPlanner;
use veloxc::optimizer::{optimize_vir_module, MlaAbsorptionPass};
use veloxc::parser::Parser;
use veloxc::sema::SemanticAnalyzer;
use veloxc::types::ScopeKind;

/// Deterministic pseudo-random number generator (PCG-32 / XorShift64) with zero dependencies.
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let mult = self.state.wrapping_mul(0x2545F4914F6CDD1D);
        (mult >> 32) as u32
    }

    /// Random f32 in range [min, max]
    fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        let u = self.next_u32();
        let unit = (u as f64) / (u32::MAX as f64);
        (min as f64 + unit * (max as f64 - min as f64)) as f32
    }

    fn random_vec(&mut self, len: usize, min: f32, max: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32_range(min, max)).collect()
    }
}

/// Helper to parse and lower a `.vlx` file to optimized VIR module.
fn load_and_lower_model(path: &str) -> (VirModule, VirModule) {
    let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    let tokens = Lexer::tokenize_source(&src, path).expect("Lexing failed");
    let mut parser = Parser::new(tokens, path, &src);
    let program = parser.parse_program_safe().expect("Parsing failed");
    let typed = SemanticAnalyzer::new().analyze(&program).expect("Sema failed");
    let plans = StaticMemoryPlanner::new().plan_program(&typed).expect("Memory planning failed");

    let unfused_modules = lower_program_to_vir(&typed, &plans);
    assert_eq!(unfused_modules.len(), 1);
    let unfused = unfused_modules[0].clone();

    let mut fused = unfused.clone();
    let _report = optimize_vir_module(&mut fused);

    (unfused, fused)
}

/// Extracts all virtual registers used across all instructions and terminators in a VIR module.
fn extract_all_registers(module: &VirModule) -> Vec<VirReg> {
    let mut regs = Vec::new();

    for func in &module.functions {
        for bb in &func.blocks {
            for inst in &bb.instructions {
                match inst {
                    VirInst::Embed { dst, .. } => regs.push(dst.clone()),
                    VirInst::RmsNorm { dst, input, .. } => {
                        regs.push(dst.clone());
                        regs.push(input.clone());
                    }
                    VirInst::Matmul { dst, lhs, .. } => {
                        regs.push(dst.clone());
                        regs.push(lhs.clone());
                    }
                    VirInst::RoPE { dst, input, .. } => {
                        regs.push(dst.clone());
                        regs.push(input.clone());
                    }
                    VirInst::KVWrite { k, v, .. } => {
                        regs.push(k.clone());
                        regs.push(v.clone());
                    }
                    VirInst::LatentKVAppend { c_kv, k_rope, .. } => {
                        regs.push(c_kv.clone());
                        regs.push(k_rope.clone());
                    }
                    VirInst::FlashPrefill { dst, q, k, v, .. } => {
                        regs.push(dst.clone());
                        regs.push(q.clone());
                        regs.push(k.clone());
                        regs.push(v.clone());
                    }
                    VirInst::FlashDecode { dst, q, .. } => {
                        regs.push(dst.clone());
                        regs.push(q.clone());
                    }
                    VirInst::MlaDecodeAbsorbed { dst, q_absorbed, q_rope, .. } => {
                        regs.push(dst.clone());
                        regs.push(q_absorbed.clone());
                        regs.push(q_rope.clone());
                    }
                    VirInst::FlashChunkedPiggyback {
                        dst_chunk,
                        dst_decode,
                        q_chunk,
                        k_chunk,
                        v_chunk,
                        q_decode,
                        ..
                    } => {
                        regs.push(dst_chunk.clone());
                        regs.push(dst_decode.clone());
                        regs.push(q_chunk.clone());
                        regs.push(k_chunk.clone());
                        regs.push(v_chunk.clone());
                        regs.push(q_decode.clone());
                    }
                    VirInst::FlashTreeAttention { dst, q, k, v, .. } => {
                        regs.push(dst.clone());
                        regs.push(q.clone());
                        regs.push(k.clone());
                        regs.push(v.clone());
                    }
                    VirInst::Silu { dst, input } => {
                        regs.push(dst.clone());
                        regs.push(input.clone());
                    }
                    VirInst::ElementwiseMul { dst, lhs, rhs } => {
                        regs.push(dst.clone());
                        regs.push(lhs.clone());
                        regs.push(rhs.clone());
                    }
                    VirInst::ResidualAdd { dst, lhs, rhs } => {
                        regs.push(dst.clone());
                        regs.push(lhs.clone());
                        regs.push(rhs.clone());
                    }
                    VirInst::SwiGlu { dst, input, .. } => {
                        regs.push(dst.clone());
                        regs.push(input.clone());
                    }
                    VirInst::MoeGateTopK { dst_indices, dst_weights, input, .. } => {
                        regs.push(dst_indices.clone());
                        regs.push(dst_weights.clone());
                        regs.push(input.clone());
                    }
                    VirInst::MoeDispatchTopK { dst, input, indices, weights, .. } => {
                        regs.push(dst.clone());
                        regs.push(input.clone());
                        regs.push(indices.clone());
                        regs.push(weights.clone());
                    }
                    VirInst::SliceLastToken { dst, input } => {
                        regs.push(dst.clone());
                        regs.push(input.clone());
                    }
                    VirInst::SampleArgmax { dst, logits } => {
                        regs.push(dst.clone());
                        regs.push(logits.clone());
                    }
                    VirInst::AsyncDma { src, dst, .. } => {
                        regs.push(src.clone());
                        regs.push(dst.clone());
                    }
                    VirInst::TmaLoad { src, dst, .. } => {
                        regs.push(src.clone());
                        regs.push(dst.clone());
                    }
                    VirInst::TmaStore { src, dst } => {
                        regs.push(src.clone());
                        regs.push(dst.clone());
                    }
                    VirInst::FusedMlaAbsorbedAttn { dst_hidden, input_hidden, .. } => {
                        regs.push(dst_hidden.clone());
                        regs.push(input_hidden.clone());
                    }
                    VirInst::FusedNormQkvRopeAttn { dst_attn, input_hidden, .. } => {
                        regs.push(dst_attn.clone());
                        regs.push(input_hidden.clone());
                    }
                    VirInst::FusedOutProjResSwiGluRes { dst_hidden, attn_in, residual_in, .. } => {
                        regs.push(dst_hidden.clone());
                        regs.push(attn_in.clone());
                        regs.push(residual_in.clone());
                    }
                    VirInst::FlashAttn3Fp8 { dst_attn, q, k, v, q_scale, k_scale, v_scale, .. } => {
                        regs.push(dst_attn.clone());
                        regs.push(q.clone());
                        regs.push(k.clone());
                        regs.push(v.clone());
                        regs.push(q_scale.clone());
                        regs.push(k_scale.clone());
                        regs.push(v_scale.clone());
                    }
                    VirInst::PersistentGemm { dst, a, b, .. } => {
                        regs.push(dst.clone());
                        regs.push(a.clone());
                        regs.push(b.clone());
                    }
                    VirInst::StrataFluxFetch { dst_high_precision, src_low_precision, scale, token_ids, .. } => {
                        regs.push(dst_high_precision.clone());
                        regs.push(src_low_precision.clone());
                        regs.push(scale.clone());
                        regs.push(token_ids.clone());
                    }
                    VirInst::StrataFluxRefine { dst_refined, base_low_precision, refinement } => {
                        regs.push(dst_refined.clone());
                        regs.push(base_low_precision.clone());
                        regs.push(refinement.clone());
                    }
                }
            }
            if let VirTerminator::Return(Some(reg)) = &bb.terminator {
                regs.push(reg.clone());
            }
        }
    }

    regs
}

// ============================================================================
// STRESS-TEST 1: MLA Matrix Absorption Numerical Equivalence (< 10^-5 error)
// ============================================================================

#[test]
fn test_adversarial_mla_matrix_absorption_randomized() {
    let mut rng = DeterministicRng::new(0xDEADBEEFCAFE1234);

    let test_shapes = [
        // (d_q, d_c, d_h, d_model, description)
        (1, 1, 1, 1, "minimal 1x1 scalar projection"),
        (4, 3, 2, 6, "toy rectangular matrix"),
        (17, 13, 23, 37, "prime non-power-of-2 dimensions"),
        (64, 32, 16, 128, "small power-of-2 model head"),
        (128, 64, 32, 256, "medium dimension head"),
        (512, 128, 64, 1024, "large dimension head"),
        (1536, 512, 128, 7168, "exact DeepSeek-V3 head & model dimensions"),
    ];

    for &(d_q, d_c, d_h, d_model, desc) in &test_shapes {
        // Run 2 randomized trials on huge DeepSeek-V3 shape to avoid unoptimized O(10^9) debug latency, 10 on others
        let trials = if d_q > 512 { 2 } else { 10 };
        for trial in 0..trials {
            // Keep matrix values normalized to avoid catastrophic float32 accumulation drift
            let scale = 1.0 / (d_h as f32).sqrt();
            let w_uq = rng.random_vec(d_q * d_h, -scale, scale);
            let w_uk = rng.random_vec(d_c * d_h, -scale, scale);
            let w_uv = rng.random_vec(d_c * d_h, -scale, scale);
            let w_o = rng.random_vec(d_h * d_model, -scale, scale);

            // In transformer inference, inputs into MLA are RMSNormed: RMS(c_q) ~= 1.0,
            // so individual elements are scaled by 1 / sqrt(d)
            let q_scale = 1.0 / (d_q as f32).sqrt();
            let kv_scale = 1.0 / (d_c as f32).sqrt();
            let c_q = rng.random_vec(d_q, -q_scale, q_scale);
            let c_kv = rng.random_vec(d_c, -kv_scale, kv_scale);

            // -------------------------------------------------------------
            // Check 1: Direct Matrix-Level Equivalence: W_UQ' vs (W_UQ @ W_UK^T)
            // -------------------------------------------------------------
            let w_absorbed_q = MlaAbsorptionPass::compute_absorbed_q_f32(&w_uq, &w_uk, d_q, d_c, d_h);
            assert_eq!(w_absorbed_q.len(), d_q * d_c);

            // Reference double-precision calculation of W_UQ @ W_UK^T
            for r in 0..d_q {
                for c in 0..d_c {
                    let mut acc_f64 = 0.0f64;
                    for k in 0..d_h {
                        acc_f64 += (w_uq[r * d_h + k] as f64) * (w_uk[c * d_h + k] as f64);
                    }
                    let diff_mat = ((w_absorbed_q[r * d_c + c] as f64) - acc_f64).abs();
                    assert!(
                        diff_mat < 1e-5,
                        "Matrix entry [{r}, {c}] drift in {desc}: absorbed={} vs f64_ref={acc_f64}, diff={diff_mat} >= 1e-5",
                        w_absorbed_q[r * d_c + c]
                    );
                }
            }

            // -------------------------------------------------------------
            // Check 2: Query projection absorption numerical equivalence
            // -------------------------------------------------------------
            // Unabsorbed computation:
            // q = c_q @ W_UQ  (1 x d_h)
            // k = c_kv @ W_UK (1 x d_h)
            // score = q . k
            let mut q = vec![0.0f32; d_h];
            for k_idx in 0..d_h {
                for r in 0..d_q {
                    q[k_idx] += c_q[r] * w_uq[r * d_h + k_idx];
                }
            }
            let mut k_vec = vec![0.0f32; d_h];
            for k_idx in 0..d_h {
                for r in 0..d_c {
                    k_vec[k_idx] += c_kv[r] * w_uk[r * d_h + k_idx];
                }
            }
            let score_unabsorbed: f32 = q.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();

            // Absorbed computation:
            // W_UQ' = W_UQ @ W_UK^T (d_q x d_c)
            // q_abs = c_q @ W_UQ'   (1 x d_c)
            // score = q_abs . c_kv
            let mut q_abs = vec![0.0f32; d_c];
            for c in 0..d_c {
                for r in 0..d_q {
                    q_abs[c] += c_q[r] * w_absorbed_q[r * d_c + c];
                }
            }
            let score_absorbed: f32 = q_abs.iter().zip(c_kv.iter()).map(|(a, b)| a * b).sum();

            let diff_score = (score_unabsorbed - score_absorbed).abs();
            assert!(
                diff_score < 1e-5,
                "MLA Query Absorption drift failure on {desc} (trial {trial}): unabsorbed={score_unabsorbed}, absorbed={score_absorbed}, diff={diff_score} >= 1e-5"
            );

            // -------------------------------------------------------------
            // Check 3: Output projection absorption W_O' = W_UV @ W_O
            // -------------------------------------------------------------
            // Unabsorbed computation:
            // v = c_kv @ W_UV (1 x d_h)
            // out = v @ W_O   (1 x d_model)
            let mut v = vec![0.0f32; d_h];
            for k_idx in 0..d_h {
                for r in 0..d_c {
                    v[k_idx] += c_kv[r] * w_uv[r * d_h + k_idx];
                }
            }
            let mut out_unabsorbed = vec![0.0f32; d_model];
            for m in 0..d_model {
                for k_idx in 0..d_h {
                    out_unabsorbed[m] += v[k_idx] * w_o[k_idx * d_model + m];
                }
            }

            // Absorbed computation:
            // W_O' = W_UV @ W_O (d_c x d_model)
            // out_abs = c_kv @ W_O' (1 x d_model)
            let w_absorbed_o = MlaAbsorptionPass::compute_absorbed_o_f32(&w_uv, &w_o, d_c, d_h, d_model);
            let mut out_absorbed = vec![0.0f32; d_model];
            for m in 0..d_model {
                for r in 0..d_c {
                    out_absorbed[m] += c_kv[r] * w_absorbed_o[r * d_model + m];
                }
            }

            for m in 0..d_model {
                let diff_out = (out_unabsorbed[m] - out_absorbed[m]).abs();
                assert!(
                    diff_out < 1e-5,
                    "MLA Output Absorption drift failure on {desc} at dim {m} (trial {trial}): unabsorbed={}, absorbed={}, diff={diff_out} >= 1e-5",
                    out_unabsorbed[m],
                    out_absorbed[m]
                );
            }
        }
    }
}

// ============================================================================
// STRESS-TEST 2: Kernel Fusion Launch Reduction (14 ops -> 2 launches / layer)
// ============================================================================

#[test]
fn test_adversarial_kernel_fusion_launch_reduction_smollm_30_layers() {
    let (unfused, fused) = load_and_lower_model("examples/tinystories_smollm.vlx");

    // SmolLM-135M has 30 layers
    let num_layers = 30;

    for (u_func, f_func) in unfused.functions.iter().zip(fused.functions.iter()) {
        assert_eq!(u_func.name, f_func.name);
        assert_eq!(u_func.scope_kind, f_func.scope_kind);

        // In decode scope: 1 Embed + 30 * 17 ops + 3 final ops = 514 ops
        // In prefill scope: 1 Embed + 30 * 16 ops + 3 final ops = 484 ops
        if u_func.scope_kind.is_decode_phase() {
            assert_eq!(u_func.unoptimized_op_count, 1 + num_layers * 17 + 3);
        } else {
            assert_eq!(u_func.unoptimized_op_count, 1 + num_layers * 16 + 3);
        }

        // Fused function: exactly 1 Embed + 30 * 2 layer kernels + 3 final ops = 64 launches!
        let expected_fused_launches = 1 + (num_layers * 2) + 3;
        assert_eq!(
            f_func.optimized_launch_count, expected_fused_launches,
            "Expected exactly {expected_fused_launches} launches (60 layer dispatches + 4 setup/final dispatches) in {}",
            f_func.name
        );

        // Verify layer-by-layer instructions: exactly 60 fused layer dispatches alternating Attn and MLP
        let insts = &f_func.blocks[0].instructions;
        assert_eq!(insts.len(), expected_fused_launches);

        // Op 0: Embed
        assert!(matches!(insts[0], VirInst::Embed { .. }));

        // Ops 1..61: exactly 30 pairs of (FusedNormQkvRopeAttn, FusedOutProjResSwiGluRes)
        for layer in 0..num_layers {
            let attn_idx = 1 + layer * 2;
            let mlp_idx = 1 + layer * 2 + 1;

            match &insts[attn_idx] {
                VirInst::FusedNormQkvRopeAttn {
                    layer: l,
                    is_decode,
                    ..
                } => {
                    assert_eq!(*l, layer, "Layer mismatch at attn idx {attn_idx}");
                    assert_eq!(*is_decode, u_func.scope_kind.is_decode_phase());
                }
                other => panic!("Expected FusedNormQkvRopeAttn at {attn_idx}, got {other:?}"),
            }

            match &insts[mlp_idx] {
                VirInst::FusedOutProjResSwiGluRes { layer: l, .. } => {
                    assert_eq!(*l, layer, "Layer mismatch at mlp idx {mlp_idx}");
                }
                other => panic!("Expected FusedOutProjResSwiGluRes at {mlp_idx}, got {other:?}"),
            }
        }

        // Final 3 ops
        assert!(matches!(insts[61], VirInst::RmsNorm { layer: None, .. }));
        assert!(matches!(insts[62], VirInst::Matmul { layer: None, .. }));
        assert!(matches!(insts[63], VirInst::SampleArgmax { .. }));

        // Reduction ratio must be > 80% (from ~514 down to 64 is 87.5% reduction!)
        let ratio = 1.0 - (f_func.optimized_launch_count as f64 / u_func.unoptimized_op_count as f64);
        assert!(
            ratio > 0.80,
            "Launch reduction ratio {ratio} was not > 80% in {}",
            f_func.name
        );
    }
}

#[test]
fn test_adversarial_kernel_fusion_across_depths_and_models() {
    // Test all 4 standard dense models: SmolLM (30L), Llama3-8B (32L), Qwen2.5-7B (28L), Nanbeige-33B (60L)
    let dense_models = [
        ("examples/tinystories_smollm.vlx", 30),
        ("examples/llama3_8b.vlx", 32),
        ("examples/qwen2_5_7b.vlx", 28),
        ("examples/nanbeige_33b.vlx", 60),
    ];

    for (path, expected_layers) in dense_models {
        let (unfused, fused) = load_and_lower_model(path);

        for (u_func, f_func) in unfused.functions.iter().zip(fused.functions.iter()) {
            let expected_fused_launches = 1 + (expected_layers * 2) + 3;
            assert_eq!(
                f_func.optimized_launch_count, expected_fused_launches,
                "Model {path} function {} failed launch count: got {}, expected {}",
                f_func.name, f_func.optimized_launch_count, expected_fused_launches
            );

            // Verify fused layer kernels count is exactly 2 * expected_layers
            let fused_layer_ops = f_func.blocks[0]
                .instructions
                .iter()
                .filter(|i| {
                    matches!(
                        i,
                        VirInst::FusedNormQkvRopeAttn { .. } | VirInst::FusedOutProjResSwiGluRes { .. }
                    )
                })
                .count();

            assert_eq!(
                fused_layer_ops,
                expected_layers * 2,
                "Model {path} layer ops count mismatch: got {fused_layer_ops}, expected {}",
                expected_layers * 2
            );

            let ratio = 1.0 - (f_func.optimized_launch_count as f64 / u_func.unoptimized_op_count as f64);
            assert!(
                ratio > 0.80,
                "Model {path} function {} launch reduction ratio {:.2}% not > 80%",
                f_func.name,
                ratio * 100.0
            );
        }
    }
}

// ============================================================================
// STRESS-TEST 3: DAG Monomorphization Disjointness & Predication
// ============================================================================

#[test]
fn test_adversarial_dag_monomorphization_disjointness_and_predication() {
    let all_models = [
        "examples/tinystories_smollm.vlx",
        "examples/llama3_8b.vlx",
        "examples/qwen2_5_7b.vlx",
        "examples/deepseek_v3_mla.vlx",
        "examples/nanbeige_33b.vlx",
    ];

    for path in all_models {
        let (_unfused, fused) = load_and_lower_model(path);
        assert!(fused.dag_monomorphized, "Module {path} must have dag_monomorphized = true");

        let prefill_func = fused
            .functions
            .iter()
            .find(|f| f.scope_kind == ScopeKind::Prefill)
            .expect("Must have prefill function");
        let decode_func = fused
            .functions
            .iter()
            .find(|f| f.scope_kind == ScopeKind::Decode)
            .expect("Must have decode function");

        // 1. Both functions must be monomorphized
        assert!(prefill_func.is_monomorphized);
        assert!(decode_func.is_monomorphized);

        // 2. Monomorphization predication contract:
        // Prefill operates over the entire batch/sequence statically -> active_mask_predicated == false
        assert!(
            !prefill_func.active_mask_predicated,
            "Prefill function {} must NOT be active_mask predicated",
            prefill_func.name
        );
        // Decode operates over dynamic active requests with masked execution -> active_mask_predicated == true
        assert!(
            decode_func.active_mask_predicated,
            "Decode function {} MUST be active_mask predicated",
            decode_func.name
        );

        // 3. Parameters must be phase-specialized and disjoint
        let prefill_param_names: Vec<String> = prefill_func.params.iter().map(|p| p.name.clone()).collect();
        let decode_param_names: Vec<String> = decode_func.params.iter().map(|p| p.name.clone()).collect();

        assert!(
            prefill_param_names.contains(&"tokens".to_string()),
            "Prefill must have 'tokens' param"
        );
        assert!(
            prefill_param_names.contains(&"seq_len".to_string()),
            "Prefill must have 'seq_len' param"
        );
        assert!(
            !prefill_param_names.contains(&"active_mask".to_string()),
            "Prefill must not have 'active_mask' param"
        );

        assert!(
            decode_param_names.contains(&"token".to_string()),
            "Decode must have 'token' param"
        );
        assert!(
            decode_param_names.contains(&"step".to_string()),
            "Decode must have 'step' param"
        );
        assert!(
            decode_param_names.contains(&"active_mask".to_string()),
            "Decode must have 'active_mask' param"
        );

        // 4. Non-branching verification:
        // Single basic block with unconditional Return terminator, ZERO branches
        for func in [&prefill_func, &decode_func] {
            assert_eq!(
                func.blocks.len(),
                1,
                "Function {} must be a single basic block (non-branching DAG)",
                func.name
            );
            let bb = &func.blocks[0];
            assert_eq!(bb.label, "entry");
            assert!(
                matches!(bb.terminator, VirTerminator::Return(_)),
                "Function {} terminator must be Return, found {:?}",
                func.name,
                bb.terminator
            );
        }

        // 5. Instruction stream disjointness:
        // Prefill instruction stream must NOT contain decode instructions (KVWrite, FlashDecode, is_decode: true)
        for inst in &prefill_func.blocks[0].instructions {
            match inst {
                VirInst::KVWrite { .. } | VirInst::FlashDecode { .. } | VirInst::MlaDecodeAbsorbed { .. } => {
                    panic!("Prefill stream in {path} illegally contains decode instruction: {inst:?}");
                }
                VirInst::FusedNormQkvRopeAttn { is_decode, .. } => {
                    assert!(!is_decode, "Prefill stream in {path} contains fused decode kernel");
                }
                _ => {}
            }
        }

        // Decode instruction stream must NOT contain prefill instructions (FlashPrefill, is_decode: false)
        for inst in &decode_func.blocks[0].instructions {
            match inst {
                VirInst::FlashPrefill { .. } => {
                    panic!("Decode stream in {path} illegally contains prefill instruction: {inst:?}");
                }
                VirInst::FusedNormQkvRopeAttn { is_decode, .. } => {
                    assert!(is_decode, "Decode stream in {path} contains fused prefill kernel");
                }
                _ => {}
            }
        }
    }
}

// ============================================================================
// STRESS-TEST 4: Virtual Register 256-Byte Alignment Invariant
// ============================================================================

#[test]
fn test_adversarial_virtual_register_256_byte_alignment_all_models() {
    let all_models = [
        "examples/tinystories_smollm.vlx",
        "examples/llama3_8b.vlx",
        "examples/qwen2_5_7b.vlx",
        "examples/deepseek_v3_mla.vlx",
        "examples/nanbeige_33b.vlx",
    ];

    for path in all_models {
        let (unfused, fused) = load_and_lower_model(path);

        for (stage, module) in [("unfused", &unfused), ("fused", &fused)] {
            let regs = extract_all_registers(module);
            assert!(
                !regs.is_empty(),
                "Model {path} in {stage} stage produced 0 virtual registers"
            );

            for reg in &regs {
                assert_eq!(
                    reg.arena_offset % 256,
                    0,
                    "Register {} ({}) in model {path} stage {stage} violates 256-byte alignment: offset = 0x{:X} ({} mod 256 != 0)",
                    reg.id,
                    reg.name,
                    reg.arena_offset,
                    reg.arena_offset % 256
                );

                assert_eq!(
                    reg.aligned_bytes % 256,
                    0,
                    "Register {} ({}) in model {path} stage {stage} aligned_bytes {} mod 256 != 0",
                    reg.id,
                    reg.name,
                    reg.aligned_bytes
                );

                assert!(
                    reg.aligned_bytes >= 256,
                    "Register {} ({}) in model {path} stage {stage} aligned_bytes {} < 256",
                    reg.id,
                    reg.name,
                    reg.aligned_bytes
                );
            }
        }
    }
}
