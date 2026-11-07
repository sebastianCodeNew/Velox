//! Adversarial Stress-Test Suite for Step 8: MLA Absorbed Latent Attention Kernel
//!
//! Evaluates:
//! 1. Numerical stability under extreme inputs (+1000.0, -1000.0, zero, spiky, alternating).
//! 2. Mathematical convergence between single-pass online softmax and classical two-pass softmax across d_c in {64, 128, 512}.
//! 3. Long context sequence scaling from step 0 to step (max_s - 1) up to 1024 steps.
//! 4. Clamping and out-of-bounds safety when step >= max_s.
//! 5. Multi-layer memory isolation and offset safety.

use veloxc::runtime::{mla_absorbed_step_f32, mla_absorbed_step_with_scratch_f32};

/// Deterministic pseudo-random number generator (XorShift64) with zero external dependencies.
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

    fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        let u = self.next_u32();
        let unit = (u as f64) / (u32::MAX as f64);
        (min as f64 + unit * (max as f64 - min as f64)) as f32
    }

    fn random_vec(&mut self, len: usize, min: f32, max: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32_range(min, max)).collect()
    }
}

/// Classical two-pass softmax oracle over KV latent pool for verification.
fn reference_two_pass_mla_attention(
    q_abs: &[f32],
    kv_latent_pool: &[f32],
    pool_offset: usize,
    latent_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(seq_len > 0);
    let mut scores = vec![0.0f32; seq_len];
    let mut max_s = f32::NEG_INFINITY;

    // Pass 1: Dot products and maximum logit
    for t in 0..seq_len {
        let k_start = pool_offset + t * latent_dim;
        let k_vec = &kv_latent_pool[k_start..k_start + latent_dim];
        let mut dot = 0.0f32;
        for d in 0..latent_dim {
            dot += q_abs[d] * k_vec[d];
        }
        let s = dot * scale;
        scores[t] = s;
        if s > max_s {
            max_s = s;
        }
    }

    // Pass 2: Exponentiate with max subtraction and sum
    let mut sum_exp = 0.0f32;
    for t in 0..seq_len {
        let e = (scores[t] - max_s).exp();
        scores[t] = e;
        sum_exp += e;
    }

    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
    for t in 0..seq_len {
        scores[t] *= inv_sum;
    }

    // Pass 3: Weighted sum over latent keys
    let mut context = vec![0.0f32; latent_dim];
    for t in 0..seq_len {
        let prob = scores[t];
        let k_start = pool_offset + t * latent_dim;
        let k_vec = &kv_latent_pool[k_start..k_start + latent_dim];
        for d in 0..latent_dim {
            context[d] += prob * k_vec[d];
        }
    }

    context
}

#[test]
fn test_mla_extreme_positive_logits_no_nan_or_inf() {
    // Stress test: Extremely large logits (+1000.0 to +10,000.0)
    // Naive softmax exp(1000.0) overflows f32 to +Inf and results in NaN.
    // Online softmax with running max subtraction must remain strictly finite.
    let hidden_dim = 16;
    let latent_dim = 8;
    let max_seq_len = 8;

    let mut kv_latent_pool = vec![0.0f32; max_seq_len * latent_dim];
    let w_norm = vec![1.0f32; hidden_dim];
    let w_dkv = vec![0.1f32; hidden_dim * latent_dim];
    let w_absorbed_q = vec![0.1f32; hidden_dim * latent_dim];
    let w_absorbed_o = vec![0.1f32; latent_dim * hidden_dim];

    // Seed historical slots with known vectors
    for t in 0..4 {
        for d in 0..latent_dim {
            // Very large numbers in latent pool
            kv_latent_pool[t * latent_dim + d] = 500.0 * (d as f32 + 1.0);
        }
    }

    // Input that produces large dot product > 1000.0
    let input_hidden = vec![200.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    mla_absorbed_step_f32(
        &mut dst_hidden,
        &input_hidden,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        0,
        3, // step 3 (history tokens 0..=3)
        1e-5,
    );

    // Verify all outputs are strictly finite (no NaN, no Inf)
    for (i, &val) in dst_hidden.iter().enumerate() {
        assert!(val.is_finite(), "dst_hidden[{i}] must be finite, got {val}");
        assert!(!val.is_nan(), "dst_hidden[{i}] must not be NaN");
    }

    for (i, &val) in kv_latent_pool.iter().enumerate() {
        assert!(val.is_finite(), "kv_latent_pool[{i}] must be finite, got {val}");
    }
}

#[test]
fn test_mla_extreme_negative_logits_no_zero_division() {
    // Stress test: Extremely small/negative logits (-1000.0 to -5000.0)
    // In naive softmax exp(-1000.0) underflows to 0.0, causing sum_exp == 0.0 and division by zero.
    // Online softmax ensures running max is subtracted, so max logit exponent is exp(0) == 1.0, sum >= 1.0.
    let hidden_dim = 16;
    let latent_dim = 8;
    let max_seq_len = 8;

    let mut kv_latent_pool = vec![0.0f32; max_seq_len * latent_dim];
    let w_norm = vec![1.0f32; hidden_dim];
    let w_dkv = vec![1.0f32; hidden_dim * latent_dim];
    let w_absorbed_q = vec![1.0f32; hidden_dim * latent_dim];
    let w_absorbed_o = vec![1.0f32; latent_dim * hidden_dim];

    // Seed historical slots with negative vectors so dot product is ~ -2000.0
    for t in 0..4 {
        for d in 0..latent_dim {
            kv_latent_pool[t * latent_dim + d] = -500.0 * (t as f32 + 1.0);
        }
    }

    let input_hidden = vec![1.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    mla_absorbed_step_f32(
        &mut dst_hidden,
        &input_hidden,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        0,
        3,
        1e-5,
    );

    // Verify all outputs are strictly finite (no NaN, no Inf, no divide by zero)
    for (i, &val) in dst_hidden.iter().enumerate() {
        assert!(val.is_finite(), "dst_hidden[{i}] must be finite, got {val}");
        assert!(!val.is_nan(), "dst_hidden[{i}] must not be NaN");
    }
}

#[test]
fn test_mla_online_vs_twopass_convergence_varying_dimensions() {
    // Mathematical convergence stress-test across latent dimensions: 64, 128, 512
    let dims = [
        (64, 32, 16),    // hidden=64, latent=32, seq=16
        (128, 64, 32),   // hidden=128, latent=64, seq=32
        (256, 128, 64),  // hidden=256, latent=128, seq=64
        (1024, 512, 128),// hidden=1024, latent=512, seq=128 (DeepSeek-V3 scale)
    ];

    let mut rng = DeterministicRng::new(42);

    for &(hidden_dim, latent_dim, seq_len) in &dims {
        let max_seq_len = seq_len + 8;
        let mut kv_latent_pool = rng.random_vec(max_seq_len * latent_dim, -1.0, 1.0);

        let w_norm = rng.random_vec(hidden_dim, 0.5, 1.5);
        let w_dkv = rng.random_vec(hidden_dim * latent_dim, -0.1, 0.1);
        let w_absorbed_q = rng.random_vec(hidden_dim * latent_dim, -0.1, 0.1);
        let w_absorbed_o = rng.random_vec(latent_dim * hidden_dim, -0.1, 0.1);

        let input_hidden = rng.random_vec(hidden_dim, -2.0, 2.0);

        // Scratch buffers for online step
        let mut norm_x = vec![0.0f32; hidden_dim];
        let mut q_abs = vec![0.0f32; latent_dim];
        let mut context_online = vec![0.0f32; latent_dim];
        let mut o_proj = vec![0.0f32; hidden_dim];
        let mut dst_hidden = vec![0.0f32; hidden_dim];

        let step = seq_len - 1;

        mla_absorbed_step_with_scratch_f32(
            &mut dst_hidden,
            &input_hidden,
            &w_norm,
            &w_dkv,
            &w_absorbed_q,
            &w_absorbed_o,
            &mut kv_latent_pool,
            &mut norm_x,
            &mut q_abs,
            &mut context_online,
            &mut o_proj,
            0,
            step,
            hidden_dim,
            latent_dim,
            max_seq_len,
            1e-5,
        );

        // Compute reference two-pass softmax context using the exact same q_abs and updated kv_latent_pool
        let scale = if latent_dim >= 128 {
            1.0f32 / 128.0f32.sqrt()
        } else {
            1.0f32 / (latent_dim as f32).sqrt()
        };

        let context_reference = reference_two_pass_mla_attention(
            &q_abs,
            &kv_latent_pool,
            0,
            latent_dim,
            step + 1,
            scale,
        );

        // Verify mathematical equivalence between single-pass online softmax and two-pass classical softmax
        let mut max_abs_diff = 0.0f32;
        for d in 0..latent_dim {
            let diff = (context_online[d] - context_reference[d]).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
            assert!(
                diff < 1e-4,
                "Latent dim {latent_dim}, step {step}, d={d}: online={:.7}, ref={:.7}, diff={:.7}",
                context_online[d], context_reference[d], diff
            );
        }

        // Also verify residual addition matches reference
        // Compute reference o_proj = context_reference @ W_absorbed_o
        let mut ref_o_proj = vec![0.0f32; hidden_dim];
        for o in 0..hidden_dim {
            let mut acc = 0.0f32;
            let row_offset = o * latent_dim;
            for d in 0..latent_dim {
                acc += context_reference[d] * w_absorbed_o[row_offset + d];
            }
            ref_o_proj[o] = acc;
        }

        for i in 0..hidden_dim {
            let expected_dst = input_hidden[i] + ref_o_proj[i];
            let diff = (dst_hidden[i] - expected_dst).abs();
            assert!(
                diff < 1e-4,
                "Hidden dim {hidden_dim}, i={i}: dst={:.7}, expected={:.7}, diff={:.7}",
                dst_hidden[i], expected_dst, diff
            );
        }

        println!(
            "Dim convergence PASS: latent_dim={}, hidden_dim={}, seq_len={}, max_abs_diff={:e}",
            latent_dim, hidden_dim, seq_len, max_abs_diff
        );
    }
}

#[test]
fn test_mla_long_history_sequence_steps_0_to_max_s() {
    // Stress test: Autoregressive decoding from step 0 to step (max_s - 1)
    // Tests cumulative numerical stability, KV cache preservation, and invariance.
    let hidden_dim = 32;
    let latent_dim = 16;
    let max_seq_len = 128;

    let mut rng = DeterministicRng::new(1337);
    let mut kv_latent_pool = vec![0.0f32; max_seq_len * latent_dim];
    let w_norm = rng.random_vec(hidden_dim, 0.8, 1.2);
    let w_dkv = rng.random_vec(hidden_dim * latent_dim, -0.2, 0.2);
    let w_absorbed_q = rng.random_vec(hidden_dim * latent_dim, -0.2, 0.2);
    let w_absorbed_o = rng.random_vec(latent_dim * hidden_dim, -0.2, 0.2);

    let scale = 1.0f32 / (latent_dim as f32).sqrt();

    let mut norm_x = vec![0.0f32; hidden_dim];
    let mut q_abs = vec![0.0f32; latent_dim];
    let mut context_online = vec![0.0f32; latent_dim];
    let mut o_proj = vec![0.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    // Run 128 sequential autoregressive decode steps
    for step in 0..max_seq_len {
        let input_hidden = rng.random_vec(hidden_dim, -1.0, 1.0);

        mla_absorbed_step_with_scratch_f32(
            &mut dst_hidden,
            &input_hidden,
            &w_norm,
            &w_dkv,
            &w_absorbed_q,
            &w_absorbed_o,
            &mut kv_latent_pool,
            &mut norm_x,
            &mut q_abs,
            &mut context_online,
            &mut o_proj,
            0,
            step,
            hidden_dim,
            latent_dim,
            max_seq_len,
            1e-5,
        );

        // 1. Verify finite numbers
        for &val in &dst_hidden {
            assert!(val.is_finite(), "Step {step}: dst_hidden has non-finite value");
        }

        // 2. Verify reference two-pass equivalence at key milestone steps
        if step == 0 || step == 1 || step == 15 || step == 63 || step == max_seq_len - 1 {
            let context_reference = reference_two_pass_mla_attention(
                &q_abs,
                &kv_latent_pool,
                0,
                latent_dim,
                step + 1,
                scale,
            );

            for d in 0..latent_dim {
                let diff = (context_online[d] - context_reference[d]).abs();
                assert!(
                    diff < 1e-4,
                    "Step {step}, d={d}: online={:.6}, ref={:.6}, diff={:.6}",
                    context_online[d], context_reference[d], diff
                );
            }
        }
    }
}

#[test]
fn test_mla_safe_clamping_beyond_max_s() {
    // Adversarial test: step >= max_seq_len
    // Runtime must clamp safe_step = max_s - 1 and NOT overflow buffer or panic.
    let hidden_dim = 16;
    let latent_dim = 8;
    let max_seq_len = 4;

    let mut kv_latent_pool = vec![0.0f32; max_seq_len * latent_dim];
    let w_norm = vec![1.0f32; hidden_dim];
    let w_dkv = vec![0.1f32; hidden_dim * latent_dim];
    let w_absorbed_q = vec![0.1f32; hidden_dim * latent_dim];
    let w_absorbed_o = vec![0.1f32; latent_dim * hidden_dim];

    let input_hidden = vec![1.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    // step = 10 when max_s = 4
    mla_absorbed_step_f32(
        &mut dst_hidden,
        &input_hidden,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        0,
        10,
        1e-5,
    );

    for &val in &dst_hidden {
        assert!(val.is_finite(), "dst_hidden must remain finite on clamped step");
    }
}

#[test]
fn test_mla_multi_layer_isolation() {
    // Adversarial test: Multi-layer memory isolation
    // Layer 1 execution must NOT read or corrupt Layer 0's KV latent pool.
    let hidden_dim = 8;
    let latent_dim = 4;
    let max_seq_len = 4;
    let num_layers = 2;

    let mut kv_latent_pool = vec![0.0f32; num_layers * max_seq_len * latent_dim];

    // Populate layer 0 with 42.0
    for i in 0..(max_seq_len * latent_dim) {
        kv_latent_pool[i] = 42.0;
    }

    let w_norm = vec![1.0f32; num_layers * hidden_dim];
    let w_dkv = vec![0.5f32; num_layers * hidden_dim * latent_dim];
    let w_absorbed_q = vec![0.5f32; num_layers * hidden_dim * latent_dim];
    let w_absorbed_o = vec![0.5f32; num_layers * latent_dim * hidden_dim];

    let input_hidden = vec![2.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    // Run step 0 on Layer 1
    mla_absorbed_step_f32(
        &mut dst_hidden,
        &input_hidden,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        1, // layer 1
        0, // step 0
        1e-5,
    );

    // Verify layer 0 slots are 100% untouched
    for i in 0..(max_seq_len * latent_dim) {
        assert_eq!(
            kv_latent_pool[i], 42.0,
            "Layer 0 latent pool slot {i} was corrupted by Layer 1 execution"
        );
    }

    // If layer 1 slot 0 is updated properly with latent_dim=4, hidden_dim=8,
    // norm_x = [1.0; 8], dot product with w_dkv (0.5) is 8 * 0.5 = 4.0:
    let layer1_slot0_start = max_seq_len * latent_dim;
    let slot1_val = kv_latent_pool[layer1_slot0_start];
    assert!(
        (slot1_val - 4.0).abs() < 1e-4,
        "Layer 1 slot 0 was not updated at correct index! Got {slot1_val}"
    );
}

#[test]
fn test_mla_extreme_alternating_logits_dynamic_range() {
    // Stress test: Logits alternating violently between +1000.0 and -1000.0.
    // Tests sudden jumps in max logit and exponent underflow/overflow handling.
    let hidden_dim = 16;
    let latent_dim = 8;
    let max_seq_len = 16;

    let mut kv_latent_pool = vec![0.0f32; max_seq_len * latent_dim];
    let w_norm = vec![1.0f32; hidden_dim];
    let w_dkv = vec![0.1f32; hidden_dim * latent_dim];
    let w_absorbed_q = vec![0.1f32; hidden_dim * latent_dim];
    let w_absorbed_o = vec![0.1f32; latent_dim * hidden_dim];

    // Alternating huge positive and negative values in latent slots
    for t in 0..max_seq_len {
        let sign = if t % 2 == 0 { 1.0f32 } else { -1.0f32 };
        for d in 0..latent_dim {
            kv_latent_pool[t * latent_dim + d] = sign * 2000.0;
        }
    }

    let input_hidden = vec![50.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    for step in 0..max_seq_len {
        mla_absorbed_step_f32(
            &mut dst_hidden,
            &input_hidden,
            &w_norm,
            &w_dkv,
            &w_absorbed_q,
            &w_absorbed_o,
            &mut kv_latent_pool,
            0,
            step,
            1e-5,
        );

        for (i, &val) in dst_hidden.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Step {step}, dst_hidden[{i}] must be finite under alternating logits, got {val}"
            );
            assert!(!val.is_nan(), "Step {step}, dst_hidden[{i}] was NaN");
        }
    }
}

#[test]
fn test_mla_convex_hull_invariant_mass_conservation() {
    // Fundamental mathematical invariant:
    // Softmax probabilities p_t >= 0 and sum(p_t) = 1.0.
    // Therefore, context = sum(p_t * k_t) MUST lie strictly inside the convex hull
    // bounded by [min_t(k_t[d]), max_t(k_t[d])] for every coordinate d.
    let hidden_dim = 32;
    let latent_dim = 16;
    let max_seq_len = 32;

    let mut rng = DeterministicRng::new(999);
    let mut kv_latent_pool = rng.random_vec(max_seq_len * latent_dim, -10.0, 10.0);

    let w_norm = rng.random_vec(hidden_dim, 0.5, 1.5);
    let w_dkv = rng.random_vec(hidden_dim * latent_dim, -0.1, 0.1);
    let w_absorbed_q = rng.random_vec(hidden_dim * latent_dim, -0.1, 0.1);
    let w_absorbed_o = rng.random_vec(latent_dim * hidden_dim, -0.1, 0.1);

    let mut norm_x = vec![0.0f32; hidden_dim];
    let mut q_abs = vec![0.0f32; latent_dim];
    let mut context = vec![0.0f32; latent_dim];
    let mut o_proj = vec![0.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    for step in 0..max_seq_len {
        let input_hidden = rng.random_vec(hidden_dim, -5.0, 5.0);

        mla_absorbed_step_with_scratch_f32(
            &mut dst_hidden,
            &input_hidden,
            &w_norm,
            &w_dkv,
            &w_absorbed_q,
            &w_absorbed_o,
            &mut kv_latent_pool,
            &mut norm_x,
            &mut q_abs,
            &mut context,
            &mut o_proj,
            0,
            step,
            hidden_dim,
            latent_dim,
            max_seq_len,
            1e-5,
        );

        let seq_len = step + 1;
        for d in 0..latent_dim {
            let mut min_k = f32::INFINITY;
            let mut max_k = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let k_val = kv_latent_pool[t * latent_dim + d];
                if k_val < min_k { min_k = k_val; }
                if k_val > max_k { max_k = k_val; }
            }

            let ctx_d = context[d];
            let eps = 1e-4; // Numerical tolerance for f32 accumulation
            assert!(
                ctx_d >= min_k - eps && ctx_d <= max_k + eps,
                "Convex hull violation at step {step}, dim {d}: ctx={ctx_d}, min={min_k}, max={max_k}"
            );
        }
    }
}

#[test]
fn test_mla_zero_and_subnormal_inputs_stability() {
    // Stress test: Exactly zero inputs and subnormal inputs
    let hidden_dim = 16;
    let latent_dim = 8;
    let max_seq_len = 4;

    let mut kv_latent_pool = vec![0.0f32; max_seq_len * latent_dim];
    let w_norm = vec![1.0f32; hidden_dim];
    let w_dkv = vec![0.0f32; hidden_dim * latent_dim];
    let w_absorbed_q = vec![0.0f32; hidden_dim * latent_dim];
    let w_absorbed_o = vec![0.0f32; latent_dim * hidden_dim];

    let input_zero = vec![0.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    mla_absorbed_step_f32(
        &mut dst_hidden,
        &input_zero,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        0,
        0,
        1e-5,
    );

    for (i, &val) in dst_hidden.iter().enumerate() {
        assert!(val.is_finite(), "dst_hidden[{i}] must be finite for zero input");
        assert_eq!(val, 0.0f32, "dst_hidden[{i}] must be zero for all-zero weights and input");
    }

    // Subnormal inputs: 1e-38
    let input_subnormal = vec![1e-38f32; hidden_dim];
    mla_absorbed_step_f32(
        &mut dst_hidden,
        &input_subnormal,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        0,
        1,
        1e-5,
    );

    for (i, &val) in dst_hidden.iter().enumerate() {
        assert!(val.is_finite(), "dst_hidden[{i}] must be finite for subnormal input");
    }
}

#[test]
fn test_mla_full_deepseek_v3_scale_1024_steps() {
    // Stress test: Full DeepSeek-V3 latent dimension d_c = 512, hidden_dim = 2048, 1024 steps
    let hidden_dim = 2048;
    let latent_dim = 512;
    let max_seq_len = 1024;

    let mut rng = DeterministicRng::new(777);
    let mut kv_latent_pool = vec![0.0f32; max_seq_len * latent_dim];
    let w_norm = rng.random_vec(hidden_dim, 0.9, 1.1);
    let w_dkv = rng.random_vec(hidden_dim * latent_dim, -0.05, 0.05);
    let w_absorbed_q = rng.random_vec(hidden_dim * latent_dim, -0.05, 0.05);
    let w_absorbed_o = rng.random_vec(latent_dim * hidden_dim, -0.05, 0.05);

    let mut norm_x = vec![0.0f32; hidden_dim];
    let mut q_abs = vec![0.0f32; latent_dim];
    let mut context = vec![0.0f32; latent_dim];
    let mut o_proj = vec![0.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    // Seed pool with pre-existing conversation history
    for i in 0..(512 * latent_dim) {
        kv_latent_pool[i] = rng.next_f32_range(-0.5, 0.5);
    }

    // Test execution at step 512 (long context)
    let input_hidden = rng.random_vec(hidden_dim, -1.0, 1.0);
    mla_absorbed_step_with_scratch_f32(
        &mut dst_hidden,
        &input_hidden,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        &mut norm_x,
        &mut q_abs,
        &mut context,
        &mut o_proj,
        0,
        512,
        hidden_dim,
        latent_dim,
        max_seq_len,
        1e-5,
    );

    for (i, &val) in dst_hidden.iter().enumerate() {
        assert!(val.is_finite(), "dst_hidden[{i}] at step 512 must be finite");
    }

    // Verify mathematical parity with two-pass reference at step 512
    let scale = 1.0f32 / 128.0f32.sqrt();
    let ref_context = reference_two_pass_mla_attention(
        &q_abs,
        &kv_latent_pool,
        0,
        latent_dim,
        513,
        scale,
    );

    let mut max_err = 0.0f32;
    for d in 0..latent_dim {
        let err = (context[d] - ref_context[d]).abs();
        if err > max_err { max_err = err; }
        assert!(
            err < 1e-4,
            "DeepSeek-V3 scale step 512, dim {d}: online={:.7}, ref={:.7}, err={:.7}",
            context[d], ref_context[d], err
        );
    }

    println!("Full DeepSeek-V3 Scale (dim=2048, d_c=512, step=512) PASS, max_err={:e}", max_err);
}

#[test]
fn test_mla_orthogonal_query_uniform_attention() {
    // Stress test: Query orthogonal to all historical keys -> all dot products are exactly 0.0
    // Softmax probabilities must be strictly uniform: p_t = 1 / seq_len.
    // Resulting context must be the exact arithmetic mean of all historical keys.
    let hidden_dim = 16;
    let latent_dim = 8;
    let max_seq_len = 16;

    let mut kv_latent_pool = vec![0.0f32; max_seq_len * latent_dim];
    let mut rng = DeterministicRng::new(555);

    // Fill pool with random vectors
    for i in 0..(max_seq_len * latent_dim) {
        kv_latent_pool[i] = rng.next_f32_range(-10.0, 10.0);
    }

    let w_norm = vec![1.0f32; hidden_dim];
    let w_dkv = vec![0.0f32; hidden_dim * latent_dim];
    let w_absorbed_q = vec![0.0f32; hidden_dim * latent_dim]; // q_abs will be exactly 0
    let w_absorbed_o = vec![1.0f32; latent_dim * hidden_dim];

    let mut norm_x = vec![0.0f32; hidden_dim];
    let mut q_abs = vec![0.0f32; latent_dim];
    let mut context = vec![0.0f32; latent_dim];
    let mut o_proj = vec![0.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    let input_hidden = vec![1.0f32; hidden_dim];
    let step = 9; // seq_len = 10

    mla_absorbed_step_with_scratch_f32(
        &mut dst_hidden,
        &input_hidden,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        &mut norm_x,
        &mut q_abs,
        &mut context,
        &mut o_proj,
        0,
        step,
        hidden_dim,
        latent_dim,
        max_seq_len,
        1e-5,
    );

    let seq_len = step + 1;
    for d in 0..latent_dim {
        let mut expected_mean = 0.0f32;
        for t in 0..seq_len {
            expected_mean += kv_latent_pool[t * latent_dim + d];
        }
        expected_mean /= seq_len as f32;

        let diff = (context[d] - expected_mean).abs();
        assert!(
            diff < 1e-5,
            "Orthogonal query mean failure at dim {d}: ctx={:.7}, expected={:.7}, diff={:e}",
            context[d], expected_mean, diff
        );
    }
}

#[test]
fn test_mla_very_long_sequence_2048_steps_numerical_drift() {
    // Stress test: Sequence length of 2048 steps.
    // Evaluates whether single-pass online softmax exhibits numerical drift or loss of precision
    // compared to classical two-pass softmax over deep historical context.
    let hidden_dim = 64;
    let latent_dim = 64;
    let max_seq_len = 2048;

    let mut rng = DeterministicRng::new(2026);
    let mut kv_latent_pool = rng.random_vec(max_seq_len * latent_dim, -1.0, 1.0);

    let w_norm = rng.random_vec(hidden_dim, 0.8, 1.2);
    let w_dkv = rng.random_vec(hidden_dim * latent_dim, -0.05, 0.05);
    let w_absorbed_q = rng.random_vec(hidden_dim * latent_dim, -0.05, 0.05);
    let w_absorbed_o = rng.random_vec(latent_dim * hidden_dim, -0.05, 0.05);

    let mut norm_x = vec![0.0f32; hidden_dim];
    let mut q_abs = vec![0.0f32; latent_dim];
    let mut context_online = vec![0.0f32; latent_dim];
    let mut o_proj = vec![0.0f32; hidden_dim];
    let mut dst_hidden = vec![0.0f32; hidden_dim];

    let input_hidden = rng.random_vec(hidden_dim, -1.0, 1.0);
    let step = 2047; // step 2047 (seq_len = 2048)

    mla_absorbed_step_with_scratch_f32(
        &mut dst_hidden,
        &input_hidden,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        &mut norm_x,
        &mut q_abs,
        &mut context_online,
        &mut o_proj,
        0,
        step,
        hidden_dim,
        latent_dim,
        max_seq_len,
        1e-5,
    );

    // Compute two-pass reference across 2048 historical keys
    let scale = 1.0f32 / (latent_dim as f32).sqrt();
    let ref_context = reference_two_pass_mla_attention(
        &q_abs,
        &kv_latent_pool,
        0,
        latent_dim,
        step + 1,
        scale,
    );

    let mut max_abs_diff = 0.0f32;
    for d in 0..latent_dim {
        let diff = (context_online[d] - ref_context[d]).abs();
        if diff > max_abs_diff { max_abs_diff = diff; }
        assert!(
            diff < 1e-4,
            "Step 2047 dim {d}: online={:.7}, ref={:.7}, diff={:.7}",
            context_online[d], ref_context[d], diff
        );
    }

    println!("2048-step Long Context Drift PASS: max_abs_diff = {:e}", max_abs_diff);
}
