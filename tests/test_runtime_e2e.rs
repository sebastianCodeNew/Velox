//! Integration Tests for Steps 8 & 9: Zero-Copy Win32 `.safetensors` Memory Mapping,
//! Numerical Kernels, and End-to-End Token Generation on Real `SmolLM-135M` Weights.

use std::path::Path;
use veloxc::cli::{run_cli_with_io, EXIT_SUCCESS};
use veloxc::diagnostics::DiagnosticCode;
use veloxc::runtime::{
    gqa_attention_step_f32, matmul_transposed_f32, mla_absorbed_step_f32, rmsnorm_f32,
    rope_rotate_half_in_place_f32, sample_argmax_f32, sample_top_p_f32, swiglu_mlp_residual_f32,
    MmapSafetensors, TransformerConfig, VeloxStaticExecutionEngine,
};

#[test]
fn test_step8_mmap_safetensors_header_and_e013_missing_key() {
    let smollm_path = Path::new("smollm-135m.safetensors");
    if !smollm_path.exists() {
        return;
    }

    let st = MmapSafetensors::open(smollm_path).expect("Win32 zero-copy mmap must succeed");
    assert!(st.tensors.contains_key("model.embed_tokens.weight"));
    assert!(st.tensors.contains_key("model.norm.weight"));
    assert!(st.tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));

    // Verify weight tying fallback: lm_head.weight -> model.embed_tokens.weight
    let (lm_head, lm_shape) = st
        .get_tensor_f32("lm_head.weight", veloxc::types::ScalarType::F32)
        .expect("Tied lm_head.weight must resolve to model.embed_tokens.weight");
    assert_eq!(lm_head.len(), 49152 * 576);
    assert_eq!(lm_shape, &[49152, 576]);

    // Verify missing key returns E013 diagnostic code
    let err = st.get_entry("non_existent.tensor.weight").unwrap_err();
    assert_eq!(err, DiagnosticCode::E013);
}

#[test]
fn test_step8_numerical_kernels_edge_cases() {
    // RMSNorm on uniform vector
    let x = [2.0f32, 2.0, 2.0, 2.0];
    let w = [1.0f32, 0.5, -1.0, 2.0];
    let mut out = [0.0f32; 4];
    rmsnorm_f32(&mut out, &x, &w, 1e-5);
    assert!((out[0] - 1.0).abs() < 1e-5);
    assert!((out[1] - 0.5).abs() < 1e-5);
    assert!((out[2] + 1.0).abs() < 1e-5);
    assert!((out[3] - 2.0).abs() < 1e-5);

    // RoPE at pos=0 is identity
    let mut q = [1.25f32, -0.75, 0.5, 2.5];
    let q_orig = q;
    rope_rotate_half_in_place_f32(&mut q, 1, 4, 0, 10000.0);
    for i in 0..4 {
        assert!((q[i] - q_orig[i]).abs() < 1e-6);
    }

    // GQA Attention at pos=0 copies v_cache[0] directly
    let q_vec = [1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5]; // 2 q heads, dim=4
    let k_cache = [1.0f32, 0.0, 0.0, 1.0]; // 1 kv head, dim=4
    let v_cache = [3.5f32, -1.5, 0.25, 7.0]; // 1 kv head, dim=4
    let mut attn_out = [0.0f32; 8];
    let mut scores = [0.0f32; 4];
    gqa_attention_step_f32(&mut attn_out, &mut scores, &q_vec, &k_cache, &v_cache, 2, 1, 4, 0);
    assert_eq!(&attn_out[0..4], &v_cache);
    assert_eq!(&attn_out[4..8], &v_cache);

    // SwiGLU MLP + Residual
    let mut res = [1.0f32, 2.0];
    let mut gate = [0.0f32; 2];
    let mut up = [0.0f32; 2];
    let mut down = [0.0f32; 2];
    let norm_x = [1.0f32, 1.0];
    let eye = [1.0f32, 0.0, 0.0, 1.0];
    swiglu_mlp_residual_f32(
        &mut res, &mut gate, &mut up, &mut down, &norm_x, &eye, &eye, &eye, 2, 2,
    );
    let silu_1 = 1.0f32 / (1.0f32 + (-1.0f32).exp());
    assert!((res[0] - (1.0 + silu_1)).abs() < 1e-5);
    assert!((res[1] - (2.0 + silu_1)).abs() < 1e-5);

    // Matmul transposed
    let mut mm_out = [0.0f32; 2];
    matmul_transposed_f32(&mut mm_out, &[2.0, 3.0], &[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!((mm_out[0] - 8.0).abs() < 1e-6);
    assert!((mm_out[1] - 18.0).abs() < 1e-6);

    // Argmax and Top-P sampling
    let logits = [-2.0f32, 0.5, 9.2, 3.1];
    assert_eq!(sample_argmax_f32(&logits), 2);
    let mut probs = [0.0f32; 4];
    let mut idx = [0u32; 4];
    assert_eq!(
        sample_top_p_f32(&logits, &mut probs, &mut idx, 0.0, 0.9, 0.5),
        2
    );
}

#[test]
fn test_step8_mla_absorbed_step_mathematical_online_softmax() {
    // Test DeepSeek-V3 Multi-Head Latent Attention with Absorbed Projections:
    // Dimensions: hidden_dim = 4, latent_dim = 2, max_seq_len = 4
    let max_s = 4;
    let d_c = 2;
    let mut kv_latent_pool = vec![0.0f32; max_s * d_c];

    let w_norm = [1.0f32; 4];
    // Projection matrices:
    // W_DKV (2 rows x 4 cols): projects [norm_x[0], norm_x[1]] into latent slot
    let w_dkv = [
        1.0f32, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ];
    // W_absorbed_q (2 rows x 4 cols): projects [norm_x[0], norm_x[1]] into q_abs
    let w_absorbed_q = [
        1.0f32, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ];
    // W_absorbed_o (4 rows x 2 cols): projects context into hidden dimension
    let w_absorbed_o = [
        1.0f32, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
    ];

    // --- Step 0 ---
    let input_0 = [2.0f32, 2.0, 2.0, 2.0];
    let mut dst_0 = [0.0f32; 4];
    mla_absorbed_step_f32(
        &mut dst_0,
        &input_0,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        0,
        0,
        1e-5,
    );

    // Verification for Step 0:
    // norm_x ~ [1.0, 1.0, 1.0, 1.0]
    // slot_0 = [1.0, 1.0] stored in kv_latent_pool[0..2]
    // q_abs_0 = [1.0, 1.0]
    // Attention over 1 key: softmax(score_0) = 1.0, ctx = k_0 = [1.0, 1.0]
    // o_proj = [1.0, 1.0, 0.0, 0.0]
    // dst_0 = input_0 + o_proj = [3.0, 3.0, 2.0, 2.0]
    assert!((kv_latent_pool[0] - 1.0).abs() < 1e-4, "KV latent slot 0 written correctly");
    assert!((kv_latent_pool[1] - 1.0).abs() < 1e-4, "KV latent slot 0 written correctly");
    assert!((dst_0[0] - 3.0).abs() < 1e-4);
    assert!((dst_0[1] - 3.0).abs() < 1e-4);
    assert!((dst_0[2] - 2.0).abs() < 1e-4);
    assert!((dst_0[3] - 2.0).abs() < 1e-4);

    // --- Step 1 ---
    let input_1 = [2.0f32, -2.0, 2.0, -2.0];
    let mut dst_1 = [0.0f32; 4];
    mla_absorbed_step_f32(
        &mut dst_1,
        &input_1,
        &w_norm,
        &w_dkv,
        &w_absorbed_q,
        &w_absorbed_o,
        &mut kv_latent_pool,
        0,
        1,
        1e-5,
    );

    // Verification for Step 1:
    // Slot 1 written into kv_latent_pool[2..4]
    // Reference two-pass softmax calculation across history [k0, k1]:
    let k0 = &kv_latent_pool[0..2];
    let k1 = &kv_latent_pool[2..4];
    let q = [k1[0], k1[1]];
    let scale = 1.0f32 / (2.0f32).sqrt();
    let s0 = (q[0] * k0[0] + q[1] * k0[1]) * scale;
    let s1 = (q[0] * k1[0] + q[1] * k1[1]) * scale;
    let max_s_val = s0.max(s1);
    let e0 = (s0 - max_s_val).exp();
    let e1 = (s1 - max_s_val).exp();
    let p0 = e0 / (e0 + e1);
    let p1 = e1 / (e0 + e1);
    let expected_ctx_0 = p0 * k0[0] + p1 * k1[0];
    let expected_ctx_1 = p0 * k0[1] + p1 * k1[1];
    let expected_o0 = expected_ctx_0;
    let expected_o1 = expected_ctx_1;

    let expected_dst_0 = input_1[0] + expected_o0;
    let expected_dst_1 = input_1[1] + expected_o1;
    let expected_dst_2 = input_1[2];
    let expected_dst_3 = input_1[3];

    assert!((dst_1[0] - expected_dst_0).abs() < 1e-4, "Online softmax step 1 dim 0");
    assert!((dst_1[1] - expected_dst_1).abs() < 1e-4, "Online softmax step 1 dim 1");
    assert!((dst_1[2] - expected_dst_2).abs() < 1e-4, "Residual dim 2");
    assert!((dst_1[3] - expected_dst_3).abs() < 1e-4, "Residual dim 3");
}

#[test]
fn test_step9_real_smollm_135m_e2e_generation_and_cli() {
    let smollm_path = Path::new("smollm-135m.safetensors");
    assert!(
        smollm_path.exists(),
        "Real smollm-135m.safetensors must exist for Step 9 E2E verification"
    );

    let config = TransformerConfig::smollm_135m(32);
    let mut engine = VeloxStaticExecutionEngine::load_safetensors(smollm_path, config)
        .expect("Failed to load real smollm-135m.safetensors");

    let prompt = [1u32, 504, 346];
    let mut generated = [0u32; 4];
    let count = engine.generate_greedy_into(&prompt, 4, &mut generated);
    assert_eq!(count, 4);

    // Exact Hugging Face SmolLM-135M ground truth continuation for prompt [1, 504, 346]
    assert_eq!(
        generated,
        [359, 253, 4028, 282],
        "Generated tokens must match Hugging Face SmolLM-135M ground truth"
    );

    // Ensure all generated token IDs are valid vocabulary indices (< 49152) and finite logits
    for &tok in &generated {
        assert!((tok as usize) < 49152, "Generated token ID out of vocab range");
    }
    assert!(engine.logits.iter().all(|v| v.is_finite()));

    // Verify CLI --run flag produces deterministic output matching direct engine execution
    let mut out = Vec::new();
    let mut err = Vec::new();
    let args = vec![
        "--run".to_string(),
        "--tokens".to_string(),
        "1,504,346".to_string(),
        "--steps".to_string(),
        "4".to_string(),
        "examples/tinystories_smollm.vlx".to_string(),
    ];
    let code = run_cli_with_io(&args, &mut out, &mut err);
    assert_eq!(
        code,
        EXIT_SUCCESS,
        "CLI --run failed: {}",
        String::from_utf8_lossy(&err)
    );
    let report = String::from_utf8(out).unwrap();
    assert!(report.contains("VELOX ZERO-COPY NATIVE RUNTIME EXECUTION REPORT"));
    assert!(report.contains("Generated Tokens (4)   : [359, 253, 4028, 282]"));
    assert!(report.contains("Full Token Sequence    : [1, 504, 346, 359, 253, 4028, 282]"));
    assert!(
        report.contains("Dynamic Heap Allocs    : 0 (measured by Velox counting allocator over prefill + decode)"),
        "measured inference heap allocations must be 0, report:\n{report}"
    );
}

#[test]
fn test_step9_zero_dynamic_allocation_invariance() {
    let smollm_path = Path::new("smollm-135m.safetensors");
    if !smollm_path.exists() {
        return;
    }

    let config = TransformerConfig::smollm_135m(32);
    let mut engine = VeloxStaticExecutionEngine::load_safetensors(smollm_path, config)
        .expect("Failed to load real smollm-135m.safetensors");

    // Pointer stability check: buffer memory addresses must remain invariant during inference
    let addr_hidden = engine.hidden.as_ptr();
    let addr_norm = engine.norm_x.as_ptr();
    let addr_q = engine.q_buf.as_ptr();
    let addr_attn_out = engine.attn_out.as_ptr();
    let addr_k = engine.k_cache.as_ptr();
    let addr_v = engine.v_cache.as_ptr();
    let addr_scores = engine.scores_buf.as_ptr();
    let addr_gate = engine.gate_buf.as_ptr();
    let addr_up = engine.up_buf.as_ptr();
    let addr_down = engine.down_buf.as_ptr();
    let addr_logits = engine.logits.as_ptr();

    let mut out_tokens = [0u32; 8];
    // Empirical measurement: count every allocator call issued by this thread during inference.
    let (count, heap_allocs) = veloxc::alloc_counter::count_allocations(|| {
        engine.generate_greedy_into(&[1, 504, 346], 8, &mut out_tokens)
    });
    assert_eq!(count, 8);
    assert_eq!(
        heap_allocs, 0,
        "prefill + 8-step decode must issue zero heap allocations (measured {heap_allocs})"
    );

    // Single forward steps at increasing positions must also be allocation-free.
    for pos in 11..16 {
        let (_, step_allocs) = veloxc::alloc_counter::count_allocations(|| engine.forward_step(282, pos));
        assert_eq!(step_allocs, 0, "forward_step at pos {pos} allocated {step_allocs} times");
    }

    assert_eq!(engine.hidden.as_ptr(), addr_hidden, "hidden buffer must not reallocate");
    assert_eq!(engine.norm_x.as_ptr(), addr_norm, "norm_x buffer must not reallocate");
    assert_eq!(engine.q_buf.as_ptr(), addr_q, "q_buf must not reallocate");
    assert_eq!(engine.attn_out.as_ptr(), addr_attn_out, "attn_out must not reallocate");
    assert_eq!(engine.k_cache.as_ptr(), addr_k, "k_cache must not reallocate");
    assert_eq!(engine.v_cache.as_ptr(), addr_v, "v_cache must not reallocate");
    assert_eq!(engine.scores_buf.as_ptr(), addr_scores, "scores_buf must not reallocate");
    assert_eq!(engine.gate_buf.as_ptr(), addr_gate, "gate_buf must not reallocate");
    assert_eq!(engine.up_buf.as_ptr(), addr_up, "up_buf must not reallocate");
    assert_eq!(engine.down_buf.as_ptr(), addr_down, "down_buf must not reallocate");
    assert_eq!(engine.logits.as_ptr(), addr_logits, "logits must not reallocate");
}

#[test]
fn test_step8_hardening_empty_file_unexpected_eof() {
    let tmp_path = std::env::temp_dir().join(format!("velox_empty_hardening_{}.safetensors", std::process::id()));
    {
        let _f = std::fs::File::create(&tmp_path).expect("Failed to create empty test file");
    }
    let res = MmapSafetensors::open(&tmp_path);
    assert!(res.is_err(), "Empty safetensors file must fail");
    let err_msg = res.err().unwrap();
    assert!(
        err_msg.contains("Safetensors file is empty (0 bytes)"),
        "Expected 'Safetensors file is empty (0 bytes)', got: {err_msg}"
    );
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_step8_hardening_unaligned_tensor_pointer_e013() {
    use std::io::Write;
    let tmp_path = std::env::temp_dir().join(format!("velox_unaligned_hardening_{}.safetensors", std::process::id()));
    {
        // Write header with 1-byte unaligned data offset: start at 1, end at 17
        let header_json = "{\"unaligned.tensor\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[1,17]}}";
        let mut header_padded = header_json.to_string();
        while !(8 + header_padded.len()).is_multiple_of(8) {
            header_padded.push(' ');
        }
        let mut f = std::fs::File::create(&tmp_path).expect("Failed to create unaligned test file");
        f.write_all(&(header_padded.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header_padded.as_bytes()).unwrap();
        // 20 bytes of payload
        f.write_all(&[0u8; 20]).unwrap();
        f.flush().unwrap();
    }
    let st = MmapSafetensors::open(&tmp_path).expect("Open should succeed");
    let res = st.get_tensor_f32("unaligned.tensor", veloxc::types::ScalarType::F32);
    assert_eq!(res, Err(DiagnosticCode::E013), "Unaligned tensor pointer must emit E013");
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_step8_hardening_mla_latent_dim_branch_ordering() {
    // Test that multi-layer slice where layer > 0 correctly calculates latent_dim
    let hidden_dim = 64;
    let latent_dim = 128;
    let num_layers = 2;
    // w_dkv has size num_layers * latent_dim * hidden_dim = 2 * 128 * 64 = 16384
    let w_dkv = vec![0.01f32; num_layers * latent_dim * hidden_dim];
    let w_norm = vec![1.0f32; num_layers * hidden_dim];
    let w_absorbed_q = vec![0.01f32; num_layers * latent_dim * hidden_dim];
    let w_absorbed_o = vec![0.01f32; num_layers * hidden_dim * latent_dim];
    let max_seq_len = 16;
    let mut kv_latent_pool = vec![0.0f32; num_layers * max_seq_len * latent_dim];

    let mut dst_hidden = vec![0.0f32; hidden_dim];
    let input_hidden = vec![0.5f32; hidden_dim];

    // For layer = 1, (layer + 1) = 2.
    // (layer + 1) * hidden_dim = 2 * 64 = 128.
    // w_dkv.len() = 16384, which is a multiple of 128.
    // 16384 / 128 = 128.
    // Since latent_dim <= 1024 and hidden_dim <= 8192, mla_absorbed_step_f32 should succeed
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

    for val in &dst_hidden {
        assert!(val.is_finite(), "Output must be finite");
    }
}


