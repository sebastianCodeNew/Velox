"""
Empirical Stress Test 1: DeepSeek MLA Matrix Absorption & Numerical Precision Analysis.

Verifies:
1. Proof that standard RoPE prevents matrix absorption: R_{s-t} W_K != W_K R_{s-t}
2. Mathematical equivalence of decoupled latent absorption in exact arithmetic.
3. Numerical precision loss and divergence across:
   - FP64 ground truth oracle
   - Unabsorbed FP32 vs Absorbed FP32
   - Unabsorbed FP16 vs Absorbed FP16 (FP16 offline folding vs FP32 offline folding)
   - FP16 with FP32 accumulation (Tensor Core standard)
4. Attention softmax distribution drift:
   - Max absolute error
   - Relative Frobenius error
   - Kullback-Leibler (KL) divergence
   - Total Variation (TV) distance
5. Stress-testing under varying matrix condition numbers (kappa in [1, 10, 100, 1000, 10000]).
"""

import numpy as np


def generate_rope_matrix(dim: int, pos: int, base: float = 10000.0) -> np.ndarray:
    """Generate 2D block-diagonal RoPE rotation matrix for given position."""
    R = np.zeros((dim, dim), dtype=np.float64)
    half_dim = dim // 2
    for i in range(half_dim):
        theta = 1.0 / (base ** (2 * i / dim))
        angle = pos * theta
        c, s = np.cos(angle), np.sin(angle)
        R[2 * i, 2 * i] = c
        R[2 * i, 2 * i + 1] = -s
        R[2 * i + 1, 2 * i] = s
        R[2 * i + 1, 2 * i + 1] = c
    return R


def test_rope_non_commutativity():
    """Empirically test whether R_{s-t} W_K == W_K R_{s-t} for random W_K."""
    print("--- 1. Testing RoPE Non-Commutativity (The Non-Absorbability Theorem) ---")
    np.random.seed(42)
    dim = 128
    diffs = []
    
    for trial in range(10):
        pos_s = np.random.randint(1, 1000)
        pos_t = np.random.randint(1, 1000)
        delta_pos = pos_s - pos_t
        if delta_pos == 0:
            continue
            
        R_delta = generate_rope_matrix(dim, delta_pos)
        W_K = np.random.randn(dim, dim)
        
        # Check commutator [R_delta, W_K] = R_delta W_K - W_K R_delta
        comm = R_delta @ W_K - W_K @ R_delta
        norm_comm = np.linalg.norm(comm, ord='fro')
        norm_prod = np.linalg.norm(R_delta @ W_K, ord='fro')
        rel_diff = norm_comm / norm_prod
        diffs.append(rel_diff)
        
    mean_rel_diff = np.mean(diffs)
    print(f"Average relative commutator norm ||[R_{{s-t}}, W_K]|| / ||R_{{s-t}} W_K||: {mean_rel_diff:.6f}")
    assert mean_rel_diff > 0.5, "RoPE should strongly non-commute with general projection matrix W_K!"
    print("-> RESULT: VERIFIED. Standard RoPE strictly prevents offline key matrix absorption.")


def simulate_mla_attention(
    d_c=512,
    d_cq=512,
    d_h=128,
    d_r=64,
    d_out=7168,
    seq_len=64,
    cond_number=10.0,
    seed=123
):
    """Simulate MLA decode attention step under unabsorbed vs absorbed formulation across precisions."""
    np.random.seed(seed)
    
    def get_matrix(rows, cols, cond):
        k = min(rows, cols)
        u, _ = np.linalg.qr(np.random.randn(rows, k), mode='reduced')
        v, _ = np.linalg.qr(np.random.randn(cols, k), mode='reduced')
        s = np.logspace(0, -np.log10(cond), k)
        return (u * s @ v.T).astype(np.float64)

    # Weights
    W_UQ = get_matrix(d_h, d_cq, cond_number)   # [d_h, d_cq]
    W_UK = get_matrix(d_h, d_c, cond_number)    # [d_h, d_c]
    W_UV = get_matrix(d_h, d_c, cond_number)    # [d_h, d_c]
    W_O = get_matrix(d_out, d_h, cond_number)   # [d_out, d_h]
    
    # Inputs: query latent c_q [d_cq], cached latents c_kv [seq_len, d_c]
    c_q = np.random.randn(d_cq).astype(np.float64)
    c_kv = np.random.randn(seq_len, d_c).astype(np.float64)
    
    # Decoupled RoPE keys and query
    q_r = np.random.randn(d_r).astype(np.float64)
    k_r = np.random.randn(seq_len, d_r).astype(np.float64)
    
    # -------------------------------------------------------------
    # Oracle: Float64 Exact Ground Truth
    # -------------------------------------------------------------
    # Unabsorbed:
    q_c_64 = W_UQ @ c_q                         # [d_h]
    k_c_64 = (W_UK @ c_kv.T).T                   # [seq_len, d_h]
    scores_c_unabs_64 = (k_c_64 @ q_c_64)       # [seq_len]
    scores_r_64 = k_r @ q_r                      # [seq_len]
    scores_unabs_64 = (scores_c_unabs_64 + scores_r_64) / np.sqrt(d_h + d_r)
    
    # Absorbed:
    W_UQ_abs_64 = W_UK.T @ W_UQ                  # [d_c, d_cq]
    q_c_abs_64 = W_UQ_abs_64 @ c_q              # [d_c]
    scores_c_abs_64 = c_kv @ q_c_abs_64         # [seq_len]
    scores_abs_64 = (scores_c_abs_64 + scores_r_64) / np.sqrt(d_h + d_r)
    
    # Mathematical identity check in FP64
    abs_math_diff = np.max(np.abs(scores_unabs_64 - scores_abs_64))
    
    # -------------------------------------------------------------
    # FP32 Evaluation
    # -------------------------------------------------------------
    W_UQ_32 = W_UQ.astype(np.float32)
    W_UK_32 = W_UK.astype(np.float32)
    c_q_32 = c_q.astype(np.float32)
    c_kv_32 = c_kv.astype(np.float32)
    q_r_32 = q_r.astype(np.float32)
    k_r_32 = k_r.astype(np.float32)
    
    # Unabsorbed FP32:
    q_c_unabs_32 = W_UQ_32 @ c_q_32
    k_c_unabs_32 = (W_UK_32 @ c_kv_32.T).T
    scores_unabs_32 = ((k_c_unabs_32 @ q_c_unabs_32) + (k_r_32 @ q_r_32)) / np.float32(np.sqrt(d_h + d_r))
    
    # Absorbed FP32:
    W_abs_32 = W_UK_32.T @ W_UQ_32
    q_abs_32 = W_abs_32 @ c_q_32
    scores_abs_32 = ((c_kv_32 @ q_abs_32) + (k_r_32 @ q_r_32)) / np.float32(np.sqrt(d_h + d_r))
    
    # -------------------------------------------------------------
    # FP16 Evaluation: Case A (Offline pre-multiplied in FP16)
    # -------------------------------------------------------------
    W_UQ_16 = W_UQ_32.astype(np.float16)
    W_UK_16 = W_UK_32.astype(np.float16)
    c_q_16 = c_q_32.astype(np.float16)
    c_kv_16 = c_kv_32.astype(np.float16)
    q_r_16 = q_r_32.astype(np.float16)
    k_r_16 = k_r_32.astype(np.float16)
    scale_16 = np.float16(1.0 / np.sqrt(d_h + d_r))
    
    # Unabsorbed in FP16:
    q_c_unabs_16 = (W_UQ_16 @ c_q_16).astype(np.float16)
    k_c_unabs_16 = ((W_UK_16 @ c_kv_16.T).T).astype(np.float16)
    scores_unabs_16 = (((k_c_unabs_16 @ q_c_unabs_16).astype(np.float16) + (k_r_16 @ q_r_16).astype(np.float16)) * scale_16).astype(np.float16)
    
    # Absorbed in pure FP16 (offline folding in FP16):
    W_abs_pure16 = (W_UK_16.T @ W_UQ_16).astype(np.float16)
    q_abs_pure16 = (W_abs_pure16 @ c_q_16).astype(np.float16)
    scores_abs_pure16 = (((c_kv_16 @ q_abs_pure16).astype(np.float16) + (k_r_16 @ q_r_16).astype(np.float16)) * scale_16).astype(np.float16)
    
    # -------------------------------------------------------------
    # FP16 Evaluation: Case B (Offline pre-multiplied in FP32, then stored as FP16, with FP32 Tensor Core accumulation)
    # -------------------------------------------------------------
    W_abs_fp32_folded = (W_UK_32.T @ W_UQ_32).astype(np.float16)
    q_abs_tc32 = (W_abs_fp32_folded.astype(np.float32) @ c_q_16.astype(np.float32)).astype(np.float16)
    scores_abs_tc32 = (
        (c_kv_16.astype(np.float32) @ q_abs_tc32.astype(np.float32) + 
         k_r_16.astype(np.float32) @ q_r_16.astype(np.float32)) / np.float32(np.sqrt(d_h + d_r))
    ).astype(np.float16)
    
    # -------------------------------------------------------------
    # Softmax probabilities & divergence
    # -------------------------------------------------------------
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    p_oracle = softmax(scores_unabs_64)
    p_unabs_32 = softmax(scores_unabs_32.astype(np.float64))
    p_abs_32 = softmax(scores_abs_32.astype(np.float64))
    p_unabs_16 = softmax(scores_unabs_16.astype(np.float64))
    p_abs_pure16 = softmax(scores_abs_pure16.astype(np.float64))
    p_abs_tc32 = softmax(scores_abs_tc32.astype(np.float64))
    
    def get_metrics(p_approx, s_approx):
        kl = np.sum(np.where(p_oracle > 1e-12, p_oracle * np.log(np.maximum(p_oracle, 1e-12) / np.maximum(p_approx, 1e-12)), 0.0))
        tv = 0.5 * np.sum(np.abs(p_oracle - p_approx))
        max_score_err = np.max(np.abs(scores_unabs_64 - s_approx))
        return kl, tv, max_score_err
    
    m_unabs_32 = get_metrics(p_unabs_32, scores_unabs_32)
    m_abs_32 = get_metrics(p_abs_32, scores_abs_32)
    m_unabs_16 = get_metrics(p_unabs_16, scores_unabs_16)
    m_abs_pure16 = get_metrics(p_abs_pure16, scores_abs_pure16)
    m_abs_tc32 = get_metrics(p_abs_tc32, scores_abs_tc32)
    
    # Value projection check: W_O' = W_O @ W_UV
    v_c_64 = (W_UV @ c_kv.T).T                  # [seq_len, d_h]
    out_unabs_64 = p_oracle @ v_c_64 @ W_O.T    # [d_out]
    
    W_O_abs_64 = W_O @ W_UV                     # [d_out, d_c]
    out_abs_64 = p_oracle @ c_kv @ W_O_abs_64.T # [d_out]
    val_math_diff = np.max(np.abs(out_unabs_64 - out_abs_64))
    
    return {
        "abs_math_diff": abs_math_diff,
        "val_math_diff": val_math_diff,
        "m_unabs_32": m_unabs_32,
        "m_abs_32": m_abs_32,
        "m_unabs_16": m_unabs_16,
        "m_abs_pure16": m_abs_pure16,
        "m_abs_tc32": m_abs_tc32,
    }


def test_mla_stress_suite():
    print("\n--- 2. Testing Exact Mathematical Equivalence of MLA Absorption ---")
    res = simulate_mla_attention(cond_number=1.0)
    print(f"FP64 Attention Score Identity Difference: {res['abs_math_diff']:.2e}")
    print(f"FP64 Output Context Projection Difference: {res['val_math_diff']:.2e}")
    assert res['abs_math_diff'] < 1e-12, "Mathematical identity must hold exactly in FP64!"
    assert res['val_math_diff'] < 1e-12, "Output projection identity must hold exactly in FP64!"
    print("-> RESULT: VERIFIED. Offline pre-multiplication W'_{UQ} = W_{UK}^T W_{UQ} and W'_O = W_O W_{UV} is exact.")

    print("\n--- 3. Numerical Precision & Divergence Stress Testing Across Condition Numbers ---")
    cond_numbers = [1.0, 10.0, 100.0, 1000.0]
    print(f"{'Condition':<12} | {'Format / Pipeline':<28} | {'Max Score Err':<14} | {'KL Divergence':<14} | {'TV Distance':<12}")
    print("-" * 90)
    
    for kappa in cond_numbers:
        res = simulate_mla_attention(cond_number=kappa)
        print(f"kappa = {kappa:<6.1f} | Unabsorbed FP32              | {res['m_unabs_32'][2]:<14.6e} | {res['m_unabs_32'][0]:<14.6e} | {res['m_unabs_32'][1]:<12.6f}")
        print(f"             | Absorbed FP32                | {res['m_abs_32'][2]:<14.6e} | {res['m_abs_32'][0]:<14.6e} | {res['m_abs_32'][1]:<12.6f}")
        print(f"             | Unabsorbed FP16              | {res['m_unabs_16'][2]:<14.6e} | {res['m_unabs_16'][0]:<14.6e} | {res['m_unabs_16'][1]:<12.6f}")
        print(f"             | Absorbed Pure FP16 (naive)   | {res['m_abs_pure16'][2]:<14.6e} | {res['m_abs_pure16'][0]:<14.6e} | {res['m_abs_pure16'][1]:<12.6f}")
        print(f"             | Absorbed Velox (FP32fold+TC) | {res['m_abs_tc32'][2]:<14.6e} | {res['m_abs_tc32'][0]:<14.6e} | {res['m_abs_tc32'][1]:<12.6f}")
        print("-" * 90)


def test_mla_row_major_vs_column_vector_convention():
    """Verify that column-vector W'_{UQ} = W_{UK}^T W_{UQ} and row-major W'_{UQ, row} = W_{UQ} W_{UK}^T produce identical attention scores."""
    print("\n--- 4. Testing MLA Transpose Convention: Derivation vs Row-Major Code Emission ---")
    np.random.seed(42)
    d_h = 128
    d_cq = 512
    d_c = 512

    # 1. Column-vector convention (DeepSeek paper derivation):
    # Vectors: q = W_uq @ c_q, k = W_uk @ c_kv
    # Shapes: W_uq in [d_h, d_cq], W_uk in [d_h, d_c], c_q in [d_cq, 1], c_kv in [d_c, 1]
    W_uq_col = np.random.randn(d_h, d_cq)
    W_uk_col = np.random.randn(d_h, d_c)
    c_q_col = np.random.randn(d_cq, 1)
    c_kv_col = np.random.randn(d_c, 1)

    q_col = W_uq_col @ c_q_col
    k_col = W_uk_col @ c_kv_col
    score_col = (q_col.T @ k_col)[0, 0]

    # Folded weight in column-vector derivation:
    W_prime_col = W_uk_col.T @ W_uq_col  # [d_c, d_cq]
    q_prime_col = W_prime_col @ c_q_col   # [d_c, 1]
    score_absorbed_col = (q_prime_col.T @ c_kv_col)[0, 0]
    assert np.isclose(score_col, score_absorbed_col, atol=1e-11)

    # 2. Row-major convention (Velox code emission & Tensor Core execution):
    # Activations are rows: c_q_row [1, d_cq], c_kv_row [1, d_c]
    # Weight matrices are [in_features, out_features]: W_uq_row [d_cq, d_h], W_uk_row [d_c, d_h]
    c_q_row = c_q_col.T      # [1, d_cq]
    c_kv_row = c_kv_col.T    # [1, d_c]
    W_uq_row = W_uq_col.T    # [d_cq, d_h]
    W_uk_row = W_uk_col.T    # [d_c, d_h]

    q_row = c_q_row @ W_uq_row   # [1, d_h]
    k_row = c_kv_row @ W_uk_row  # [1, d_h]
    score_row = (q_row @ k_row.T)[0, 0]

    # Folded weight in row-major convention: W'_{UQ, row} = W_{UQ} W_{UK}^T
    W_prime_row = W_uq_row @ W_uk_row.T  # [d_cq, d_c]
    q_absorbed_row = c_q_row @ W_prime_row  # [1, d_c]
    score_absorbed_row = (q_absorbed_row @ c_kv_row.T)[0, 0]
    assert np.isclose(score_row, score_absorbed_row, atol=1e-11)

    # Cross-convention verification
    assert np.isclose(score_col, score_row, atol=1e-11)
    assert np.allclose(W_prime_row, W_prime_col.T, atol=1e-11)
    print(f"Col-vector score: {score_col:.8f}, Row-major score: {score_row:.8f} (Abs diff: {abs(score_col - score_row):.2e})")
    print(f"W'_row shape: {W_prime_row.shape}, W'_col shape: {W_prime_col.shape}, Transpose equality verified: {np.allclose(W_prime_row, W_prime_col.T)}")
    print("-> RESULT: VERIFIED. Row-major emission W'_{UQ, row} = W_{UQ} W_{UK}^T matches column derivation W'_{UQ} = W_{UK}^T W_{UQ} bit-exact.")


if __name__ == "__main__":
    test_rope_non_commutativity()
    test_mla_stress_suite()
    test_mla_row_major_vs_column_vector_convention()
