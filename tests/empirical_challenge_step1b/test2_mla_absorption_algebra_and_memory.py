"""
Verification Test 2: Multi-Head Latent Attention (MLA) Matrix Absorption & Memory Footprint Verification
Project Velox - Step 1B SOTA Algorithms & Papers Challenge
"""

import numpy as np


def verify_mla_matrix_absorption():
    print("=== Step 2.1: MLA Matrix Absorption Numerical & Algebraic Proof ===")
    np.random.seed(42)
    
    # Architectural dimensions (DeepSeek-V2/V3 configuration)
    d = 5120         # Hidden dimension
    n_h = 16         # Test with 16 heads (identical math for 128)
    d_h = 128        # Head dimension
    d_c = 512        # KV latent dimension
    d_cp = 1536      # Query latent dimension (d_c')
    d_h_r = 64       # Decoupled RoPE dimension
    seq_len = 32     # Sequence length for decoding
    
    print(f"Dimensions: d={d}, n_h={n_h}, d_h={d_h}, d_c={d_c}, d_c'={d_cp}, d_h^R={d_h_r}, S={seq_len}")
    
    # 1. Random weight matrices
    # Query up-projection per head: W_{UQ, i} in R^{d_h x d_c'}
    # Key up-projection per head: W_{UK, i} in R^{d_h x d_c}
    # Value up-projection per head: W_{UV, i} in R^{d_h x d_c}
    # Output projection per head: W_{O, i} in R^{d x d_h}
    W_UQ = np.random.randn(n_h, d_h, d_cp) * 0.02
    W_UK = np.random.randn(n_h, d_h, d_c) * 0.02
    W_UV = np.random.randn(n_h, d_h, d_c) * 0.02
    W_O = np.random.randn(n_h, d, d_h) * 0.02
    
    # 2. Input latent activations
    # Query latent vector: c_t^Q in R^{d_c'}
    # Historical KV latent vectors: c_j^{KV} in R^{seq_len x d_c}
    c_Q = np.random.randn(d_cp)
    c_KV = np.random.randn(seq_len, d_c)
    
    # RoPE positional keys and query
    k_R = np.random.randn(seq_len, d_h_r)
    q_R = np.random.randn(n_h, d_h_r)
    
    # ==========================================
    # PATH A: Standard Uncompressed Computation
    # Materializing full uncompressed keys and values in memory
    # ==========================================
    logits_uncompressed = np.zeros((n_h, seq_len))
    heads_out_uncompressed = np.zeros((n_h, d_h))
    
    for i in range(n_h):
        # Uncompressed query: q_{t,i}^C = W_{UQ, i} @ c_t^Q (in R^{d_h})
        q_C_i = W_UQ[i] @ c_Q
        # Uncompressed keys: k_{j,i}^C = W_{UK, i} @ c_j^{KV} (in R^{seq_len x d_h})
        k_C_i = c_KV @ W_UK[i].T
        # Content score
        content_score = k_C_i @ q_C_i  # R^{seq_len}
        # RoPE score
        rope_score = k_R @ q_R[i]      # R^{seq_len}
        # Full score
        logits_uncompressed[i] = (content_score + rope_score) / np.sqrt(d_h + d_h_r)
        
        # Softmax
        attn_weights = np.exp(logits_uncompressed[i] - np.max(logits_uncompressed[i]))
        attn_weights /= np.sum(attn_weights)
        
        # Uncompressed values: v_{j,i}^C = W_{UV, i} @ c_j^{KV} (in R^{seq_len x d_h})
        v_C_i = c_KV @ W_UV[i].T
        # Attention output: u_{t,i} = sum_j alpha_j * v_{j,i}^C (in R^{d_h})
        heads_out_uncompressed[i] = attn_weights @ v_C_i
        
    # Final projection
    o_uncompressed = np.zeros(d)
    for i in range(n_h):
        o_uncompressed += W_O[i] @ heads_out_uncompressed[i]
        
    # ==========================================
    # PATH B: Compile-Time Absorbed Computation
    # Zero decompression of keys and values in memory!
    # ==========================================
    # Offline precomputation by compiler:
    # Absorbed query projection: W_{UQ, i}' = W_{UK, i}^T @ W_{UQ, i} in R^{d_c x d_c'}
    # Absorbed output projection: W_{O, i}' = W_{O, i} @ W_{UV, i} in R^{d x d_c}
    W_UQ_prime = np.zeros((n_h, d_c, d_cp))
    W_O_prime = np.zeros((n_h, d, d_c))
    for i in range(n_h):
        W_UQ_prime[i] = W_UK[i].T @ W_UQ[i]
        W_O_prime[i] = W_O[i] @ W_UV[i]
        
    logits_absorbed = np.zeros((n_h, seq_len))
    o_absorbed = np.zeros(d)
    
    for i in range(n_h):
        # Absorbed query: q_tilde_{t,i}^C = W_{UQ, i}' @ c_t^Q (in R^{d_c})
        q_tilde_C_i = W_UQ_prime[i] @ c_Q
        # Content score computed directly against latent vector c_KV without decompression!
        content_score_abs = c_KV @ q_tilde_C_i  # (seq_len x d_c) @ d_c = seq_len
        # RoPE score
        rope_score = k_R @ q_R[i]
        # Full score
        logits_absorbed[i] = (content_score_abs + rope_score) / np.sqrt(d_h + d_h_r)
        
        # Softmax
        attn_weights = np.exp(logits_absorbed[i] - np.max(logits_absorbed[i]))
        attn_weights /= np.sum(attn_weights)
        
        # Latent accumulation vector: u_tilde_{t,i} = sum_j alpha_j * c_j^{KV} (in R^{d_c})
        u_tilde_i = attn_weights @ c_KV
        
        # Direct absorbed projection to output hidden dimension
        o_absorbed += W_O_prime[i] @ u_tilde_i
        
    # Check max absolute differences
    logit_diff = np.max(np.abs(logits_uncompressed - logits_absorbed))
    output_diff = np.max(np.abs(o_uncompressed - o_absorbed))
    rel_output_diff = output_diff / np.linalg.norm(o_uncompressed)
    
    print(f"Max Absolute Logit Difference: {logit_diff:.2e}")
    print(f"Max Absolute Output Difference: {output_diff:.2e}")
    print(f"Relative Output Difference: {rel_output_diff:.2e}")
    
    assert logit_diff < 1e-12, f"Logit mismatch: {logit_diff}"
    assert rel_output_diff < 1e-12, f"Output mismatch: {rel_output_diff}"
    print(">> Algebraic and Numerical Equivalence of Matrix Absorption: PASSED!\n")


def verify_memory_footprint():
    print("=== Step 2.2: Memory Footprint Reduction Verification ===")
    # Parameters from DeepSeek-V2 / V3
    n_h = 128
    d_h = 128
    d_c = 512
    d_h_r = 64
    bytes_per_elem = 2  # FP16
    
    # 1. MHA
    mha_bytes = 2 * n_h * d_h * bytes_per_elem
    print(f"MHA: 2 * {n_h} * {d_h} * {bytes_per_elem} = {mha_bytes:,} bytes / token / layer")
    assert mha_bytes == 65536, f"MHA bytes mismatch: {mha_bytes}"
    
    # 2. GQA-8
    n_kv = 8
    gqa8_bytes = 2 * n_kv * d_h * bytes_per_elem
    print(f"GQA-8: 2 * {n_kv} * {d_h} * {bytes_per_elem} = {gqa8_bytes:,} bytes / token / layer")
    assert gqa8_bytes == 4096, f"GQA-8 bytes mismatch: {gqa8_bytes}"
    
    # 3. MLA
    mla_bytes = (d_c + d_h_r) * bytes_per_elem
    print(f"MLA: ({d_c} + {d_h_r}) * {bytes_per_elem} = {mla_bytes:,} bytes / token / layer")
    assert mla_bytes == 1152, f"MLA bytes mismatch: {mla_bytes}"
    
    # Reductions
    red_vs_mha = (mha_bytes - mla_bytes) / mha_bytes
    red_vs_gqa = (gqa8_bytes - mla_bytes) / gqa8_bytes
    print(f"Reduction vs MHA: {red_vs_mha * 100:.6f}% (Claimed: 98.24%)")
    print(f"Reduction vs GQA-8: {red_vs_gqa * 100:.6f}% (Claimed: 71.88%)")
    
    assert np.isclose(red_vs_mha, 64384.0 / 65536.0)
    assert np.isclose(red_vs_gqa, 2944.0 / 4096.0)
    
    # 60 layers, 128,000 tokens context
    num_layers = 60
    seq_tokens = 128000
    
    mha_total_bytes = num_layers * seq_tokens * mha_bytes
    gqa_total_bytes = num_layers * seq_tokens * gqa8_bytes
    mla_total_bytes = num_layers * seq_tokens * mla_bytes
    
    print(f"\n128k Context on 60 layers:")
    print(f"  MHA Total: {mha_total_bytes / 1e9:.2f} GB (decimal) | {mha_total_bytes / (1024**3):.2f} GiB")
    print(f"  GQA Total: {gqa_total_bytes / 1e9:.2f} GB (decimal) | {gqa_total_bytes / (1024**3):.2f} GiB")
    print(f"  MLA Total: {mla_total_bytes / 1e9:.2f} GB (decimal) | {mla_total_bytes / (1024**3):.2f} GiB")
    
    assert np.isclose(mha_total_bytes / 1e9, 503.31648, atol=0.1)
    assert np.isclose(gqa_total_bytes / 1e9, 31.45728, atol=0.1)
    assert np.isclose(mla_total_bytes / 1e9, 8.84736, atol=0.1)
    
    print(">> Memory Footprint and Reduction Claims: FULLY VERIFIED!\n")


if __name__ == "__main__":
    verify_mla_matrix_absorption()
    verify_memory_footprint()
    print("ALL MLA VERIFICATIONS COMPLETED SUCCESSFULLY!")
