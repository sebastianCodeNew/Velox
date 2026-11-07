"""
Verification Test 4: Roofline Models (Prefill/Decode) & GPUDirect RDMA Pipelining
Project Velox - Step 1B SOTA Algorithms & Papers Challenge
"""

import numpy as np


def verify_llama3_70b_parameters():
    print("=== Step 4.1: First-Principles Llama-3-70B Parameter Count ===")
    L = 80
    d = 8192
    n_q = 64
    n_kv = 8
    d_head = 128
    d_ffn = 28672
    
    # Per layer:
    # Attention:
    # W_Q: d x (n_q * d_head) = 8192 x 8192
    # W_K: d x (n_kv * d_head) = 8192 x 1024
    # W_V: d x (n_kv * d_head) = 8192 x 1024
    # W_O: (n_q * d_head) x d = 8192 x 8192
    attn_params = (d * (n_q * d_head)) + (d * (n_kv * d_head)) + (d * (n_kv * d_head)) + ((n_q * d_head) * d)
    
    # SwiGLU FFN:
    # W_gate: d x d_ffn = 8192 x 28672
    # W_up: d x d_ffn = 8192 x 28672
    # W_down: d_ffn x d = 28672 x 8192
    ffn_params = 3 * (d * d_ffn)
    
    params_per_layer = attn_params + ffn_params
    total_params = L * params_per_layer
    
    weight_bytes_fp16 = total_params * 2
    
    print(f"Per-layer Attention Params: {attn_params:,}")
    print(f"Per-layer FFN Params:       {ffn_params:,}")
    print(f"Per-layer Total Params:     {params_per_layer:,}")
    print(f"Total Model Parameters:     {total_params:,} ({total_params / 1e9:.2f} B)")
    print(f"FP16 Weight Footprint:      {weight_bytes_fp16:,} bytes ({weight_bytes_fp16 / 1e9:.2f} GB)")
    
    assert total_params == 68451041280
    assert np.isclose(total_params / 1e9, 68.45, atol=0.01)
    assert np.isclose(weight_bytes_fp16 / 1e9, 136.90, atol=0.01)
    print(">> Parameter Count Derivation: PASSED!\n")
    return total_params, weight_bytes_fp16, L, d, n_kv, d_head


def verify_arithmetic_intensities(total_params, weight_bytes, L, d, n_kv, d_head):
    print("=== Step 4.2: Arithmetic Intensity Derivation for Prefill & Decode ===")
    
    # --- Prefill Phase: S_p = 4096 ---
    S_p = 4096
    flops_gemm_prefill = 2 * total_params * S_p
    flops_attn_prefill = 4 * L * d * (S_p**2)
    total_flops_prefill = flops_gemm_prefill + flops_attn_prefill
    
    bytes_kv_write = 2 * L * n_kv * d_head * S_p * 2  # FP16
    total_bytes_prefill = weight_bytes + bytes_kv_write
    
    I_prefill = total_flops_prefill / total_bytes_prefill
    
    print(f"Prefill (S_p = {S_p}):")
    print(f"  GEMM FLOPs:       {flops_gemm_prefill:.4e} ({flops_gemm_prefill / 1e12:.2f} TFLOPs)")
    print(f"  Attn FLOPs:       {flops_attn_prefill:.4e} ({flops_attn_prefill / 1e12:.2f} TFLOPs)")
    print(f"  Total FLOPs:      {total_flops_prefill:.4e} ({total_flops_prefill / 1e12:.2f} TFLOPs)")
    print(f"  Weight Bytes:     {weight_bytes / 1e9:.3f} GB")
    print(f"  KV Write Bytes:   {bytes_kv_write / 1e9:.3f} GB")
    print(f"  Total Bytes:      {total_bytes_prefill / 1e9:.3f} GB ({total_bytes_prefill:.4e} B)")
    print(f"  Arithmetic Intensity: {I_prefill:.2f} FLOPs/byte (Claimed: 4,374.3 FLOPs/byte)")
    assert np.isclose(I_prefill, 4374.3, atol=0.2)
    
    # --- Decode Phase: B_d = 16, S_d = 4096 ---
    B_d = 16
    S_d = 4096
    flops_gemm_decode16 = 2 * total_params * B_d
    flops_attn_decode16 = 4 * L * d * B_d * S_d
    total_flops_decode16 = flops_gemm_decode16 + flops_attn_decode16
    
    bytes_kv_read16 = 2 * L * n_kv * d_head * B_d * S_d * 2
    total_bytes_decode16 = weight_bytes + bytes_kv_read16
    I_decode16 = total_flops_decode16 / total_bytes_decode16
    
    print(f"\nDecode (B_d = {B_d}, S_d = {S_d}):")
    print(f"  Total FLOPs:      {total_flops_decode16:.4e} ({total_flops_decode16 / 1e12:.2f} TFLOPs)")
    print(f"  KV Read Bytes:    {bytes_kv_read16 / 1e9:.3f} GB")
    print(f"  Total Bytes:      {total_bytes_decode16 / 1e9:.3f} GB")
    print(f"  Arithmetic Intensity: {I_decode16:.2f} FLOPs/byte (Claimed: 14.9 FLOPs/byte)")
    assert np.isclose(I_decode16, 14.9, atol=0.1)
    
    # --- Decode Phase: B_d = 1, S_d = 4096 ---
    flops_gemm_decode1 = 2 * total_params * 1
    flops_attn_decode1 = 4 * L * d * 1 * S_d
    total_flops_decode1 = flops_gemm_decode1 + flops_attn_decode1
    bytes_kv_read1 = 2 * L * n_kv * d_head * 1 * S_d * 2
    total_bytes_decode1 = weight_bytes + bytes_kv_read1
    I_decode1 = total_flops_decode1 / total_bytes_decode1
    
    print(f"\nDecode (B_d = 1, S_d = {S_d}):")
    print(f"  Total FLOPs:      {total_flops_decode1:.4e} ({total_flops_decode1 / 1e9:.2f} GFLOPs)")
    print(f"  Arithmetic Intensity: {I_decode1:.2f} FLOPs/byte (Claimed: 1.07 FLOPs/byte)")
    assert np.isclose(I_decode1, 1.07, atol=0.01)
    
    # --- Roofline Analysis on H100 SXM5 ---
    R_peak = 989.5e12   # FLOP/s
    BW_hbm = 3350e9     # B/s
    I_knee = R_peak / BW_hbm
    print(f"\nH100 SXM5 Hardware Roofline:")
    print(f"  Peak FP16 Tensor Core: {R_peak / 1e12:.1f} TFLOPs/s")
    print(f"  HBM3 Bandwidth:        {BW_hbm / 1e9:.0f} GB/s")
    print(f"  Knee Point I_knee:     {I_knee:.2f} FLOPs/byte (Claimed: 295.4 FLOPs/byte)")
    assert np.isclose(I_knee, 295.4, atol=0.2)
    
    # Compute utilization in decode B=16
    t_mem = total_bytes_decode16 / BW_hbm
    max_flops_window = R_peak * t_mem
    utilization = (total_flops_decode16 / max_flops_window) * 100
    print(f"  Decode Step Physical Memory Time: {t_mem * 1000:.2f} ms")
    print(f"  Tensor Core Utilization (Decode B=16): {utilization:.2f}% (Claimed: 5.05%)")
    assert np.isclose(utilization, 5.05, atol=0.05)
    print(">> Arithmetic Intensity and Roofline Derivation: PASSED!\n")


def verify_layerwise_rdma_transfer():
    print("=== Step 4.3: Layer-wise GPUDirect RDMA Pipelining Verification ===")
    L = 80
    n_kv = 8
    d_head = 128
    S_p = 4096
    sizeof_fp16 = 2
    
    # Size of KV cache generated by 1 layer:
    # 2 * n_kv * d_head * S_p * sizeof(fp16)
    size_layer_kv_bytes = 2 * n_kv * d_head * S_p * sizeof_fp16
    size_layer_kv_mb = size_layer_kv_bytes / (1024 * 1024)
    print(f"Layer KV Cache Size: {size_layer_kv_bytes:,} bytes = {size_layer_kv_mb:.1f} MiB")
    assert size_layer_kv_bytes == 16777216
    
    # Transfer latencies:
    # 1. Standard single 400 Gbps link = 50 GB/s
    bw_single = 50e9 # B/s
    # In text: T_layer_xfer = 16.0 MB / 50 GB/s = 0.320 ms
    # If 16.0e6 / 50e9 = 0.320 ms; if 16777216 / 50e9 = 0.3355 ms
    t_xfer_text_claim = (16.0e6) / bw_single
    t_xfer_exact = size_layer_kv_bytes / bw_single
    print(f"Transfer Time (Single 400 Gbps, 50 GB/s):")
    print(f"  Using 16.0 MB: {t_xfer_text_claim * 1000:.3f} ms ({t_xfer_text_claim * 1e6:.1f} us)")
    print(f"  Using 16.78 MB: {t_xfer_exact * 1000:.3f} ms ({t_xfer_exact * 1e6:.1f} us)")
    
    # 2. 8x H100 with 8x ConnectX-7 = 400 GB/s
    bw_cluster = 400e9 # B/s
    t_xfer_cluster_claim = (16.0e6) / bw_cluster
    t_xfer_cluster_exact = size_layer_kv_bytes / bw_cluster
    print(f"Transfer Time (8x ConnectX-7, 400 GB/s):")
    print(f"  Using 16.0 MB: {t_xfer_cluster_claim * 1000:.3f} ms ({t_xfer_cluster_claim * 1e6:.1f} us)")
    print(f"  Using 16.78 MB: {t_xfer_cluster_exact * 1000:.3f} ms ({t_xfer_cluster_exact * 1e6:.1f} us)")
    
    # Layer compute time:
    # 8x H100 TP=8: peak FP16 = 8 * 989.5 TFLOPs = 7916 TFLOPs/s
    # 65% MFU sustained throughput = 0.65 * 7916 = 5145.4 TFLOPs/s
    # Total prefill FLOPs = 604.7 TFLOPs / 80 layers = 7.559 TFLOPs/layer
    flops_per_layer = 604.7e12 / 80
    sustained_throughput = 0.65 * (8 * 989.5e12)
    t_compute_layer = flops_per_layer / sustained_throughput
    print(f"\nLayer Compute Time (8x H100, 65% MFU):")
    print(f"  FLOPs per layer:        {flops_per_layer / 1e12:.3f} TFLOPs")
    print(f"  Sustained Throughput:   {sustained_throughput / 1e12:.1f} TFLOPs/s")
    print(f"  Compute Time per layer: {t_compute_layer * 1000:.3f} ms (Claimed: 1.469 ms)")
    assert np.isclose(t_compute_layer * 1000, 1.469, atol=0.01)
    
    # Check 100% hiding condition:
    # T_layer_xfer < T_layer_compute
    ratio_single = t_xfer_text_claim / t_compute_layer
    ratio_cluster = t_xfer_cluster_claim / t_compute_layer
    print(f"\nHiding Ratios (T_xfer / T_compute):")
    print(f"  Single 400 Gbps Link: {ratio_single:.3f} (Transfer takes only {ratio_single * 100:.1f}% of compute window)")
    print(f"  8x 400 Gbps Cluster:  {ratio_cluster:.3f} (Transfer takes only {ratio_cluster * 100:.1f}% of compute window)")
    print(f"Exposed End-to-End Latency Delta: {t_xfer_text_claim * 1000:.2f} ms (at layer L)")
    
    assert t_xfer_text_claim < t_compute_layer
    assert t_xfer_exact < t_compute_layer
    print(">> 100% Transfer Latency Hiding Proof: FULLY VERIFIED!\n")


if __name__ == "__main__":
    params, weight_bytes, L, d, n_kv, d_head = verify_llama3_70b_parameters()
    verify_arithmetic_intensities(params, weight_bytes, L, d, n_kv, d_head)
    verify_layerwise_rdma_transfer()
    print("ALL ROOFLINE & RDMA VERIFICATIONS COMPLETED SUCCESSFULLY!")
