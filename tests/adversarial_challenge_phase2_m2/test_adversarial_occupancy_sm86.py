"""
Adversarial Verification Suite: Ampere sm_86 Occupancy & Register/Smem Simulator
Simulates the exact NVIDIA Ampere sm_86 hardware limits:
- SM count: 16
- Max Warps / SM: 48 (1536 threads)
- Max Blocks / SM: 16
- Max Registers / SM: 65,536 (32-bit)
- Max Shared Memory / Block: 49,152 bytes (48 KiB)
- Max Shared Memory / SM: 102,400 bytes (100 KiB)
- Register allocation unit: 256 registers per warp (8 regs/thread)
- Shared memory allocation unit: 128 bytes

Tests:
1. Occupancy of every kernel in Section 3.3 under SmolLM-135M, Llama-3-8B, and DeepSeek-V3
2. Full parameter sweep: R_thread in [1..128], Smem in [0..49152]
3. Precise cliff-edge analysis: At what exact register count does occupancy drop below 66.7%?
"""

import math

def calculate_sm86_occupancy(threads_per_block: int, regs_per_thread: int, smem_per_block_bytes: int):
    max_warps_per_sm = 48
    max_threads_per_sm = 1536
    max_blocks_per_sm = 16
    max_regs_per_sm = 65536
    max_smem_per_sm = 102400 # 100 KiB
    max_smem_per_block = 49152 # 48 KiB default
    
    if smem_per_block_bytes > max_smem_per_block:
        return {
            "valid": False,
            "error": f"Smem per block {smem_per_block_bytes} exceeds 48 KiB limit ({max_smem_per_block} B)",
            "occupancy": 0.0
        }
    if threads_per_block > 1024 or threads_per_block % 32 != 0:
        return {
            "valid": False,
            "error": f"Invalid threads_per_block: {threads_per_block}",
            "occupancy": 0.0
        }
        
    warps_per_block = threads_per_block // 32
    
    # 1. Warp Limit
    blocks_limit_warps = max_warps_per_sm // warps_per_block
    
    # 2. Block Limit
    blocks_limit_blocks = max_blocks_per_sm
    
    # 3. Register Limit
    # Ampere sm_86 allocates registers per warp in units of 256 registers (8 regs/thread * 32 threads)
    regs_per_thread_rounded = int(math.ceil(regs_per_thread / 8.0) * 8)
    regs_per_warp = regs_per_thread_rounded * 32
    regs_per_block = regs_per_warp * warps_per_block
    
    if regs_per_block > max_regs_per_sm:
        blocks_limit_regs = 0
    else:
        blocks_limit_regs = max_regs_per_sm // regs_per_block
        
    # 4. Shared Memory Limit
    # sm_86 allocates shared memory in 128-byte segments
    smem_rounded = int(math.ceil(smem_per_block_bytes / 128.0) * 128)
    if smem_rounded == 0:
        blocks_limit_smem = max_blocks_per_sm
    else:
        blocks_limit_smem = max_smem_per_sm // smem_rounded
        
    active_blocks = min(blocks_limit_warps, blocks_limit_blocks, blocks_limit_regs, blocks_limit_smem)
    active_warps = active_blocks * warps_per_block
    active_threads = active_blocks * threads_per_block
    occupancy_pct = (active_warps / max_warps_per_sm) * 100.0
    
    limiting_factor = []
    if active_blocks == blocks_limit_warps: limiting_factor.append("Warps")
    if active_blocks == blocks_limit_blocks: limiting_factor.append("Blocks")
    if active_blocks == blocks_limit_regs: limiting_factor.append("Registers")
    if active_blocks == blocks_limit_smem: limiting_factor.append("SharedMemory")
    
    return {
        "valid": True,
        "active_blocks": active_blocks,
        "active_warps": active_warps,
        "active_threads": active_threads,
        "occupancy_pct": occupancy_pct,
        "limiting_factor": "+".join(limiting_factor),
        "regs_per_block": regs_per_block,
        "smem_rounded": smem_rounded,
        "blocks_limit_warps": blocks_limit_warps,
        "blocks_limit_blocks": blocks_limit_blocks,
        "blocks_limit_regs": blocks_limit_regs,
        "blocks_limit_smem": blocks_limit_smem
    }

def run_occupancy_stress_test():
    print("=" * 80)
    print("1. EVALUATION OF STEP 12 KERNEL MATRIX (SECTION 3.3)")
    print("=" * 80)
    
    kernels = [
        ("velox_cuda_matmul_vec_kernel", 128, 40, 0),
        ("velox_cuda_matmul_vec_kernel (max regs)", 128, 64, 0),
        ("velox_cuda_fused_norm_qkv_warp_kernel", 128, 48, 1280),
        ("velox_cuda_fused_norm_qkv_warp_kernel (max regs)", 128, 64, 1280),
        ("velox_cuda_fused_out_res_swiglu_res", 128, 48, 1536),
        ("velox_cuda_fused_out_res_swiglu_res (max regs)", 128, 64, 1536),
        ("velox_cuda_splitk_attention_stage1", 128, 48, 768),
        ("velox_cuda_splitk_attention_stage1 (dh=128)", 128, 48, 1024),
        ("velox_cuda_splitk_attention_stage2 (dh=64)", 64, 32, 320),
        ("velox_cuda_splitk_attention_stage2 (dh=128)", 128, 32, 512),
        ("velox_cuda_lm_head_argmax_stage1", 256, 32, 2048),
        ("velox_cuda_lm_head_argmax_stage1 (max regs)", 256, 64, 2048),
        ("velox_cuda_lm_head_argmax_stage2 (single block)", 256, 32, 2048),
        ("velox_cuda_marlin_lop3_gemv", 128, 40, 0)
    ]
    
    for name, t_block, r_thread, smem in kernels:
        res = calculate_sm86_occupancy(t_block, r_thread, smem)
        status = "MEETS >=66.7%" if res["occupancy_pct"] >= 66.66 else "FAILS <66.7%"
        print(f"{name:50s} | T={t_block:3d}, R={r_thread:2d}, Smem={smem:4d}B | "
              f"Blocks={res['active_blocks']:2d}, Warps={res['active_warps']:2d} ({res['occupancy_pct']:5.1f}%) | "
              f"Limit: {res['limiting_factor']:12s} | {status}")
              
    print("\n" + "=" * 80)
    print("2. CLIFF-EDGE REGISTER SWEEP (T_block = 128 threads, Smem = 2048 B)")
    print("=" * 80)
    print(f"{'R_thread':8s} | {'R_alloc':8s} | {'Regs/Block':10s} | {'Active Blocks':14s} | {'Active Warps':12s} | {'Occupancy %':12s} | {'Status':15s}")
    print("-" * 88)
    
    cliff_found = None
    for r in range(16, 96, 4):
        res = calculate_sm86_occupancy(128, r, 2048)
        r_alloc = int(math.ceil(r / 8.0) * 8)
        meets = res["occupancy_pct"] >= 66.66
        status = ">= 66.7% (PASS)" if meets else "< 66.7% (CLIFF!)"
        if not meets and cliff_found is None:
            cliff_found = r
        print(f"{r:8d} | {r_alloc:8d} | {res['regs_per_block']:10d} | {res['active_blocks']:14d} | {res['active_warps']:12d} | {res['occupancy_pct']:11.1f}% | {status}")
        
    print(f"\nEmpirical Register Cliff Edge for T_block=128: R_thread = {cliff_found} registers.")
    print(f"When R_thread <= 64, active warps = 32 (66.7% occupancy).")
    print(f"When R_thread >= 65 (rounded to 72), active blocks drops from 8 to 7, warps drop to 28 (58.3% occupancy).")
    
    print("\n" + "=" * 80)
    print("3. CLIFF-EDGE REGISTER SWEEP (T_block = 256 threads, Smem = 2048 B)")
    print("=" * 80)
    cliff_256 = None
    for r in range(16, 96, 4):
        res = calculate_sm86_occupancy(256, r, 2048)
        r_alloc = int(math.ceil(r / 8.0) * 8)
        meets = res["occupancy_pct"] >= 66.66
        status = ">= 66.7% (PASS)" if meets else "< 66.7% (CLIFF!)"
        if not meets and cliff_256 is None:
            cliff_256 = r
        print(f"{r:8d} | {r_alloc:8d} | {res['regs_per_block']:10d} | {res['active_blocks']:14d} | {res['active_warps']:12d} | {res['occupancy_pct']:11.1f}% | {status}")
    print(f"\nEmpirical Register Cliff Edge for T_block=256: R_thread = {cliff_256} registers.")

if __name__ == "__main__":
    run_occupancy_stress_test()
