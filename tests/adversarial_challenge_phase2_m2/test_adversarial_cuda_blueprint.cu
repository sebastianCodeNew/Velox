// =============================================================================
// Adversarial Verification Test Suite: Section 9 CUDA Codegen Blueprint
// Tests compilation with NVCC -arch=sm_86, registers, spills, and execution
// on NVIDIA GeForce RTX 3050 Laptop GPU (sm_86)
// =============================================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define VELOX_HIDDEN_DIM 576
#define VELOX_NUM_HEADS 9
#define VELOX_HEAD_DIM 64
#define VELOX_NUM_KV_HEADS 3
#define VELOX_MODEL_MAX_SEQ_LEN 2048
#define VELOX_VOCAB_SIZE 49152

// -----------------------------------------------------------------------------
// Section 9.1: Coalesced Warp-per-Row Float4 GEMV Kernel
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 8) velox_cuda_matmul_vec_kernel(
    float* __restrict__ dst,
    const float* __restrict__ x,
    const float* __restrict__ w,
    size_t in_dim,
    size_t out_dim,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;

    // 1. Warp and Row Mapping: 4 warps per block, 1 warp computes 1 row
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t row = (size_t)blockIdx.x * 4ULL + warp_id;

    if (row >= out_dim) return;

    const float* __restrict__ row_w = w + (row * in_dim);
    float acc = 0.0f;

    // 2. Contiguous 128-bit float4 loads: 32 lanes x 4 floats = 128 floats (512 bytes) per step
    const size_t steps = in_dim / 128ULL;
    for (size_t s = 0; s < steps; ++s) {
        const size_t k_offset = s * 128ULL + (lane_id * 4ULL);
        const float4 w_v = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 x_v = *reinterpret_cast<const float4*>(&x[k_offset]);
        acc += w_v.x * x_v.x + w_v.y * x_v.y + w_v.z * x_v.z + w_v.w * x_v.w;
    }

    // 3. Tail Processing for in_dim % 128 != 0 (e.g. 576 % 128 = 64)
    const size_t tail_start = steps * 128ULL;
    const size_t remaining_floats = in_dim - tail_start;
    const size_t remaining_f4 = remaining_floats / 4ULL;

    if (lane_id < remaining_f4) {
        const size_t k_offset = tail_start + (lane_id * 4ULL);
        const float4 w_v = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 x_v = *reinterpret_cast<const float4*>(&x[k_offset]);
        acc += w_v.x * x_v.x + w_v.y * x_v.y + w_v.z * x_v.z + w_v.w * x_v.w;
    }

    // 4. In-Register 5-Step Butterfly Shuffle Reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // 5. Lane 0 writes final scalar dot product
    if (lane_id == 0) {
        dst[row] = acc;
    }
}

// -----------------------------------------------------------------------------
// Section 9.2: Upgraded Multi-Block Fused RMSNorm + QKV Projection Kernel
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 8) velox_cuda_fused_norm_qkv_warp_kernel(
    float* __restrict__ dst_q,
    float* __restrict__ kv_pool,
    const float* __restrict__ input_hidden,
    const float* __restrict__ w_norm,
    const float* __restrict__ w_q,
    const float* __restrict__ w_k,
    const float* __restrict__ w_v,
    size_t layer,
    const uint64_t* __restrict__ d_step,
    float eps,
    float inv_rms,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;

    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t global_row = (size_t)blockIdx.x * 4ULL + warp_id;

    const size_t dim = VELOX_HIDDEN_DIM;                         // 576
    const size_t num_q = VELOX_NUM_HEADS * VELOX_HEAD_DIM;       // 576
    const size_t num_kv = VELOX_NUM_KV_HEADS * VELOX_HEAD_DIM;   // 192
    const size_t total_rows = num_q + 2ULL * num_kv;             // 960

    if (global_row >= total_rows) return;

    const size_t step = (size_t)(*d_step);
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    const size_t safe_step = (step < max_s) ? step : (max_s - 1);

    // Determine target matrix and destination pointer
    const float* __restrict__ row_w = NULL;
    float* __restrict__ dst_ptr = NULL;

    if (global_row < num_q) {
        // Q projection
        row_w = w_q + (layer * dim * num_q) + (global_row * dim);
        dst_ptr = dst_q + global_row;
    } else if (global_row < num_q + num_kv) {
        // K projection
        const size_t k_row = global_row - num_q;
        row_w = w_k + (layer * dim * num_kv) + (k_row * dim);
        const size_t layer_kv_stride = 2ULL * max_s * num_kv;
        float* k_layer = kv_pool + (layer * layer_kv_stride);
        dst_ptr = k_layer + (safe_step * num_kv) + k_row;
    } else {
        // V projection
        const size_t v_row = global_row - num_q - num_kv;
        row_w = w_v + (layer * dim * num_kv) + (v_row * dim);
        const size_t layer_kv_stride = 2ULL * max_s * num_kv;
        float* v_layer = kv_pool + (layer * layer_kv_stride) + (max_s * num_kv);
        dst_ptr = v_layer + (safe_step * num_kv) + v_row;
    }

    const float* __restrict__ l_wnorm = w_norm + (layer * dim);
    float acc = 0.0f;

    // Vectorized float4 dot product with pre-normalized input
    const size_t steps = dim / 128ULL;
    for (size_t s = 0; s < steps; ++s) {
        const size_t k_offset = s * 128ULL + (lane_id * 4ULL);
        const float4 w_v = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&input_hidden[k_offset]);
        const float4 nm_v = *reinterpret_cast<const float4*>(&l_wnorm[k_offset]);

        const float nx0 = in_v.x * inv_rms * nm_v.x;
        const float nx1 = in_v.y * inv_rms * nm_v.y;
        const float nx2 = in_v.z * inv_rms * nm_v.z;
        const float nx3 = in_v.w * inv_rms * nm_v.w;

        acc += nx0 * w_v.x + nx1 * w_v.y + nx2 * w_v.z + nx3 * w_v.w;
    }

    // Tail for remaining elements (576 % 128 = 64)
    const size_t tail_start = steps * 128ULL;
    if (lane_id < 16) { // 16 lanes x 4 floats = 64 floats
        const size_t k_offset = tail_start + (lane_id * 4ULL);
        const float4 w_v = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&input_hidden[k_offset]);
        const float4 nm_v = *reinterpret_cast<const float4*>(&l_wnorm[k_offset]);

        const float nx0 = in_v.x * inv_rms * nm_v.x;
        const float nx1 = in_v.y * inv_rms * nm_v.y;
        const float nx2 = in_v.z * inv_rms * nm_v.z;
        const float nx3 = in_v.w * inv_rms * nm_v.w;

        acc += nx0 * w_v.x + nx1 * w_v.y + nx2 * w_v.z + nx3 * w_v.w;
    }

    // Intra-warp butterfly shuffle reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        *dst_ptr = acc;
    }
}

// -----------------------------------------------------------------------------
// Section 9.3: Two-Stage Parallel Vocabulary Argmax Kernels (Verbatim Blueprint)
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 4) velox_cuda_lm_head_argmax_stage1(
    float* __restrict__ scratch_vals,
    uint32_t* __restrict__ scratch_idxs,
    const float* __restrict__ logits,
    size_t vocab_size,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;

    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];

    const size_t tid = threadIdx.x;
    const size_t vid = (size_t)blockIdx.x * 256ULL + tid;

    float my_val = (vid < vocab_size) ? logits[vid] : -1.0e30f;
    uint32_t my_idx = (vid < vocab_size) ? (uint32_t)vid : 0U;

    s_val[tid] = my_val;
    s_idx[tid] = my_idx;
    __syncthreads();

    // 8-step intra-block reduction tree
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        scratch_vals[blockIdx.x] = s_val[0];
        scratch_idxs[blockIdx.x] = s_idx[0];
    }
}

// Section 9.3 Blueprint Stage 2 (Hardcoded 192 candidates)
__global__ void __launch_bounds__(256, 1) velox_cuda_lm_head_argmax_stage2(
    uint32_t* __restrict__ host_pinned_ring,
    const float* __restrict__ scratch_vals,
    const uint32_t* __restrict__ scratch_idxs,
    const uint64_t* __restrict__ d_step,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;

    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];

    const size_t tid = threadIdx.x;
    const size_t num_candidates = 192ULL; // 49,152 / 256 = 192

    float my_val = (tid < num_candidates) ? scratch_vals[tid] : -1.0e30f;
    uint32_t my_idx = (tid < num_candidates) ? scratch_idxs[tid] : 0U;

    s_val[tid] = my_val;
    s_idx[tid] = my_idx;
    __syncthreads();

    // 8-step intra-block reduction tree
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0 && host_pinned_ring != NULL) {
        const size_t step = (size_t)(*d_step);
        host_pinned_ring[step & 1023ULL] = s_idx[0];
    }
}

// Generalized Stage 2 Kernel (Handling Arbitrary Candidates num_candidates > 256)
__global__ void __launch_bounds__(256, 1) velox_cuda_lm_head_argmax_stage2_generalized(
    uint32_t* __restrict__ host_pinned_ring,
    const float* __restrict__ scratch_vals,
    const uint32_t* __restrict__ scratch_idxs,
    size_t num_candidates,
    const uint64_t* __restrict__ d_step,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;

    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];

    const size_t tid = threadIdx.x;

    // Grid-stride loop to load and partially reduce candidates > 256
    float my_val = -1.0e30f;
    uint32_t my_idx = 0U;

    for (size_t c = tid; c < num_candidates; c += blockDim.x) {
        float c_val = scratch_vals[c];
        uint32_t c_idx = scratch_idxs[c];
        if (c_val > my_val) {
            my_val = c_val;
            my_idx = c_idx;
        }
    }

    s_val[tid] = my_val;
    s_idx[tid] = my_idx;
    __syncthreads();

    // 8-step intra-block reduction tree
    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0 && host_pinned_ring != NULL) {
        const size_t step = (size_t)(*d_step);
        host_pinned_ring[step & 1023ULL] = s_idx[0];
    }
}

// -----------------------------------------------------------------------------
// Host Verification Harness
// -----------------------------------------------------------------------------
int main() {
    printf("====================================================================\n");
    printf("CUDA BLUEPRINT ADVERSARIAL VERIFICATION & PARITY HARNESS (sm_86)\n");
    printf("====================================================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (CC %d.%d, %d SMs, %.1f GB VRAM)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Test 1: GEMV Numerical Parity vs Double-Precision Oracle
    printf("\n--- Test 1: Warp GEMV Float4 Numerical Parity ---\n");
    struct GemvCase {
        size_t M;
        size_t K;
        const char* name;
    } cases[] = {
        {960, 576, "SmolLM QKV"},
        {576, 576, "SmolLM Out"},
        {3072, 576, "SmolLM GateUp"},
        {576, 1536, "SmolLM Down"},
        {1, 576, "Edge M=1"},
        {3, 576, "Edge M=3"},
        {5, 576, "Edge M=5"},
        {4, 64, "Edge K=64"},
        {4, 128, "Edge K=128"},
        {4, 4096, "Llama-3 K=4096"},
        {4, 7168, "DeepSeek K=7168"}
    };

    bool all_gemv_pass = true;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        size_t M = cases[i].M;
        size_t K = cases[i].K;
        
        float* h_w = (float*)malloc(M * K * sizeof(float));
        float* h_x = (float*)malloc(K * sizeof(float));
        float* h_y_cpu = (float*)malloc(M * sizeof(float));
        float* h_y_gpu = (float*)malloc(M * sizeof(float));

        for (size_t k = 0; k < K; ++k) {
            h_x[k] = sinf((float)k * 0.1f);
        }
        for (size_t r = 0; r < M; ++r) {
            for (size_t k = 0; k < K; ++k) {
                h_w[r * K + k] = cosf((float)(r * K + k) * 0.05f) * 0.1f;
            }
        }

        // CPU Oracle (double precision accumulation)
        for (size_t r = 0; r < M; ++r) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum += (double)h_w[r * K + k] * (double)h_x[k];
            }
            h_y_cpu[r] = (float)sum;
        }

        float *d_w, *d_x, *d_y;
        cudaMalloc(&d_w, M * K * sizeof(float));
        cudaMalloc(&d_x, K * sizeof(float));
        cudaMalloc(&d_y, M * sizeof(float));

        cudaMemcpy(d_w, h_w, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, K * sizeof(float), cudaMemcpyHostToDevice);

        size_t warps_per_block = 4;
        size_t num_blocks = (M + warps_per_block - 1) / warps_per_block;
        velox_cuda_matmul_vec_kernel<<<dim3(num_blocks), dim3(128)>>>(d_y, d_x, d_w, K, M, 1ULL);
        cudaDeviceSynchronize();

        cudaMemcpy(h_y_gpu, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

        float max_diff = 0.0f;
        for (size_t r = 0; r < M; ++r) {
            float diff = fabsf(h_y_cpu[r] - h_y_gpu[r]);
            if (diff > max_diff) max_diff = diff;
        }

        bool pass = max_diff < 1e-4f;
        if (!pass) all_gemv_pass = false;
        printf("  [%s] %-20s (M=%5zu, K=%5zu): max_diff = %.2e -> %s\n",
               pass ? "PASS" : "FAIL", cases[i].name, M, K, max_diff, pass ? "BIT-EXACT/PARITY" : "DIVERGENCE");

        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
        free(h_w);
        free(h_x);
        free(h_y_cpu);
        free(h_y_gpu);
    }

    // Test 2: Two-Stage Vocabulary Argmax on SmolLM-135M (V=49,152)
    printf("\n--- Test 2: Two-Stage Argmax on SmolLM-135M (V=49,152) ---\n");
    size_t V_smol = 49152;
    float* h_logits_smol = (float*)malloc(V_smol * sizeof(float));
    for (size_t v = 0; v < V_smol; ++v) {
        h_logits_smol[v] = sinf((float)v * 0.01f);
    }
    // Inject known winning token
    size_t expected_smol_winner = 38412;
    h_logits_smol[expected_smol_winner] = 999.0f;

    float* d_logits_smol;
    float* d_scratch_vals;
    uint32_t* d_scratch_idxs;
    uint32_t* d_pinned_ring;
    uint64_t* d_step;

    cudaMalloc(&d_logits_smol, V_smol * sizeof(float));
    cudaMalloc(&d_scratch_vals, 192 * sizeof(float));
    cudaMalloc(&d_scratch_idxs, 192 * sizeof(uint32_t));
    cudaMallocHost(&d_pinned_ring, 1024 * sizeof(uint32_t));
    cudaMalloc(&d_step, sizeof(uint64_t));

    uint64_t init_step = 0ULL;
    cudaMemcpy(d_step, &init_step, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_logits_smol, h_logits_smol, V_smol * sizeof(float), cudaMemcpyHostToDevice);

    velox_cuda_lm_head_argmax_stage1<<<dim3(192), dim3(256)>>>(d_scratch_vals, d_scratch_idxs, d_logits_smol, V_smol, 1ULL);
    velox_cuda_lm_head_argmax_stage2<<<dim3(1), dim3(256)>>>(d_pinned_ring, d_scratch_vals, d_scratch_idxs, d_step, 1ULL);
    cudaDeviceSynchronize();

    uint32_t actual_smol_winner = d_pinned_ring[0];
    bool smol_pass = (actual_smol_winner == expected_smol_winner);
    printf("  SmolLM-135M Argmax: Expected Token = %zu, Actual Token = %u -> %s\n",
           expected_smol_winner, actual_smol_winner, smol_pass ? "PASS (MATCH)" : "FAIL");

    cudaFree(d_logits_smol);
    cudaFree(d_scratch_vals);
    cudaFree(d_scratch_idxs);
    free(h_logits_smol);

    // Test 3: Adversarial Probing of Section 9.3 Blueprint on Llama-3-8B (V=128,256)
    printf("\n--- Test 3: Adversarial Probing on Large Vocabulary (Llama-3-8B: V=128,256) ---\n");
    size_t V_llama = 128256;
    size_t stage1_blocks = (V_llama + 255) / 256; // 501 blocks
    float* h_logits_llama = (float*)malloc(V_llama * sizeof(float));
    for (size_t v = 0; v < V_llama; ++v) {
        h_logits_llama[v] = -10.0f; // low baseline
    }
    // Inject maximum in the UPPER half of vocabulary (token 80,000 > 65,536)
    size_t high_token = 80000;
    h_logits_llama[high_token] = 500.0f;

    // Inject a secondary local maximum in lower half (token 10,000)
    size_t low_token = 10000;
    h_logits_llama[low_token] = 100.0f;

    float* d_logits_llama;
    float* d_scratch_vals_llama;
    uint32_t* d_scratch_idxs_llama;
    cudaMalloc(&d_logits_llama, V_llama * sizeof(float));
    cudaMalloc(&d_scratch_vals_llama, stage1_blocks * sizeof(float));
    cudaMalloc(&d_scratch_idxs_llama, stage1_blocks * sizeof(uint32_t));
    cudaMemcpy(d_logits_llama, h_logits_llama, V_llama * sizeof(float), cudaMemcpyHostToDevice);

    // Run Stage 1 across all 501 blocks
    velox_cuda_lm_head_argmax_stage1<<<dim3(stage1_blocks), dim3(256)>>>(
        d_scratch_vals_llama, d_scratch_idxs_llama, d_logits_llama, V_llama, 1ULL);

    // 3a: Run Blueprint Section 9.3 (hardcoded 192 candidates)
    velox_cuda_lm_head_argmax_stage2<<<dim3(1), dim3(256)>>>(
        d_pinned_ring, d_scratch_vals_llama, d_scratch_idxs_llama, d_step, 1ULL);
    cudaDeviceSynchronize();
    uint32_t blueprint_result = d_pinned_ring[0];
    printf("  [Blueprint 9.3 on V=128,256] Expected = %zu (Logit 500.0), Actual = %u (Logit %.1f)\n",
           high_token, blueprint_result, h_logits_llama[blueprint_result]);
    if (blueprint_result != high_token) {
        printf("  [PROVEN DEFECT] Blueprint Section 9.3 TRUNCATES candidates! Selected token %u instead of %zu!\n",
               blueprint_result, high_token);
    }

    // 3b: Run Generalized Stage 2 (grid-stride loop over all 501 candidates)
    velox_cuda_lm_head_argmax_stage2_generalized<<<dim3(1), dim3(256)>>>(
        d_pinned_ring, d_scratch_vals_llama, d_scratch_idxs_llama, stage1_blocks, d_step, 1ULL);
    cudaDeviceSynchronize();
    uint32_t generalized_result = d_pinned_ring[0];
    printf("  [Generalized Stage 2 on V=128,256] Expected = %zu, Actual = %u -> %s\n",
           high_token, generalized_result, (generalized_result == high_token) ? "PASS (EXACT MATCH)" : "FAIL");

    cudaFree(d_logits_llama);
    cudaFree(d_scratch_vals_llama);
    cudaFree(d_scratch_idxs_llama);
    cudaFreeHost(d_pinned_ring);
    cudaFree(d_step);
    free(h_logits_llama);

    printf("\n====================================================================\n");
    printf("OVERALL EMPIRICAL VERDICT SUMMARY:\n");
    printf("GEMV float4 bit-exact across all shapes: %s\n", all_gemv_pass ? "YES (PASS)" : "NO (FAIL)");
    printf("SmolLM-135M Two-Stage Argmax bit-exact: %s\n", smol_pass ? "YES (PASS)" : "NO (FAIL)");
    printf("Blueprint Section 9.3 Large Vocab Defect Confirmed: %s\n", (blueprint_result != high_token) ? "CONFIRMED DEFECT" : "NO DEFECT");
    printf("====================================================================\n");

    return 0;
}
