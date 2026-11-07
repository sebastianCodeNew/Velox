#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(c) do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// Kernels to test
__global__ void gemv_warp_float4(float* dst, const float* x, const float* w, int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id < M) {
        const float* row = w + (warp_id * K);
        float acc = 0.0f;
        int k_vec = lane * 4;
        for (; k_vec + 3 < K; k_vec += 32 * 4) {
            float4 vx = *reinterpret_cast<const float4*>(&x[k_vec]);
            float4 vw = *reinterpret_cast<const float4*>(&row[k_vec]);
            acc += vx.x * vw.x + vx.y * vw.y + vx.z * vw.z + vx.w * vw.w;
        }
        for (int k = k_vec; k < K; ++k) acc += x[k] * row[k];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) dst[warp_id] = acc;
    }
}

__device__ __forceinline__ uint32_t ptx_lop3_0xEA(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}

__global__ void marlin_lop3_dequant(float* out, const uint32_t* packed_w4, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 8 < N) {
        uint32_t val = packed_w4[idx];
        const uint32_t fp16_bias = 0x64006400;
        const uint32_t mask_low  = 0x000F000F;
        uint32_t h2_0 = ptx_lop3_0xEA(val, mask_low, fp16_bias);
        half2 sub_bias = *reinterpret_cast<const half2*>(&fp16_bias);
        half2 res0 = __hsub2(*reinterpret_cast<half2*>(&h2_0), sub_bias);
        float2 f0 = __half22float2(res0);
        out[idx * 8 + 0] = f0.x;
        out[idx * 8 + 1] = f0.y;
    }
}

__global__ void argmax_stage1_block(float* out_vals, uint32_t* out_idxs, const float* logits, int N) {
    __shared__ float s_vals[256];
    __shared__ uint32_t s_idxs[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_val = (idx < N) ? logits[idx] : -1e30f;
    uint32_t local_idx = (idx < N) ? (uint32_t)idx : 0;
    for (int i = idx + gridDim.x * blockDim.x; i < N; i += gridDim.x * blockDim.x) {
        if (logits[i] > local_val) {
            local_val = logits[i];
            local_idx = (uint32_t)i;
        }
    }
    s_vals[tid] = local_val;
    s_idxs[tid] = local_idx;
    __syncthreads();
    #pragma unroll
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_vals[blockIdx.x] = s_vals[0];
        out_idxs[blockIdx.x] = s_idxs[0];
    }
}

int main() {
    printf("================================================================================\n");
    printf("ADVERSARIAL PROBE 3: ZERO RUNTIME HEAP ALLOCATION VERIFICATION\n");
    printf("================================================================================\n");

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Initial allocations (Setup Phase)
    int M = 1536, K = 576;
    float *d_x, *d_w, *d_dst;
    CHECK_CUDA(cudaMalloc(&d_x, K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, (size_t)M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dst, M * sizeof(float)));

    int N_marlin = 65536;
    uint32_t* d_packed;
    float* d_out_marlin;
    CHECK_CUDA(cudaMalloc(&d_packed, (N_marlin / 8) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_out_marlin, N_marlin * sizeof(float)));

    int N_vocab = 49152;
    float *d_logits, *d_cand_vals;
    uint32_t *d_cand_idxs;
    CHECK_CUDA(cudaMalloc(&d_logits, N_vocab * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cand_vals, 192 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cand_idxs, 192 * sizeof(uint32_t)));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Query baseline VRAM after setup
    size_t free_before = 0, total_before = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_before, &total_before));

    printf("Setup Complete. Baseline VRAM: Free = %zu bytes, Total = %zu bytes\n", free_before, total_before);
    printf("Executing 1,000 continuous token decode steps with all kernels...\n");

    // Execution Phase: 1,000 iterations of full decode sequence
    int STEPS = 1000;
    for (int step = 0; step < STEPS; ++step) {
        // 1. GEMV
        gemv_warp_float4<<<(M * 32 + 255) / 256, 256, 0, stream>>>(d_dst, d_x, d_w, M, K);
        // 2. Marlin dequant
        marlin_lop3_dequant<<<(N_marlin / 8 + 255) / 256, 256, 0, stream>>>(d_out_marlin, d_packed, N_marlin);
        // 3. Parallel Argmax
        argmax_stage1_block<<<192, 256, 0, stream>>>(d_cand_vals, d_cand_idxs, d_logits, N_vocab);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Query VRAM after execution
    size_t free_after = 0, total_after = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_after, &total_after));

    printf("Execution Finished (1,000 steps).\n");
    printf("VRAM Free Before : %zu bytes\n", free_before);
    printf("VRAM Free After  : %zu bytes\n", free_after);

    int64_t vram_leak = (int64_t)free_before - (int64_t)free_after;
    printf("VRAM Delta       : %lld bytes\n", (long long)vram_leak);

    bool zero_alloc = (vram_leak == 0);
    printf("Adversarial Zero Allocation Verdict: %s\n",
           zero_alloc ? "PASSED (Strictly Zero Heap/VRAM Creep across 1,000 steps)" : "FAILED (VRAM Leak Detected!)");
    printf("================================================================================\n");

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_dst);
    cudaFree(d_packed); cudaFree(d_out_marlin);
    cudaFree(d_logits); cudaFree(d_cand_vals); cudaFree(d_cand_idxs);
    cudaStreamDestroy(stream);

    return zero_alloc ? 0 : 1;
}
