#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(c) do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define NUM_WEIGHTS (1024 * 1024) // 1,048,576 weights
#define NUM_PACKED  (NUM_WEIGHTS / 8) // 131,072 uint32 words
#define TRIALS 100
#define TOLERANCE 1e-4f

// Inline PTX for Ampere LOP3 instruction: computes (b & c) | a using LUT 0xEA
__device__ __forceinline__ uint32_t ptx_lop3_0xEA(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}

// Naive dequantization baseline: uses 8 shifts, 8 bitwise ANDs, and scalar float casts
__global__ void naive_dequant_kernel(
    float* __restrict__ out,
    const uint32_t* __restrict__ packed_w4,
    float scale,
    float zero_bias,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 8 < N) {
        uint32_t val = packed_w4[idx];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t nib = (val >> (i * 4)) & 0xF;
            out[idx * 8 + i] = ((float)nib - zero_bias) * scale;
        }
    }
}

// SOTA Marlin-style LOP3 dequantization kernel:
// 1. Uses lop3.b32 with LUT 0xEA to fuse nibble masking and FP16 exponent insertion into 1 cycle.
// 2. Uses __hsub2 to subtract the FP16 bias (1024.0) in SIMD half2 registers.
__global__ void marlin_lop3_dequant_kernel(
    float* __restrict__ out,
    const uint32_t* __restrict__ packed_w4,
    float scale,
    float zero_bias,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 8 < N) {
        uint32_t val = packed_w4[idx];

        // FP16 representation of 1024.0 is 0x6400 (sign=0, exp=19-15=4? No: 2^10 = 1024. Exp=25, 0x6400)
        // 0x64006400 holds two halfs with value 1024.0
        const uint32_t fp16_bias = 0x64006400;
        const uint32_t mask_low  = 0x000F000F;

        // Extract pairs of nibbles:
        // Pair 0: nibble 0 (bits 0..3) and nibble 4 (bits 16..19)
        // Pair 1: nibble 1 (bits 4..7) and nibble 5 (bits 20..23)
        // Pair 2: nibble 2 (bits 8..11) and nibble 6 (bits 24..27)
        // Pair 3: nibble 3 (bits 12..15) and nibble 7 (bits 28..31)

        uint32_t v0 = val;
        uint32_t v1 = val >> 4;
        uint32_t v2 = val >> 8;
        uint32_t v3 = val >> 12;

        // LOP3 0xEA computes: (a & b) | c
        // Here a = v, b = mask_low, c = fp16_bias -> (v & mask_low) | fp16_bias
        uint32_t h2_0 = ptx_lop3_0xEA(v0, mask_low, fp16_bias);
        uint32_t h2_1 = ptx_lop3_0xEA(v1, mask_low, fp16_bias);
        uint32_t h2_2 = ptx_lop3_0xEA(v2, mask_low, fp16_bias);
        uint32_t h2_3 = ptx_lop3_0xEA(v3, mask_low, fp16_bias);

        // Subtract 1024.0 in half2 to get exact nibble value in FP16
        half2 sub_bias = *reinterpret_cast<const half2*>(&fp16_bias);

        half2 res0 = __hsub2(*reinterpret_cast<half2*>(&h2_0), sub_bias);
        half2 res1 = __hsub2(*reinterpret_cast<half2*>(&h2_1), sub_bias);
        half2 res2 = __hsub2(*reinterpret_cast<half2*>(&h2_2), sub_bias);
        half2 res3 = __hsub2(*reinterpret_cast<half2*>(&h2_3), sub_bias);

        // Convert to float and apply scale & zero_bias
        float2 f0 = __half22float2(res0);
        float2 f1 = __half22float2(res1);
        float2 f2 = __half22float2(res2);
        float2 f3 = __half22float2(res3);

        out[idx * 8 + 0] = (f0.x - zero_bias) * scale;
        out[idx * 8 + 1] = (f1.x - zero_bias) * scale;
        out[idx * 8 + 2] = (f2.x - zero_bias) * scale;
        out[idx * 8 + 3] = (f3.x - zero_bias) * scale;
        out[idx * 8 + 4] = (f0.y - zero_bias) * scale;
        out[idx * 8 + 5] = (f1.y - zero_bias) * scale;
        out[idx * 8 + 6] = (f2.y - zero_bias) * scale;
        out[idx * 8 + 7] = (f3.y - zero_bias) * scale;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Target Device: %s (Compute %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    size_t packed_bytes = NUM_PACKED * sizeof(uint32_t);
    size_t out_bytes    = NUM_WEIGHTS * sizeof(float);

    uint32_t* h_packed     = (uint32_t*)malloc(packed_bytes);
    float*    h_out_naive  = (float*)malloc(out_bytes);
    float*    h_out_marlin = (float*)malloc(out_bytes);
    double*   h_out_ref    = (double*)malloc(NUM_WEIGHTS * sizeof(double));

    float scale = 0.03125f;
    float zero_bias = 8.0f;

    // Seed packed values with pseudo-random 4-bit nibbles [0..15]
    for (int i = 0; i < NUM_PACKED; ++i) {
        uint32_t word = 0;
        for (int nib = 0; nib < 8; ++nib) {
            uint32_t val = (uint32_t)((i * 8 + nib) % 16);
            word |= (val << (nib * 4));
            h_out_ref[i * 8 + nib] = ((double)val - (double)zero_bias) * (double)scale;
        }
        h_packed[i] = word;
    }

    uint32_t* d_packed;
    float *d_out_naive, *d_out_marlin;
    CHECK_CUDA(cudaMalloc(&d_packed, packed_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_naive, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_marlin, out_bytes));

    CHECK_CUDA(cudaMemcpy(d_packed, h_packed, packed_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int block_dim = 256;
    int grid_dim = (NUM_PACKED + block_dim - 1) / block_dim;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        naive_dequant_kernel<<<grid_dim, block_dim, 0, stream>>>(d_out_naive, d_packed, scale, zero_bias, NUM_WEIGHTS);
        marlin_lop3_dequant_kernel<<<grid_dim, block_dim, 0, stream>>>(d_out_marlin, d_packed, scale, zero_bias, NUM_WEIGHTS);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Profile Naive Shift & Mask Kernel
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        naive_dequant_kernel<<<grid_dim, block_dim, 0, stream>>>(d_out_naive, d_packed, scale, zero_bias, NUM_WEIGHTS);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_naive = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));
    ms_naive /= (float)TRIALS;

    // Profile Marlin LOP3 Kernel
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        marlin_lop3_dequant_kernel<<<grid_dim, block_dim, 0, stream>>>(d_out_marlin, d_packed, scale, zero_bias, NUM_WEIGHTS);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_marlin = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_marlin, start, stop));
    ms_marlin /= (float)TRIALS;

    CHECK_CUDA(cudaMemcpy(h_out_naive, d_out_naive, out_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_marlin, d_out_marlin, out_bytes, cudaMemcpyDeviceToHost));

    // Numerical Parity Check vs Double-Precision Oracle
    double max_err_naive = 0.0;
    double max_err_marlin = 0.0;
    for (int i = 0; i < NUM_WEIGHTS; ++i) {
        double err_n = fabs((double)h_out_naive[i] - h_out_ref[i]);
        double err_m = fabs((double)h_out_marlin[i] - h_out_ref[i]);
        if (err_n > max_err_naive) max_err_naive = err_n;
        if (err_m > max_err_marlin) max_err_marlin = err_m;
    }


    double gb_streamed = (double)(packed_bytes + out_bytes) / 1e9;
    double bw_naive = gb_streamed / (ms_naive * 1e-3);
    double bw_marlin = gb_streamed / (ms_marlin * 1e-3);
    double speedup = ms_naive / ms_marlin;
    bool pass = (max_err_marlin <= TOLERANCE);

    printf("================================================================================\n");
    printf("STEP 11B TEST 3: MARLIN W4A16 LOP3 DEQUANTIZATION VERIFICATION\n");
    printf("================================================================================\n");
    printf("Dequantized Values      : %d elements (Packed 4-bit words: %d)\n", NUM_WEIGHTS, NUM_PACKED);
    printf("Naive Shift-Mask Time   : %7.4f ms (%6.2f GB/s)\n", ms_naive, bw_naive);
    printf("Marlin LOP3 0xEA Time   : %7.4f ms (%6.2f GB/s)\n", ms_marlin, bw_marlin);
    printf("LOP3 Speedup Factor     : %5.2fx over naive shift-mask\n", speedup);
    printf("Max Error vs Oracle     : Naive=%.2e, Marlin=%.2e (Tol=%.2e)\n", max_err_naive, max_err_marlin, TOLERANCE);
    printf("Parity Status           : %s\n", pass ? "PASSED" : "FAILED");
    printf("JSON_RECORD: {\"weights\": %d, \"naive_ms\": %.4f, \"marlin_ms\": %.4f, \"speedup\": %.2f, \"bandwidth_gbps\": %.2f, \"max_err\": %.2e, \"pass\": %s}\n",
           NUM_WEIGHTS, ms_naive, ms_marlin, speedup, bw_marlin, max_err_marlin, pass ? "true" : "false");
    printf("================================================================================\n");

    free(h_packed); free(h_out_naive); free(h_out_marlin); free(h_out_ref);
    cudaFree(d_packed); cudaFree(d_out_naive); cudaFree(d_out_marlin);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (pass) {
        printf("\n>>> STEP 11B TEST 3 PASSED 100%% <<<\n");
        return 0;
    } else {
        printf("\n>>> STEP 11B TEST 3 FAILED <<<\n");
        return 1;
    }
}
