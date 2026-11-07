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
        return 1; \
    } \
} while(0)

// Inline PTX for Ampere LOP3 instruction: computes (b & c) | a using LUT 0xEA
__device__ __forceinline__ uint32_t ptx_lop3_0xEA(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(res) : "r"(a), "r"(b), "r"(c));
    return res;
}

// SOTA Marlin-style LOP3 dequantization kernel
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

        const uint32_t fp16_bias = 0x64006400; // Two FP16 numbers equal to 1024.0
        const uint32_t mask_low  = 0x000F000F;

        uint32_t v0 = val;
        uint32_t v1 = val >> 4;
        uint32_t v2 = val >> 8;
        uint32_t v3 = val >> 12;

        uint32_t h2_0 = ptx_lop3_0xEA(v0, mask_low, fp16_bias);
        uint32_t h2_1 = ptx_lop3_0xEA(v1, mask_low, fp16_bias);
        uint32_t h2_2 = ptx_lop3_0xEA(v2, mask_low, fp16_bias);
        uint32_t h2_3 = ptx_lop3_0xEA(v3, mask_low, fp16_bias);

        half2 sub_bias = *reinterpret_cast<const half2*>(&fp16_bias);

        half2 res0 = __hsub2(*reinterpret_cast<half2*>(&h2_0), sub_bias);
        half2 res1 = __hsub2(*reinterpret_cast<half2*>(&h2_1), sub_bias);
        half2 res2 = __hsub2(*reinterpret_cast<half2*>(&h2_2), sub_bias);
        half2 res3 = __hsub2(*reinterpret_cast<half2*>(&h2_3), sub_bias);

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
    cudaGetDeviceProperties(&prop, 0);
    printf("================================================================================\n");
    printf("ADVERSARIAL CHALLENGE TEST 3: MARLIN LOP3 DEQUANTIZATION BIT-EXACTNESS AUDIT\n");
    printf("Target Device: %s (Compute %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("================================================================================\n");

    float scale = 0.03125f;
    float zero_bias = 8.0f;

    // --- Test Section 1: Exhaustive Sweep of All 16 Nibbles (0x0..0xF) in All 8 Positions ---
    printf("\n--- Section 1: Exhaustive Sweep of All 16 Nibbles (0x0..0xF) across 8 Positions ---\n");
    // We create a test vector of 16 * 8 = 128 words, each word isolates one nibble v at position p
    int num_isolated = 16 * 8;
    int num_isolated_elements = num_isolated * 8;
    uint32_t* h_isolated_packed = (uint32_t*)malloc(num_isolated * sizeof(uint32_t));
    float* h_isolated_out = (float*)malloc(num_isolated_elements * sizeof(float));

    for (int p = 0; p < 8; ++p) {
        for (int v = 0; v < 16; ++v) {
            int w_idx = p * 16 + v;
            h_isolated_packed[w_idx] = ((uint32_t)v << (p * 4));
        }
    }

    uint32_t* d_isolated_packed;
    float* d_isolated_out;
    CHECK_CUDA(cudaMalloc(&d_isolated_packed, num_isolated * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_isolated_out, num_isolated_elements * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_isolated_packed, h_isolated_packed, num_isolated * sizeof(uint32_t), cudaMemcpyHostToDevice));

    marlin_lop3_dequant_kernel<<<1, 256>>>(d_isolated_out, d_isolated_packed, scale, zero_bias, num_isolated_elements);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_isolated_out, d_isolated_out, num_isolated_elements * sizeof(float), cudaMemcpyDeviceToHost));

    int nibble_errors = 0;
    for (int p = 0; p < 8; ++p) {
        for (int v = 0; v < 16; ++v) {
            int w_idx = p * 16 + v;
            for (int k = 0; k < 8; ++k) {
                float expected_nib = (k == p) ? (float)v : 0.0f;
                float expected_val = (expected_nib - zero_bias) * scale;
                float actual_val = h_isolated_out[w_idx * 8 + k];
                if (actual_val != expected_val) {
                    printf("[-] ERROR at pos %d, nibble 0x%X, element %d: Expected %.6f, got %.6f\n",
                           p, v, k, expected_val, actual_val);
                    nibble_errors++;
                }
            }
        }
    }
    printf("Exhaustive 16 Nibble x 8 Position Status: %s (Errors: %d / 1024)\n",
           nibble_errors == 0 ? "PASSED (BIT-EXACT 0.0 ERROR)" : "FAILED", nibble_errors);

    cudaFree(d_isolated_packed);
    cudaFree(d_isolated_out);
    free(h_isolated_packed);
    free(h_isolated_out);

    // --- Test Section 2: All 256 Half2 Pair Combinations (Nibble A in Low, Nibble B in High) ---
    printf("\n--- Section 2: All 256 Half2 Pair Combinations (Cross-Lane Interference Check) ---\n");
    int num_pairs = 256;
    int num_pair_elements = num_pairs * 8;
    uint32_t* h_pair_packed = (uint32_t*)malloc(num_pairs * sizeof(uint32_t));
    float* h_pair_out = (float*)malloc(num_pair_elements * sizeof(float));

    for (int a = 0; a < 16; ++a) {
        for (int b = 0; b < 16; ++b) {
            int idx = a * 16 + b;
            // Pack into nibble 0 (bits 0..3) and nibble 4 (bits 16..19) -> h2_0
            h_pair_packed[idx] = ((uint32_t)a) | ((uint32_t)b << 16);
        }
    }

    uint32_t* d_pair_packed;
    float* d_pair_out;
    CHECK_CUDA(cudaMalloc(&d_pair_packed, num_pairs * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_pair_out, num_pair_elements * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_pair_packed, h_pair_packed, num_pairs * sizeof(uint32_t), cudaMemcpyHostToDevice));

    marlin_lop3_dequant_kernel<<<1, 256>>>(d_pair_out, d_pair_packed, scale, zero_bias, num_pair_elements);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_pair_out, d_pair_out, num_pair_elements * sizeof(float), cudaMemcpyDeviceToHost));

    int pair_errors = 0;
    for (int a = 0; a < 16; ++a) {
        for (int b = 0; b < 16; ++b) {
            int idx = a * 16 + b;
            float exp_a = ((float)a - zero_bias) * scale;
            float exp_b = ((float)b - zero_bias) * scale;
            float act_a = h_pair_out[idx * 8 + 0]; // nibble 0
            float act_b = h_pair_out[idx * 8 + 4]; // nibble 4
            if (act_a != exp_a || act_b != exp_b) {
                printf("[-] Pair error (0x%X, 0x%X): Act=(%.6f, %.6f) vs Exp=(%.6f, %.6f)\n",
                       a, b, act_a, act_b, exp_a, exp_b);
                pair_errors++;
            }
        }
    }
    printf("256 Half2 Cross-Pair Combinations Status: %s (Errors: %d / 256)\n",
           pair_errors == 0 ? "PASSED (BIT-EXACT 0.0 ERROR)" : "FAILED", pair_errors);

    cudaFree(d_pair_packed);
    cudaFree(d_pair_out);
    free(h_pair_packed);
    free(h_pair_out);

    // --- Test Section 3: Boundary Bit Patterns (0x00000000, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA) ---
    printf("\n--- Section 3: Boundary & Alternating Bit Patterns ---\n");
    uint32_t patterns[] = {
        0x00000000, // all 0x0
        0xFFFFFFFF, // all 0xF
        0x55555555, // all 0x5
        0xAAAAAAAA, // all 0xA
        0x12345678, // sequential
        0x87654321, // reverse
        0x0F0F0F0F, // alternating 0 and F
        0xF0F0F0F0  // alternating F and 0
    };
    int num_patterns = sizeof(patterns) / sizeof(patterns[0]);
    int pattern_elements = num_patterns * 8;
    float* h_patt_out = (float*)malloc(pattern_elements * sizeof(float));

    uint32_t* d_patt_packed;
    float* d_patt_out;
    CHECK_CUDA(cudaMalloc(&d_patt_packed, num_patterns * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_patt_out, pattern_elements * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_patt_packed, patterns, num_patterns * sizeof(uint32_t), cudaMemcpyHostToDevice));

    marlin_lop3_dequant_kernel<<<1, 256>>>(d_patt_out, d_patt_packed, scale, zero_bias, pattern_elements);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_patt_out, d_patt_out, pattern_elements * sizeof(float), cudaMemcpyDeviceToHost));

    int pattern_errors = 0;
    for (int i = 0; i < num_patterns; ++i) {
        uint32_t word = patterns[i];
        for (int nib = 0; nib < 8; ++nib) {
            uint32_t v = (word >> (nib * 4)) & 0xF;
            float exp_val = ((float)v - zero_bias) * scale;
            float act_val = h_patt_out[i * 8 + nib];
            if (act_val != exp_val) {
                printf("[-] Pattern 0x%08X nibble %d (0x%X): Expected %.6f, got %.6f\n",
                       word, nib, v, exp_val, act_val);
                pattern_errors++;
            }
        }
    }
    printf("Boundary & Alternating Patterns Status: %s (Errors: %d / %d)\n",
           pattern_errors == 0 ? "PASSED (BIT-EXACT 0.0 ERROR)" : "FAILED", pattern_errors, pattern_elements);

    cudaFree(d_patt_packed);
    cudaFree(d_patt_out);
    free(h_patt_out);

    // --- Test Section 4: 1,048,576 True Pseudo-Random Packed 32-bit Words ---
    printf("\n--- Section 4: 1,048,576 Large Scale Random Packed Words (8,388,608 Weights) ---\n");
    int big_words = 1048576;
    int big_elements = big_words * 8;
    uint32_t* h_big_packed = (uint32_t*)malloc(big_words * sizeof(uint32_t));
    float* h_big_out = (float*)malloc(big_elements * sizeof(float));

    srand(777);
    for (int i = 0; i < big_words; ++i) {
        uint32_t w = ((uint32_t)rand() & 0xFFFF) | (((uint32_t)rand() & 0xFFFF) << 16);
        h_big_packed[i] = w;
    }

    uint32_t* d_big_packed;
    float* d_big_out;
    CHECK_CUDA(cudaMalloc(&d_big_packed, big_words * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_big_out, big_elements * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_big_packed, h_big_packed, big_words * sizeof(uint32_t), cudaMemcpyHostToDevice));

    int block_dim = 256;
    int grid_dim = (big_words + block_dim - 1) / block_dim;
    marlin_lop3_dequant_kernel<<<grid_dim, block_dim>>>(d_big_out, d_big_packed, scale, zero_bias, big_elements);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_big_out, d_big_out, big_elements * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    int big_mismatches = 0;
    for (int i = 0; i < big_words; ++i) {
        uint32_t w = h_big_packed[i];
        for (int nib = 0; nib < 8; ++nib) {
            uint32_t v = (w >> (nib * 4)) & 0xF;
            float exp_val = ((float)v - zero_bias) * scale;
            float act_val = h_big_out[i * 8 + nib];
            double diff = fabs((double)act_val - (double)exp_val);
            if (diff > max_diff) max_diff = diff;
            if (diff != 0.0) big_mismatches++;
        }
    }
    printf("Large Scale 8.38M Random Weights Status: %s | Max Diff: %.2e | Mismatches: %d\n",
           big_mismatches == 0 ? "PASSED (BIT-EXACT 0.0 ERROR)" : "FAILED", max_diff, big_mismatches);

    cudaFree(d_big_packed);
    cudaFree(d_big_out);
    free(h_big_packed);
    free(h_big_out);

    int total_failures = nibble_errors + pair_errors + pattern_errors + big_mismatches;
    printf("\n================================================================================\n");
    printf("TEST 3 TOTAL AUDIT SUMMARY: Total Failures = %d\n", total_failures);
    printf("================================================================================\n");

    return (total_failures == 0) ? 0 : 1;
}
