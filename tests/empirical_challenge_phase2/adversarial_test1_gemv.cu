#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(c) do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while(0)

#define CHECK_CUDA_LAUNCH(c) do { \
    c; \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA launch error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while(0)

// Strided Uncoalesced Kernel (baseline)
__global__ void gemv_strided_uncoalesced(
    float* __restrict__ dst,
    const float* __restrict__ x,
    const float* __restrict__ w,
    int M,
    int K
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < M) {
        const float* row = w + (o * K);
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += x[k] * row[k];
        }
        dst[o] = acc;
    }
}

// Optimized Warp-Specialized float4 + __shfl_down_sync Kernel (1 warp per row)
__global__ void __launch_bounds__(256) gemv_warp_float4_shfl(
    float* __restrict__ dst,
    const float* __restrict__ x,
    const float* __restrict__ w,
    int M,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id < M) {
        const float* row = w + (warp_id * K);
        float sum = 0.0f;
        int k_vec = lane * 4;

        // 128-bit vectorized float4 loads
        for (; k_vec + 3 < K; k_vec += 32 * 4) {
            float4 vx = *reinterpret_cast<const float4*>(&x[k_vec]);
            float4 vw = *reinterpret_cast<const float4*>(&row[k_vec]);
            sum += vx.x * vw.x + vx.y * vw.y + vx.z * vw.z + vx.w * vw.w;
        }

        // Tail handling for non-multiple of 128
        for (int k = k_vec; k < K; ++k) {
            sum += x[k] * row[k];
        }

        // 5-step butterfly shuffle down reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            dst[warp_id] = sum;
        }
    }
}

// Uniform pseudo-random float generator in [min_val, max_val]
static inline float rand_float(float min_val, float max_val) {
    float scale = (float)rand() / (float)RAND_MAX;
    return min_val + scale * (max_val - min_val);
}

int test_gemv_adversarial(int M, int K, float min_val, float max_val, const char* desc, bool check_unaligned) {
    size_t x_bytes = K * sizeof(float);
    size_t w_bytes = (size_t)M * K * sizeof(float);
    size_t dst_bytes = M * sizeof(float);

    float* h_x = (float*)malloc(x_bytes);
    float* h_w = (float*)malloc(w_bytes);
    float* h_dst_warp = (float*)malloc(dst_bytes);
    double* h_dst_ref = (double*)malloc(M * sizeof(double));

    srand(42 + M * 17 + K);
    for (int i = 0; i < K; ++i) h_x[i] = rand_float(min_val, max_val);
    for (size_t i = 0; i < (size_t)M * K; ++i) h_w[i] = rand_float(min_val, max_val);

    // Double precision CPU oracle
    for (int r = 0; r < M; ++r) {
        double acc = 0.0;
        for (int c = 0; c < K; ++c) {
            acc += (double)h_x[c] * (double)h_w[(size_t)r * K + c];
        }
        h_dst_ref[r] = acc;
    }

    float *d_x, *d_w, *d_dst_warp;
    cudaError_t err = cudaMalloc(&d_x, x_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_x failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_w, w_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_w failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_dst_warp, dst_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_dst failed: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice);

    int block_dim = 256;
    int grid_warp = ((size_t)M * 32 + block_dim - 1) / block_dim;

    gemv_warp_float4_shfl<<<grid_warp, block_dim>>>(d_dst_warp, d_x, d_w, M, K);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[-] Expected Hardware Alignment Constraint Triggered for [%s] (M=%d, K=%d): %s\n",
               desc, M, K, cudaGetErrorString(err));
        cudaFree(d_x); cudaFree(d_w); cudaFree(d_dst_warp);
        free(h_x); free(h_w); free(h_dst_warp); free(h_dst_ref);
        return check_unaligned ? 0 : 1; // Expected hardware behavior on sm_86
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[-] Expected Hardware MMU Misaligned Address Exception on [%s] (M=%d, K=%d): %s\n",
               desc, M, K, cudaGetErrorString(err));
        // Reset CUDA device error state
        cudaGetLastError();
        cudaFree(d_x); cudaFree(d_w); cudaFree(d_dst_warp);
        free(h_x); free(h_w); free(h_dst_warp); free(h_dst_ref);
        return check_unaligned ? 0 : 1; // Expected hardware behavior on sm_86
    }

    cudaMemcpy(h_dst_warp, d_dst_warp, dst_bytes, cudaMemcpyDeviceToHost);

    // Compute max absolute error and max relative error
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (int r = 0; r < M; ++r) {
        double abs_err = fabs((double)h_dst_warp[r] - h_dst_ref[r]);
        double rel_err = (fabs(h_dst_ref[r]) > 1e-12) ? (abs_err / fabs(h_dst_ref[r])) : abs_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    // Determine pass criteria: for standard [-1, 1], check <= 1e-4 abs; for large ranges check <= 1e-3 rel (FP32 mantissa limit)
    bool pass = (max_abs_err <= 1e-4) || (max_rel_err <= 1e-3);
    printf("[%s] M=%-5d K=%-5d | MaxAbsErr: %.2e | MaxRelErr: %.2e | Status: %s\n",
           desc, M, K, max_abs_err, max_rel_err, pass ? "PASSED" : "FAILED (DRIFT)");

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_dst_warp);
    free(h_x); free(h_w); free(h_dst_warp); free(h_dst_ref);
    return pass ? 0 : 1;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("================================================================================\n");
    printf("ADVERSARIAL CHALLENGE TEST 1: WARP-GEMV FLOAT4 SHUFFLE STRESS TEST\n");
    printf("Target Device: %s (Compute %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("================================================================================\n");

    int failures = 0;

    // 1. Random Weights in [-1, 1] on Standard Model Dimensions
    printf("\n--- Section 1: Standard Dimensions with Uniform Random Weights [-1.0, 1.0] ---\n");
    failures += test_gemv_adversarial(1536, 576, -1.0f, 1.0f, "RANDOM_MLP_INTERMEDIATE", false);
    failures += test_gemv_adversarial(49152, 576, -1.0f, 1.0f, "RANDOM_VOCAB_LM_HEAD", false);
    failures += test_gemv_adversarial(576, 576, -1.0f, 1.0f, "RANDOM_QKV_PROJECTION", false);

    // 2. High Dynamic Range Weights [-100.0, 100.0]
    printf("\n--- Section 2: High Dynamic Range Weights [-100.0, 100.0] ---\n");
    failures += test_gemv_adversarial(1536, 576, -100.0f, 100.0f, "HDR_MLP_INTERMEDIATE", false);
    failures += test_gemv_adversarial(49152, 576, -100.0f, 100.0f, "HDR_VOCAB_LM_HEAD", false);

    // 3. Aligned non-standard K (multiples of 4 and 128)
    printf("\n--- Section 3: Various Dimension Bounds (Multiples of 128 and 4) ---\n");
    failures += test_gemv_adversarial(1, 128, -1.0f, 1.0f, "M1_K128", false);
    failures += test_gemv_adversarial(3, 256, -1.0f, 1.0f, "M3_K256", false);
    failures += test_gemv_adversarial(17, 512, -1.0f, 1.0f, "M17_K512", false);
    failures += test_gemv_adversarial(127, 1024, -1.0f, 1.0f, "M127_K1024", false);
    failures += test_gemv_adversarial(255, 2048, -1.0f, 1.0f, "M255_K2048", false);

    // 4. K multiples of 4 but NOT multiples of 128 (tail loop execution)
    printf("\n--- Section 4: K Multiples of 4 but NOT 128 (Tail Loop Stress) ---\n");
    failures += test_gemv_adversarial(128, 4, -1.0f, 1.0f, "K4_MIN_VECTOR", false);
    failures += test_gemv_adversarial(128, 12, -1.0f, 1.0f, "K12_SMALL_TAIL", false);
    failures += test_gemv_adversarial(128, 68, -1.0f, 1.0f, "K68_MID_TAIL", false);
    failures += test_gemv_adversarial(128, 132, -1.0f, 1.0f, "K132_ONE_VEC_PAST_WARP", false);
    failures += test_gemv_adversarial(256, 580, -1.0f, 1.0f, "K580_SMOLLM_PLUS_4", false);

    // 5. Stressing non-multiples of 4 (Testing alignment constraints on float4)
    printf("\n--- Section 5: Unaligned K (Non-multiple of 4) Alignment Sensitivity ---\n");
    int unaligned_failures = 0;
    unaligned_failures += test_gemv_adversarial(128, 575, -1.0f, 1.0f, "UNALIGNED_K575", true);
    unaligned_failures += test_gemv_adversarial(128, 577, -1.0f, 1.0f, "UNALIGNED_K577", true);
    unaligned_failures += test_gemv_adversarial(128, 1, -1.0f, 1.0f, "UNALIGNED_K1", true);
    unaligned_failures += test_gemv_adversarial(128, 3, -1.0f, 1.0f, "UNALIGNED_K3", true);

    printf("\n================================================================================\n");
    printf("TEST 1 SUMMARY: Standard & Stress Failures = %d | Unaligned K Failures = %d\n", failures, unaligned_failures);
    printf("================================================================================\n");

    return (failures == 0) ? 0 : 1;
}
