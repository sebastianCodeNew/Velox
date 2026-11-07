#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(c) do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define TOLERANCE 1e-4f
#define TRIALS 50

// Strided Uncoalesced Kernel (Velox v0.1 baseline: 1 thread per row)
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

int run_test_dimension(int M, int K, const char* label) {
    size_t x_bytes = K * sizeof(float);
    size_t w_bytes = (size_t)M * K * sizeof(float);
    size_t dst_bytes = M * sizeof(float);

    float* h_x = (float*)malloc(x_bytes);
    float* h_w = (float*)malloc(w_bytes);
    float* h_dst_strided = (float*)malloc(dst_bytes);
    float* h_dst_warp = (float*)malloc(dst_bytes);
    double* h_dst_ref = (double*)malloc(M * sizeof(double));

    for (int i = 0; i < K; ++i) h_x[i] = sinf((float)i * 0.1f);
    for (size_t i = 0; i < (size_t)M * K; ++i) h_w[i] = cosf((float)(i % 17) * 0.1f) * 0.02f;

    // Double-precision CPU Oracle
    for (int r = 0; r < M; ++r) {
        double acc = 0.0;
        for (int c = 0; c < K; ++c) {
            acc += (double)h_x[c] * (double)h_w[(size_t)r * K + c];
        }
        h_dst_ref[r] = acc;
    }

    float *d_x, *d_w, *d_dst_strided, *d_dst_warp;
    CHECK_CUDA(cudaMalloc(&d_x, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_w, w_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst_strided, dst_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst_warp, dst_bytes));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int block_dim = 256;
    int grid_strided = (M + block_dim - 1) / block_dim;
    int grid_warp = ((size_t)M * 32 + block_dim - 1) / block_dim;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        gemv_strided_uncoalesced<<<grid_strided, block_dim, 0, stream>>>(d_dst_strided, d_x, d_w, M, K);
        gemv_warp_float4_shfl<<<grid_warp, block_dim, 0, stream>>>(d_dst_warp, d_x, d_w, M, K);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Profile Strided Baseline
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        gemv_strided_uncoalesced<<<grid_strided, block_dim, 0, stream>>>(d_dst_strided, d_x, d_w, M, K);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_strided = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_strided, start, stop));
    ms_strided /= (float)TRIALS;

    // Profile Optimized Warp float4 Kernel
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        gemv_warp_float4_shfl<<<grid_warp, block_dim, 0, stream>>>(d_dst_warp, d_x, d_w, M, K);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_warp = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_warp, start, stop));
    ms_warp /= (float)TRIALS;

    CHECK_CUDA(cudaMemcpy(h_dst_strided, d_dst_strided, dst_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dst_warp, d_dst_warp, dst_bytes, cudaMemcpyDeviceToHost));

    // Parity verification against CPU Oracle
    double max_err_strided = 0.0;
    double max_err_warp = 0.0;
    for (int r = 0; r < M; ++r) {
        double err_s = fabs((double)h_dst_strided[r] - h_dst_ref[r]);
        double err_w = fabs((double)h_dst_warp[r] - h_dst_ref[r]);
        if (err_s > max_err_strided) max_err_strided = err_s;
        if (err_w > max_err_warp) max_err_warp = err_w;
    }

    double total_bytes = (double)(w_bytes + x_bytes + dst_bytes);
    double bw_strided = total_bytes / (ms_strided * 1e6);
    double bw_warp = total_bytes / (ms_warp * 1e6);
    double speedup = ms_strided / ms_warp;
    double roofline_pct = (bw_warp / 192.0) * 100.0;

    printf("================================================================================\n");
    printf("TEST 1 [%s]: M=%d, K=%d (Weights: %.2f MB)\n", label, M, K, (double)w_bytes / 1e6);
    printf("Strided Uncoalesced Latency : %.4f ms (Bandwidth: %6.2f GB/s, %5.2f%% Roofline)\n",
           ms_strided, bw_strided, (bw_strided / 192.0) * 100.0);
    printf("Warp float4 Shuffle Latency : %.4f ms (Bandwidth: %6.2f GB/s, %5.2f%% Roofline)\n",
           ms_warp, bw_warp, roofline_pct);
    printf("Warp-GEMV Speedup           : %.2fx over strided baseline\n", speedup);
    printf("Max Error vs Double Oracle  : Strided=%.2e, Warp=%.2e (Tol=%.2e)\n",
           max_err_strided, max_err_warp, TOLERANCE);
    bool pass = (max_err_warp <= TOLERANCE);
    printf("Parity Status               : %s\n", pass ? "PASSED" : "FAILED");
    printf("JSON_RECORD: {\"label\": \"%s\", \"M\": %d, \"K\": %d, \"strided_ms\": %.4f, \"warp_ms\": %.4f, \"speedup\": %.2f, \"bandwidth_gbps\": %.2f, \"roofline_pct\": %.2f, \"max_err\": %.2e, \"pass\": %s}\n",
           label, M, K, ms_strided, ms_warp, speedup, bw_warp, roofline_pct, max_err_warp, pass ? "true" : "false");
    printf("================================================================================\n");

    free(h_x); free(h_w); free(h_dst_strided); free(h_dst_warp); free(h_dst_ref);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_dst_strided); cudaFree(d_dst_warp);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return pass ? 0 : 1;
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Target Device: %s (Compute %d.%d, %d SMs, %.2f GB/s Peak)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, 192.0);

    int res1 = run_test_dimension(1536, 576, "MLP_INTERMEDIATE");
    int res2 = run_test_dimension(49152, 576, "VOCAB_LM_HEAD");

    if (res1 == 0 && res2 == 0) {
        printf("\n>>> STEP 11B TEST 1 ALL DIMENSIONS PASSED 100%% <<<\n");
        return 0;
    } else {
        printf("\n>>> STEP 11B TEST 1 VERIFICATION FAILED <<<\n");
        return 1;
    }
}
