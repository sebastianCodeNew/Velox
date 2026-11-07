#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(c) do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define M_DIM 49152
#define K_DIM 576
#define TRIALS 50

// Config A: Strided Uncoalesced GEMV (Velox v0.1 baseline: 1 thread per row)
__global__ void gemv_strided_uncoalesced(float* dst, const float* x, const float* w, int M, int K) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < M) {
        const float* row = w + (o * K);
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += x[k] * row[k];
        dst[o] = acc;
    }
}

// Config B: Coalesced Warp-per-Row Scalar GEMV
__global__ void gemv_coalesced_warp_scalar(float* dst, const float* x, const float* w, int M, int K) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id < M) {
        const float* row = w + (warp_id * K);
        float acc = 0.0f;
        for (int k = lane; k < K; k += 32) {
            acc += x[k] * row[k];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) dst[warp_id] = acc;
    }
}

// Config C: Coalesced Warp-per-Row Vectorized float4 GEMV (128-bit loads)
__global__ void __launch_bounds__(256) gemv_coalesced_warp_float4(float* dst, const float* x, const float* w, int M, int K) {
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
        for (int k = k_vec; k < K; ++k) {
            acc += x[k] * row[k];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) dst[warp_id] = acc;
    }
}

int main() {
    float *d_x, *d_w, *d_dst_a, *d_dst_b, *d_dst_c;
    size_t x_bytes = K_DIM * sizeof(float);
    size_t w_bytes = (size_t)M_DIM * K_DIM * sizeof(float);
    size_t dst_bytes = M_DIM * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_x, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_w, w_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst_a, dst_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst_b, dst_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst_c, dst_bytes));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    gemv_strided_uncoalesced<<<(M_DIM + 255) / 256, 256, 0, stream>>>(d_dst_a, d_x, d_w, M_DIM, K_DIM);
    gemv_coalesced_warp_scalar<<<(M_DIM * 32 + 255) / 256, 256, 0, stream>>>(d_dst_b, d_x, d_w, M_DIM, K_DIM);
    gemv_coalesced_warp_float4<<<(M_DIM * 32 + 255) / 256, 256, 0, stream>>>(d_dst_c, d_x, d_w, M_DIM, K_DIM);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Time Config A
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        gemv_strided_uncoalesced<<<(M_DIM + 255) / 256, 256, 0, stream>>>(d_dst_a, d_x, d_w, M_DIM, K_DIM);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_a = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_a, start, stop));
    ms_a /= (float)TRIALS;

    // Time Config B
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        gemv_coalesced_warp_scalar<<<(M_DIM * 32 + 255) / 256, 256, 0, stream>>>(d_dst_b, d_x, d_w, M_DIM, K_DIM);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_b = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_b, start, stop));
    ms_b /= (float)TRIALS;

    // Time Config C
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        gemv_coalesced_warp_float4<<<(M_DIM * 32 + 255) / 256, 256, 0, stream>>>(d_dst_c, d_x, d_w, M_DIM, K_DIM);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_c = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_c, start, stop));
    ms_c /= (float)TRIALS;

    double total_bytes = (double)(w_bytes + x_bytes + dst_bytes);
    double bw_a = total_bytes / (ms_a * 1e6);
    double bw_b = total_bytes / (ms_b * 1e6);
    double bw_c = total_bytes / (ms_c * 1e6);

    printf("================================================================================\n");
    printf("STEP 11 BENCHMARK 2: STRIDED UNCOALESCED VS FLOAT4 COALESCED GEMV\n");
    printf("================================================================================\n");
    printf("Matrix Dimensions       : M=%d, K=%d (Weights: %.2f MB)\n", M_DIM, K_DIM, (double)w_bytes / 1e6);
    printf("Strided Uncoalesced     : %7.4f ms (Bandwidth: %6.2f GB/s, %5.2f%% Roofline)\n", ms_a, bw_a, (bw_a / 192.0) * 100.0);
    printf("Coalesced Warp Scalar   : %7.4f ms (Bandwidth: %6.2f GB/s, %5.2f%% Roofline)\n", ms_b, bw_b, (bw_b / 192.0) * 100.0);
    printf("Coalesced Warp float4   : %7.4f ms (Bandwidth: %6.2f GB/s, %5.2f%% Roofline)\n", ms_c, bw_c, (bw_c / 192.0) * 100.0);
    printf("Vectorized Speedup      : %5.2fx over strided baseline\n", ms_a / ms_c);
    printf("JSON_OUTPUT: {\"benchmark\": \"bench2_gemv_coalescing\", \"strided_ms\": %.4f, \"scalar_ms\": %.4f, \"float4_ms\": %.4f, \"speedup\": %.2f, \"float4_bw_gbps\": %.2f}\n",
           ms_a, ms_b, ms_c, ms_a / ms_c, bw_c);
    printf("================================================================================\n");

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_dst_a); cudaFree(d_dst_b); cudaFree(d_dst_c);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
