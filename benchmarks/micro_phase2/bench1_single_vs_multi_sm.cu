#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <windows.h>

#define CHECK_CUDA(c) do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define M_DIM 1536
#define K_DIM 576
#define TRIALS 100

// Config A: Single-SM loop (Velox v0.1 style, 1 block of 256 threads on SM 0)
__global__ void __launch_bounds__(256) gemv_single_sm(
    float* __restrict__ dst,
    const float* __restrict__ x,
    const float* __restrict__ w,
    int M,
    int K
) {
    if (blockIdx.x > 0) return; // Confines execution to SM 0
    for (int o = threadIdx.x; o < M; o += blockDim.x) {
        const float* row = w + (o * K);
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += x[k] * row[k];
        dst[o] = acc;
    }
}

// Config B: Multi-SM grid (partitioned across blocks, all 16 SMs active)
__global__ void __launch_bounds__(256) gemv_multi_sm(
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
        for (int k = 0; k < K; ++k) acc += x[k] * row[k];
        dst[o] = acc;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (Compute %d.%d, %d SMs, Peak Bandwidth: 192.0 GB/s)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    float *d_x, *d_w, *d_dst_single, *d_dst_multi;
    size_t x_bytes = K_DIM * sizeof(float);
    size_t w_bytes = (size_t)M_DIM * K_DIM * sizeof(float);
    size_t dst_bytes = M_DIM * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_x, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_w, w_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst_single, dst_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst_multi, dst_bytes));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        gemv_single_sm<<<1, 256, 0, stream>>>(d_dst_single, d_x, d_w, M_DIM, K_DIM);
        gemv_multi_sm<<<(M_DIM + 255) / 256, 256, 0, stream>>>(d_dst_multi, d_x, d_w, M_DIM, K_DIM);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Profile Config A (Single-SM)
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        gemv_single_sm<<<1, 256, 0, stream>>>(d_dst_single, d_x, d_w, M_DIM, K_DIM);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_single = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_single, start, stop));
    ms_single /= (float)TRIALS;

    // Profile Config B (Multi-SM)
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        gemv_multi_sm<<<(M_DIM + 255) / 256, 256, 0, stream>>>(d_dst_multi, d_x, d_w, M_DIM, K_DIM);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_multi = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_multi, start, stop));
    ms_multi /= (float)TRIALS;

    double total_bytes = (double)(w_bytes + x_bytes + dst_bytes);
    double single_bw = total_bytes / (ms_single * 1e6);
    double multi_bw  = total_bytes / (ms_multi * 1e6);
    double speedup   = ms_single / ms_multi;

    printf("================================================================================\n");
    printf("STEP 11 BENCHMARK 1: SINGLE-SM VS MULTI-SM GRID PERFORMANCE\n");
    printf("================================================================================\n");
    printf("Matrix Dimensions       : M=%d, K=%d (Bytes: %.2f MB)\n", M_DIM, K_DIM, (double)w_bytes / 1e6);
    printf("Single-SM Latency       : %.4f ms (Bandwidth: %6.2f GB/s, %5.2f%% Roofline)\n",
           ms_single, single_bw, (single_bw / 192.0) * 100.0);
    printf("Multi-SM Latency        : %.4f ms (Bandwidth: %6.2f GB/s, %5.2f%% Roofline)\n",
           ms_multi, multi_bw, (multi_bw / 192.0) * 100.0);
    printf("Multi-SM Speedup        : %.2fx\n", speedup);
    printf("JSON_OUTPUT: {\"benchmark\": \"bench1_single_vs_multi_sm\", \"single_sm_ms\": %.4f, \"multi_sm_ms\": %.4f, \"speedup\": %.2f, \"multi_bw_gbps\": %.2f}\n",
           ms_single, ms_multi, speedup, multi_bw);
    printf("================================================================================\n");

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_dst_single); cudaFree(d_dst_multi);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
