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

#define K_DIM 576
#define L2_CACHE_BYTES (2048 * 1024) // 2 MB L2 cache on RTX 3050 Laptop GPU
#define FLUSH_BUFFER_BYTES (8 * 1024 * 1024) // 8 MB (> 4x L2 cache) to guarantee cache eviction

// Warp float4 GEMV kernel
__global__ void __launch_bounds__(256) gemv_warp_float4(float* dst, const float* x, const float* w, int M, int K) {
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

// Dummy kernel to flush L2 cache by writing to an 8 MB buffer
__global__ void l2_cache_flush_kernel(int* flush_buf, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        flush_buf[idx] += 1;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("L2 Cache Size Reported: %d bytes (%.2f MB)\n", prop.l2CacheSize, (double)prop.l2CacheSize / (1024.0 * 1024.0));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Allocate 8 MB L2 flush buffer
    int* d_flush_buf;
    int flush_elements = FLUSH_BUFFER_BYTES / sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_flush_buf, FLUSH_BUFFER_BYTES));
    CHECK_CUDA(cudaMemset(d_flush_buf, 0, FLUSH_BUFFER_BYTES));

    // -------------------------------------------------------------------------
    // TEST CASE 1: L2 Cache-Resident Matrix (M=128, K=576 -> 288 KiB << 2 MB)
    // -------------------------------------------------------------------------
    int M_small = 128;
    size_t w_small_bytes = (size_t)M_small * K_DIM * sizeof(float);
    float *d_x_small, *d_w_small, *d_dst_small;
    CHECK_CUDA(cudaMalloc(&d_x_small, K_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_small, w_small_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst_small, M_small * sizeof(float)));

    // Warmup in cache
    for (int i = 0; i < 20; ++i) {
        gemv_warp_float4<<<(M_small * 32 + 255) / 256, 256, 0, stream>>>(d_dst_small, d_x_small, d_w_small, M_small, K_DIM);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Timed run: Matrix stays in L2 cache across repeated iterations
    int trials_small = 1000;
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < trials_small; ++i) {
        gemv_warp_float4<<<(M_small * 32 + 255) / 256, 256, 0, stream>>>(d_dst_small, d_x_small, d_w_small, M_small, K_DIM);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_small_cached = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_small_cached, start, stop));
    ms_small_cached /= (float)trials_small;
    double bw_small_cached = (double)w_small_bytes / (ms_small_cached * 1e6);

    // Timed run with L2 cache flushed before every iteration
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < 100; ++i) {
        l2_cache_flush_kernel<<<(flush_elements + 255) / 256, 256, 0, stream>>>(d_flush_buf, flush_elements);
        gemv_warp_float4<<<(M_small * 32 + 255) / 256, 256, 0, stream>>>(d_dst_small, d_x_small, d_w_small, M_small, K_DIM);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    // (Note: includes flush kernel, so we isolate the flush time)
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < 100; ++i) {
        l2_cache_flush_kernel<<<(flush_elements + 255) / 256, 256, 0, stream>>>(d_flush_buf, flush_elements);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_flush_only = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_flush_only, start, stop));
    ms_flush_only /= 100.0f;

    // -------------------------------------------------------------------------
    // TEST CASE 2: Large Vocabulary Matrix (M=49152, K=576 -> 113.25 MB >> 2 MB)
    // -------------------------------------------------------------------------
    int M_large = 49152;
    size_t w_large_bytes = (size_t)M_large * K_DIM * sizeof(float);
    float *d_x_large, *d_w_large, *d_dst_large;
    CHECK_CUDA(cudaMalloc(&d_x_large, K_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_large, w_large_bytes));
    CHECK_CUDA(cudaMalloc(&d_dst_large, M_large * sizeof(float)));

    // Warmup large matrix
    for (int i = 0; i < 5; ++i) {
        gemv_warp_float4<<<(M_large * 32 + 255) / 256, 256, 0, stream>>>(d_dst_large, d_x_large, d_w_large, M_large, K_DIM);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Profile standard repeated execution (as done in Worker 1's benchmark)
    int trials_large = 50;
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < trials_large; ++i) {
        gemv_warp_float4<<<(M_large * 32 + 255) / 256, 256, 0, stream>>>(d_dst_large, d_x_large, d_w_large, M_large, K_DIM);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_large_standard = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_large_standard, start, stop));
    ms_large_standard /= (float)trials_large;
    double bw_large_standard = (double)(w_large_bytes + K_DIM * 4 + M_large * 4) / (ms_large_standard * 1e6);

    // Profile execution with EXPLICIT L2 CACHE FLUSH between every single run
    float total_ms_with_flush = 0.0f;
    for (int i = 0; i < 20; ++i) {
        // Flush L2 completely
        l2_cache_flush_kernel<<<(flush_elements + 255) / 256, 256, 0, stream>>>(d_flush_buf, flush_elements);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // Time the GEMV alone
        CHECK_CUDA(cudaEventRecord(start, stream));
        gemv_warp_float4<<<(M_large * 32 + 255) / 256, 256, 0, stream>>>(d_dst_large, d_x_large, d_w_large, M_large, K_DIM);
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float trial_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&trial_ms, start, stop));
        total_ms_with_flush += trial_ms;
    }
    float ms_large_flushed = total_ms_with_flush / 20.0f;
    double bw_large_flushed = (double)(w_large_bytes + K_DIM * 4 + M_large * 4) / (ms_large_flushed * 1e6);

    printf("\n================================================================================\n");
    printf("ADVERSARIAL PROBE 1: L2 CACHE RESIDENCY VS GDDR6 STREAMING PROOF\n");
    printf("================================================================================\n");
    printf("GPU L2 Cache Capacity   : %d bytes (%.2f MB)\n", prop.l2CacheSize, (double)prop.l2CacheSize / 1e6);
    printf("Working Set (M=49152)   : %.2f MB (%.1fx L2 Cache Capacity)\n",
           (double)w_large_bytes / 1e6,
           (double)w_large_bytes / (double)prop.l2CacheSize);
    printf("Standard Multi-Trial    : %7.4f ms -> Effective Bandwidth: %6.2f GB/s (%.2f%% Peak)\n",
           ms_large_standard, bw_large_standard, (bw_large_standard / 192.0) * 100.0);
    printf("Cold Run (Flushed L2)   : %7.4f ms -> Effective Bandwidth: %6.2f GB/s (%.2f%% Peak)\n",
           ms_large_flushed, bw_large_flushed, (bw_large_flushed / 192.0) * 100.0);

    double delta_pct = fabs(ms_large_flushed - ms_large_standard) / ms_large_standard * 100.0;
    printf("Delta Flushed vs Std    : %.2f%%\n", delta_pct);

    printf("Small Matrix (M=128)    : Size: %.2f KiB (Fits in L2: Yes)\n", (double)w_small_bytes / 1024.0);
    printf("Small Matrix L2 Latency : %7.4f ms -> Apparent L2 Bandwidth: %6.2f GB/s\n",
           ms_small_cached, bw_small_cached);

    bool genuine_gddr6 = (delta_pct < 5.0) && (bw_large_standard > 175.0);
    printf("Adversarial Verdict     : %s\n",
           genuine_gddr6 ? "PROVEN GENUINE GDDR6 STREAMING (Delta < 5%, Bandwidth > 175 GB/s)" : "REJECTED");
    printf("================================================================================\n");

    cudaFree(d_flush_buf);
    cudaFree(d_x_small); cudaFree(d_w_small); cudaFree(d_dst_small);
    cudaFree(d_x_large); cudaFree(d_w_large); cudaFree(d_dst_large);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return genuine_gddr6 ? 0 : 1;
}
