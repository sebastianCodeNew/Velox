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

#define NUM_KERNELS 64
#define TRIALS 100

__global__ void dummy_kernel(float* buf, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) buf[idx] += 1.0f;
}

int main() {
    float* d_buf;
    CHECK_CUDA(cudaMalloc(&d_buf, 1024 * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    // 1. Host Synchronized Sequential Launches (1 per kernel)
    for (int k = 0; k < 5; ++k) dummy_kernel<<<4, 256, 0, stream>>>(d_buf, 1024);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    QueryPerformanceCounter(&t0);
    for (int t = 0; t < 20; ++t) {
        for (int k = 0; k < NUM_KERNELS; ++k) {
            dummy_kernel<<<4, 256, 0, stream>>>(d_buf, 1024);
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
    }
    QueryPerformanceCounter(&t1);
    double ms_sync = ((double)(t1.QuadPart - t0.QuadPart) * 1000.0) / ((double)freq.QuadPart * 20);

    // 2. Asynchronous Stream Enqueue (all 64 queued, then synchronized once)
    QueryPerformanceCounter(&t0);
    for (int t = 0; t < TRIALS; ++t) {
        for (int k = 0; k < NUM_KERNELS; ++k) {
            dummy_kernel<<<4, 256, 0, stream>>>(d_buf, 1024);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    QueryPerformanceCounter(&t1);
    double ms_async = ((double)(t1.QuadPart - t0.QuadPart) * 1000.0) / ((double)freq.QuadPart * TRIALS);

    // 3. CUDA Graph Capture & Single-Shot Launch
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < NUM_KERNELS; ++k) {
        dummy_kernel<<<4, 256, 0, stream>>>(d_buf, 1024);
    }
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Warmup graph
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    QueryPerformanceCounter(&t0);
    for (int t = 0; t < TRIALS; ++t) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    QueryPerformanceCounter(&t1);
    double ms_graph = ((double)(t1.QuadPart - t0.QuadPart) * 1000.0) / ((double)freq.QuadPart * TRIALS);

    printf("================================================================================\n");
    printf("STEP 11 BENCHMARK 4: KERNEL LAUNCH OVERHEAD VS CUDA GRAPHS\n");
    printf("================================================================================\n");
    printf("Sequence Size             : %d Kernels / Token Step\n", NUM_KERNELS);
    printf("Host Synchronous Sync     : %7.4f ms/token (%.2f us/launch)\n", ms_sync, (ms_sync * 1000.0) / NUM_KERNELS);
    printf("Async Stream Enqueue      : %7.4f ms/token (%.2f us/launch)\n", ms_async, (ms_async * 1000.0) / NUM_KERNELS);
    printf("CUDA Graph Single-Shot    : %7.4f ms/token\n", ms_graph);
    printf("Graph Speedup vs Sync     : %5.2fx\n", ms_sync / ms_graph);
    printf("Graph Speedup vs Async    : %5.2fx\n", ms_async / ms_graph);
    printf("Dispatch Latency Saved    : %7.4f ms (%.1f%% dispatch reduction)\n",
           ms_async - ms_graph, ((ms_async - ms_graph) / ms_async) * 100.0);
    printf("JSON_OUTPUT: {\"benchmark\": \"bench4_kernel_launch_overhead\", \"sync_ms\": %.4f, \"async_ms\": %.4f, \"graph_ms\": %.4f, \"speedup_vs_sync\": %.2f, \"speedup_vs_async\": %.2f}\n",
           ms_sync, ms_async, ms_graph, ms_sync / ms_graph, ms_async / ms_graph);
    printf("================================================================================\n");

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_buf);
    cudaStreamDestroy(stream);

    return 0;
}
