#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(c) do { \
    cudaError_t e = (c); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define VOCAB_SIZE 49152
#define NUM_BLOCKS 192
#define TRIALS 100

// Config A: Single-thread global memory linear scan (Velox v0.1 baseline)
__global__ void argmax_single_thread(uint32_t* dst, const float* logits, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float best_val = logits[0];
        uint32_t best_idx = 0;
        for (int i = 1; i < N; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = (uint32_t)i;
            }
        }
        *dst = best_idx;
    }
}

// Config B Stage 1: Parallel block reduction (192 blocks x 256 threads)
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

// Config B Stage 2: Final tree reduction across 192 candidate winners
__global__ void argmax_stage2_final(uint32_t* dst, const float* cand_vals, const uint32_t* cand_idxs, int num_cands) {
    __shared__ float s_vals[256];
    __shared__ uint32_t s_idxs[256];
    int tid = threadIdx.x;

    s_vals[tid] = (tid < num_cands) ? cand_vals[tid] : -1e30f;
    s_idxs[tid] = (tid < num_cands) ? cand_idxs[tid] : 0;
    __syncthreads();

    #pragma unroll
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) *dst = s_idxs[0];
}

int main() {
    float *h_logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        h_logits[i] = sinf((float)i * 0.013f);
    }
    // Set a known max element
    h_logits[34567] = 999.0f;

    float *d_logits, *d_cand_vals;
    uint32_t *d_cand_idxs, *d_out_a, *d_out_b;

    CHECK_CUDA(cudaMalloc(&d_logits, VOCAB_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cand_vals, NUM_BLOCKS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cand_idxs, NUM_BLOCKS * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_out_a, sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_out_b, sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpy(d_logits, h_logits, VOCAB_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    argmax_single_thread<<<1, 1, 0, stream>>>(d_out_a, d_logits, VOCAB_SIZE);
    argmax_stage1_block<<<NUM_BLOCKS, 256, 0, stream>>>(d_cand_vals, d_cand_idxs, d_logits, VOCAB_SIZE);
    argmax_stage2_final<<<1, 256, 0, stream>>>(d_out_b, d_cand_vals, d_cand_idxs, NUM_BLOCKS);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Profile Config A
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        argmax_single_thread<<<1, 1, 0, stream>>>(d_out_a, d_logits, VOCAB_SIZE);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_a = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_a, start, stop));
    ms_a /= (float)TRIALS;

    // Profile Config B
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        argmax_stage1_block<<<NUM_BLOCKS, 256, 0, stream>>>(d_cand_vals, d_cand_idxs, d_logits, VOCAB_SIZE);
        argmax_stage2_final<<<1, 256, 0, stream>>>(d_out_b, d_cand_vals, d_cand_idxs, NUM_BLOCKS);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_b = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_b, start, stop));
    ms_b /= (float)TRIALS;

    uint32_t h_idx_a = 0, h_idx_b = 0;
    CHECK_CUDA(cudaMemcpy(&h_idx_a, d_out_a, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_idx_b, d_out_b, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    double speedup = ms_a / ms_b;
    bool pass = (h_idx_a == 34567 && h_idx_b == 34567);

    printf("================================================================================\n");
    printf("STEP 11 BENCHMARK 3: SINGLE-BLOCK VS PARALLEL MULTI-BLOCK LM HEAD\n");
    printf("================================================================================\n");
    printf("Vocabulary Size         : %d elements\n", VOCAB_SIZE);
    printf("Single-Thread Scan      : %7.4f ms (%6.2f us) -> Winner: %u\n", ms_a, ms_a * 1000.0, h_idx_a);
    printf("Two-Stage Parallel Tree : %7.4f ms (%6.2f us) -> Winner: %u\n", ms_b, ms_b * 1000.0, h_idx_b);
    printf("Parallel Argmax Speedup : %5.2fx\n", speedup);
    printf("Parity Verification     : %s\n", pass ? "PASSED" : "FAILED");
    printf("JSON_OUTPUT: {\"benchmark\": \"bench3_parallel_lm_head\", \"single_thread_us\": %.2f, \"parallel_tree_us\": %.2f, \"speedup\": %.2f, \"pass\": %s}\n",
           ms_a * 1000.0, ms_b * 1000.0, speedup, pass ? "true" : "false");
    printf("================================================================================\n");

    free(h_logits);
    cudaFree(d_logits); cudaFree(d_cand_vals); cudaFree(d_cand_idxs); cudaFree(d_out_a); cudaFree(d_out_b);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return pass ? 0 : 1;
}
