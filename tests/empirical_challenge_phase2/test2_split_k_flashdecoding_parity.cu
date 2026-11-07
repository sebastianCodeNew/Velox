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

#define HEAD_DIM 64
#define NUM_HEADS_Q 9
#define NUM_HEADS_KV 3
#define GQA_RATIO 3  // NUM_HEADS_Q / NUM_HEADS_KV
#define TOLERANCE 1e-4f
#define TRIALS 50

// Stage 1: Split-K Partial Attention with Online Softmax per split
// Grid: <<<dim3(NUM_SPLITS, NUM_HEADS_Q), HEAD_DIM>>>
__global__ void splitk_stage1_kernel(
    float* __restrict__ partial_out,    // [NUM_HEADS_Q, NUM_SPLITS, HEAD_DIM]
    float* __restrict__ partial_max,    // [NUM_HEADS_Q, NUM_SPLITS]
    float* __restrict__ partial_lse,    // [NUM_HEADS_Q, NUM_SPLITS]
    const float* __restrict__ q,        // [NUM_HEADS_Q, HEAD_DIM]
    const float* __restrict__ k_cache,  // [SEQ_LEN, NUM_HEADS_KV, HEAD_DIM]
    const float* __restrict__ v_cache,  // [SEQ_LEN, NUM_HEADS_KV, HEAD_DIM]
    int seq_len,
    int num_splits,
    float sm_scale
) {
    int split_id = blockIdx.x;
    int q_head = blockIdx.y;
    int tid = threadIdx.x; // [0..HEAD_DIM-1]
    int kv_head = q_head / GQA_RATIO;

    int chunk_size = (seq_len + num_splits - 1) / num_splits;
    int t_start = split_id * chunk_size;
    int t_end = t_start + chunk_size;
    if (t_end > seq_len) t_end = seq_len;

    float m = -1e30f;
    float l = 0.0f;
    float acc_d = 0.0f;

    const float* q_h = q + (q_head * HEAD_DIM);

    // Online Softmax loop across temporal tokens allocated to this split
    for (int t = t_start; t < t_end; ++t) {
        const float* k_t = k_cache + (t * NUM_HEADS_KV * HEAD_DIM + kv_head * HEAD_DIM);
        const float* v_t = v_cache + (t * NUM_HEADS_KV * HEAD_DIM + kv_head * HEAD_DIM);

        // Dot product Q . K_t in shared memory
        __shared__ float s_dot[HEAD_DIM];
        s_dot[tid] = q_h[tid] * k_t[tid];
        __syncthreads();

        for (int s = HEAD_DIM / 2; s > 0; s >>= 1) {
            if (tid < s) s_dot[tid] += s_dot[tid + s];
            __syncthreads();
        }
        float dot = s_dot[0] * sm_scale;

        // Online Softmax update
        float new_m = fmaxf(m, dot);
        float alpha = expf(m - new_m);
        float beta = expf(dot - new_m);
        l = l * alpha + beta;
        m = new_m;

        acc_d = acc_d * alpha + beta * v_t[tid];
        __syncthreads();
    }

    // Write partial results
    int base_idx = q_head * num_splits + split_id;
    partial_out[base_idx * HEAD_DIM + tid] = acc_d;
    if (tid == 0) {
        partial_max[base_idx] = m;
        partial_lse[base_idx] = l;
    }
}

// Stage 2: Merge Partial Splits via LogSumExp Rescaling
// Grid: <<<NUM_HEADS_Q, HEAD_DIM>>>
__global__ void splitk_stage2_kernel(
    float* __restrict__ out,                  // [NUM_HEADS_Q, HEAD_DIM]
    const float* __restrict__ partial_out,    // [NUM_HEADS_Q, NUM_SPLITS, HEAD_DIM]
    const float* __restrict__ partial_max,    // [NUM_HEADS_Q, NUM_SPLITS]
    const float* __restrict__ partial_lse,    // [NUM_HEADS_Q, NUM_SPLITS]
    int num_splits
) {
    int q_head = blockIdx.x;
    int tid = threadIdx.x; // [0..HEAD_DIM-1]

    int head_split_base = q_head * num_splits;

    // 1. Find global maximum logit across splits
    float m_global = -1e30f;
    for (int s = 0; s < num_splits; ++s) {
        float ms = partial_max[head_split_base + s];
        if (ms > m_global) m_global = ms;
    }

    // 2. Compute rescaled denominator sum
    float l_global = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float ms = partial_max[head_split_base + s];
        float ls = partial_lse[head_split_base + s];
        l_global += ls * expf(ms - m_global);
    }

    // 3. Accumulate rescaled V vector contributions
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float ms = partial_max[head_split_base + s];
        float scale = expf(ms - m_global);
        float val = partial_out[(head_split_base + s) * HEAD_DIM + tid];
        acc += val * scale;
    }

    out[q_head * HEAD_DIM + tid] = (l_global > 0.0f) ? (acc / l_global) : 0.0f;
}

// Single-Block Naive Reference Kernel on GPU for latency comparison
__global__ void attention_single_block_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    int seq_len,
    float sm_scale
) {
    int q_head = blockIdx.x;
    int tid = threadIdx.x;
    int kv_head = q_head / GQA_RATIO;

    const float* q_h = q + (q_head * HEAD_DIM);
    float m = -1e30f;
    float l = 0.0f;
    float acc_d = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        const float* k_t = k_cache + (t * NUM_HEADS_KV * HEAD_DIM + kv_head * HEAD_DIM);
        const float* v_t = v_cache + (t * NUM_HEADS_KV * HEAD_DIM + kv_head * HEAD_DIM);

        __shared__ float s_dot[HEAD_DIM];
        s_dot[tid] = q_h[tid] * k_t[tid];
        __syncthreads();

        for (int s = HEAD_DIM / 2; s > 0; s >>= 1) {
            if (tid < s) s_dot[tid] += s_dot[tid + s];
            __syncthreads();
        }
        float dot = s_dot[0] * sm_scale;

        float new_m = fmaxf(m, dot);
        float alpha = expf(m - new_m);
        float beta = expf(dot - new_m);
        l = l * alpha + beta;
        m = new_m;

        acc_d = acc_d * alpha + beta * v_t[tid];
        __syncthreads();
    }

    out[q_head * HEAD_DIM + tid] = (l > 0.0f) ? (acc_d / l) : 0.0f;
}

int test_context_length(int S, int num_splits) {
    float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);

    size_t q_bytes = NUM_HEADS_Q * HEAD_DIM * sizeof(float);
    size_t kv_bytes = (size_t)S * NUM_HEADS_KV * HEAD_DIM * sizeof(float);
    size_t out_bytes = NUM_HEADS_Q * HEAD_DIM * sizeof(float);

    float* h_q = (float*)malloc(q_bytes);
    float* h_k = (float*)malloc(kv_bytes);
    float* h_v = (float*)malloc(kv_bytes);
    float* h_out_splitk = (float*)malloc(out_bytes);
    double* h_out_ref = (double*)malloc(NUM_HEADS_Q * HEAD_DIM * sizeof(double));

    for (int i = 0; i < NUM_HEADS_Q * HEAD_DIM; ++i) h_q[i] = 0.1f * sinf((float)i * 0.13f);
    for (size_t i = 0; i < (size_t)S * NUM_HEADS_KV * HEAD_DIM; ++i) {
        h_k[i] = 0.08f * cosf((float)(i % 23) * 0.11f);
        h_v[i] = 0.06f * sinf((float)(i % 19) * 0.17f);
    }

    // Double-Precision CPU Reference Oracle
    for (int qh = 0; qh < NUM_HEADS_Q; ++qh) {
        int kvh = qh / GQA_RATIO;
        const float* q_ptr = h_q + (qh * HEAD_DIM);

        double* scores = (double*)malloc(S * sizeof(double));
        double max_score = -1e30;

        for (int t = 0; t < S; ++t) {
            const float* k_ptr = h_k + (t * NUM_HEADS_KV * HEAD_DIM + kvh * HEAD_DIM);
            double dot = 0.0;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += (double)q_ptr[d] * (double)k_ptr[d];
            }
            dot *= (double)sm_scale;
            scores[t] = dot;
            if (dot > max_score) max_score = dot;
        }

        double sum_exp = 0.0;
        for (int t = 0; t < S; ++t) {
            scores[t] = exp(scores[t] - max_score);
            sum_exp += scores[t];
        }

        for (int d = 0; d < HEAD_DIM; ++d) {
            double acc = 0.0;
            for (int t = 0; t < S; ++t) {
                const float* v_ptr = h_v + (t * NUM_HEADS_KV * HEAD_DIM + kvh * HEAD_DIM);
                acc += scores[t] * (double)v_ptr[d];
            }
            h_out_ref[qh * HEAD_DIM + d] = acc / sum_exp;
        }
        free(scores);
    }

    // GPU Allocations
    float *d_q, *d_k, *d_v, *d_out_splitk, *d_out_single;
    float *d_p_out, *d_p_max, *d_p_lse;

    size_t p_out_bytes = (size_t)NUM_HEADS_Q * num_splits * HEAD_DIM * sizeof(float);
    size_t p_meta_bytes = (size_t)NUM_HEADS_Q * num_splits * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_q, q_bytes));
    CHECK_CUDA(cudaMalloc(&d_k, kv_bytes));
    CHECK_CUDA(cudaMalloc(&d_v, kv_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_splitk, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_single, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_p_out, p_out_bytes));
    CHECK_CUDA(cudaMalloc(&d_p_max, p_meta_bytes));
    CHECK_CUDA(cudaMalloc(&d_p_lse, p_meta_bytes));

    CHECK_CUDA(cudaMemcpy(d_q, h_q, q_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k, kv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_v, kv_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 stage1_grid(num_splits, NUM_HEADS_Q);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        splitk_stage1_kernel<<<stage1_grid, HEAD_DIM, 0, stream>>>(
            d_p_out, d_p_max, d_p_lse, d_q, d_k, d_v, S, num_splits, sm_scale
        );
        splitk_stage2_kernel<<<NUM_HEADS_Q, HEAD_DIM, 0, stream>>>(
            d_out_splitk, d_p_out, d_p_max, d_p_lse, num_splits
        );
        attention_single_block_kernel<<<NUM_HEADS_Q, HEAD_DIM, 0, stream>>>(
            d_out_single, d_q, d_k, d_v, S, sm_scale
        );
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Profile Split-K FlashDecoding
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        splitk_stage1_kernel<<<stage1_grid, HEAD_DIM, 0, stream>>>(
            d_p_out, d_p_max, d_p_lse, d_q, d_k, d_v, S, num_splits, sm_scale
        );
        splitk_stage2_kernel<<<NUM_HEADS_Q, HEAD_DIM, 0, stream>>>(
            d_out_splitk, d_p_out, d_p_max, d_p_lse, num_splits
        );
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_splitk = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_splitk, start, stop));
    ms_splitk /= (float)TRIALS;

    // Profile Single-Block Baseline
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < TRIALS; ++i) {
        attention_single_block_kernel<<<NUM_HEADS_Q, HEAD_DIM, 0, stream>>>(
            d_out_single, d_q, d_k, d_v, S, sm_scale
        );
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    float ms_single = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_single, start, stop));
    ms_single /= (float)TRIALS;

    CHECK_CUDA(cudaMemcpy(h_out_splitk, d_out_splitk, out_bytes, cudaMemcpyDeviceToHost));

    // Parity Check vs Double Precision Oracle
    double max_err = 0.0;
    for (int i = 0; i < NUM_HEADS_Q * HEAD_DIM; ++i) {
        double err = fabs((double)h_out_splitk[i] - h_out_ref[i]);
        if (err > max_err) max_err = err;
    }

    double speedup = ms_single / ms_splitk;
    bool pass = (max_err <= TOLERANCE);

    printf("================================================================================\n");
    printf("TEST 2: S=%4d tokens (Q_heads=%d, KV_heads=%d, Splits=%d)\n", S, NUM_HEADS_Q, NUM_HEADS_KV, num_splits);
    printf("Single-Block Attention : %7.4f ms (%6.2f us)\n", ms_single, ms_single * 1000.0);
    printf("Split-K FlashDecoding  : %7.4f ms (%6.2f us)\n", ms_splitk, ms_splitk * 1000.0);
    printf("Split-K Speedup        : %5.2fx over single block\n", speedup);
    printf("Max Absolute Error     : %.2e (Tolerance: %.2e)\n", max_err, TOLERANCE);
    printf("Parity Status          : %s\n", pass ? "PASSED" : "FAILED");
    printf("JSON_RECORD: {\"S\": %d, \"splits\": %d, \"single_block_us\": %.2f, \"splitk_us\": %.2f, \"speedup\": %.2f, \"max_err\": %.2e, \"pass\": %s}\n",
           S, num_splits, ms_single * 1000.0, ms_splitk * 1000.0, speedup, max_err, pass ? "true" : "false");
    printf("================================================================================\n");

    free(h_q); free(h_k); free(h_v); free(h_out_splitk); free(h_out_ref);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out_splitk); cudaFree(d_out_single);
    cudaFree(d_p_out); cudaFree(d_p_max); cudaFree(d_p_lse);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return pass ? 0 : 1;
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Target Device: %s (Compute %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    int test_lengths[4] = {64, 512, 1024, 2048};
    int test_splits[4]  = { 4,   8,    8,   16};

    int total_failed = 0;
    for (int i = 0; i < 4; ++i) {
        int res = test_context_length(test_lengths[i], test_splits[i]);
        if (res != 0) total_failed++;
    }

    if (total_failed == 0) {
        printf("\n>>> STEP 11B TEST 2 ALL CONTEXT LENGTHS S=64..2048 PASSED 100%% <<<\n");
        return 0;
    } else {
        printf("\n>>> STEP 11B TEST 2 VERIFICATION FAILED (%d failures) <<<\n", total_failed);
        return 1;
    }
}
