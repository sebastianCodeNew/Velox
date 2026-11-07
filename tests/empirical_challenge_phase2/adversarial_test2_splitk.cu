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

#define HEAD_DIM 64
#define NUM_HEADS_Q 9
#define NUM_HEADS_KV 3
#define GQA_RATIO 3
#define TOLERANCE 1e-4f

// Stage 1: Split-K Partial Attention with Online Softmax per split
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
    int tid = threadIdx.x;
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

    // Write partial results
    int base_idx = q_head * num_splits + split_id;
    partial_out[base_idx * HEAD_DIM + tid] = acc_d;
    if (tid == 0) {
        partial_max[base_idx] = m;
        partial_lse[base_idx] = l;
    }
}

// Stage 2: Merge Partial Splits via LogSumExp Rescaling
__global__ void splitk_stage2_kernel(
    float* __restrict__ out,                  // [NUM_HEADS_Q, HEAD_DIM]
    const float* __restrict__ partial_out,    // [NUM_HEADS_Q, NUM_SPLITS, HEAD_DIM]
    const float* __restrict__ partial_max,    // [NUM_HEADS_Q, NUM_SPLITS]
    const float* __restrict__ partial_lse,    // [NUM_HEADS_Q, NUM_SPLITS]
    int num_splits
) {
    int q_head = blockIdx.x;
    int tid = threadIdx.x;

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

static inline float rand_float(float min_val, float max_val) {
    float scale = (float)rand() / (float)RAND_MAX;
    return min_val + scale * (max_val - min_val);
}

int test_splitk_adversarial(int S, int num_splits, float scale_mult, const char* label) {
    float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);

    size_t q_bytes = NUM_HEADS_Q * HEAD_DIM * sizeof(float);
    size_t kv_bytes = (size_t)S * NUM_HEADS_KV * HEAD_DIM * sizeof(float);
    size_t out_bytes = NUM_HEADS_Q * HEAD_DIM * sizeof(float);

    float* h_q = (float*)malloc(q_bytes);
    float* h_k = (float*)malloc(kv_bytes);
    float* h_v = (float*)malloc(kv_bytes);
    float* h_out_splitk = (float*)malloc(out_bytes);
    double* h_out_ref = (double*)malloc(NUM_HEADS_Q * HEAD_DIM * sizeof(double));

    srand(1001 + S * 31 + num_splits);
    for (int i = 0; i < NUM_HEADS_Q * HEAD_DIM; ++i) h_q[i] = rand_float(-1.0f, 1.0f) * scale_mult;
    for (size_t i = 0; i < (size_t)S * NUM_HEADS_KV * HEAD_DIM; ++i) {
        h_k[i] = rand_float(-1.0f, 1.0f) * scale_mult;
        h_v[i] = rand_float(-1.0f, 1.0f);
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

    float *d_q, *d_k, *d_v, *d_out_splitk;
    float *d_p_out, *d_p_max, *d_p_lse;

    size_t p_out_bytes = (size_t)NUM_HEADS_Q * num_splits * HEAD_DIM * sizeof(float);
    size_t p_meta_bytes = (size_t)NUM_HEADS_Q * num_splits * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_q, q_bytes));
    CHECK_CUDA(cudaMalloc(&d_k, kv_bytes));
    CHECK_CUDA(cudaMalloc(&d_v, kv_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_splitk, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_p_out, p_out_bytes));
    CHECK_CUDA(cudaMalloc(&d_p_max, p_meta_bytes));
    CHECK_CUDA(cudaMalloc(&d_p_lse, p_meta_bytes));

    CHECK_CUDA(cudaMemcpy(d_q, h_q, q_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k, kv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_v, kv_bytes, cudaMemcpyHostToDevice));

    dim3 stage1_grid(num_splits, NUM_HEADS_Q);
    splitk_stage1_kernel<<<stage1_grid, HEAD_DIM>>>(
        d_p_out, d_p_max, d_p_lse, d_q, d_k, d_v, S, num_splits, sm_scale
    );
    CHECK_CUDA(cudaGetLastError());

    splitk_stage2_kernel<<<NUM_HEADS_Q, HEAD_DIM>>>(
        d_out_splitk, d_p_out, d_p_max, d_p_lse, num_splits
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_splitk, d_out_splitk, out_bytes, cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    int nan_count = 0;
    for (int i = 0; i < NUM_HEADS_Q * HEAD_DIM; ++i) {
        if (isnan(h_out_splitk[i]) || isinf(h_out_splitk[i])) {
            nan_count++;
        }
        double err = fabs((double)h_out_splitk[i] - h_out_ref[i]);
        if (err > max_abs_err) max_abs_err = err;
    }

    bool pass = (nan_count == 0) && (max_abs_err <= TOLERANCE);
    printf("[%s] S=%-4d Splits=%-2d | NaNs: %d | MaxAbsErr: %.2e | Tol: %.2e | Status: %s\n",
           label, S, num_splits, nan_count, max_abs_err, TOLERANCE, pass ? "PASSED" : "FAILED");

    free(h_q); free(h_k); free(h_v); free(h_out_splitk); free(h_out_ref);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out_splitk);
    cudaFree(d_p_out); cudaFree(d_p_max); cudaFree(d_p_lse);

    return pass ? 0 : 1;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("================================================================================\n");
    printf("ADVERSARIAL CHALLENGE TEST 2: SPLIT-K FLASHDECODING BOUNDARY & PARITY STRESS\n");
    printf("Target Device: %s (Compute %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("================================================================================\n");

    int failures = 0;

    // 1. Extreme Small Boundary Contexts (S=1, S=3) where S < num_splits
    printf("\n--- Section 1: Extreme Small Contexts S=1 and S=3 (Testing S < num_splits) ---\n");
    failures += test_splitk_adversarial(1, 1, 1.0f, "S1_SPLITS1");
    failures += test_splitk_adversarial(1, 2, 1.0f, "S1_SPLITS2");
    failures += test_splitk_adversarial(1, 4, 1.0f, "S1_SPLITS4_OVERCOMMIT");
    failures += test_splitk_adversarial(1, 8, 1.0f, "S1_SPLITS8_OVERCOMMIT");
    failures += test_splitk_adversarial(1, 16, 1.0f, "S1_SPLITS16_OVERCOMMIT");

    failures += test_splitk_adversarial(3, 1, 1.0f, "S3_SPLITS1");
    failures += test_splitk_adversarial(3, 2, 1.0f, "S3_SPLITS2");
    failures += test_splitk_adversarial(3, 4, 1.0f, "S3_SPLITS4_OVERCOMMIT");
    failures += test_splitk_adversarial(3, 8, 1.0f, "S3_SPLITS8_OVERCOMMIT");

    // 2. Prime / Unaligned Sequence Lengths (S=127)
    printf("\n--- Section 2: Prime / Unaligned Length S=127 ---\n");
    failures += test_splitk_adversarial(127, 2, 1.0f, "S127_SPLITS2");
    failures += test_splitk_adversarial(127, 4, 1.0f, "S127_SPLITS4");
    failures += test_splitk_adversarial(127, 8, 1.0f, "S127_SPLITS8");
    failures += test_splitk_adversarial(127, 16, 1.0f, "S127_SPLITS16");

    // 3. Long Context Boundary (S=2048)
    printf("\n--- Section 3: Long Context S=2048 ---\n");
    failures += test_splitk_adversarial(2048, 8, 1.0f, "S2048_SPLITS8");
    failures += test_splitk_adversarial(2048, 16, 1.0f, "S2048_SPLITS16");
    failures += test_splitk_adversarial(2048, 32, 1.0f, "S2048_SPLITS32");

    // 4. Extreme Dynamic Range & Softmax Scale Variations
    printf("\n--- Section 4: Softmax Numerical Dynamics (Underflow / Overflow Stress) ---\n");
    failures += test_splitk_adversarial(127, 8, 5.0f, "S127_HIGH_SCALE_SHARP_ATTN");
    failures += test_splitk_adversarial(127, 8, 0.01f, "S127_TINY_SCALE_UNIFORM_ATTN");
    failures += test_splitk_adversarial(1, 8, 10.0f, "S1_HIGH_SCALE");
    failures += test_splitk_adversarial(2048, 16, 3.0f, "S2048_HIGH_SCALE");

    printf("\n================================================================================\n");
    printf("TEST 2 SUMMARY: Total Failures = %d / 17\n", failures);
    printf("================================================================================\n");

    return (failures == 0) ? 0 : 1;
}
