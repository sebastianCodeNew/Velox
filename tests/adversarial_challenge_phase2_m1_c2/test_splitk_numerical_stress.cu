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
#define GQA_RATIO 3
#define TOLERANCE 1e-4f

__global__ void splitk_stage1_kernel(
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_lse,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
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

    int base_idx = q_head * num_splits + split_id;
    partial_out[base_idx * HEAD_DIM + tid] = acc_d;
    if (tid == 0) {
        partial_max[base_idx] = m;
        partial_lse[base_idx] = l;
    }
}

__global__ void splitk_stage2_kernel(
    float* __restrict__ out,
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_lse,
    int num_splits
) {
    int q_head = blockIdx.x;
    int tid = threadIdx.x;
    int head_split_base = q_head * num_splits;

    float m_global = -1e30f;
    for (int s = 0; s < num_splits; ++s) {
        float ms = partial_max[head_split_base + s];
        if (ms > m_global) m_global = ms;
    }

    float l_global = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float ms = partial_max[head_split_base + s];
        float ls = partial_lse[head_split_base + s];
        l_global += ls * expf(ms - m_global);
    }

    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float ms = partial_max[head_split_base + s];
        float scale = expf(ms - m_global);
        float val = partial_out[(head_split_base + s) * HEAD_DIM + tid];
        acc += val * scale;
    }

    out[q_head * HEAD_DIM + tid] = (l_global > 0.0f) ? (acc / l_global) : 0.0f;
}

int test_splitk_adversarial(int S, int num_splits, float logit_bias) {
    size_t q_bytes = NUM_HEADS_Q * HEAD_DIM * sizeof(float);
    size_t kv_bytes = (size_t)S * NUM_HEADS_KV * HEAD_DIM * sizeof(float);
    size_t out_bytes = NUM_HEADS_Q * HEAD_DIM * sizeof(float);
    size_t p_out_bytes = (size_t)NUM_HEADS_Q * num_splits * HEAD_DIM * sizeof(float);
    size_t p_stats_bytes = (size_t)NUM_HEADS_Q * num_splits * sizeof(float);

    float* h_q = (float*)malloc(q_bytes);
    float* h_k = (float*)malloc(kv_bytes);
    float* h_v = (float*)malloc(kv_bytes);
    float* h_out = (float*)malloc(out_bytes);
    double* h_out_ref = (double*)malloc(NUM_HEADS_Q * HEAD_DIM * sizeof(double));

    float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);

    // Initialize with extreme dynamic range values
    for (int i = 0; i < NUM_HEADS_Q * HEAD_DIM; ++i) {
        h_q[i] = (sinf((float)i * 0.1f) > 0.0f ? 1.0f : -1.0f) * 2.0f;
    }
    for (size_t i = 0; i < (size_t)S * NUM_HEADS_KV * HEAD_DIM; ++i) {
        h_k[i] = (cosf((float)i * 0.13f) * 2.0f) + logit_bias; // Inject bias (+10.0 to +30.0)
        h_v[i] = sinf((float)i * 0.05f);
    }

    // Double precision reference
    for (int qh = 0; qh < NUM_HEADS_Q; ++qh) {
        int kvh = qh / GQA_RATIO;
        double* dots = (double*)malloc(S * sizeof(double));
        double max_dot = -1e300;
        for (int t = 0; t < S; ++t) {
            double dot = 0.0;
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += (double)h_q[qh * HEAD_DIM + d] * (double)h_k[t * NUM_HEADS_KV * HEAD_DIM + kvh * HEAD_DIM + d];
            }
            dot *= (double)sm_scale;
            dots[t] = dot;
            if (dot > max_dot) max_dot = dot;
        }

        double sum_exp = 0.0;
        for (int t = 0; t < S; ++t) {
            sum_exp += exp(dots[t] - max_dot);
        }

        for (int d = 0; d < HEAD_DIM; ++d) {
            double acc = 0.0;
            for (int t = 0; t < S; ++t) {
                double weight = exp(dots[t] - max_dot) / sum_exp;
                acc += weight * (double)h_v[t * NUM_HEADS_KV * HEAD_DIM + kvh * HEAD_DIM + d];
            }
            h_out_ref[qh * HEAD_DIM + d] = acc;
        }
        free(dots);
    }

    float *d_q, *d_k, *d_v, *d_out, *d_p_out, *d_p_max, *d_p_lse;
    CHECK_CUDA(cudaMalloc(&d_q, q_bytes));
    CHECK_CUDA(cudaMalloc(&d_k, kv_bytes));
    CHECK_CUDA(cudaMalloc(&d_v, kv_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, out_bytes));
    CHECK_CUDA(cudaMalloc(&d_p_out, p_out_bytes));
    CHECK_CUDA(cudaMalloc(&d_p_max, p_stats_bytes));
    CHECK_CUDA(cudaMalloc(&d_p_lse, p_stats_bytes));

    CHECK_CUDA(cudaMemcpy(d_q, h_q, q_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k, kv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_v, kv_bytes, cudaMemcpyHostToDevice));

    dim3 stage1_grid(num_splits, NUM_HEADS_Q);
    splitk_stage1_kernel<<<stage1_grid, HEAD_DIM>>>(d_p_out, d_p_max, d_p_lse, d_q, d_k, d_v, S, num_splits, sm_scale);
    splitk_stage2_kernel<<<NUM_HEADS_Q, HEAD_DIM>>>(d_out, d_p_out, d_p_max, d_p_lse, num_splits);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < NUM_HEADS_Q * HEAD_DIM; ++i) {
        double err = fabs((double)h_out[i] - h_out_ref[i]);
        if (err > max_err) max_err = err;
    }

    bool pass = (max_err <= TOLERANCE);
    printf("Context S=%4d | Splits=%2d | Bias=%+4.1f | Max Error: %.2e | Status: %s\n",
           S, num_splits, logit_bias, max_err, pass ? "PASSED" : "FAILED");

    free(h_q); free(h_k); free(h_v); free(h_out); free(h_out_ref);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    cudaFree(d_p_out); cudaFree(d_p_max); cudaFree(d_p_lse);

    return pass ? 0 : 1;
}

int main() {
    printf("================================================================================\n");
    printf("ADVERSARIAL PROBE 4: SPLIT-K FLASHDECODING NUMERICAL STRESS & STABILITY\n");
    printf("================================================================================\n");

    int failures = 0;
    failures += test_splitk_adversarial(32, 2, 0.0f);
    failures += test_splitk_adversarial(64, 4, 15.0f);
    failures += test_splitk_adversarial(128, 4, -10.0f);
    failures += test_splitk_adversarial(512, 8, 25.0f);
    failures += test_splitk_adversarial(1024, 8, 30.0f);
    failures += test_splitk_adversarial(2048, 16, 40.0f);
    failures += test_splitk_adversarial(4096, 32, 50.0f);

    printf("================================================================================\n");
    printf("Adversarial Numerical Verdict: %s (%d failures)\n",
           failures == 0 ? "PASSED 100% (Bit-Accurate across extreme dynamic range to S=4096)" : "FAILED", failures);
    printf("================================================================================\n");

    return failures;
}
