#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define VELOX_HIDDEN_DIM 576
#define VELOX_INTERMEDIATE_DIM 1536

// Exact SOTA Fused Kernel from src/codegen.rs with shared memory isolation
__global__ void __launch_bounds__(256) velox_cuda_fused_out_res_swiglu_res_isolated(
    float* __restrict__ dst_hidden,
    const float* __restrict__ attn_in,
    const float* __restrict__ residual_in,
    const float* __restrict__ w_out,
    const float* __restrict__ w_post_norm,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    size_t layer,
    float eps,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    if (blockIdx.x > 0) return;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    size_t tid = threadIdx.x;

    const float* l_wout = w_out + (layer * dim * dim);
    const float* l_wpost = w_post_norm + (layer * dim);
    const float* l_wgate = w_gate + (layer * dim * inter_dim);
    const float* l_wup = w_up + (layer * dim * inter_dim);
    const float* l_wdown = w_down + (layer * dim * inter_dim);

    // 1. Compute mid[o] = attn_in @ w_out[o] + residual_in[o] into dst_hidden[o]
    for (size_t o = tid; o < dim; o += blockDim.x) {
        const float* row = l_wout + (o * dim);
        float acc = 0.0f;
        for (size_t k = 0; k < dim; ++k) {
            acc += attn_in[k] * row[k];
        }
        dst_hidden[o] = acc + residual_in[o];
    }
    __syncthreads();

    // 2. Pre-SwiGLU RMSNorm reduction over mid (dst_hidden)
    __shared__ float s_ss[256];
    float local_ss = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        float v = dst_hidden[i];
        local_ss += v * v;
    }
    s_ss[tid] = local_ss;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_ss[tid] += s_ss[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf((s_ss[0] / (float)dim) + eps);
    __syncthreads();

    // Precompute normalized intermediate hidden state into shared memory
    // to prevent down-projection in-place chunk updates to dst_hidden from corrupting n_mid
    __shared__ float s_norm_mid[VELOX_HIDDEN_DIM];
    for (size_t k = tid; k < dim; k += blockDim.x) {
        s_norm_mid[k] = dst_hidden[k] * inv_rms * l_wpost[k];
    }
    __syncthreads();

    // 3. Chunked SwiGLU projections and down accumulation
    const size_t CHUNK = 128;
    __shared__ float s_act[CHUNK];

    for (size_t j_base = 0; j_base < inter_dim; j_base += CHUNK) {
        size_t chunk_len = ((j_base + CHUNK) <= inter_dim) ? CHUNK : (inter_dim - j_base);
        for (size_t c = tid; c < chunk_len; c += blockDim.x) {
            size_t j = j_base + c;
            const float* row_gate = l_wgate + (j * dim);
            const float* row_up = l_wup + (j * dim);
            float acc_gate = 0.0f;
            float acc_up = 0.0f;
            for (size_t k = 0; k < dim; ++k) {
                float n_mid = s_norm_mid[k];
                acc_gate += n_mid * row_gate[k];
                acc_up += n_mid * row_up[k];
            }
            float silu = acc_gate / (1.0f + expf(-acc_gate));
            s_act[c] = silu * acc_up;
        }
        __syncthreads();

        for (size_t o = tid; o < dim; o += blockDim.x) {
            const float* row_down = l_wdown + (o * inter_dim);
            float chunk_acc = 0.0f;
            for (size_t c = 0; c < chunk_len; ++c) {
                chunk_acc += s_act[c] * row_down[j_base + c];
            }
            dst_hidden[o] += chunk_acc;
        }
        __syncthreads();
    }
}

// Buggy Un-isolated kernel (counter-example demonstrating the defect)
__global__ void __launch_bounds__(256) velox_cuda_fused_swiglu_buggy_unisolated(
    float* __restrict__ dst_hidden,
    const float* __restrict__ attn_in,
    const float* __restrict__ residual_in,
    const float* __restrict__ w_out,
    const float* __restrict__ w_post_norm,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    size_t layer,
    float eps,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    if (blockIdx.x > 0) return;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    size_t tid = threadIdx.x;

    const float* l_wout = w_out + (layer * dim * dim);
    const float* l_wpost = w_post_norm + (layer * dim);
    const float* l_wgate = w_gate + (layer * dim * inter_dim);
    const float* l_wup = w_up + (layer * dim * inter_dim);
    const float* l_wdown = w_down + (layer * dim * inter_dim);

    for (size_t o = tid; o < dim; o += blockDim.x) {
        const float* row = l_wout + (o * dim);
        float acc = 0.0f;
        for (size_t k = 0; k < dim; ++k) acc += attn_in[k] * row[k];
        dst_hidden[o] = acc + residual_in[o];
    }
    __syncthreads();

    __shared__ float s_ss[256];
    float local_ss = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        float v = dst_hidden[i];
        local_ss += v * v;
    }
    s_ss[tid] = local_ss;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_ss[tid] += s_ss[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf((s_ss[0] / (float)dim) + eps);
    __syncthreads();

    // BUGGY: Does NOT cache in shared memory, reads dst_hidden in-place!
    const size_t CHUNK = 128;
    __shared__ float s_act[CHUNK];

    for (size_t j_base = 0; j_base < inter_dim; j_base += CHUNK) {
        size_t chunk_len = ((j_base + CHUNK) <= inter_dim) ? CHUNK : (inter_dim - j_base);
        for (size_t c = tid; c < chunk_len; c += blockDim.x) {
            size_t j = j_base + c;
            const float* row_gate = l_wgate + (j * dim);
            const float* row_up = l_wup + (j * dim);
            float acc_gate = 0.0f;
            float acc_up = 0.0f;
            for (size_t k = 0; k < dim; ++k) {
                // Reading corrupted dst_hidden after chunk 0!
                float n_mid = dst_hidden[k] * inv_rms * l_wpost[k];
                acc_gate += n_mid * row_gate[k];
                acc_up += n_mid * row_up[k];
            }
            float silu = acc_gate / (1.0f + expf(-acc_gate));
            s_act[c] = silu * acc_up;
        }
        __syncthreads();

        for (size_t o = tid; o < dim; o += blockDim.x) {
            const float* row_down = l_wdown + (o * inter_dim);
            float chunk_acc = 0.0f;
            for (size_t c = 0; c < chunk_len; ++c) {
                chunk_acc += s_act[c] * row_down[j_base + c];
            }
            dst_hidden[o] += chunk_acc;
        }
        __syncthreads();
    }
}

// CPU Mathematical Oracle (Exact unchunked calculation)
void cpu_swiglu_oracle(
    float* out_hidden,
    const float* attn_in,
    const float* residual_in,
    const float* w_out,
    const float* w_post_norm,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float eps
) {
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;

    double* mid = (double*)malloc(dim * sizeof(double));
    double ss = 0.0;
    for (size_t o = 0; o < dim; ++o) {
        double acc = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            acc += (double)attn_in[k] * (double)w_out[o * dim + k];
        }
        mid[o] = acc + (double)residual_in[o];
        ss += mid[o] * mid[o];
    }
    double inv_rms = 1.0 / sqrt((ss / (double)dim) + (double)eps);

    double* norm_mid = (double*)malloc(dim * sizeof(double));
    for (size_t k = 0; k < dim; ++k) {
        norm_mid[k] = mid[k] * inv_rms * (double)w_post_norm[k];
    }

    double* act = (double*)malloc(inter_dim * sizeof(double));
    for (size_t j = 0; j < inter_dim; ++j) {
        double acc_gate = 0.0;
        double acc_up = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            acc_gate += norm_mid[k] * (double)w_gate[j * dim + k];
            acc_up += norm_mid[k] * (double)w_up[j * dim + k];
        }
        double silu = acc_gate / (1.0 + exp(-acc_gate));
        act[j] = silu * acc_up;
    }

    for (size_t o = 0; o < dim; ++o) {
        double down_acc = 0.0;
        for (size_t j = 0; j < inter_dim; ++j) {
            down_acc += act[j] * (double)w_down[o * inter_dim + j];
        }
        out_hidden[o] = (float)(mid[o] + down_acc);
    }

    free(mid);
    free(norm_mid);
    free(act);
}

int main() {
    printf("====================================================================\n");
    printf("[EMPIRICAL CHALLENGER] SwiGLU 12-Chunk Isolation Stress Suite\n");
    printf("====================================================================\n");

    const size_t dim = VELOX_HIDDEN_DIM;          // 576
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM; // 1536
    const float eps = 1e-5f;

    printf("Dimensions: HIDDEN_DIM=%zu, INTERMEDIATE_DIM=%zu, CHUNK=128 (12 chunks)\n",
           dim, inter_dim);

    // Allocate host buffers
    float* h_attn = (float*)malloc(dim * sizeof(float));
    float* h_res = (float*)malloc(dim * sizeof(float));
    float* h_wout = (float*)malloc(dim * dim * sizeof(float));
    float* h_wpost = (float*)malloc(dim * sizeof(float));
    float* h_wgate = (float*)malloc(inter_dim * dim * sizeof(float));
    float* h_wup = (float*)malloc(inter_dim * dim * sizeof(float));
    float* h_wdown = (float*)malloc(dim * inter_dim * sizeof(float));

    float* h_gpu_isolated = (float*)malloc(dim * sizeof(float));
    float* h_gpu_buggy = (float*)malloc(dim * sizeof(float));
    float* h_oracle = (float*)malloc(dim * sizeof(float));

    // Seeded initialization
    srand(1337);
    for (size_t i = 0; i < dim; ++i) h_attn[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (size_t i = 0; i < dim; ++i) h_res[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (size_t i = 0; i < dim * dim; ++i) h_wout[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (size_t i = 0; i < dim; ++i) h_wpost[i] = 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (size_t i = 0; i < inter_dim * dim; ++i) h_wgate[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (size_t i = 0; i < inter_dim * dim; ++i) h_wup[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    for (size_t i = 0; i < dim * inter_dim; ++i) h_wdown[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;

    // Allocate device buffers
    float *d_dst, *d_attn, *d_res, *d_wout, *d_wpost, *d_wgate, *d_wup, *d_wdown;
    CHECK_CUDA(cudaMalloc(&d_dst, dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn, dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_res, dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wout, dim * dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wpost, dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wgate, inter_dim * dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wup, inter_dim * dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_wdown, dim * inter_dim * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_attn, h_attn, dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_res, h_res, dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wout, h_wout, dim * dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wpost, h_wpost, dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wgate, h_wgate, inter_dim * dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wup, h_wup, inter_dim * dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_wdown, h_wdown, dim * inter_dim * sizeof(float), cudaMemcpyHostToDevice));

    // Compute CPU Reference Oracle
    cpu_swiglu_oracle(h_oracle, h_attn, h_res, h_wout, h_wpost, h_wgate, h_wup, h_wdown, eps);

    // 1. Run Isolated Kernel
    CHECK_CUDA(cudaMemset(d_dst, 0, dim * sizeof(float)));
    velox_cuda_fused_out_res_swiglu_res_isolated<<<1, 256>>>(
        d_dst, d_attn, d_res, d_wout, d_wpost, d_wgate, d_wup, d_wdown, 0, eps, 1ULL
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu_isolated, d_dst, dim * sizeof(float), cudaMemcpyDeviceToHost));

    // 2. Run Buggy Un-isolated Kernel
    CHECK_CUDA(cudaMemset(d_dst, 0, dim * sizeof(float)));
    velox_cuda_fused_swiglu_buggy_unisolated<<<1, 256>>>(
        d_dst, d_attn, d_res, d_wout, d_wpost, d_wgate, d_wup, d_wdown, 0, eps, 1ULL
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu_buggy, d_dst, dim * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare Errors
    float max_err_isolated = 0.0f;
    float max_err_buggy = 0.0f;
    for (size_t o = 0; o < dim; ++o) {
        float err_iso = fabsf(h_gpu_isolated[o] - h_oracle[o]);
        if (err_iso > max_err_isolated) max_err_isolated = err_iso;

        float err_bug = fabsf(h_gpu_buggy[o] - h_oracle[o]);
        if (err_bug > max_err_buggy) max_err_buggy = err_bug;
    }

    printf("Oracle Comparison Results:\n");
    printf("  Isolated Kernel Max Abs Error vs Oracle : %e\n", max_err_isolated);
    printf("  Buggy Kernel    Max Abs Error vs Oracle : %e\n", max_err_buggy);

    // 3. Stress Test: 50 Randomized Consecutive Trials
    int stress_trials = 50;
    int stress_passed = 0;
    float peak_stress_err = 0.0f;

    for (int t = 0; t < stress_trials; ++t) {
        for (size_t i = 0; i < dim; ++i) {
            h_attn[i] = ((float)rand() / RAND_MAX - 0.5f) * (float)(t + 1) * 0.1f;
            h_res[i] = ((float)rand() / RAND_MAX - 0.5f) * (float)(t + 1) * 0.1f;
        }
        CHECK_CUDA(cudaMemcpy(d_attn, h_attn, dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_res, h_res, dim * sizeof(float), cudaMemcpyHostToDevice));

        cpu_swiglu_oracle(h_oracle, h_attn, h_res, h_wout, h_wpost, h_wgate, h_wup, h_wdown, eps);

        velox_cuda_fused_out_res_swiglu_res_isolated<<<1, 256>>>(
            d_dst, d_attn, d_res, d_wout, d_wpost, d_wgate, d_wup, d_wdown, 0, eps, 1ULL
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_gpu_isolated, d_dst, dim * sizeof(float), cudaMemcpyDeviceToHost));

        float trial_max_err = 0.0f;
        bool trial_ok = true;
        for (size_t o = 0; o < dim; ++o) {
            if (isnan(h_gpu_isolated[o]) || isinf(h_gpu_isolated[o])) {
                trial_ok = false;
                break;
            }
            float err = fabsf(h_gpu_isolated[o] - h_oracle[o]);
            if (err > trial_max_err) trial_max_err = err;
        }
        if (trial_max_err > peak_stress_err) peak_stress_err = trial_max_err;

        // Tolerance for float32 reduction differences across 576*1536 operations
        if (trial_ok && trial_max_err < 1e-4f) {
            stress_passed++;
        }
    }

    printf("Stress Suite Results (%d randomized trials):\n", stress_trials);
    printf("  Passed Trials   : %d / %d\n", stress_passed, stress_trials);
    printf("  Peak Max Error  : %e\n", peak_stress_err);

    // Clean up
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_attn));
    CHECK_CUDA(cudaFree(d_res));
    CHECK_CUDA(cudaFree(d_wout));
    CHECK_CUDA(cudaFree(d_wpost));
    CHECK_CUDA(cudaFree(d_wgate));
    CHECK_CUDA(cudaFree(d_wup));
    CHECK_CUDA(cudaFree(d_wdown));

    free(h_attn); free(h_res); free(h_wout); free(h_wpost);
    free(h_wgate); free(h_wup); free(h_wdown);
    free(h_gpu_isolated); free(h_gpu_buggy); free(h_oracle);

    bool overall_pass = (max_err_isolated < 1e-4f) && (max_err_buggy > 1e-3f) && (stress_passed == stress_trials);

    printf("====================================================================\n");
    if (overall_pass) {
        printf("[EMPIRICAL VERDICT] PASS: SwiGLU 12-Chunk Isolation is mathematically verified.\n");
        printf("                    Shared memory caching completely eliminates in-place corruption.\n");
    } else {
        printf("[EMPIRICAL VERDICT] FAIL: SwiGLU chunk isolation stress failed.\n");
    }
    printf("====================================================================\n");

    return overall_pass ? 0 : 1;
}
