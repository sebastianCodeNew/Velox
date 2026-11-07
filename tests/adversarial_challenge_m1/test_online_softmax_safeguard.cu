#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <assert.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

#define HEAD_DIM 64

// Online Softmax Kernel mirroring Velox codegen
__global__ void velox_test_online_softmax_kernel(
    float* d_out,
    const float* d_scores, // [seq_len]
    const float* d_v,      // [seq_len * HEAD_DIM]
    size_t seq_len,
    size_t D
) {
    size_t tid = threadIdx.x;
    float acc_d = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (size_t t = 0; t < seq_len; ++t) {
        float dot = d_scores[t];
        const float* v_t = d_v + (t * D);

        float new_m = (dot > m) ? dot : m;
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - new_m);
        float beta = (new_m == -INFINITY) ? 0.0f : expf(dot - new_m);
        l = l * alpha + beta;
        m = new_m;

        if (tid < D) {
            acc_d = acc_d * alpha + beta * v_t[tid];
        }
        __syncthreads();
    }

    if (tid < D) {
        d_out[tid] = (l > 0.0f) ? (acc_d / l) : 0.0f;
    }
}

// Host reference oracle (double precision two-pass stable softmax)
void host_reference_oracle(
    float* h_out,
    const float* h_scores,
    const float* h_v,
    size_t seq_len,
    size_t D
) {
    double max_s = -INFINITY;
    for (size_t t = 0; t < seq_len; ++t) {
        if (h_scores[t] > max_s) {
            max_s = h_scores[t];
        }
    }

    double sum_exp = 0.0;
    double* weights = (double*)malloc(seq_len * sizeof(double));
    for (size_t t = 0; t < seq_len; ++t) {
        if (max_s == -INFINITY) {
            weights[t] = 0.0;
        } else {
            weights[t] = exp((double)h_scores[t] - max_s);
        }
        sum_exp += weights[t];
    }

    for (size_t d = 0; d < D; ++d) {
        double acc = 0.0;
        for (size_t t = 0; t < seq_len; ++t) {
            acc += weights[t] * (double)h_v[t * D + d];
        }
        h_out[d] = (sum_exp > 0.0) ? (float)(acc / sum_exp) : 0.0f;
    }
    free(weights);
}

int main() {
    printf("====================================================================\n");
    printf("[EMPIRICAL CHALLENGER] Online Softmax Adversarial Edge Cases Suite\n");
    printf("====================================================================\n");

    int device = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Running on Device: %s (Compute Capability %d.%d, VRAM: %zu MiB)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));

    const size_t D = HEAD_DIM;
    const size_t max_seq = 256;

    float* h_scores = (float*)malloc(max_seq * sizeof(float));
    float* h_v = (float*)malloc(max_seq * D * sizeof(float));
    float* h_gpu_out = (float*)malloc(D * sizeof(float));
    float* h_oracle_out = (float*)malloc(D * sizeof(float));

    // Populate V with non-trivial values
    for (size_t i = 0; i < max_seq * D; ++i) {
        h_v[i] = sinf((float)i * 0.1f) + 1.5f;
    }

    float *d_out, *d_scores, *d_v;
    CHECK_CUDA(cudaMalloc(&d_out, D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, max_seq * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, max_seq * D * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_v, h_v, max_seq * D * sizeof(float), cudaMemcpyHostToDevice));

    int total_tests = 0;
    int passed_tests = 0;

    // Test 1: All -INFINITY (Complete Causal/Padding Masking)
    {
        total_tests++;
        size_t seq = 64;
        for (size_t t = 0; t < seq; ++t) h_scores[t] = -INFINITY;

        CHECK_CUDA(cudaMemcpy(d_scores, h_scores, seq * sizeof(float), cudaMemcpyHostToDevice));
        velox_test_online_softmax_kernel<<<1, 256>>>(d_out, d_scores, d_v, seq, D);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_gpu_out, d_out, D * sizeof(float), cudaMemcpyDeviceToHost));
        host_reference_oracle(h_oracle_out, h_scores, h_v, seq, D);

        bool pass = true;
        for (size_t d = 0; d < D; ++d) {
            if (isnan(h_gpu_out[d]) || isinf(h_gpu_out[d]) || h_gpu_out[d] != 0.0f) {
                pass = false;
                printf("FAIL: Test 1 d=%zu got %f (expected 0.0f, no NaN/inf)\n", d, h_gpu_out[d]);
                break;
            }
        }
        if (pass) {
            printf("[PASS] Test 1: All -INFINITY yields zero vector with zero NaN/inf.\n");
            passed_tests++;
        }
    }

    // Test 2: Prefixed -INFINITY (First N tokens masked, followed by valid tokens)
    {
        total_tests++;
        size_t seq = 32;
        for (size_t t = 0; t < 20; ++t) h_scores[t] = -INFINITY;
        for (size_t t = 20; t < seq; ++t) h_scores[t] = 2.0f + (float)t;

        CHECK_CUDA(cudaMemcpy(d_scores, h_scores, seq * sizeof(float), cudaMemcpyHostToDevice));
        velox_test_online_softmax_kernel<<<1, 256>>>(d_out, d_scores, d_v, seq, D);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_gpu_out, d_out, D * sizeof(float), cudaMemcpyDeviceToHost));
        host_reference_oracle(h_oracle_out, h_scores, h_v, seq, D);

        bool pass = true;
        float max_err = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            if (isnan(h_gpu_out[d]) || isinf(h_gpu_out[d])) { pass = false; break; }
            float err = fabsf(h_gpu_out[d] - h_oracle_out[d]);
            if (err > max_err) max_err = err;
        }
        if (pass && max_err < 1e-5f) {
            printf("[PASS] Test 2: Masked prefix (-inf) recovers cleanly (max_err: %e)\n", max_err);
            passed_tests++;
        } else {
            printf("FAIL: Test 2 max_err=%e pass=%d\n", max_err, pass);
        }
    }

    // Test 3: Extreme Large Positive Values (e.g. 500.0, 1000.0, 50000.0)
    {
        total_tests++;
        size_t seq = 16;
        for (size_t t = 0; t < seq; ++t) h_scores[t] = 10000.0f + (float)t * 0.5f;

        CHECK_CUDA(cudaMemcpy(d_scores, h_scores, seq * sizeof(float), cudaMemcpyHostToDevice));
        velox_test_online_softmax_kernel<<<1, 256>>>(d_out, d_scores, d_v, seq, D);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_gpu_out, d_out, D * sizeof(float), cudaMemcpyDeviceToHost));
        host_reference_oracle(h_oracle_out, h_scores, h_v, seq, D);

        bool pass = true;
        float max_err = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            if (isnan(h_gpu_out[d]) || isinf(h_gpu_out[d])) { pass = false; break; }
            float err = fabsf(h_gpu_out[d] - h_oracle_out[d]);
            if (err > max_err) max_err = err;
        }
        if (pass && max_err < 1e-5f) {
            printf("[PASS] Test 3: Extreme positive values (+10000.0) without exp overflow (max_err: %e)\n", max_err);
            passed_tests++;
        } else {
            printf("FAIL: Test 3 max_err=%e pass=%d\n", max_err, pass);
        }
    }

    // Test 4: Extreme Negative Values (e.g. -1000.0, -100000.0)
    {
        total_tests++;
        size_t seq = 16;
        for (size_t t = 0; t < seq; ++t) h_scores[t] = -50000.0f + (float)t;

        CHECK_CUDA(cudaMemcpy(d_scores, h_scores, seq * sizeof(float), cudaMemcpyHostToDevice));
        velox_test_online_softmax_kernel<<<1, 256>>>(d_out, d_scores, d_v, seq, D);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_gpu_out, d_out, D * sizeof(float), cudaMemcpyDeviceToHost));
        host_reference_oracle(h_oracle_out, h_scores, h_v, seq, D);

        bool pass = true;
        float max_err = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            if (isnan(h_gpu_out[d]) || isinf(h_gpu_out[d])) { pass = false; break; }
            float err = fabsf(h_gpu_out[d] - h_oracle_out[d]);
            if (err > max_err) max_err = err;
        }
        if (pass && max_err < 1e-5f) {
            printf("[PASS] Test 4: Extreme negative values (-50000.0) without underflow NaN (max_err: %e)\n", max_err);
            passed_tests++;
        } else {
            printf("FAIL: Test 4 max_err=%e pass=%d\n", max_err, pass);
        }
    }

    // Test 5: Alternating Extreme Dynamic Range (+1000.0, -1000.0, -INFINITY, 0.0, +1002.0)
    {
        total_tests++;
        size_t seq = 10;
        h_scores[0] = -INFINITY;
        h_scores[1] = 1000.0f;
        h_scores[2] = -1000.0f;
        h_scores[3] = -INFINITY;
        h_scores[4] = 1001.0f;
        h_scores[5] = -INFINITY;
        h_scores[6] = 500.0f;
        h_scores[7] = 1001.5f;
        h_scores[8] = -INFINITY;
        h_scores[9] = 1001.5f;

        CHECK_CUDA(cudaMemcpy(d_scores, h_scores, seq * sizeof(float), cudaMemcpyHostToDevice));
        velox_test_online_softmax_kernel<<<1, 256>>>(d_out, d_scores, d_v, seq, D);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_gpu_out, d_out, D * sizeof(float), cudaMemcpyDeviceToHost));
        host_reference_oracle(h_oracle_out, h_scores, h_v, seq, D);

        bool pass = true;
        float max_err = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            if (isnan(h_gpu_out[d]) || isinf(h_gpu_out[d])) { pass = false; break; }
            float err = fabsf(h_gpu_out[d] - h_oracle_out[d]);
            if (err > max_err) max_err = err;
        }
        if (pass && max_err < 1e-5f) {
            printf("[PASS] Test 5: Mixed dynamic range (-inf, +-1000) matches double oracle (max_err: %e)\n", max_err);
            passed_tests++;
        } else {
            printf("FAIL: Test 5 max_err=%e pass=%d\n", max_err, pass);
        }
    }

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_v));
    free(h_scores);
    free(h_v);
    free(h_gpu_out);
    free(h_oracle_out);

    printf("====================================================================\n");
    printf("Online Softmax Adversarial Verdict: %d / %d PASSED\n", passed_tests, total_tests);
    printf("====================================================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
