#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>
#include <windows.h>

#define TOLERANCE 1e-4f
#define TRIALS 200

// Config A: Naive Scalar GEMV (Velox v0.1 baseline: reloads x row-by-row)
void gemv_scalar(float* dst, const float* x, const float* w, int M, int K) {
    for (int m = 0; m < M; ++m) {
        const float* row = w + (m * K);
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += x[k] * row[k];
        }
        dst[m] = acc;
    }
}

// Config B: 4-Row Register-Blocked AVX2+FMA3 GEMV
// Holds 4 output accumulators in YMM0..YMM3. Loads each 8-float chunk of x into YMM4 once,
// performing 4 FMAs per vector load (75% reduction in L1 activation read traffic).
void gemv_avx2_4row(float* dst, const float* x, const float* w, int M, int K) {
    int m = 0;
    for (; m + 3 < M; m += 4) {
        const float* r0 = w + ((m + 0) * K);
        const float* r1 = w + ((m + 1) * K);
        const float* r2 = w + ((m + 2) * K);
        const float* r3 = w + ((m + 3) * K);

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int k = 0;
        for (; k + 7 < K; k += 8) {
            __m256 vx = _mm256_loadu_ps(&x[k]);
            acc0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r0[k]), acc0);
            acc1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r1[k]), acc1);
            acc2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r2[k]), acc2);
            acc3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r3[k]), acc3);
        }

        // Horizontal addition of 8 floats per YMM register into scalar float
        __m128 h0 = _mm_add_ps(_mm256_extractf128_ps(acc0, 0), _mm256_extractf128_ps(acc0, 1));
        h0 = _mm_hadd_ps(h0, h0);
        h0 = _mm_hadd_ps(h0, h0);
        float s0 = _mm_cvtss_f32(h0);

        __m128 h1 = _mm_add_ps(_mm256_extractf128_ps(acc1, 0), _mm256_extractf128_ps(acc1, 1));
        h1 = _mm_hadd_ps(h1, h1);
        h1 = _mm_hadd_ps(h1, h1);
        float s1 = _mm_cvtss_f32(h1);

        __m128 h2 = _mm_add_ps(_mm256_extractf128_ps(acc2, 0), _mm256_extractf128_ps(acc2, 1));
        h2 = _mm_hadd_ps(h2, h2);
        h2 = _mm_hadd_ps(h2, h2);
        float s2 = _mm_cvtss_f32(h2);

        __m128 h3 = _mm_add_ps(_mm256_extractf128_ps(acc3, 0), _mm256_extractf128_ps(acc3, 1));
        h3 = _mm_hadd_ps(h3, h3);
        h3 = _mm_hadd_ps(h3, h3);
        float s3 = _mm_cvtss_f32(h3);

        // Tail handling for K not multiple of 8
        for (int kt = k; kt < K; ++kt) {
            s0 += x[kt] * r0[kt];
            s1 += x[kt] * r1[kt];
            s2 += x[kt] * r2[kt];
            s3 += x[kt] * r3[kt];
        }

        dst[m + 0] = s0;
        dst[m + 1] = s1;
        dst[m + 2] = s2;
        dst[m + 3] = s3;
    }

    // Residual rows if M is not a multiple of 4
    for (; m < M; ++m) {
        const float* row = w + (m * K);
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += x[k] * row[k];
        dst[m] = acc;
    }
}

int test_dimension(int M, int K, const char* label) {
    float* x = (float*)_aligned_malloc(K * sizeof(float), 32);
    float* w = (float*)_aligned_malloc((size_t)M * K * sizeof(float), 32);
    float* dst_scalar = (float*)_aligned_malloc(M * sizeof(float), 32);
    float* dst_avx2 = (float*)_aligned_malloc(M * sizeof(float), 32);
    double* dst_ref = (double*)malloc(M * sizeof(double));

    for (int i = 0; i < K; ++i) x[i] = sinf((float)i * 0.05f);
    for (size_t i = 0; i < (size_t)M * K; ++i) w[i] = cosf((float)(i % 31) * 0.02f) * 0.01f;

    // Double-Precision CPU Reference Oracle
    for (int r = 0; r < M; ++r) {
        double acc = 0.0;
        for (int c = 0; c < K; ++c) {
            acc += (double)x[c] * (double)w[(size_t)r * K + c];
        }
        dst_ref[r] = acc;
    }

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        gemv_scalar(dst_scalar, x, w, M, K);
        gemv_avx2_4row(dst_avx2, x, w, M, K);
    }

    // Profile Scalar Baseline
    QueryPerformanceCounter(&t0);
    for (int t = 0; t < TRIALS; ++t) {
        gemv_scalar(dst_scalar, x, w, M, K);
    }
    QueryPerformanceCounter(&t1);
    double ms_scalar = ((double)(t1.QuadPart - t0.QuadPart) * 1000.0) / ((double)freq.QuadPart * TRIALS);

    // Profile AVX2+FMA3 4-Row Blocked Kernel
    QueryPerformanceCounter(&t0);
    for (int t = 0; t < TRIALS; ++t) {
        gemv_avx2_4row(dst_avx2, x, w, M, K);
    }
    QueryPerformanceCounter(&t1);
    double ms_avx2 = ((double)(t1.QuadPart - t0.QuadPart) * 1000.0) / ((double)freq.QuadPart * TRIALS);

    // Numerical Parity Check vs Oracle
    double max_err_scalar = 0.0;
    double max_err_avx2 = 0.0;
    for (int r = 0; r < M; ++r) {
        double err_s = fabs((double)dst_scalar[r] - dst_ref[r]);
        double err_a = fabs((double)dst_avx2[r] - dst_ref[r]);
        if (err_s > max_err_scalar) max_err_scalar = err_s;
        if (err_a > max_err_avx2) max_err_avx2 = err_a;
    }

    double gflops_compute = (2.0 * (double)M * (double)K) / 1e9;
    double gflops_scalar = gflops_compute / (ms_scalar * 1e-3);
    double gflops_avx2 = gflops_compute / (ms_avx2 * 1e-3);
    double speedup = ms_scalar / ms_avx2;
    bool pass = (max_err_avx2 <= TOLERANCE);

    printf("================================================================================\n");
    printf("TEST 4 [%s]: M=%d, K=%d\n", label, M, K);
    printf("Scalar Single-Thread Latency   : %7.4f ms (%6.2f GFLOP/s)\n", ms_scalar, gflops_scalar);
    printf("AVX2+FMA3 4-Row Blocked Latency: %7.4f ms (%6.2f GFLOP/s)\n", ms_avx2, gflops_avx2);
    printf("AVX2 Register Blocking Speedup : %5.2fx over scalar baseline\n", speedup);
    printf("Max Error vs Double Oracle     : Scalar=%.2e, AVX2=%.2e (Tol=%.2e)\n", max_err_scalar, max_err_avx2, TOLERANCE);
    printf("Parity Status                  : %s\n", pass ? "PASSED" : "FAILED");
    printf("JSON_RECORD: {\"label\": \"%s\", \"M\": %d, \"K\": %d, \"scalar_ms\": %.4f, \"avx2_ms\": %.4f, \"speedup\": %.2f, \"gflops_avx2\": %.2f, \"max_err\": %.2e, \"pass\": %s}\n",
           label, M, K, ms_scalar, ms_avx2, speedup, gflops_avx2, max_err_avx2, pass ? "true" : "false");
    printf("================================================================================\n");

    _aligned_free(x);
    _aligned_free(w);
    _aligned_free(dst_scalar);
    _aligned_free(dst_avx2);
    free(dst_ref);

    return pass ? 0 : 1;
}

int main() {
    printf("CPU Micro-Kernel Benchmark: Intel Core i5-12500H AVX2+FMA3 Register Blocking\n");

    int res1 = test_dimension(256, 576, "TINY_BLOCK");
    int res2 = test_dimension(1536, 576, "MLP_PROJECTION");

    if (res1 == 0 && res2 == 0) {
        printf("\n>>> STEP 11B TEST 4 PASSED 100%% <<<\n");
        return 0;
    } else {
        printf("\n>>> STEP 11B TEST 4 FAILED <<<\n");
        return 1;
    }
}
