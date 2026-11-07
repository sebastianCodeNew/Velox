#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include <windows.h>

#define M_DIM 1536
#define K_DIM 576
#define TRIALS 500

// Config A: Scalar Single-Thread
void gemv_scalar(float* dst, const float* x, const float* w, int M, int K) {
    for (int m = 0; m < M; ++m) {
        const float* row = w + (m * K);
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += x[k] * row[k];
        dst[m] = acc;
    }
}

// Config B: 4-Row Register-Blocked AVX2+FMA3 (Single-Thread)
void gemv_avx2_4row_st(float* dst, const float* x, const float* w, int M, int K) {
    for (int m = 0; m < M; m += 4) {
        const float* r0 = w + ((m + 0) * K);
        const float* r1 = w + ((m + 1) * K);
        const float* r2 = w + ((m + 2) * K);
        const float* r3 = w + ((m + 3) * K);

        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();

        for (int k = 0; k < K; k += 8) {
            __m256 vx = _mm256_loadu_ps(&x[k]);
            a0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r0[k]), a0);
            a1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r1[k]), a1);
            a2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r2[k]), a2);
            a3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r3[k]), a3);
        }

        __m128 h0 = _mm_add_ps(_mm256_extractf128_ps(a0, 0), _mm256_extractf128_ps(a0, 1));
        h0 = _mm_hadd_ps(h0, h0);
        h0 = _mm_hadd_ps(h0, h0);
        dst[m + 0] = _mm_cvtss_f32(h0);

        __m128 h1 = _mm_add_ps(_mm256_extractf128_ps(a1, 0), _mm256_extractf128_ps(a1, 1));
        h1 = _mm_hadd_ps(h1, h1);
        h1 = _mm_hadd_ps(h1, h1);
        dst[m + 1] = _mm_cvtss_f32(h1);

        __m128 h2 = _mm_add_ps(_mm256_extractf128_ps(a2, 0), _mm256_extractf128_ps(a2, 1));
        h2 = _mm_hadd_ps(h2, h2);
        h2 = _mm_hadd_ps(h2, h2);
        dst[m + 2] = _mm_cvtss_f32(h2);

        __m128 h3 = _mm_add_ps(_mm256_extractf128_ps(a3, 0), _mm256_extractf128_ps(a3, 1));
        h3 = _mm_hadd_ps(h3, h3);
        h3 = _mm_hadd_ps(h3, h3);
        dst[m + 3] = _mm_cvtss_f32(h3);
    }
}

// Config C: 16-Thread OpenMP + 4-Row AVX2+FMA3
void gemv_avx2_4row_mt(float* dst, const float* x, const float* w, int M, int K) {
    int m;
    #pragma omp parallel for schedule(static)
    for (m = 0; m < M; m += 4) {
        const float* r0 = w + ((m + 0) * K);
        const float* r1 = w + ((m + 1) * K);
        const float* r2 = w + ((m + 2) * K);
        const float* r3 = w + ((m + 3) * K);

        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();

        for (int k = 0; k < K; k += 8) {
            __m256 vx = _mm256_loadu_ps(&x[k]);
            a0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r0[k]), a0);
            a1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r1[k]), a1);
            a2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r2[k]), a2);
            a3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r3[k]), a3);
        }

        __m128 h0 = _mm_add_ps(_mm256_extractf128_ps(a0, 0), _mm256_extractf128_ps(a0, 1));
        h0 = _mm_hadd_ps(h0, h0);
        h0 = _mm_hadd_ps(h0, h0);
        dst[m + 0] = _mm_cvtss_f32(h0);

        __m128 h1 = _mm_add_ps(_mm256_extractf128_ps(a1, 0), _mm256_extractf128_ps(a1, 1));
        h1 = _mm_hadd_ps(h1, h1);
        h1 = _mm_hadd_ps(h1, h1);
        dst[m + 1] = _mm_cvtss_f32(h1);

        __m128 h2 = _mm_add_ps(_mm256_extractf128_ps(a2, 0), _mm256_extractf128_ps(a2, 1));
        h2 = _mm_hadd_ps(h2, h2);
        h2 = _mm_hadd_ps(h2, h2);
        dst[m + 2] = _mm_cvtss_f32(h2);

        __m128 h3 = _mm_add_ps(_mm256_extractf128_ps(a3, 0), _mm256_extractf128_ps(a3, 1));
        h3 = _mm_hadd_ps(h3, h3);
        h3 = _mm_hadd_ps(h3, h3);
        dst[m + 3] = _mm_cvtss_f32(h3);
    }
}

int main() {
    int max_threads = omp_get_max_threads();
    printf("Host Processor: Intel Core i5-12500H (OpenMP Threads: %d)\n", max_threads);

    float* x = (float*)_aligned_malloc(K_DIM * sizeof(float), 32);
    float* w = (float*)_aligned_malloc(M_DIM * K_DIM * sizeof(float), 32);
    float* dst_a = (float*)_aligned_malloc(M_DIM * sizeof(float), 32);
    float* dst_b = (float*)_aligned_malloc(M_DIM * sizeof(float), 32);
    float* dst_c = (float*)_aligned_malloc(M_DIM * sizeof(float), 32);

    for (int i = 0; i < K_DIM; ++i) x[i] = 1.0f / (float)(i + 1);
    for (int i = 0; i < M_DIM * K_DIM; ++i) w[i] = 0.001f * (float)(i % 17);

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        gemv_scalar(dst_a, x, w, M_DIM, K_DIM);
        gemv_avx2_4row_st(dst_b, x, w, M_DIM, K_DIM);
        gemv_avx2_4row_mt(dst_c, x, w, M_DIM, K_DIM);
    }

    // Profile Config A (Scalar ST)
    QueryPerformanceCounter(&t0);
    for (int t = 0; t < TRIALS; ++t) gemv_scalar(dst_a, x, w, M_DIM, K_DIM);
    QueryPerformanceCounter(&t1);
    double ms_a = ((double)(t1.QuadPart - t0.QuadPart) * 1000.0) / ((double)freq.QuadPart * TRIALS);

    // Profile Config B (AVX2 ST)
    QueryPerformanceCounter(&t0);
    for (int t = 0; t < TRIALS; ++t) gemv_avx2_4row_st(dst_b, x, w, M_DIM, K_DIM);
    QueryPerformanceCounter(&t1);
    double ms_b = ((double)(t1.QuadPart - t0.QuadPart) * 1000.0) / ((double)freq.QuadPart * TRIALS);

    // Profile Config C (AVX2 MT)
    QueryPerformanceCounter(&t0);
    for (int t = 0; t < TRIALS; ++t) gemv_avx2_4row_mt(dst_c, x, w, M_DIM, K_DIM);
    QueryPerformanceCounter(&t1);
    double ms_c = ((double)(t1.QuadPart - t0.QuadPart) * 1000.0) / ((double)freq.QuadPart * TRIALS);

    double gflops = (2.0 * (double)M_DIM * (double)K_DIM) / 1e9;
    double gflops_a = gflops / (ms_a * 1e-3);
    double gflops_b = gflops / (ms_b * 1e-3);
    double gflops_c = gflops / (ms_c * 1e-3);

    printf("================================================================================\n");
    printf("STEP 11 BENCHMARK 5: SINGLE-THREAD VS MULTI-THREAD AVX2 CPU GEMV\n");
    printf("================================================================================\n");
    printf("Matrix Dimensions       : M=%d, K=%d\n", M_DIM, K_DIM);
    printf("Scalar Single-Thread    : %7.4f ms (%6.2f GFLOP/s)\n", ms_a, gflops_a);
    printf("AVX2 4-Row Single-Thread: %7.4f ms (%6.2f GFLOP/s, Speedup: %5.2fx)\n", ms_b, gflops_b, ms_a / ms_b);
    printf("AVX2 4-Row Multi-Thread : %7.4f ms (%6.2f GFLOP/s, Speedup: %5.2fx over scalar, %5.2fx over ST)\n",
           ms_c, gflops_c, ms_a / ms_c, ms_b / ms_c);
    printf("JSON_OUTPUT: {\"benchmark\": \"bench5_cpu_avx2_gemv\", \"scalar_ms\": %.4f, \"avx2_st_ms\": %.4f, \"avx2_mt_ms\": %.4f, \"speedup_st\": %.2f, \"speedup_mt\": %.2f, \"gflops_mt\": %.2f}\n",
           ms_a, ms_b, ms_c, ms_a / ms_b, ms_a / ms_c, gflops_c);
    printf("================================================================================\n");

    _aligned_free(x); _aligned_free(w);
    _aligned_free(dst_a); _aligned_free(dst_b); _aligned_free(dst_c);

    return 0;
}
