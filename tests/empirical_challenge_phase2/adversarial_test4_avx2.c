#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>
#include <windows.h>

#define TOLERANCE 1e-4f

// Config B: 4-Row Register-Blocked AVX2+FMA3 GEMV
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

static inline float rand_float(float min_val, float max_val) {
    float scale = (float)rand() / (float)RAND_MAX;
    return min_val + scale * (max_val - min_val);
}

int test_unaligned_dimension(int M, int K, int ptr_offset_elements, const char* desc) {
    // Allocate with padding to allow pointer misalignment testing
    size_t x_count = K + ptr_offset_elements + 16;
    size_t w_count = (size_t)M * K + ptr_offset_elements + 16;
    size_t dst_count = M + ptr_offset_elements + 16;

    float* raw_x = (float*)malloc(x_count * sizeof(float));
    float* raw_w = (float*)malloc(w_count * sizeof(float));
    float* raw_dst = (float*)malloc(dst_count * sizeof(float));
    double* dst_ref = (double*)malloc(M * sizeof(double));

    float* x = raw_x + ptr_offset_elements;
    float* w = raw_w + ptr_offset_elements;
    float* dst = raw_dst + ptr_offset_elements;

    srand(555 + M * 19 + K * 7);
    for (int i = 0; i < K; ++i) x[i] = rand_float(-1.0f, 1.0f);
    for (size_t i = 0; i < (size_t)M * K; ++i) w[i] = rand_float(-1.0f, 1.0f);

    // Double-Precision CPU Reference Oracle
    for (int r = 0; r < M; ++r) {
        double acc = 0.0;
        for (int c = 0; c < K; ++c) {
            acc += (double)x[c] * (double)w[(size_t)r * K + c];
        }
        dst_ref[r] = acc;
    }

    // Execute AVX2+FMA3 register-blocked GEMV
    gemv_avx2_4row(dst, x, w, M, K);

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (int r = 0; r < M; ++r) {
        double abs_err = fabs((double)dst[r] - dst_ref[r]);
        double rel_err = (fabs(dst_ref[r]) > 1e-12) ? (abs_err / fabs(dst_ref[r])) : abs_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    bool pass = (max_abs_err <= TOLERANCE) || (max_rel_err <= 1e-5);
    printf("[%s] M=%-4d K=%-4d PtrOffset=%d | MaxAbsErr: %.2e | MaxRelErr: %.2e | Status: %s\n",
           desc, M, K, ptr_offset_elements, max_abs_err, max_rel_err, pass ? "PASSED" : "FAILED");

    free(raw_x);
    free(raw_w);
    free(raw_dst);
    free(dst_ref);

    return pass ? 0 : 1;
}

int main() {
    printf("================================================================================\n");
    printf("ADVERSARIAL CHALLENGE TEST 4: AVX2+FMA3 REGISTER BLOCKING UNALIGNED STRESS\n");
    printf("Host CPU: Intel Core i5-12500H AVX2+FMA3\n");
    printf("================================================================================\n");

    int failures = 0;

    // 1. Extreme Small Boundary Dimensions
    printf("\n--- Section 1: Extreme Boundary Matrix Sizes ---\n");
    failures += test_unaligned_dimension(1, 1, 0, "M1_K1");
    failures += test_unaligned_dimension(1, 7, 0, "M1_K7");
    failures += test_unaligned_dimension(1, 8, 0, "M1_K8");
    failures += test_unaligned_dimension(1, 9, 0, "M1_K9");
    failures += test_unaligned_dimension(3, 1, 0, "M3_K1");
    failures += test_unaligned_dimension(4, 1, 0, "M4_K1");
    failures += test_unaligned_dimension(5, 1, 0, "M5_K1");
    failures += test_unaligned_dimension(3, 7, 0, "M3_K7");
    failures += test_unaligned_dimension(5, 9, 0, "M5_K9");

    // 2. Unaligned M (Residual Rows: M % 4 != 0) with Standard K=576
    printf("\n--- Section 2: Residual Rows (M %% 4 != 0) with K=576 ---\n");
    failures += test_unaligned_dimension(255, 576, 0, "M255_K576 (M%4=3)");
    failures += test_unaligned_dimension(257, 576, 0, "M257_K576 (M%4=1)");
    failures += test_unaligned_dimension(258, 576, 0, "M258_K576 (M%4=2)");
    failures += test_unaligned_dimension(1535, 576, 0, "M1535_K576 (M%4=3)");
    failures += test_unaligned_dimension(1537, 576, 0, "M1537_K576 (M%4=1)");

    // 3. Unaligned K (Tail Handling: K % 8 != 0) with Standard M=1536
    printf("\n--- Section 3: Residual Columns (K %% 8 != 0) with M=1536 ---\n");
    failures += test_unaligned_dimension(1536, 575, 0, "M1536_K575 (K%8=7)");
    failures += test_unaligned_dimension(1536, 577, 0, "M1536_K577 (K%8=1)");
    failures += test_unaligned_dimension(1536, 578, 0, "M1536_K578 (K%8=2)");
    failures += test_unaligned_dimension(1536, 579, 0, "M1536_K579 (K%8=3)");
    failures += test_unaligned_dimension(1536, 581, 0, "M1536_K581 (K%8=5)");

    // 4. Simultaneously Unaligned M and K
    printf("\n--- Section 4: Simultaneously Unaligned M and K ---\n");
    failures += test_unaligned_dimension(7, 13, 0, "M7_K13");
    failures += test_unaligned_dimension(17, 31, 0, "M17_K31");
    failures += test_unaligned_dimension(127, 255, 0, "M127_K255");
    failures += test_unaligned_dimension(577, 575, 0, "M577_K575");
    failures += test_unaligned_dimension(1535, 577, 0, "M1535_K577");

    // 5. Unaligned Pointer Offsets (Testing _mm256_loadu_ps with 4-byte, 8-byte, 12-byte unaligned offsets)
    printf("\n--- Section 5: Unaligned Pointer Memory Offsets ---\n");
    failures += test_unaligned_dimension(256, 576, 1, "OFFSET_1_ELEM (4-byte)");
    failures += test_unaligned_dimension(256, 576, 2, "OFFSET_2_ELEM (8-byte)");
    failures += test_unaligned_dimension(256, 576, 3, "OFFSET_3_ELEM (12-byte)");
    failures += test_unaligned_dimension(577, 575, 3, "UNALIGNED_M_K_WITH_OFFSET_3");

    printf("\n================================================================================\n");
    printf("TEST 4 TOTAL AUDIT SUMMARY: Total Failures = %d / 28\n", failures);
    printf("================================================================================\n");

    return (failures == 0) ? 0 : 1;
}
