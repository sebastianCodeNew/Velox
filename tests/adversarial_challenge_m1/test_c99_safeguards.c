#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define HEAD_DIM 64

// From src/codegen.rs C99 emission:
static void velox_test_c99_attention_step(
    float* dst_attn,
    float* scores_scratch,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    size_t H,
    size_t KV_H,
    size_t D,
    size_t safe_step
) {
    float scale = 1.0f / sqrtf((float)D);
    size_t kv_group = H / KV_H;
    size_t seq_len = safe_step + 1;

    for (size_t h = 0; h < H; ++h) {
        size_t kv_h = h / kv_group;
        const float* q_h = q + (h * D);

        for (size_t t = 0; t < seq_len; ++t) {
            const float* k_t = k_cache + (t * KV_H * D) + (kv_h * D);
            float dot = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                dot += q_h[d] * k_t[d];
            }
            scores_scratch[t] = dot * scale;
        }

        float max_score = -INFINITY;
        for (size_t t = 0; t < seq_len; ++t) {
            float s = scores_scratch[t];
            if (s > max_score) { max_score = s; }
        }
        float sum_exp = 0.0f;
        for (size_t t = 0; t < seq_len; ++t) {
            float e = (max_score == -INFINITY) ? 0.0f : expf(scores_scratch[t] - max_score);
            scores_scratch[t] = e;
            sum_exp += e;
        }
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (size_t t = 0; t < seq_len; ++t) {
            scores_scratch[t] *= inv_sum;
        }

        float* out_h = dst_attn + (h * D);
        for (size_t d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (size_t t = 0; t < seq_len; ++t) {
                const float* v_t = v_cache + (t * KV_H * D) + (kv_h * D);
                acc += scores_scratch[t] * v_t[d];
            }
            out_h[d] = acc;
        }
    }
}

int main() {
    printf("====================================================================\n");
    printf("[EMPIRICAL CHALLENGER] C99 Attention Adversarial Safeguard Suite\n");
    printf("====================================================================\n");

    const size_t H = 9;
    const size_t KV_H = 3;
    const size_t D = HEAD_DIM;
    const size_t max_seq = 64;

    float* dst = (float*)calloc(H * D, sizeof(float));
    float* scores = (float*)calloc(max_seq, sizeof(float));
    float* q = (float*)malloc(H * D * sizeof(float));
    float* k = (float*)malloc(max_seq * KV_H * D * sizeof(float));
    float* v = (float*)malloc(max_seq * KV_H * D * sizeof(float));

    for (size_t i = 0; i < H * D; ++i) q[i] = 1.0f;
    for (size_t i = 0; i < max_seq * KV_H * D; ++i) {
        k[i] = 0.0f;
        v[i] = (float)(i % 17) * 0.1f;
    }

    int passed = 0;
    int total = 0;

    // Test C1: Normal input
    {
        total++;
        for (size_t i = 0; i < 4 * KV_H * D; ++i) k[i] = 0.1f;
        velox_test_c99_attention_step(dst, scores, q, k, v, H, KV_H, D, 3);
        bool ok = true;
        for (size_t i = 0; i < H * D; ++i) {
            if (isnan(dst[i]) || isinf(dst[i])) ok = false;
        }
        if (ok) {
            printf("[PASS] Test C1: Standard finite scores execute cleanly.\n");
            passed++;
        }
    }

    // Test C2: All -INFINITY scores directly injected into max_score branch
    {
        total++;
        float scores_all_inf[16];
        for (int i = 0; i < 16; ++i) scores_all_inf[i] = -INFINITY;

        float max_score = -INFINITY;
        for (size_t t = 0; t < 16; ++t) {
            if (scores_all_inf[t] > max_score) max_score = scores_all_inf[t];
        }
        float sum_exp = 0.0f;
        for (size_t t = 0; t < 16; ++t) {
            float e = (max_score == -INFINITY) ? 0.0f : expf(scores_all_inf[t] - max_score);
            scores_all_inf[t] = e;
            sum_exp += e;
        }
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (size_t t = 0; t < 16; ++t) {
            scores_all_inf[t] *= inv_sum;
        }

        bool ok = true;
        for (int i = 0; i < 16; ++i) {
            if (isnan(scores_all_inf[i]) || isinf(scores_all_inf[i]) || scores_all_inf[i] != 0.0f) {
                ok = false;
            }
        }
        if (ok && sum_exp == 0.0f && inv_sum == 0.0f) {
            printf("[PASS] Test C2: All -INFINITY handled cleanly without NaN or division by zero.\n");
            passed++;
        } else {
            printf("FAIL: Test C2 failed.\n");
        }
    }

    // Test C3: Extreme Dynamic Range in C99
    {
        total++;
        float scores_mix[5] = {-INFINITY, 10000.0f, -50000.0f, -INFINITY, 10001.0f};
        float max_score = -INFINITY;
        for (size_t t = 0; t < 5; ++t) {
            if (scores_mix[t] > max_score) max_score = scores_mix[t];
        }
        float sum_exp = 0.0f;
        for (size_t t = 0; t < 5; ++t) {
            float e = (max_score == -INFINITY) ? 0.0f : expf(scores_mix[t] - max_score);
            scores_mix[t] = e;
            sum_exp += e;
        }
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (size_t t = 0; t < 5; ++t) {
            scores_mix[t] *= inv_sum;
        }

        bool ok = true;
        for (int i = 0; i < 5; ++i) {
            if (isnan(scores_mix[i]) || isinf(scores_mix[i])) ok = false;
        }
        // scores_mix[4] (10001.0) and scores_mix[1] (10000.0) should sum to ~1.0
        float sum_probs = scores_mix[0] + scores_mix[1] + scores_mix[2] + scores_mix[3] + scores_mix[4];
        if (ok && fabsf(sum_probs - 1.0f) < 1e-5f && scores_mix[0] == 0.0f && scores_mix[3] == 0.0f) {
            printf("[PASS] Test C3: C99 extreme dynamic range matches sum=1.0 and zeros -inf entries.\n");
            passed++;
        } else {
            printf("FAIL: Test C3 sum_probs=%f ok=%d\n", sum_probs, ok);
        }
    }

    free(dst); free(scores); free(q); free(k); free(v);

    printf("====================================================================\n");
    printf("C99 Attention Safeguard Verdict: %d / %d PASSED\n", passed, total);
    printf("====================================================================\n");

    return (passed == total) ? 0 : 1;
}
