/* ========================================================================== */
/* Velox Standalone Zero-Allocation C99 Serving Module                        */
/* Model:             SmolLM_135M                                             */
/* Target Device:     nvidia_rtx_3050_laptop (sm_86)                             */
/* Memory Planner:    Velox Static Memory Planner (256-Byte Aligned Arena)    */
/* Dynamic Heap Ops:  0 (Strictly Static Arena Offset Addressing)             */
/* ========================================================================== */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

/* --- 1. Compile-Time Model & Memory Arena Constants ----------------------- */
#define VELOX_MODEL_HIDDEN_DIM               576ULL
#define VELOX_MODEL_INTERMEDIATE_DIM         1536ULL
#define VELOX_MODEL_NUM_LAYERS               30ULL
#define VELOX_MODEL_NUM_HEADS                9ULL
#define VELOX_MODEL_NUM_KV_HEADS             3ULL
#define VELOX_MODEL_HEAD_DIM                 64ULL
#define VELOX_MODEL_VOCAB_SIZE               49152ULL
#define VELOX_MODEL_MAX_SEQ_LEN              2048ULL
#define VELOX_MODEL_NUM_EXPERTS              256ULL
#define VELOX_MODEL_ROPE_THETA               10000.0f
#define VELOX_ARENA_ALIGNMENT                256ULL

#define VELOX_WEIGHT_ARENA_BYTES             651306240ULL
#define VELOX_KV_CACHE_POOL_BYTES            377491456ULL
#define VELOX_ACTIVATION_ARENA_BYTES         424676608ULL
#define VELOX_SYSTEM_RESERVE_BYTES           268435456ULL
#define VELOX_TOTAL_REQUIRED_VRAM_BYTES      1721909760ULL
#define VELOX_SAFE_VRAM_BUDGET_BYTES         4080218931ULL

#define VELOX_W_OFFSET_W_EMBED                        0ULL
#define VELOX_W_OFFSET_W_Q                            113246208ULL
#define VELOX_W_OFFSET_W_K                            153059328ULL
#define VELOX_W_OFFSET_W_V                            166330368ULL
#define VELOX_W_OFFSET_W_OUT                          179601408ULL
#define VELOX_W_OFFSET_W_GATE                         219414528ULL
#define VELOX_W_OFFSET_W_UP                           325582848ULL
#define VELOX_W_OFFSET_W_DOWN                         431751168ULL
#define VELOX_W_OFFSET_W_ATTN_NORM                    537919488ULL
#define VELOX_W_OFFSET_W_MLP_NORM                     537988608ULL
#define VELOX_W_OFFSET_W_FINAL_NORM                   538057728ULL
#define VELOX_W_OFFSET_W_HEAD                         538060032ULL

/* --- 2. Velox Zero-Allocation Static Arena Context ------------------------ */
typedef struct VeloxArenaContext {
    const uint8_t* weight_arena;      /* @vram_immutable mmap base */
    uint8_t*       kv_cache_pool;     /* @paged_vram static pool   */
    uint8_t*       activation_arena;  /* @bump_sram 256B scratch   */
    uint32_t*      host_pinned_ring;  /* @host_pinned SPSC ring    */
} VeloxArenaContext;

#define VELOX_ACT_PTR(ctx, offset) ((float*)((uint8_t*)((ctx)->activation_arena) + (offset)))
#define VELOX_W_PTR(ctx, offset)   ((const float*)((const uint8_t*)((ctx)->weight_arena) + (offset)))

/* --- 3. Exact Numerical Kernels (Zero Dynamic Allocations) ---------------- */
static inline void velox_c99_embed_f32(
    float* dst, uint32_t token_id, const float* w_embed, size_t hidden_dim
) {
    const float* row = w_embed + ((size_t)token_id * hidden_dim);
    for (size_t i = 0; i < hidden_dim; ++i) {
        dst[i] = row[i];
    }
}

static inline void velox_c99_rmsnorm_f32(
    float* dst, const float* x, const float* weight, size_t dim, float eps
) {
    float ss = 0.0f;
    for (size_t i = 0; i < dim; ++i) { ss += x[i] * x[i]; }
    float inv_rms = 1.0f / sqrtf((ss / (float)dim) + eps);
    for (size_t i = 0; i < dim; ++i) { dst[i] = (x[i] * inv_rms) * weight[i]; }
}

static inline void velox_c99_matmul_vec_f32(
    float* dst, const float* x, const float* w, size_t in_dim, size_t out_dim
) {
    for (size_t o = 0; o < out_dim; ++o) {
        const float* w_row = w + (o * in_dim);
        float acc = 0.0f;
        size_t chunks = in_dim / 4;
        for (size_t c = 0; c < chunks; ++c) {
            size_t base = c * 4;
            acc += x[base] * w_row[base]
                + x[base + 1] * w_row[base + 1]
                + x[base + 2] * w_row[base + 2]
                + x[base + 3] * w_row[base + 3];
        }
        for (size_t k = chunks * 4; k < in_dim; ++k) {
            acc += x[k] * w_row[k];
        }
        dst[o] = acc;
    }
}

static inline void velox_c99_rope_rotate_half_in_place_f32(
    float* tensor, size_t num_heads, size_t head_dim, size_t pos, float theta
) {
    size_t half = head_dim / 2;
    float pos_f = (float)pos;
    for (size_t h = 0; h < num_heads; ++h) {
        float* head_slice = tensor + (h * head_dim);
        for (size_t j = 0; j < half; ++j) {
            float freq = 1.0f / powf(theta, (float)(2 * j) / (float)head_dim);
            float angle = pos_f * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float x1 = head_slice[j];
            float x2 = head_slice[j + half];
            head_slice[j] = x1 * cos_a - x2 * sin_a;
            head_slice[j + half] = x2 * cos_a + x1 * sin_a;
        }
    }
}

static inline void velox_c99_gqa_attention_step_f32(
    float* attn_out, float* scores_scratch, const float* q,
    const float* k_cache_layer, const float* v_cache_layer,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, size_t pos
) {
    size_t kv_group_size = num_q_heads / num_kv_heads;
    size_t kv_stride = num_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t seq_len = pos + 1;
    for (size_t h = 0; h < num_q_heads; ++h) {
        size_t kv_h = h / kv_group_size;
        const float* q_head = q + (h * head_dim);
        float max_score = -INFINITY;
        for (size_t t = 0; t < seq_len; ++t) {
            size_t k_offset = t * kv_stride + kv_h * head_dim;
            const float* k_head = k_cache_layer + k_offset;
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                dot += q_head[d] * k_head[d];
            }
            float s = dot * scale;
            scores_scratch[t] = s;
            if (s > max_score) max_score = s;
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
        float* out_head = attn_out + (h * head_dim);
        for (size_t d = 0; d < head_dim; ++d) {
            out_head[d] = 0.0f;
        }
        for (size_t t = 0; t < seq_len; ++t) {
            float prob = scores_scratch[t];
            size_t v_offset = t * kv_stride + kv_h * head_dim;
            const float* v_head = v_cache_layer + v_offset;
            for (size_t d = 0; d < head_dim; ++d) {
                out_head[d] += prob * v_head[d];
            }
        }
    }
}

/* SOTA Fused Kernel 1: Pre-Attn RMSNorm + QKV + RoPE + KV Write + GQA Attention */
static void velox_c99_fused_norm_qkv_rope_attn(
    float* dst_attn, const float* input_hidden,
    const float* w_norm, const float* w_q, const float* w_k, const float* w_v,
    float* kv_pool, size_t layer, size_t step, float eps, float theta
) {
    const size_t H = VELOX_MODEL_NUM_HEADS;
    const size_t KV_H = VELOX_MODEL_NUM_KV_HEADS;
    const size_t D = VELOX_MODEL_HEAD_DIM;
    const size_t dim = VELOX_MODEL_HIDDEN_DIM;
    const size_t kv_dim = KV_H * D;
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    float norm_x[VELOX_MODEL_HIDDEN_DIM];
    velox_c99_rmsnorm_f32(norm_x, input_hidden, w_norm + (layer * dim), dim, eps);

    float q_buf[VELOX_MODEL_NUM_HEADS * VELOX_MODEL_HEAD_DIM];
    velox_c99_matmul_vec_f32(q_buf, norm_x, w_q + (layer * dim * (H * D)), dim, H * D);

    size_t layer_kv_stride = 2ULL * max_s * kv_dim;
    float* k_layer = kv_pool + (layer * layer_kv_stride);
    float* v_layer = k_layer + (max_s * kv_dim);
    size_t safe_step = (step < max_s) ? step : (max_s - 1);
    float* k_slot = k_layer + (safe_step * kv_dim);
    float* v_slot = v_layer + (safe_step * kv_dim);
    velox_c99_matmul_vec_f32(k_slot, norm_x, w_k + (layer * dim * kv_dim), dim, kv_dim);
    velox_c99_matmul_vec_f32(v_slot, norm_x, w_v + (layer * dim * kv_dim), dim, kv_dim);

    velox_c99_rope_rotate_half_in_place_f32(q_buf, H, D, safe_step, theta);
    velox_c99_rope_rotate_half_in_place_f32(k_slot, KV_H, D, safe_step, theta);

    float scores_scratch[VELOX_MODEL_MAX_SEQ_LEN];
    velox_c99_gqa_attention_step_f32(
        dst_attn, scores_scratch, q_buf,
        k_layer, v_layer, H, KV_H, D, safe_step
    );
}

/* SOTA Fused Kernel 2: OutProj + Residual + Post-Norm + SwiGLU + Residual */
static void velox_c99_fused_out_res_swiglu_res(
    float* dst_hidden, const float* attn_in, const float* residual_in,
    const float* w_out, const float* w_post_norm,
    const float* w_gate, const float* w_up, const float* w_down,
    size_t layer, float eps
) {
    const size_t dim = VELOX_MODEL_HIDDEN_DIM;
    const size_t inter_dim = VELOX_MODEL_INTERMEDIATE_DIM;
    float mid[VELOX_MODEL_HIDDEN_DIM];
    velox_c99_matmul_vec_f32(mid, attn_in, w_out + (layer * dim * dim), dim, dim);
    for (size_t i = 0; i < dim; ++i) { mid[i] += residual_in[i]; }

    float norm_mid[VELOX_MODEL_HIDDEN_DIM];
    velox_c99_rmsnorm_f32(norm_mid, mid, w_post_norm + (layer * dim), dim, eps);

    float gate_buf[VELOX_MODEL_INTERMEDIATE_DIM];
    float up_buf[VELOX_MODEL_INTERMEDIATE_DIM];
    velox_c99_matmul_vec_f32(gate_buf, norm_mid, w_gate + (layer * dim * inter_dim), dim, inter_dim);
    velox_c99_matmul_vec_f32(up_buf, norm_mid, w_up + (layer * dim * inter_dim), dim, inter_dim);

    float act_buf[VELOX_MODEL_INTERMEDIATE_DIM];
    for (size_t i = 0; i < inter_dim; ++i) {
        float g = gate_buf[i];
        float silu = g / (1.0f + expf(-g));
        act_buf[i] = silu * up_buf[i];
    }

    float down_buf[VELOX_MODEL_HIDDEN_DIM];
    velox_c99_matmul_vec_f32(down_buf, act_buf, w_down + (layer * inter_dim * dim), inter_dim, dim);
    for (size_t i = 0; i < dim; ++i) { dst_hidden[i] = mid[i] + down_buf[i]; }
}

/* SOTA Absorbed MLA Kernel: Latent Decode without KV Decompression */
static void velox_c99_fused_mla_absorbed_attn(
    float* dst_hidden, const float* input_hidden,
    const float* w_norm, const float* w_dkv, const float* w_absorbed_q, const float* w_absorbed_o,
    float* kv_latent_pool, size_t layer, size_t step, float eps
) {
    const size_t dim = VELOX_MODEL_HIDDEN_DIM;
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    const size_t d_c = 512;
    float norm_x[VELOX_MODEL_HIDDEN_DIM];
    velox_c99_rmsnorm_f32(norm_x, input_hidden, w_norm + (layer * dim), dim, eps);

    float* layer_latent = kv_latent_pool + (layer * max_s * d_c);
    size_t safe_step = (step < max_s) ? step : (max_s - 1);
    float* slot = layer_latent + (safe_step * d_c);
    velox_c99_matmul_vec_f32(slot, norm_x, w_dkv + (layer * dim * d_c), dim, d_c);

    float q_abs[512];
    velox_c99_matmul_vec_f32(q_abs, norm_x, w_absorbed_q + (layer * dim * d_c), dim, d_c);

    float scores[VELOX_MODEL_MAX_SEQ_LEN];
    float scale = 1.0f / sqrtf(128.0f);
    float max_s_val = -INFINITY;
    size_t seq_len = safe_step + 1;
    for (size_t t = 0; t < seq_len; ++t) {
        const float* k_vec = layer_latent + (t * d_c);
        float dot = 0.0f;
        for (size_t d = 0; d < d_c; ++d) { dot += q_abs[d] * k_vec[d]; }
        float s = dot * scale;
        scores[t] = s;
        if (s > max_s_val) max_s_val = s;
    }
    float sum_exp = 0.0f;
    for (size_t t = 0; t < seq_len; ++t) {
        float e = (max_s_val == -INFINITY) ? 0.0f : expf(scores[t] - max_s_val);
        scores[t] = e;
        sum_exp += e;
    }
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (size_t t = 0; t < seq_len; ++t) { scores[t] *= inv_sum; }

    float context[512];
    for (size_t d = 0; d < d_c; ++d) { context[d] = 0.0f; }
    for (size_t t = 0; t < seq_len; ++t) {
        float prob = scores[t];
        const float* v_vec = layer_latent + (t * d_c);
        for (size_t d = 0; d < d_c; ++d) { context[d] += prob * v_vec[d]; }
    }

    float o_proj[VELOX_MODEL_HIDDEN_DIM];
    velox_c99_matmul_vec_f32(o_proj, context, w_absorbed_o + (layer * d_c * dim), d_c, dim);
    for (size_t i = 0; i < dim; ++i) { dst_hidden[i] = input_hidden[i] + o_proj[i]; }
}

static inline void velox_c99_moe_gate_topk(
    uint32_t* top_indices, float* top_weights, const float* input,
    const float* w_gate, size_t layer, size_t hidden_dim, size_t num_experts, size_t top_k
) {
    const float* l_gate = w_gate + (layer * num_experts * hidden_dim);
    size_t safe_experts = (num_experts <= 256) ? num_experts : 256;
    float logits[256];
    for (size_t e = 0; e < safe_experts; ++e) {
        const float* row = l_gate + (e * hidden_dim);
        float acc = 0.0f;
        for (size_t k = 0; k < hidden_dim; ++k) { acc += input[k] * row[k]; }
        logits[e] = acc;
    }
    for (size_t k = 0; k < top_k; ++k) {
        size_t best_idx = 0;
        float best_val = -INFINITY;
        for (size_t e = 0; e < safe_experts; ++e) {
            int already = 0;
            for (size_t prev = 0; prev < k; ++prev) {
                if (top_indices[prev] == (uint32_t)e) { already = 1; break; }
            }
            if (!already && logits[e] > best_val) {
                best_val = logits[e];
                best_idx = e;
            }
        }
        top_indices[k] = (uint32_t)best_idx;
    }
    float max_l = logits[top_indices[0]];
    for (size_t k = 1; k < top_k; ++k) {
        if (logits[top_indices[k]] > max_l) max_l = logits[top_indices[k]];
    }
    float sum_exp = 0.0f;
    for (size_t k = 0; k < top_k; ++k) {
        float e = expf(logits[top_indices[k]] - max_l);
        top_weights[k] = e;
        sum_exp += e;
    }
    float inv_sum = 1.0f / sum_exp;
    for (size_t k = 0; k < top_k; ++k) { top_weights[k] *= inv_sum; }
}

static inline void velox_c99_moe_dispatch_topk(
    float* dst, const float* input, const uint32_t* indices, const float* weights,
    const float* w_experts, size_t layer, size_t hidden_dim, size_t expert_inter_dim, size_t num_experts, size_t top_k, float capacity_factor
) {
    for (size_t i = 0; i < hidden_dim; ++i) { dst[i] = 0.0f; }
    size_t expert_size = expert_inter_dim * hidden_dim;
    size_t max_dispatched = (size_t)(top_k * capacity_factor);
    if (max_dispatched > top_k) max_dispatched = top_k;
    float inter_buf[VELOX_MODEL_INTERMEDIATE_DIM];

    for (size_t k = 0; k < max_dispatched; ++k) {
        uint32_t exp_id = indices[k];
        float w = weights[k];
        const float* exp_w = w_experts + (layer * num_experts * expert_size) + ((size_t)exp_id * expert_size);

        /* Expert up-projection with GEMV: inter = silu(input @ exp_w) */
        for (size_t j = 0; j < expert_inter_dim; ++j) {
            float acc = 0.0f;
            for (size_t i = 0; i < hidden_dim; ++i) {
                acc += input[i] * exp_w[i * expert_inter_dim + j];
            }
            float silu = acc / (1.0f + expf(-acc));
            inter_buf[j] = silu;
        }

        /* Expert down-projection with GEMV and weighted blending: dst += w * (inter @ exp_w^T) */
        for (size_t o = 0; o < hidden_dim; ++o) {
            float acc = 0.0f;
            for (size_t j = 0; j < expert_inter_dim; ++j) {
                acc += inter_buf[j] * exp_w[o * expert_inter_dim + j];
            }
            dst[o] += w * acc;
        }
    }
}

static inline void velox_c99_residual_add_f32(float* dst, const float* lhs, const float* rhs, size_t dim) {
    for (size_t i = 0; i < dim; ++i) { dst[i] = lhs[i] + rhs[i]; }
}

static inline void velox_c99_silu_f32(float* dst, const float* x, size_t dim) {
    for (size_t i = 0; i < dim; ++i) {
        float val = x[i];
        dst[i] = val / (1.0f + expf(-val));
    }
}

static inline void velox_c99_elementwise_mul_f32(float* dst, const float* lhs, const float* rhs, size_t dim) {
    for (size_t i = 0; i < dim; ++i) { dst[i] = lhs[i] * rhs[i]; }
}

static inline void velox_c99_slice_last_token(float* dst, const float* src, size_t dim) {
    for (size_t i = 0; i < dim; ++i) { dst[i] = src[i]; }
}

static inline uint32_t velox_c99_sample_argmax_f32(const float* logits, size_t vocab_size) {
    uint32_t best_id = 0;
    float best_val = logits[0];
    for (size_t v = 1; v < vocab_size; ++v) {
        if (logits[v] > best_val) { best_val = logits[v]; best_id = (uint32_t)v; }
    }
    return best_id;
}

static inline uint32_t velox_c99_sample_top_p_f32(
    const float* logits, size_t vocab_size, float temperature, float top_p, float random_coin
) {
    if (temperature <= 1e-5f || top_p <= 0.0f) {
        return velox_c99_sample_argmax_f32(logits, vocab_size);
    }
    float inv_temp = 1.0f / temperature;
    float max_l = logits[0] * inv_temp;
    for (size_t v = 1; v < vocab_size; ++v) {
        float val = logits[v] * inv_temp;
        if (val > max_l) max_l = val;
    }
    float sum_exp = 0.0f;
    for (size_t v = 0; v < vocab_size; ++v) {
        sum_exp += expf(logits[v] * inv_temp - max_l);
    }
    float target = random_coin * top_p * sum_exp;
    float cum = 0.0f;
    for (size_t v = 0; v < vocab_size; ++v) {
        cum += expf(logits[v] * inv_temp - max_l);
        if (cum >= target) return (uint32_t)v;
    }
    return 0U;
}

/* --- 4. Monomorphized Execution DAG Entry Points -------------------------- */
uint32_t velox_SmolLM_135M_prefill(VeloxArenaContext* ctx, uint32_t token_in, size_t step, uint64_t active_mask) {
    /* Basic Block: entry */
    velox_c99_embed_f32(VELOX_ACT_PTR(ctx, 301989888ULL), token_in, VELOX_W_PTR(ctx, 0ULL), VELOX_MODEL_HIDDEN_DIM);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 0ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 0ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 1ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 1ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 2ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 2ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 3ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 3ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 4ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 4ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 5ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 5ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 6ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 6ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 7ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 7ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 8ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 8ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 9ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 9ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 10ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 10ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 11ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 11ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 12ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 12ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 13ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 13ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 14ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 14ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 15ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 15ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 16ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 16ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 17ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 17ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 18ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 18ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 19ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 19ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 20ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 20ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 21ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 21ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 22ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 22ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 23ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 23ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 24ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 24ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 25ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 25ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 26ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 26ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 27ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 27ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 28ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 28ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 29ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 29ULL, 0.000010f);
    velox_c99_rmsnorm_f32(VELOX_ACT_PTR(ctx, 786432ULL), VELOX_ACT_PTR(ctx, 301989888ULL), VELOX_W_PTR(ctx, 538057728ULL) + (0ULL * VELOX_MODEL_HIDDEN_DIM), VELOX_MODEL_HIDDEN_DIM, 0.000010f);
    velox_c99_matmul_vec_f32(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 786432ULL), VELOX_W_PTR(ctx, 538060032ULL), 576ULL, 49152ULL);
    uint32_t sampled_333 = velox_c99_sample_argmax_f32(VELOX_ACT_PTR(ctx, 0ULL), VELOX_MODEL_VOCAB_SIZE);
    if (ctx->host_pinned_ring) { ctx->host_pinned_ring[step & 1023ULL] = sampled_333; }
    return sampled_333;
}

uint32_t velox_SmolLM_135M_decode(VeloxArenaContext* ctx, uint32_t token_in, size_t step, uint64_t active_mask) {
    /* Single-Graph Continuous Batching Active Mask Predication */
    if ((active_mask & 1ULL) == 0ULL) { return 0U; }
    /* Basic Block: entry */
    velox_c99_embed_f32(VELOX_ACT_PTR(ctx, 147456ULL), token_in, VELOX_W_PTR(ctx, 0ULL), VELOX_MODEL_HIDDEN_DIM);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 0ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 0ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 1ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 1ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 2ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 2ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 3ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 3ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 4ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 4ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 5ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 5ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 6ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 6ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 7ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 7ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 8ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 8ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 9ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 9ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 10ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 10ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 11ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 11ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 12ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 12ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 13ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 13ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 14ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 14ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 15ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 15ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 16ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 16ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 17ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 17ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 18ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 18ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 19ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 19ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 20ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 20ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 21ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 21ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 22ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 22ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 23ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 23ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 24ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 24ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 25ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 25ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 26ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 26ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 27ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 27ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 28ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 28ULL, 0.000010f);
    velox_c99_fused_norm_qkv_rope_attn(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 537919488ULL), VELOX_W_PTR(ctx, 113246208ULL), VELOX_W_PTR(ctx, 153059328ULL), VELOX_W_PTR(ctx, 166330368ULL), (float*)ctx->kv_cache_pool, 29ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA);
    velox_c99_fused_out_res_swiglu_res(VELOX_ACT_PTR(ctx, 147456ULL), VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 179601408ULL), VELOX_W_PTR(ctx, 537988608ULL), VELOX_W_PTR(ctx, 219414528ULL), VELOX_W_PTR(ctx, 325582848ULL), VELOX_W_PTR(ctx, 431751168ULL), 29ULL, 0.000010f);
    velox_c99_rmsnorm_f32(VELOX_ACT_PTR(ctx, 1572864ULL), VELOX_ACT_PTR(ctx, 147456ULL), VELOX_W_PTR(ctx, 538057728ULL) + (0ULL * VELOX_MODEL_HIDDEN_DIM), VELOX_MODEL_HIDDEN_DIM, 0.000010f);
    velox_c99_matmul_vec_f32(VELOX_ACT_PTR(ctx, 0ULL), VELOX_ACT_PTR(ctx, 1572864ULL), VELOX_W_PTR(ctx, 538060032ULL), 576ULL, 49152ULL);
    uint32_t sampled_333 = velox_c99_sample_argmax_f32(VELOX_ACT_PTR(ctx, 0ULL), VELOX_MODEL_VOCAB_SIZE);
    if (ctx->host_pinned_ring) { ctx->host_pinned_ring[step & 1023ULL] = sampled_333; }
    return sampled_333;
}

