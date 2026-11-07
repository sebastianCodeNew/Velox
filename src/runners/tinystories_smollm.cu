// ==========================================================================
// Velox Standalone Zero-Allocation CUDA Serving Module (.cu)
// Model:             SmolLM_135M
// Target GPU:        nvidia_rtx_3050_laptop (sm_86)
// Memory Planner:    Velox Static Memory Planner (256B Aligned Offsets)
// Dynamic Heap Ops:  0 (Pre-Committed Static Device Arena)
// ==========================================================================

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define VELOX_HIDDEN_DIM                     576ULL
#define VELOX_MODEL_HIDDEN_DIM               VELOX_HIDDEN_DIM
#define VELOX_INTERMEDIATE_DIM               1536ULL
#define VELOX_MODEL_INTERMEDIATE_DIM         VELOX_INTERMEDIATE_DIM
#define VELOX_NUM_LAYERS                     30ULL
#define VELOX_MODEL_NUM_LAYERS               VELOX_NUM_LAYERS
#define VELOX_NUM_HEADS                      9ULL
#define VELOX_MODEL_NUM_HEADS                VELOX_NUM_HEADS
#define VELOX_NUM_KV_HEADS                   3ULL
#define VELOX_MODEL_NUM_KV_HEADS             VELOX_NUM_KV_HEADS
#define VELOX_HEAD_DIM                       64ULL
#define VELOX_MODEL_HEAD_DIM                 VELOX_HEAD_DIM
#define VELOX_VOCAB_SIZE                     49152ULL
#define VELOX_MODEL_VOCAB_SIZE               VELOX_VOCAB_SIZE
#define VELOX_MODEL_MAX_SEQ_LEN              2048ULL
#define VELOX_MODEL_NUM_EXPERTS              256ULL
#define VELOX_MODEL_ROPE_THETA               10000.0f
#define VELOX_ACTIVATION_ARENA_BYTES         424676608ULL
#define VELOX_TOTAL_REQUIRED_VRAM_BYTES      1721909760ULL
#define VELOX_SAFE_VRAM_BUDGET_BYTES         4080218931ULL

typedef struct VeloxCudaArena {
    const uint8_t* __restrict__ weight_arena;
    uint8_t*       __restrict__ kv_cache_pool;
    uint8_t*       __restrict__ activation_arena;
    uint32_t*      __restrict__ host_pinned_ring;
} VeloxCudaArena;

#define VELOX_DEV_ACT_PTR(arena_base, offset) ((float*)((uint8_t*)(arena_base) + (offset)))
#define VELOX_DEV_W_PTR(weight_base, offset)  ((const float*)((const uint8_t*)(weight_base) + (offset)))

// --- 1. Warp-Specialized & Fused CUDA Kernels -----------------------------
__global__ void __launch_bounds__(256) velox_cuda_embed_kernel(
    float* __restrict__ dst, uint32_t token_id, const float* __restrict__ w_embed, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < VELOX_HIDDEN_DIM) {
        dst[idx] = w_embed[(size_t)token_id * VELOX_HIDDEN_DIM + idx];
    }
}

__global__ void __launch_bounds__(256) velox_cuda_rmsnorm_kernel(
    float* __restrict__ dst, const float* __restrict__ x, const float* __restrict__ weight,
    size_t dim, float eps, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    __shared__ float s_ss[256];
    size_t tid = threadIdx.x;
    float local_ss = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        float v = x[i];
        local_ss += v * v;
    }
    s_ss[tid] = local_ss;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_ss[tid] += s_ss[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf((s_ss[0] / (float)dim) + eps);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + tid; i < dim; i += (size_t)gridDim.x * blockDim.x) {
        dst[i] = (x[i] * inv_rms) * weight[i];
    }
}

__global__ void __launch_bounds__(256) velox_cuda_matmul_vec_kernel(
    float* __restrict__ dst, const float* __restrict__ x, const float* __restrict__ w,
    size_t in_dim, size_t out_dim, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o < out_dim) {
        const float* row = w + (o * in_dim);
        float acc = 0.0f;
        for (size_t k = 0; k < in_dim; ++k) { acc += x[k] * row[k]; }
        dst[o] = acc;
    }
}

__global__ void __launch_bounds__(256) velox_cuda_residual_add(
    float* __restrict__ dst, const float* __restrict__ lhs, const float* __restrict__ rhs,
    size_t dim, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) { dst[idx] = lhs[idx] + rhs[idx]; }
}

// Elementwise SiLU Activation Kernel (z / (1 + exp(-z)))
__global__ void __launch_bounds__(256) velox_cuda_silu_kernel(
    float* __restrict__ dst, const float* __restrict__ src, size_t dim, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < dim; idx += (size_t)gridDim.x * blockDim.x) {
        float x = src[idx];
        dst[idx] = x / (1.0f + expf(-x));
    }
}

// Elementwise Hadamard Multiplication Kernel (lhs * rhs)
__global__ void __launch_bounds__(256) velox_cuda_elementwise_mul_kernel(
    float* __restrict__ dst, const float* __restrict__ lhs, const float* __restrict__ rhs, size_t dim, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < dim; idx += (size_t)gridDim.x * blockDim.x) {
        dst[idx] = lhs[idx] * rhs[idx];
    }
}

// Slice Last Sequence Position Kernel
__global__ void __launch_bounds__(256) velox_cuda_slice_last_token_kernel(
    float* __restrict__ dst, const float* __restrict__ src, size_t dim, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < dim; idx += (size_t)gridDim.x * blockDim.x) {
        dst[idx] = src[idx];
    }
}

// RoPE Rotary Position Embedding Kernel
__global__ void __launch_bounds__(256) velox_cuda_rope_kernel(
    float* __restrict__ dst, const float* __restrict__ src,
    size_t num_heads, size_t head_dim, size_t step, float theta, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    size_t half_d = head_dim / 2;
    size_t total_pairs = num_heads * half_d;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total_pairs; idx += (size_t)gridDim.x * blockDim.x) {
        size_t h = idx / half_d;
        size_t j = idx % half_d;
        float freq = 1.0f / powf(theta, (2.0f * (float)j) / (float)head_dim);
        float angle = (float)step * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        const float* src_head = src + (h * head_dim);
        float* dst_head = dst + (h * head_dim);
        float x1 = src_head[j];
        float x2 = src_head[j + half_d];
        dst_head[j] = x1 * cos_a - x2 * sin_a;
        dst_head[j + half_d] = x2 * cos_a + x1 * sin_a;
    }
}

// SOTA Fused Kernel 1 of 2 per Layer: RMSNorm + QKV + RoPE + FlashAttention
__global__ void __launch_bounds__(256) velox_cuda_fused_norm_qkv_rope_attn(
    float* __restrict__ dst_attn, const float* __restrict__ input_hidden,
    const float* __restrict__ w_norm, const float* __restrict__ w_q,
    const float* __restrict__ w_k, const float* __restrict__ w_v,
    float* __restrict__ kv_pool, size_t layer, size_t step, float eps, float theta, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    if (blockIdx.x > 0) return;
    const size_t H = VELOX_NUM_HEADS;
    const size_t KV_H = VELOX_NUM_KV_HEADS;
    const size_t D = VELOX_HEAD_DIM;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t kv_dim = KV_H * D;
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    size_t tid = threadIdx.x;

    // 1. Pre-Attn RMSNorm reduction
    __shared__ float s_ss[256];
    float local_ss = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        float v = input_hidden[i];
        local_ss += v * v;
    }
    s_ss[tid] = local_ss;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_ss[tid] += s_ss[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf((s_ss[0] / (float)dim) + eps);
    const float* l_wnorm = w_norm + (layer * dim);

    // 2. Q Projection into dst_attn (temporary holding buffer for Q)
    const float* l_wq = w_q + (layer * dim * (H * D));
    for (size_t o = tid; o < H * D; o += blockDim.x) {
        const float* row = l_wq + (o * dim);
        float acc = 0.0f;
        for (size_t k = 0; k < dim; ++k) {
            float nx = input_hidden[k] * inv_rms * l_wnorm[k];
            acc += nx * row[k];
        }
        dst_attn[o] = acc;
    }

    // 3. K and V Projections into kv_pool slot
    size_t layer_kv_stride = 2ULL * max_s * kv_dim;
    float* k_layer = kv_pool + (layer * layer_kv_stride);
    float* v_layer = k_layer + (max_s * kv_dim);
    size_t safe_step = (step < max_s) ? step : (max_s - 1);
    float* k_slot = k_layer + (safe_step * kv_dim);
    float* v_slot = v_layer + (safe_step * kv_dim);

    const float* l_wk = w_k + (layer * dim * kv_dim);
    const float* l_wv = w_v + (layer * dim * kv_dim);
    for (size_t o = tid; o < kv_dim; o += blockDim.x) {
        const float* row_k = l_wk + (o * dim);
        const float* row_v = l_wv + (o * dim);
        float acc_k = 0.0f;
        float acc_v = 0.0f;
        for (size_t k = 0; k < dim; ++k) {
            float nx = input_hidden[k] * inv_rms * l_wnorm[k];
            acc_k += nx * row_k[k];
            acc_v += nx * row_v[k];
        }
        k_slot[o] = acc_k;
        v_slot[o] = acc_v;
    }
    __syncthreads();

    // 4. RoPE rotation on Q (dst_attn) and K (k_slot)
    size_t half_d = D / 2;
    for (size_t idx = tid; idx < H * half_d; idx += blockDim.x) {
        size_t h = idx / half_d;
        size_t j = idx % half_d;
        float freq = 1.0f / powf(theta, (2.0f * (float)j) / (float)D);
        float angle = (float)safe_step * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float* q_head = dst_attn + (h * D);
        float x1 = q_head[j];
        float x2 = q_head[j + half_d];
        q_head[j] = x1 * cos_a - x2 * sin_a;
        q_head[j + half_d] = x2 * cos_a + x1 * sin_a;
    }
    for (size_t idx = tid; idx < KV_H * half_d; idx += blockDim.x) {
        size_t h = idx / half_d;
        size_t j = idx % half_d;
        float freq = 1.0f / powf(theta, (2.0f * (float)j) / (float)D);
        float angle = (float)safe_step * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float* k_head = k_slot + (h * D);
        float x1 = k_head[j];
        float x2 = k_head[j + half_d];
        k_head[j] = x1 * cos_a - x2 * sin_a;
        k_head[j + half_d] = x2 * cos_a + x1 * sin_a;
    }
    __syncthreads();

    // 5. Causal FlashAttention reduction (Online Softmax) per query head
    float scale = 1.0f / sqrtf((float)D);
    size_t kv_group = H / KV_H;
    size_t seq_len = safe_step + 1;

    for (size_t h = 0; h < H; ++h) {
        size_t kv_h = h / kv_group;
        __shared__ float s_q[128];
        if (tid < D && tid < 128) {
            s_q[tid] = dst_attn[h * D + tid];
        }
        __syncthreads();

        float acc_d = 0.0f;
        float m = -INFINITY;
        float l = 0.0f;

        for (size_t t = 0; t < seq_len; ++t) {
            const float* k_t = k_layer + (t * kv_dim) + (kv_h * D);
            const float* v_t = v_layer + (t * kv_dim) + (kv_h * D);

            __shared__ float s_dot[256];
            float my_dot = 0.0f;
            if (tid < D) {
                my_dot = s_q[tid] * k_t[tid];
            }
            s_dot[tid] = my_dot;
            __syncthreads();
            for (size_t s = 128; s > 0; s >>= 1) {
                if (tid < s && tid + s < 256) s_dot[tid] += s_dot[tid + s];
                __syncthreads();
            }
            float dot = s_dot[0] * scale;

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
            dst_attn[h * D + tid] = (l > 0.0f) ? (acc_d / l) : 0.0f;
        }
        __syncthreads();
    }
}

// SOTA Fused Kernel 2 of 2 per Layer: OutProj + Residual + Post-Norm + SwiGLU + Residual
__global__ void __launch_bounds__(256) velox_cuda_fused_out_res_swiglu_res(
    float* __restrict__ dst_hidden, const float* __restrict__ attn_in, const float* __restrict__ residual_in,
    const float* __restrict__ w_out, const float* __restrict__ w_post_norm,
    const float* __restrict__ w_gate, const float* __restrict__ w_up, const float* __restrict__ w_down,
    size_t layer, float eps, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    if (blockIdx.x > 0) return;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    size_t tid = threadIdx.x;
    const float* l_wout = w_out + (layer * dim * dim);
    const float* l_wpost = w_post_norm + (layer * dim);
    const float* l_wgate = w_gate + (layer * inter_dim * dim);
    const float* l_wup = w_up + (layer * inter_dim * dim);
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

// SOTA Absorbed MLA Decode Kernel: Direct Latent Contraction without KV Decompression
__global__ void __launch_bounds__(256) velox_cuda_fused_mla_absorbed_attn(
    float* __restrict__ dst_hidden, const float* __restrict__ input_hidden,
    const float* __restrict__ w_norm, const float* __restrict__ w_dkv,
    const float* __restrict__ w_absorbed_q, const float* __restrict__ w_absorbed_o,
    float* __restrict__ kv_latent, size_t layer, size_t step, float eps, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    if (blockIdx.x > 0) return;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    const size_t d_c = 512;
    size_t tid = threadIdx.x;

    // 1. RMSNorm of input_hidden
    __shared__ float s_ss[256];
    float local_ss = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        float v = input_hidden[i];
        local_ss += v * v;
    }
    s_ss[tid] = local_ss;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_ss[tid] += s_ss[tid + stride];
        __syncthreads();
    }
    float inv_rms = rsqrtf((s_ss[0] / (float)dim) + eps);
    const float* l_wnorm = w_norm + (layer * dim);
    __syncthreads();

    // 2. Project KV compressed latent slot: c_t^{KV} = norm_x @ W_{DKV}
    size_t safe_step = (step < max_s) ? step : (max_s - 1);
    float* layer_latent = kv_latent + (layer * max_s * d_c);
    float* slot = layer_latent + (safe_step * d_c);
    const float* l_wdkv = w_dkv + (layer * dim * d_c);
    for (size_t o = tid; o < d_c; o += blockDim.x) {
        const float* row = l_wdkv + (o * dim);
        float acc = 0.0f;
        for (size_t k = 0; k < dim; ++k) {
            float nx = input_hidden[k] * inv_rms * l_wnorm[k];
            acc += nx * row[k];
        }
        slot[o] = acc;
    }

    // 3. Project absorbed query: q_abs = norm_x @ W_{absorbed_q}
    __shared__ float s_qabs[512];
    const float* l_wqabs = w_absorbed_q + (layer * dim * d_c);
    for (size_t o = tid; o < d_c; o += blockDim.x) {
        const float* row = l_wqabs + (o * dim);
        float acc = 0.0f;
        for (size_t k = 0; k < dim; ++k) {
            float nx = input_hidden[k] * inv_rms * l_wnorm[k];
            acc += nx * row[k];
        }
        s_qabs[o] = acc;
    }
    __syncthreads();

    // 4. Causal FlashAttention Online Softmax reduction over latent space
    __shared__ float s_ctx[512];
    for (size_t d = tid; d < d_c; d += blockDim.x) { s_ctx[d] = 0.0f; }
    __syncthreads();

    float scale = 1.0f / sqrtf(128.0f);
    size_t seq_len = safe_step + 1;
    float m = -INFINITY;
    float l = 0.0f;
    __shared__ float s_dot[256];

    for (size_t t = 0; t < seq_len; ++t) {
        const float* k_vec = layer_latent + (t * d_c);
        float my_dot = 0.0f;
        for (size_t d = tid; d < d_c; d += blockDim.x) {
            my_dot += s_qabs[d] * k_vec[d];
        }
        s_dot[tid] = my_dot;
        __syncthreads();
        for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) s_dot[tid] += s_dot[tid + s];
            __syncthreads();
        }
        float dot = s_dot[0] * scale;
        float new_m = (dot > m) ? dot : m;
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - new_m);
        float beta = (new_m == -INFINITY) ? 0.0f : expf(dot - new_m);
        l = l * alpha + beta;
        m = new_m;
        for (size_t d = tid; d < d_c; d += blockDim.x) {
            s_ctx[d] = s_ctx[d] * alpha + beta * k_vec[d];
        }
        __syncthreads();
    }
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (size_t d = tid; d < d_c; d += blockDim.x) { s_ctx[d] *= inv_l; }
    __syncthreads();

    // 5. Down-projection context @ W_{absorbed_o} + residual add
    const float* l_woabs = w_absorbed_o + (layer * d_c * dim);
    for (size_t o = tid; o < dim; o += blockDim.x) {
        const float* row = l_woabs + (o * d_c);
        float acc = 0.0f;
        for (size_t d = 0; d < d_c; ++d) {
            acc += s_ctx[d] * row[d];
        }
        dst_hidden[o] = input_hidden[o] + acc;
    }
}

__global__ void __launch_bounds__(256) velox_cuda_moe_gate_topk(
    uint32_t* __restrict__ dst_indices, float* __restrict__ dst_weights,
    const float* __restrict__ input, const float* __restrict__ w_gate,
    size_t layer, size_t hidden_dim, size_t num_experts, size_t top_k, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    if (blockIdx.x > 0) return;
    size_t tid = threadIdx.x;
    const float* l_gate = w_gate + (layer * num_experts * hidden_dim);
    size_t safe_experts = (num_experts <= 256) ? num_experts : 256;
    __shared__ float s_logits[256];

    for (size_t e = tid; e < safe_experts; e += blockDim.x) {
        const float* row = l_gate + (e * hidden_dim);
        float acc = 0.0f;
        for (size_t k = 0; k < hidden_dim; ++k) { acc += input[k] * row[k]; }
        s_logits[e] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        for (size_t k = 0; k < top_k; ++k) {
            size_t best_idx = 0;
            float best_val = -INFINITY;
            for (size_t e = 0; e < safe_experts; ++e) {
                int already = 0;
                for (size_t prev = 0; prev < k; ++prev) {
                    if (dst_indices[prev] == (uint32_t)e) { already = 1; break; }
                }
                if (!already && s_logits[e] > best_val) {
                    best_val = s_logits[e];
                    best_idx = e;
                }
            }
            dst_indices[k] = (uint32_t)best_idx;
        }

        float max_l = s_logits[dst_indices[0]];
        for (size_t k = 1; k < top_k; ++k) {
            if (s_logits[dst_indices[k]] > max_l) max_l = s_logits[dst_indices[k]];
        }
        float sum_exp = 0.0f;
        for (size_t k = 0; k < top_k; ++k) {
            float e = expf(s_logits[dst_indices[k]] - max_l);
            dst_weights[k] = e;
            sum_exp += e;
        }
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (size_t k = 0; k < top_k; ++k) { dst_weights[k] *= inv_sum; }
    }
}

__global__ void __launch_bounds__(256) velox_cuda_moe_dispatch_topk(
    float* __restrict__ dst, const float* __restrict__ input,
    const uint32_t* __restrict__ indices, const float* __restrict__ weights,
    const float* __restrict__ w_experts, size_t layer, size_t hidden_dim,
    size_t expert_inter_dim, size_t num_experts, size_t top_k, float capacity_factor,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    if (blockIdx.x > 0) return;
    size_t tid = threadIdx.x;
    for (size_t o = tid; o < hidden_dim; o += blockDim.x) { dst[o] = 0.0f; }
    __syncthreads();

    size_t expert_size = expert_inter_dim * hidden_dim;
    size_t max_dispatched = (size_t)((float)top_k * capacity_factor);
    if (max_dispatched > top_k) max_dispatched = top_k;
    const size_t CHUNK = 128;
    __shared__ float s_inter[CHUNK];

    for (size_t k = 0; k < max_dispatched; ++k) {
        uint32_t exp_id = indices[k];
        float w = weights[k];
        const float* exp_w = w_experts + (layer * num_experts * expert_size) + ((size_t)exp_id * expert_size);

        for (size_t j_base = 0; j_base < expert_inter_dim; j_base += CHUNK) {
            size_t chunk_len = ((j_base + CHUNK) <= expert_inter_dim) ? CHUNK : (expert_inter_dim - j_base);
            for (size_t c = tid; c < chunk_len; c += blockDim.x) {
                size_t j = j_base + c;
                float acc = 0.0f;
                for (size_t i = 0; i < hidden_dim; ++i) {
                    acc += input[i] * exp_w[i * expert_inter_dim + j];
                }
                s_inter[c] = acc / (1.0f + expf(-acc));
            }
            __syncthreads();

            for (size_t o = tid; o < hidden_dim; o += blockDim.x) {
                float acc = 0.0f;
                for (size_t c = 0; c < chunk_len; ++c) {
                    acc += s_inter[c] * exp_w[o * expert_inter_dim + (j_base + c)];
                }
                dst[o] += w * acc;
            }
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(256) velox_cuda_final_norm_argmax(
    uint32_t* __restrict__ host_ring, const float* __restrict__ logits,
    size_t vocab_size, size_t step, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    if (threadIdx.x == 0 && blockIdx.x == 0 && host_ring != NULL) {
        uint32_t best_id = 0;
        float best_val = logits[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (logits[v] > best_val) { best_val = logits[v]; best_id = (uint32_t)v; }
        }
        host_ring[step & 1023ULL] = best_id;
    }
}

extern "C" void velox_SmolLM_135M_prefill_cuda(VeloxCudaArena* ctx, uint32_t token_in, size_t step, uint64_t active_mask, cudaStream_t stream) {
    dim3 grid((VELOX_HIDDEN_DIM + 255) / 256);
    dim3 block(256);
    velox_cuda_embed_kernel<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), token_in, VELOX_DEV_W_PTR(ctx->weight_arena, 0ULL), active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 0ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 0ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 1ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 1ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 2ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 2ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 3ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 3ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 4ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 4ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 5ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 5ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 6ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 6ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 7ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 7ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 8ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 8ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 9ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 9ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 10ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 10ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 11ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 11ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 12ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 12ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 13ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 13ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 14ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 14ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 15ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 15ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 16ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 16ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 17ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 17ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 18ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 18ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 19ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 19ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 20ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 20ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 21ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 21ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 22ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 22ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 23ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 23ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 24ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 24ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 25ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 25ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 26ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 26ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 27ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 27ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 28ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 28ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 29ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 29ULL, 0.000010f, active_mask);
    velox_cuda_rmsnorm_kernel<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 786432ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 538057728ULL), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    velox_cuda_matmul_vec_kernel<<<dim3((49152ULL + 255) / 256), block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 786432ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 538060032ULL), 576ULL, 49152ULL, active_mask);
    velox_cuda_final_norm_argmax<<<1, 32, 0, stream>>>(ctx->host_pinned_ring, VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_VOCAB_SIZE, step, active_mask);
}

extern "C" void velox_SmolLM_135M_decode_cuda(VeloxCudaArena* ctx, uint32_t token_in, size_t step, uint64_t active_mask, cudaStream_t stream) {
    dim3 grid((VELOX_HIDDEN_DIM + 255) / 256);
    dim3 block(256);
    velox_cuda_embed_kernel<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), token_in, VELOX_DEV_W_PTR(ctx->weight_arena, 0ULL), active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 0ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 0ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 1ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 1ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 2ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 2ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 3ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 3ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 4ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 4ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 5ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 5ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 6ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 6ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 7ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 7ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 8ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 8ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 9ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 9ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 10ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 10ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 11ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 11ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 12ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 12ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 13ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 13ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 14ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 14ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 15ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 15ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 16ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 16ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 17ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 17ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 18ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 18ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 19ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 19ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 20ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 20ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 21ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 21ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 22ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 22ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 23ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 23ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 24ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 24ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 25ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 25ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 26ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 26ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 27ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 27ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 28ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 28ULL, 0.000010f, active_mask);
    velox_cuda_fused_norm_qkv_rope_attn<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537919488ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), (float*)ctx->kv_cache_pool, 29ULL, step, 0.000010f, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_fused_out_res_swiglu_res<<<1, 256, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 537988608ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 29ULL, 0.000010f, active_mask);
    velox_cuda_rmsnorm_kernel<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 1572864ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 538057728ULL), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    velox_cuda_matmul_vec_kernel<<<dim3((49152ULL + 255) / 256), block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 1572864ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 538060032ULL), 576ULL, 49152ULL, active_mask);
    velox_cuda_final_norm_argmax<<<1, 32, 0, stream>>>(ctx->host_pinned_ring, VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_VOCAB_SIZE, step, active_mask);
}

