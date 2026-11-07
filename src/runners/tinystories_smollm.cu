// ==========================================================================
// Velox Standalone Zero-Allocation CUDA Serving Module (.cu)
// Model:             SmolLM_135M
// Target GPU:        nvidia_rtx_3050_laptop (sm_86)
// Memory Planner:    Velox Static Memory Planner (256B Aligned Offsets)
// Dynamic Heap Ops:  0 (Pre-Committed Static Device Arena)
// ==========================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
#define VELOX_SCRATCH_NORM_OFFSET            2097152ULL
#define VELOX_SCRATCH_ACT_OFFSET             2162688ULL
#define VELOX_SCRATCH_STAGE1_VALS_OFFSET     2228224ULL
#define VELOX_SCRATCH_STAGE1_IDXS_OFFSET     2244608ULL
#define VELOX_SCRATCH_DECODE_STATE_OFFSET    2260992ULL
#define VELOX_HOST_RING_SLOTS                1024ULL
#define VELOX_HOST_RING_EMPTY                0xFFFFFFFFu
#define VELOX_QUANT_FP32                     0u
#define VELOX_QUANT_FP16                     1u
#define VELOX_QUANT_Q4_0                     2u

typedef struct VeloxCudaArena {
    const uint8_t* __restrict__ weight_arena;
    uint8_t*       __restrict__ kv_cache_pool;
    uint8_t*       __restrict__ activation_arena;
    uint32_t*      __restrict__ host_pinned_ring;
    const uint8_t* __restrict__ norm_weight_arena;
    const uint8_t* __restrict__ weight_scales;
    uint32_t                    quant_mode;
    uint32_t                    tied_lm_head;
} VeloxCudaArena;

// Device-resident autoregressive state. Every step-dependent kernel reads `step`/`token` from here
// instead of taking them by value, so a single captured CUDA Graph can be replayed for every
// decode position. The final argmax kernel feeds the sampled token back and advances `step`.
typedef struct VeloxDecodeState {
    unsigned long long step;
    uint32_t token;
    uint32_t reserved;
} VeloxDecodeState;

#define VELOX_W_NORM_START_FP32              537919488ULL
#define VELOX_W_HEAD_START_FP32              538060032ULL
#define VELOX_EFF_W_OFF(ctx, offset)         (((ctx)->tied_lm_head && (offset) >= VELOX_W_HEAD_START_FP32) ? 0ULL : (offset))
#define VELOX_DEV_ACT_PTR(arena_base, offset) ((float*)((uint8_t*)(arena_base) + (((offset) >= 4194304ULL) ? 147456ULL : (offset))))
#define VELOX_DEV_W_PTR(weight_base, offset)  ((const float*)((const uint8_t*)(weight_base) + (offset)))
#define VELOX_DEV_W_F16_PTR(ctx, offset)      ((const __half*)((const uint8_t*)(ctx)->weight_arena + (VELOX_EFF_W_OFF(ctx, offset) >> 1)))
#define VELOX_DEV_W_Q4_PTR(ctx, offset)       ((const uint32_t*)((const uint8_t*)(ctx)->weight_arena + (VELOX_EFF_W_OFF(ctx, offset) >> 3)))
#define VELOX_DEV_S_Q4_PTR(ctx, offset)       (((const half2*)(ctx)->weight_scales) + (VELOX_EFF_W_OFF(ctx, offset) >> 8))
#define VELOX_DEV_NORM_PTR(ctx, offset)       ((const float*)(((ctx)->norm_weight_arena != NULL) ? ((ctx)->norm_weight_arena + ((offset) - VELOX_W_NORM_START_FP32)) : ((ctx)->weight_arena + (offset))))
#define VELOX_DEV_STATE(ctx) ((VeloxDecodeState*)((uint8_t*)(ctx)->activation_arena + VELOX_SCRATCH_DECODE_STATE_OFFSET))

// Writes the host-provided (token, step) into the device state (used by the eager API entry points).
__global__ void velox_cuda_set_decode_state(VeloxDecodeState* __restrict__ state, uint32_t token, unsigned long long step) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        state->step = step;
        state->token = token;
        state->reserved = 0u;
    }
}

// --- 1. Warp-Specialized & Fused CUDA Kernels -----------------------------
__global__ void __launch_bounds__(256) velox_cuda_embed_kernel(
    float* __restrict__ dst, const VeloxDecodeState* __restrict__ state, const float* __restrict__ w_embed, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t token_id = (size_t)state->token;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < VELOX_HIDDEN_DIM && token_id < VELOX_VOCAB_SIZE) {
        dst[idx] = w_embed[token_id * VELOX_HIDDEN_DIM + idx];
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

__global__ void __launch_bounds__(128, 8) velox_cuda_matmul_vec_kernel(
    float* __restrict__ dst, const float* __restrict__ x, const float* __restrict__ w,
    size_t in_dim, size_t out_dim, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t row = (size_t)blockIdx.x * 4ULL + warp_id;
    if (row >= out_dim) return;

    const float* __restrict__ row_w = w + (row * in_dim);
    float acc = 0.0f;

    const size_t steps = in_dim / 128ULL;
    for (size_t s = 0; s < steps; ++s) {
        const size_t k_offset = s * 128ULL + (lane_id * 4ULL);
        const float4 w_v = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 x_v = *reinterpret_cast<const float4*>(&x[k_offset]);
        acc += w_v.x * x_v.x + w_v.y * x_v.y + w_v.z * x_v.z + w_v.w * x_v.w;
    }

    const size_t tail_start = steps * 128ULL;
    const size_t remaining_floats = in_dim - tail_start;
    const size_t remaining_f4 = remaining_floats / 4ULL;
    if (lane_id < remaining_f4) {
        const size_t k_offset = tail_start + (lane_id * 4ULL);
        const float4 w_v = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 x_v = *reinterpret_cast<const float4*>(&x[k_offset]);
        acc += w_v.x * x_v.x + w_v.y * x_v.y + w_v.z * x_v.z + w_v.w * x_v.w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        dst[row] = acc;
    }
}

// 1.3b Dedicated Layer RMSNorm Kernel
__global__ void __launch_bounds__(256) velox_cuda_layer_norm_kernel(
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
    for (size_t i = tid; i < dim; i += blockDim.x) {
        dst[i] = (x[i] * inv_rms) * weight[i];
    }
}

// 1.3c Multi-Block Coalesced Warp QKV Kernel (240 blocks on SmolLM-135M)
__global__ void __launch_bounds__(128, 8) velox_cuda_fused_qkv_warp_kernel(
    float* __restrict__ dst_q,
    float* __restrict__ kv_pool,
    const float* __restrict__ norm_input,
    const float* __restrict__ w_q,
    const float* __restrict__ w_k,
    const float* __restrict__ w_v,
    size_t layer,
    const VeloxDecodeState* __restrict__ state,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t step = (size_t)state->step;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t global_row = (size_t)blockIdx.x * 4ULL + warp_id;

    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t num_q = VELOX_NUM_HEADS * VELOX_HEAD_DIM;
    const size_t num_kv = VELOX_NUM_KV_HEADS * VELOX_HEAD_DIM;
    const size_t total_rows = num_q + 2ULL * num_kv;
    if (global_row >= total_rows) return;

    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    const size_t safe_step = (step < max_s) ? step : (max_s - 1);
    const size_t layer_kv_stride = 2ULL * max_s * num_kv;
    float* k_layer = kv_pool + (layer * layer_kv_stride);
    float* v_layer = k_layer + (max_s * num_kv);

    const float* __restrict__ row_w = NULL;
    float* __restrict__ dst_ptr = NULL;

    if (global_row < num_q) {
        row_w = w_q + (layer * dim * num_q) + (global_row * dim);
        dst_ptr = dst_q + global_row;
    } else if (global_row < num_q + num_kv) {
        const size_t k_row = global_row - num_q;
        row_w = w_k + (layer * dim * num_kv) + (k_row * dim);
        dst_ptr = k_layer + (safe_step * num_kv) + k_row;
    } else {
        const size_t v_row = global_row - num_q - num_kv;
        row_w = w_v + (layer * dim * num_kv) + (v_row * dim);
        dst_ptr = v_layer + (safe_step * num_kv) + v_row;
    }

    float acc = 0.0f;
    const size_t steps = dim / 128ULL;
    for (size_t s = 0; s < steps; ++s) {
        const size_t k_offset = s * 128ULL + (lane_id * 4ULL);
        const float4 w_v_val = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&norm_input[k_offset]);
        acc += in_v.x * w_v_val.x + in_v.y * w_v_val.y + in_v.z * w_v_val.z + in_v.w * w_v_val.w;
    }

    const size_t tail_start = steps * 128ULL;
    const size_t remaining_floats = dim - tail_start;
    const size_t remaining_f4 = remaining_floats / 4ULL;
    if (lane_id < remaining_f4) {
        const size_t k_offset = tail_start + (lane_id * 4ULL);
        const float4 w_v_val = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&norm_input[k_offset]);
        acc += in_v.x * w_v_val.x + in_v.y * w_v_val.y + in_v.z * w_v_val.z + in_v.w * w_v_val.w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        *dst_ptr = acc;
    }
}

// 1.3d Multi-Block Fused RoPE Kernel (Q & K in 3 blocks)
__global__ void __launch_bounds__(128, 8) velox_cuda_fused_rope_kernel(
    float* __restrict__ q,
    float* __restrict__ kv_pool,
    size_t layer,
    const VeloxDecodeState* __restrict__ state,
    float theta,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t step = (size_t)state->step;
    const size_t H = VELOX_NUM_HEADS;
    const size_t KV_H = VELOX_NUM_KV_HEADS;
    const size_t D = VELOX_HEAD_DIM;
    const size_t half_d = D / 2;
    const size_t q_pairs = H * half_d;
    const size_t kv_pairs = KV_H * half_d;
    const size_t total_pairs = q_pairs + kv_pairs;

    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    const size_t safe_step = (step < max_s) ? step : (max_s - 1);
    const size_t kv_dim = KV_H * D;
    const size_t layer_kv_stride = 2ULL * max_s * kv_dim;
    float* k_layer = kv_pool + (layer * layer_kv_stride);
    float* k_slot = k_layer + (safe_step * kv_dim);

    if (idx < q_pairs) {
        size_t h = idx / half_d;
        size_t j = idx % half_d;
        float freq = 1.0f / powf(theta, (2.0f * (float)j) / (float)D);
        float angle = (float)safe_step * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float* q_head = q + (h * D);
        float x1 = q_head[j];
        float x2 = q_head[j + half_d];
        q_head[j] = x1 * cos_a - x2 * sin_a;
        q_head[j + half_d] = x2 * cos_a + x1 * sin_a;
    } else {
        size_t kv_idx = idx - q_pairs;
        size_t h = kv_idx / half_d;
        size_t j = kv_idx % half_d;
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
}

// 1.3e Parallel GQA Attention Kernel (9 blocks across heads)
__global__ void __launch_bounds__(256, 4) velox_cuda_gqa_attn_kernel(
    float* __restrict__ dst_attn,
    const float* __restrict__ kv_pool,
    size_t layer,
    const VeloxDecodeState* __restrict__ state,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t step = (size_t)state->step;
    const size_t H = VELOX_NUM_HEADS;
    const size_t KV_H = VELOX_NUM_KV_HEADS;
    const size_t D = VELOX_HEAD_DIM;
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    const size_t safe_step = (step < max_s) ? step : (max_s - 1);
    const size_t seq_len = safe_step + 1;
    const size_t kv_dim = KV_H * D;
    const size_t layer_kv_stride = 2ULL * max_s * kv_dim;

    const size_t h = blockIdx.x;
    if (h >= H) return;
    const size_t tid = threadIdx.x;
    const size_t kv_group = H / KV_H;
    const size_t kv_h = h / kv_group;

    const float* k_layer = kv_pool + (layer * layer_kv_stride);
    const float* v_layer = k_layer + (max_s * kv_dim);

    __shared__ float s_q[128];
    if (tid < D) {
        s_q[tid] = dst_attn[h * D + tid];
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)D);
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
        // Tree reduction bounded by the launched block size (power of two, <= 256). Reducing a
        // fixed 256-wide span would read never-written shared memory when blockDim.x < 256.
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

        if (tid < D) {
            acc_d = acc_d * alpha + beta * v_t[tid];
        }
        __syncthreads();
    }

    if (tid < D) {
        dst_attn[h * D + tid] = (l > 0.0f) ? (acc_d / l) : 0.0f;
    }
}

// 1.3f Multi-Block Out Projection + Residual Kernel (144 blocks on SmolLM-135M)
__global__ void __launch_bounds__(128, 8) velox_cuda_out_proj_residual_kernel(
    float* __restrict__ dst_hidden,
    const float* __restrict__ attn_in,
    const float* __restrict__ residual_in,
    const float* __restrict__ w_out,
    size_t layer,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t row = (size_t)blockIdx.x * 4ULL + warp_id;

    const size_t dim = VELOX_HIDDEN_DIM;
    if (row >= dim) return;

    const float* __restrict__ row_w = w_out + (layer * dim * dim) + (row * dim);
    float acc = 0.0f;

    const size_t steps = dim / 128ULL;
    for (size_t s = 0; s < steps; ++s) {
        const size_t k_offset = s * 128ULL + (lane_id * 4ULL);
        const float4 w_v = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&attn_in[k_offset]);
        acc += in_v.x * w_v.x + in_v.y * w_v.y + in_v.z * w_v.z + in_v.w * w_v.w;
    }

    const size_t tail_start = steps * 128ULL;
    const size_t remaining_floats = dim - tail_start;
    const size_t remaining_f4 = remaining_floats / 4ULL;
    if (lane_id < remaining_f4) {
        const size_t k_offset = tail_start + (lane_id * 4ULL);
        const float4 w_v = *reinterpret_cast<const float4*>(&row_w[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&attn_in[k_offset]);
        acc += in_v.x * w_v.x + in_v.y * w_v.y + in_v.z * w_v.z + in_v.w * w_v.w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        dst_hidden[row] = acc + residual_in[row];
    }
}

// 1.3g Multi-Block SwiGLU Gate & Up Kernel (384 blocks on SmolLM-135M)
__global__ void __launch_bounds__(128, 8) velox_cuda_swiglu_gate_up_kernel(
    float* __restrict__ dst_act,
    const float* __restrict__ norm_input,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    size_t layer,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t j = (size_t)blockIdx.x * 4ULL + warp_id;

    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    if (j >= inter_dim) return;

    const float* __restrict__ row_gate = w_gate + (layer * inter_dim * dim) + (j * dim);
    const float* __restrict__ row_up = w_up + (layer * inter_dim * dim) + (j * dim);

    float acc_gate = 0.0f;
    float acc_up = 0.0f;

    const size_t steps = dim / 128ULL;
    for (size_t s = 0; s < steps; ++s) {
        const size_t k_offset = s * 128ULL + (lane_id * 4ULL);
        const float4 wg_v = *reinterpret_cast<const float4*>(&row_gate[k_offset]);
        const float4 wu_v = *reinterpret_cast<const float4*>(&row_up[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&norm_input[k_offset]);
        acc_gate += in_v.x * wg_v.x + in_v.y * wg_v.y + in_v.z * wg_v.z + in_v.w * wg_v.w;
        acc_up += in_v.x * wu_v.x + in_v.y * wu_v.y + in_v.z * wu_v.z + in_v.w * wu_v.w;
    }

    const size_t tail_start = steps * 128ULL;
    const size_t remaining_floats = dim - tail_start;
    const size_t remaining_f4 = remaining_floats / 4ULL;
    if (lane_id < remaining_f4) {
        const size_t k_offset = tail_start + (lane_id * 4ULL);
        const float4 wg_v = *reinterpret_cast<const float4*>(&row_gate[k_offset]);
        const float4 wu_v = *reinterpret_cast<const float4*>(&row_up[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&norm_input[k_offset]);
        acc_gate += in_v.x * wg_v.x + in_v.y * wg_v.y + in_v.z * wg_v.z + in_v.w * wg_v.w;
        acc_up += in_v.x * wu_v.x + in_v.y * wu_v.y + in_v.z * wu_v.z + in_v.w * wu_v.w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_gate += __shfl_down_sync(0xffffffff, acc_gate, offset);
        acc_up += __shfl_down_sync(0xffffffff, acc_up, offset);
    }

    if (lane_id == 0) {
        float silu = acc_gate / (1.0f + expf(-acc_gate));
        dst_act[j] = silu * acc_up;
    }
}

// 1.3h Multi-Block SwiGLU Down Projection + Residual Kernel (144 blocks on SmolLM-135M)
__global__ void __launch_bounds__(128, 8) velox_cuda_swiglu_down_residual_kernel(
    float* __restrict__ dst_hidden,
    const float* __restrict__ act_in,
    const float* __restrict__ w_down,
    size_t layer,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t o = (size_t)blockIdx.x * 4ULL + warp_id;

    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    if (o >= dim) return;

    const float* __restrict__ row_down = w_down + (layer * dim * inter_dim) + (o * inter_dim);
    float acc = 0.0f;

    const size_t steps = inter_dim / 128ULL;
    for (size_t s = 0; s < steps; ++s) {
        const size_t k_offset = s * 128ULL + (lane_id * 4ULL);
        const float4 wd_v = *reinterpret_cast<const float4*>(&row_down[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&act_in[k_offset]);
        acc += in_v.x * wd_v.x + in_v.y * wd_v.y + in_v.z * wd_v.z + in_v.w * wd_v.w;
    }

    const size_t tail_start = steps * 128ULL;
    const size_t remaining_floats = inter_dim - tail_start;
    const size_t remaining_f4 = remaining_floats / 4ULL;
    if (lane_id < remaining_f4) {
        const size_t k_offset = tail_start + (lane_id * 4ULL);
        const float4 wd_v = *reinterpret_cast<const float4*>(&row_down[k_offset]);
        const float4 in_v = *reinterpret_cast<const float4*>(&act_in[k_offset]);
        acc += in_v.x * wd_v.x + in_v.y * wd_v.y + in_v.z * wd_v.z + in_v.w * wd_v.w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane_id == 0) {
        dst_hidden[o] += acc;
    }
}

// 1.3i Two-Stage Parallel Vocabulary Argmax (Stage 1: Multi-Block Reduction)
__global__ void __launch_bounds__(256, 4) velox_cuda_lm_head_argmax_stage1(
    float* __restrict__ scratch_vals,
    uint32_t* __restrict__ scratch_idxs,
    const float* __restrict__ logits,
    size_t vocab_size,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];
    const size_t tid = threadIdx.x;
    const size_t vid = (size_t)blockIdx.x * 256ULL + tid;

    float my_val = (vid < vocab_size) ? logits[vid] : -1.0e30f;
    uint32_t my_idx = (vid < vocab_size) ? (uint32_t)vid : 0U;
    s_val[tid] = my_val;
    s_idx[tid] = my_idx;
    __syncthreads();

    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        scratch_vals[blockIdx.x] = s_val[0];
        scratch_idxs[blockIdx.x] = s_idx[0];
    }
}

// 1.3j Two-Stage Parallel Vocabulary Argmax (Stage 2: Candidate Stride Reduction)
__global__ void __launch_bounds__(256, 1) velox_cuda_lm_head_argmax_stage2(
    uint32_t* __restrict__ host_pinned_ring,
    const float* __restrict__ scratch_vals,
    const uint32_t* __restrict__ scratch_idxs,
    size_t num_candidates,
    VeloxDecodeState* __restrict__ state,
    uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];
    const size_t tid = threadIdx.x;
    const unsigned long long step = state->step;

    float my_val = -1.0e30f;
    uint32_t my_idx = 0U;
    for (size_t c = tid; c < num_candidates; c += blockDim.x) {
        float c_val = scratch_vals[c];
        uint32_t c_idx = scratch_idxs[c];
        if (c_val > my_val) {
            my_val = c_val;
            my_idx = c_idx;
        }
    }
    s_val[tid] = my_val;
    s_idx[tid] = my_idx;
    __syncthreads();

    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const uint32_t winner = s_idx[0];
        if (host_pinned_ring != NULL) {
            host_pinned_ring[step & (VELOX_HOST_RING_SLOTS - 1ULL)] = winner;
            // Make the zero-copy ring write visible to the polling host thread.
            __threadfence_system();
        }
        // Device-side autoregressive feedback: next step embeds `winner` at position step + 1.
        state->token = winner;
        state->step = step + 1ULL;
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
    size_t num_heads, size_t head_dim, const VeloxDecodeState* __restrict__ state, float theta, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t step = (size_t)state->step;
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
    float* __restrict__ kv_latent, size_t layer, const VeloxDecodeState* __restrict__ state, float eps, uint64_t active_mask
) {
    if ((active_mask & 1ULL) == 0ULL) return;
    if (blockIdx.x > 0) return;
    const size_t step = (size_t)state->step;
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

__device__ __forceinline__ uint32_t velox_ptx_lop3_0xEA(uint32_t a, uint32_t b, uint32_t c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    uint32_t result;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
#else
    return (a & b) | c;
#endif
}

__device__ __forceinline__ half2 velox_u32_as_half2(uint32_t v) {
    union { uint32_t u; half2 h2; } conv;
    conv.u = v;
    return conv.h2;
}

__device__ __forceinline__ float velox_marlin_dot8_q4(uint32_t packed, half2 sm_h2, float4 x0, float4 x1) {
    const uint32_t fp16_bias = 0x64006400u;
    const uint32_t mask_low  = 0x000F000Fu;
    uint32_t h2_0 = velox_ptx_lop3_0xEA(packed,       mask_low, fp16_bias);
    uint32_t h2_1 = velox_ptx_lop3_0xEA(packed >> 4,  mask_low, fp16_bias);
    uint32_t h2_2 = velox_ptx_lop3_0xEA(packed >> 8,  mask_low, fp16_bias);
    uint32_t h2_3 = velox_ptx_lop3_0xEA(packed >> 12, mask_low, fp16_bias);
    const half2 sub_bias = velox_u32_as_half2(fp16_bias);
    const float2 f0 = __half22float2(__hsub2(velox_u32_as_half2(h2_0), sub_bias));
    const float2 f1 = __half22float2(__hsub2(velox_u32_as_half2(h2_1), sub_bias));
    const float2 f2 = __half22float2(__hsub2(velox_u32_as_half2(h2_2), sub_bias));
    const float2 f3 = __half22float2(__hsub2(velox_u32_as_half2(h2_3), sub_bias));
    const float sum_qx = f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w
                       + f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
    const float sum_x  = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w);
    const float2 sm    = __half22float2(sm_h2);
    return sm.x * sum_qx + sm.y * sum_x;
}

__device__ __forceinline__ float velox_warp_dot_f16(const __half* __restrict__ row_w, const float* __restrict__ x, size_t dim, size_t lane_id) {
    float acc = 0.0f;
    const size_t steps = dim / 256ULL;
    for (size_t s = 0; s < steps; ++s) {
        const size_t k_offset = s * 256ULL + (lane_id * 8ULL);
        const uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(&row_w[k_offset]));
        const float4 in0 = *reinterpret_cast<const float4*>(&x[k_offset]);
        const float4 in1 = *reinterpret_cast<const float4*>(&x[k_offset + 4ULL]);
        const float2 w01 = __half22float2(velox_u32_as_half2(w_u4.x));
        const float2 w23 = __half22float2(velox_u32_as_half2(w_u4.y));
        const float2 w45 = __half22float2(velox_u32_as_half2(w_u4.z));
        const float2 w67 = __half22float2(velox_u32_as_half2(w_u4.w));
        acc += in0.x * w01.x + in0.y * w01.y + in0.z * w23.x + in0.w * w23.y
             + in1.x * w45.x + in1.y * w45.y + in1.z * w67.x + in1.w * w67.y;
    }
    const size_t tail_start = steps * 256ULL;
    const size_t remaining_f8 = (dim - tail_start) / 8ULL;
    if (lane_id < remaining_f8) {
        const size_t k_offset = tail_start + (lane_id * 8ULL);
        const uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(&row_w[k_offset]));
        const float4 in0 = *reinterpret_cast<const float4*>(&x[k_offset]);
        const float4 in1 = *reinterpret_cast<const float4*>(&x[k_offset + 4ULL]);
        const float2 w01 = __half22float2(velox_u32_as_half2(w_u4.x));
        const float2 w23 = __half22float2(velox_u32_as_half2(w_u4.y));
        const float2 w45 = __half22float2(velox_u32_as_half2(w_u4.z));
        const float2 w67 = __half22float2(velox_u32_as_half2(w_u4.w));
        acc += in0.x * w01.x + in0.y * w01.y + in0.z * w23.x + in0.w * w23.y
             + in1.x * w45.x + in1.y * w45.y + in1.z * w67.x + in1.w * w67.y;
    }
    const size_t tail_start_4 = tail_start + remaining_f8 * 8ULL;
    const size_t remaining_f4 = (dim - tail_start_4) / 4ULL;
    if (lane_id < remaining_f4) {
        const size_t k_offset = tail_start_4 + (lane_id * 4ULL);
        const uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(&row_w[k_offset]));
        const float4 in_v = *reinterpret_cast<const float4*>(&x[k_offset]);
        const float2 w01 = __half22float2(velox_u32_as_half2(w_u2.x));
        const float2 w23 = __half22float2(velox_u32_as_half2(w_u2.y));
        acc += in_v.x * w01.x + in_v.y * w01.y + in_v.z * w23.x + in_v.w * w23.y;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    return acc;
}

__device__ __forceinline__ float velox_warp_dot_q4(const uint32_t* __restrict__ row_w_q4, const half2* __restrict__ row_s_q4, const float* __restrict__ x, size_t dim, size_t lane_id) {
    float acc = 0.0f;
    for (size_t k = lane_id * 8ULL; k + 7ULL < dim; k += 256ULL) {
        const uint32_t packed = __ldg(&row_w_q4[k >> 3]);
        const half2 sm_h2     = __ldg(&row_s_q4[k >> 6]);
        const float4 x0       = *reinterpret_cast<const float4*>(&x[k]);
        const float4 x1       = *reinterpret_cast<const float4*>(&x[k + 4ULL]);
        acc += velox_marlin_dot8_q4(packed, sm_h2, x0, x1);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    return acc;
}

// Startup GPU weight conversion kernels (FP32 -> FP16 and FP32 -> Marlin INT4 group-64)
__global__ void velox_cuda_f32_to_f16_kernel(__half* __restrict__ dst, const float* __restrict__ src, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += (size_t)gridDim.x * blockDim.x) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ void velox_cuda_quantize_q4_group64_kernel(uint32_t* __restrict__ dst_packed, half2* __restrict__ dst_scales, const float* __restrict__ src, size_t num_groups) {
    size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; g < num_groups; g += (size_t)gridDim.x * blockDim.x) {
        const float* grp = src + g * 64ULL;
        float fmin = grp[0];
        float fmax = grp[0];
        for (int i = 1; i < 64; ++i) {
            float v = grp[i];
            if (v < fmin) fmin = v;
            if (v > fmax) fmax = v;
        }
        float scale_f = (fmax > fmin) ? ((fmax - fmin) / 15.0f) : 0.0f;
        __half h_scale = __float2half(scale_f);
        __half h_min   = __float2half(fmin);
        dst_scales[g]  = make_half2(h_scale, h_min);
        float s_exact  = __half2float(h_scale);
        float m_exact  = __half2float(h_min);
        float inv_s    = (s_exact > 1.0e-12f) ? (1.0f / s_exact) : 0.0f;
        for (int w = 0; w < 8; ++w) {
            const float* wp = grp + w * 8;
            uint32_t q[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int qi = __float2int_rn((wp[j] - m_exact) * inv_s);
                if (qi < 0) qi = 0;
                if (qi > 15) qi = 15;
                q[j] = (uint32_t)qi;
            }
            uint32_t packed = (q[0] <<  0) | (q[2] <<  4) | (q[4] <<  8) | (q[6] << 12)
                            | (q[1] << 16) | (q[3] << 20) | (q[5] << 24) | (q[7] << 28);
            dst_packed[g * 8ULL + (size_t)w] = packed;
        }
    }
}

__global__ void __launch_bounds__(256) velox_cuda_embed_kernel_f16(float* __restrict__ dst, const VeloxDecodeState* __restrict__ state, const __half* __restrict__ w_embed, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t token_id = (size_t)state->token;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < VELOX_HIDDEN_DIM && token_id < VELOX_VOCAB_SIZE) {
        dst[idx] = __half2float(__ldg(&w_embed[token_id * VELOX_HIDDEN_DIM + idx]));
    }
}

__global__ void __launch_bounds__(256) velox_cuda_embed_kernel_q4(float* __restrict__ dst, const VeloxDecodeState* __restrict__ state, const uint32_t* __restrict__ w_embed_q4, const half2* __restrict__ s_embed_q4, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t token_id = (size_t)state->token;
    size_t word_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t base_k = word_idx * 8ULL;
    if (base_k + 7ULL < VELOX_HIDDEN_DIM && token_id < VELOX_VOCAB_SIZE) {
        size_t elem_off = token_id * VELOX_HIDDEN_DIM + base_k;
        uint32_t packed = __ldg(&w_embed_q4[elem_off >> 3]);
        float2 sm = __half22float2(__ldg(&s_embed_q4[elem_off >> 6]));
        const uint32_t fp16_bias = 0x64006400u;
        const uint32_t mask_low  = 0x000F000Fu;
        uint32_t h2_0 = velox_ptx_lop3_0xEA(packed,       mask_low, fp16_bias);
        uint32_t h2_1 = velox_ptx_lop3_0xEA(packed >> 4,  mask_low, fp16_bias);
        uint32_t h2_2 = velox_ptx_lop3_0xEA(packed >> 8,  mask_low, fp16_bias);
        uint32_t h2_3 = velox_ptx_lop3_0xEA(packed >> 12, mask_low, fp16_bias);
        const half2 sub_bias = velox_u32_as_half2(fp16_bias);
        const float2 f0 = __half22float2(__hsub2(velox_u32_as_half2(h2_0), sub_bias));
        const float2 f1 = __half22float2(__hsub2(velox_u32_as_half2(h2_1), sub_bias));
        const float2 f2 = __half22float2(__hsub2(velox_u32_as_half2(h2_2), sub_bias));
        const float2 f3 = __half22float2(__hsub2(velox_u32_as_half2(h2_3), sub_bias));
        dst[base_k + 0ULL] = fmaf(f0.x, sm.x, sm.y);
        dst[base_k + 1ULL] = fmaf(f0.y, sm.x, sm.y);
        dst[base_k + 2ULL] = fmaf(f1.x, sm.x, sm.y);
        dst[base_k + 3ULL] = fmaf(f1.y, sm.x, sm.y);
        dst[base_k + 4ULL] = fmaf(f2.x, sm.x, sm.y);
        dst[base_k + 5ULL] = fmaf(f2.y, sm.x, sm.y);
        dst[base_k + 6ULL] = fmaf(f3.x, sm.x, sm.y);
        dst[base_k + 7ULL] = fmaf(f3.y, sm.x, sm.y);
    }
}

__global__ void __launch_bounds__(128, 8) velox_cuda_matmul_vec_kernel_f16(float* __restrict__ dst, const float* __restrict__ x, const __half* __restrict__ w, size_t in_dim, size_t out_dim, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t row = (size_t)blockIdx.x * 4ULL + warp_id;
    if (row >= out_dim) return;
    float acc = velox_warp_dot_f16(w + (row * in_dim), x, in_dim, lane_id);
    if (lane_id == 0) dst[row] = acc;
}

__global__ void __launch_bounds__(128, 8) velox_cuda_matmul_vec_kernel_q4(float* __restrict__ dst, const float* __restrict__ x, const uint32_t* __restrict__ w_q4, const half2* __restrict__ s_q4, size_t in_dim, size_t out_dim, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t row = (size_t)blockIdx.x * 4ULL + warp_id;
    if (row >= out_dim) return;
    float acc = velox_warp_dot_q4(w_q4 + ((row * in_dim) >> 3), s_q4 + ((row * in_dim) >> 6), x, in_dim, lane_id);
    if (lane_id == 0) dst[row] = acc;
}

__global__ void __launch_bounds__(128, 8) velox_cuda_fused_qkv_warp_kernel_f16(float* __restrict__ dst_q, float* __restrict__ kv_pool, const float* __restrict__ norm_input, const __half* __restrict__ w_q, const __half* __restrict__ w_k, const __half* __restrict__ w_v, size_t layer, const VeloxDecodeState* __restrict__ state, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t step = (size_t)state->step;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t global_row = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t num_q = VELOX_NUM_HEADS * VELOX_HEAD_DIM;
    const size_t num_kv = VELOX_NUM_KV_HEADS * VELOX_HEAD_DIM;
    if (global_row >= num_q + 2ULL * num_kv) return;
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    const size_t safe_step = (step < max_s) ? step : (max_s - 1);
    float* k_layer = kv_pool + (layer * 2ULL * max_s * num_kv);
    float* v_layer = k_layer + (max_s * num_kv);
    const __half* __restrict__ row_w = NULL;
    float* __restrict__ dst_ptr = NULL;
    if (global_row < num_q) {
        row_w = w_q + (layer * dim * num_q) + (global_row * dim);
        dst_ptr = dst_q + global_row;
    } else if (global_row < num_q + num_kv) {
        const size_t k_row = global_row - num_q;
        row_w = w_k + (layer * dim * num_kv) + (k_row * dim);
        dst_ptr = k_layer + (safe_step * num_kv) + k_row;
    } else {
        const size_t v_row = global_row - num_q - num_kv;
        row_w = w_v + (layer * dim * num_kv) + (v_row * dim);
        dst_ptr = v_layer + (safe_step * num_kv) + v_row;
    }
    float acc = velox_warp_dot_f16(row_w, norm_input, dim, lane_id);
    if (lane_id == 0) *dst_ptr = acc;
}

__global__ void __launch_bounds__(128, 8) velox_cuda_fused_qkv_warp_kernel_q4(float* __restrict__ dst_q, float* __restrict__ kv_pool, const float* __restrict__ norm_input, const uint32_t* __restrict__ w_q, const half2* __restrict__ s_q, const uint32_t* __restrict__ w_k, const half2* __restrict__ s_k, const uint32_t* __restrict__ w_v, const half2* __restrict__ s_v, size_t layer, const VeloxDecodeState* __restrict__ state, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t step = (size_t)state->step;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t global_row = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t num_q = VELOX_NUM_HEADS * VELOX_HEAD_DIM;
    const size_t num_kv = VELOX_NUM_KV_HEADS * VELOX_HEAD_DIM;
    if (global_row >= num_q + 2ULL * num_kv) return;
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    const size_t safe_step = (step < max_s) ? step : (max_s - 1);
    float* k_layer = kv_pool + (layer * 2ULL * max_s * num_kv);
    float* v_layer = k_layer + (max_s * num_kv);
    const uint32_t* __restrict__ row_w = NULL;
    const half2* __restrict__ row_s = NULL;
    float* __restrict__ dst_ptr = NULL;
    if (global_row < num_q) {
        size_t off = (layer * dim * num_q) + (global_row * dim);
        row_w = w_q + (off >> 3); row_s = s_q + (off >> 6);
        dst_ptr = dst_q + global_row;
    } else if (global_row < num_q + num_kv) {
        const size_t k_row = global_row - num_q;
        size_t off = (layer * dim * num_kv) + (k_row * dim);
        row_w = w_k + (off >> 3); row_s = s_k + (off >> 6);
        dst_ptr = k_layer + (safe_step * num_kv) + k_row;
    } else {
        const size_t v_row = global_row - num_q - num_kv;
        size_t off = (layer * dim * num_kv) + (v_row * dim);
        row_w = w_v + (off >> 3); row_s = s_v + (off >> 6);
        dst_ptr = v_layer + (safe_step * num_kv) + v_row;
    }
    float acc = velox_warp_dot_q4(row_w, row_s, norm_input, dim, lane_id);
    if (lane_id == 0) *dst_ptr = acc;
}

__global__ void __launch_bounds__(128, 8) velox_cuda_out_proj_residual_kernel_f16(float* __restrict__ dst_hidden, const float* __restrict__ attn_in, const float* __restrict__ residual_in, const __half* __restrict__ w_out, size_t layer, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t row = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    if (row >= dim) return;
    float acc = velox_warp_dot_f16(w_out + (layer * dim * dim) + (row * dim), attn_in, dim, lane_id);
    if (lane_id == 0) dst_hidden[row] = acc + residual_in[row];
}

__global__ void __launch_bounds__(128, 8) velox_cuda_out_proj_residual_kernel_q4(float* __restrict__ dst_hidden, const float* __restrict__ attn_in, const float* __restrict__ residual_in, const uint32_t* __restrict__ w_out, const half2* __restrict__ s_out, size_t layer, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t row = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    if (row >= dim) return;
    size_t off = (layer * dim * dim) + (row * dim);
    float acc = velox_warp_dot_q4(w_out + (off >> 3), s_out + (off >> 6), attn_in, dim, lane_id);
    if (lane_id == 0) dst_hidden[row] = acc + residual_in[row];
}

__global__ void __launch_bounds__(128, 8) velox_cuda_swiglu_gate_up_kernel_f16(float* __restrict__ dst_act, const float* __restrict__ norm_input, const __half* __restrict__ w_gate, const __half* __restrict__ w_up, size_t layer, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t j = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    if (j >= inter_dim) return;
    size_t off = (layer * inter_dim * dim) + (j * dim);
    float acc_gate = velox_warp_dot_f16(w_gate + off, norm_input, dim, lane_id);
    float acc_up   = velox_warp_dot_f16(w_up + off, norm_input, dim, lane_id);
    if (lane_id == 0) {
        float silu = acc_gate / (1.0f + expf(-acc_gate));
        dst_act[j] = silu * acc_up;
    }
}

__global__ void __launch_bounds__(128, 8) velox_cuda_swiglu_gate_up_kernel_q4(float* __restrict__ dst_act, const float* __restrict__ norm_input, const uint32_t* __restrict__ w_gate, const half2* __restrict__ s_gate, const uint32_t* __restrict__ w_up, const half2* __restrict__ s_up, size_t layer, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t j = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    if (j >= inter_dim) return;
    size_t off = (layer * inter_dim * dim) + (j * dim);
    float acc_gate = velox_warp_dot_q4(w_gate + (off >> 3), s_gate + (off >> 6), norm_input, dim, lane_id);
    float acc_up   = velox_warp_dot_q4(w_up + (off >> 3), s_up + (off >> 6), norm_input, dim, lane_id);
    if (lane_id == 0) {
        float silu = acc_gate / (1.0f + expf(-acc_gate));
        dst_act[j] = silu * acc_up;
    }
}

__global__ void __launch_bounds__(128, 8) velox_cuda_swiglu_down_residual_kernel_f16(float* __restrict__ dst_hidden, const float* __restrict__ act_in, const __half* __restrict__ w_down, size_t layer, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t o = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    if (o >= dim) return;
    float acc = velox_warp_dot_f16(w_down + (layer * dim * inter_dim) + (o * inter_dim), act_in, inter_dim, lane_id);
    if (lane_id == 0) dst_hidden[o] += acc;
}

__global__ void __launch_bounds__(128, 8) velox_cuda_swiglu_down_residual_kernel_q4(float* __restrict__ dst_hidden, const float* __restrict__ act_in, const uint32_t* __restrict__ w_down, const half2* __restrict__ s_down, size_t layer, uint64_t active_mask) {
    if ((active_mask & 1ULL) == 0ULL) return;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t o = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    if (o >= dim) return;
    size_t off = (layer * dim * inter_dim) + (o * inter_dim);
    float acc = velox_warp_dot_q4(w_down + (off >> 3), s_down + (off >> 6), act_in, inter_dim, lane_id);
    if (lane_id == 0) dst_hidden[o] += acc;
}

extern "C" void velox_SmolLM_135M_prefill_graph_step_cuda(VeloxCudaArena* ctx, uint64_t active_mask, cudaStream_t stream) {
    dim3 grid((VELOX_HIDDEN_DIM + 255) / 256);
    dim3 block(256);
    VeloxDecodeState* d_state = VELOX_DEV_STATE(ctx);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_embed_kernel_f16<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), d_state, VELOX_DEV_W_F16_PTR(ctx, 0ULL), active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_embed_kernel_q4<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), d_state, VELOX_DEV_W_Q4_PTR(ctx, 0ULL), VELOX_DEV_S_Q4_PTR(ctx, 0ULL), active_mask);
    } else {
        velox_cuda_embed_kernel<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), d_state, VELOX_DEV_W_PTR(ctx->weight_arena, 0ULL), active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (0ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 0ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 0ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 0ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 0ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 0ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 0ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 0ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 0ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (0ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 0ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 0ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 0ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 0ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 0ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 0ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (1ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 1ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 1ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 1ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 1ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 1ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 1ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 1ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 1ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (1ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 1ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 1ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 1ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 1ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 1ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 1ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (2ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 2ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 2ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 2ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 2ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 2ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 2ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 2ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 2ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (2ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 2ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 2ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 2ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 2ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 2ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 2ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (3ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 3ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 3ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 3ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 3ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 3ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 3ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 3ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 3ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (3ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 3ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 3ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 3ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 3ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 3ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 3ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (4ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 4ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 4ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 4ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 4ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 4ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 4ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 4ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 4ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (4ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 4ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 4ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 4ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 4ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 4ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 4ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (5ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 5ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 5ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 5ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 5ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 5ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 5ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 5ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 5ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (5ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 5ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 5ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 5ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 5ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 5ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 5ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (6ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 6ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 6ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 6ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 6ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 6ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 6ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 6ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 6ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (6ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 6ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 6ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 6ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 6ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 6ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 6ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (7ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 7ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 7ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 7ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 7ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 7ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 7ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 7ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 7ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (7ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 7ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 7ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 7ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 7ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 7ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 7ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (8ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 8ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 8ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 8ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 8ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 8ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 8ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 8ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 8ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (8ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 8ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 8ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 8ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 8ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 8ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 8ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (9ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 9ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 9ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 9ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 9ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 9ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 9ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 9ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 9ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (9ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 9ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 9ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 9ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 9ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 9ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 9ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (10ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 10ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 10ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 10ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 10ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 10ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 10ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 10ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 10ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (10ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 10ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 10ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 10ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 10ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 10ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 10ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (11ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 11ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 11ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 11ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 11ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 11ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 11ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 11ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 11ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (11ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 11ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 11ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 11ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 11ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 11ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 11ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (12ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 12ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 12ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 12ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 12ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 12ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 12ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 12ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 12ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (12ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 12ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 12ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 12ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 12ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 12ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 12ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (13ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 13ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 13ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 13ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 13ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 13ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 13ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 13ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 13ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (13ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 13ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 13ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 13ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 13ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 13ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 13ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (14ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 14ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 14ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 14ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 14ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 14ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 14ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 14ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 14ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (14ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 14ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 14ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 14ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 14ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 14ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 14ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (15ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 15ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 15ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 15ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 15ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 15ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 15ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 15ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 15ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (15ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 15ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 15ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 15ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 15ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 15ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 15ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (16ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 16ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 16ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 16ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 16ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 16ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 16ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 16ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 16ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (16ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 16ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 16ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 16ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 16ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 16ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 16ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (17ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 17ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 17ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 17ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 17ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 17ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 17ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 17ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 17ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (17ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 17ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 17ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 17ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 17ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 17ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 17ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (18ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 18ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 18ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 18ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 18ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 18ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 18ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 18ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 18ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (18ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 18ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 18ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 18ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 18ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 18ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 18ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (19ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 19ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 19ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 19ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 19ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 19ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 19ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 19ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 19ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (19ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 19ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 19ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 19ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 19ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 19ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 19ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (20ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 20ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 20ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 20ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 20ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 20ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 20ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 20ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 20ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (20ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 20ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 20ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 20ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 20ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 20ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 20ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (21ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 21ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 21ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 21ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 21ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 21ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 21ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 21ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 21ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (21ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 21ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 21ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 21ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 21ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 21ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 21ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (22ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 22ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 22ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 22ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 22ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 22ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 22ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 22ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 22ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (22ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 22ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 22ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 22ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 22ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 22ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 22ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (23ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 23ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 23ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 23ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 23ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 23ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 23ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 23ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 23ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (23ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 23ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 23ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 23ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 23ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 23ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 23ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (24ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 24ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 24ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 24ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 24ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 24ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 24ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 24ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 24ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (24ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 24ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 24ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 24ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 24ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 24ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 24ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (25ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 25ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 25ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 25ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 25ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 25ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 25ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 25ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 25ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (25ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 25ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 25ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 25ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 25ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 25ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 25ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (26ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 26ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 26ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 26ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 26ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 26ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 26ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 26ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 26ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (26ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 26ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 26ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 26ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 26ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 26ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 26ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (27ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 27ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 27ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 27ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 27ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 27ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 27ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 27ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 27ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (27ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 27ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 27ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 27ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 27ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 27ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 27ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (28ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 28ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 28ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 28ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 28ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 28ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 28ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 28ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 28ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (28ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 28ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 28ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 28ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 28ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 28ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 28ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (29ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 29ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 29ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 29ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 29ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 29ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 29ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 29ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 29ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (29ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 29ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 29ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 29ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 29ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 29ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 29ULL, active_mask);
    }
    velox_cuda_rmsnorm_kernel<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 786432ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 301989888ULL), VELOX_DEV_NORM_PTR(ctx, 538057728ULL), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_matmul_vec_kernel_f16<<<dim3(12288ULL), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 786432ULL), VELOX_DEV_W_F16_PTR(ctx, 538060032ULL), 576ULL, 49152ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_matmul_vec_kernel_q4<<<dim3(12288ULL), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 786432ULL), VELOX_DEV_W_Q4_PTR(ctx, 538060032ULL), VELOX_DEV_S_Q4_PTR(ctx, 538060032ULL), 576ULL, 49152ULL, active_mask);
    } else {
        velox_cuda_matmul_vec_kernel<<<dim3(12288ULL), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 786432ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 538060032ULL), 576ULL, 49152ULL, active_mask);
    }
    velox_cuda_lm_head_argmax_stage1<<<dim3((VELOX_VOCAB_SIZE + 255ULL) / 256ULL), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_VALS_OFFSET), (uint32_t*)VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_IDXS_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_VOCAB_SIZE, active_mask);
    velox_cuda_lm_head_argmax_stage2<<<dim3(1), dim3(256), 0, stream>>>(ctx->host_pinned_ring, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_VALS_OFFSET), (const uint32_t*)VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_IDXS_OFFSET), (VELOX_VOCAB_SIZE + 255ULL) / 256ULL, d_state, active_mask);
}

extern "C" void velox_SmolLM_135M_prefill_cuda(VeloxCudaArena* ctx, uint32_t token_in, size_t step, uint64_t active_mask, cudaStream_t stream) {
    velox_cuda_set_decode_state<<<1, 1, 0, stream>>>(VELOX_DEV_STATE(ctx), token_in, (unsigned long long)step);
    velox_SmolLM_135M_prefill_graph_step_cuda(ctx, active_mask, stream);
}

extern "C" void velox_SmolLM_135M_decode_graph_step_cuda(VeloxCudaArena* ctx, uint64_t active_mask, cudaStream_t stream) {
    dim3 grid((VELOX_HIDDEN_DIM + 255) / 256);
    dim3 block(256);
    VeloxDecodeState* d_state = VELOX_DEV_STATE(ctx);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_embed_kernel_f16<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), d_state, VELOX_DEV_W_F16_PTR(ctx, 0ULL), active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_embed_kernel_q4<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), d_state, VELOX_DEV_W_Q4_PTR(ctx, 0ULL), VELOX_DEV_S_Q4_PTR(ctx, 0ULL), active_mask);
    } else {
        velox_cuda_embed_kernel<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), d_state, VELOX_DEV_W_PTR(ctx->weight_arena, 0ULL), active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (0ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 0ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 0ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 0ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 0ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 0ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 0ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 0ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 0ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (0ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 0ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 0ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 0ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 0ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 0ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 0ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (1ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 1ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 1ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 1ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 1ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 1ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 1ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 1ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 1ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (1ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 1ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 1ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 1ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 1ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 1ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 1ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (2ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 2ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 2ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 2ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 2ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 2ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 2ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 2ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 2ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (2ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 2ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 2ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 2ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 2ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 2ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 2ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (3ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 3ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 3ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 3ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 3ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 3ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 3ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 3ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 3ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (3ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 3ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 3ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 3ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 3ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 3ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 3ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (4ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 4ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 4ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 4ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 4ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 4ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 4ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 4ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 4ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (4ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 4ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 4ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 4ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 4ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 4ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 4ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (5ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 5ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 5ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 5ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 5ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 5ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 5ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 5ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 5ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (5ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 5ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 5ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 5ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 5ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 5ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 5ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (6ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 6ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 6ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 6ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 6ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 6ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 6ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 6ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 6ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (6ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 6ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 6ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 6ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 6ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 6ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 6ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (7ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 7ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 7ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 7ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 7ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 7ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 7ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 7ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 7ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (7ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 7ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 7ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 7ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 7ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 7ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 7ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (8ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 8ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 8ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 8ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 8ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 8ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 8ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 8ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 8ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (8ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 8ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 8ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 8ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 8ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 8ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 8ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (9ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 9ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 9ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 9ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 9ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 9ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 9ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 9ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 9ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (9ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 9ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 9ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 9ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 9ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 9ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 9ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (10ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 10ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 10ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 10ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 10ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 10ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 10ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 10ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 10ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (10ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 10ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 10ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 10ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 10ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 10ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 10ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (11ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 11ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 11ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 11ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 11ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 11ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 11ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 11ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 11ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (11ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 11ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 11ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 11ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 11ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 11ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 11ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (12ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 12ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 12ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 12ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 12ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 12ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 12ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 12ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 12ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (12ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 12ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 12ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 12ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 12ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 12ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 12ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (13ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 13ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 13ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 13ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 13ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 13ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 13ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 13ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 13ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (13ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 13ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 13ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 13ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 13ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 13ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 13ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (14ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 14ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 14ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 14ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 14ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 14ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 14ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 14ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 14ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (14ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 14ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 14ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 14ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 14ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 14ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 14ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (15ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 15ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 15ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 15ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 15ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 15ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 15ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 15ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 15ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (15ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 15ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 15ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 15ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 15ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 15ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 15ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (16ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 16ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 16ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 16ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 16ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 16ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 16ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 16ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 16ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (16ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 16ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 16ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 16ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 16ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 16ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 16ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (17ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 17ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 17ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 17ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 17ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 17ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 17ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 17ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 17ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (17ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 17ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 17ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 17ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 17ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 17ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 17ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (18ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 18ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 18ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 18ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 18ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 18ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 18ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 18ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 18ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (18ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 18ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 18ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 18ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 18ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 18ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 18ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (19ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 19ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 19ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 19ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 19ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 19ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 19ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 19ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 19ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (19ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 19ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 19ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 19ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 19ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 19ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 19ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (20ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 20ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 20ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 20ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 20ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 20ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 20ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 20ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 20ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (20ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 20ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 20ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 20ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 20ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 20ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 20ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (21ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 21ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 21ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 21ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 21ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 21ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 21ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 21ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 21ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (21ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 21ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 21ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 21ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 21ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 21ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 21ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (22ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 22ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 22ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 22ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 22ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 22ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 22ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 22ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 22ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (22ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 22ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 22ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 22ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 22ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 22ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 22ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (23ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 23ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 23ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 23ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 23ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 23ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 23ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 23ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 23ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (23ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 23ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 23ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 23ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 23ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 23ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 23ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (24ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 24ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 24ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 24ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 24ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 24ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 24ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 24ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 24ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (24ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 24ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 24ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 24ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 24ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 24ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 24ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (25ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 25ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 25ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 25ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 25ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 25ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 25ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 25ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 25ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (25ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 25ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 25ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 25ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 25ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 25ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 25ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (26ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 26ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 26ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 26ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 26ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 26ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 26ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 26ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 26ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (26ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 26ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 26ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 26ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 26ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 26ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 26ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (27ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 27ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 27ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 27ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 27ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 27ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 27ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 27ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 27ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (27ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 27ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 27ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 27ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 27ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 27ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 27ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (28ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 28ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 28ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 28ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 28ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 28ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 28ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 28ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 28ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (28ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 28ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 28ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 28ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 28ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 28ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 28ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537919488ULL) + (29ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_fused_qkv_warp_kernel_f16<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 113246208ULL), VELOX_DEV_W_F16_PTR(ctx, 153059328ULL), VELOX_DEV_W_F16_PTR(ctx, 166330368ULL), 29ULL, d_state, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_fused_qkv_warp_kernel_q4<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_S_Q4_PTR(ctx, 113246208ULL), VELOX_DEV_W_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_S_Q4_PTR(ctx, 153059328ULL), VELOX_DEV_W_Q4_PTR(ctx, 166330368ULL), VELOX_DEV_S_Q4_PTR(ctx, 166330368ULL), 29ULL, d_state, active_mask);
    } else {
        velox_cuda_fused_qkv_warp_kernel<<<dim3(240), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 113246208ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 153059328ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 166330368ULL), 29ULL, d_state, active_mask);
    }
    velox_cuda_fused_rope_kernel<<<dim3(3), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 29ULL, d_state, VELOX_MODEL_ROPE_THETA, active_mask);
    velox_cuda_gqa_attn_kernel<<<dim3(VELOX_NUM_HEADS), dim3(VELOX_HEAD_DIM), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), (float*)ctx->kv_cache_pool, 29ULL, d_state, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_out_proj_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_F16_PTR(ctx, 179601408ULL), 29ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_out_proj_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_Q4_PTR(ctx, 179601408ULL), VELOX_DEV_S_Q4_PTR(ctx, 179601408ULL), 29ULL, active_mask);
    } else {
        velox_cuda_out_proj_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 179601408ULL), 29ULL, active_mask);
    }
    velox_cuda_layer_norm_kernel<<<dim3(1), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 537988608ULL) + (29ULL * VELOX_HIDDEN_DIM), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_swiglu_gate_up_kernel_f16<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 219414528ULL), VELOX_DEV_W_F16_PTR(ctx, 325582848ULL), 29ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_f16<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_F16_PTR(ctx, 431751168ULL), 29ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_swiglu_gate_up_kernel_q4<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_S_Q4_PTR(ctx, 219414528ULL), VELOX_DEV_W_Q4_PTR(ctx, 325582848ULL), VELOX_DEV_S_Q4_PTR(ctx, 325582848ULL), 29ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel_q4<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_Q4_PTR(ctx, 431751168ULL), VELOX_DEV_S_Q4_PTR(ctx, 431751168ULL), 29ULL, active_mask);
    } else {
        velox_cuda_swiglu_gate_up_kernel<<<dim3(384), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_NORM_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 219414528ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 325582848ULL), 29ULL, active_mask);
        velox_cuda_swiglu_down_residual_kernel<<<dim3(144), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_ACT_OFFSET), VELOX_DEV_W_PTR(ctx->weight_arena, 431751168ULL), 29ULL, active_mask);
    }
    velox_cuda_rmsnorm_kernel<<<grid, block, 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 1572864ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 147456ULL), VELOX_DEV_NORM_PTR(ctx, 538057728ULL), VELOX_HIDDEN_DIM, 0.000010f, active_mask);
    if (ctx->quant_mode == VELOX_QUANT_FP16) {
        velox_cuda_matmul_vec_kernel_f16<<<dim3(12288ULL), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 1572864ULL), VELOX_DEV_W_F16_PTR(ctx, 538060032ULL), 576ULL, 49152ULL, active_mask);
    } else if (ctx->quant_mode == VELOX_QUANT_Q4_0) {
        velox_cuda_matmul_vec_kernel_q4<<<dim3(12288ULL), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 1572864ULL), VELOX_DEV_W_Q4_PTR(ctx, 538060032ULL), VELOX_DEV_S_Q4_PTR(ctx, 538060032ULL), 576ULL, 49152ULL, active_mask);
    } else {
        velox_cuda_matmul_vec_kernel<<<dim3(12288ULL), dim3(128), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_DEV_ACT_PTR(ctx->activation_arena, 1572864ULL), VELOX_DEV_W_PTR(ctx->weight_arena, 538060032ULL), 576ULL, 49152ULL, active_mask);
    }
    velox_cuda_lm_head_argmax_stage1<<<dim3((VELOX_VOCAB_SIZE + 255ULL) / 256ULL), dim3(256), 0, stream>>>(VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_VALS_OFFSET), (uint32_t*)VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_IDXS_OFFSET), VELOX_DEV_ACT_PTR(ctx->activation_arena, 0ULL), VELOX_VOCAB_SIZE, active_mask);
    velox_cuda_lm_head_argmax_stage2<<<dim3(1), dim3(256), 0, stream>>>(ctx->host_pinned_ring, VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_VALS_OFFSET), (const uint32_t*)VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_IDXS_OFFSET), (VELOX_VOCAB_SIZE + 255ULL) / 256ULL, d_state, active_mask);
}

extern "C" void velox_SmolLM_135M_decode_cuda(VeloxCudaArena* ctx, uint32_t token_in, size_t step, uint64_t active_mask, cudaStream_t stream) {
    velox_cuda_set_decode_state<<<1, 1, 0, stream>>>(VELOX_DEV_STATE(ctx), token_in, (unsigned long long)step);
    velox_SmolLM_135M_decode_graph_step_cuda(ctx, active_mask, stream);
}

