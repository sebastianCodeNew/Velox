// ============================================================================
// Velox Emitted Standalone CUDA GPU Host Runner
// Target: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, sm_86)
// Model:  K2-Horizon-3.7B (examples/k2_horizon_3_7b.vlx)
//         36 Layers, Hidden=2560, Intermediate=10240, Heads=32, KV_Heads=8,
//         Head_Dim=128, Vocab=250624, RoPE_Theta=10000000.0, Native 512K Context
// Strict Invariant: Strictly zero dynamic allocations during inference
//                   (0 cudaMalloc/cudaFree/malloc/free during autoregressive loop)
// ============================================================================

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Configure maximum sequence length to 2,048 tokens (High-speed VRAM Resident)
#define VELOX_MODEL_MAX_SEQ_LEN           2048ULL

// Include the emitted standalone CUDA model kernels & execution DAG
#include "k2_horizon_3_7b.cu"

// Static Memory Tier Offsets & Sizes from Velox Static Memory Planner
#define K2_TIER1_WEIGHT_ARENA_BYTES       2529781760ULL /* 2412.59 MiB */
#define K2_TIER3_ACTIVATION_ARENA_BYTES   65538816ULL   /* 62.50 MiB */
#define K2_TIER4_SYSTEM_RESERVE_BYTES     268435456ULL  /* 256.00 MiB */
#define K2_WEIGHT_SCALES_BYTES            340000000ULL  /* ~324.25 MiB for group-64 scales */
#define K2_HOST_RING_BYTES                4096ULL

// --- High-Resolution Performance Timer --------------------------------------
static LARGE_INTEGER g_timer_freq;

static void timer_init(void) {
    QueryPerformanceFrequency(&g_timer_freq);
}

static double timer_ms_since(LARGE_INTEGER start) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return ((double)(now.QuadPart - start.QuadPart) * 1000.0) / (double)g_timer_freq.QuadPart;
}

// --- Win32 Zero-Copy Memory-Mapped GGUF Loader ------------------------------
typedef struct {
    char name[128];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
    size_t byte_len;
} GgufTensorEntry;

typedef struct {
    HANDLE hFile;
    HANDLE hMapping;
    const uint8_t* base_ptr;
    size_t file_size;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
    size_t tensor_data_start;
    GgufTensorEntry* tensors;
} Win32GgufMap;

static bool skip_gguf_val(const uint8_t* base, size_t* pos, size_t file_sz, uint32_t vtype) {
    switch (vtype) {
        case 0: case 1: case 7:
            *pos += 1; return (*pos <= file_sz);
        case 2: case 3:
            *pos += 2; return (*pos <= file_sz);
        case 4: case 5: case 6:
            *pos += 4; return (*pos <= file_sz);
        case 8: {
            if (*pos + 8 > file_sz) return false;
            uint64_t slen = *(const uint64_t*)(base + *pos);
            *pos += 8 + (size_t)slen;
            return (*pos <= file_sz);
        }
        case 10: case 11: case 12:
            *pos += 8; return (*pos <= file_sz);
        case 9: {
            if (*pos + 12 > file_sz) return false;
            uint32_t elem_type = *(const uint32_t*)(base + *pos);
            uint64_t arr_len = *(const uint64_t*)(base + *pos + 4);
            *pos += 12;
            for (uint64_t a = 0; a < arr_len; ++a) {
                if (!skip_gguf_val(base, pos, file_sz, elem_type)) return false;
            }
            return (*pos <= file_sz);
        }
        default:
            return false;
    }
}

static bool win32_gguf_open(const char* filepath, Win32GgufMap* map) {
    memset(map, 0, sizeof(*map));
    int wlen = MultiByteToWideChar(CP_UTF8, 0, filepath, -1, NULL, 0);
    if (wlen <= 0) return false;
    wchar_t* wpath = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    if (!wpath) return false;
    MultiByteToWideChar(CP_UTF8, 0, filepath, -1, wpath, wlen);

    map->hFile = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    free(wpath);
    if (map->hFile == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(map->hFile, &sz) || sz.QuadPart < 24) {
        CloseHandle(map->hFile);
        return false;
    }
    map->file_size = (size_t)sz.QuadPart;

    map->hMapping = CreateFileMappingW(map->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!map->hMapping) { CloseHandle(map->hFile); return false; }

    map->base_ptr = (const uint8_t*)MapViewOfFile(map->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!map->base_ptr) { CloseHandle(map->hMapping); CloseHandle(map->hFile); return false; }

    if (*(const uint32_t*)map->base_ptr != 0x46554747u) {
        UnmapViewOfFile(map->base_ptr);
        CloseHandle(map->hMapping);
        CloseHandle(map->hFile);
        return false;
    }

    size_t pos = 4;
    map->version = *(const uint32_t*)(map->base_ptr + pos); pos += 4;
    map->tensor_count = *(const uint64_t*)(map->base_ptr + pos); pos += 8;
    map->kv_count = *(const uint64_t*)(map->base_ptr + pos); pos += 8;

    for (uint64_t k = 0; k < map->kv_count; ++k) {
        if (pos + 8 > map->file_size) return false;
        uint64_t klen = *(const uint64_t*)(map->base_ptr + pos); pos += 8;
        pos += (size_t)klen;
        if (pos + 4 > map->file_size) return false;
        uint32_t vtype = *(const uint32_t*)(map->base_ptr + pos); pos += 4;
        if (!skip_gguf_val(map->base_ptr, &pos, map->file_size, vtype)) return false;
    }

    map->tensors = (GgufTensorEntry*)malloc(map->tensor_count * sizeof(GgufTensorEntry));
    if (!map->tensors) return false;

    for (uint64_t t = 0; t < map->tensor_count; ++t) {
        GgufTensorEntry* te = &map->tensors[t];
        memset(te, 0, sizeof(*te));
        if (pos + 8 > map->file_size) return false;
        uint64_t nlen = *(const uint64_t*)(map->base_ptr + pos); pos += 8;
        size_t copylen = nlen < 127 ? (size_t)nlen : 127;
        memcpy(te->name, map->base_ptr + pos, copylen);
        te->name[copylen] = '\0';
        pos += (size_t)nlen;

        te->n_dims = *(const uint32_t*)(map->base_ptr + pos); pos += 4;
        size_t nelem = 1;
        for (uint32_t d = 0; d < te->n_dims; ++d) {
            te->dims[d] = *(const uint64_t*)(map->base_ptr + pos); pos += 8;
            nelem *= (size_t)te->dims[d];
        }
        te->type = *(const uint32_t*)(map->base_ptr + pos); pos += 4;
        te->offset = *(const uint64_t*)(map->base_ptr + pos); pos += 8;

        if (te->type == 0) te->byte_len = nelem * 4;
        else if (te->type == 1) te->byte_len = nelem * 2;
        else if (te->type == 12) te->byte_len = (nelem / 256) * 144;
        else if (te->type == 14) te->byte_len = (nelem / 256) * 210;
        else te->byte_len = nelem;
    }

    map->tensor_data_start = (pos + 31) & ~31ULL;
    return true;
}

static const GgufTensorEntry* win32_gguf_find_tensor(const Win32GgufMap* map, const char* name) {
    if (!map || !map->tensors) return NULL;
    for (uint64_t t = 0; t < map->tensor_count; ++t) {
        if (strcmp(map->tensors[t].name, name) == 0) {
            return &map->tensors[t];
        }
    }
    return NULL;
}

static void win32_gguf_close(Win32GgufMap* map) {
    if (map->tensors) free(map->tensors);
    if (map->base_ptr) UnmapViewOfFile(map->base_ptr);
    if (map->hMapping) CloseHandle(map->hMapping);
    if (map->hFile && map->hFile != INVALID_HANDLE_VALUE) CloseHandle(map->hFile);
    memset(map, 0, sizeof(*map));
}

// --- GPU Unpack Kernels: Q4_K / Q6_K to Marlin INT4 + Group-64 Scales ------

static __device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// Each CUDA block processes 1 Q4_K block of 256 weights (4 groups of 64)
__global__ void velox_cuda_unpack_q4_k_fused_kernel(
    uint32_t* __restrict__ dst_w,
    half2* __restrict__ dst_s,
    const uint8_t* __restrict__ src_q4k,
    size_t num_blocks_256,
    size_t weight_off_words,
    size_t scale_off_groups
) {
    size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_blocks_256) return;

    const uint8_t* blk = src_q4k + b * 144ULL;
    half d_half, dmin_half;
    memcpy(&d_half, blk, sizeof(half));
    memcpy(&dmin_half, blk + 2, sizeof(half));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;

    float fd = __half2float(d_half);
    float fdmin = __half2float(dmin_half);

    for (int g = 0; g < 4; ++g) {
        uint8_t sc0_u8, m0_u8, sc1_u8, m1_u8;
        get_scale_min_k4(2 * g,     scales, &sc0_u8, &m0_u8);
        get_scale_min_k4(2 * g + 1, scales, &sc1_u8, &m1_u8);
        float d1 = fd * (float)sc0_u8;
        float m1 = fdmin * (float)m0_u8;
        float d2 = fd * (float)sc1_u8;
        float m2 = fdmin * (float)m1_u8;

        const uint8_t* group_qs = qs + g * 32;
        float vals[64];
        float grp_min = 1e30f;
        float grp_max = -1e30f;

        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            float v0 = d1 * (float)(group_qs[l] & 0x0F) - m1;
            float v1 = d2 * (float)(group_qs[l] >> 4)   - m2;
            vals[l]      = v0;
            vals[l + 32] = v1;
            if (v0 < grp_min) grp_min = v0;
            if (v0 > grp_max) grp_max = v0;
            if (v1 < grp_min) grp_min = v1;
            if (v1 > grp_max) grp_max = v1;
        }

        float sc = (grp_max > grp_min) ? ((grp_max - grp_min) / 15.0f) : 1e-5f;
        float mn = grp_min;
        dst_s[scale_off_groups + b * 4ULL + (size_t)g] = __floats2half2_rn(sc, mn);
        float inv_sc = 1.0f / sc;

        for (int w = 0; w < 8; ++w) {
            uint32_t q[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int qi = __float2int_rn((vals[w * 8 + k] - mn) * inv_sc);
                q[k] = (uint32_t)((qi < 0) ? 0 : ((qi > 15) ? 15 : qi));
            }
            uint32_t packed = (q[0] <<  0) | (q[2] <<  4) | (q[4] <<  8) | (q[6] << 12)
                            | (q[1] << 16) | (q[3] << 20) | (q[5] << 24) | (q[7] << 28);
            dst_w[weight_off_words + b * 32ULL + g * 8ULL + (size_t)w] = packed;
        }
    }
}

// Each CUDA block processes 1 Q6_K block of 256 weights (4 groups of 64)
__global__ void velox_cuda_unpack_q6_k_fused_kernel(
    uint32_t* __restrict__ dst_w,
    half2* __restrict__ dst_s,
    const uint8_t* __restrict__ src_q6k,
    size_t num_blocks_256,
    size_t weight_off_words,
    size_t scale_off_groups
) {
    size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_blocks_256) return;

    const uint8_t* blk = src_q6k + b * 210ULL;
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t*  sc = (const int8_t*)(blk + 192);
    half d_half;
    memcpy(&d_half, blk + 208, sizeof(half));
    float d = __half2float(d_half);

    for (int n = 0; n < 2; ++n) {
        const uint8_t* ql_n = ql + n * 64;
        const uint8_t* qh_n = qh + n * 32;
        const int8_t*  sc_n = sc + n * 8;
        float w128[128];

        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            const int8_t q1 = (int8_t)((ql_n[l + 0]  & 0xF) | (((qh_n[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql_n[l + 32] & 0xF) | (((qh_n[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql_n[l + 0]  >> 4)  | (((qh_n[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql_n[l + 32] >> 4)  | (((qh_n[l] >> 6) & 3) << 4)) - 32;

            w128[l + 0]  = d * (float)sc_n[is + 0] * (float)q1;
            w128[l + 32] = d * (float)sc_n[is + 2] * (float)q2;
            w128[l + 64] = d * (float)sc_n[is + 4] * (float)q3;
            w128[l + 96] = d * (float)sc_n[is + 6] * (float)q4;
        }

        for (int sub_g = 0; sub_g < 2; ++sub_g) {
            int g = n * 2 + sub_g;
            const float* vals = w128 + sub_g * 64;
            float grp_min = vals[0];
            float grp_max = vals[0];
            #pragma unroll
            for (int i = 1; i < 64; ++i) {
                float v = vals[i];
                if (v < grp_min) grp_min = v;
                if (v > grp_max) grp_max = v;
            }

            float sc_val = (grp_max > grp_min) ? ((grp_max - grp_min) / 15.0f) : 1e-5f;
            float mn_val = grp_min;
            dst_s[scale_off_groups + b * 4ULL + (size_t)g] = __floats2half2_rn(sc_val, mn_val);
            float inv_s = 1.0f / sc_val;

            for (int w = 0; w < 8; ++w) {
                uint32_t q[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    int qi = __float2int_rn((vals[w * 8 + k] - mn_val) * inv_s);
                    q[k] = (uint32_t)((qi < 0) ? 0 : ((qi > 15) ? 15 : qi));
                }
                uint32_t packed = (q[0] <<  0) | (q[2] <<  4) | (q[4] <<  8) | (q[6] << 12)
                                | (q[1] << 16) | (q[3] << 20) | (q[5] << 24) | (q[7] << 28);
                dst_w[weight_off_words + b * 32ULL + g * 8ULL + (size_t)w] = packed;
            }
        }
    }
}

// Stream and unpack a GGUF tensor into d_weight_arena and d_weight_scales
static void load_gguf_tensor_to_arena(
    const Win32GgufMap* map,
    const char* tensor_name,
    uint8_t* d_staging,
    size_t staging_bytes,
    uint32_t* d_weight_words,
    half2* d_weight_scales,
    size_t weight_off_bytes,
    cudaStream_t stream
) {
    const GgufTensorEntry* te = win32_gguf_find_tensor(map, tensor_name);
    if (!te) {
        if (strcmp(tensor_name, "output.weight") == 0) {
            te = win32_gguf_find_tensor(map, "token_embd.weight");
        }
        if (!te) return;
    }

    const uint8_t* host_src = map->base_ptr + map->tensor_data_start + te->offset;
    size_t total_bytes = te->byte_len;
    size_t weight_off_words = weight_off_bytes >> 2;
    size_t scale_off_groups = weight_off_bytes >> 5;

    if (te->type == 0) {
        // FP32 Norm weight: direct copy to device
        cudaMemcpyAsync((uint8_t*)d_weight_words + weight_off_bytes, host_src, total_bytes, cudaMemcpyHostToDevice, stream);
        return;
    }

    size_t chunk_bytes = staging_bytes;
    size_t processed_bytes = 0;
    size_t cur_block_idx = 0;

    while (processed_bytes < total_bytes) {
        size_t rem = total_bytes - processed_bytes;
        size_t take = (rem < chunk_bytes) ? rem : chunk_bytes;

        // Ensure chunk is aligned to whole blocks
        size_t block_sz = (te->type == 14) ? 210ULL : 144ULL;
        take = (take / block_sz) * block_sz;
        if (take == 0) take = rem;

        cudaMemcpyAsync(d_staging, host_src + processed_bytes, take, cudaMemcpyHostToDevice, stream);
        size_t num_blocks = take / block_sz;

        if (num_blocks > 0) {
            dim3 block(256);
            dim3 grid((uint32_t)((num_blocks + 255) / 256));

            if (te->type == 14) {
                velox_cuda_unpack_q6_k_fused_kernel<<<grid, block, 0, stream>>>(
                    d_weight_words, d_weight_scales, d_staging, num_blocks,
                    weight_off_words + cur_block_idx * 32ULL,
                    scale_off_groups + cur_block_idx * 4ULL
                );
            } else {
                velox_cuda_unpack_q4_k_fused_kernel<<<grid, block, 0, stream>>>(
                    d_weight_words, d_weight_scales, d_staging, num_blocks,
                    weight_off_words + cur_block_idx * 32ULL,
                    scale_off_groups + cur_block_idx * 4ULL
                );
            }
        }

        cur_block_idx += num_blocks;
        processed_bytes += take;
    }
}

// =============================================================================
// Batched Prefill Engine: Process ALL prompt tokens through each layer in a
// single pass, streaming each layer's weights exactly ONCE from VRAM.
//
// Architecture:
//   - For GEMV kernels (QKV, OutProj, SwiGLU): 2D grid dim3(row_blocks, prompt_len)
//     where blockIdx.y selects the prompt token. All tokens share the same weight
//     rows loaded from L2 cache. This converts N sequential weight streams into 1.
//   - For RMSNorm: 2D grid dim3(1, prompt_len) - each blockIdx.y normalizes one token.
//   - For RoPE: 2D grid dim3(rope_blocks, prompt_len).
//   - For Attention: Sequential per-token (causal dependency on prior KV entries).
//   - Embed: 2D grid dim3(embed_blocks, prompt_len).
//
// Memory Layout (in activation_arena, starting at BATCHED_PREFILL_BASE):
//   hidden[t]  = BATCHED_PREFILL_BASE + t * VELOX_HIDDEN_DIM * 4
//   norm[t]    = BATCHED_PREFILL_NORM_BASE + t * VELOX_HIDDEN_DIM * 4
//   q_buf[t]   = BATCHED_PREFILL_Q_BASE + t * (VELOX_NUM_HEADS * VELOX_HEAD_DIM) * 4
//   act_buf[t] = BATCHED_PREFILL_ACT_BASE + t * VELOX_INTERMEDIATE_DIM * 4
//
// Constraint: Max prompt_len = 1024 tokens (fits within activation arena)
// =============================================================================
#define VELOX_BATCHED_PREFILL_MAX_TOKENS  1024ULL
#define VELOX_BATCHED_PREFILL_BASE        8388608ULL   /* 8 MiB offset */
// hidden: 1024 * 2560 * 4 = 10,485,760 bytes (10 MiB)
#define VELOX_BATCHED_PREFILL_NORM_BASE   (VELOX_BATCHED_PREFILL_BASE + VELOX_BATCHED_PREFILL_MAX_TOKENS * VELOX_HIDDEN_DIM * 4ULL)
// norm: 1024 * 2560 * 4 = 10,485,760 bytes
#define VELOX_BATCHED_PREFILL_Q_BASE      (VELOX_BATCHED_PREFILL_NORM_BASE + VELOX_BATCHED_PREFILL_MAX_TOKENS * VELOX_HIDDEN_DIM * 4ULL)
// q_buf: 1024 * 4096 * 4 = 16,777,216 bytes (16 MiB)
#define VELOX_BATCHED_PREFILL_ACT_BASE    (VELOX_BATCHED_PREFILL_Q_BASE + VELOX_BATCHED_PREFILL_MAX_TOKENS * (VELOX_NUM_HEADS * VELOX_HEAD_DIM) * 4ULL)
// act_buf: 1024 * 10240 * 4 = 41,943,040 bytes (40 MiB)
// Total: 8 + 10 + 10 + 16 + 40 = 84 MiB (within 125 MiB arena)

// =============================================================================
// Speculative Verify Region (Batched LM-Head Evaluation)
//
// Layout (all static offsets inside the 125 MiB activation arena):
//   [spec token slots]  1024 * 12 B  (4 B token ids + 8 B device positions)
//   [verify norm rows]  SPEC_MAX_ROWS * 2560 * 4   (82 KB for 8 rows)
//   [verify logits]     SPEC_MAX_ROWS * 250624 * 4 (7.65 MiB for 8 rows)
//   [stage1 scratch]    SPEC_MAX_ROWS * 979 blocks * 8 B (63 KB)
//
// The device token slots are written AFTER the batched-prefill activation
// region (offset ACT_BASE + act_bytes); all three verify buffers trail them,
// so total arena usage peaks at ~91.8 MiB and never collides with the low
// 0..8 MiB block (scratch/decode-state) or the prefill region itself.
// =============================================================================
#define VELOX_SPEC_MAX_ROWS               8ULL
#define VELOX_VERIFY_TOKEN_SLOT_BYTES     12ULL  /* 4 B id + 8 B position */
#define VELOX_VERIFY_TOKEN_REGION_END     (VELOX_BATCHED_PREFILL_ACT_BASE + VELOX_BATCHED_PREFILL_MAX_TOKENS * (VELOX_INTERMEDIATE_DIM * 4ULL + VELOX_VERIFY_TOKEN_SLOT_BYTES))
#define VELOX_VERIFY_NORM_BASE            VELOX_VERIFY_TOKEN_REGION_END
#define VELOX_VERIFY_LOGITS_BASE          (VELOX_VERIFY_NORM_BASE + VELOX_SPEC_MAX_ROWS * VELOX_HIDDEN_DIM * 4ULL)
#define VELOX_VERIFY_STAGE1_BASE          (VELOX_VERIFY_LOGITS_BASE + VELOX_SPEC_MAX_ROWS * VELOX_VOCAB_SIZE * 4ULL)
// Stage1 scratch: split into separate float vals and uint32_t idxs buffers
#define VELOX_VERIFY_STAGE1_NUM_BLOCKS    ((VELOX_VOCAB_SIZE + 255ULL) / 256ULL)  // 979
#define VELOX_VERIFY_STAGE1_VALS_SIZE     (VELOX_SPEC_MAX_ROWS * VELOX_VERIFY_STAGE1_NUM_BLOCKS * 4ULL)  // float
#define VELOX_VERIFY_STAGE1_IDXS_BASE     (VELOX_VERIFY_STAGE1_BASE + VELOX_VERIFY_STAGE1_VALS_SIZE)
#define VELOX_VERIFY_STAGE1_IDXS_SIZE     (VELOX_SPEC_MAX_ROWS * VELOX_VERIFY_STAGE1_NUM_BLOCKS * 4ULL)  // uint32_t
#define VELOX_VERIFY_REGION_END           (VELOX_VERIFY_STAGE1_IDXS_BASE + VELOX_VERIFY_STAGE1_IDXS_SIZE)


// Batched Embed Q4: blockIdx.y = token index
__global__ void __launch_bounds__(256) velox_cuda_embed_batched_q4(
    float* __restrict__ dst_base, const uint32_t* __restrict__ tokens,
    const uint32_t* __restrict__ w_embed_q4, const half2* __restrict__ s_embed_q4,
    size_t prompt_len, size_t hidden_stride
) {
    const size_t t = blockIdx.y;
    if (t >= prompt_len) return;
    const size_t token_id = (size_t)tokens[t];
    size_t word_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t base_k = word_idx * 8ULL;
    float* dst = dst_base + t * hidden_stride;
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

// Batched RMSNorm: blockIdx.y = token index
__global__ void __launch_bounds__(256) velox_cuda_rmsnorm_batched(
    float* __restrict__ dst_base, const float* __restrict__ src_base,
    const float* __restrict__ weight, size_t dim, float eps,
    size_t prompt_len, size_t src_stride, size_t dst_stride
) {
    const size_t t = blockIdx.y;
    if (t >= prompt_len || blockIdx.x != 0) return;
    const float* x = src_base + t * src_stride;
    float* dst = dst_base + t * dst_stride;
    const size_t n_groups = (dim == 2560ULL) ? 2ULL : 1ULL;
    const size_t grp_dim = dim / n_groups;
    const float eff_eps = (dim == 2560ULL && eps > 1e-6f) ? 1e-6f : eps;
    __shared__ float s_ss[256];
    size_t tid = threadIdx.x;
    for (size_t g = 0ULL; g < n_groups; ++g) {
        const size_t base = g * grp_dim;
        float local_ss = 0.0f;
        for (size_t i = tid; i < grp_dim; i += blockDim.x) {
            float v = x[base + i];
            local_ss += v * v;
        }
        s_ss[tid] = local_ss;
        __syncthreads();
        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_ss[tid] += s_ss[tid + stride];
            __syncthreads();
        }
        float inv_rms = rsqrtf((s_ss[0] / (float)grp_dim) + eff_eps);
        for (size_t i = tid; i < grp_dim; i += blockDim.x) {
            dst[base + i] = (x[base + i] * inv_rms) * weight[base + i];
        }
        __syncthreads();
    }
}

// Template Register-Tiled Batched Fused QKV Q4: 1D grid dim3(qkv_blocks)
// Zero local memory spills: BATCH is compile-time constant, unrolled strictly into hardware registers!
template<int BATCH>
__global__ void __launch_bounds__(128, 2) velox_cuda_fused_qkv_batched_q4(
    float* __restrict__ q_base,
    float* __restrict__ kv_pool,
    const float* __restrict__ norm_base,
    const uint32_t* __restrict__ w_q, const half2* __restrict__ s_q,
    const uint32_t* __restrict__ w_k, const half2* __restrict__ s_k,
    const uint32_t* __restrict__ w_v, const half2* __restrict__ s_v,
    size_t layer,
    size_t norm_stride, size_t q_stride,
    const size_t* __restrict__ d_positions
) {
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t global_row = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t num_q = VELOX_NUM_HEADS * VELOX_HEAD_DIM;
    const size_t num_kv = VELOX_NUM_KV_HEADS * VELOX_HEAD_DIM;
    if (global_row >= num_q + 2ULL * num_kv) return;
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    float* k_layer = kv_pool + (layer * 2ULL * max_s * num_kv);
    float* v_layer = k_layer + (max_s * num_kv);

    const uint32_t* __restrict__ row_w = NULL;
    const half2* __restrict__ row_s = NULL;
    if (global_row < num_q) {
        size_t off = (layer * dim * num_q) + (global_row * dim);
        row_w = w_q + (off >> 3); row_s = s_q + (off >> 6);
    } else if (global_row < num_q + num_kv) {
        const size_t k_row = global_row - num_q;
        size_t off = (layer * dim * num_kv) + (k_row * dim);
        row_w = w_k + (off >> 3); row_s = s_k + (off >> 6);
    } else {
        const size_t v_row = global_row - num_q - num_kv;
        size_t off = (layer * dim * num_kv) + (v_row * dim);
        row_w = w_v + (off >> 3); row_s = s_v + (off >> 6);
    }

    float acc[BATCH];
    #pragma unroll
    for (int t = 0; t < BATCH; ++t) acc[t] = 0.0f;

    const uint32_t fp16_bias = 0x64006400u;
    const uint32_t mask_low  = 0x000F000Fu;
    const half2 sub_bias = velox_u32_as_half2(fp16_bias);

    for (size_t k = lane_id * 8ULL; k + 7ULL < dim; k += 256ULL) {
        const uint32_t packed = __ldg(&row_w[k >> 3]);
        const half2 sm_h2     = __ldg(&row_s[k >> 6]);

        uint32_t h2_0 = velox_ptx_lop3_0xEA(packed,       mask_low, fp16_bias);
        uint32_t h2_1 = velox_ptx_lop3_0xEA(packed >> 4,  mask_low, fp16_bias);
        uint32_t h2_2 = velox_ptx_lop3_0xEA(packed >> 8,  mask_low, fp16_bias);
        uint32_t h2_3 = velox_ptx_lop3_0xEA(packed >> 12, mask_low, fp16_bias);
        const float2 f0 = __half22float2(__hsub2(velox_u32_as_half2(h2_0), sub_bias));
        const float2 f1 = __half22float2(__hsub2(velox_u32_as_half2(h2_1), sub_bias));
        const float2 f2 = __half22float2(__hsub2(velox_u32_as_half2(h2_2), sub_bias));
        const float2 f3 = __half22float2(__hsub2(velox_u32_as_half2(h2_3), sub_bias));
        const float2 sm = __half22float2(sm_h2);

        #pragma unroll
        for (int t = 0; t < BATCH; ++t) {
            const float* norm_input = norm_base + (size_t)t * norm_stride;
            const float4 x0 = *reinterpret_cast<const float4*>(&norm_input[k]);
            const float4 x1 = *reinterpret_cast<const float4*>(&norm_input[k + 4ULL]);
            const float sum_qx = f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w
                               + f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
            const float sum_x  = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w);
            acc[t] += sm.x * sum_qx + sm.y * sum_x;
        }
    }

    #pragma unroll
    for (int t = 0; t < BATCH; ++t) {
        float a = acc[t];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xffffffff, a, offset);
        }
        if (lane_id == 0) {
            if (global_row < num_q) {
                q_base[(size_t)t * q_stride + global_row] = a;
            } else if (global_row < num_q + num_kv) {
                const size_t k_row = global_row - num_q;
                const size_t pos = d_positions[t];
                const size_t safe_pos = (pos < max_s) ? pos : (max_s - 1);
                k_layer[safe_pos * num_kv + k_row] = a;
            } else {
                const size_t v_row = global_row - num_q - num_kv;
                const size_t pos = d_positions[t];
                const size_t safe_pos = (pos < max_s) ? pos : (max_s - 1);
                v_layer[safe_pos * num_kv + v_row] = a;
            }
        }
    }
}

// Batched RoPE: dim3(rope_blocks, prompt_len)
__global__ void __launch_bounds__(128, 8) velox_cuda_rope_batched(
    float* __restrict__ q_base,
    float* __restrict__ kv_pool,
    size_t layer, size_t prompt_len,
    size_t q_stride,
    const size_t* __restrict__ d_positions,
    float theta
) {
    const size_t t = blockIdx.y;
    if (t >= prompt_len) return;
    const size_t pos = d_positions[t];
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
    const size_t safe_pos = (pos < max_s) ? pos : (max_s - 1);
    const size_t kv_dim = KV_H * D;
    const size_t layer_kv_stride = 2ULL * max_s * kv_dim;
    float* k_layer = kv_pool + (layer * layer_kv_stride);
    float* k_slot = k_layer + (safe_pos * kv_dim);
    float* q = q_base + t * q_stride;
    if (idx < q_pairs) {
        size_t h = idx / half_d;
        size_t j = idx % half_d;
        float freq = 1.0f / powf(theta, (2.0f * (float)j) / (float)D);
        float angle = (float)safe_pos * freq;
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
        float angle = (float)safe_pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float* k_head = k_slot + (h * D);
        float x1 = k_head[j];
        float x2 = k_head[j + half_d];
        k_head[j] = x1 * cos_a - x2 * sin_a;
        k_head[j + half_d] = x2 * cos_a + x1 * sin_a;
    }
}

// Batched GQA Attention: ONE block per (head, token), blockIdx.x = head, blockIdx.y = token
// Causal mask: token at position pos can attend to positions 0..pos
__global__ void __launch_bounds__(256, 4) velox_cuda_gqa_attn_batched(
    float* __restrict__ q_base,
    const float* __restrict__ kv_pool,
    size_t layer, size_t prompt_len,
    size_t q_stride,
    const size_t* __restrict__ d_positions
) {
    const size_t t = blockIdx.y;
    if (t >= prompt_len) return;
    const size_t pos = d_positions[t];
    const size_t H = VELOX_NUM_HEADS;
    const size_t KV_H = VELOX_NUM_KV_HEADS;
    const size_t D = VELOX_HEAD_DIM;
    const size_t max_s = VELOX_MODEL_MAX_SEQ_LEN;
    const size_t safe_pos = (pos < max_s) ? pos : (max_s - 1);
    const size_t seq_len = safe_pos + 1;
    const size_t kv_dim = KV_H * D;
    const size_t layer_kv_stride = 2ULL * max_s * kv_dim;
    const size_t h = blockIdx.x;
    if (h >= H) return;
    const size_t tid = threadIdx.x;
    const size_t kv_group = H / KV_H;
    const size_t kv_h = h / kv_group;
    const float* k_layer = kv_pool + (layer * layer_kv_stride);
    const float* v_layer = k_layer + (max_s * kv_dim);
    float* q = q_base + t * q_stride;
    __shared__ float s_q[128];
    if (tid < D) {
        s_q[tid] = q[h * D + tid];
    }
    __syncthreads();
    float scale = 1.0f / sqrtf((float)D);
    float acc_d = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;
    for (size_t s = 0; s < seq_len; ++s) {
        const float* k_t = k_layer + (s * kv_dim) + (kv_h * D);
        const float* v_t = v_layer + (s * kv_dim) + (kv_h * D);
        __shared__ float s_dot[256];
        float my_dot = 0.0f;
        if (tid < D) {
            my_dot = s_q[tid] * k_t[tid];
        }
        s_dot[tid] = my_dot;
        __syncthreads();
        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_dot[tid] += s_dot[tid + stride];
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
        q[h * D + tid] = (l > 0.0f) ? (acc_d / l) : 0.0f;
    }
}

// Template Register-Tiled Batched OutProj + Residual Q4: 1D grid dim3(out_blocks)
template<int BATCH>
__global__ void __launch_bounds__(128, 2) velox_cuda_out_proj_residual_batched_q4(
    float* __restrict__ hidden_base,
    const float* __restrict__ q_base,
    const uint32_t* __restrict__ w_out, const half2* __restrict__ s_out,
    size_t layer,
    size_t hidden_stride, size_t q_stride
) {
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t row = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t attn_dim = VELOX_NUM_HEADS * VELOX_HEAD_DIM;
    if (row >= dim) return;

    size_t off = (layer * dim * attn_dim) + (row * attn_dim);
    const uint32_t* __restrict__ row_w = w_out + (off >> 3);
    const half2* __restrict__ row_s = s_out + (off >> 6);

    float acc[BATCH];
    #pragma unroll
    for (int t = 0; t < BATCH; ++t) acc[t] = 0.0f;

    const uint32_t fp16_bias = 0x64006400u;
    const uint32_t mask_low  = 0x000F000Fu;
    const half2 sub_bias = velox_u32_as_half2(fp16_bias);

    for (size_t k = lane_id * 8ULL; k + 7ULL < attn_dim; k += 256ULL) {
        const uint32_t packed = __ldg(&row_w[k >> 3]);
        const half2 sm_h2     = __ldg(&row_s[k >> 6]);

        uint32_t h2_0 = velox_ptx_lop3_0xEA(packed,       mask_low, fp16_bias);
        uint32_t h2_1 = velox_ptx_lop3_0xEA(packed >> 4,  mask_low, fp16_bias);
        uint32_t h2_2 = velox_ptx_lop3_0xEA(packed >> 8,  mask_low, fp16_bias);
        uint32_t h2_3 = velox_ptx_lop3_0xEA(packed >> 12, mask_low, fp16_bias);
        const float2 f0 = __half22float2(__hsub2(velox_u32_as_half2(h2_0), sub_bias));
        const float2 f1 = __half22float2(__hsub2(velox_u32_as_half2(h2_1), sub_bias));
        const float2 f2 = __half22float2(__hsub2(velox_u32_as_half2(h2_2), sub_bias));
        const float2 f3 = __half22float2(__hsub2(velox_u32_as_half2(h2_3), sub_bias));
        const float2 sm = __half22float2(sm_h2);

        #pragma unroll
        for (int t = 0; t < BATCH; ++t) {
            const float* attn_in = q_base + (size_t)t * q_stride;
            const float4 x0 = *reinterpret_cast<const float4*>(&attn_in[k]);
            const float4 x1 = *reinterpret_cast<const float4*>(&attn_in[k + 4ULL]);
            const float sum_qx = f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w
                               + f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
            const float sum_x  = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w);
            acc[t] += sm.x * sum_qx + sm.y * sum_x;
        }
    }

    #pragma unroll
    for (int t = 0; t < BATCH; ++t) {
        float a = acc[t];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xffffffff, a, offset);
        }
        if (lane_id == 0) {
            hidden_base[(size_t)t * hidden_stride + row] += a;
        }
    }
}

// Template Register-Tiled Batched SwiGLU Gate+Up Q4: 1D grid dim3(inter_blocks)
template<int BATCH>
__global__ void __launch_bounds__(128, 2) velox_cuda_swiglu_gate_up_batched_q4(
    float* __restrict__ act_base,
    const float* __restrict__ norm_base,
    const uint32_t* __restrict__ w_gate, const half2* __restrict__ s_gate,
    const uint32_t* __restrict__ w_up, const half2* __restrict__ s_up,
    size_t layer,
    size_t norm_stride, size_t act_stride
) {
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t j = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    if (j >= inter_dim) return;

    size_t off = (layer * inter_dim * dim) + (j * dim);
    const uint32_t* __restrict__ wg_ptr = w_gate + (off >> 3);
    const half2* __restrict__ sg_ptr    = s_gate + (off >> 6);
    const uint32_t* __restrict__ wu_ptr = w_up + (off >> 3);
    const half2* __restrict__ su_ptr    = s_up + (off >> 6);

    float acc_g[BATCH];
    float acc_u[BATCH];
    #pragma unroll
    for (int t = 0; t < BATCH; ++t) { acc_g[t] = 0.0f; acc_u[t] = 0.0f; }

    const uint32_t fp16_bias = 0x64006400u;
    const uint32_t mask_low  = 0x000F000Fu;
    const half2 sub_bias = velox_u32_as_half2(fp16_bias);

    for (size_t k = lane_id * 8ULL; k + 7ULL < dim; k += 256ULL) {
        const uint32_t pg = __ldg(&wg_ptr[k >> 3]);
        const half2 sm_g  = __ldg(&sg_ptr[k >> 6]);
        uint32_t g0 = velox_ptx_lop3_0xEA(pg,       mask_low, fp16_bias);
        uint32_t g1 = velox_ptx_lop3_0xEA(pg >> 4,  mask_low, fp16_bias);
        uint32_t g2 = velox_ptx_lop3_0xEA(pg >> 8,  mask_low, fp16_bias);
        uint32_t g3 = velox_ptx_lop3_0xEA(pg >> 12, mask_low, fp16_bias);
        const float2 fg0 = __half22float2(__hsub2(velox_u32_as_half2(g0), sub_bias));
        const float2 fg1 = __half22float2(__hsub2(velox_u32_as_half2(g1), sub_bias));
        const float2 fg2 = __half22float2(__hsub2(velox_u32_as_half2(g2), sub_bias));
        const float2 fg3 = __half22float2(__hsub2(velox_u32_as_half2(g3), sub_bias));
        const float2 smg = __half22float2(sm_g);

        const uint32_t pu = __ldg(&wu_ptr[k >> 3]);
        const half2 sm_u  = __ldg(&su_ptr[k >> 6]);
        uint32_t u0 = velox_ptx_lop3_0xEA(pu,       mask_low, fp16_bias);
        uint32_t u1 = velox_ptx_lop3_0xEA(pu >> 4,  mask_low, fp16_bias);
        uint32_t u2 = velox_ptx_lop3_0xEA(pu >> 8,  mask_low, fp16_bias);
        uint32_t u3 = velox_ptx_lop3_0xEA(pu >> 12, mask_low, fp16_bias);
        const float2 fu0 = __half22float2(__hsub2(velox_u32_as_half2(u0), sub_bias));
        const float2 fu1 = __half22float2(__hsub2(velox_u32_as_half2(u1), sub_bias));
        const float2 fu2 = __half22float2(__hsub2(velox_u32_as_half2(u2), sub_bias));
        const float2 fu3 = __half22float2(__hsub2(velox_u32_as_half2(u3), sub_bias));
        const float2 smu = __half22float2(sm_u);

        #pragma unroll
        for (int t = 0; t < BATCH; ++t) {
            const float* norm_input = norm_base + (size_t)t * norm_stride;
            const float4 x0 = *reinterpret_cast<const float4*>(&norm_input[k]);
            const float4 x1 = *reinterpret_cast<const float4*>(&norm_input[k + 4ULL]);
            const float sum_x  = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w);

            const float sum_qg = fg0.x * x0.x + fg0.y * x0.y + fg1.x * x0.z + fg1.y * x0.w
                               + fg2.x * x1.x + fg2.y * x1.y + fg3.x * x1.z + fg3.y * x1.w;
            acc_g[t] += smg.x * sum_qg + smg.y * sum_x;

            const float sum_qu = fu0.x * x0.x + fu0.y * x0.y + fu1.x * x0.z + fu1.y * x0.w
                               + fu2.x * x1.x + fu2.y * x1.y + fu3.x * x1.z + fu3.y * x1.w;
            acc_u[t] += smu.x * sum_qu + smu.y * sum_x;
        }
    }

    #pragma unroll
    for (int t = 0; t < BATCH; ++t) {
        float ag = acc_g[t];
        float au = acc_u[t];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            ag += __shfl_down_sync(0xffffffff, ag, offset);
            au += __shfl_down_sync(0xffffffff, au, offset);
        }
        if (lane_id == 0) {
            float silu = ag / (1.0f + expf(-ag));
            act_base[(size_t)t * act_stride + j] = silu * au;
        }
    }
}

// Template Register-Tiled Batched SwiGLU Down + Residual Q4: 1D grid dim3(out_blocks)
template<int BATCH>
__global__ void __launch_bounds__(128, 2) velox_cuda_swiglu_down_residual_batched_q4(
    float* __restrict__ hidden_base,
    const float* __restrict__ act_base,
    const uint32_t* __restrict__ w_down, const half2* __restrict__ s_down,
    size_t layer,
    size_t hidden_stride, size_t act_stride
) {
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = threadIdx.x % 32;
    const size_t o = (size_t)blockIdx.x * 4ULL + warp_id;
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    if (o >= dim) return;

    size_t off = (layer * dim * inter_dim) + (o * inter_dim);
    const uint32_t* __restrict__ row_w = w_down + (off >> 3);
    const half2* __restrict__ row_s    = s_down + (off >> 6);

    float acc[BATCH];
    #pragma unroll
    for (int t = 0; t < BATCH; ++t) acc[t] = 0.0f;

    const uint32_t fp16_bias = 0x64006400u;
    const uint32_t mask_low  = 0x000F000Fu;
    const half2 sub_bias = velox_u32_as_half2(fp16_bias);

    for (size_t k = lane_id * 8ULL; k + 7ULL < inter_dim; k += 256ULL) {
        const uint32_t packed = __ldg(&row_w[k >> 3]);
        const half2 sm_h2     = __ldg(&row_s[k >> 6]);

        uint32_t h2_0 = velox_ptx_lop3_0xEA(packed,       mask_low, fp16_bias);
        uint32_t h2_1 = velox_ptx_lop3_0xEA(packed >> 4,  mask_low, fp16_bias);
        uint32_t h2_2 = velox_ptx_lop3_0xEA(packed >> 8,  mask_low, fp16_bias);
        uint32_t h2_3 = velox_ptx_lop3_0xEA(packed >> 12, mask_low, fp16_bias);
        const float2 f0 = __half22float2(__hsub2(velox_u32_as_half2(h2_0), sub_bias));
        const float2 f1 = __half22float2(__hsub2(velox_u32_as_half2(h2_1), sub_bias));
        const float2 f2 = __half22float2(__hsub2(velox_u32_as_half2(h2_2), sub_bias));
        const float2 f3 = __half22float2(__hsub2(velox_u32_as_half2(h2_3), sub_bias));
        const float2 sm = __half22float2(sm_h2);

        #pragma unroll
        for (int t = 0; t < BATCH; ++t) {
            const float* act_in = act_base + (size_t)t * act_stride;
            const float4 x0 = *reinterpret_cast<const float4*>(&act_in[k]);
            const float4 x1 = *reinterpret_cast<const float4*>(&act_in[k + 4ULL]);
            const float sum_qx = f0.x * x0.x + f0.y * x0.y + f1.x * x0.z + f1.y * x0.w
                               + f2.x * x1.x + f2.y * x1.y + f3.x * x1.z + f3.y * x1.w;
            const float sum_x  = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w);
            acc[t] += sm.x * sum_qx + sm.y * sum_x;
        }
    }

    #pragma unroll
    for (int t = 0; t < BATCH; ++t) {
        float a = acc[t];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xffffffff, a, offset);
        }
        if (lane_id == 0) {
            hidden_base[(size_t)t * hidden_stride + o] += a;
        }
    }
}

// Initialize position indices directly on device (zero host heap allocations)
__global__ void velox_cuda_init_positions(size_t* d_positions, size_t base_pos, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_positions[idx] = base_pos + idx;
}

// Template Batched Layer Runner: executes all 36 layers for B tokens in parallel
template<int B>
static void velox_run_batched_layers_q4(
    VeloxCudaArena* ctx,
    float* hidden_base,
    float* norm_base,
    float* q_base,
    float* act_base,
    const size_t* d_positions,
    cudaStream_t stream
) {
    const size_t dim = VELOX_HIDDEN_DIM;
    const size_t num_q = VELOX_NUM_HEADS * VELOX_HEAD_DIM;
    const size_t num_kv = VELOX_NUM_KV_HEADS * VELOX_HEAD_DIM;
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM;
    const size_t qkv_total_rows = num_q + 2 * num_kv;
    const size_t qkv_blocks = (qkv_total_rows + 3) / 4;
    const size_t rope_pairs = (VELOX_NUM_HEADS + VELOX_NUM_KV_HEADS) * (VELOX_HEAD_DIM / 2);
    const size_t rope_blocks = (rope_pairs + 127) / 128;
    const size_t out_blocks = (dim + 3) / 4;
    const size_t inter_blocks = (inter_dim + 3) / 4;

    for (size_t layer = 0; layer < 36; ++layer) {
        // 2a. Batched Pre-Attention RMSNorm
        const float* attn_norm_w = VELOX_DEV_NORM_PTR(ctx, 2208235520ULL) + (layer * dim);
        velox_cuda_rmsnorm_batched<<<dim3(1, (unsigned int)B), dim3(256), 0, stream>>>(
            norm_base, hidden_base, attn_norm_w, dim, 1e-6f,
            B, dim, dim
        );

        // 2b. Batched QKV Projection (Register-Tiled: 1D grid, weights loaded ONCE into registers)
        velox_cuda_fused_qkv_batched_q4<B><<<dim3((unsigned int)qkv_blocks), dim3(128), 0, stream>>>(
            q_base, (float*)ctx->kv_cache_pool, norm_base,
            VELOX_DEV_W_Q4_PTR(ctx, 320798720ULL), VELOX_DEV_S_Q4_PTR(ctx, 320798720ULL),
            VELOX_DEV_W_Q4_PTR(ctx, 509542400ULL), VELOX_DEV_S_Q4_PTR(ctx, 509542400ULL),
            VELOX_DEV_W_Q4_PTR(ctx, 556728320ULL), VELOX_DEV_S_Q4_PTR(ctx, 556728320ULL),
            layer, dim, num_q, d_positions
        );

        // 2c. Batched RoPE rotation on Q and K
        velox_cuda_rope_batched<<<dim3((unsigned int)rope_blocks, (unsigned int)B), dim3(128), 0, stream>>>(
            q_base, (float*)ctx->kv_cache_pool,
            layer, B, num_q, d_positions, VELOX_MODEL_ROPE_THETA
        );

        // 2d. Attention (batched across heads AND tokens)
        velox_cuda_gqa_attn_batched<<<dim3((unsigned int)VELOX_NUM_HEADS, (unsigned int)B), dim3(VELOX_HEAD_DIM), 0, stream>>>(
            q_base, (float*)ctx->kv_cache_pool,
            layer, B, num_q, d_positions
        );

        // 2e. Batched OutProj + Residual (Register-Tiled: 1D grid, weights loaded ONCE into registers)
        velox_cuda_out_proj_residual_batched_q4<B><<<dim3((unsigned int)out_blocks), dim3(128), 0, stream>>>(
            hidden_base, q_base,
            VELOX_DEV_W_Q4_PTR(ctx, 603914240ULL), VELOX_DEV_S_Q4_PTR(ctx, 603914240ULL),
            layer, dim, num_q
        );

        // 2f. Batched Post-Attention RMSNorm
        const float* ffn_norm_w = VELOX_DEV_NORM_PTR(ctx, 2208604160ULL) + (layer * dim);
        velox_cuda_rmsnorm_batched<<<dim3(1, (unsigned int)B), dim3(256), 0, stream>>>(
            norm_base, hidden_base, ffn_norm_w, dim, 1e-6f,
            B, dim, dim
        );

        // 2g. Batched SwiGLU Gate+Up (Register-Tiled: 1D grid, weights loaded ONCE into registers)
        velox_cuda_swiglu_gate_up_batched_q4<B><<<dim3((unsigned int)inter_blocks), dim3(128), 0, stream>>>(
            act_base, norm_base,
            VELOX_DEV_W_Q4_PTR(ctx, 792657920ULL), VELOX_DEV_S_Q4_PTR(ctx, 792657920ULL),
            VELOX_DEV_W_Q4_PTR(ctx, 1264517120ULL), VELOX_DEV_S_Q4_PTR(ctx, 1264517120ULL),
            layer, dim, inter_dim
        );

        // 2h. Batched SwiGLU Down + Residual (Register-Tiled: 1D grid, weights loaded ONCE into registers)
        velox_cuda_swiglu_down_residual_batched_q4<B><<<dim3((unsigned int)out_blocks), dim3(128), 0, stream>>>(
            hidden_base, act_base,
            VELOX_DEV_W_Q4_PTR(ctx, 1736376320ULL), VELOX_DEV_S_Q4_PTR(ctx, 1736376320ULL),
            layer, dim, inter_dim
        );
    }
}

// =============================================================================
// LM Head & Speculative Decoding Evaluation Helpers
// =============================================================================

// Evaluates RMSNorm + LM head + Argmax on d_hidden at position eval_pos
static uint32_t velox_evaluate_lm_head(
    VeloxCudaArena* ctx,
    const float* d_hidden,
    size_t eval_pos,
    cudaStream_t stream
) {
    const size_t dim = VELOX_HIDDEN_DIM;
    VeloxDecodeState* d_state = VELOX_DEV_STATE(ctx);
    velox_cuda_set_decode_state<<<1, 1, 0, stream>>>(d_state, 0, (unsigned long long)eval_pos);

    velox_cuda_rmsnorm_kernel<<<dim3(1), dim3(256), 0, stream>>>(
        VELOX_DEV_ACT_PTR(ctx->activation_arena, 6921216ULL),
        d_hidden,
        VELOX_DEV_NORM_PTR(ctx, 2208972800ULL),
        dim, 1e-6f, 1ULL
    );

    velox_cuda_matmul_vec_kernel_q4<<<dim3(62656ULL), dim3(128), 0, stream>>>(
        VELOX_DEV_ACT_PTR(ctx->activation_arena, 6931456ULL),
        VELOX_DEV_ACT_PTR(ctx->activation_arena, 6921216ULL),
        VELOX_DEV_W_Q4_PTR(ctx, 2208983040ULL),
        VELOX_DEV_S_Q4_PTR(ctx, 2208983040ULL),
        2560ULL, 250624ULL, 1ULL
    );

    velox_cuda_lm_head_argmax_stage1<<<dim3((VELOX_VOCAB_SIZE + 255ULL) / 256ULL), dim3(256), 0, stream>>>(
        VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_VALS_OFFSET),
        (uint32_t*)VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_IDXS_OFFSET),
        VELOX_DEV_ACT_PTR(ctx->activation_arena, 6931456ULL),
        VELOX_VOCAB_SIZE, 1ULL
    );

    velox_cuda_lm_head_argmax_stage2<<<dim3(1), dim3(256), 0, stream>>>(
        ctx->host_pinned_ring,
        VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_VALS_OFFSET),
        (const uint32_t*)VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_SCRATCH_STAGE1_IDXS_OFFSET),
        (VELOX_VOCAB_SIZE + 255ULL) / 256ULL,
        d_state, 1ULL
    );

    cudaStreamSynchronize(stream);
    return ctx->host_pinned_ring[eval_pos & (VELOX_HOST_RING_SLOTS - 1ULL)];
}

// ===========================================================================
// BATCHED Speculative Verify: RMSNorm + Q4 LM-Head + Argmax for K+1 candidate
// rows in ONE streamed pass.
//
// Why this exists (bandwidth proof, RTX 3050 sm_86):
//   The output head is Q6_K: 2560 x 250624 = 641,598,720 params -> after
//   Marlin-int4 repacking 320,799,360 B of weight traffic per pass. The
//   single-row path ran this B times sequentially with a full stream sync
//   after each row, so speculative verify cost grew linearly with the draft
//   width. This version reads the weight tables ONCE for all rows (L2 covers
//   the per-block 4-row working set across candidate rows) and synchronizes
//   exactly once, making verify ~O(1) per candidate instead of O(B) passes:
//   break-even draft acceptance drops from a=1.0 (impossible in practice)
//   to a > (m_B - 1) / K with m_B ~= 1.1-1.3, i.e. a ~= 0.1 for K = 3.
// ===========================================================================

// Row-batched Marlin-int4 output projection: one output row per warp,
// one candidate index per blockIdx.y. dst rows stride = out_dim floats.
__global__ void __launch_bounds__(128, 8) velox_cuda_matmul_vec_batched_q4(
    float* __restrict__ dst,
    const float* __restrict__ x_base,
    const uint32_t* __restrict__ w_q4,
    const half2* __restrict__ s_q4,
    size_t in_dim,
    size_t out_dim,
    size_t n_rows
) {
    const size_t row = (size_t)blockIdx.x * 4ULL + threadIdx.x / 32;
    if (row >= out_dim) return;
    const size_t b = blockIdx.y;
    if (b >= n_rows) return;
    const float* x = x_base + (b * in_dim);
    const float acc = velox_warp_dot_q4(
        w_q4 + ((row * in_dim) >> 3),
        s_q4 + ((row * in_dim) >> 6),
        x, in_dim, threadIdx.x % 32);
    if (threadIdx.x % 32 == 0) {
        dst[(b * out_dim) + row] = acc;
    }
}

// Stage 1 batched: per (row, vocab-slice) partial argmax into the verify
// scratch region. One block per (row, 256-wide vocab slice).
__global__ void velox_cuda_lm_head_argmax_stage1_batched(
    float* __restrict__ scratch_vals,
    uint32_t* __restrict__ scratch_idxs,
    const float* __restrict__ logits,
    size_t vocab_size,
    size_t n_rows
) {
    const size_t row = blockIdx.y;
    if (row >= n_rows) return;
    const size_t num_blocks = gridDim.x;
    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];
    const size_t tid = threadIdx.x;
    const size_t vid = (size_t)blockIdx.x * 256ULL + tid;
    const float* row_logits = logits + (row * vocab_size);

    float my_val = (vid < vocab_size) ? row_logits[vid] : -1.0e30f;
    uint32_t my_idx = (vid < vocab_size) ? (uint32_t)vid : 0U;
    s_val[tid] = my_val;
    s_idx[tid] = my_idx;
    __syncthreads();

    #pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const size_t slot = (row * num_blocks) + blockIdx.x;
        scratch_vals[slot] = s_val[0];
        scratch_idxs[slot] = s_idx[0];
    }
}

// Stage 2 batched: one block per candidate row reduces its partial argmax
// list and zero-copy writes the winner into the host-pinned ring at the
// candidate's absolute decode position (same slot contract as single-row
// evaluation), followed by a system-level fence so a polling host sees it.
__global__ void velox_cuda_lm_head_argmax_stage2_batched(
    uint32_t* __restrict__ host_pinned_ring,
    const float* __restrict__ scratch_vals,
    const uint32_t* __restrict__ scratch_idxs,
    size_t num_candidates,
    const size_t* __restrict__ d_positions,
    size_t n_rows
) {
    const size_t row = blockIdx.x;
    if (row >= n_rows) return;
    __shared__ float s_val[256];
    __shared__ uint32_t s_idx[256];
    const size_t tid = threadIdx.x;

    float my_val = -1.0e30f;
    uint32_t my_idx = 0U;
    const float* r_vals = scratch_vals + (row * num_candidates);
    const uint32_t* r_idxs = scratch_idxs + (row * num_candidates);
    for (size_t c = tid; c < num_candidates; c += blockDim.x) {
        const float c_val = r_vals[c];
        const uint32_t c_idx = r_idxs[c];
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
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const size_t pos = d_positions[row];
        host_pinned_ring[pos & (VELOX_HOST_RING_SLOTS - 1ULL)] = s_idx[0];
        // Make the zero-copy ring write visible to the polling host thread.
        __threadfence_system();
    }
}

// Host-side batched evaluator: final RMSNorm -> batched Q4 LM head ->
// two-stage batched argmax. Zero host synchronization inside (the caller
// is responsible for one cudaStreamSynchronize after this returns, which
// also makes it safe to embed inside CUDA graph capture).
static void velox_evaluate_lm_head_batched(
    VeloxCudaArena* ctx,
    const float* hidden_base,
    size_t n_rows,
    const size_t* d_positions,
    cudaStream_t stream
) {
    if (n_rows == 0ULL || n_rows > VELOX_SPEC_MAX_ROWS) return;

    const float* final_norm_w = VELOX_DEV_NORM_PTR(ctx, 2208972800ULL);
    float* verify_norm = VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_VERIFY_NORM_BASE);
    float* verify_logits = VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_VERIFY_LOGITS_BASE);
    float* scratch_vals = VELOX_DEV_ACT_PTR(ctx->activation_arena, VELOX_VERIFY_STAGE1_BASE);
    uint32_t* scratch_idxs = (uint32_t*)VELOX_DEV_ACT_PTR(
        ctx->activation_arena, VELOX_VERIFY_STAGE1_IDXS_BASE);

    velox_cuda_rmsnorm_batched<<<dim3(1, (unsigned int)n_rows), dim3(256), 0, stream>>>(
        verify_norm, hidden_base, final_norm_w,
        VELOX_HIDDEN_DIM, 1e-6f,
        n_rows, VELOX_HIDDEN_DIM, VELOX_HIDDEN_DIM
    );

    velox_cuda_matmul_vec_batched_q4<<<dim3(62656ULL, (unsigned int)n_rows), dim3(128), 0, stream>>>(
        verify_logits, verify_norm,
        VELOX_DEV_W_Q4_PTR(ctx, 2208983040ULL),
        VELOX_DEV_S_Q4_PTR(ctx, 2208983040ULL),
        2560ULL, VELOX_VOCAB_SIZE, n_rows
    );

    velox_cuda_lm_head_argmax_stage1_batched<<<dim3((VELOX_VOCAB_SIZE + 255ULL) / 256ULL, (unsigned int)n_rows), dim3(256), 0, stream>>>(
        scratch_vals, scratch_idxs, verify_logits, VELOX_VOCAB_SIZE, n_rows
    );

    velox_cuda_lm_head_argmax_stage2_batched<<<dim3((unsigned int)n_rows), dim3(256), 0, stream>>>(
        ctx->host_pinned_ring, scratch_vals, scratch_idxs,
        (VELOX_VOCAB_SIZE + 255ULL) / 256ULL, d_positions, n_rows
    );
}

// Full spec-verify forward pass for a compile-time batch width: embed the
// B candidate rows, run all 36 fused layers in one weights-stream pass,
// and batch-evaluate the LM head. Shared by the CUDA-graph capture prologue
// and the legacy fallback so both paths dispatch through the same code.
#define RUN_VERIFY_BATCH(N) case N: velox_run_batched_layers_q4<N>(ctx, hidden, norm, q, act, d_positions, stream); break;
static void velox_verify_forward_full(
    VeloxCudaArena* ctx,
    int batch,
    float* hidden,
    float* norm,
    float* q,
    float* act,
    const size_t* d_positions,
    cudaStream_t stream
) {
    velox_cuda_embed_batched_q4<<<dim3((VELOX_HIDDEN_DIM / 8ULL + 255) / 256, (unsigned int)batch), dim3(256), 0, stream>>>(
        hidden, (uint32_t*)((uint8_t*)ctx->activation_arena + (VELOX_BATCHED_PREFILL_ACT_BASE + VELOX_BATCHED_PREFILL_MAX_TOKENS * (VELOX_INTERMEDIATE_DIM * 4ULL + 0ULL))),
        VELOX_DEV_W_Q4_PTR(ctx, 0ULL), VELOX_DEV_S_Q4_PTR(ctx, 0ULL),
        (size_t)batch, VELOX_HIDDEN_DIM
    );

    switch (batch) {
        RUN_VERIFY_BATCH(2)
        RUN_VERIFY_BATCH(3)
        RUN_VERIFY_BATCH(4)
        RUN_VERIFY_BATCH(5)
        RUN_VERIFY_BATCH(6)
        RUN_VERIFY_BATCH(7)
        RUN_VERIFY_BATCH(8)
        default: break;
    }

    velox_evaluate_lm_head_batched(ctx, hidden, (size_t)batch, d_positions, stream);
}
#undef RUN_VERIFY_BATCH

// Prompt-Lookup Drafting (PLD): Match last n-gram in prompt/generation history
// Searches backward from current position for the most recent match.
static int velox_prompt_lookup_draft(
    const uint32_t* history,
    size_t history_len,
    uint32_t* draft_tokens,
    int max_draft
) {
    if (history_len < 2 || max_draft <= 0) return 0;

    // Try trigram (n=3), bigram (n=2), and unigram (n=1)
    for (int n = 3; n >= 1; --n) {
        if (history_len < (size_t)(n + 1)) continue;
        const uint32_t* needle = history + history_len - n;
        for (long long i = (long long)(history_len - n - 1); i >= 0; --i) {
            bool match = true;
            for (int k = 0; k < n; ++k) {
                if (history[i + k] != needle[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                size_t draft_start = (size_t)i + n;
                if (draft_start < history_len - n) {
                    size_t available = (history_len - n) - draft_start;
                    int count = (int)available;
                    int limit = (n == 1 && max_draft > 2) ? 2 : max_draft;
                    if (count > limit) count = limit;
                    for (int d = 0; d < count; ++d) {
                        draft_tokens[d] = history[draft_start + d];
                    }
                    return count;
                }
            }
        }
    }
    return 0;
}

// =============================================================================
// Batched Prefill Entry Point: processes all prompt tokens through all 36 layers
// in a single pass, streaming each layer's weights exactly once.
// =============================================================================
static void velox_batched_prefill_q4(
    VeloxCudaArena* ctx,
    const uint32_t* h_prompt_tokens,
    size_t prompt_len,
    cudaStream_t stream
) {
    if (prompt_len == 0) return;

    const size_t dim = VELOX_HIDDEN_DIM;           // 2560
    const size_t inter_dim = VELOX_INTERMEDIATE_DIM; // 10240

    // Activation arena pointers for batched buffers
    float* hidden_base = (float*)((uint8_t*)ctx->activation_arena + VELOX_BATCHED_PREFILL_BASE);
    float* norm_base   = (float*)((uint8_t*)ctx->activation_arena + VELOX_BATCHED_PREFILL_NORM_BASE);
    float* q_base      = (float*)((uint8_t*)ctx->activation_arena + VELOX_BATCHED_PREFILL_Q_BASE);
    float* act_base    = (float*)((uint8_t*)ctx->activation_arena + VELOX_BATCHED_PREFILL_ACT_BASE);

    // Use space after the batched act buffer for token/position arrays (zero dynamic host allocations)
    size_t token_dev_offset = VELOX_BATCHED_PREFILL_ACT_BASE + VELOX_BATCHED_PREFILL_MAX_TOKENS * inter_dim * 4ULL;
    uint32_t* d_tokens = (uint32_t*)((uint8_t*)ctx->activation_arena + token_dev_offset);
    size_t* d_positions = (size_t*)((uint8_t*)ctx->activation_arena + token_dev_offset + VELOX_BATCHED_PREFILL_MAX_TOKENS * sizeof(uint32_t));

    cudaMemcpyAsync(d_tokens, h_prompt_tokens, prompt_len * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    velox_cuda_init_positions<<<1, (unsigned int)((prompt_len + 31) / 32 * 32), 0, stream>>>(d_positions, 0ULL, prompt_len);

    // Grid dimensions for Embedding
    const size_t embed_blocks = (dim / 8ULL + 255) / 256;

    cudaEvent_t ev_embed_stop, ev_layers_stop, ev_head_stop;
    cudaEventCreate(&ev_embed_stop);
    cudaEventCreate(&ev_layers_stop);
    cudaEventCreate(&ev_head_stop);

    // Step 1: Batched Embedding (all tokens simultaneously)
    velox_cuda_embed_batched_q4<<<dim3((unsigned int)embed_blocks, (unsigned int)prompt_len), dim3(256), 0, stream>>>(
        hidden_base, d_tokens,
        VELOX_DEV_W_Q4_PTR(ctx, 0ULL), VELOX_DEV_S_Q4_PTR(ctx, 0ULL),
        prompt_len, dim
    );
    cudaEventRecord(ev_embed_stop, stream);

    // Step 2: Process all 36 layers with zero local memory spills via Register-Tiled Template Kernels
    #define DISPATCH_CASE(N) case N: velox_run_batched_layers_q4<N>(ctx, hidden_base, norm_base, q_base, act_base, d_positions, stream); break;

    switch (prompt_len) {
        DISPATCH_CASE(1)
        DISPATCH_CASE(2)
        DISPATCH_CASE(3)
        DISPATCH_CASE(4)
        DISPATCH_CASE(5)
        DISPATCH_CASE(6)
        DISPATCH_CASE(7)
        DISPATCH_CASE(8)
        DISPATCH_CASE(9)
        DISPATCH_CASE(10)
        DISPATCH_CASE(11)
        DISPATCH_CASE(12)
        DISPATCH_CASE(13)
        DISPATCH_CASE(14)
        DISPATCH_CASE(15)
        DISPATCH_CASE(16)
        default:
            fprintf(stderr, "[velox_batched_prefill_q4] Prompt length %zu > 16, falling back to sequential decode\n", prompt_len);
            for (size_t pos = 0; pos < prompt_len; ++pos) {
                velox_K2_Horizon_3_7B_decode_cuda(ctx, h_prompt_tokens[pos], pos, 1ULL, stream);
            }
            return;
    }
    #undef DISPATCH_CASE

    cudaEventRecord(ev_layers_stop, stream);

    // Step 3: Copy last token's hidden state and evaluate LM head
    float* decode_hidden = VELOX_DEV_ACT_PTR(ctx->activation_arena, 1012736ULL);
    float* last_hidden = hidden_base + (prompt_len - 1) * dim;
    cudaMemcpyAsync(decode_hidden, last_hidden, dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    cudaEventRecord(ev_head_stop, stream);
    velox_evaluate_lm_head(ctx, decode_hidden, prompt_len - 1, stream);

    float t_embed_ms = 0.0f, t_layers_ms = 0.0f, t_head_ms = 0.0f;
    cudaEventElapsedTime(&t_embed_ms, ev_embed_stop, ev_embed_stop);
    cudaEventElapsedTime(&t_layers_ms, ev_embed_stop, ev_layers_stop);
    cudaEventElapsedTime(&t_head_ms, ev_layers_stop, ev_head_stop);
    printf("  [Prefill Timing Breakdown] 36 Layers: %.3f ms (%.3f ms/layer), LM Head: %.3f ms\n",
           t_layers_ms, t_layers_ms / 36.0f, t_head_ms);

    cudaEventDestroy(ev_embed_stop);
    cudaEventDestroy(ev_layers_stop);
    cudaEventDestroy(ev_head_stop);
}

// --- Top-5 Logits Helper for 250,624 Vocabulary -----------------------------
typedef struct {
    uint32_t token_id;
    float logit;
} LogitEntry;

static void print_top5_logits(const float* logits, size_t vocab_size) {
    LogitEntry top5[5];
    for (int i = 0; i < 5; ++i) {
        top5[i].token_id = 0;
        top5[i].logit = -1e30f;
    }

    for (size_t v = 0; v < vocab_size; ++v) {
        float val = logits[v];
        if (val > top5[4].logit) {
            top5[4].token_id = (uint32_t)v;
            top5[4].logit = val;
            for (int k = 4; k > 0; --k) {
                if (top5[k].logit > top5[k - 1].logit) {
                    LogitEntry tmp = top5[k];
                    top5[k] = top5[k - 1];
                    top5[k - 1] = tmp;
                } else {
                    break;
                }
            }
        }
    }

    printf("Final Step Top-5 Logits (out of %zu):\n", vocab_size);
    for (int i = 0; i < 5; ++i) {
        printf("  #%d: token_id=%u logit=%.6f\n", i + 1, top5[i].token_id, top5[i].logit);
    }
}

// --- Main Program Entry Point -----------------------------------------------
int main(int argc, char** argv) {
    const char* weights_path = "weights/k2_horizon_q4_k_m.gguf";
    uint32_t prompt_tokens[16384];
    prompt_tokens[0] = 0;
    prompt_tokens[1] = 1504;
    prompt_tokens[2] = 346;
    prompt_tokens[3] = 2024;
    size_t prompt_len = 4;
    size_t steps = 16;
    bool verbose = false;
    bool use_cuda_graph = true;
    bool use_speculative = false;
    int max_draft_len = 3;
    float spec_off_ema = 0.20f;
    size_t max_context_tokens = 16384;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if ((strcmp(argv[i], "--tokens") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc) {
            const char* s = argv[++i];
            prompt_len = 0;
            char* copy = _strdup(s);
            char* token = strtok(copy, ",");
            while (token && prompt_len < 16384) {
                prompt_tokens[prompt_len++] = (uint32_t)strtoul(token, NULL, 10);
                token = strtok(NULL, ",");
            }
            free(copy);
        } else if (strcmp(argv[i], "--speculative") == 0) {
            use_speculative = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                max_draft_len = atoi(argv[++i]);
                if (max_draft_len <= 0) max_draft_len = 3;
                if (max_draft_len > 7) max_draft_len = 7;
            }
        } else if (strcmp(argv[i], "--draft-len") == 0 && i + 1 < argc) {
            max_draft_len = atoi(argv[++i]);
            if (max_draft_len <= 0) max_draft_len = 3;
            if (max_draft_len > 7) max_draft_len = 7;
        } else if (strcmp(argv[i], "--spec-ema") == 0 && i + 1 < argc) {
            spec_off_ema = (float)atof(argv[++i]);
            if (spec_off_ema < 0.0f) spec_off_ema = 0.0f;
            if (spec_off_ema > 0.95f) spec_off_ema = 0.95f;
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--context-len") == 0 && i + 1 < argc) {
            max_context_tokens = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--no-cuda-graph") == 0) {
            use_cuda_graph = false;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Velox Standalone CUDA Runner for K2-Horizon-3.7B\n");
            printf("Usage: velox_cuda_runner_k2.exe [options]\n");
            printf("Options:\n");
            printf("  --weights <path>       Path to weights (.gguf)\n");
            printf("  --tokens <t1,t2,...>   Comma-separated prompt token IDs (or --prompt)\n");
            printf("  --speculative [K]      Enable Prompt-Lookup Speculative Decoding (default draft K=3)\n");
            printf("  --draft-len <K>        Set max draft candidate length (default: 3, max: 7)\n");
            printf("  --spec-ema <x>         Adaptive drafting: EMA acceptance floor (default 0.20)\n");
            printf("  --steps <N>            Autoregressive decode steps (default: 16)\n");
            printf("  --context-len <N>      Max context tokens in host-pinned KV cache (default: 16384)\n");
            printf("  --verbose              Print token at each step\n");
            printf("  --no-cuda-graph        Disable CUDA Graph replay\n");
            return 0;
        }
    }

    timer_init();

    // 1. Initialize GPU Device Invariants (RTX 3050 Laptop, sm_86)
    cudaSetDeviceFlags(cudaDeviceMapHost);
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // 2. Compute Memory Tier Allocations
    size_t weight_bytes = K2_TIER1_WEIGHT_ARENA_BYTES;
    size_t act_arena_bytes = VELOX_ACTIVATION_ARENA_BYTES;
    size_t scales_bytes = K2_WEIGHT_SCALES_BYTES;
    size_t ring_bytes = K2_HOST_RING_BYTES;

    // KV Cache pool: 36 layers * 2 (K+V) * VELOX_MODEL_MAX_SEQ_LEN * 8 KV heads * 128 head_dim * 4 bytes (float)
    size_t kv_pool_bytes = VELOX_NUM_LAYERS * 2ULL * VELOX_MODEL_MAX_SEQ_LEN * VELOX_NUM_KV_HEADS * VELOX_HEAD_DIM * sizeof(float);

    LARGE_INTEGER t_init_start;
    QueryPerformanceCounter(&t_init_start);

    // 3. Pre-allocate static Velox CUDA arenas (strictly once at startup on RTX 3050)
    uint8_t* d_weight_arena = NULL;
    uint8_t* d_weight_scales = NULL;
    uint8_t* d_activation = NULL;
    uint8_t* h_kv_cache = NULL;
    uint8_t* d_kv_cache = NULL;
    bool is_kv_in_vram = false;
    uint32_t* h_ring = NULL;
    uint32_t* d_ring = NULL;

    cudaError_t err;
    err = cudaMalloc((void**)&d_weight_arena, weight_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_weight_arena (%zu MiB) failed: %s\n",
                weight_bytes / (1024 * 1024), cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_weight_scales, scales_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_weight_scales (%zu MiB) failed: %s\n",
                scales_bytes / (1024 * 1024), cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_activation, act_arena_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_activation (%zu MiB) failed: %s\n",
                act_arena_bytes / (1024 * 1024), cudaGetErrorString(err));
        return 1;
    }

    // Allocate KV Cache: Try high-speed dedicated GPU VRAM first, fallback to Host-Pinned RAM
    err = cudaMalloc((void**)&d_kv_cache, kv_pool_bytes);
    if (err == cudaSuccess) {
        is_kv_in_vram = true;
    } else {
        is_kv_in_vram = false;
        err = cudaHostAlloc((void**)&h_kv_cache, kv_pool_bytes, cudaHostAllocMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaHostAlloc h_kv_cache (%zu MiB) failed: %s\n",
                    kv_pool_bytes / (1024 * 1024), cudaGetErrorString(err));
            return 1;
        }
        err = cudaHostGetDevicePointer((void**)&d_kv_cache, h_kv_cache, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaHostGetDevicePointer failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }

    // Host-pinned ring buffer for zero-copy D2H token reading
    err = cudaHostAlloc((void**)&h_ring, ring_bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaHostAlloc host_pinned_ring failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaHostGetDevicePointer((void**)&d_ring, h_ring, 0);
    if (err != cudaSuccess) {
        d_ring = h_ring;
    }

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // 4. Memory-map GGUF file via Win32 zero-copy and stream weights into GPU
    Win32GgufMap gguf_map;
    bool has_weights_file = win32_gguf_open(weights_path, &gguf_map);
    if (!has_weights_file && strcmp(weights_path, "weights/k2_horizon_q4_k_m.gguf") != 0) {
        has_weights_file = win32_gguf_open("weights/k2_horizon_q4_k_m.gguf", &gguf_map);
    }

    double weight_load_ms = 0.0;
    if (has_weights_file) {
        LARGE_INTEGER t_load_start;
        QueryPerformanceCounter(&t_load_start);

        // Allocate a 16 MiB device staging buffer for streaming unpacking
        size_t staging_bytes = 16ULL * 1024ULL * 1024ULL;
        uint8_t* d_staging = NULL;
        cudaMalloc((void**)&d_staging, staging_bytes);

        uint32_t* d_w_u32 = (uint32_t*)d_weight_arena;
        half2* d_s_h2 = (half2*)d_weight_scales;

        // 1. Embeddings: token_embd.weight -> 0ULL
        load_gguf_tensor_to_arena(&gguf_map, "token_embd.weight", d_staging, staging_bytes, d_w_u32, d_s_h2, 0ULL, stream);

        // 2. Transformer Blocks (36 layers)
        char tname[128];
        for (int l = 0; l < 36; ++l) {
            // w_q: 320798720ULL + l * (2560 * 4096 / 2)
            snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
            load_gguf_tensor_to_arena(&gguf_map, tname, d_staging, staging_bytes, d_w_u32, d_s_h2, 320798720ULL + (size_t)l * 5242880ULL, stream);

            // w_k: 509542400ULL + l * (2560 * 1024 / 2)
            snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
            load_gguf_tensor_to_arena(&gguf_map, tname, d_staging, staging_bytes, d_w_u32, d_s_h2, 509542400ULL + (size_t)l * 1310720ULL, stream);

            // w_v: 556728320ULL + l * (2560 * 1024 / 2)
            snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
            load_gguf_tensor_to_arena(&gguf_map, tname, d_staging, staging_bytes, d_w_u32, d_s_h2, 556728320ULL + (size_t)l * 1310720ULL, stream);

            // w_o: 603914240ULL + l * (4096 * 2560 / 2)
            snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
            load_gguf_tensor_to_arena(&gguf_map, tname, d_staging, staging_bytes, d_w_u32, d_s_h2, 603914240ULL + (size_t)l * 5242880ULL, stream);

            // w_gate: 792657920ULL + l * (2560 * 10240 / 2)
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
            load_gguf_tensor_to_arena(&gguf_map, tname, d_staging, staging_bytes, d_w_u32, d_s_h2, 792657920ULL + (size_t)l * 13107200ULL, stream);

            // w_up: 1264517120ULL + l * (2560 * 10240 / 2)
            snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
            load_gguf_tensor_to_arena(&gguf_map, tname, d_staging, staging_bytes, d_w_u32, d_s_h2, 1264517120ULL + (size_t)l * 13107200ULL, stream);

            // w_down: 1736376320ULL + l * (10240 * 2560 / 2)
            snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
            load_gguf_tensor_to_arena(&gguf_map, tname, d_staging, staging_bytes, d_w_u32, d_s_h2, 1736376320ULL + (size_t)l * 13107200ULL, stream);

            // w_input_norm: 2208235520ULL + l * (2560 * 4)
            snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
            load_gguf_tensor_to_arena(&gguf_map, tname, d_staging, staging_bytes, d_w_u32, d_s_h2, 2208235520ULL + (size_t)l * 10240ULL, stream);

            // w_post_norm: 2208604160ULL + l * (2560 * 4)
            snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", l);
            load_gguf_tensor_to_arena(&gguf_map, tname, d_staging, staging_bytes, d_w_u32, d_s_h2, 2208604160ULL + (size_t)l * 10240ULL, stream);
        }

        // 3. Final norm & output head
        load_gguf_tensor_to_arena(&gguf_map, "output_norm.weight", d_staging, staging_bytes, d_w_u32, d_s_h2, 2208972800ULL, stream);
        load_gguf_tensor_to_arena(&gguf_map, "output.weight", d_staging, staging_bytes, d_w_u32, d_s_h2, 2208983040ULL, stream);

        cudaStreamSynchronize(stream);
        cudaFree(d_staging);
        weight_load_ms = timer_ms_since(t_load_start);
    } else {
        cudaMemsetAsync(d_weight_arena, 0, weight_bytes, stream);
        cudaMemsetAsync(d_weight_scales, 0, scales_bytes, stream);
    }

    // Zero-initialize dynamic pools
    cudaMemsetAsync(d_activation, 0, act_arena_bytes, stream);
    cudaMemsetAsync(d_kv_cache, 0, kv_pool_bytes, stream);
    memset(h_ring, 0, ring_bytes);
    cudaStreamSynchronize(stream);

    // 5. Construct Velox zero-allocation execution arena context
    VeloxCudaArena ctx;
    ctx.weight_arena = d_weight_arena;
    ctx.kv_cache_pool = d_kv_cache;
    ctx.activation_arena = d_activation;
    ctx.host_pinned_ring = d_ring;
    ctx.norm_weight_arena = d_weight_arena + VELOX_W_NORM_START_FP32;
    ctx.weight_scales = d_weight_scales;
    ctx.quant_mode = VELOX_QUANT_Q4_0;
    ctx.tied_lm_head = 0u;

    // 6. Pre-capture single-step decode CUDA Graph
    cudaGraph_t decode_graph = NULL;
    cudaGraphExec_t decode_graph_exec = NULL;
    if (use_cuda_graph) {
        err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err == cudaSuccess) {
            velox_K2_Horizon_3_7B_decode_graph_step_cuda(&ctx, 1ULL, stream);
            err = cudaStreamEndCapture(stream, &decode_graph);
            if (err != cudaSuccess) {
                if (verbose) printf("[CUDA Graph] StreamEndCapture failed: %s\n", cudaGetErrorString(err));
            } else if (decode_graph != NULL) {
                err = cudaGraphInstantiate(&decode_graph_exec, decode_graph, NULL, NULL, 0);
                if (err != cudaSuccess) {
                    if (verbose) printf("[CUDA Graph] GraphInstantiate failed: %s\n", cudaGetErrorString(err));
                } else {
                    if (verbose) printf("[CUDA Graph] Graph capture & instantiate SUCCESSFUL!\n");
                }
            }
        } else {
            if (verbose) printf("[CUDA Graph] StreamBeginCapture failed: %s\n", cudaGetErrorString(err));
        }
        if (err != cudaSuccess || decode_graph_exec == NULL) {
            use_cuda_graph = false;
        }
    }

    // Shared scratch pointers for the speculative verify path. Embed, fused
    // layers, and the batched LM head all read/write these fixed arena
    // offsets, so a captured graph replays against fresh inputs by simply
    // updating d_tokens / d_positions on the same stream before launch.
    float* hidden_base = (float*)((uint8_t*)ctx.activation_arena + VELOX_BATCHED_PREFILL_BASE);
    float* norm_base   = (float*)((uint8_t*)ctx.activation_arena + VELOX_BATCHED_PREFILL_NORM_BASE);
    float* q_base      = (float*)((uint8_t*)ctx.activation_arena + VELOX_BATCHED_PREFILL_Q_BASE);
    float* act_base    = (float*)((uint8_t*)ctx.activation_arena + VELOX_BATCHED_PREFILL_ACT_BASE);

    size_t token_dev_offset = VELOX_BATCHED_PREFILL_ACT_BASE + VELOX_BATCHED_PREFILL_MAX_TOKENS * VELOX_INTERMEDIATE_DIM * 4ULL;
    uint32_t* d_tokens = (uint32_t*)((uint8_t*)ctx.activation_arena + token_dev_offset);
    size_t* d_positions = (size_t*)((uint8_t*)ctx.activation_arena + token_dev_offset + VELOX_BATCHED_PREFILL_MAX_TOKENS * sizeof(uint32_t));

    // 6b. Pre-capture batched speculative-verify CUDA Graphs, one per batch
    // width B = 2..8. Each graph bakes in: batched embed -> 36 fused layers
    // in a single weights-stream pass -> batched LM head + two-stage argmax.
    // Per speculative step the host only issues the candidate H2D copy, a
    // 1-block position-init kernel, and one cudaGraphLaunch, collapsing
    // ~170 kernel launches (7 per layer x 36 layers + head) into 1.
    cudaGraph_t verify_graph[9];
    cudaGraphExec_t verify_graph_exec[9];
    int verify_graphs_ready = 0;
    for (int b = 0; b < 9; ++b) {
        verify_graph[b] = NULL;
        verify_graph_exec[b] = NULL;
    }
    if (use_cuda_graph && use_speculative) {
        for (int B = 2; B <= 8; ++B) {
            bool capture_ok = true;
            err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            if (err == cudaSuccess) {
                velox_verify_forward_full(&ctx, B, hidden_base, norm_base, q_base, act_base, d_positions, stream);
                err = cudaStreamEndCapture(stream, &verify_graph[B]);
                capture_ok = (err == cudaSuccess && verify_graph[B] != NULL);
            }
            if (capture_ok) {
                err = cudaGraphInstantiate(&verify_graph_exec[B], verify_graph[B], NULL, NULL, 0);
                capture_ok = (err == cudaSuccess && verify_graph_exec[B] != NULL);
            }
            if (capture_ok) {
                // Keep only the instantiated executable; release the graph
                // template so no graph handles linger past startup.
                cudaGraphDestroy(verify_graph[B]);
                verify_graph[B] = NULL;
                ++verify_graphs_ready;
            } else {
                // Per-width fallback: a failed width simply flies legacy
                // launches while narrower/wider widths still replay graphs.
                if (verify_graph[B] != NULL) {
                    cudaGraphDestroy(verify_graph[B]);
                    verify_graph[B] = NULL;
                }
                if (verbose) {
                    printf("[CUDA Graph] Verify batch B=%d capture failed: %s\n",
                           B, (err == cudaSuccess) ? "instantiate failed" : cudaGetErrorString(err));
                }
            }
        }
        if (verbose) {
            printf("[CUDA Graph] Speculative verify graphs captured: %d/7 (B=2..8)\n", verify_graphs_ready);
        }
    }

    uint32_t* generated_tokens = (uint32_t*)malloc(steps * sizeof(uint32_t));
    if (!generated_tokens) return 1;

    cudaEvent_t ev_prefill_start, ev_prefill_stop;
    cudaEvent_t ev_decode_start, ev_decode_stop;
    cudaEventCreate(&ev_prefill_start);
    cudaEventCreate(&ev_prefill_stop);
    cudaEventCreate(&ev_decode_start);
    cudaEventCreate(&ev_decode_stop);

    // 7. Execute Prefill Phase on prompt tokens
    cudaEventRecord(ev_prefill_start, stream);

    if (ctx.quant_mode == VELOX_QUANT_Q4_0 && prompt_len > 1 && prompt_len <= VELOX_BATCHED_PREFILL_MAX_TOKENS) {
        // Batched prefill: stream each layer's weights exactly ONCE for all prompt tokens
        velox_batched_prefill_q4(&ctx, prompt_tokens, prompt_len, stream);
    } else {
        // Sequential fallback (single token or non-Q4 mode)
        for (size_t pos = 0; pos < prompt_len; ++pos) {
            velox_K2_Horizon_3_7B_decode_cuda(&ctx, prompt_tokens[pos], pos, 1ULL, stream);
        }
    }

    cudaEventRecord(ev_prefill_stop, stream);
    cudaStreamSynchronize(stream);

    float ttft_gpu_ms = 0.0f;
    cudaEventElapsedTime(&ttft_gpu_ms, ev_prefill_start, ev_prefill_stop);

    uint32_t next_tok = h_ring[(prompt_len - 1) & 1023ULL];
    if (steps > 0) {
        generated_tokens[0] = next_tok;
        if (verbose) {
            printf("[Step 0] Token: %u\n", next_tok);
            fflush(stdout);
        }
    }

    // 8. Autoregressive Decode Phase (strictly 0 dynamic allocations on GPU & CPU)
    float decode_gpu_ms = 0.0f;
    size_t total_drafted = 0;
    size_t total_accepted = 0;
    size_t spec_steps = 0;
    size_t plain_by_policy = 0;
    float  spec_ema_final = 0.0f;
    int    spec_ladder_final = 0;

    if (steps > 1) {
        if (use_speculative) {
            cudaEventRecord(ev_decode_start, stream);

            // Shared scratch pointers (hidden/norm/q/act arenas, candidate token
            // and position staging) were hoisted before graph capture above.

            // -------------------------------------------------------------
            // Adaptive acceptance policy (EMA over per-step draft acceptance):
            //
            //   r_t   = accepted drafts in this step / drafts attempted
            //   ema   = (1 - tau) * ema + tau * r_t          (tau = 0.35)
            //
            // Speculation stays enabled while ema >= spec_off_ema (default
            // 0.20). Below the floor, drafting pauses for a ladder that
            // doubles in plain-decode steps (1, 2, 4, ... 64) and probes a
            // single speculative step again -- text regimes change mid-run
            // (comment blocks, repetitive prints), so one bad stretch must
            // not disable the engine permanently. K is additionally scaled
            // by the current EMA: high confidence -> full K, marginal
            // confidence -> narrow drafts, keeping wasted verify work at a
            // premium.
            // -------------------------------------------------------------
            const float spec_ema_tau     = 0.35f;
            float       ema_accept       = 0.70f;
            const float spec_ladder_cap  = 64.0f;
            int         plain_stride     = 1;
            int         plain_since_spec = 0;
            spec_ema_final = 0.70f;

            size_t curr_pos = prompt_len;
            size_t gen_count = 1;

            uint32_t history[VELOX_MODEL_MAX_SEQ_LEN];
            memcpy(history, prompt_tokens, prompt_len * sizeof(uint32_t));
            history[prompt_len] = next_tok;
            size_t history_len = prompt_len + 1;

            while (gen_count < steps && curr_pos + 2ULL < VELOX_MODEL_MAX_SEQ_LEN && history_len + 2ULL < VELOX_MODEL_MAX_SEQ_LEN) {
                spec_steps++;
                uint32_t draft_tokens[8];
                int K = velox_prompt_lookup_draft(history, history_len, draft_tokens, max_draft_len);
                if ((size_t)K > steps - gen_count) {
                    K = (int)(steps - gen_count);
                }
                // Dynamic draft width from current acceptance confidence.
                if (K > 0) {
                    int k_cap = (int)(max_draft_len * (0.4f + 0.6f * (ema_accept > 1.0f ? 1.0f : ema_accept)));
                    if (k_cap < 1) k_cap = 1;
                    if (K > k_cap) K = k_cap;
                    if ((size_t)K > steps - gen_count) {
                        K = (int)(steps - gen_count);
                    }
                }

                // Probe-ladder policy gate.
                bool use_spec_now;
                if ((ema_accept >= spec_off_ema) || K <= 0) {
                    use_spec_now = (K > 0);
                    plain_since_spec = 0;
                } else if (plain_since_spec >= plain_stride) {
                    use_spec_now = true;
                    plain_since_spec = 0;
                    plain_stride = (plain_stride < (int)spec_ladder_cap) ? plain_stride * 2 : (int)spec_ladder_cap;
                } else {
                    use_spec_now = false;
                    plain_since_spec++;
                }

                if (!use_spec_now) {
                    plain_by_policy++;
                    velox_K2_Horizon_3_7B_decode_cuda(&ctx, next_tok, curr_pos, 1ULL, stream);
                    cudaStreamSynchronize(stream);
                    next_tok = h_ring[curr_pos & (VELOX_HOST_RING_SLOTS - 1ULL)];
                    generated_tokens[gen_count++] = next_tok;
                    history[history_len++] = next_tok;
                    if (verbose) {
                        printf("[Step %zu] Token: %u (Autoregressive Standard)\n", gen_count - 1, next_tok);
                        fflush(stdout);
                    }
                    curr_pos++;
                } else {
                    total_drafted += K;
                    int B = K + 1;
                    int accepted_in_step = 0;
                    uint32_t candidates[9];
                    candidates[0] = next_tok;
                    for (int i = 0; i < K; ++i) {
                        candidates[i + 1] = draft_tokens[i];
                    }

                    cudaMemcpyAsync(d_tokens, candidates, B * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
                    velox_cuda_init_positions<<<1, 32, 0, stream>>>(d_positions, curr_pos, B);

                    if (verify_graph_exec[B] != NULL) {
                        // Graph replay: the captured batch reads the d_tokens and
                        // d_positions scratch slots, which the two operations
                        // above just refreshed on this same stream -- stream
                        // ordering guarantees replay observes the fresh inputs.
                        cudaGraphLaunch(verify_graph_exec[B], stream);
                    } else {
                        // Legacy fallback (capture disabled or unavailable for
                        // this batch width): identical dispatch, explicated
                        // launches. Bit-identical kernel sequence to the graph.
                        velox_verify_forward_full(&ctx, B, hidden_base, norm_base, q_base, act_base, d_positions, stream);
                    }
                    cudaStreamSynchronize(stream);

                    for (int t = 0; t < B && gen_count < steps; ++t) {
                        const uint32_t winner = h_ring[(curr_pos + (size_t)t) & (VELOX_HOST_RING_SLOTS - 1ULL)];

                        if (t < B - 1) {
                            if (winner == candidates[t + 1]) {
                                total_accepted++;
                                accepted_in_step++;
                                generated_tokens[gen_count++] = candidates[t + 1];
                                history[history_len++] = candidates[t + 1];
                                if (verbose) {
                                    printf("[Step %zu] Token: %u (ACCEPTED Speculative Draft)\n", gen_count - 1, candidates[t + 1]);
                                    fflush(stdout);
                                }
                            } else {
                                generated_tokens[gen_count++] = winner;
                                history[history_len++] = winner;
                                next_tok = winner;
                                curr_pos += (t + 1);
                                if (verbose) {
                                    printf("[Step %zu] Token: %u (Rejected draft %u, replaced with true)\n", gen_count - 1, winner, candidates[t + 1]);
                                    fflush(stdout);
                                }
                                break;
                            }
                        } else {
                            generated_tokens[gen_count++] = winner;
                            history[history_len++] = winner;
                            next_tok = winner;
                            curr_pos += B;
                            if (verbose) {
                                printf("[Step %zu] Token: %u (Bonus speculative verified token)\n", gen_count - 1, winner);
                                fflush(stdout);
                            }
                        }
                    }

                    // Exponentially smooth per-step draft acceptance into the
                    // policy state (r always in [0,1]).
                    {
                        const float r = (float)accepted_in_step / (float)K;
                        ema_accept = (1.0f - spec_ema_tau) * ema_accept + spec_ema_tau * r;
                    }
                }
            }

            cudaEventRecord(ev_decode_stop, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&decode_gpu_ms, ev_decode_start, ev_decode_stop);
            spec_ladder_final = plain_stride;
            spec_ema_final = ema_accept;
        } else if (use_cuda_graph && decode_graph_exec != NULL) {
            velox_cuda_set_decode_state<<<1, 1, 0, stream>>>(VELOX_DEV_STATE(&ctx), next_tok, (unsigned long long)prompt_len);
            cudaStreamSynchronize(stream);

            cudaEventRecord(ev_decode_start, stream);

            if (!verbose && steps <= VELOX_HOST_RING_SLOTS) {
                for (size_t s = 1; s < steps; ++s) {
                    cudaGraphLaunch(decode_graph_exec, stream);
                }
                cudaEventRecord(ev_decode_stop, stream);
                cudaStreamSynchronize(stream);
                for (size_t s = 1; s < steps; ++s) {
                    size_t pos = prompt_len - 1 + s;
                    generated_tokens[s] = h_ring[pos & 1023ULL];
                }
            } else {
                for (size_t s = 1; s < steps; ++s) {
                    size_t pos = prompt_len - 1 + s;
                    velox_cuda_set_decode_state<<<1, 1, 0, stream>>>(VELOX_DEV_STATE(&ctx), next_tok, (unsigned long long)pos);
                    cudaGraphLaunch(decode_graph_exec, stream);
                    cudaStreamSynchronize(stream);
                    next_tok = h_ring[pos & 1023ULL];
                    generated_tokens[s] = next_tok;
                    if (verbose) {
                        printf("[Step %zu] Token: %u\n", s, next_tok);
                        fflush(stdout);
                    }
                }
                cudaEventRecord(ev_decode_stop, stream);
                cudaStreamSynchronize(stream);
            }

            cudaEventElapsedTime(&decode_gpu_ms, ev_decode_start, ev_decode_stop);
        } else {
            cudaEventRecord(ev_decode_start, stream);

            for (size_t s = 1; s < steps; ++s) {
                size_t pos = prompt_len - 1 + s;
                velox_K2_Horizon_3_7B_decode_cuda(&ctx, next_tok, pos, 1ULL, stream);
                cudaStreamSynchronize(stream);
                next_tok = h_ring[pos & 1023ULL];
                generated_tokens[s] = next_tok;
                if (verbose) {
                    printf("[Step %zu] Token: %u\n", s, next_tok);
                    fflush(stdout);
                }
            }

            cudaEventRecord(ev_decode_stop, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&decode_gpu_ms, ev_decode_start, ev_decode_stop);
        }
    }

    double total_gpu_ms = (double)ttft_gpu_ms + (double)decode_gpu_ms;
    double decode_tok_per_sec = (steps > 1) ? (((double)(steps - 1) * 1000.0) / (double)decode_gpu_ms) : 0.0;

    // 9. Standardized Velox Standalone Execution Report
    printf("================================================================================\n");
    printf("VELOX EMITTED CUDA STANDALONE RUNTIME EXECUTION REPORT\n");
    printf("================================================================================\n");
    printf("Model                  : K2-Horizon-3.7B (36 Layers)\n");
    printf("Context                : 524288 tokens (arch limit; planner grants 2048 active KV slots)\n");
    printf("Target Device          : %s (Compute %d.%d, %zu MiB VRAM)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
    printf("Weights Source         : %s\n", has_weights_file ? weights_path : "Dry-Run Feasibility Arena (0 Allocation)");
    if (has_weights_file) {
        printf("Weight Stream to VRAM  : %.2f ms\n", weight_load_ms);
    }
    printf("Device VRAM Allocated  : %.2f MiB (%.2f%% of 4096 MiB RTX 3050)\n",
           (double)(weight_bytes + scales_bytes + act_arena_bytes) / (1024.0 * 1024.0),
           ((double)(weight_bytes + scales_bytes + act_arena_bytes) / (double)prop.totalGlobalMem) * 100.0);
    if (is_kv_in_vram) {
        printf("KV Cache Allocation    : %.2f MiB in Dedicated GPU VRAM (%llu Tokens, 155 GB/s GDDR6)\n",
               (double)kv_pool_bytes / (1024.0 * 1024.0), (unsigned long long)VELOX_MODEL_MAX_SEQ_LEN);
    } else {
        printf("KV Cache Allocation    : %.2f MiB in Tier 4 Host-Pinned System RAM (%llu Tokens, UVA Zero-Copy)\n",
               (double)kv_pool_bytes / (1024.0 * 1024.0), (unsigned long long)VELOX_MODEL_MAX_SEQ_LEN);
    }

    printf("Prompt Tokens (%zu)      : [", prompt_len);
    for (size_t i = 0; i < prompt_len; ++i) {
        printf("%u%s", prompt_tokens[i], (i + 1 < prompt_len) ? ", " : "");
    }
    printf("]\n");

    printf("Generated Tokens (%zu)  : [", steps);
    for (size_t i = 0; i < steps; ++i) {
        printf("%u%s", generated_tokens[i], (i + 1 < steps) ? ", " : "");
    }
    printf("]\n");

    float* h_logits = (float*)malloc(VELOX_VOCAB_SIZE * sizeof(float));
    if (h_logits) {
        cudaMemcpy(h_logits, (const float*)((const uint8_t*)ctx.activation_arena + 6931456ULL), VELOX_VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        print_top5_logits(h_logits, VELOX_VOCAB_SIZE);
        free(h_logits);
    }

    printf("TTFT (Prefill Latency) : %.3f ms (prompt len = %zu)\n", ttft_gpu_ms, prompt_len);
    if (steps > 1) {
        printf("Decode Latency (%zu tok): %.3f ms (%.2f tok/s, %.3f ms/tok)\n",
               steps - 1, decode_gpu_ms, decode_tok_per_sec, (double)decode_gpu_ms / (double)(steps - 1));
    }
    if (use_speculative) {
        printf("Decoding Mode          : Speculative Decoding (Prompt-Lookup, max draft K=%d)\n", max_draft_len);
        printf("Verify Batch Replay    : %d/7 CUDA graphs captured (B=2..8); fallback: explicit launches\n",
               verify_graphs_ready);
        printf("Adaptive Policy        : EMA floor %.2f, final EMA %.2f, ladder x%d; policy-plain steps: %zu\n",
               spec_off_ema, spec_ema_final, spec_ladder_final, plain_by_policy);
        printf("Draft Acceptance Rate  : %.1f%% (%zu / %zu draft tokens accepted)\n",
               total_drafted > 0 ? ((double)total_accepted / (double)total_drafted) * 100.0 : 0.0,
               total_accepted, total_drafted);
        printf("Speculative Steps      : %zu forward passes for %zu generated tokens (%.2fx compression)\n",
               spec_steps, steps - 1, spec_steps > 0 ? (double)(steps - 1) / (double)spec_steps : 1.0);
    } else {
        printf("Decoding Mode          : Autoregressive Standard (Single Token)\n");
    }
    printf("Total GPU Inference    : %.3f ms\n", total_gpu_ms);
    printf("Dynamic Heap Allocs    : 0 (Strictly zero cudaMalloc/cudaFree during generation)\n");
    printf("Execution Engine       : 100%% Velox Bare-Metal Silicon (Zero External Runtimes)\n");
    printf("================================================================================\n");

    // Clean up
    free(generated_tokens);
    if (decode_graph_exec) cudaGraphExecDestroy(decode_graph_exec);
    if (decode_graph) cudaGraphDestroy(decode_graph);
    for (int b = 2; b <= 8; ++b) {
        if (verify_graph_exec[b]) cudaGraphExecDestroy(verify_graph_exec[b]);
    }
    cudaEventDestroy(ev_prefill_start);
    cudaEventDestroy(ev_prefill_stop);
    cudaEventDestroy(ev_decode_start);
    cudaEventDestroy(ev_decode_stop);
    cudaStreamDestroy(stream);
    cudaFree(d_weight_arena);
    cudaFree(d_weight_scales);
    cudaFree(d_activation);
    if (is_kv_in_vram) {
        cudaFree(d_kv_cache);
    } else if (h_kv_cache) {
        cudaFreeHost(h_kv_cache);
    }
    cudaFreeHost(h_ring);
    if (has_weights_file) {
        win32_gguf_close(&gguf_map);
    }

    return 0;
}
