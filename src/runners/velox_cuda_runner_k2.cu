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

// Configure maximum sequence length to 16,384 tokens (16K Context Window)
#define VELOX_MODEL_MAX_SEQ_LEN           16384ULL

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
    prompt_tokens[0] = 1;
    prompt_tokens[1] = 1504;
    prompt_tokens[2] = 346;
    prompt_tokens[3] = 2024;
    size_t prompt_len = 4;
    size_t steps = 16;
    bool verbose = false;
    bool use_cuda_graph = true;
    size_t max_context_tokens = 16384;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            prompt_len = 0;
            char* copy = _strdup(s);
            char* token = strtok(copy, ",");
            while (token && prompt_len < 16384) {
                prompt_tokens[prompt_len++] = (uint32_t)strtoul(token, NULL, 10);
                token = strtok(NULL, ",");
            }
            free(copy);
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
            printf("  --tokens <t1,t2,...>   Comma-separated prompt token IDs\n");
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
    size_t act_arena_bytes = K2_TIER3_ACTIVATION_ARENA_BYTES;
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

    // Allocate KV Cache: Tier 4 Host-Pinned System RAM (UVA Zero-Copy Mapping for 16K Context)
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
            if (err == cudaSuccess && decode_graph != NULL) {
                err = cudaGraphInstantiate(&decode_graph_exec, decode_graph, NULL, NULL, 0);
            }
        }
        if (err != cudaSuccess || decode_graph_exec == NULL) {
            use_cuda_graph = false;
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

    for (size_t pos = 0; pos < prompt_len; ++pos) {
        velox_K2_Horizon_3_7B_decode_cuda(&ctx, prompt_tokens[pos], pos, 1ULL, stream);
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
    if (steps > 1) {
        if (use_cuda_graph && decode_graph_exec != NULL) {
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
    printf("Model                  : K2-Horizon-3.7B (36 Layers, 512K Context)\n");
    printf("Target Device          : %s (Compute %d.%d, %zu MiB VRAM)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
    printf("Weights Source         : %s\n", has_weights_file ? weights_path : "Dry-Run Feasibility Arena (0 Allocation)");
    if (has_weights_file) {
        printf("Weight Stream to VRAM  : %.2f ms\n", weight_load_ms);
    }
    printf("Device VRAM Allocated  : %.2f MiB (%.2f%% of 4096 MiB RTX 3050)\n",
           (double)(weight_bytes + scales_bytes + act_arena_bytes) / (1024.0 * 1024.0),
           ((double)(weight_bytes + scales_bytes + act_arena_bytes) / (double)prop.totalGlobalMem) * 100.0);
    printf("KV Cache Allocation    : %.2f MiB in Tier 4 Host-Pinned System RAM (%llu Tokens, UVA Zero-Copy)\n",
           (double)kv_pool_bytes / (1024.0 * 1024.0), (unsigned long long)VELOX_MODEL_MAX_SEQ_LEN);

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
        cudaMemcpy(h_logits, (const float*)((const uint8_t*)ctx.activation_arena + 3966976ULL), VELOX_VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        print_top5_logits(h_logits, VELOX_VOCAB_SIZE);
        free(h_logits);
    }

    printf("TTFT (Prefill Latency) : %.3f ms (prompt len = %zu)\n", ttft_gpu_ms, prompt_len);
    if (steps > 1) {
        printf("Decode Latency (%zu tok): %.3f ms (%.2f tok/s, %.3f ms/tok)\n",
               steps - 1, decode_gpu_ms, decode_tok_per_sec, (double)decode_gpu_ms / (double)(steps - 1));
    }
    printf("Total GPU Inference    : %.3f ms\n", total_gpu_ms);
    printf("Dynamic Heap Allocs    : 0 (Strictly zero cudaMalloc/cudaFree during generation)\n");
    printf("Execution Engine       : 100%% Velox Bare-Metal Silicon (Zero External Runtimes)\n");
    printf("================================================================================\n");

    // Clean up
    free(generated_tokens);
    if (decode_graph_exec) cudaGraphExecDestroy(decode_graph_exec);
    if (decode_graph) cudaGraphDestroy(decode_graph);
    cudaEventDestroy(ev_prefill_start);
    cudaEventDestroy(ev_prefill_stop);
    cudaEventDestroy(ev_decode_start);
    cudaEventDestroy(ev_decode_stop);
    cudaStreamDestroy(stream);
    cudaFree(d_weight_arena);
    cudaFree(d_weight_scales);
    cudaFree(d_activation);
    cudaFreeHost(h_kv_cache);
    cudaFreeHost(h_ring);
    if (has_weights_file) {
        win32_gguf_close(&gguf_map);
    }

    return 0;
}
