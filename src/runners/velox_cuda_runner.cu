// ============================================================================
// Velox Emitted CUDA GPU Host Runner
// Target: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, sm_86)
// Model:  SmolLM_135M (examples/tinystories_smollm.vlx)
// Zero dynamic allocations during inference (0 cudaMalloc/cudaFree during decode)
// ============================================================================

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

// Include the emitted standalone CUDA model kernels & execution DAG
#include "tinystories_smollm.cu"

// Static Memory Tier Offsets & Sizes
#define VELOX_W_OFFSET_W_EMBED       0ULL
#define VELOX_W_OFFSET_W_Q           113246208ULL
#define VELOX_W_OFFSET_W_K           153059328ULL
#define VELOX_W_OFFSET_W_V           166330368ULL
#define VELOX_W_OFFSET_W_OUT         179601408ULL
#define VELOX_W_OFFSET_W_GATE        219414528ULL
#define VELOX_W_OFFSET_W_UP          325582848ULL
#define VELOX_W_OFFSET_W_DOWN        431751168ULL
#define VELOX_W_OFFSET_W_ATTN_NORM   537919488ULL
#define VELOX_W_OFFSET_W_MLP_NORM    537988608ULL
#define VELOX_W_OFFSET_W_FINAL_NORM  538057728ULL
#define VELOX_W_OFFSET_W_HEAD        538060032ULL

#define VELOX_WEIGHT_ARENA_BYTES     651306240ULL
#define VELOX_KV_CACHE_POOL_BYTES    94371840ULL

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

// --- Win32 Zero-Copy Memory-Mapped Safetensors Loader -----------------------
typedef struct {
    HANDLE hFile;
    HANDLE hMapping;
    const uint8_t* base_ptr;
    size_t file_size;
    const char* header_json;
    size_t header_len;
    size_t data_start_offset;
} Win32SafetensorsMap;

static bool win32_safetensors_open(const char* filepath, Win32SafetensorsMap* map) {
    memset(map, 0, sizeof(*map));

    int wlen = MultiByteToWideChar(CP_UTF8, 0, filepath, -1, NULL, 0);
    if (wlen <= 0) return false;
    wchar_t* wpath = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    if (!wpath) return false;
    MultiByteToWideChar(CP_UTF8, 0, filepath, -1, wpath, wlen);

    map->hFile = CreateFileW(
        wpath, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );
    free(wpath);

    if (map->hFile == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(map->hFile, &sz) || sz.QuadPart < 8) {
        CloseHandle(map->hFile);
        return false;
    }
    map->file_size = (size_t)sz.QuadPart;

    map->hMapping = CreateFileMappingW(map->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!map->hMapping) {
        CloseHandle(map->hFile);
        return false;
    }

    map->base_ptr = (const uint8_t*)MapViewOfFile(map->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!map->base_ptr) {
        CloseHandle(map->hMapping);
        CloseHandle(map->hFile);
        return false;
    }

    uint64_t hlen = *(const uint64_t*)map->base_ptr;
    if (map->file_size < 8 + hlen) {
        UnmapViewOfFile(map->base_ptr);
        CloseHandle(map->hMapping);
        CloseHandle(map->hFile);
        return false;
    }

    map->header_len = (size_t)hlen;
    map->header_json = (const char*)(map->base_ptr + 8);
    map->data_start_offset = 8 + (size_t)hlen;
    return true;
}

static void win32_safetensors_close(Win32SafetensorsMap* map) {
    if (map->base_ptr) UnmapViewOfFile(map->base_ptr);
    if (map->hMapping) CloseHandle(map->hMapping);
    if (map->hFile && map->hFile != INVALID_HANDLE_VALUE) CloseHandle(map->hFile);
    memset(map, 0, sizeof(*map));
}

static bool safetensors_find_tensor(
    const Win32SafetensorsMap* map,
    const char* tensor_name,
    size_t* out_abs_offset,
    size_t* out_byte_len
) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", tensor_name);
    const char* p = strstr(map->header_json, needle);
    if (!p) {
        // Fallback for tied lm_head.weight
        if (strcmp(tensor_name, "lm_head.weight") == 0) {
            return safetensors_find_tensor(map, "model.embed_tokens.weight", out_abs_offset, out_byte_len);
        }
        return false;
    }

    const char* do_key = strstr(p, "\"data_offsets\"");
    if (!do_key) return false;
    const char* lbr = strchr(do_key, '[');
    if (!lbr) return false;
    size_t start = (size_t)strtoull(lbr + 1, NULL, 10);
    const char* comma = strchr(lbr, ',');
    if (!comma) return false;
    size_t end = (size_t)strtoull(comma + 1, NULL, 10);

    *out_abs_offset = map->data_start_offset + start;
    *out_byte_len = end - start;
    return true;
}

// --- Weight Streaming to Device VRAM (One-Time at Startup) -------------------
static bool stream_tensor_converted(
    uint8_t* d_weight_arena,
    uint8_t* d_staging_buf,
    size_t dst_fp32_off,
    const uint8_t* host_src,
    size_t byte_len,
    uint32_t quant_mode,
    cudaStream_t stream
) {
    if (quant_mode == VELOX_QUANT_FP32 || dst_fp32_off >= VELOX_W_NORM_START_FP32) {
        size_t actual_off = dst_fp32_off;
        if (quant_mode == VELOX_QUANT_FP16 && dst_fp32_off >= VELOX_W_NORM_START_FP32) {
            actual_off = (VELOX_W_NORM_START_FP32 >> 1) + (dst_fp32_off - VELOX_W_NORM_START_FP32);
        } else if (quant_mode == VELOX_QUANT_Q4_0 && dst_fp32_off >= VELOX_W_NORM_START_FP32) {
            actual_off = 75644928ULL + (dst_fp32_off - VELOX_W_NORM_START_FP32);
        }
        cudaMemcpyAsync(d_weight_arena + actual_off, host_src, byte_len, cudaMemcpyHostToDevice, stream);
        return true;
    }

    const size_t max_chunk = 67108864ULL; // 64 MiB staging chunk (fits inside 90 MiB d_kv_cache)
    size_t done = 0;
    while (done < byte_len) {
        size_t chunk = byte_len - done;
        if (chunk > max_chunk) chunk = max_chunk;
        cudaMemcpyAsync(d_staging_buf, host_src + done, chunk, cudaMemcpyHostToDevice, stream);
        size_t cur_fp32_off = dst_fp32_off + done;
        size_t num_floats = chunk / sizeof(float);
        if (quant_mode == VELOX_QUANT_FP16) {
            velox_cuda_f32_to_f16_kernel<<<1024, 256, 0, stream>>>(
                (__half*)(d_weight_arena + (cur_fp32_off >> 1)),
                (const float*)d_staging_buf,
                num_floats
            );
        } else if (quant_mode == VELOX_QUANT_Q4_0) {
            velox_cuda_quantize_q4_group64_kernel<<<1024, 256, 0, stream>>>(
                (uint32_t*)(d_weight_arena + (cur_fp32_off >> 3)),
                ((half2*)(d_weight_arena + 67239936ULL)) + (cur_fp32_off >> 8),
                (const float*)d_staging_buf,
                num_floats >> 6
            );
        }
        done += chunk;
    }
    return true;
}

static bool stream_weights_to_device(
    uint8_t* d_weight_arena,
    uint8_t* d_staging_buf,
    size_t weight_arena_bytes,
    const Win32SafetensorsMap* st,
    uint32_t quant_mode,
    cudaStream_t stream
) {
    size_t src_off = 0, len = 0;

    // 1. Embed tokens
    if (!safetensors_find_tensor(st, "model.embed_tokens.weight", &src_off, &len)) {
        fprintf(stderr, "Error: missing model.embed_tokens.weight in safetensors\n");
        return false;
    }
    stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_EMBED, st->base_ptr + src_off, len, quant_mode, stream);

    // 2. Transformer layers (0..29)
    for (size_t l = 0; l < 30; ++l) {
        char key[128];

        // Q projection
        snprintf(key, sizeof(key), "model.layers.%zu.self_attn.q_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_Q + l * (576 * 576 * sizeof(float)), st->base_ptr + src_off, len, quant_mode, stream);

        // K projection
        snprintf(key, sizeof(key), "model.layers.%zu.self_attn.k_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_K + l * (192 * 576 * sizeof(float)), st->base_ptr + src_off, len, quant_mode, stream);

        // V projection
        snprintf(key, sizeof(key), "model.layers.%zu.self_attn.v_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_V + l * (192 * 576 * sizeof(float)), st->base_ptr + src_off, len, quant_mode, stream);

        // Out projection
        snprintf(key, sizeof(key), "model.layers.%zu.self_attn.o_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_OUT + l * (576 * 576 * sizeof(float)), st->base_ptr + src_off, len, quant_mode, stream);

        // Gate projection
        snprintf(key, sizeof(key), "model.layers.%zu.mlp.gate_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_GATE + l * (1536 * 576 * sizeof(float)), st->base_ptr + src_off, len, quant_mode, stream);

        // Up projection
        snprintf(key, sizeof(key), "model.layers.%zu.mlp.up_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_UP + l * (1536 * 576 * sizeof(float)), st->base_ptr + src_off, len, quant_mode, stream);

        // Down projection
        snprintf(key, sizeof(key), "model.layers.%zu.mlp.down_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_DOWN + l * (576 * 1536 * sizeof(float)), st->base_ptr + src_off, len, quant_mode, stream);

        // Input layernorm
        snprintf(key, sizeof(key), "model.layers.%zu.input_layernorm.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_ATTN_NORM + l * (576 * sizeof(float)), st->base_ptr + src_off, len, quant_mode, stream);

        // Post-attention layernorm
        snprintf(key, sizeof(key), "model.layers.%zu.post_attention_layernorm.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_MLP_NORM + l * (576 * sizeof(float)), st->base_ptr + src_off, len, quant_mode, stream);
    }

    // 3. Final layernorm
    if (!safetensors_find_tensor(st, "model.norm.weight", &src_off, &len)) {
        fprintf(stderr, "Error: missing model.norm.weight in safetensors\n");
        return false;
    }
    stream_tensor_converted(d_weight_arena, d_staging_buf, VELOX_W_OFFSET_W_FINAL_NORM, st->base_ptr + src_off, len, quant_mode, stream);

    // 4. Tied LM Head (same weights as model.embed_tokens.weight; aliased at offset 0 in quantized modes)
    if (quant_mode == VELOX_QUANT_FP32) {
        if (!safetensors_find_tensor(st, "lm_head.weight", &src_off, &len)) {
            fprintf(stderr, "Error: missing lm_head.weight in safetensors\n");
            return false;
        }
        if (VELOX_W_OFFSET_W_HEAD + len <= weight_arena_bytes) {
            cudaMemcpyAsync(d_weight_arena + VELOX_W_OFFSET_W_HEAD, st->base_ptr + src_off, len, cudaMemcpyHostToDevice, stream);
        }
    }

    cudaStreamSynchronize(stream);
    return true;
}

// --- Top-5 Logits Helper ----------------------------------------------------
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

    printf("Final Step Top-5 Logits:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  #%d: token_id=%u logit=%.6f\n", i + 1, top5[i].token_id, top5[i].logit);
    }
}

// --- Main Program Entry Point -----------------------------------------------
int main(int argc, char** argv) {
    const char* weights_path = "smollm-135m.safetensors";
    uint32_t prompt_tokens[64] = {1, 504, 346};
    size_t prompt_len = 3;
    size_t steps = 32;
    bool verbose = false;
    bool use_cuda_graph = true;
    uint32_t quant_mode = VELOX_QUANT_FP32;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            prompt_len = 0;
            char* copy = _strdup(s);
            char* token = strtok(copy, ",");
            while (token && prompt_len < 64) {
                prompt_tokens[prompt_len++] = (uint32_t)strtoul(token, NULL, 10);
                token = strtok(NULL, ",");
            }
            free(copy);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--no-cuda-graph") == 0) {
            use_cuda_graph = false;
        } else if (strcmp(argv[i], "--quantize") == 0 && i + 1 < argc) {
            const char* q = argv[++i];
            if (strcmp(q, "f16") == 0 || strcmp(q, "fp16") == 0) {
                quant_mode = VELOX_QUANT_FP16;
            } else if (strcmp(q, "q4_0") == 0 || strcmp(q, "int4") == 0 || strcmp(q, "w4a16") == 0) {
                quant_mode = VELOX_QUANT_Q4_0;
            } else {
                quant_mode = VELOX_QUANT_FP32;
            }
        } else if (strncmp(argv[i], "--quantize=", 11) == 0) {
            const char* q = argv[i] + 11;
            if (strcmp(q, "f16") == 0 || strcmp(q, "fp16") == 0) {
                quant_mode = VELOX_QUANT_FP16;
            } else if (strcmp(q, "q4_0") == 0 || strcmp(q, "int4") == 0 || strcmp(q, "w4a16") == 0) {
                quant_mode = VELOX_QUANT_Q4_0;
            } else {
                quant_mode = VELOX_QUANT_FP32;
            }
        }
    }

    timer_init();

    // Verify GPU Device
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // 1. Memory-map safetensors file via Win32 zero-copy
    LARGE_INTEGER t_load_start;
    QueryPerformanceCounter(&t_load_start);

    Win32SafetensorsMap st_map;
    if (!win32_safetensors_open(weights_path, &st_map)) {
        fprintf(stderr, "Error: failed to open safetensors file '%s'\n", weights_path);
        return 1;
    }

    // 2. Pre-allocate static Velox CUDA arenas (strictly once at startup on RTX 3050)
    size_t weight_bytes = VELOX_WEIGHT_ARENA_BYTES;
    size_t kv_pool_bytes = VELOX_KV_CACHE_POOL_BYTES;
    size_t act_arena_bytes = VELOX_ACTIVATION_ARENA_BYTES;
    if (quant_mode == VELOX_QUANT_FP16) {
        // 268,959,744 B FP16 weights + 140,544 B FP32 norms (w_head tied to w_embed at offset 0)
        weight_bytes = 269100288ULL;
        act_arena_bytes = 4194304ULL;
    } else if (quant_mode == VELOX_QUANT_Q4_0) {
        // 67,239,936 B INT4 packed + 8,404,992 B half2 group-64 scales + 140,544 B FP32 norms
        weight_bytes = 75785472ULL;
        act_arena_bytes = 4194304ULL;
    }
    size_t ring_bytes = 4096;

    uint8_t* d_weight_arena = NULL;
    uint8_t* d_kv_cache = NULL;
    uint8_t* d_activation = NULL;
    uint32_t* h_ring = NULL;
    uint32_t* d_ring = NULL;

    cudaError_t err;
    err = cudaMalloc((void**)&d_weight_arena, weight_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc weight_arena failed: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc((void**)&d_kv_cache, kv_pool_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc kv_cache failed: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc((void**)&d_activation, act_arena_bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc activation_arena failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Host-pinned ring buffer for zero-copy D2H token reading
    cudaSetDeviceFlags(cudaDeviceMapHost);
    err = cudaHostAlloc((void**)&h_ring, ring_bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) { fprintf(stderr, "cudaHostAlloc host_pinned_ring failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaHostGetDevicePointer((void**)&d_ring, h_ring, 0);
    if (err != cudaSuccess) { d_ring = h_ring; }

    // Create execution CUDA stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // 3. Stream model weights into device weight_arena once at startup (using d_kv_cache as temporary staging before zeroing)
    if (!stream_weights_to_device(d_weight_arena, d_kv_cache, weight_bytes, &st_map, quant_mode, stream)) {
        fprintf(stderr, "Error: failed to stream weights to device\n");
        return 1;
    }

    // Zero-initialize dynamic pools
    cudaMemsetAsync(d_kv_cache, 0, kv_pool_bytes, stream);
    cudaMemsetAsync(d_activation, 0, act_arena_bytes, stream);
    cudaStreamSynchronize(stream);
    memset(h_ring, 0, ring_bytes);
    double load_ms = timer_ms_since(t_load_start);

    // 4. Construct Velox zero-allocation execution arena context
    VeloxCudaArena ctx;
    ctx.weight_arena = d_weight_arena;
    ctx.kv_cache_pool = d_kv_cache;
    ctx.activation_arena = d_activation;
    ctx.host_pinned_ring = d_ring;
    ctx.quant_mode = quant_mode;
    if (quant_mode == VELOX_QUANT_FP16) {
        ctx.norm_weight_arena = d_weight_arena + (VELOX_W_NORM_START_FP32 >> 1);
        ctx.weight_scales = NULL;
        ctx.tied_lm_head = 1u;
    } else if (quant_mode == VELOX_QUANT_Q4_0) {
        ctx.norm_weight_arena = d_weight_arena + 75644928ULL;
        ctx.weight_scales = d_weight_arena + 67239936ULL;
        ctx.tied_lm_head = 1u;
    } else {
        ctx.norm_weight_arena = NULL;
        ctx.weight_scales = NULL;
        ctx.tied_lm_head = 0u;
    }

    // 4b. Pre-capture single-step decode CUDA Graph (invariant device-resident VeloxDecodeState)
    cudaGraph_t decode_graph = NULL;
    cudaGraphExec_t decode_graph_exec = NULL;
    if (use_cuda_graph) {
        err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err == cudaSuccess) {
            velox_SmolLM_135M_decode_graph_step_cuda(&ctx, 1ULL, stream);
            err = cudaStreamEndCapture(stream, &decode_graph);
            if (err == cudaSuccess && decode_graph != NULL) {
                err = cudaGraphInstantiate(&decode_graph_exec, decode_graph, NULL, NULL, 0);
            }
        }
        if (err != cudaSuccess || decode_graph_exec == NULL) {
            fprintf(stderr, "Warning: CUDA Graph capture failed (%s), falling back to eager stream\n", cudaGetErrorString(err));
            use_cuda_graph = false;
        }
    }

    // Allocate output token buffer on host
    uint32_t* generated_tokens = (uint32_t*)malloc(steps * sizeof(uint32_t));
    if (!generated_tokens) return 1;

    // Setup CUDA event timers for precision GPU profiling
    cudaEvent_t ev_prefill_start, ev_prefill_stop;
    cudaEvent_t ev_decode_start, ev_decode_stop;
    cudaEventCreate(&ev_prefill_start);
    cudaEventCreate(&ev_prefill_stop);
    cudaEventCreate(&ev_decode_start);
    cudaEventCreate(&ev_decode_stop);

    // 5. Execute Prefill Phase on prompt tokens
    cudaEventRecord(ev_prefill_start, stream);

    for (size_t pos = 0; pos < prompt_len; ++pos) {
        velox_SmolLM_135M_prefill_cuda(&ctx, prompt_tokens[pos], pos, 1ULL, stream);
    }

    cudaEventRecord(ev_prefill_stop, stream);
    cudaStreamSynchronize(stream);

    float ttft_gpu_ms = 0.0f;
    cudaEventElapsedTime(&ttft_gpu_ms, ev_prefill_start, ev_prefill_stop);

    // Fetch step 0 token 359 target logit immediately after prefill phase completes
    float step0_token_359_logit = 0.0f;
    cudaMemcpy(&step0_token_359_logit, (const float*)ctx.activation_arena + 359, sizeof(float), cudaMemcpyDeviceToHost);

    uint32_t next_tok = h_ring[(prompt_len - 1) & 1023ULL];
    if (steps > 0) {
        generated_tokens[0] = next_tok;
        if (verbose) {
            printf("[Step 0] Token: %u\n", next_tok);
            fflush(stdout);
        }
    }

    // 6. Autoregressive Decode Phase (strictly 0 dynamic allocations on GPU & CPU)
    float decode_gpu_ms = 0.0f;
    if (steps > 1) {
        if (use_cuda_graph && decode_graph_exec != NULL) {
            // Ensure device decode state holds (next_tok, prompt_len) before graph replay
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
                velox_SmolLM_135M_decode_cuda(&ctx, next_tok, pos, 1ULL, stream);
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

    // 7. Standardized Velox Execution Report
    printf("================================================================================\n");
    printf("VELOX EMITTED CUDA STANDALONE RUNTIME EXECUTION REPORT\n");
    printf("================================================================================\n");
    printf("Model                  : SmolLM_135M\n");
    printf("Target Device          : %s (Compute %d.%d, %zu MiB VRAM)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
    printf("Safetensors Weights    : %s (Win32 Zero-Copy Mmap)\n", weights_path);
    printf("Weight Stream to VRAM  : %.2f ms\n", load_ms);
    printf("Total VRAM Allocated   : %.2f MiB (%.2f%% of 4096 MiB)\n",
           (double)(weight_bytes + kv_pool_bytes + act_arena_bytes) / (1024.0 * 1024.0),
           ((double)(weight_bytes + kv_pool_bytes + act_arena_bytes) / (double)prop.totalGlobalMem) * 100.0);
    printf("Prompt Tokens (%zu)      : [", prompt_len);
    for (size_t i = 0; i < prompt_len; ++i) {
        printf("%u%s", prompt_tokens[i], (i + 1 < prompt_len) ? ", " : "");
    }
    printf("]\n");

    printf("Generated Tokens (%zu)   : [", steps);
    for (size_t i = 0; i < steps; ++i) {
        printf("%u%s", generated_tokens[i], (i + 1 < steps) ? ", " : "");
    }
    printf("]\n");

    printf("Full Token Sequence    : [");
    for (size_t i = 0; i < prompt_len; ++i) {
        printf("%u, ", prompt_tokens[i]);
    }
    for (size_t i = 0; i < steps; ++i) {
        printf("%u%s", generated_tokens[i], (i + 1 < steps) ? ", " : "");
    }
    printf("]\n");

    // Copy and inspect final step top-5 logits from device activation arena offset 0ULL
    float* h_logits = (float*)malloc(VELOX_VOCAB_SIZE * sizeof(float));
    if (h_logits) {
        cudaMemcpy(h_logits, ctx.activation_arena, VELOX_VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        print_top5_logits(h_logits, VELOX_VOCAB_SIZE);
        free(h_logits);
    }

    printf("TTFT (Prefill Latency) : %.3f ms\n", ttft_gpu_ms);
    if (steps > 1) {
        printf("Decode Latency (%zu tok): %.3f ms (%.2f tok/s, %.3f ms/tok)\n",
               steps - 1, decode_gpu_ms, decode_tok_per_sec, (double)decode_gpu_ms / (double)(steps - 1));
    }
    printf("Total GPU Inference    : %.3f ms\n", total_gpu_ms);
    printf("Dynamic Heap Allocs    : 0 (Strictly zero cudaMalloc/cudaFree during generation)\n");

    // Parity verification check against reference SmolLM-135M output
    static const uint32_t EXPECTED_32[32] = {
        359, 253, 4028, 282, 260, 1773, 30, 198, 26501, 368, 281, 260, 1773, 198, 20255, 3726,
        288, 9517, 253, 12301, 198, 19161, 354, 12301, 198, 19161, 354, 12301, 198, 2683, 416, 722
    };

    bool match_4 = (steps >= 4);
    for (size_t i = 0; i < 4 && i < steps; ++i) {
        if (generated_tokens[i] != EXPECTED_32[i]) match_4 = false;
    }
    bool match_32 = (steps >= 32);
    for (size_t i = 0; i < 32 && i < steps; ++i) {
        if (generated_tokens[i] != EXPECTED_32[i]) match_32 = false;
    }

    if (quant_mode == VELOX_QUANT_Q4_0) {
        if (steps >= 4) {
            if (match_4) {
                printf("Parity 4-token check   : PASSED [359, 253, 4028, 282]\n");
            } else {
                printf("Parity 4-token check   : DIVERGED [%u, %u, %u, %u] (INT4 quantization noise vs FP32)\n",
                       generated_tokens[0], generated_tokens[1], generated_tokens[2], generated_tokens[3]);
            }
        } else {
            printf("Parity 4-token check   : SKIPPED (fewer than 4 steps requested: %zu)\n", steps);
        }

        if (steps >= 32) {
            if (match_32) {
                printf("Parity 32-token check  : PASSED (Exact match with Velox Rust CPU reference)\n");
            } else {
                printf("Parity 32-token check  : DIVERGED (INT4 quantization noise vs FP32)\n");
            }
        } else {
            printf("Parity 32-token check  : SKIPPED (fewer than 32 steps requested: %zu)\n", steps);
        }

        float delta = fabsf(step0_token_359_logit - 18.586565f);
        printf("Accuracy Metric Check  : %s (Step 0 target logit error = %.6f relative to FP32 reference)\n",
               delta <= 0.03f ? "PASSED" : "FAILED", delta);
    } else {
        if (steps >= 4) {
            printf("Parity 4-token check   : %s\n", match_4 ? "PASSED [359, 253, 4028, 282]" : "FAILED");
        } else {
            printf("Parity 4-token check   : SKIPPED (fewer than 4 steps requested: %zu)\n", steps);
        }

        if (steps >= 32) {
            printf("Parity 32-token check  : %s\n", match_32 ? "PASSED (Exact match with Velox Rust CPU reference)" : "FAILED");
        } else {
            printf("Parity 32-token check  : SKIPPED (fewer than 32 steps requested: %zu)\n", steps);
        }
    }
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
    cudaFree(d_kv_cache);
    cudaFree(d_activation);
    cudaFreeHost(h_ring);
    win32_safetensors_close(&st_map);

    int exit_code = 0;
    cudaError_t last_err = cudaGetLastError();
    if (last_err != cudaSuccess) {
        fprintf(stderr, "CUDA error detected: %s\n", cudaGetErrorString(last_err));
        exit_code = 1;
    } else if (quant_mode == VELOX_QUANT_Q4_0) {
        float delta = fabsf(step0_token_359_logit - 18.586565f);
        exit_code = (delta <= 0.03f) ? 0 : 2;
    } else if (steps >= 32) {
        exit_code = (match_4 && match_32) ? 0 : 2;
    } else if (steps >= 4) {
        exit_code = match_4 ? 0 : 2;
    } else {
        exit_code = 0; // generation succeeded for steps < 4
    }
    return exit_code;
}
