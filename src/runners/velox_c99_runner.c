// ============================================================================
// Velox Emitted C99 Host Runner
// Target: x86_64 CPU (MSVC cl.exe /O2)
// Model:  SmolLM_135M (examples/tinystories_smollm.vlx)
// Zero dynamic allocations during inference (0 malloc/free in prefill/decode)
// ============================================================================

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

// Include the emitted standalone C99 model implementation
#include "tinystories_smollm.c"

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

// --- Weight Arena Population ------------------------------------------------
static bool populate_c99_weight_arena(
    uint8_t* weight_arena,
    size_t weight_arena_bytes,
    const Win32SafetensorsMap* st
) {
    size_t src_off = 0, len = 0;

    // 1. Embed tokens
    if (!safetensors_find_tensor(st, "model.embed_tokens.weight", &src_off, &len)) {
        fprintf(stderr, "Error: missing model.embed_tokens.weight in safetensors\n");
        return false;
    }
    memcpy(weight_arena + VELOX_W_OFFSET_W_EMBED, st->base_ptr + src_off, len);

    // 2. Transformer layers (0..29)
    for (size_t l = 0; l < 30; ++l) {
        char key[128];

        // Q projection
        snprintf(key, sizeof(key), "model.layers.%zu.self_attn.q_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        memcpy(weight_arena + VELOX_W_OFFSET_W_Q + l * (576 * 576 * sizeof(float)), st->base_ptr + src_off, len);

        // K projection
        snprintf(key, sizeof(key), "model.layers.%zu.self_attn.k_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        memcpy(weight_arena + VELOX_W_OFFSET_W_K + l * (192 * 576 * sizeof(float)), st->base_ptr + src_off, len);

        // V projection
        snprintf(key, sizeof(key), "model.layers.%zu.self_attn.v_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        memcpy(weight_arena + VELOX_W_OFFSET_W_V + l * (192 * 576 * sizeof(float)), st->base_ptr + src_off, len);

        // Out projection
        snprintf(key, sizeof(key), "model.layers.%zu.self_attn.o_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        memcpy(weight_arena + VELOX_W_OFFSET_W_OUT + l * (576 * 576 * sizeof(float)), st->base_ptr + src_off, len);

        // Gate projection
        snprintf(key, sizeof(key), "model.layers.%zu.mlp.gate_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        memcpy(weight_arena + VELOX_W_OFFSET_W_GATE + l * (1536 * 576 * sizeof(float)), st->base_ptr + src_off, len);

        // Up projection
        snprintf(key, sizeof(key), "model.layers.%zu.mlp.up_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        memcpy(weight_arena + VELOX_W_OFFSET_W_UP + l * (1536 * 576 * sizeof(float)), st->base_ptr + src_off, len);

        // Down projection
        snprintf(key, sizeof(key), "model.layers.%zu.mlp.down_proj.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        memcpy(weight_arena + VELOX_W_OFFSET_W_DOWN + l * (576 * 1536 * sizeof(float)), st->base_ptr + src_off, len);

        // Input layernorm
        snprintf(key, sizeof(key), "model.layers.%zu.input_layernorm.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        memcpy(weight_arena + VELOX_W_OFFSET_W_ATTN_NORM + l * (576 * sizeof(float)), st->base_ptr + src_off, len);

        // Post-attention layernorm
        snprintf(key, sizeof(key), "model.layers.%zu.post_attention_layernorm.weight", l);
        if (!safetensors_find_tensor(st, key, &src_off, &len)) return false;
        memcpy(weight_arena + VELOX_W_OFFSET_W_MLP_NORM + l * (576 * sizeof(float)), st->base_ptr + src_off, len);
    }

    // 3. Final layernorm
    if (!safetensors_find_tensor(st, "model.norm.weight", &src_off, &len)) {
        fprintf(stderr, "Error: missing model.norm.weight in safetensors\n");
        return false;
    }
    memcpy(weight_arena + VELOX_W_OFFSET_W_FINAL_NORM, st->base_ptr + src_off, len);

    // 4. Tied LM Head (same weights as model.embed_tokens.weight)
    if (!safetensors_find_tensor(st, "lm_head.weight", &src_off, &len)) {
        fprintf(stderr, "Error: missing lm_head.weight in safetensors\n");
        return false;
    }
    if (VELOX_W_OFFSET_W_HEAD + len <= weight_arena_bytes) {
        memcpy(weight_arena + VELOX_W_OFFSET_W_HEAD, st->base_ptr + src_off, len);
    }

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
        }
    }

    timer_init();

    // 1. Memory-map safetensors file via Win32 zero-copy
    LARGE_INTEGER t_load_start;
    QueryPerformanceCounter(&t_load_start);

    Win32SafetensorsMap st_map;
    if (!win32_safetensors_open(weights_path, &st_map)) {
        fprintf(stderr, "Error: failed to open safetensors file '%s'\n", weights_path);
        return 1;
    }

    // 2. Pre-allocate static Velox arenas (page-aligned, strictly once at startup)
    size_t weight_bytes = VELOX_WEIGHT_ARENA_BYTES;
    size_t kv_pool_bytes = VELOX_KV_CACHE_POOL_BYTES;
    size_t act_arena_bytes = VELOX_ACTIVATION_ARENA_BYTES;
    size_t ring_bytes = 4096;

    uint8_t* h_weight_arena = (uint8_t*)VirtualAlloc(NULL, weight_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    uint8_t* h_kv_cache = (uint8_t*)VirtualAlloc(NULL, kv_pool_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    uint8_t* h_activation = (uint8_t*)VirtualAlloc(NULL, act_arena_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    uint32_t* h_ring = (uint32_t*)VirtualAlloc(NULL, ring_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    if (!h_weight_arena || !h_kv_cache || !h_activation || !h_ring) {
        fprintf(stderr, "Error: failed to allocate host arenas\n");
        return 1;
    }

    // Zero-initialize dynamic pools
    memset(h_kv_cache, 0, kv_pool_bytes);
    memset(h_activation, 0, act_arena_bytes);
    memset(h_ring, 0, ring_bytes);

    // 3. Ingest weights from mmap into weight_arena once at startup
    if (!populate_c99_weight_arena(h_weight_arena, weight_bytes, &st_map)) {
        fprintf(stderr, "Error: failed to populate weight arena\n");
        return 1;
    }
    double load_ms = timer_ms_since(t_load_start);

    // 4. Construct Velox zero-allocation execution context
    VeloxArenaContext ctx;
    ctx.weight_arena = h_weight_arena;
    ctx.kv_cache_pool = h_kv_cache;
    ctx.activation_arena = h_activation;
    ctx.host_pinned_ring = h_ring;

    // Allocate output token buffer
    uint32_t* generated_tokens = (uint32_t*)malloc(steps * sizeof(uint32_t));
    if (!generated_tokens) return 1;

    // 5. Execute Prefill Phase on prompt tokens
    LARGE_INTEGER t_prefill_start;
    QueryPerformanceCounter(&t_prefill_start);

    uint32_t next_tok = prompt_tokens[0];
    for (size_t pos = 0; pos < prompt_len; ++pos) {
        next_tok = velox_SmolLM_135M_prefill(&ctx, prompt_tokens[pos], pos, 1ULL);
    }
    double ttft_ms = timer_ms_since(t_prefill_start);

    if (steps > 0) {
        generated_tokens[0] = next_tok;
    }

    // 6. Autoregressive Decode Phase (strictly 0 heap allocations)
    LARGE_INTEGER t_decode_start;
    QueryPerformanceCounter(&t_decode_start);

    for (size_t s = 1; s < steps; ++s) {
        size_t pos = prompt_len - 1 + s;
        next_tok = velox_SmolLM_135M_decode(&ctx, next_tok, pos, 1ULL);
        generated_tokens[s] = next_tok;
        if (verbose) {
            printf("[Step %zu] Token: %u\n", s, next_tok);
        }
    }
    double decode_ms = (steps > 1) ? timer_ms_since(t_decode_start) : 0.0;
    double total_inference_ms = ttft_ms + decode_ms;
    double decode_tok_per_sec = (steps > 1) ? (((double)(steps - 1) * 1000.0) / decode_ms) : 0.0;

    // 7. Standardized Velox Execution Report
    printf("================================================================================\n");
    printf("VELOX EMITTED C99 STANDALONE RUNTIME EXECUTION REPORT\n");
    printf("================================================================================\n");
    printf("Model                  : SmolLM_135M\n");
#if defined(_OPENMP)
    printf("Runner                 : Native C99 Multi-Core AVX2+FMA3 (%d threads) on Host CPU\n", omp_get_max_threads());
#else
    printf("Runner                 : Native C99 AVX2+FMA3 on Host CPU\n");
#endif
    printf("Safetensors Weights    : %s (Win32 Zero-Copy Mmap)\n", weights_path);
    printf("Weight Load / Ingest   : %.2f ms\n", load_ms);
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

    // Print final step top-5 logits
    print_top5_logits((const float*)ctx.activation_arena, VELOX_MODEL_VOCAB_SIZE);

    printf("TTFT (Prefill Latency) : %.3f ms\n", ttft_ms);
    if (steps > 1) {
        printf("Decode Latency (%zu tok): %.3f ms (%.2f tok/s, %.3f ms/tok)\n",
               steps - 1, decode_ms, decode_tok_per_sec, decode_ms / (double)(steps - 1));
    }
    printf("Total Inference Time   : %.3f ms\n", total_inference_ms);
    printf("Dynamic Heap Allocs    : 0 (Strictly zero malloc/free during generation)\n");

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
    printf("================================================================================\n");

    // Clean up
    free(generated_tokens);
    VirtualFree(h_weight_arena, 0, MEM_RELEASE);
    VirtualFree(h_kv_cache, 0, MEM_RELEASE);
    VirtualFree(h_activation, 0, MEM_RELEASE);
    VirtualFree(h_ring, 0, MEM_RELEASE);
    win32_safetensors_close(&st_map);

    int exit_code = 0;
    if (steps >= 32) {
        exit_code = (match_4 && match_32) ? 0 : 2;
    } else if (steps >= 4) {
        exit_code = match_4 ? 0 : 2;
    } else {
        exit_code = 0; // generation succeeded for steps < 4
    }
    return exit_code;
}
