#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

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

    if (memcmp(map->base_ptr, "GGUF", 4) != 0) return false;
    map->version = *(const uint32_t*)(map->base_ptr + 4);
    map->tensor_count = *(const uint64_t*)(map->base_ptr + 8);
    map->kv_count = *(const uint64_t*)(map->base_ptr + 16);

    size_t pos = 24;
    for (uint64_t i = 0; i < map->kv_count; ++i) {
        if (pos + 8 > map->file_size) return false;
        uint64_t klen = *(const uint64_t*)(map->base_ptr + pos);
        pos += 8 + (size_t)klen;
        if (pos + 4 > map->file_size) return false;
        uint32_t vtype = *(const uint32_t*)(map->base_ptr + pos);
        pos += 4;
        if (!skip_gguf_val(map->base_ptr, &pos, map->file_size, vtype)) return false;
    }

    map->tensors = (GgufTensorEntry*)calloc(map->tensor_count, sizeof(GgufTensorEntry));
    if (!map->tensors) return false;

    for (uint64_t i = 0; i < map->tensor_count; ++i) {
        if (pos + 8 > map->file_size) return false;
        uint64_t nlen = *(const uint64_t*)(map->base_ptr + pos);
        pos += 8;
        if (pos + nlen > map->file_size) return false;
        size_t cplen = (nlen < 127) ? (size_t)nlen : 127;
        memcpy(map->tensors[i].name, map->base_ptr + pos, cplen);
        map->tensors[i].name[cplen] = '\0';
        pos += (size_t)nlen;

        if (pos + 4 > map->file_size) return false;
        map->tensors[i].n_dims = *(const uint32_t*)(map->base_ptr + pos);
        pos += 4;

        for (uint32_t d = 0; d < map->tensors[i].n_dims; ++d) {
            if (pos + 8 > map->file_size) return false;
            map->tensors[i].dims[d] = *(const uint64_t*)(map->base_ptr + pos);
            pos += 8;
        }

        if (pos + 4 > map->file_size) return false;
        map->tensors[i].type = *(const uint32_t*)(map->base_ptr + pos);
        pos += 4;

        if (pos + 8 > map->file_size) return false;
        map->tensors[i].offset = *(const uint64_t*)(map->base_ptr + pos);
        pos += 8;

        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < map->tensors[i].n_dims; ++d) {
            n_elements *= map->tensors[i].dims[d];
        }

        size_t blk_size = 1;
        size_t type_bytes = 4;
        if (map->tensors[i].type == 0) {
            blk_size = 1; type_bytes = 4;
        } else if (map->tensors[i].type == 1) {
            blk_size = 1; type_bytes = 2;
        } else if (map->tensors[i].type == 12) { // Q4_K
            blk_size = 256; type_bytes = 144;
        } else if (map->tensors[i].type == 14) { // Q6_K
            blk_size = 256; type_bytes = 210;
        }
        map->tensors[i].byte_len = (size_t)((n_elements / blk_size) * type_bytes);
    }

    size_t alignment = 32;
    map->tensor_data_start = (pos + alignment - 1) & ~(alignment - 1);
    return true;
}

static const GgufTensorEntry* win32_gguf_find_tensor(const Win32GgufMap* map, const char* name) {
    for (uint64_t i = 0; i < map->tensor_count; ++i) {
        if (strcmp(map->tensors[i].name, name) == 0) {
            return &map->tensors[i];
        }
    }
    return NULL;
}

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(float));
    return out;
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

int main(void) {
    Win32GgufMap map;
    if (!win32_gguf_open("weights/k2_horizon_q4_k_m.gguf", &map)) {
        printf("Failed to open GGUF\n");
        return 1;
    }

    // 1. Test Q4_K dequantization on token_embd.weight block 0
    const GgufTensorEntry* te_emb = win32_gguf_find_tensor(&map, "token_embd.weight");
    if (te_emb) {
        const uint8_t* blk = map.base_ptr + map.tensor_data_start + te_emb->offset;
        uint16_t d_u16 = *(const uint16_t*)blk;
        uint16_t dmin_u16 = *(const uint16_t*)(blk + 2);
        float d = fp16_to_fp32(d_u16);
        float min = fp16_to_fp32(dmin_u16);
        const uint8_t* scales = blk + 4;
        const uint8_t* q = blk + 16;

        float vals[256];
        float* y = vals;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc, m;
            get_scale_min_k4(2*j, scales, &sc, &m);
            const float d1 = d * (float)sc;
            const float m1 = min * (float)m;
            get_scale_min_k4(2*j + 1, scales, &sc, &m);
            const float d2 = d * (float)sc;
            const float m2 = min * (float)m;
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (float)(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (float)(q[l] >> 4) - m2;
            }
            q += 32;
        }

        printf("Q4_K 'token_embd.weight' block 0 first 8 values:\n");
        for (int i = 0; i < 8; ++i) {
            printf("  [%d] = %f\n", i, vals[i]);
        }
    }

    // 2. Test Q6_K dequantization on output.weight block 0
    const GgufTensorEntry* te_out = win32_gguf_find_tensor(&map, "output.weight");
    if (te_out) {
        const uint8_t* blk = map.base_ptr + map.tensor_data_start + te_out->offset;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* sc = (const int8_t*)(blk + 192);
        uint16_t d_u16 = *(const uint16_t*)(blk + 208);
        float d = fp16_to_fp32(d_u16);

        float vals[256];
        for (int n = 0; n < 2; ++n) {
            const uint8_t* ql_n = ql + n * 64;
            const uint8_t* qh_n = qh + n * 32;
            const int8_t*  sc_n = sc + n * 8;
            float* y_n = vals + n * 128;
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql_n[l + 0]  & 0xF) | (((qh_n[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql_n[l + 32] & 0xF) | (((qh_n[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql_n[l + 0]  >> 4)  | (((qh_n[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql_n[l + 32] >> 4)  | (((qh_n[l] >> 6) & 3) << 4)) - 32;

                y_n[l + 0]  = d * (float)sc_n[is + 0] * (float)q1;
                y_n[l + 32] = d * (float)sc_n[is + 2] * (float)q2;
                y_n[l + 64] = d * (float)sc_n[is + 4] * (float)q3;
                y_n[l + 96] = d * (float)sc_n[is + 6] * (float)q4;
            }
        }

        printf("Q6_K 'output.weight' block 0 first 8 values:\n");
        for (int i = 0; i < 8; ++i) {
            printf("  [%d] = %f\n", i, vals[i]);
        }
    }

    return 0;
}
