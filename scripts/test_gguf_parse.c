#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

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
        case 0: case 1: case 7: // u8, i8, bool
            *pos += 1; return (*pos <= file_sz);
        case 2: case 3: // u16, i16
            *pos += 2; return (*pos <= file_sz);
        case 4: case 5: case 6: // u32, i32, f32
            *pos += 4; return (*pos <= file_sz);
        case 8: { // string
            if (*pos + 8 > file_sz) return false;
            uint64_t slen = *(const uint64_t*)(base + *pos);
            *pos += 8 + (size_t)slen;
            return (*pos <= file_sz);
        }
        case 10: case 11: case 12: // u64, i64, f64
            *pos += 8; return (*pos <= file_sz);
        case 9: { // array
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

    // Check magic
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

    // Skip KV entries
    for (uint64_t k = 0; k < map->kv_count; ++k) {
        if (pos + 8 > map->file_size) return false;
        uint64_t klen = *(const uint64_t*)(map->base_ptr + pos); pos += 8;
        pos += (size_t)klen;
        if (pos + 4 > map->file_size) return false;
        uint32_t vtype = *(const uint32_t*)(map->base_ptr + pos); pos += 4;
        if (!skip_gguf_val(map->base_ptr, &pos, map->file_size, vtype)) return false;
    }

    // Read Tensor entries
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

int main(void) {
    Win32GgufMap map;
    if (!win32_gguf_open("weights/k2_horizon_q4_k_m.gguf", &map)) {
        printf("Failed to open GGUF\n");
        return 1;
    }
    printf("Successfully opened GGUF! version=%u, tensors=%llu, kv=%llu, data_start=%zu\n",
           map.version, map.tensor_count, map.kv_count, map.tensor_data_start);

    for (uint64_t i = 0; i < map.tensor_count; ++i) {
        if (strstr(map.tensors[i].name, "blk.0.") != NULL || strstr(map.tensors[i].name, "norm") != NULL) {
            printf("  Tensor '%s': dims=[%llu, %llu], type=%u, byte_len=%zu\n",
                   map.tensors[i].name, map.tensors[i].dims[0], map.tensors[i].dims[1],
                   map.tensors[i].type, map.tensors[i].byte_len);
        }
    }
    win32_gguf_close(&map);
    return 0;
}
