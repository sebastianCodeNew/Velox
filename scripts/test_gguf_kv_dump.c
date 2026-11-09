// ==========================================================================
// Velox GGUF Header Metadata Dumper (Win32 Zero-Copy)
//
// Purpose:
//   Separates diagnosis from repair: walks the GGUF header key-value
//   metadata dictionary of a model file and prints every entry with its
//   GGUF v3 value type and decoded value, WITHOUT reading tensor data.
//   Used to cross-check Velox compile-time model constants (RoPE theta,
//   RMSNorm eps, hidden dims, KV layout) against the true weights file.
//
// Build (MSVC x64, from the repository root):
//   cl /nologo /W4 /O2 scripts\test_gguf_kv_dump.c /Fe:scripts\test_gguf_kv_dump.exe
// Usage:
//   scripts\test_gguf_kv_dump.exe [path/to/model.gguf]
//     (path resolved from the current working directory only;
//      defaults to weights/k2_horizon_q4_k_m.gguf)
// ==========================================================================

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * GGUF v3 metadata value types (GGUF spec, llama.cpp gguf.h)
 * ------------------------------------------------------------------------- */
enum {
    GGUF_VT_U8      = 0,
    GGUF_VT_I8      = 1,
    GGUF_VT_U16     = 2,
    GGUF_VT_I16     = 3,
    GGUF_VT_U32     = 4,
    GGUF_VT_I32     = 5,
    GGUF_VT_F32     = 6,
    GGUF_VT_BOOL    = 7,
    GGUF_VT_STRING  = 8,
    GGUF_VT_ARRAY   = 9,
    GGUF_VT_U64     = 10,
    GGUF_VT_I64     = 11,
    GGUF_VT_F64     = 12
};

static const char* gguf_vt_name(uint32_t t) {
    switch (t) {
        case GGUF_VT_U8:     return "u8";
        case GGUF_VT_I8:     return "i8";
        case GGUF_VT_U16:    return "u16";
        case GGUF_VT_I16:    return "i16";
        case GGUF_VT_U32:    return "u32";
        case GGUF_VT_I32:    return "i32";
        case GGUF_VT_F32:    return "f32";
        case GGUF_VT_BOOL:   return "bool";
        case GGUF_VT_STRING: return "str";
        case GGUF_VT_ARRAY:  return "arr";
        case GGUF_VT_U64:    return "u64";
        case GGUF_VT_I64:    return "i64";
        case GGUF_VT_F64:    return "f64";
        default:             return "?";
    }
}

static const char* gguf_tensor_type_name(uint32_t t) {
    switch (t) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 6:  return "Q5_0";
        case 7:  return "Q5_1";
        case 8:  return "Q8_0";
        case 9:  return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 22: return "IQ3_S";
        case 24: return "IQ4_XS";
        case 30: return "BF16";
        default: return "OTHER";
    }
}

/* Win32 file mapping handle bundle ---------------------------------------- */
typedef struct {
    HANDLE hFile;
    HANDLE hMapping;
    const uint8_t* base_ptr;
    size_t file_size;
} MappedFile;

static bool map_file(const char* path, MappedFile* mf) {
    memset(mf, 0, sizeof(*mf));
    mf->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (mf->hFile == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(mf->hFile, &sz) || sz.QuadPart <= 0) {
        CloseHandle(mf->hFile);
        mf->hFile = NULL;
        return false;
    }
    mf->file_size = (size_t)sz.QuadPart;

    mf->hMapping = CreateFileMappingA(mf->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mf->hMapping == NULL) {
        CloseHandle(mf->hFile);
        mf->hFile = NULL;
        return false;
    }
    mf->base_ptr = (const uint8_t*)MapViewOfFile(mf->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (mf->base_ptr == NULL) {
        CloseHandle(mf->hMapping);
        CloseHandle(mf->hFile);
        mf->hMapping = NULL;
        mf->hFile = NULL;
        return false;
    }
    return true;
}

static void unmap_file(MappedFile* mf) {
    if (mf->base_ptr)   UnmapViewOfFile(mf->base_ptr);
    if (mf->hMapping)   CloseHandle(mf->hMapping);
    if (mf->hFile && mf->hFile != INVALID_HANDLE_VALUE) CloseHandle(mf->hFile);
    memset(mf, 0, sizeof(*mf));
}

/* Bounds-checked reader ----------------------------------------------------- */
typedef struct {
    const uint8_t* base;
    size_t size;
    size_t pos;
} Reader;

static bool r_avail(const Reader* r, size_t n) {
    return r->pos + n <= r->size;
}

static bool r_u16(Reader* r, uint16_t* out) {
    if (!r_avail(r, 2)) return false;
    *out = *(const uint16_t*)(r->base + r->pos);
    r->pos += 2;
    return true;
}

static bool r_u32(Reader* r, uint32_t* out) {
    if (!r_avail(r, 4)) return false;
    *out = *(const uint32_t*)(r->base + r->pos);
    r->pos += 4;
    return true;
}

static bool r_u64(Reader* r, uint64_t* out) {
    if (!r_avail(r, 8)) return false;
    *out = *(const uint64_t*)(r->base + r->pos);
    r->pos += 8;
    return true;
}

static bool r_u8(Reader* r, uint8_t* out) {
    if (!r_avail(r, 1)) return false;
    *out = r->base[r->pos];
    r->pos += 1;
    return true;
}

static bool r_string(Reader* r, char** out) {
    uint64_t len;
    if (!r_u64(r, &len)) return false;
    if (len > (uint64_t)r->size) return false;
    if (!r_avail(r, (size_t)len)) return false;
    char* s = (char*)malloc((size_t)len + 1);
    if (!s) return false;
    memcpy(s, r->base + r->pos, (size_t)len);
    s[len] = '\0';
    r->pos += (size_t)len;
    *out = s;
    return true;
}

/* Print one scalar value of the given type from the CURRENT reader position,
 * or, for strings/arrays, print the full inline value. Returns false only on
 * malformed input; the reader position always advances past the value.       */
#define KV_ARRAY_PRINT_CAP 10

static void print_f32_or_f64(Reader* r, bool is_f64) {
    if (is_f64) {
        double g = 0.0;
        if (r_avail(r, 8)) memcpy(&g, r->base + r->pos, 8);
        printf(" %.10g", g);
    } else {
        float f = 0.0f;
        if (r_avail(r, 4)) memcpy(&f, r->base + r->pos, 4);
        printf(" %.6g", (double)f);
    }
}

static bool print_scalar(Reader* r, uint32_t vt) {
    uint8_t  b = 0;
    uint16_t w = 0;
    uint16_t w16 = 0;
    uint32_t d = 0;
    uint64_t q = 0;

    switch (vt) {
        case GGUF_VT_U8:
            if (!r_u8(r, &b)) return false;
            printf(" %u\n", b);
            return true;
        case GGUF_VT_I8:
            if (!r_u8(r, &b)) return false;
            printf(" %d\n", (int)(char)b);
            return true;
        case GGUF_VT_U16:
            if (!r_u16(r, &w)) return false;
            printf(" %u\n", w);
            return true;
        case GGUF_VT_I16:
            if (!r_u16(r, &w16)) return false;
            printf(" %d\n", (int)(short)w16);
            return true;
        case GGUF_VT_U32:
            if (!r_u32(r, &d)) return false;
            printf(" %lu\n", (unsigned long)d);
            return true;
        case GGUF_VT_I32:
            if (!r_u32(r, &d)) return false;
            printf(" %ld\n", (long)(int32_t)d);
            return true;
        case GGUF_VT_F32:
            if (!r_avail(r, 4)) return false;
            print_f32_or_f64(r, false);
            r->pos += 4;
            printf("\n");
            return true;
        case GGUF_VT_BOOL:
            if (!r_u8(r, &b)) return false;
            printf(" %s\n", b ? "true" : "false");
            return true;
        case GGUF_VT_U64:
            if (!r_u64(r, &q)) return false;
            printf(" %llu\n", (unsigned long long)q);
            return true;
        case GGUF_VT_I64:
            if (!r_u64(r, &q)) return false;
            printf(" %lld\n", (long long)q);
            return true;
        case GGUF_VT_F64:
            if (!r_avail(r, 8)) return false;
            print_f32_or_f64(r, true);
            r->pos += 8;
            printf("\n");
            return true;
        default:
            return false;
    }
}

static bool array_elem_size(uint32_t et, size_t* esz) {
    switch (et) {
        case GGUF_VT_U8: case GGUF_VT_I8: case GGUF_VT_BOOL:  *esz = 1; break;
        case GGUF_VT_U16: case GGUF_VT_I16:                   *esz = 2; break;
        case GGUF_VT_U32: case GGUF_VT_I32: case GGUF_VT_F32: *esz = 4; break;
        case GGUF_VT_U64: case GGUF_VT_I64: case GGUF_VT_F64: *esz = 8; break;
        default: return false;
    }
    return true;
}

static bool print_array_elem(Reader* r, uint32_t et, char sep_ch) {
    uint8_t  b;
    uint16_t w;
    uint16_t w16;
    uint32_t d;
    uint64_t q;
    char*    s;

    if (sep_ch) printf("%c", sep_ch);

    switch (et) {
        case GGUF_VT_U8:
            if (!r_u8(r, &b)) return false;
            printf("%u", b);
            return true;
        case GGUF_VT_I8:
            if (!r_u8(r, &b)) return false;
            printf("%d", (int)(char)b);
            return true;
        case GGUF_VT_U16:
            if (!r_u16(r, &w)) return false;
            printf("%u", w);
            return true;
        case GGUF_VT_I16:
            if (!r_u16(r, &w16)) return false;
            printf("%d", (int)(short)w16);
            return true;
        case GGUF_VT_U32:
            if (!r_u32(r, &d)) return false;
            printf("%lu", (unsigned long)d);
            return true;
        case GGUF_VT_I32:
            if (!r_u32(r, &d)) return false;
            printf("%ld", (long)(int32_t)d);
            return true;
        case GGUF_VT_F32:
            if (!r_avail(r, 4)) return false;
            print_f32_or_f64(r, false);
            r->pos += 4;
            return true;
        case GGUF_VT_U64:
            if (!r_u64(r, &q)) return false;
            printf("%llu", (unsigned long long)q);
            return true;
        case GGUF_VT_I64:
            if (!r_u64(r, &q)) return false;
            printf("%lld", (long long)q);
            return true;
        case GGUF_VT_F64:
            if (!r_avail(r, 8)) return false;
            print_f32_or_f64(r, true);
            r->pos += 8;
            return true;
        case GGUF_VT_BOOL:
            if (!r_u8(r, &b)) return false;
            printf("%s", b ? "true" : "false");
            return true;
        case GGUF_VT_STRING:
            if (!r_string(r, &s)) return false;
            printf("\"%s\"", s);
            free(s);
            return true;
        default:
            return false;
    }
}

static bool print_array(Reader* r, uint32_t et, uint64_t count, int depth) {
    printf("\n");
    (void)depth;
    printf("    value: %s len=%llu: [",
           gguf_vt_name(et), (unsigned long long)count);

    const uint64_t shown = count < (uint64_t)KV_ARRAY_PRINT_CAP
                               ? count : (uint64_t)KV_ARRAY_PRINT_CAP;
    for (uint64_t i = 0; i < shown; ++i) {
        if (!print_array_elem(r, et, i == 0 ? ' ' : ',')) return false;
    }
    if (shown < count) {
        /* Fast bulk skip for fixed-size element types; walk for strings. */
        size_t esz;
        if (array_elem_size(et, &esz)) {
            const uint64_t rest = count - shown;
            /* Overflow-safe skip: refuse products that cannot be expressed. */
            if (rest != 0 && (SIZE_MAX / esz) < rest) return false;
            if (!r_avail(r, (size_t)rest * esz)) return false;
            r->pos += (size_t)rest * esz;
        } else if (et == GGUF_VT_STRING) {
            for (uint64_t i = count - shown; i > 0; --i) {
                uint64_t slen;
                if (!r_u64(r, &slen) || slen > r->size || !r_avail(r, (size_t)slen)) {
                    return false;
                }
                r->pos += (size_t)slen;
            }
        } else {
            return false;
        }
        printf(", ... %llu more", (unsigned long long)(count - shown));
    }
    printf(" ]\n");
    return true;
}

int main(int argc, char** argv) {
    const char* default_path = "weights/k2_horizon_q4_k_m.gguf";
    const char* path = (argc > 1 && argv[1][0] != '\0') ? argv[1] : default_path;

    MappedFile mf;
    if (!map_file(path, &mf)) {
        fprintf(stderr, "Failed to open/mmap '%s' (GetLastError=%lu)\n",
                path, (unsigned long)GetLastError());
        return 1;
    }

    if (mf.file_size < 8 || memcmp(mf.base_ptr, "GGUF", 4) != 0) {
        fprintf(stderr, "Not a GGUF file (bad magic)\n");
        unmap_file(&mf);
        return 1;
    }

    Reader r = { mf.base_ptr, mf.file_size, 4 };
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;

    if (!r_u32(&r, &version) || !r_u64(&r, &tensor_count) || !r_u64(&r, &kv_count)) {
        fprintf(stderr, "Header walk failed\n");
        unmap_file(&mf);
        return 2;
    }

    printf("GGUF file : %s\n", path);
    printf("version=%u  tensors=%llu  kv=%llu  file_bytes=%zu\n\n",
           version, (unsigned long long)tensor_count, (unsigned long long)kv_count,
           mf.file_size);

    for (uint64_t k = 0; k < kv_count; ++k) {
        char* key = NULL;
        if (!r_string(&r, &key)) {
            fprintf(stderr, "key walk failed at kv#%llu\n", (unsigned long long)k);
            free(key);
            unmap_file(&mf);
            return 3;
        }
        uint32_t vt;
        if (!r_u32(&r, &vt)) {
            fprintf(stderr, "valtype walk failed for '%s'\n", key);
            free(key);
            unmap_file(&mf);
            return 3;
        }

        printf("kv[%3llu] %-34s (%s)", (unsigned long long)k, key, gguf_vt_name(vt));

        bool ok = false;
        switch (vt) {
            case GGUF_VT_U8: case GGUF_VT_I8: case GGUF_VT_U16: case GGUF_VT_I16:
            case GGUF_VT_U32: case GGUF_VT_I32: case GGUF_VT_F32:
            case GGUF_VT_BOOL: case GGUF_VT_U64: case GGUF_VT_I64: case GGUF_VT_F64:
                ok = print_scalar(&r, vt);
                break;
            case GGUF_VT_STRING: {
                char* s = NULL;
                if (r_string(&r, &s)) {
                    printf(": \"%s\"\n", s);
                    free(s);
                    ok = true;
                }
                break;
            }
            case GGUF_VT_ARRAY: {
                uint32_t et;
                uint64_t ac;
                if (r_u32(&r, &et) && r_u64(&r, &ac)) {
                    ok = print_array(&r, et, ac, 0);
                }
                break;
            }
            default:
                break;
        }

        if (!ok) {
            fprintf(stderr, "  -> failed to decode value of '%s' (type %s)\n",
                    key, gguf_vt_name(vt));
            free(key);
            unmap_file(&mf);
            return 4;
        }
        free(key);
    }

    /* Tensor entry dictionary immediately follows the KV dictionary. */
    printf("\nfirst tensor entries (header only, no data touched):\n");
    for (uint64_t t = 0; t < tensor_count; ++t) {
        if (t == 8) {
            printf("  ... (%llu more entries; tensor data region then starts)\n",
                   (unsigned long long)(tensor_count - 8));
            break;
        }
        char* name;
        if (!r_string(&r, &name)) {
            fprintf(stderr, "tensor name walk failed at #%llu\n", (unsigned long long)t);
            free(name);
            unmap_file(&mf);
            return 5;
        }
        uint32_t ndims;
        if (!r_u32(&r, &ndims) || ndims == 0 || ndims > 4) {
            fprintf(stderr, "tensor '%s' invalid ndims=%u\n", name, ndims);
            free(name);
            unmap_file(&mf);
            return 5;
        }
        uint64_t ne[4] = { 1, 1, 1, 1 };
        uint64_t offs = 0;
        for (uint32_t d = 0; d < ndims; ++d) {
            if (!r_u64(&r, &ne[d])) {
                fprintf(stderr, "tensor '%s' dim walk failed\n", name);
                free(name);
                unmap_file(&mf);
                return 5;
            }
        }
        uint32_t ttype;
        if (!r_u32(&r, &ttype) || !r_u64(&r, &offs)) {
            fprintf(stderr, "tensor '%s' type/offset walk failed\n", name);
            free(name);
            unmap_file(&mf);
            return 5;
        }
        printf("  #%llu %-36s type=%-5s ne=[%llu %llu %llu %llu] data_offset=%llu\n",
               (unsigned long long)t, name, gguf_tensor_type_name(ttype),
               (unsigned long long)ne[0], (unsigned long long)ne[1],
               (unsigned long long)ne[2], (unsigned long long)ne[3],
               (unsigned long long)offs);
        free(name);
    }

    unmap_file(&mf);
    return 0;
}
