#include "zendnn_matmul_bench.h"
#include "ggml_utils.h"
#include "routing_utils.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// #define GGML_COMMON_DECL_CPP
// #include "ggml-common.h"
// #undef GGML_COMMON_DECL_CPP

#include "zendnnl.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstring>
#include <cmath>
#include <omp.h>



#define QK8_0  32
#define QK4_0  32

/**
 * @file ggml_weight_unpacking.c
 * @brief GGML weight unpacking for ZenDNN kernels
 */

typedef struct { uint16_t d; int8_t  qs[32];  } block_q8_0;
typedef struct { uint16_t d; uint8_t qs[16];  } block_q4_0;
typedef struct { uint16_t d[8]; uint8_t qs[128]; } block_q4_0x8;

static inline uint32_t fp32_to_bits(float f) {
    uint32_t bits; memcpy(&bits, &f, sizeof(f)); return bits;
}

static inline float fp32_from_bits(uint32_t bits) {
    float f; memcpy(&f, &bits, sizeof(bits)); return f;
}

/* fp16 -> fp32, from llama.cpp */
static float fp16_to_fp32(uint16_t h) {
    const uint32_t w             = (uint32_t)h << 16;
    const uint32_t sign          = w & UINT32_C(0x80000000);
    const uint32_t two_w         = w + w;
    const uint32_t exp_offset    = UINT32_C(0xE0) << 23;
    const float    exp_scale     = fp32_from_bits(UINT32_C(0x7800000));
    const float    normalized    = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask    = UINT32_C(126) << 23;
    const float    denormalized  = fp32_from_bits((two_w >> 17) | magic_mask) - 0.5f;
    const uint32_t result = sign | (two_w < (UINT32_C(1) << 27) ? fp32_to_bits(denormalized) : fp32_to_bits(normalized));
    return fp32_from_bits(result);
}

/* fp32 -> bf16, from llama.cpp */
static uint16_t fp32_to_bf16(float s) {
    union { float f; uint32_t i; } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) return (u.i >> 16) | 64; /* NaN: force quiet */
    return (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
}

/* write one scale (fp32 or bf16) at index idx */
static void write_scale(void* scale_buffer, bool use_bf16, int64_t idx, float val) {
    if (use_bf16) ((uint16_t*)scale_buffer)[idx] = fp32_to_bf16(val);
    else          ((float*)scale_buffer)[idx]    = val;
}



#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

// Helper to extract top 16 bits of a float to make a bf16 natively
static inline uint16_t f32_to_bf16_truncate(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}


int ggml_unpack_weight_buffer(
    const void* weight_data,
    int         ggml_type,
    bool        is_superblock,
    bool        use_bf16_scales,
    bool        use_unsigned_q4,
    int64_t     M,
    int64_t     K,
    int8_t* weight_buffer,
    void* scale_buffer)
{
    if (!weight_data || !weight_buffer || !scale_buffer) return -1;
    if (M <= 0 || K <= 0 || K % 32 != 0) return -1;

    int64_t num_groups = K / 32;

    /* Q8_0: 4-row tiled implementation for interleaved {scale, weights} */
    if (ggml_type == 8) {
        const block_q8_0* blocks = (const block_q8_0*)weight_data;

        // 1. BF16 Scales Path
        if (use_bf16_scales) {
            uint16_t* out_scales = (uint16_t*)scale_buffer;
            #pragma omp parallel for schedule(static)
            for (int64_t r4 = 0; r4 < M / 4; r4++) {
                for (int64_t g = 0; g < num_groups; g++) {
                    int64_t row0 = r4 * 4 + 0;
                    int64_t row1 = r4 * 4 + 1;
                    int64_t row2 = r4 * 4 + 2;
                    int64_t row3 = r4 * 4 + 3;

                    const block_q8_0* b0 = &blocks[row0 * num_groups + g];
                    const block_q8_0* b1 = &blocks[row1 * num_groups + g];
                    const block_q8_0* b2 = &blocks[row2 * num_groups + g];
                    const block_q8_0* b3 = &blocks[row3 * num_groups + g];

                    out_scales[g * M + row0] = f32_to_bf16_truncate(fp16_to_fp32(b0->d));
                    out_scales[g * M + row1] = f32_to_bf16_truncate(fp16_to_fp32(b1->d));
                    out_scales[g * M + row2] = f32_to_bf16_truncate(fp16_to_fp32(b2->d));
                    out_scales[g * M + row3] = f32_to_bf16_truncate(fp16_to_fp32(b3->d));

                    memcpy(&weight_buffer[row0 * K + g * 32], b0->qs, 32);
                    memcpy(&weight_buffer[row1 * K + g * 32], b1->qs, 32);
                    memcpy(&weight_buffer[row2 * K + g * 32], b2->qs, 32);
                    memcpy(&weight_buffer[row3 * K + g * 32], b3->qs, 32);
                }
            }
            // Cleanup loop for rows if M % 4 != 0
            for (int64_t row = (M / 4) * 4; row < M; row++) {
                for (int64_t g = 0; g < num_groups; g++) {
                    const block_q8_0* b = &blocks[row * num_groups + g];
                    out_scales[g * M + row] = f32_to_bf16_truncate(fp16_to_fp32(b->d));
                    memcpy(&weight_buffer[row * K + g * 32], b->qs, 32);
                }
            }
        } 
        // 2. F32 Scales Path
        else {
            float* out_scales = (float*)scale_buffer;
            #pragma omp parallel for schedule(static)
            for (int64_t r4 = 0; r4 < M / 4; r4++) {
                for (int64_t g = 0; g < num_groups; g++) {
                    int64_t row0 = r4 * 4 + 0;
                    int64_t row1 = r4 * 4 + 1;
                    int64_t row2 = r4 * 4 + 2;
                    int64_t row3 = r4 * 4 + 3;

                    const block_q8_0* b0 = &blocks[row0 * num_groups + g];
                    const block_q8_0* b1 = &blocks[row1 * num_groups + g];
                    const block_q8_0* b2 = &blocks[row2 * num_groups + g];
                    const block_q8_0* b3 = &blocks[row3 * num_groups + g];

                    out_scales[g * M + row0] = fp16_to_fp32(b0->d);
                    out_scales[g * M + row1] = fp16_to_fp32(b1->d);
                    out_scales[g * M + row2] = fp16_to_fp32(b2->d);
                    out_scales[g * M + row3] = fp16_to_fp32(b3->d);

                    memcpy(&weight_buffer[row0 * K + g * 32], b0->qs, 32);
                    memcpy(&weight_buffer[row1 * K + g * 32], b1->qs, 32);
                    memcpy(&weight_buffer[row2 * K + g * 32], b2->qs, 32);
                    memcpy(&weight_buffer[row3 * K + g * 32], b3->qs, 32);
                }
            }
            // Cleanup loop for rows if M % 4 != 0
            for (int64_t row = (M / 4) * 4; row < M; row++) {
                for (int64_t g = 0; g < num_groups; g++) {
                    const block_q8_0* b = &blocks[row * num_groups + g];
                    out_scales[g * M + row] = fp16_to_fp32(b->d);
                    memcpy(&weight_buffer[row * K + g * 32], b->qs, 32);
                }
            }
        }
        return 0;
    }
    
    /* Q4_0 superblock: 8 rows interleaved, nibbles pre-signed via XOR 0x88 by make_block_q4_0x8 */
    if (ggml_type == 2 && is_superblock) {
        if (M % 8 != 0) return -1;
        const block_q4_0x8* blocks = (const block_q4_0x8*)weight_data;
        
        #pragma omp parallel for schedule(static)
        for (int64_t r8 = 0; r8 < M / 8; r8++) {
            for (int64_t g = 0; g < num_groups; g++) {
                const block_q4_0x8* b = &blocks[r8 * num_groups + g];
                for (int ri = 0; ri < 8; ri++) {
                    int64_t row = r8 * 8 + ri;
                    write_scale(scale_buffer, use_bf16_scales, g * M + row, fp16_to_fp32(b->d[ri]));

                    /* un-swizzle AVX interleave: low 8 bytes then high 8 bytes per row */
                    uint8_t src[16];
                    for (int i = 0; i < 8; i++) {
                        src[i]     = b->qs[ri * 8 + i];
                        src[i + 8] = b->qs[64 + ri * 8 + i];
                    }

                    int8_t* dst = &weight_buffer[(row * K + g * 32) / 2];
                    for (int i = 0; i < 8; i++) {
                        uint8_t lo0 = src[2*i]     & 0x0F;
                        uint8_t lo1 = src[2*i + 1] & 0x0F;
                        uint8_t hi0 = src[2*i]     >> 4;
                        uint8_t hi1 = src[2*i + 1] >> 4;

                        if (use_unsigned_q4) { /* reverse GGML's XOR 0x88 to restore unsigned nibbles */
                            lo0 ^= 8; lo1 ^= 8; hi0 ^= 8; hi1 ^= 8;
                        }

                        dst[i]     = (int8_t)(lo0 | (lo1 << 4));
                        dst[8 + i] = (int8_t)(hi0 | (hi1 << 4));
                    }
                }
            }
        }
        return 0;
    }

    /* Q4_0 regular: unsigned nibbles 0-15, subtract 8 to convert to signed S4 (8-Row Tiled) */
    if (ggml_type == 2) {
        const block_q4_0* blocks = (const block_q4_0*)weight_data;
        
        #pragma omp parallel for schedule(static)
        for (int64_t r8 = 0; r8 < M / 8; r8++) {
            for (int64_t g = 0; g < num_groups; g++) {
                
                for (int ri = 0; ri < 8; ri++) {
                    int64_t row = r8 * 8 + ri;
                    const block_q4_0* b = &blocks[row * num_groups + g];
                    write_scale(scale_buffer, use_bf16_scales, g * M + row, fp16_to_fp32(b->d));

                    const uint8_t* src = b->qs;
                    int8_t* dst = &weight_buffer[(row * K + g * 32) / 2];
                    
                    for (int i = 0; i < 8; i++) {
                        uint8_t lo0 = src[2*i]     & 0x0F;
                        uint8_t lo1 = src[2*i + 1] & 0x0F;
                        uint8_t hi0 = src[2*i]     >> 4;
                        uint8_t hi1 = src[2*i + 1] >> 4;

                        if (!use_unsigned_q4) { /* convert unsigned nibbles to signed S4 */
                            lo0 = (lo0 - 8) & 0x0F;
                            lo1 = (lo1 - 8) & 0x0F;
                            hi0 = (hi0 - 8) & 0x0F;
                            hi1 = (hi1 - 8) & 0x0F;
                        }

                        dst[i]     = (int8_t)(lo0 | (lo1 << 4));
                        dst[8 + i] = (int8_t)(hi0 | (hi1 << 4));
                    }
                }
            }
        }
        
        // Cleanup loop for rows if M % 8 != 0
        for (int64_t row = (M / 8) * 8; row < M; row++) {
            for (int64_t g = 0; g < num_groups; g++) {
                const block_q4_0* b = &blocks[row * num_groups + g];
                write_scale(scale_buffer, use_bf16_scales, g * M + row, fp16_to_fp32(b->d));

                const uint8_t* src = b->qs;
                int8_t* dst = &weight_buffer[(row * K + g * 32) / 2];
                for (int i = 0; i < 8; i++) {
                    uint8_t lo0 = src[2*i]     & 0x0F;
                    uint8_t lo1 = src[2*i + 1] & 0x0F;
                    uint8_t hi0 = src[2*i]     >> 4;
                    uint8_t hi1 = src[2*i + 1] >> 4;

                    if (!use_unsigned_q4) { 
                        lo0 = (lo0 - 8) & 0x0F;
                        lo1 = (lo1 - 8) & 0x0F;
                        hi0 = (hi0 - 8) & 0x0F;
                        hi1 = (hi1 - 8) & 0x0F;
                    }

                    dst[i]     = (int8_t)(lo0 | (lo1 << 4));
                    dst[8 + i] = (int8_t)(hi0 | (hi1 << 4));
                }
            }
        }
        return 0;
    }

    return -1; /* unsupported type */
}

using namespace zendnnl::lowoha::matmul;

// ===========================================================================
//  Internal helpers
// ===========================================================================

static zendnnl::common::data_type_t ggml_type_to_zendnn(ggml_type t)
{
    switch (t) {
        case GGML_TYPE_F32:  return zendnnl::common::data_type_t::f32;
        case GGML_TYPE_BF16: return zendnnl::common::data_type_t::bf16;
        default:
            fprintf(stderr, "error: unsupported ggml_type %d for ZenDNN\n", t);
            exit(1);
    }
}

static uint32_t xorshift32(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

template <typename T>
static void fill_buffer(T* data, size_t n, uint32_t seed)
{
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t r   = xorshift32(state);
        float    val = static_cast<float>(static_cast<int32_t>(r))
                     / static_cast<float>(INT32_MAX);
        data[i] = static_cast<T>(val);
    }
}

static void fill_buffer_bf16(void* data, size_t n, uint32_t seed)
{
    uint16_t* ptr   = reinterpret_cast<uint16_t*>(data);
    uint32_t  state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t r   = xorshift32(state);
        float    val = static_cast<float>(static_cast<int32_t>(r))
                     / static_cast<float>(INT32_MAX);
        uint32_t u   = *reinterpret_cast<uint32_t*>(&val);
        if ((u & 0x7fffffff) > 0x7f800000)
            ptr[i] = (u >> 16) | 64;
        else
            ptr[i] = static_cast<uint16_t>(
                         (u + (0x7fff + ((u >> 16) & 1))) >> 16);
    }
}

// ===========================================================================
//  generate_native_q4_0x8_superblocks
//
//  Uses GGML's internal CPU backend repack buffer to generate authentic
//  Q4_0x8 superblocks for benchmark testing.
// ===========================================================================
static std::vector<block_q4_0x8> generate_native_q4_0x8_superblocks(int64_t m, int64_t k, uint32_t seed) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "error: failed to init CPU backend for repacking\n");
        exit(1);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    bool using_repack = false;

    // Probe for the repack extra buffer
    ggml_backend_dev_t cpu_dev = ggml_backend_get_device(backend);
    if (cpu_dev) {
        ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        if (cpu_reg) {
            typedef ggml_backend_buffer_type_t * (*get_extra_bufts_t)(ggml_backend_dev_t);
            auto get_extra_bufts = (get_extra_bufts_t)ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

            if (get_extra_bufts) {
                ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts(cpu_dev);
                if (extra_bufts && *extra_bufts) {
                    buft = *extra_bufts; 
                    using_repack = true;
                }
            }
        }
    }

    if (!using_repack) {
        fprintf(stderr, "error: GGML CPU backend does not support Q4_0x8 repacking on this machine architecture.\n");
        exit(1);
    }

    struct ggml_init_params params = {};
    params.mem_size   = ggml_tensor_overhead() * 2 + 1024; 
    params.mem_buffer = nullptr;
    params.no_alloc   = true;
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, m);
    
    // Allocate the tensor on the repack buffer
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        fprintf(stderr, "error: failed to allocate repack buffer\n");
        exit(1);
    }

    // Trigger internal GGML quantize + repack
    fill_tensor_deterministic(a, seed, using_repack);

    // Extract the physical Q4_0x8 superblock layout out of the repacked tensor payload
    const int64_t num_superblocks = (m * k) / 256;
    std::vector<block_q4_0x8> repacked_data(num_superblocks);
    memcpy(repacked_data.data(), a->data, num_superblocks * sizeof(block_q4_0x8));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return repacked_data;
}

// ===========================================================================
//  custom_matmul_q8_0
// ===========================================================================
static bool custom_matmul_q8_0(
    int               n_threads,
    int64_t           m,
    int64_t           n,
    int64_t           k,
    const block_q8_0* weights,
    const float* src_fp32,
    float* C,
    int64_t           ldc)
{
    const int64_t BS    = QK8_0; 
    const int64_t n_blk = k / BS;

    if (k % BS != 0) return false;

    std::vector<block_q8_0> src_q8(n * n_blk);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src_fp32, src_q8.data(), 0, n, k, nullptr);

    std::vector<int8_t> src_s8(n * k);
    std::vector<float>  src_scales(n_blk * n);

    if (ggml_unpack_weight_buffer(src_q8.data(), GGML_TYPE_Q8_0, false, false, false, n, k, src_s8.data(), src_scales.data()) != 0) 
        return false;

    std::vector<int8_t> wei_s8(m * k);
    std::vector<float>  wei_scales(n_blk * m);

    if (ggml_unpack_weight_buffer(weights, GGML_TYPE_Q8_0, false, false, false, m, k, wei_s8.data(), wei_scales.data()) != 0) 
        return false;

    const int8_t* __restrict__ pSrc   = src_s8.data();
    const int8_t* __restrict__ pWei   = wei_s8.data();
    const float* __restrict__ pSrcSc = src_scales.data();
    const float* __restrict__ pWeiSc = wei_scales.data();

    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            float acc = 0.0f;
            for (int64_t b = 0; b < n_blk; ++b) {
                const int8_t* s = pSrc + i * k + b * BS;
                const int8_t* w = pWei + j * k + b * BS;

                int32_t dot = 0;
                for (int64_t ki = 0; ki < BS; ++ki)
                    dot += (int32_t)s[ki] * (int32_t)w[ki];

                acc += pSrcSc[b * n + i] * pWeiSc[b * m + j] * (float)dot;
            }
            C[i * ldc + j] = acc;
        }
    }
    return true;
}

// ===========================================================================
//  custom_matmul_q4_0
// ===========================================================================
static bool custom_matmul_q4_0(
    int               n_threads,
    int64_t           m,
    int64_t           n,
    int64_t           k,
    const block_q4_0* weights,
    const float* src_fp32,
    float* C,
    int64_t           ldc)
{
    const int64_t BS    = QK4_0;
    const int64_t n_grp = k / BS;

    if (k % BS != 0) return false;

    const int64_t k_s4 = k / 2;
    std::vector<int8_t> wei_s4(m * k_s4);
    std::vector<float>  wei_scales(n_grp * m);

    if (ggml_unpack_weight_buffer(weights, GGML_TYPE_Q4_0, false, false, false, m, k, wei_s4.data(), wei_scales.data()) != 0) 
        return false;

    std::vector<block_q8_0> src_q8(n * n_grp);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src_fp32, src_q8.data(), 0, n, k, nullptr);

    std::vector<int8_t> src_s8(n * k);
    std::vector<float>  src_scales(n_grp * n);

    if (ggml_unpack_weight_buffer(src_q8.data(), GGML_TYPE_Q8_0, false, false, false, n, k, src_s8.data(), src_scales.data()) != 0) 
        return false;

    const int8_t* __restrict__ pSrc   = src_s8.data();
    const int8_t* __restrict__ pWei   = wei_s4.data();
    const float* __restrict__ pSrcSc = src_scales.data();
    const float* __restrict__ pWeiSc = wei_scales.data();

    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            float acc = 0.0f;
            for (int64_t g = 0; g < n_grp; ++g) {
                const int8_t* s  = pSrc + i * k    + g * BS;
                const int8_t* wp = pWei + j * k_s4 + g * (BS / 2);

                int32_t dot = 0;
                for (int64_t bi = 0; bi < BS / 2; ++bi) {
                    int8_t p = wp[bi];
                    const int32_t w0 = (int32_t)((int8_t)(p << 4) >> 4);
                    const int32_t w1 = (int32_t)(p >> 4);
                    dot += (int32_t)s[2*bi    ] * w0;
                    dot += (int32_t)s[2*bi + 1] * w1;
                }
                acc += pSrcSc[g * n + i] * pWeiSc[g * m + j] * (float)dot;
            }
            C[i * ldc + j] = acc;
        }
    }
    return true;
}

// ===========================================================================
//  custom_matmul_q4_0x8
// ===========================================================================
static bool custom_matmul_q4_0x8(
    int                 n_threads,
    int64_t             m,
    int64_t             n,
    int64_t             k,
    const block_q4_0x8* weights,
    const float* src_fp32,
    float* C,
    int64_t             ldc)
{
    const int64_t BS    = QK4_0; 
    const int64_t n_grp = k / BS;

    if (k % 256 != 0) {
        fprintf(stderr, "error [Q4_0x8 custom]: K=%ld not divisible by 256\n", k);
        return false;
    }

    const int64_t k_s4 = k / 2;
    std::vector<int8_t> wei_s4(m * k_s4);
    std::vector<float>  wei_scales(n_grp * m);

    if (ggml_unpack_weight_buffer(weights, GGML_TYPE_Q4_0, true, false, false, m, k, wei_s4.data(), wei_scales.data()) != 0) {
        fprintf(stderr, "error [Q4_0x8 custom]: ggml_unpack_weight_buffer failed for weights\n");
        return false;
    }

    std::vector<block_q8_0> src_q8(n * n_grp);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src_fp32, src_q8.data(), 0, n, k, nullptr);

    std::vector<int8_t> src_s8(n * k);
    std::vector<float>  src_scales(n_grp * n);

    if (ggml_unpack_weight_buffer(src_q8.data(), GGML_TYPE_Q8_0, false, false, false, n, k, src_s8.data(), src_scales.data()) != 0) {
        fprintf(stderr, "error [Q4_0x8 custom]: ggml_unpack_weight_buffer failed for src\n");
        return false;
    }

    const int8_t* __restrict__ pSrc   = src_s8.data();
    const int8_t* __restrict__ pWei   = wei_s4.data();
    const float* __restrict__ pSrcSc = src_scales.data();
    const float* __restrict__ pWeiSc = wei_scales.data();

    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            float acc = 0.0f;
            for (int64_t g = 0; g < n_grp; ++g) {
                const int8_t* s  = pSrc + i * k    + g * BS;
                const int8_t* wp = pWei + j * k_s4 + g * (BS / 2);

                int32_t dot = 0;
                for (int64_t bi = 0; bi < BS / 2; ++bi) {
                    int8_t p = wp[bi];
                    const int32_t w0 = (int32_t)((int8_t)(p << 4) >> 4); 
                    const int32_t w1 = (int32_t)(p >> 4);               
                    dot += (int32_t)s[2*bi    ] * w0;
                    dot += (int32_t)s[2*bi + 1] * w1;
                }
                acc += pSrcSc[g * n + i] * pWeiSc[g * m + j] * (float)dot;
            }
            C[i * ldc + j] = acc;
        }
    }
    return true;
}


static bool ggml_zendnn_matmul_q8_0_f32s8(
    int               n_threads,
    int64_t           m,          // output features (weight rows)
    int64_t           n,          // tokens (src rows)
    int64_t           k,
    const block_q8_0* weights,
    const float*      src_fp32,
    float*            C,
    int64_t           ldc)
{
    using namespace zendnnl::lowoha::matmul;

    const int64_t group_size = QK8_0;       // 32
    const int64_t num_groups = k / group_size;

    if (k % group_size != 0) {
        fprintf(stderr,
                "error [Q8_0 ZenDNN S8S8]: K=%ld not divisible by %ld\n",
                k, group_size);
        return false;
    }

    matmul_data_types dtypes;
    dtypes.src     = data_type_t::f32;
    dtypes.wei     = data_type_t::s8;
    dtypes.dst     = data_type_t::f32;
    dtypes.bias    = data_type_t::none;
    dtypes.compute = data_type_t::s8;   // symmetric quantisation path

    matmul_params params;
    params.dtypes       = dtypes;
    params.dynamic_quant = true;
    params.num_threads = n_threads;

    params.quant_params.src_scale.buff = nullptr;
    params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::bf16;
    params.quant_params.src_scale.dims = {static_cast<int>(n), static_cast<int>(num_groups)};

    params.packing.pack_format_b = 1;
    

    matmul_batch_params_t batch_params; 
    
    status_t status = matmul_direct(
        'r',                            // row-major layout
        false,                          // transA = false  (src  is n×k)
        true,                           // transB = true   (wei  is m×k, used as k×m)
        static_cast<int>(n),            // M — src rows
        static_cast<int>(m),            // N — weight rows / output features
        static_cast<int>(k),            // K — inner dimension
        1.0f,                           // alpha
        src_fp32,                       // A  (n × k)
        static_cast<int>(k),            // lda
        weights,                  // B  (m × k, transposed to k × m)
        static_cast<int>(k),            // ldb  (stride along K before transpose)
        nullptr,                        // bias
        0.0f,                           // beta
        C,                              // output (n × m)
        static_cast<int>(ldc),          // ldc
        true,                           // const weight
        batch_params,
        params
    );

    return status == status_t::success;
}


static bool ggml_zendnn_matmul_q8_0_f32s8_static(
    int               n_threads,
    int64_t           m,          // output features (weight rows)
    int64_t           n,          // tokens (src rows)
    int64_t           k,
    const block_q8_0* weights,
    const float* src_fp32,
    float* C,
    int64_t           ldc)
{
    using namespace zendnnl::lowoha::matmul;

    const int64_t group_size = QK8_0;       // 32
    const int64_t num_groups = k / group_size;

    if (k % group_size != 0) {
        fprintf(stderr,
                "error [Q8_0 ZenDNN S8S8 API]: K=%ld not divisible by %ld\n",
                k, group_size);
        return false;
    }

    // ------------------------------------------------------------------ //
    //  1. Unpack Weights (Q8_0 -> S8 + scales)                           //
    //     wei_s8    : (m × k)                                            //
    //     wei_scales: (num_groups × m) layout [g * m + out_feat]         //
    // ------------------------------------------------------------------ //
    std::vector<int8_t> wei_s8(m * k);
    std::vector<float>  wei_scales(num_groups * m);

    if (ggml_unpack_weight_buffer(weights, GGML_TYPE_Q8_0, false, false, false, 
                                  m, k, wei_s8.data(), wei_scales.data()) != 0) 
    {
        fprintf(stderr, "error [Q8_0 ZenDNN S8S8 API]: extract failed for weights\n");
        return false;
    }

    // ------------------------------------------------------------------ //
    //  2. Quantize & Unpack Source (F32 -> Q8_0 -> S8 + scales)          //
    // ------------------------------------------------------------------ //
    std::vector<block_q8_0> src_q8(n * num_groups);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src_fp32, src_q8.data(), 0, n, k, nullptr);

    std::vector<int8_t> src_s8(n * k);
    
    // ggml_unpack_weight_buffer naturally outputs group-major: [num_groups × n]
    std::vector<float> src_scales_unpacked(num_groups * n); 

    if (ggml_unpack_weight_buffer(src_q8.data(), GGML_TYPE_Q8_0, false, false, false, 
                                  n, k, src_s8.data(), src_scales_unpacked.data()) != 0) 
    {
        fprintf(stderr, "error [Q8_0 ZenDNN S8S8 API]: extract failed for src\n");
        return false;
    }

    // ZenDNN's API for src scales expects dims = {n, num_groups}. 
    // We must transpose the unpacked scales from [num_groups, n] -> [n, num_groups].
    std::vector<float> src_scales(n * num_groups);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t g = 0; g < num_groups; ++g) {
            src_scales[i * num_groups + g] = src_scales_unpacked[g * n + i];
        }
    }

    // ------------------------------------------------------------------ //
    //  3. Build matmul descriptor                                        //
    // ------------------------------------------------------------------ //
    matmul_data_types dtypes;
    dtypes.src     = data_type_t::s8;   // <--- explicitly S8
    dtypes.wei     = data_type_t::s8;
    dtypes.dst     = data_type_t::f32;
    dtypes.bias    = data_type_t::none;
    dtypes.compute = data_type_t::s8;   // symmetric quantisation path

    matmul_params params;
    params.dtypes        = dtypes;
    params.dynamic_quant = false;       // <--- disabled, we are feeding scales manually

    // Source scale: provided manually, dims {n, num_groups}
    params.quant_params.src_scale.buff = src_scales.data();
    params.quant_params.src_scale.dt   = data_type_t::f32;
    params.quant_params.src_scale.dims = {static_cast<int>(n), static_cast<int>(num_groups)};

    // Weight scale: provided manually, dims {num_groups, m}
    params.quant_params.wei_scale.buff = wei_scales.data();
    params.quant_params.wei_scale.dt   = data_type_t::f32;
    params.quant_params.wei_scale.dims = {static_cast<int>(num_groups), static_cast<int>(m)};

    matmul_batch_params_t batch_params;  // default / unused

    // ------------------------------------------------------------------ //
    //  4. Execute                                                        //
    // ------------------------------------------------------------------ //
    status_t status = matmul_direct(
        'r',                            // row-major layout
        false,                          // transA = false  (src is n×k)
        true,                           // transB = true   (wei is m×k, used as k×m)
        static_cast<int>(n),            // M — src rows
        static_cast<int>(m),            // N — weight rows / output features
        static_cast<int>(k),            // K — inner dimension
        1.0f,                           // alpha
        src_s8.data(),                  // A (n × k) 
        static_cast<int>(k),            // lda
        wei_s8.data(),                  // B (m × k, transposed to k × m)
        static_cast<int>(k),            // ldb (stride along K before transpose)
        nullptr,                        // bias
        0.0f,                           // beta
        C,                              // output (n × m)
        static_cast<int>(ldc),          // ldc
        true,                          // is_weights_const
        batch_params,
        params
    );

    return status == status_t::success;
}



// ===========================================================================
//  ggml_zendnn_matmul_q4_0_woq_bf16
// ===========================================================================
static bool ggml_zendnn_matmul_q4_0_woq_bf16(
    int               n_threads,
    int64_t           m,
    int64_t           n,
    int64_t           k,
    const block_q4_0* weights,
    const float* src_fp32,
    float* C,
    int64_t           ldc)
{
    // printf("q4 dynamic quant started\n");
    const int64_t group_size = 32;
    const int64_t num_groups = k / group_size;

    // std::vector<ggml_bf16_t> src_bf16(n * k);
    // ggml_fp32_to_bf16_row(src_fp32, src_bf16.data(),n*k);

    std::vector<int8_t> packed_s4((m * k ) / 2);
    std::vector<float>  scales(num_groups * m);

    
    // --- Start Unpack Timer ---
    auto unpack_start = std::chrono::steady_clock::now();
    if (ggml_unpack_weight_buffer(weights, GGML_TYPE_Q4_0, false, false, false, m, k, packed_s4.data(), scales.data()) != 0) {
        fprintf(stderr, "error [Q4_0 ZenDNN]: ggml_unpack_weight_buffer failed\n");
        return false;
    }

    
    // --- End Unpack Timer ---
    auto unpack_end = std::chrono::steady_clock::now();
    double unpack_us = std::chrono::duration<double, std::micro>(unpack_end - unpack_start).count();
    
    // Optional: Only print if it exceeds a certain threshold to avoid spamming stdout
    // if (unpack_ms > 1.0) { 
    printf("[Profile] Q4_0 Unpacking took: %.3f us\n", unpack_us);
    // }




    std::vector<ggml_bf16_t> src_bf16(n*k);
    ggml_fp32_to_bf16_row(src_fp32,src_bf16.data(),n*k);

    matmul_data_types dtypes;
    dtypes.src     = zendnnl::common::data_type_t::bf16;
    // dtypes.src     = zendnnl::common::data_type_t::f32;
    dtypes.wei     = zendnnl::common::data_type_t::s4;
    dtypes.dst     = zendnnl::common::data_type_t::f32;
    dtypes.bias    = zendnnl::common::data_type_t::none;
    dtypes.compute = zendnnl::common::data_type_t::s8;

    matmul_params params;
    params.dtypes      = dtypes;
    params.num_threads = n_threads;
    // params.dynamic_quant = true;

    params.quant_params.src_scale.buff = nullptr;
    params.quant_params.src_scale.dt   = zendnnl::common::data_type_t::f32;
    params.quant_params.src_scale.dims = {n, num_groups};

    params.quant_params.wei_scale.buff = scales.data();
    params.quant_params.wei_scale.dt   = zendnnl::common::data_type_t::f32;
    params.quant_params.wei_scale.dims = {num_groups, m};
    params.quant_params.wei_zp.buff    = nullptr;
    params.quant_params.wei_zp.dims    = {};

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    status_t status = matmul_direct(
        'r', false, true, 
        n, m, k, 
        1.0f,
        src_bf16.data(), k, 
        packed_s4.data(), k, 
        nullptr, 0.0f,
        C, ldc, 
        true, batch_params, params);

    
    // printf("q4 dynamic quant ended\n");
    return (status == status_t::success);
}

// ===========================================================================
//  ggml_zendnn_matmul_q4_0x8_woq_bf16
// ===========================================================================
static bool ggml_zendnn_matmul_q4_0x8_woq_bf16(
    int                 n_threads,
    int64_t             m,          
    int64_t             n,          
    int64_t             k,
    const block_q4_0x8* weights,
    const float* src_fp32,
    float* C,
    int64_t             ldc)        
{
    const int64_t group_size = 32;
    const int64_t num_groups = k / group_size;

    if (k % 256 != 0) return false;

    std::vector<ggml_bf16_t> src_bf16(n * k);
    ggml_fp32_to_bf16_row(src_fp32, src_bf16.data(),n*k);

    std::vector<int8_t> packed_s4((m * k) / 2);
    std::vector<float>  scales(num_groups * m);

    
    // --- Start Unpack Timer ---
    auto unpack_start = std::chrono::steady_clock::now();
    if (ggml_unpack_weight_buffer(weights, GGML_TYPE_Q4_0, true, false, false, m, k, packed_s4.data(), scales.data()) != 0) {
        fprintf(stderr, "error [Q4_0x8 ZenDNN]: ggml_unpack_weight_buffer failed\n");
        return false;
    }
        
    // --- End Unpack Timer ---
    auto unpack_end = std::chrono::steady_clock::now();
    double unpack_us = std::chrono::duration<double, std::micro>(unpack_end - unpack_start).count();
    
    // Optional: Only print if it exceeds a certain threshold to avoid spamming stdout
    // if (unpack_ms > 1.0) { 
    printf("[Profile] Q4_0 Superblock Unpacking took: %.3f us\n", unpack_us);
    // }


    matmul_data_types dtypes;
    dtypes.src     = zendnnl::common::data_type_t::bf16;
    dtypes.wei     = zendnnl::common::data_type_t::s4;
    dtypes.dst     = zendnnl::common::data_type_t::f32;
    dtypes.bias    = zendnnl::common::data_type_t::none;
    dtypes.compute = zendnnl::common::data_type_t::f32;

    matmul_params params;
    params.dtypes      = dtypes;
    params.num_threads = n_threads;

    params.quant_params.wei_scale.buff = scales.data();
    params.quant_params.wei_scale.dt   = zendnnl::common::data_type_t::f32;
    params.quant_params.wei_scale.dims = {num_groups, m};
    params.quant_params.wei_zp.buff    = nullptr;
    params.quant_params.wei_zp.dims    = {};

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    status_t status = matmul_direct(
        'r', false, true, n, m, k, 1.0f,
        src_bf16.data(), k, packed_s4.data(), k, nullptr, 0.0f,
        C, ldc, true, batch_params, params);

    return (status == status_t::success);
}

// ===========================================================================
//  bench_matmul_zendnn
// ===========================================================================
BenchResult bench_matmul_zendnn(const OpDesc& desc)
{
    const int64_t output_features = desc.m; 
    const int64_t tokens          = desc.n; 
    const int64_t K               = desc.k;
    const ggml_type src_ggml      = desc.src_dtype;
    const ggml_type wei_ggml      = desc.wei_dtype;

    auto t_ctx_start = std::chrono::steady_clock::now();

    // =========================================================================
    //  Q8_0 — ZenDNN S8×S8 kernel  (+ custom OMP kernel for verification)
    // =========================================================================
    if (wei_ggml == GGML_TYPE_Q8_0) {
        if (src_ggml != GGML_TYPE_F32) {
            fprintf(stderr, "error: Q8_0 requires F32 source\n"); exit(1);
        }

        const int64_t BS         = QK8_0;
        const int64_t num_blocks = K / BS;

        if (K % BS != 0) {
            fprintf(stderr, "error: K=%ld not divisible by %ld for Q8_0\n",
                    K, BS);
            exit(1);
        }

        printf("Q8_0 ZenDNN f32×S8 (dynamic quant)\n");

        std::vector<block_q8_0> weights_q8(output_features * num_blocks);
        std::vector<float>      input_f32 (tokens * K);

        // Two output buffers — custom OMP and ZenDNN.
        // GGML reference is computed independently by bench_matmul_ggml.
        std::vector<float> output_custom(tokens * output_features, 0.0f);
        std::vector<float> output_zendnn(tokens * output_features, 0.0f);

        // ── context creation timer ────────────────────────────────────────
        auto t_ctx_end = std::chrono::steady_clock::now();
        double ctx_ms  = std::chrono::duration<double, std::milli>(
                             t_ctx_end - t_ctx_start).count();

        // ── operator setup ────────────────────────────────────────────────
        // Use the SAME seed as bench_matmul_ggml so weights and src are
        // identical across both benchmark calls.
        auto t_op_start = std::chrono::steady_clock::now();
        {
            std::vector<float> tmp_wei(output_features * K);
            fill_buffer(tmp_wei.data(), output_features * K, desc.data_seed);
            ggml_quantize_chunk(GGML_TYPE_Q8_0,
                                tmp_wei.data(), weights_q8.data(),
                                0, output_features, K, nullptr);
        }
        fill_buffer(input_f32.data(), tokens * K, desc.data_seed + 95);
        auto t_op_end = std::chrono::steady_clock::now();
        double op_ms  = std::chrono::duration<double, std::milli>(
                            t_op_end - t_op_start).count();

        // ── 1. Custom OMP kernel (only when verifying) ────────────────────
        if (desc.verify_output) {
            bool ok = custom_matmul_q8_0(
                desc.threads, output_features, tokens, K,
                weights_q8.data(), input_f32.data(),
                output_custom.data(), output_features);
            if (!ok) {
                fprintf(stderr,
                        "error: Q8_0 custom OMP kernel failed\n");
                exit(1);
            }
        }

        // ── 2. ZenDNN warmup ──────────────────────────────────────────────
        for (int i = 0; i < desc.warmup; i++) {
            bool ok = ggml_zendnn_matmul_q8_0_f32s8(
                desc.threads, output_features, tokens, K,
                weights_q8.data(), input_f32.data(),
                output_zendnn.data(), output_features);
            if (!ok) {
                fprintf(stderr, "error: Q8_0 ZenDNN warmup failed\n");
                exit(1);
            }
        }

        // ── 3. Timed ZenDNN repeats ───────────────────────────────────────
        double min_ms = std::numeric_limits<double>::max();
        double max_ms = 0.0, sum_ms = 0.0;

        for (int i = 0; i < desc.repeats; i++) {
            auto t0 = std::chrono::steady_clock::now();
            bool ok = ggml_zendnn_matmul_q8_0_f32s8(
                desc.threads, output_features, tokens, K,
                weights_q8.data(), input_f32.data(),
                output_zendnn.data(), output_features);
            auto t1 = std::chrono::steady_clock::now();
            if (!ok) {
                fprintf(stderr, "error: Q8_0 ZenDNN matmul failed\n");
                exit(1);
            }
            double ms = std::chrono::duration<double, std::milli>(
                            t1 - t0).count();
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
            sum_ms += ms;
        }

        double avg_ms = sum_ms / desc.repeats;
        double tflops = (2.0 * output_features * tokens * K)
                        / (avg_ms * 1e-3) / 1e12;

        // ── 4. Pack result ────────────────────────────────────────────────
        BenchResult result;
        result.min_ms          = min_ms;
        result.avg_ms          = avg_ms;
        result.max_ms          = max_ms;
        result.tflops          = tflops;
        result.ctx_creation_ms = ctx_ms;
        result.op_creation_ms  = op_ms;
        result.op_execution_ms = avg_ms;
        result.other_ms        = 0.0;

        // output_data = ZenDNN result (backward compat for two-way path)
        if (desc.verify_output) {
            result.out_custom  = output_custom;
            result.out_zendnn  = output_zendnn;
            result.output_data = output_zendnn;
        }

        return result;
    }

    
    // =========================================================================
    //  Q4_0 / Q4_0x8 — ZenDNN kernel + custom kernel verify
    // =========================================================================
    if (wei_ggml == GGML_TYPE_Q4_0) {
        if (src_ggml != GGML_TYPE_F32) {
            fprintf(stderr, "error: Q4_0 requires F32 source\n"); exit(1);
        }

        // --- Q4_0x8 SUPERBLOCK PATH ---
        if (desc.is_superblock) {
            printf("Q4_0x8 Superblocks\n");
            std::vector<float> input_f32(tokens * K);
            std::vector<float> output_zendnn(tokens * output_features);
            std::vector<float> output_custom(tokens * output_features);

            auto t_ctx_end = std::chrono::steady_clock::now();
            double ctx_ms  = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

            auto t_op_start = std::chrono::steady_clock::now();

            // Native generation of Q4_0x8
            std::vector<block_q4_0x8> weights_q4_0x8 = generate_native_q4_0x8_superblocks(output_features, K, desc.data_seed);
            
            fill_buffer(input_f32.data(), tokens * K, desc.data_seed + 95);

            auto t_op_end = std::chrono::steady_clock::now();
            double op_ms  = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

            // Custom run for verification
            bool ok_custom = custom_matmul_q4_0x8(
                desc.threads, output_features, tokens, K,
                weights_q4_0x8.data(), input_f32.data(),
                output_custom.data(), output_features);
            if (!ok_custom) { fprintf(stderr, "error: Q4_0x8 custom fail\n"); exit(1); }

            // ZenDNN warmup
            for (int i = 0; i < desc.warmup; i++) {
                bool ok = ggml_zendnn_matmul_q4_0x8_woq_bf16(
                    desc.threads, output_features, tokens, K,
                    weights_q4_0x8.data(), input_f32.data(),
                    output_zendnn.data(), output_features);
                if (!ok) { fprintf(stderr, "error: Q4_0x8 ZenDNN warmup\n"); exit(1); }
            }

            double min_ms = std::numeric_limits<double>::max();
            double max_ms = 0.0, sum_ms = 0.0;

            for (int i = 0; i < desc.repeats; i++) {
                auto t0 = std::chrono::steady_clock::now();
                bool ok = ggml_zendnn_matmul_q4_0x8_woq_bf16(
                    desc.threads, output_features, tokens, K,
                    weights_q4_0x8.data(), input_f32.data(),
                    output_zendnn.data(), output_features);
                auto t1 = std::chrono::steady_clock::now();
                if (!ok) { fprintf(stderr, "error: Q4_0x8 ZenDNN matmul\n"); exit(1); }

                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                min_ms = std::min(min_ms, ms);
                max_ms = std::max(max_ms, ms);
                sum_ms += ms;
            }

            double avg_ms = sum_ms / desc.repeats;
            double tflops = (2.0 * output_features * tokens * K) / (avg_ms * 1e-3) / 1e12;

            BenchResult result;
            result.min_ms = min_ms; result.avg_ms = avg_ms;
            result.max_ms = max_ms; result.tflops = tflops;
            result.ctx_creation_ms = ctx_ms;
            result.op_creation_ms  = op_ms;
            result.op_execution_ms = avg_ms;
            result.other_ms        = 0.0;

            if (desc.verify_output) {
                result.output_data.assign(output_custom.begin(), output_custom.end());
            }
            return result;
        } 
        // --- Q4_0 STANDARD PATH ---
        else {
            
            printf("Q4_0 Regular blocks\n");
            const int64_t group_size = 32;
            const int64_t num_groups = K / group_size;

            std::vector<block_q4_0> weights_q4(output_features * num_groups);
            std::vector<float>      input_f32(tokens * K);
            std::vector<float>      output_zendnn(tokens * output_features);
            std::vector<float>      output_custom(tokens * output_features);

            auto t_ctx_end = std::chrono::steady_clock::now();
            double ctx_ms  = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

            auto t_op_start = std::chrono::steady_clock::now();

            std::vector<float> tmp_wei(output_features * K);
            fill_buffer(tmp_wei.data(), output_features * K, desc.data_seed);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, tmp_wei.data(), weights_q4.data(), 0, output_features, K, nullptr);

            fill_buffer(input_f32.data(), tokens * K, desc.data_seed + 95);

            auto t_op_end = std::chrono::steady_clock::now();
            double op_ms  = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

            {
                bool ok = custom_matmul_q4_0(
                    desc.threads, output_features, tokens, K,
                    weights_q4.data(), input_f32.data(),
                    output_custom.data(), output_features);
                if (!ok) {
                    fprintf(stderr, "error: Q4_0 custom kernel run failed\n"); exit(1);
                }
            }

            for (int i = 0; i < desc.warmup; i++) {
                bool ok = ggml_zendnn_matmul_q4_0_woq_bf16(
                    desc.threads, output_features, tokens, K,
                    weights_q4.data(), input_f32.data(),
                    output_zendnn.data(), output_features);

                if (!ok) { fprintf(stderr, "error: Q4_0 ZenDNN warmup\n"); exit(1); }
            }

            double min_ms = std::numeric_limits<double>::max();
            double max_ms = 0.0, sum_ms = 0.0;

            for (int i = 0; i < desc.repeats; i++) {
                auto t0 = std::chrono::steady_clock::now();

                bool ok = ggml_zendnn_matmul_q4_0_woq_bf16(
                    desc.threads, output_features, tokens, K,
                    weights_q4.data(), input_f32.data(),
                    output_zendnn.data(), output_features);
                

                auto t1 = std::chrono::steady_clock::now();
                if (!ok) { fprintf(stderr, "error: Q4_0 ZenDNN matmul\n"); exit(1); }

                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                min_ms = std::min(min_ms, ms);
                max_ms = std::max(max_ms, ms);
                sum_ms += ms;
            }

            double avg_ms = sum_ms / desc.repeats;
            double tflops = (2.0 * output_features * tokens * K) / (avg_ms * 1e-3) / 1e12;

            BenchResult result;
            result.min_ms = min_ms; result.avg_ms = avg_ms;
            result.max_ms = max_ms; result.tflops = tflops;
            result.ctx_creation_ms = ctx_ms;
            result.op_creation_ms  = op_ms;
            result.op_execution_ms = avg_ms;
            result.other_ms        = 0.0;

            if (desc.verify_output) {
                result.output_data.assign(output_custom.begin(), output_custom.end());
            }

            return result;
        }
    }

    // =========================================================================
    //  Regular FP32 / BF16 — ZenDNN
    // =========================================================================
    zendnnl::common::data_type_t src_dt = ggml_type_to_zendnn(src_ggml);
    zendnnl::common::data_type_t wei_dt = ggml_type_to_zendnn(wei_ggml);
    zendnnl::common::data_type_t dst_dt = zendnnl::common::data_type_t::f32;

    size_t a_size = output_features * K;  
    size_t b_size = tokens * K;           
    size_t c_size = tokens * output_features; 

    size_t wei_elem = (wei_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);
    size_t src_elem = (src_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);

    std::vector<char> a_data(a_size * wei_elem);
    std::vector<char> b_data(b_size * src_elem);
    std::vector<char> c_data(c_size * sizeof(float));

    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_ms  = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    auto t_op_start = std::chrono::steady_clock::now();
    void* a_ptr = a_data.data();
    void* b_ptr = b_data.data();
    void* c_ptr = c_data.data();

    if (wei_dt == zendnnl::common::data_type_t::f32)
        fill_buffer(reinterpret_cast<float*>(a_ptr), a_size, desc.data_seed);
    else
        fill_buffer_bf16(a_ptr, a_size, desc.data_seed);

    if (src_dt == zendnnl::common::data_type_t::f32)
        fill_buffer(reinterpret_cast<float*>(b_ptr), b_size, desc.data_seed + 95);
    else
        fill_buffer_bf16(b_ptr, b_size, desc.data_seed + 95);

    matmul_data_types dtypes;
    dtypes.src     = src_dt; dtypes.wei = wei_dt; dtypes.dst = dst_dt;
    dtypes.bias    = zendnnl::common::data_type_t::none;
    dtypes.compute = zendnnl::common::data_type_t::f32;

    matmul_params params;
    params.dtypes      = dtypes;
    params.num_threads = desc.threads;

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    auto t_op_end = std::chrono::steady_clock::now();
    double op_ms  = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    for (int i = 0; i < desc.warmup; i++) {
        status_t s = matmul_direct(
            'r', false, true, tokens, output_features, K, 1.0f,
            b_ptr, K, a_ptr, K, nullptr, 0.0f,
            c_ptr, output_features, true, batch_params, params);
        if (s != status_t::success) {
            fprintf(stderr, "error: ZenDNN warmup\n"); exit(1);
        }
    }

    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0, sum_ms = 0.0;

    for (int i = 0; i < desc.repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();

        status_t s = matmul_direct(
            'r', false, true, tokens, output_features, K, 1.0f,
            b_ptr, K, a_ptr, K, nullptr, 0.0f,
            c_ptr, output_features, true, batch_params, params);

        auto t1 = std::chrono::steady_clock::now();
        if (s != status_t::success) {
            fprintf(stderr, "error: ZenDNN matmul\n"); exit(1);
        }

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / desc.repeats;
    double tflops = (2.0 * output_features * tokens * K) / (avg_ms * 1e-3) / 1e12;

    BenchResult result;
    result.min_ms = min_ms; result.avg_ms = avg_ms;
    result.max_ms = max_ms; result.tflops = tflops;
    result.ctx_creation_ms = ctx_ms;
    result.op_creation_ms  = op_ms;
    result.op_execution_ms = avg_ms;
    result.other_ms        = 0.0;

    if (desc.verify_output) {
        result.output_data.resize(c_size);
        std::copy(reinterpret_cast<const float*>(c_ptr),
                  reinterpret_cast<const float*>(c_ptr) + c_size,
                  result.output_data.begin());
    }

    return result;
}

// ===========================================================================
//  bench_matmul_id_zendnn   (MoE group GEMM — unchanged)
// ===========================================================================
BenchResult bench_matmul_id_zendnn(const OpDesc& desc)
{
    const int64_t output_features = desc.m;
    const int64_t tokens          = desc.n;
    const int64_t K               = desc.k;
    const int64_t n_exp           = desc.n_experts;
    const int64_t n_used          = desc.n_experts_used;
    const ggml_type src_ggml      = desc.src_dtype;
    const ggml_type wei_ggml      = desc.wei_dtype;

    zendnnl::common::data_type_t src_dt = ggml_type_to_zendnn(src_ggml);
    zendnnl::common::data_type_t wei_dt = ggml_type_to_zendnn(wei_ggml);
    zendnnl::common::data_type_t dst_dt = zendnnl::common::data_type_t::f32;

    auto t_ctx_start = std::chrono::steady_clock::now();

    size_t expert_size = K * output_features;
    size_t input_size  = K * n_used * tokens;
    size_t output_size = output_features * n_used * tokens;

    size_t wei_elem = (wei_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);
    size_t src_elem = (src_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);
    size_t dst_elem = sizeof(float);

    std::vector<char>    expert_data(expert_size * n_exp * wei_elem);
    std::vector<char>    input_data(input_size   * src_elem);
    std::vector<char>    output_data(output_size  * dst_elem);
    std::vector<int32_t> ids_data(n_used * tokens);

    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_ms  = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    auto t_op_start = std::chrono::steady_clock::now();
    void* ep = expert_data.data();
    void* ip = input_data.data();
    void* op = output_data.data();

    if (wei_dt == zendnnl::common::data_type_t::f32)
        fill_buffer(reinterpret_cast<float*>(ep), expert_size*n_exp, desc.data_seed);
    else
        fill_buffer_bf16(ep, expert_size*n_exp, desc.data_seed);

    if (src_dt == zendnnl::common::data_type_t::f32)
        fill_buffer(reinterpret_cast<float*>(ip), input_size, desc.data_seed+95);
    else
        fill_buffer_bf16(ip, input_size, desc.data_seed+95);

    std::vector<int32_t> routing_ids = generate_routing_ids(
        tokens, n_exp, n_used,
        desc.expert_token_counts, desc.routing_pattern, desc.routing_seed);
    std::copy(routing_ids.begin(), routing_ids.end(), ids_data.begin());

    struct TS { int32_t slot, token; };
    std::vector<std::vector<TS>> e2t(n_exp);
    for (int64_t t = 0; t < tokens; t++)
        for (int64_t s = 0; s < n_used; s++) {
            int32_t eid = ids_data[t*n_used+s];
            e2t[eid].push_back({(int32_t)s, (int32_t)t});
        }

    std::vector<char>        layout;
    std::vector<bool>        transA, transB, iwc;
    std::vector<int>         Mv, Nv, Kv, lda_v, ldb_v, ldc_v;
    std::vector<float>       alph, bet;
    std::vector<const void*> sptrs, wptrs, bptrs;
    std::vector<void*>       dptrs;
    std::vector<matmul_params> pvec;
    std::vector<std::vector<char>> eib, eob;

    matmul_data_types dtypes;
    dtypes.src = src_dt; dtypes.wei = wei_dt; dtypes.dst = dst_dt;
    dtypes.bias = zendnnl::common::data_type_t::none;
    dtypes.compute = zendnnl::common::data_type_t::f32;

    matmul_params pt;
    pt.dtypes = dtypes; pt.num_threads = desc.threads;

    for (int64_t eid = 0; eid < n_exp; eid++) {
        const auto& tv = e2t[eid];
        if (tv.empty()) continue;
        int64_t nr = (int64_t)tv.size();
        eib.emplace_back(nr * K * src_elem);
        eob.emplace_back(nr * output_features * dst_elem);
        char* ib = eib.back().data();
        char* ob = eob.back().data();
        layout.push_back('r');
        transA.push_back(false); transB.push_back(true);
        Mv.push_back((int)nr); Nv.push_back((int)output_features);
        Kv.push_back((int)K);
        alph.push_back(1.f); bet.push_back(0.f);
        lda_v.push_back((int)K); ldb_v.push_back((int)K);
        ldc_v.push_back((int)output_features);
        iwc.push_back(true);
        sptrs.push_back(ib);
        wptrs.push_back(reinterpret_cast<const char*>(ep) + eid*expert_size*wei_elem);
        bptrs.push_back(nullptr);
        dptrs.push_back(ob);
        pvec.push_back(pt);
    }

    auto t_op_end = std::chrono::steady_clock::now();
    double op_ms  = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    auto gather = [&]() {
        size_t ei = 0;
        for (int64_t eid = 0; eid < n_exp; eid++) {
            const auto& tv = e2t[eid];
            if (tv.empty()) continue;
            char* ib = eib[ei].data();
            for (size_t i = 0; i < tv.size(); i++) {
                size_t off = ((size_t)tv[i].token*n_used*K + (size_t)tv[i].slot*K) * src_elem;
                memcpy(ib + i*K*src_elem, reinterpret_cast<const char*>(ip)+off, K*src_elem);
            }
            ei++;
        }
    };

    auto scatter = [&]() {
        size_t ei = 0;
        for (int64_t eid = 0; eid < n_exp; eid++) {
            const auto& tv = e2t[eid];
            if (tv.empty()) continue;
            const char* ob = eob[ei].data();
            for (size_t i = 0; i < tv.size(); i++) {
                size_t off = ((size_t)tv[i].token*n_used*output_features + (size_t)tv[i].slot*output_features) * dst_elem;
                memcpy(reinterpret_cast<char*>(op)+off, ob + i*output_features*dst_elem, output_features*dst_elem);
            }
            ei++;
        }
    };

    for (int i = 0; i < desc.warmup; i++) {
        if (!layout.empty()) {
            gather();
            status_t s = group_matmul_direct(
                layout,transA,transB,Mv,Nv,Kv,alph,
                sptrs,lda_v,wptrs,ldb_v,bptrs,bet,
                dptrs,ldc_v,iwc,pvec);
            if (s != status_t::success) {
                fprintf(stderr,"error: group GEMM warmup\n"); exit(1);
            }
            scatter();
        }
    }

    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0, sum_ms = 0.0;

    for (int i = 0; i < desc.repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();
        if (!layout.empty()) {
            gather();
            status_t s = group_matmul_direct(
                layout,transA,transB,Mv,Nv,Kv,alph,
                sptrs,lda_v,wptrs,ldb_v,bptrs,bet,
                dptrs,ldc_v,iwc,pvec);
            if (s != status_t::success) {
                fprintf(stderr,"error: group GEMM\n"); exit(1);
            }
            scatter();
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        min_ms = std::min(min_ms,ms);
        max_ms = std::max(max_ms,ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / desc.repeats;
    double tflops = (2.0*output_features*K*n_used*tokens) / (avg_ms*1e-3) / 1e12;

    BenchResult result;
    result.min_ms = min_ms; result.avg_ms = avg_ms;
    result.max_ms = max_ms; result.tflops = tflops;
    result.ctx_creation_ms = ctx_ms;
    result.op_creation_ms  = op_ms;
    result.op_execution_ms = avg_ms;
    result.other_ms        = 0.0;

    if (desc.verify_output) {
        result.output_data.resize(output_size);
        std::copy(reinterpret_cast<const float*>(op),
                  reinterpret_cast<const float*>(op) + output_size,
                  result.output_data.begin());
    }

    return result;
}