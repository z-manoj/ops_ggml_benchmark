#include "custom_moe.h"
#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>
#include <cpuid.h>

// ---------------------------------------------------------------------------
// Cache-blocking tile sizes (tuned for AMD EPYC 9R14: 32KB L1d, 1MB L2, 384MB L3)
// Kept at AVX2-tuned values: cache locality more important than matching vector width
// AVX-512 benefit comes from wider vectors (32 floats/iter), not larger tiles
// ---------------------------------------------------------------------------
static constexpr int64_t N_TILE = 128;   // output rows per tile
static constexpr int64_t K_TILE = 1024;  // inner-dim elements per tile
// Working set: 128 × 1024 × 2 bytes (f16) = 256KB (25% of 1MB L2, optimal)

// ---------------------------------------------------------------------------
// Runtime CPU detection
// ---------------------------------------------------------------------------
static bool cpu_has_avx512f() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 16)) != 0;  // AVX512F bit
}

static bool cpu_has_avx512_bf16() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 1, eax, ebx, ecx, edx);
    return (eax & (1 << 5)) != 0;  // AVX512_BF16 bit
}

static const bool g_has_avx512 = cpu_has_avx512f();
static const bool g_has_avx512_bf16 = cpu_has_avx512_bf16();

// NOTE: BF16 native compute via _mm512_dpbf16_ps() would eliminate FP16→FP32
// conversion overhead and reduce memory bandwidth by ~30%, but requires:
// 1. GGML to store weights in BF16 format (currently uses FP16)
// 2. FP16→BF16 conversion layer (or upstream GGML changes)
// Current implementation uses AVX-512 FP32 compute with FP16 input conversion.

// ---------------------------------------------------------------------------
// AVX-512 horizontal reduction (much simpler than AVX2!)
// ---------------------------------------------------------------------------
static inline float hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

// ---------------------------------------------------------------------------
// AVX2 horizontal reduction helper (fallback)
// ---------------------------------------------------------------------------
static inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ===========================================================================
// AVX-512 KERNELS (2x wider, processes 16 floats per vector)
// ===========================================================================

// ---------------------------------------------------------------------------
// Multi-row dot product (AVX-512): compute 4 output rows at once
// Processes 32 floats per iteration (2× __m512 vectors)
// ---------------------------------------------------------------------------
static inline void dot4_f32_avx512(
        const float* w0, const float* w1,
        const float* w2, const float* w3,
        const float* inp, int64_t len,
        float& out0, float& out1, float& out2, float& out3) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    int64_t i = 0;
    // Process 32 floats per iteration
    for (; i + 31 < len; i += 32) {
        __m512 in0 = _mm512_loadu_ps(inp + i);
        __m512 in1 = _mm512_loadu_ps(inp + i + 16);

        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + i),      in0, acc0);
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + i + 16), in1, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + i),      in0, acc1);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + i + 16), in1, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + i),      in0, acc2);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + i + 16), in1, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + i),      in0, acc3);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + i + 16), in1, acc3);
    }
    // Process remaining 16 floats
    for (; i + 15 < len; i += 16) {
        __m512 in0 = _mm512_loadu_ps(inp + i);
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(w0 + i), in0, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(w1 + i), in0, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(w2 + i), in0, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(w3 + i), in0, acc3);
    }

    out0 = hsum_avx512(acc0);
    out1 = hsum_avx512(acc1);
    out2 = hsum_avx512(acc2);
    out3 = hsum_avx512(acc3);

    // Scalar tail
    for (; i < len; i++) {
        float v = inp[i];
        out0 += w0[i] * v;
        out1 += w1[i] * v;
        out2 += w2[i] * v;
        out3 += w3[i] * v;
    }
}

static inline void dot4_f16_f32_avx512(
        const ggml_fp16_t* w0, const ggml_fp16_t* w1,
        const ggml_fp16_t* w2, const ggml_fp16_t* w3,
        const float* inp, int64_t len,
        float& out0, float& out1, float& out2, float& out3) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();

    int64_t i = 0;
    // Process 32 floats per iteration (load 32 FP16 → convert to 2× __m512)
    for (; i + 31 < len; i += 32) {
        __m512 in0 = _mm512_loadu_ps(inp + i);
        __m512 in1 = _mm512_loadu_ps(inp + i + 16);

        acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w0+i))),    in0, acc0);
        acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w0+i+16))), in1, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w1+i))),    in0, acc1);
        acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w1+i+16))), in1, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w2+i))),    in0, acc2);
        acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w2+i+16))), in1, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w3+i))),    in0, acc3);
        acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w3+i+16))), in1, acc3);
    }
    // Process remaining 16 floats
    for (; i + 15 < len; i += 16) {
        __m512 in0 = _mm512_loadu_ps(inp + i);
        acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w0+i))), in0, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w1+i))), in0, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w2+i))), in0, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(w3+i))), in0, acc3);
    }

    out0 = hsum_avx512(acc0);
    out1 = hsum_avx512(acc1);
    out2 = hsum_avx512(acc2);
    out3 = hsum_avx512(acc3);

    // Scalar tail
    for (; i < len; i++) {
        float v = inp[i];
        out0 += ggml_fp16_to_fp32(w0[i]) * v;
        out1 += ggml_fp16_to_fp32(w1[i]) * v;
        out2 += ggml_fp16_to_fp32(w2[i]) * v;
        out3 += ggml_fp16_to_fp32(w3[i]) * v;
    }
}

// Single-row fallback (AVX-512)
static inline float dot_f32_avx512(const float* a, const float* b, int64_t K) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 31 < K; i += 32) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), sum1);
    }
    for (; i + 15 < K; i += 16) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sum0);
    }
    sum0 = _mm512_add_ps(sum0, sum1);
    float result = hsum_avx512(sum0);
    for (; i < K; i++) result += a[i] * b[i];
    return result;
}

static inline float dot_f16_f32_avx512(const ggml_fp16_t* a, const float* b, int64_t K) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 31 < K; i += 32) {
        sum0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(a+i))),    _mm512_loadu_ps(b+i),    sum0);
        sum1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(a+i+16))), _mm512_loadu_ps(b+i+16), sum1);
    }
    for (; i + 15 < K; i += 16) {
        sum0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(a+i))), _mm512_loadu_ps(b+i), sum0);
    }
    sum0 = _mm512_add_ps(sum0, sum1);
    float result = hsum_avx512(sum0);
    for (; i < K; i++) result += ggml_fp16_to_fp32(a[i]) * b[i];
    return result;
}

// ===========================================================================
// AVX2 KERNELS (fallback for CPUs without AVX-512)
// ===========================================================================

static inline void dot4_f32_avx2(
        const float* w0, const float* w1,
        const float* w2, const float* w3,
        const float* inp, int64_t len,
        float& out0, float& out1, float& out2, float& out3) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        __m256 in0 = _mm256_loadu_ps(inp + i);
        __m256 in1 = _mm256_loadu_ps(inp + i + 8);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i),     in0, acc0);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i + 8), in1, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i),     in0, acc1);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i + 8), in1, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + i),     in0, acc2);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + i + 8), in1, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + i),     in0, acc3);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + i + 8), in1, acc3);
    }
    for (; i + 7 < len; i += 8) {
        __m256 in0 = _mm256_loadu_ps(inp + i);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i), in0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i), in0, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + i), in0, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + i), in0, acc3);
    }

    out0 = hsum_avx2(acc0);
    out1 = hsum_avx2(acc1);
    out2 = hsum_avx2(acc2);
    out3 = hsum_avx2(acc3);

    for (; i < len; i++) {
        float v = inp[i];
        out0 += w0[i] * v;
        out1 += w1[i] * v;
        out2 += w2[i] * v;
        out3 += w3[i] * v;
    }
}

static inline void dot4_f16_f32_avx2(
        const ggml_fp16_t* w0, const ggml_fp16_t* w1,
        const ggml_fp16_t* w2, const ggml_fp16_t* w3,
        const float* inp, int64_t len,
        float& out0, float& out1, float& out2, float& out3) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int64_t i = 0;
    for (; i + 15 < len; i += 16) {
        __m256 in0 = _mm256_loadu_ps(inp + i);
        __m256 in1 = _mm256_loadu_ps(inp + i + 8);

        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0+i))),     in0, acc0);
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0+i+8))),   in1, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1+i))),     in0, acc1);
        acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1+i+8))),   in1, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2+i))),     in0, acc2);
        acc2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2+i+8))),   in1, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3+i))),     in0, acc3);
        acc3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3+i+8))),   in1, acc3);
    }
    for (; i + 7 < len; i += 8) {
        __m256 in0 = _mm256_loadu_ps(inp + i);
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w0+i))), in0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w1+i))), in0, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w2+i))), in0, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w3+i))), in0, acc3);
    }

    out0 = hsum_avx2(acc0);
    out1 = hsum_avx2(acc1);
    out2 = hsum_avx2(acc2);
    out3 = hsum_avx2(acc3);

    for (; i < len; i++) {
        float v = inp[i];
        out0 += ggml_fp16_to_fp32(w0[i]) * v;
        out1 += ggml_fp16_to_fp32(w1[i]) * v;
        out2 += ggml_fp16_to_fp32(w2[i]) * v;
        out3 += ggml_fp16_to_fp32(w3[i]) * v;
    }
}

static inline float dot_f32_avx2(const float* a, const float* b, int64_t K) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 15 < K; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    for (; i + 7 < K; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    float result = hsum_avx2(sum0);
    for (; i < K; i++) result += a[i] * b[i];
    return result;
}

static inline float dot_f16_f32_avx2(const ggml_fp16_t* a, const float* b, int64_t K) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 15 < K; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a+i))),   _mm256_loadu_ps(b+i),   sum0);
        sum1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a+i+8))), _mm256_loadu_ps(b+i+8), sum1);
    }
    for (; i + 7 < K; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(a+i))), _mm256_loadu_ps(b+i), sum0);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    float result = hsum_avx2(sum0);
    for (; i < K; i++) result += ggml_fp16_to_fp32(a[i]) * b[i];
    return result;
}

// ===========================================================================
// DISPATCH WRAPPERS (select AVX-512 or AVX2 at runtime)
// ===========================================================================

static inline void dot4_f32(const float* w0, const float* w1, const float* w2, const float* w3,
                            const float* inp, int64_t len,
                            float& out0, float& out1, float& out2, float& out3) {
    if (g_has_avx512) {
        dot4_f32_avx512(w0, w1, w2, w3, inp, len, out0, out1, out2, out3);
    } else {
        dot4_f32_avx2(w0, w1, w2, w3, inp, len, out0, out1, out2, out3);
    }
}

static inline void dot4_f16_f32(const ggml_fp16_t* w0, const ggml_fp16_t* w1,
                                const ggml_fp16_t* w2, const ggml_fp16_t* w3,
                                const float* inp, int64_t len,
                                float& out0, float& out1, float& out2, float& out3) {
    if (g_has_avx512) {
        dot4_f16_f32_avx512(w0, w1, w2, w3, inp, len, out0, out1, out2, out3);
    } else {
        dot4_f16_f32_avx2(w0, w1, w2, w3, inp, len, out0, out1, out2, out3);
    }
}

static inline float dot_f32(const float* a, const float* b, int64_t K) {
    if (g_has_avx512) {
        return dot_f32_avx512(a, b, K);
    } else {
        return dot_f32_avx2(a, b, K);
    }
}

static inline float dot_f16_f32(const ggml_fp16_t* a, const float* b, int64_t K) {
    if (g_has_avx512) {
        return dot_f16_f32_avx512(a, b, K);
    } else {
        return dot_f16_f32_avx2(a, b, K);
    }
}

// ---------------------------------------------------------------------------
// Expert grouping
// ---------------------------------------------------------------------------
struct TokenSlot {
    int32_t token;
    int32_t slot;
};

static void group_by_expert(const int32_t* ids_data,
                            int64_t n_tok, int64_t n_used, int64_t n_exp,
                            std::vector<std::vector<TokenSlot>>& groups) {
    groups.resize(n_exp);
    for (auto& g : groups) g.clear();
    for (int64_t t = 0; t < n_tok; t++) {
        for (int64_t s = 0; s < n_used; s++) {
            int32_t eidx = ids_data[t * n_used + s];
            groups[eidx].push_back({(int32_t)t, (int32_t)s});
        }
    }
}

// ---------------------------------------------------------------------------
// custom_moe_compute  --  optimized with GEMM-style weight reuse
//
// Key insight: for each expert, process ALL routed tokens within each
// weight row-tile. The weight tile is loaded once into L2 and reused
// across every token assigned to that expert (~128 tokens on average
// for typical MoE configs). This converts scattered GEMV into batched
// GEMM, dramatically improving arithmetic intensity.
//
// Loop order per expert:
//   parallel for row_tile in [0, N):
//     for k_tile in [0, K):          <- weight tile in L2
//       for token in expert_group:   <- all tokens reuse weight tile
//         4-row micro-kernel
//
// Additional optimizations:
//   - 4-row micro-kernel: loads input vector once, dots 4 weight rows
//   - Software prefetching for next weight rows
//   - Dynamic scheduling across row tiles for load balance
//   - Runtime dispatch: AVX-512 (32 floats/iter) or AVX2 (16 floats/iter)
// ---------------------------------------------------------------------------
void custom_moe_compute(struct ggml_tensor* dst,
                        const struct ggml_tensor* experts,
                        const struct ggml_tensor* input,
                        const struct ggml_tensor* ids,
                        int n_threads) {
    const int64_t K      = experts->ne[0];
    const int64_t N      = experts->ne[1];
    const int64_t n_exp  = experts->ne[2];
    const int64_t n_used = ids->ne[0];
    const int64_t n_tok  = ids->ne[1];

    const bool is_f16 = (experts->type == GGML_TYPE_F16);

    const int64_t exp_stride_row    = K;
    const int64_t exp_stride_expert = N * K;
    const int64_t inp_stride_slot   = K;
    const int64_t inp_stride_tok    = n_used * K;
    const int64_t dst_stride_slot   = N;
    const int64_t dst_stride_tok    = n_used * N;

    const int32_t* ids_data = reinterpret_cast<const int32_t*>(ids->data);
    const float*   inp_data = reinterpret_cast<const float*>(input->data);
    float*         dst_data = reinterpret_cast<float*>(dst->data);

    omp_set_num_threads(n_threads);

    // Phase 1: Group tokens by expert
    std::vector<std::vector<TokenSlot>> groups;
    group_by_expert(ids_data, n_tok, n_used, n_exp, groups);

    const int64_t n_row_tiles = (N + N_TILE - 1) / N_TILE;

    // Build compact list of active experts (those with routed tokens)
    std::vector<int32_t> active_experts;
    active_experts.reserve(n_exp);
    for (int32_t e = 0; e < (int32_t)n_exp; e++) {
        if (!groups[e].empty()) active_experts.push_back(e);
    }
    const int64_t n_active = (int64_t)active_experts.size();
    const int64_t total_tiles = n_active * n_row_tiles;

    // Phase 2: Single dispatch over all (expert × row_tile) pairs
    if (is_f16) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t flat_idx = 0; flat_idx < total_tiles; flat_idx++) {
            const int64_t expert_idx = flat_idx / n_row_tiles;
            const int64_t tile_idx   = flat_idx % n_row_tiles;
            const int32_t e = active_experts[expert_idx];
            const auto& grp = groups[e];
            const int64_t count_e = (int64_t)grp.size();

            const ggml_fp16_t* exp_base = reinterpret_cast<const ggml_fp16_t*>(experts->data)
                                          + e * exp_stride_expert;
            const int64_t r0    = tile_idx * N_TILE;
            const int64_t r_end = std::min(r0 + N_TILE, N);

            for (int64_t k0 = 0; k0 < K; k0 += K_TILE) {
                const int64_t k_end = std::min(k0 + K_TILE, K);
                const int64_t k_len = k_end - k0;
                const bool first_k = (k0 == 0);

                for (int64_t gi = 0; gi < count_e; gi++) {
                    const int32_t t = grp[gi].token;
                    const int32_t s = grp[gi].slot;
                    const float* inp_vec = inp_data + t * inp_stride_tok
                                         + s * inp_stride_slot + k0;
                    float* dst_vec = dst_data + t * dst_stride_tok
                                   + s * dst_stride_slot;

                    int64_t r = r0;
                    for (; r + 3 < r_end; r += 4) {
                        const ggml_fp16_t* w0 = exp_base + r * exp_stride_row + k0;
                        const ggml_fp16_t* w1 = w0 + exp_stride_row;
                        const ggml_fp16_t* w2 = w1 + exp_stride_row;
                        const ggml_fp16_t* w3 = w2 + exp_stride_row;

                        if (r + 7 < r_end) {
                            _mm_prefetch((const char*)(w3 + exp_stride_row), _MM_HINT_T0);
                            _mm_prefetch((const char*)(w3 + 2*exp_stride_row), _MM_HINT_T0);
                        }

                        float d0, d1, d2, d3;
                        dot4_f16_f32(w0, w1, w2, w3, inp_vec, k_len,
                                     d0, d1, d2, d3);
                        if (first_k) {
                            dst_vec[r]   = d0; dst_vec[r+1] = d1;
                            dst_vec[r+2] = d2; dst_vec[r+3] = d3;
                        } else {
                            dst_vec[r]   += d0; dst_vec[r+1] += d1;
                            dst_vec[r+2] += d2; dst_vec[r+3] += d3;
                        }
                    }
                    for (; r < r_end; r++) {
                        float d = dot_f16_f32(exp_base + r * exp_stride_row + k0,
                                              inp_vec, k_len);
                        if (first_k) dst_vec[r] = d;
                        else         dst_vec[r] += d;
                    }
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t flat_idx = 0; flat_idx < total_tiles; flat_idx++) {
            const int64_t expert_idx = flat_idx / n_row_tiles;
            const int64_t tile_idx   = flat_idx % n_row_tiles;
            const int32_t e = active_experts[expert_idx];
            const auto& grp = groups[e];
            const int64_t count_e = (int64_t)grp.size();

            const float* exp_base = reinterpret_cast<const float*>(experts->data)
                                    + e * exp_stride_expert;
            const int64_t r0    = tile_idx * N_TILE;
            const int64_t r_end = std::min(r0 + N_TILE, N);

            for (int64_t k0 = 0; k0 < K; k0 += K_TILE) {
                const int64_t k_end = std::min(k0 + K_TILE, K);
                const int64_t k_len = k_end - k0;
                const bool first_k = (k0 == 0);

                for (int64_t gi = 0; gi < count_e; gi++) {
                    const int32_t t = grp[gi].token;
                    const int32_t s = grp[gi].slot;
                    const float* inp_vec = inp_data + t * inp_stride_tok
                                         + s * inp_stride_slot + k0;
                    float* dst_vec = dst_data + t * dst_stride_tok
                                   + s * dst_stride_slot;

                    int64_t r = r0;
                    for (; r + 3 < r_end; r += 4) {
                        const float* w0 = exp_base + r * exp_stride_row + k0;
                        const float* w1 = w0 + exp_stride_row;
                        const float* w2 = w1 + exp_stride_row;
                        const float* w3 = w2 + exp_stride_row;

                        if (r + 7 < r_end) {
                            _mm_prefetch((const char*)(w3 + exp_stride_row), _MM_HINT_T0);
                            _mm_prefetch((const char*)(w3 + 2*exp_stride_row), _MM_HINT_T0);
                        }

                        float d0, d1, d2, d3;
                        dot4_f32(w0, w1, w2, w3, inp_vec, k_len,
                                 d0, d1, d2, d3);
                        if (first_k) {
                            dst_vec[r]   = d0; dst_vec[r+1] = d1;
                            dst_vec[r+2] = d2; dst_vec[r+3] = d3;
                        } else {
                            dst_vec[r]   += d0; dst_vec[r+1] += d1;
                            dst_vec[r+2] += d2; dst_vec[r+3] += d3;
                        }
                    }
                    for (; r < r_end; r++) {
                        float d = dot_f32(exp_base + r * exp_stride_row + k0,
                                          inp_vec, k_len);
                        if (first_k) dst_vec[r] = d;
                        else         dst_vec[r] += d;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// custom_matmul_compute  --  2D-tiled GEMM with K-tiling
//
// C(M,N) = A^T(M,K) * B(K,N)
// GGML layout: A:[K,M], B:[K,N], dst:[M,N]  (col j of dst at dst + j*M)
//
// 2D tiling: parallel over (row_tile × col_group) pairs for load balance.
// K-tiling: for each tile, iterate K in blocks so A tile stays in L2
// and is reused across J_TILE columns of B.
//
// Working set per (row_tile, k_tile, column):
//   A tile: N_TILE × K_TILE × elem_size  (~256KB f16, 25% of L2)
//   B col:  K_TILE × 4                   (~4KB, fits L1d)
//
// Runtime dispatch: AVX-512 (32 floats/iter) or AVX2 (16 floats/iter)
// ---------------------------------------------------------------------------
static constexpr int64_t J_TILE = 24;  // columns of B per tile

void custom_matmul_compute(struct ggml_tensor* dst,
                           const struct ggml_tensor* A,
                           const struct ggml_tensor* B,
                           int n_threads) {
    const int64_t K = A->ne[0];
    const int64_t M = A->ne[1];
    const int64_t N = B->ne[1];

    const bool is_f16 = (A->type == GGML_TYPE_F16);

    const float* b_data   = reinterpret_cast<const float*>(B->data);
    float*       dst_data = reinterpret_cast<float*>(dst->data);

    const int64_t n_row_tiles  = (M + N_TILE - 1) / N_TILE;
    const int64_t n_col_groups = (N + J_TILE - 1) / J_TILE;
    const int64_t total_tiles  = n_row_tiles * n_col_groups;

    omp_set_num_threads(n_threads);

    if (is_f16) {
        const ggml_fp16_t* a_data = reinterpret_cast<const ggml_fp16_t*>(A->data);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t tile = 0; tile < total_tiles; tile++) {
            const int64_t i0    = (tile / n_col_groups) * N_TILE;
            const int64_t j0    = (tile % n_col_groups) * J_TILE;
            const int64_t i_end = std::min(i0 + N_TILE, M);
            const int64_t j_end = std::min(j0 + J_TILE, N);

            for (int64_t k0 = 0; k0 < K; k0 += K_TILE) {
                const int64_t k_len = std::min(k0 + K_TILE, K) - k0;
                const bool first_k = (k0 == 0);

                for (int64_t j = j0; j < j_end; j++) {
                    const float* b_col = b_data + j * K + k0;
                    float*       d_col = dst_data + j * M;

                    int64_t i = i0;
                    for (; i + 3 < i_end; i += 4) {
                        const ggml_fp16_t* w0 = a_data + i * K + k0;

                        float d0, d1, d2, d3;
                        dot4_f16_f32(w0, w0+K, w0+2*K, w0+3*K,
                                     b_col, k_len, d0, d1, d2, d3);
                        if (first_k) { d_col[i]=d0; d_col[i+1]=d1; d_col[i+2]=d2; d_col[i+3]=d3; }
                        else         { d_col[i]+=d0; d_col[i+1]+=d1; d_col[i+2]+=d2; d_col[i+3]+=d3; }
                    }
                    for (; i < i_end; i++) {
                        float d = dot_f16_f32(a_data + i*K + k0, b_col, k_len);
                        if (first_k) d_col[i] = d;
                        else         d_col[i] += d;
                    }
                }
            }
        }
    } else {
        const float* a_data = reinterpret_cast<const float*>(A->data);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t tile = 0; tile < total_tiles; tile++) {
            const int64_t i0    = (tile / n_col_groups) * N_TILE;
            const int64_t j0    = (tile % n_col_groups) * J_TILE;
            const int64_t i_end = std::min(i0 + N_TILE, M);
            const int64_t j_end = std::min(j0 + J_TILE, N);

            for (int64_t k0 = 0; k0 < K; k0 += K_TILE) {
                const int64_t k_len = std::min(k0 + K_TILE, K) - k0;
                const bool first_k = (k0 == 0);

                for (int64_t j = j0; j < j_end; j++) {
                    const float* b_col = b_data + j * K + k0;
                    float*       d_col = dst_data + j * M;

                    int64_t i = i0;
                    for (; i + 3 < i_end; i += 4) {
                        const float* w0 = a_data + i * K + k0;

                        float d0, d1, d2, d3;
                        dot4_f32(w0, w0+K, w0+2*K, w0+3*K,
                                 b_col, k_len, d0, d1, d2, d3);
                        if (first_k) { d_col[i]=d0; d_col[i+1]=d1; d_col[i+2]=d2; d_col[i+3]=d3; }
                        else         { d_col[i]+=d0; d_col[i+1]+=d1; d_col[i+2]+=d2; d_col[i+3]+=d3; }
                    }
                    for (; i < i_end; i++) {
                        float d = dot_f32(a_data + i*K + k0, b_col, k_len);
                        if (first_k) d_col[i] = d;
                        else         d_col[i] += d;
                    }
                }
            }
        }
    }
}
