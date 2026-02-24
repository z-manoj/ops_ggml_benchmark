#include "zendnn_matmul_bench.h"
#include "ggml_utils.h"

#include "zendnnl.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstring>

using namespace zendnnl::lowoha::matmul;

// ---------------------------------------------------------------------------
// Helper: Convert ggml_type to zendnnl::common::data_type_t
// ---------------------------------------------------------------------------
static zendnnl::common::data_type_t ggml_type_to_zendnn(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return zendnnl::common::data_type_t::f32;
        case GGML_TYPE_BF16: return zendnnl::common::data_type_t::bf16;
        default:
            fprintf(stderr, "error: unsupported ggml_type %d for ZenDNN (only f32 and bf16 supported)\n", t);
            exit(1);
    }
}

// ---------------------------------------------------------------------------
// Helper: Simple xorshift32 for deterministic data
// ---------------------------------------------------------------------------
static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// ---------------------------------------------------------------------------
// Helper: Fill memory with deterministic pseudo-random data
// ---------------------------------------------------------------------------
template <typename T>
static void fill_buffer(T* data, size_t n, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t r = xorshift32(state);
        float val = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
        data[i] = static_cast<T>(val);
    }
}

// Specialization for bf16
static void fill_buffer_bf16(void* data, size_t n, uint32_t seed) {
    uint16_t* ptr = reinterpret_cast<uint16_t*>(data);
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t r = xorshift32(state);
        float val = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
        // BF16 is just top 16 bits of float32
        ptr[i] = static_cast<uint16_t>(*reinterpret_cast<uint32_t*>(&val) >> 16);
    }
}

// ---------------------------------------------------------------------------
// ZenDNN matmul benchmark
//
// ZenDNN LoWoHA matmul: C(M,N) = A(M,K) * B(K,N)
// To match GGML semantics where weights are [K, N] (column-major),
// we transpose: C(N,M) = A(N,K) * B(K,M)
// FLOPs = 2 * M * N * K
// ---------------------------------------------------------------------------
BenchResult bench_matmul_zendnn(const OpDesc& desc) {
    const int64_t M = desc.m;
    const int64_t N = desc.n;
    const int64_t K = desc.k;
    const ggml_type ggml_dt = desc.dtype;

    zendnnl::common::data_type_t dt = ggml_type_to_zendnn(ggml_dt);

    // Track timing breakdowns
    auto t_ctx_start = std::chrono::steady_clock::now();

    // 1. Allocate memory for matrices
    // To match GGML: A:[N,K] (weights), B:[K,M] (input), C:[N,M] (output)
    size_t a_size = N * K;
    size_t b_size = K * M;
    size_t c_size = N * M;

    std::vector<char> a_data, b_data, c_data;

    size_t elem_size;
    if (dt == zendnnl::common::data_type_t::f32) {
        elem_size = sizeof(float);
    } else if (dt == zendnnl::common::data_type_t::bf16) {
        elem_size = sizeof(uint16_t);
    } else {
        fprintf(stderr, "error: unsupported data type\n");
        exit(1);
    }

    a_data.resize(a_size * elem_size);
    b_data.resize(b_size * elem_size);
    c_data.resize(c_size * elem_size);
    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_creation_ms = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    // 2. Fill inputs with deterministic data
    auto t_op_start = std::chrono::steady_clock::now();
    void* a_ptr = a_data.data();
    void* b_ptr = b_data.data();
    void* c_ptr = c_data.data();

    if (dt == zendnnl::common::data_type_t::f32) {
        fill_buffer(reinterpret_cast<float*>(a_ptr), a_size, 42);
        fill_buffer(reinterpret_cast<float*>(b_ptr), b_size, 137);
    } else if (dt == zendnnl::common::data_type_t::bf16) {
        fill_buffer_bf16(a_ptr, a_size, 42);
        fill_buffer_bf16(b_ptr, b_size, 137);
    }

    // 3. Setup ZenDNN matmul parameters
    matmul_data_types dtypes;
    dtypes.src = dt;
    dtypes.wei = dt;
    dtypes.dst = dt;
    dtypes.bias = zendnnl::common::data_type_t::none;
    dtypes.compute = dt;

    matmul_params params;
    params.dtypes = dtypes;
    params.num_threads = desc.threads;

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;
    auto t_op_end = std::chrono::steady_clock::now();
    double op_creation_ms = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    // 4. Warmup
    for (int i = 0; i < desc.warmup; i++) {
        status_t status = matmul_direct(
            'r',                      // row-major
            false,                    // don't transpose src
            true,                     // transpose weights
            M,                        // M: rows of B and C
            N,                        // N: cols of A^T and C
            K,                        // K: cols of B, rows of A
            1.0f,                     // alpha
            b_ptr,                    // src: B[M, K]
            K,                        // lda
            a_ptr,                    // weights: A[N, K] (will be transposed)
            K,                        // ldb
            nullptr,                  // bias
            0.0f,                     // beta
            c_ptr,                    // output C[M, N]
            N,                        // ldc
            true,                     // is_weights_const
            batch_params,
            params
        );

        if (status != status_t::success) {
            fprintf(stderr, "error: ZenDNN matmul warmup failed: status=%d\n",
                   static_cast<int>(status));
            exit(1);
        }
    }

    // 5. Timed iterations
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    for (int i = 0; i < desc.repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();

        status_t status = matmul_direct(
            'r', false, true,
            M, N, K,
            1.0f,
            b_ptr, K,
            a_ptr, K,
            nullptr,
            0.0f,
            c_ptr, N,
            true,
            batch_params,
            params
        );

        auto t1 = std::chrono::steady_clock::now();

        if (status != status_t::success) {
            fprintf(stderr, "error: ZenDNN matmul failed: status=%d\n",
                   static_cast<int>(status));
            exit(1);
        }

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / desc.repeats;

    // TFLOPS = 2*M*N*K / avg_time_s / 1e12
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    BenchResult result;
    result.min_ms = min_ms;
    result.avg_ms = avg_ms;
    result.max_ms = max_ms;
    result.tflops = tflops;
    result.ctx_creation_ms = ctx_creation_ms;
    result.op_creation_ms = op_creation_ms;
    result.op_execution_ms = avg_ms;  // Per-iteration average
    result.other_ms = 0.0;

    return result;
}
