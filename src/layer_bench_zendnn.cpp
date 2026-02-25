#ifdef ENABLE_ZENDNN

#include "layer_bench.h"
#include "ggml_utils.h"
#include "zendnnl.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using namespace zendnnl::lowoha::matmul;

// Helper: Convert ggml_type to zendnnl::common::data_type_t
static zendnnl::common::data_type_t ggml_type_to_zendnn_layer(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return zendnnl::common::data_type_t::f32;
        case GGML_TYPE_BF16: return zendnnl::common::data_type_t::bf16;
        default:
            fprintf(stderr, "[ZenDNN] error: unsupported ggml_type %d (only f32 and bf16 supported)\n", t);
            exit(1);
    }
}

// Simple xorshift32 for deterministic data
static uint32_t xorshift32_zendnn(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Fill memory with deterministic pseudo-random data
template <typename T>
static void fill_buffer_layer(T* data, size_t n, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t r = xorshift32_zendnn(state);
        float val = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
        data[i] = static_cast<T>(val);
    }
}

static void fill_buffer_bf16_layer(void* data, size_t n, uint32_t seed) {
    uint16_t* ptr = reinterpret_cast<uint16_t*>(data);
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t r = xorshift32_zendnn(state);
        float val = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
        ptr[i] = static_cast<uint16_t>(*reinterpret_cast<uint32_t*>(&val) >> 16);
    }
}

// Estimate total weight memory in bytes for the layer config.
static size_t estimate_weight_bytes_zendnn(const LayerConfig& cfg, ggml_type dtype) {
    const size_t elem_size = ggml_type_size(dtype);
    size_t total = 0;
    for (const auto& op : cfg.ops) {
        if (op.op_type == "mul_mat") {
            // A: [N, K] weight matrix (to match GGML's [K, N] transposed)
            total += (size_t)op.n * op.k * elem_size;
        } else {
            fprintf(stderr, "[ZenDNN] error: mul_mat_id not supported (MoE models)\n");
            exit(1);
        }
    }
    return total;
}

LayerBenchResult bench_layer_zendnn(const LayerConfig& cfg,
                                    ggml_type wei_dtype, int threads,
                                    int warmup, int repeats,
                                    ggml_type src_dtype) {
    // Track timing breakdowns
    auto t_ctx_start = std::chrono::steady_clock::now();

    // Validate: no mul_mat_id ops allowed
    for (const auto& op : cfg.ops) {
        if (op.op_type != "mul_mat") {
            fprintf(stderr, "[ZenDNN] error: only supports non-MoE models (mul_mat)\n");
            fprintf(stderr, "         Found '%s' op. Use GGML backend for MoE models.\n", op.op_type.c_str());
            exit(1);
        }
    }

    zendnnl::common::data_type_t src_dt = ggml_type_to_zendnn_layer(src_dtype);
    zendnnl::common::data_type_t wei_dt = ggml_type_to_zendnn_layer(wei_dtype);
    zendnnl::common::data_type_t dst_dt = zendnnl::common::data_type_t::f32;  // Output always f32

    // Print estimated memory
    size_t weight_bytes = estimate_weight_bytes_zendnn(cfg, wei_dtype);
    fprintf(stderr, "[ZenDNN] estimated weight memory: %.2f GB\n",
            weight_bytes / (1024.0 * 1024.0 * 1024.0));

    // Determine element sizes
    size_t wei_elem_size = (wei_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);
    size_t src_elem_size = (src_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);
    size_t dst_elem_size = sizeof(float);  // Output always f32

    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_creation_ms = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    // Allocate memory for all operations
    auto t_op_start = std::chrono::steady_clock::now();

    struct OpData {
        std::vector<char> a_data;  // weights [N, K]
        std::vector<char> b_data;  // input [M, K]
        std::vector<char> c_data;  // output [M, N]
        int64_t M, N, K;
        double gflops;
        std::string op_type;
        std::string label;
    };

    std::vector<OpData> ops;
    LayerBenchResult result;
    uint32_t seed_counter = 42;

    // Create memory and fill data for each operation
    for (const auto& op : cfg.ops) {
        const int64_t M = op.m;
        const int64_t N = op.n;
        const int64_t K = op.k;

        OpData op_data;
        op_data.M = M;
        op_data.N = N;
        op_data.K = K;
        op_data.op_type = op.op_type;
        op_data.label = op.label;

        size_t a_size = N * K;
        size_t b_size = M * K;
        size_t c_size = M * N;

        op_data.a_data.resize(a_size * wei_elem_size);  // weights
        op_data.b_data.resize(b_size * src_elem_size);  // source/input
        op_data.c_data.resize(c_size * dst_elem_size);  // output

        // Fill inputs with deterministic data
        void* a_ptr = op_data.a_data.data();
        void* b_ptr = op_data.b_data.data();

        // Fill weights (a)
        if (wei_dt == zendnnl::common::data_type_t::f32) {
            fill_buffer_layer(reinterpret_cast<float*>(a_ptr), a_size, seed_counter++);
        } else if (wei_dt == zendnnl::common::data_type_t::bf16) {
            fill_buffer_bf16_layer(a_ptr, a_size, seed_counter++);
        }

        // Fill source/input (b)
        if (src_dt == zendnnl::common::data_type_t::f32) {
            fill_buffer_layer(reinterpret_cast<float*>(b_ptr), b_size, seed_counter++);
        } else if (src_dt == zendnnl::common::data_type_t::bf16) {
            fill_buffer_bf16_layer(b_ptr, b_size, seed_counter++);
        }

        double flops = 2.0 * M * N * K;
        double gflops = flops / 1e9;
        op_data.gflops = gflops;

        ops.push_back(std::move(op_data));

        result.ops.push_back({op.op_type, op.label,
                              static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                              gflops, 0.0, 0.0, 0.0, 0.0});
        result.total_gflops += gflops;
    }

    // Setup ZenDNN matmul parameters (common for all ops)
    matmul_data_types dtypes;
    dtypes.src = src_dt;
    dtypes.wei = wei_dt;
    dtypes.dst = dst_dt;  // Always f32
    dtypes.bias = zendnnl::common::data_type_t::none;
    dtypes.compute = zendnnl::common::data_type_t::f32;  // Compute in f32

    matmul_params params;
    params.dtypes = dtypes;
    params.num_threads = threads;

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    auto t_op_end = std::chrono::steady_clock::now();
    double op_creation_ms = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    // Warmup: execute all ops sequentially
    for (int w = 0; w < warmup; w++) {
        for (auto& op : ops) {
            status_t status = matmul_direct(
                'r', false, true,
                op.M, op.N, op.K,
                1.0f,
                op.b_data.data(), op.K,
                op.a_data.data(), op.K,
                nullptr, 0.0f,
                op.c_data.data(), op.N,
                true,
                batch_params,
                params
            );

            if (status != status_t::success) {
                fprintf(stderr, "[ZenDNN] error: warmup failed for op %s: status=%d\n",
                       op.label.c_str(), static_cast<int>(status));
                exit(1);
            }
        }
    }

    // Timed iterations: measure total and per-op times
    double total_min_ms = std::numeric_limits<double>::max();
    double total_max_ms = 0.0;
    double total_sum_ms = 0.0;

    // Per-op timing storage
    std::vector<std::vector<double>> op_times(ops.size());

    for (int r = 0; r < repeats; r++) {
        auto iter_t0 = std::chrono::steady_clock::now();

        // Execute and time each operation individually
        for (size_t i = 0; i < ops.size(); i++) {
            auto& op = ops[i];

            auto op_t0 = std::chrono::steady_clock::now();

            status_t status = matmul_direct(
                'r', false, true,
                op.M, op.N, op.K,
                1.0f,
                op.b_data.data(), op.K,
                op.a_data.data(), op.K,
                nullptr, 0.0f,
                op.c_data.data(), op.N,
                true,
                batch_params,
                params
            );

            auto op_t1 = std::chrono::steady_clock::now();

            if (status != status_t::success) {
                fprintf(stderr, "[ZenDNN] error: matmul failed for op %s: status=%d\n",
                       op.label.c_str(), static_cast<int>(status));
                exit(1);
            }

            double op_ms = std::chrono::duration<double, std::milli>(op_t1 - op_t0).count();
            op_times[i].push_back(op_ms);
        }

        auto iter_t1 = std::chrono::steady_clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(iter_t1 - iter_t0).count();
        total_min_ms = std::min(total_min_ms, iter_ms);
        total_max_ms = std::max(total_max_ms, iter_ms);
        total_sum_ms += iter_ms;
    }

    double total_avg_ms = total_sum_ms / repeats;

    // Compute per-op statistics
    for (size_t i = 0; i < result.ops.size(); i++) {
        const auto& times = op_times[i];
        double min_ms = *std::min_element(times.begin(), times.end());
        double max_ms = *std::max_element(times.begin(), times.end());
        double sum_ms = 0.0;
        for (double t : times) sum_ms += t;
        double avg_ms = sum_ms / repeats;

        result.ops[i].min_ms = min_ms;
        result.ops[i].avg_ms = avg_ms;
        result.ops[i].max_ms = max_ms;
        result.ops[i].tflops = (result.ops[i].gflops * 1e9) / (avg_ms * 1e-3) / 1e12;
    }

    // Aggregate results
    result.min_ms = total_min_ms;
    result.avg_ms = total_avg_ms;
    result.max_ms = total_max_ms;
    result.tflops = (result.total_gflops * 1e9) / (total_avg_ms * 1e-3) / 1e12;

    // Set timing breakdowns (all per-iteration averages)
    result.ctx_creation_ms = ctx_creation_ms;
    result.op_creation_ms = op_creation_ms;
    result.op_execution_ms = total_avg_ms;  // Per-iteration average
    result.other_ms = 0.0;

    return result;
}

#endif // ENABLE_ZENDNN
