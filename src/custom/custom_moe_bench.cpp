#include "custom_moe_bench.h"
#include "custom_moe.h"
#include "ggml_utils.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers (same as layer_bench.cpp)
// ---------------------------------------------------------------------------

static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static void fill_routing_ids(struct ggml_tensor* ids, int n_experts,
                             int n_experts_used, int n_tokens, uint32_t seed) {
    int32_t* data = reinterpret_cast<int32_t*>(ids->data);
    uint32_t state = seed;

    for (int t = 0; t < n_tokens; t++) {
        for (int e = 0; e < n_experts_used; e++) {
            uint32_t r1 = xorshift32(state) % n_experts;
            uint32_t r2 = xorshift32(state) % n_experts;
            int32_t chosen = static_cast<int32_t>(std::min(r1, r2));

            bool dup = false;
            for (int j = 0; j < e; j++) {
                if (data[t * n_experts_used + j] == chosen) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                chosen = static_cast<int32_t>((chosen + 1) % n_experts);
            }
            data[t * n_experts_used + e] = chosen;
        }
    }
}

// ---------------------------------------------------------------------------
// bench_custom_moe -- benchmark standalone custom MoE op
// ---------------------------------------------------------------------------
BenchResult bench_custom_moe(const OpDesc& desc) {
    const int64_t M = desc.m;
    const int64_t N = desc.n;
    const int64_t K = desc.k;
    const int64_t n_exp = desc.n_experts;
    const int64_t n_used = desc.n_experts_used;
    const ggml_type dtype = desc.dtype;

    // CPU backend for buffer allocation only
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        exit(1);
    }

    const size_t ctx_size = 4 * ggml_tensor_overhead();
    struct ggml_init_params params = {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "error: failed to init ggml context\n");
        exit(1);
    }

    // Create tensors (same layout as ggml_mul_mat_id)
    struct ggml_tensor* experts = ggml_new_tensor_3d(ctx, dtype, K, M, n_exp);
    struct ggml_tensor* input   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_used, N);
    struct ggml_tensor* ids     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, N);
    struct ggml_tensor* dst     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, M, n_used, N);

    // Allocate all tensors in the context via backend buffer
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "error: buffer allocation failed\n");
        exit(1);
    }

    // Fill inputs
    fill_tensor_deterministic(experts, 42);
    fill_tensor_deterministic(input, 137);
    {
        int32_t* id_data = reinterpret_cast<int32_t*>(ids->data);
        for (int64_t t = 0; t < N; t++) {
            for (int64_t e = 0; e < n_used; e++) {
                id_data[t * n_used + e] = static_cast<int32_t>((t + e) % n_exp);
            }
        }
    }

    // Warmup
    for (int i = 0; i < desc.warmup; i++) {
        custom_moe_compute(dst, experts, input, ids, desc.threads);
    }

    // Timed iterations
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    for (int i = 0; i < desc.repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();
        custom_moe_compute(dst, experts, input, ids, desc.threads);
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / desc.repeats;
    double flops = 2.0 * M * K * n_used * N;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return BenchResult{min_ms, avg_ms, max_ms, tflops};
}

// ---------------------------------------------------------------------------
// bench_custom_layer -- full layer using custom matmul + custom MoE
// ---------------------------------------------------------------------------

static size_t estimate_weight_bytes(const LayerConfig& cfg, ggml_type dtype) {
    const size_t elem_size = ggml_type_size(dtype);
    size_t total = 0;
    for (const auto& op : cfg.ops) {
        if (op.op_type == "mul_mat") {
            total += (size_t)op.k * op.n * elem_size;
        } else {
            total += (size_t)op.k * op.n * op.n_experts * elem_size;
        }
    }
    return total;
}

LayerBenchResult bench_custom_layer(const LayerConfig& cfg,
                                    ggml_type dtype, int threads,
                                    int warmup, int repeats) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        exit(1);
    }

    // Count tensors needed
    size_t n_tensors = 0;
    for (const auto& op : cfg.ops) {
        if (op.op_type == "mul_mat") {
            n_tensors += 3;  // A, B, dst
        } else {
            n_tensors += 4;  // experts, B, ids, dst
        }
    }

    size_t weight_bytes = estimate_weight_bytes(cfg, dtype);
    fprintf(stderr, "estimated weight memory: %.2f GB\n",
            weight_bytes / (1024.0 * 1024.0 * 1024.0));

    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    struct ggml_init_params params = {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "error: failed to init ggml context\n");
        exit(1);
    }

    // Track ops for custom dispatch
    struct CustomOp {
        enum { MATMUL, MOE } type;
        struct ggml_tensor* dst;
        struct ggml_tensor* src0;  // A or experts
        struct ggml_tensor* src1;  // B
        struct ggml_tensor* ids;   // only for MOE
    };
    std::vector<CustomOp> custom_ops;

    struct TensorFillInfo {
        struct ggml_tensor* tensor;
        uint32_t seed;
        bool is_routing_ids;
        int n_experts;
        int n_experts_used;
        int n_tokens;
    };
    std::vector<TensorFillInfo> fill_list;

    LayerBenchResult result;
    uint32_t seed_counter = 42;

    for (size_t i = 0; i < cfg.ops.size(); i++) {
        const auto& op = cfg.ops[i];
        const int64_t M = op.m;
        const int64_t N = op.n;
        const int64_t K = op.k;
        double flops;

        if (op.op_type == "mul_mat") {
            struct ggml_tensor* a   = ggml_new_tensor_2d(ctx, dtype, K, N);
            struct ggml_tensor* b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            struct ggml_tensor* d   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);

            fill_list.push_back({a, seed_counter++, false, 0, 0, 0});
            fill_list.push_back({b, seed_counter++, false, 0, 0, 0});

            custom_ops.push_back({CustomOp::MATMUL, d, a, b, nullptr});
            flops = 2.0 * M * N * K;
        } else {
            const int64_t n_exp  = op.n_experts;
            const int64_t n_used = op.n_experts_used;

            struct ggml_tensor* exp = ggml_new_tensor_3d(ctx, dtype, K, N, n_exp);
            struct ggml_tensor* b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_used, M);
            struct ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, M);
            struct ggml_tensor* d   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N, n_used, M);

            fill_list.push_back({exp, seed_counter++, false, 0, 0, 0});
            fill_list.push_back({b,   seed_counter++, false, 0, 0, 0});
            fill_list.push_back({ids, seed_counter++, true,
                                 static_cast<int>(n_exp),
                                 static_cast<int>(n_used),
                                 static_cast<int>(M)});

            custom_ops.push_back({CustomOp::MOE, d, exp, b, ids});
            flops = 2.0 * M * N * K * n_used;
        }

        double gflops = flops / 1e9;
        result.ops.push_back({op.op_type, op.label, static_cast<int>(M),
                              static_cast<int>(N), static_cast<int>(K), gflops});
        result.total_gflops += gflops;
    }

    // Allocate all tensors via backend buffer
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "error: buffer allocation failed\n");
        exit(1);
    }

    // Fill tensors
    for (const auto& fi : fill_list) {
        if (fi.is_routing_ids) {
            fill_routing_ids(fi.tensor, fi.n_experts, fi.n_experts_used,
                             fi.n_tokens, fi.seed);
        } else {
            fill_tensor_deterministic(fi.tensor, fi.seed);
        }
    }

    // Lambda: run all custom ops sequentially
    auto run_all = [&]() {
        for (const auto& cop : custom_ops) {
            if (cop.type == CustomOp::MATMUL) {
                custom_matmul_compute(cop.dst, cop.src0, cop.src1, threads);
            } else {
                custom_moe_compute(cop.dst, cop.src0, cop.src1, cop.ids, threads);
            }
        }
    };

    // Warmup
    for (int i = 0; i < warmup; i++) {
        run_all();
    }

    // Timed iterations
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    for (int i = 0; i < repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();
        run_all();
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / repeats;
    result.min_ms = min_ms;
    result.avg_ms = avg_ms;
    result.max_ms = max_ms;
    result.tflops = (result.total_gflops * 1e9) / (avg_ms * 1e-3) / 1e12;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return result;
}
