#include "layer_bench.h"
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
#include <cstring>
#include <limits>
#include <vector>

// Estimate total weight memory in bytes for the layer config.
static size_t estimate_weight_bytes_ggml(const LayerConfig& cfg, ggml_type dtype) {
    // For quantized types, ggml_type_size returns bytes per block, not per element.
    // We need to account for the block size.
    const size_t type_size = ggml_type_size(dtype);    // bytes per block
    const int64_t blck_size = ggml_blck_size(dtype);    // elements per block
    const double bytes_per_elem = (double)type_size / blck_size;

    size_t total = 0;
    for (const auto& op : cfg.ops) {
        if (op.op_type == "mul_mat") {
            // A: [K, N] weight matrix
            total += (size_t)((double)op.k * op.n * bytes_per_elem);
        } else {
            // as: [K, N, n_experts]
            total += (size_t)((double)op.k * op.n * op.n_experts * bytes_per_elem);
        }
    }
    return total;
}

// Simple xorshift32 for routing ID generation.
static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Fill routing IDs with a skewed distribution (min-of-two-random-draws).
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

LayerBenchResult bench_layer_ggml(const LayerConfig& cfg,
                                  ggml_type dtype, int threads,
                                  int warmup, int repeats) {
    // Track timing breakdowns
    auto t_ctx_start = std::chrono::steady_clock::now();

    // Init CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "[GGML] error: failed to init CPU backend\n");
        exit(1);
    }
    ggml_backend_cpu_set_n_threads(backend, threads);

    // Count tensors needed
    size_t n_tensors = 0;
    for (const auto& op : cfg.ops) {
        n_tensors += (op.op_type == "mul_mat") ? 3 : 4;
    }

    // Print estimated memory
    size_t weight_bytes = estimate_weight_bytes_ggml(cfg, dtype);
    fprintf(stderr, "[GGML] estimated weight memory: %.2f GB\n",
            weight_bytes / (1024.0 * 1024.0 * 1024.0));

    // Create GGML context
    const size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params params = {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "[GGML] error: failed to init ggml context\n");
        exit(1);
    }

    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_creation_ms = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    // Build graph
    auto t_op_start = std::chrono::steady_clock::now();
    struct ggml_cgraph* graph = ggml_new_graph(ctx);

    struct TensorFillInfo {
        struct ggml_tensor* tensor;
        uint32_t seed;
        bool is_routing_ids;
        int n_experts, n_experts_used, n_tokens;
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
            struct ggml_tensor* a = ggml_new_tensor_2d(ctx, dtype, K, N);
            struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);

            char name_a[64], name_b[64], name_c[64];
            snprintf(name_a, sizeof(name_a), "%s_A", op.label.c_str());
            snprintf(name_b, sizeof(name_b), "%s_B", op.label.c_str());
            snprintf(name_c, sizeof(name_c), "%s_C", op.label.c_str());
            ggml_set_name(a, name_a);
            ggml_set_name(b, name_b);
            ggml_set_input(a);
            ggml_set_input(b);

            struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
            ggml_set_name(c, name_c);
            ggml_set_output(c);
            ggml_build_forward_expand(graph, c);

            fill_list.push_back({a, seed_counter++, false, 0, 0, 0});
            fill_list.push_back({b, seed_counter++, false, 0, 0, 0});

            flops = 2.0 * M * N * K;
        } else {
            const int64_t n_exp  = op.n_experts;
            const int64_t n_used = op.n_experts_used;

            struct ggml_tensor* as  = ggml_new_tensor_3d(ctx, dtype, K, N, n_exp);
            struct ggml_tensor* b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_used, M);
            struct ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, M);

            char name_as[64], name_b[64], name_ids[64], name_c[64];
            snprintf(name_as,  sizeof(name_as),  "%s_experts", op.label.c_str());
            snprintf(name_b,   sizeof(name_b),   "%s_B",       op.label.c_str());
            snprintf(name_ids, sizeof(name_ids), "%s_ids",     op.label.c_str());
            snprintf(name_c,   sizeof(name_c),   "%s_C",       op.label.c_str());
            ggml_set_name(as, name_as);
            ggml_set_name(b, name_b);
            ggml_set_name(ids, name_ids);
            ggml_set_input(as);
            ggml_set_input(b);
            ggml_set_input(ids);

            struct ggml_tensor* c = ggml_mul_mat_id(ctx, as, b, ids);
            ggml_set_name(c, name_c);
            ggml_set_output(c);
            ggml_build_forward_expand(graph, c);

            fill_list.push_back({as, seed_counter++, false, 0, 0, 0});
            fill_list.push_back({b,  seed_counter++, false, 0, 0, 0});
            fill_list.push_back({ids, seed_counter++, true,
                                 static_cast<int>(n_exp),
                                 static_cast<int>(n_used),
                                 static_cast<int>(M)});

            flops = 2.0 * M * N * K * n_used;
        }

        double gflops = flops / 1e9;
        result.ops.push_back({op.op_type, op.label, static_cast<int>(M),
                              static_cast<int>(N), static_cast<int>(K), gflops,
                              0.0, 0.0, 0.0, 0.0});
        result.total_gflops += gflops;
    }

    // Allocate buffers - for q4_0, try to use repack buffer type for q4_0x8 optimization
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    bool using_repack = false;

    if (dtype == GGML_TYPE_Q4_0) {
        ggml_backend_dev_t cpu_dev = ggml_backend_get_device(backend);
        if (cpu_dev) {
            ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
            if (cpu_reg) {
                typedef ggml_backend_buffer_type_t * (*ggml_backend_dev_get_extra_bufts_t)(ggml_backend_dev_t);
                auto get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t)
                    ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

                if (get_extra_bufts) {
                    ggml_backend_buffer_type_t * extra_bufts = get_extra_bufts(cpu_dev);
                    if (extra_bufts && *extra_bufts) {
                        buft = *extra_bufts;  // Use first extra buffer type (repack)
                        using_repack = true;
                    }
                }
            }
        }
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(allocr, graph)) {
        fprintf(stderr, "[GGML] error: graph allocation failed\n");
        exit(1);
    }

    // Fill tensors
    // Use backend_set for repack buffer to trigger q4_0 → q4_0x8 conversion
    for (const auto& fi : fill_list) {
        if (fi.is_routing_ids) {
            fill_routing_ids(fi.tensor, fi.n_experts, fi.n_experts_used,
                             fi.n_tokens, fi.seed);
        } else {
            // Use backend_set for weight tensors (type matches dtype) when using repack
            bool is_weight = (fi.tensor->type == dtype);
            fill_tensor_deterministic(fi.tensor, fi.seed, using_repack && is_weight);
        }
    }

    auto t_op_end = std::chrono::steady_clock::now();
    double op_creation_ms = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    // Warmup
    for (int i = 0; i < warmup; i++) {
        ggml_backend_graph_compute(backend, graph);
    }

    // Timed iterations - measure total and per-op
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    // Per-op timing accumulators
    std::vector<std::vector<double>> op_times(cfg.ops.size());

    for (int i = 0; i < repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, graph);
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / repeats;

    // Compute aggregate throughput
    result.min_ms = min_ms;
    result.avg_ms = avg_ms;
    result.max_ms = max_ms;
    result.tflops = (result.total_gflops * 1e9) / (avg_ms * 1e-3) / 1e12;

    // Set timing breakdowns (all per-iteration averages)
    result.ctx_creation_ms = ctx_creation_ms;
    result.op_creation_ms = op_creation_ms;
    result.op_execution_ms = avg_ms;  // Per-iteration average
    result.other_ms = 0.0;

    // Estimate per-op times (proportional to GFLOPs)
    for (size_t i = 0; i < result.ops.size(); i++) {
        double proportion = result.ops[i].gflops / result.total_gflops;
        result.ops[i].avg_ms = avg_ms * proportion;
        result.ops[i].min_ms = min_ms * proportion;
        result.ops[i].max_ms = max_ms * proportion;
        result.ops[i].tflops = (result.ops[i].gflops * 1e9) / (result.ops[i].avg_ms * 1e-3) / 1e12;
    }

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return result;
}
