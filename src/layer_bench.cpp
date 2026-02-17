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
static size_t estimate_weight_bytes(const LayerConfig& cfg, ggml_type dtype) {
    const size_t elem_size = ggml_type_size(dtype);
    size_t total = 0;
    for (const auto& op : cfg.ops) {
        if (op.op_type == "mul_mat") {
            // A: [K, N] weight matrix
            total += (size_t)op.k * op.n * elem_size;
        } else {
            // as: [K, N, n_experts]
            total += (size_t)op.k * op.n * op.n_experts * elem_size;
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
// This biases toward lower-index experts, which is more realistic than
// uniform random for MoE gating.
static void fill_routing_ids(struct ggml_tensor* ids, int n_experts,
                             int n_experts_used, int n_tokens, uint32_t seed) {
    int32_t* data = reinterpret_cast<int32_t*>(ids->data);
    uint32_t state = seed;

    for (int t = 0; t < n_tokens; t++) {
        // Generate n_experts_used unique expert indices per token
        for (int e = 0; e < n_experts_used; e++) {
            // Min-of-two: draw two random experts, keep the lower index
            uint32_t r1 = xorshift32(state) % n_experts;
            uint32_t r2 = xorshift32(state) % n_experts;
            int32_t chosen = static_cast<int32_t>(std::min(r1, r2));

            // Ensure uniqueness within this token's selections
            bool dup = false;
            for (int j = 0; j < e; j++) {
                if (data[t * n_experts_used + j] == chosen) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                // Fallback: sequential offset from last chosen
                chosen = static_cast<int32_t>((chosen + 1) % n_experts);
            }

            data[t * n_experts_used + e] = chosen;
        }
    }
}

LayerBenchResult bench_layer(const LayerConfig& cfg,
                             ggml_type dtype, int threads,
                             int warmup, int repeats) {
    // --- Phase 1: Init CPU backend ---
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        exit(1);
    }
    ggml_backend_cpu_set_n_threads(backend, threads);

    // --- Phase 2: Count tensors needed ---
    // mul_mat: 2 inputs (A, B) + 1 output = 3 tensors
    // mul_mat_id: 3 inputs (as, B, ids) + 1 output = 4 tensors
    size_t n_tensors = 0;
    for (const auto& op : cfg.ops) {
        if (op.op_type == "mul_mat") {
            n_tensors += 3;  // A, B, C
        } else {
            n_tensors += 4;  // as, B, ids, C
        }
    }

    // Print estimated memory
    size_t weight_bytes = estimate_weight_bytes(cfg, dtype);
    fprintf(stderr, "estimated weight memory: %.2f GB\n",
            weight_bytes / (1024.0 * 1024.0 * 1024.0));

    // --- Phase 3: Create GGML context ---
    const size_t ctx_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params params = {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "error: failed to init ggml context\n");
        exit(1);
    }

    // --- Phase 4: Create tensors and build graph ---
    struct ggml_cgraph* graph = ggml_new_graph(ctx);

    // Track tensors that need filling after allocation
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

        // Per-op FLOP calculation
        double flops;

        if (op.op_type == "mul_mat") {
            // A: [K, N] weight, B: [K, M] input, C: [N, M] output
            // GGML convention: ggml_mul_mat(A, B) => C = A^T * B
            // A: [K, N], B: [K, M] => C: [N, M]
            // FLOPs = 2 * M * N * K
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
            // mul_mat_id
            const int64_t n_exp  = op.n_experts;
            const int64_t n_used = op.n_experts_used;

            // as: [K, N, n_experts], B: [K, n_used, M], ids: [n_used, M]
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
                              static_cast<int>(N), static_cast<int>(K), gflops});
        result.total_gflops += gflops;
    }

    // --- Phase 5: Allocate buffers ---
    ggml_gallocr_t allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocr, graph)) {
        fprintf(stderr, "error: graph allocation failed\n");
        exit(1);
    }

    // --- Phase 6: Fill tensors ---
    for (const auto& fi : fill_list) {
        if (fi.is_routing_ids) {
            fill_routing_ids(fi.tensor, fi.n_experts, fi.n_experts_used,
                             fi.n_tokens, fi.seed);
        } else {
            fill_tensor_deterministic(fi.tensor, fi.seed);
        }
    }

    // --- Phase 7: Warmup ---
    for (int i = 0; i < warmup; i++) {
        ggml_backend_graph_compute(backend, graph);
    }

    // --- Phase 8: Timed iterations ---
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;

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

    // --- Phase 9: Compute aggregate throughput ---
    result.min_ms = min_ms;
    result.avg_ms = avg_ms;
    result.max_ms = max_ms;
    result.tflops = (result.total_gflops * 1e9) / (avg_ms * 1e-3) / 1e12;

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return result;
}

void print_layer_results(const LayerConfig& cfg, const LayerBenchResult& result,
                         ggml_type dtype, int threads, int warmup, int repeats) {
    printf("op: layer\n");
    printf("model: %s\n", cfg.model_name.c_str());
    printf("dtype: %s\n", dtype_to_string(dtype));
    printf("threads: %d\n", threads);
    printf("warmup: %d\n", warmup);
    printf("repeats: %d\n", repeats);
    printf("\n");

    printf("graph nodes: %zu\n", result.ops.size());
    for (size_t i = 0; i < result.ops.size(); i++) {
        const auto& op = result.ops[i];
        printf("  [%zu] %-12s %-16s m=%-5d n=%-5d k=%-5d (%7.2f GFLOPs)\n",
               i, op.op_type.c_str(), op.label.c_str(),
               op.m, op.n, op.k, op.gflops);
    }
    printf("total: %.2f GFLOPs\n", result.total_gflops);
    printf("\n");
    printf("time(ms): min=%.2f avg=%.2f max=%.2f\n",
           result.min_ms, result.avg_ms, result.max_ms);
    printf("throughput: %.2f TFLOPS\n", result.tflops);
}
