#include "ggml_matmul_bench.h"
#include "ggml_utils.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstring>

// ---------------------------------------------------------------------------
// ggml_mul_mat benchmark
//
// GGML semantics for ggml_mul_mat(A, B):
//   A  : [K, M]    (weight matrix, transposed internally)
//   B  : [K, N]    (input matrix)
//   out: [M, N]
//
// This matches the standard GEMM: C(M,N) = A^T(M,K) * B(K,N)
// FLOPs = 2 * M * N * K
// ---------------------------------------------------------------------------
BenchResult bench_matmul_ggml(const OpDesc& desc) {
    const int64_t M = desc.m;
    const int64_t N = desc.n;
    const int64_t K = desc.k;
    const ggml_type src_dtype = desc.src_dtype;  // f32 or bf16
    const ggml_type wei_dtype = desc.wei_dtype;  // f32, f16, bf16, q8_0, q4_0

    // Track timing breakdowns
    auto t_start = std::chrono::steady_clock::now();

    // 1. Init CPU backend
    auto t_ctx_start = std::chrono::steady_clock::now();
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        exit(1);
    }
    ggml_backend_cpu_set_n_threads(backend, desc.threads);

    // 2. Create GGML context (no_alloc -- backend allocates tensor data)
    const size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params params = {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "error: failed to init ggml context\n");
        exit(1);
    }
    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_creation_ms = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    // 3. Create tensors
    //    A: weight [K, M], B: input [K, N]
    auto t_op_start = std::chrono::steady_clock::now();
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, wei_dtype, K, M);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, src_dtype, K, N);
    ggml_set_name(a, "A");
    ggml_set_name(b, "B");
    ggml_set_input(a);
    ggml_set_input(b);

    // 4. Build graph: single mul_mat node (output is F32)
    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    ggml_set_name(c, "C");
    ggml_set_output(c);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);

    // 5. Allocate tensor buffers via graph allocator
    // For q4_0, try to use repack buffer type for q4_0x8 optimization
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    bool using_repack = false;

    if (wei_dtype == GGML_TYPE_Q4_0) {
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
        fprintf(stderr, "error: graph allocation failed\n");
        exit(1);
    }

    // 6. Fill inputs with deterministic data
    // Use backend_set for repack buffer to trigger q4_0 → q4_0x8 conversion
    fill_tensor_deterministic(a, 42, using_repack);
    fill_tensor_deterministic(b, 137, false);  // b is always F32, no repack needed
    auto t_op_end = std::chrono::steady_clock::now();
    double op_creation_ms = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    // 7. Warmup
    for (int i = 0; i < desc.warmup; i++) {
        ggml_backend_graph_compute(backend, graph);
    }

    // 8. Timed iterations
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    for (int i = 0; i < desc.repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, graph);
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / desc.repeats;

    // TFLOPS = 2*M*N*K / avg_time_s / 1e12
    double flops = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    // 9. Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(backend);

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

// ---------------------------------------------------------------------------
// ggml_mul_mat_id benchmark (MoE expert-routed matmul)
//
// GGML ggml_mul_mat_id tensor layout (from ggml.c):
//   as  : [K, M, n_experts]            stacked expert weight matrices (3D)
//   b   : [K, n_experts_used, N]       input (3D; ne[2]=N must match ids ne[1])
//   ids : [n_experts_used, N]           routing indices (2D, i32)
//   out : [M, n_experts_used, N]
//
// FLOPs approx = 2 * M * K * n_experts_used * N
// ---------------------------------------------------------------------------
BenchResult bench_matmul_id_ggml(const OpDesc& desc) {
    const int64_t M = desc.m;
    const int64_t N = desc.n;           // number of tokens
    const int64_t K = desc.k;
    const int64_t n_exp = desc.n_experts;
    const int64_t n_used = desc.n_experts_used;
    const ggml_type src_dtype = desc.src_dtype;  // f32 or bf16
    const ggml_type wei_dtype = desc.wei_dtype;  // f32, f16, bf16, q8_0, q4_0

    // Track timing breakdowns
    auto t_ctx_start = std::chrono::steady_clock::now();
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        exit(1);
    }
    ggml_backend_cpu_set_n_threads(backend, desc.threads);

    const size_t ctx_size = 5 * ggml_tensor_overhead() + ggml_graph_overhead();
    struct ggml_init_params params = {};
    params.mem_size   = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = true;

    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "error: failed to init ggml context\n");
        exit(1);
    }
    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_creation_ms = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    // Operator creation timing
    auto t_op_start = std::chrono::steady_clock::now();
    // Expert weight matrices stacked: [K, M, n_experts]
    struct ggml_tensor* as = ggml_new_tensor_3d(ctx, wei_dtype, K, M, n_exp);
    ggml_set_name(as, "experts");
    ggml_set_input(as);

    // Input: [K, n_experts_used, N]  (3D -- ne[2]=N must match ids ne[1])
    struct ggml_tensor* b = ggml_new_tensor_3d(ctx, src_dtype, K, n_used, N);
    ggml_set_name(b, "input");
    ggml_set_input(b);

    // Routing indices: [n_experts_used, N] (2D, i32)
    struct ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, N);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    // Build graph (output is F32)
    struct ggml_tensor* c = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_set_name(c, "C");
    ggml_set_output(c);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);

    // For q4_0, try to use repack buffer type for q4_0x8 optimization
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    bool using_repack = false;

    if (wei_dtype == GGML_TYPE_Q4_0) {
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
        fprintf(stderr, "error: graph allocation failed\n");
        exit(1);
    }

    // Fill expert weights and input
    // Use backend_set for repack buffer to trigger q4_0 → q4_0x8 conversion
    fill_tensor_deterministic(as, 42, using_repack);
    fill_tensor_deterministic(b, 137, false);  // b is always F32, no repack needed

    // Fill routing ids: cycle through experts deterministically
    {
        int32_t* id_data = reinterpret_cast<int32_t*>(ids->data);
        for (int64_t token = 0; token < N; token++) {
            for (int64_t e = 0; e < n_used; e++) {
                id_data[token * n_used + e] = static_cast<int32_t>((token + e) % n_exp);
            }
        }
    }
    auto t_op_end = std::chrono::steady_clock::now();
    double op_creation_ms = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    // Warmup
    for (int i = 0; i < desc.warmup; i++) {
        ggml_backend_graph_compute(backend, graph);
    }

    // Timed iterations
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    for (int i = 0; i < desc.repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, graph);
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / desc.repeats;
    double flops = 2.0 * M * K * n_used * N;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(backend);

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
