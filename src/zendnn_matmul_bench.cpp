#include "zendnn_matmul_bench.h"
#include "ggml_utils.h"
#include "routing_utils.h"

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

// Specialization for bf16 (matches GGML's ggml_compute_fp32_to_bf16 with rounding)
static void fill_buffer_bf16(void* data, size_t n, uint32_t seed) {
    uint16_t* ptr = reinterpret_cast<uint16_t*>(data);
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t r = xorshift32(state);
        float val = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
        uint32_t u = *reinterpret_cast<uint32_t*>(&val);

        // Match GGML's fp32_to_bf16 conversion with round-to-nearest-even
        if ((u & 0x7fffffff) > 0x7f800000) { // NaN
            ptr[i] = (u >> 16) | 64; // force to quiet NaN
        } else {
            // Round to nearest even: add 0x7fff plus LSB of result for tie-breaking
            ptr[i] = static_cast<uint16_t>((u + (0x7fff + ((u >> 16) & 1))) >> 16);
        }
    }
}

// ---------------------------------------------------------------------------
// ZenDNN matmul benchmark
//
// Formula: C(N,M) = A(N,K) × B_t(K,M) where N=tokens, M=features
// Following matmul_id pattern: output[N,M] = input[N,K] × weights[K,M]
// Weights stored as [M,K] and transposed by ZenDNN to [K,M]
// Memory: weights[M,K], input[N,K], output[N,M]
// FLOPs = 2 * M * N * K
// ---------------------------------------------------------------------------
BenchResult bench_matmul_zendnn(const OpDesc& desc) {
    const int64_t M = desc.m;
    const int64_t N = desc.n;
    const int64_t K = desc.k;
    const ggml_type src_ggml = desc.src_dtype;  // f32 or bf16
    const ggml_type wei_ggml = desc.wei_dtype;  // f32 or bf16

    zendnnl::common::data_type_t src_dt = ggml_type_to_zendnn(src_ggml);
    zendnnl::common::data_type_t wei_dt = ggml_type_to_zendnn(wei_ggml);
    zendnnl::common::data_type_t dst_dt = zendnnl::common::data_type_t::f32;  // Output always f32

    // Track timing breakdowns
    auto t_ctx_start = std::chrono::steady_clock::now();

    // 1. Allocate memory for matrices
    // Following matmul_id pattern: output[N,M] = input[N,K] × weights[K,M]
    // Where N = desc.n (tokens), M = desc.m (features), K = desc.k
    // Weights stored as [M,K] and transposed, so allocate M*K
    size_t a_size = M * K;  // weights: [M, K] will be transposed
    size_t b_size = N * K;  // input: [N, K]
    size_t c_size = N * M;  // output: [N, M]

    std::vector<char> a_data, b_data, c_data;

    size_t wei_elem_size = (wei_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);
    size_t src_elem_size = (src_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);
    size_t dst_elem_size = sizeof(float);  // Output always f32

    a_data.resize(a_size * wei_elem_size);  // weights
    b_data.resize(b_size * src_elem_size);  // source/input
    c_data.resize(c_size * dst_elem_size);  // output
    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_creation_ms = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    // 2. Fill inputs with deterministic data
    auto t_op_start = std::chrono::steady_clock::now();
    void* a_ptr = a_data.data();
    void* b_ptr = b_data.data();
    void* c_ptr = c_data.data();

    // Fill weights (a) - use data_seed for weights
    if (wei_dt == zendnnl::common::data_type_t::f32) {
        fill_buffer(reinterpret_cast<float*>(a_ptr), a_size, desc.data_seed);
    } else if (wei_dt == zendnnl::common::data_type_t::bf16) {
        fill_buffer_bf16(a_ptr, a_size, desc.data_seed);
    }

    // Fill source/input (b) - use data_seed + 95 for inputs (to get different values than weights)
    if (src_dt == zendnnl::common::data_type_t::f32) {
        fill_buffer(reinterpret_cast<float*>(b_ptr), b_size, desc.data_seed + 95);
    } else if (src_dt == zendnnl::common::data_type_t::bf16) {
        fill_buffer_bf16(b_ptr, b_size, desc.data_seed + 95);
    }

    // 3. Setup ZenDNN matmul parameters
    matmul_data_types dtypes;
    dtypes.src = src_dt;
    dtypes.wei = wei_dt;
    dtypes.dst = dst_dt;  // Always f32
    dtypes.bias = zendnnl::common::data_type_t::none;
    dtypes.compute = zendnnl::common::data_type_t::f32;  // Compute in f32

    matmul_params params;
    params.dtypes = dtypes;
    params.num_threads = desc.threads;

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;
    auto t_op_end = std::chrono::steady_clock::now();
    double op_creation_ms = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    // 4. Warmup
    // Following matmul_id pattern: output[N,M] = input[N,K] × weights[K,M]
    // Weights stored as [M,K], transposed inside ZenDNN
    for (int i = 0; i < desc.warmup; i++) {
        status_t status = matmul_direct(
            'r',                      // row-major
            false,                    // don't transpose input
            true,                     // transpose weights [M,K] -> [K,M]
            N,                        // M: rows of input and output (tokens)
            M,                        // N: cols of output (features)
            K,                        // K: inner dimension
            1.0f,                     // alpha
            b_ptr,                    // src: input[N, K]
            K,                        // lda: leading dim of input
            a_ptr,                    // weights: [M, K] (will be transposed)
            K,                        // ldb: leading dim of weights
            nullptr,                  // bias
            0.0f,                     // beta
            c_ptr,                    // output: [N, M]
            M,                        // ldc: leading dim of output
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
            N, M, K,
            1.0f,
            b_ptr, K,
            a_ptr, K,
            nullptr,
            0.0f,
            c_ptr, M,
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

    // Copy output for verification if requested
    if (desc.verify_output) {
        size_t output_size = N * M;  // Output: [N, M] where N=tokens, M=features
        result.output_data.resize(output_size);
        const float* output_f32 = reinterpret_cast<const float*>(c_ptr);
        std::copy(output_f32, output_f32 + output_size, result.output_data.begin());
    }

    return result;
}
// ---------------------------------------------------------------------------
// ZenDNN matmul_id benchmark (MoE expert-routed matmul)
//
// Matches GGML's routing strategy:
// 1. Group (token, expert_slot) pairs by the expert they route to
// 2. For each expert, process all routed tokens in a single batched GEMM
// 3. Use group_gemm_direct with one operation per expert (parallel mode)
//
// FLOPs approx = 2 * M * K * n_experts_used * N
// ---------------------------------------------------------------------------
BenchResult bench_matmul_id_zendnn(const OpDesc& desc) {
    const int64_t M = desc.m;
    const int64_t N = desc.n;           // number of tokens
    const int64_t K = desc.k;
    const int64_t n_exp = desc.n_experts;
    const int64_t n_used = desc.n_experts_used;
    const ggml_type src_ggml = desc.src_dtype;  // f32 or bf16
    const ggml_type wei_ggml = desc.wei_dtype;  // f32 or bf16

    zendnnl::common::data_type_t src_dt = ggml_type_to_zendnn(src_ggml);
    zendnnl::common::data_type_t wei_dt = ggml_type_to_zendnn(wei_ggml);
    zendnnl::common::data_type_t dst_dt = zendnnl::common::data_type_t::f32;

    // Track timing breakdowns
    auto t_ctx_start = std::chrono::steady_clock::now();

    // 1. Allocate memory
    // Expert weights: [K, M, n_experts]
    size_t expert_size = K * M;
    size_t input_size = K * n_used * N;
    size_t output_size = M * n_used * N;
    size_t ids_size = n_used * N;

    size_t wei_elem_size = (wei_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);
    size_t src_elem_size = (src_dt == zendnnl::common::data_type_t::f32) ? sizeof(float) : sizeof(uint16_t);
    size_t dst_elem_size = sizeof(float);

    std::vector<char> expert_data(expert_size * n_exp * wei_elem_size);
    std::vector<char> input_data(input_size * src_elem_size);
    std::vector<char> output_data(output_size * dst_elem_size);
    std::vector<int32_t> ids_data(ids_size);

    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_creation_ms = std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    // 2. Fill inputs with deterministic data
    auto t_op_start = std::chrono::steady_clock::now();
    void* expert_ptr = expert_data.data();
    void* input_ptr = input_data.data();
    void* output_ptr = output_data.data();

    // Fill expert weights - use data_seed for weights
    if (wei_dt == zendnnl::common::data_type_t::f32) {
        fill_buffer(reinterpret_cast<float*>(expert_ptr), expert_size * n_exp, desc.data_seed);
    } else if (wei_dt == zendnnl::common::data_type_t::bf16) {
        fill_buffer_bf16(expert_ptr, expert_size * n_exp, desc.data_seed);
    }

    // Fill input - use data_seed + 95 for inputs (to get different values than weights)
    if (src_dt == zendnnl::common::data_type_t::f32) {
        fill_buffer(reinterpret_cast<float*>(input_ptr), input_size, desc.data_seed + 95);
    } else if (src_dt == zendnnl::common::data_type_t::bf16) {
        fill_buffer_bf16(input_ptr, input_size, desc.data_seed + 95);
    }

    // Generate routing ids using shared routing utilities (identical to GGML)
    std::vector<int32_t> routing_ids = generate_routing_ids(
        N, n_exp, n_used,
        desc.expert_token_counts,
        desc.routing_pattern,
        desc.routing_seed
    );
    std::copy(routing_ids.begin(), routing_ids.end(), ids_data.begin());

    // 3. Group (token, slot) pairs by expert (matching GGML's strategy)
    struct TokenSlot {
        int32_t slot;
        int32_t token;
    };
    std::vector<std::vector<TokenSlot>> expert_to_tokens(n_exp);

    for (int64_t token = 0; token < N; token++) {
        for (int64_t slot = 0; slot < n_used; slot++) {
            int32_t expert_id = ids_data[token * n_used + slot];
            expert_to_tokens[expert_id].push_back({static_cast<int32_t>(slot), static_cast<int32_t>(token)});
        }
    }

    // 4. Setup group GEMM with one operation per expert that has routed tokens
    std::vector<char> layout;
    std::vector<bool> transA;
    std::vector<bool> transB;
    std::vector<int> M_vec;
    std::vector<int> N_vec;
    std::vector<int> K_vec;
    std::vector<float> alpha;
    std::vector<float> beta;
    std::vector<int> lda;
    std::vector<int> ldb;
    std::vector<int> ldc;
    std::vector<bool> is_weights_const;
    std::vector<const void*> src_ptrs;
    std::vector<const void*> weight_ptrs;
    std::vector<const void*> bias_ptrs;
    std::vector<void*> dst_ptrs;
    std::vector<matmul_params> params_vec;

    // Temporary buffers for gathering inputs/scattering outputs per expert
    std::vector<std::vector<char>> expert_input_bufs;
    std::vector<std::vector<char>> expert_output_bufs;

    matmul_data_types dtypes;
    dtypes.src = src_dt;
    dtypes.wei = wei_dt;
    dtypes.dst = dst_dt;
    dtypes.bias = zendnnl::common::data_type_t::none;
    dtypes.compute = zendnnl::common::data_type_t::f32;

    matmul_params params_template;
    params_template.dtypes = dtypes;
    params_template.num_threads = desc.threads;

    for (int64_t expert_id = 0; expert_id < n_exp; expert_id++) {
        const auto& tokens = expert_to_tokens[expert_id];
        if (tokens.empty()) continue;

        int64_t num_rows = tokens.size();

        // Allocate gather buffer for this expert's inputs
        expert_input_bufs.emplace_back(num_rows * K * src_elem_size);
        char* input_buf = expert_input_bufs.back().data();

        // Allocate scatter buffer for this expert's outputs
        expert_output_bufs.emplace_back(num_rows * M * dst_elem_size);
        char* output_buf = expert_output_bufs.back().data();

        // NOTE: Gather is done in timing loop, not here!

        // Setup GEMM operation for this expert
        // GEMM: output[num_rows, M] = input[num_rows, K] × weights[K, M]
        // With transB=true, weights are provided as [M, K] and transposed
        layout.push_back('r');
        transA.push_back(false);  // don't transpose input
        transB.push_back(true);   // transpose expert weights [M,K] -> [K,M]
        M_vec.push_back(num_rows);  // M: rows of input and output
        N_vec.push_back(M);         // N: cols of output (output features)
        K_vec.push_back(K);         // K: inner dimension
        alpha.push_back(1.0f);
        beta.push_back(0.0f);
        lda.push_back(K);           // leading dim of input: [num_rows, K]
        ldb.push_back(K);           // leading dim of weights: [M, K]
        ldc.push_back(M);           // leading dim of output: [num_rows, M]
        is_weights_const.push_back(true);

        src_ptrs.push_back(input_buf);

        // Expert weight pointer: [K, M, n_experts] -> [:, :, expert_id]
        size_t weight_offset = expert_id * expert_size * wei_elem_size;
        weight_ptrs.push_back(reinterpret_cast<const char*>(expert_ptr) + weight_offset);

        bias_ptrs.push_back(nullptr);
        dst_ptrs.push_back(output_buf);
        params_vec.push_back(params_template);

        // Debug output (disabled)
        // fprintf(stderr, "Expert %ld: M=%d, N=%ld, K=%d, num_tokens=%ld\n",
        //        expert_id, (int)M, num_rows, (int)K, num_rows);
    }

    // fprintf(stderr, "Total group GEMM operations: %zu\n", layout.size());

    auto t_op_end = std::chrono::steady_clock::now();
    double op_creation_ms = std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    // Lambda to gather inputs for each expert
    auto gather_inputs = [&]() {
        size_t expert_idx = 0;
        for (int64_t expert_id = 0; expert_id < n_exp; expert_id++) {
            const auto& tokens = expert_to_tokens[expert_id];
            if (tokens.empty()) continue;

            if (expert_idx >= expert_input_bufs.size()) {
                fprintf(stderr, "error: expert_idx %zu out of bounds (size %zu)\n",
                       expert_idx, expert_input_bufs.size());
                exit(1);
            }

            char* input_buf = expert_input_bufs[expert_idx].data();

            for (size_t i = 0; i < tokens.size(); i++) {
                int32_t slot = tokens[i].slot;
                int32_t token = tokens[i].token;

                // Bounds check
                if (token < 0 || token >= N || slot < 0 || slot >= n_used) {
                    fprintf(stderr, "error: invalid token=%d slot=%d during gather (N=%d n_used=%d)\n",
                           token, slot, (int)N, (int)n_used);
                    exit(1);
                }

                // Source: input[:, slot, token]
                size_t input_offset = (token * n_used * K + slot * K) * src_elem_size;
                const char* src = reinterpret_cast<const char*>(input_ptr) + input_offset;

                // Copy K elements to gather buffer
                memcpy(input_buf + i * K * src_elem_size, src, K * src_elem_size);
            }

            expert_idx++;
        }
    };

    // Lambda to scatter outputs back to correct positions
    auto scatter_outputs = [&]() {
        size_t expert_idx = 0;
        for (int64_t expert_id = 0; expert_id < n_exp; expert_id++) {
            const auto& tokens = expert_to_tokens[expert_id];
            if (tokens.empty()) continue;

            if (expert_idx >= expert_output_bufs.size()) {
                fprintf(stderr, "error: expert_idx %zu out of bounds (size %zu)\n",
                       expert_idx, expert_output_bufs.size());
                exit(1);
            }

            const char* output_buf = expert_output_bufs[expert_idx].data();

            for (size_t i = 0; i < tokens.size(); i++) {
                int32_t slot = tokens[i].slot;
                int32_t token = tokens[i].token;

                // Bounds check
                if (token < 0 || token >= N || slot < 0 || slot >= n_used) {
                    fprintf(stderr, "error: invalid token=%d slot=%d (N=%d n_used=%d)\n",
                           token, slot, (int)N, (int)n_used);
                    exit(1);
                }

                // Destination: output[:, slot, token]
                size_t output_offset = (token * n_used * M + slot * M) * dst_elem_size;
                char* dst = reinterpret_cast<char*>(output_ptr) + output_offset;

                // Copy M elements from scatter buffer
                memcpy(dst, output_buf + i * M * dst_elem_size, M * dst_elem_size);
            }

            expert_idx++;
        }
    };

    // 5. Warmup
    for (int i = 0; i < desc.warmup; i++) {
        if (!layout.empty()) {
            gather_inputs();  // Gather inputs for each expert

            status_t status = group_gemm_direct(
                layout, transA, transB,
                M_vec, N_vec, K_vec,
                alpha,
                src_ptrs, lda,
                weight_ptrs, ldb,
                bias_ptrs,
                beta,
                dst_ptrs, ldc,
                is_weights_const,
                params_vec
            );

            if (status != status_t::success) {
                fprintf(stderr, "error: ZenDNN group GEMM warmup failed: status=%d\n",
                       static_cast<int>(status));
                exit(1);
            }

            scatter_outputs();
        }
    }

    // 6. Timed iterations
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    for (int i = 0; i < desc.repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();

        if (!layout.empty()) {
            gather_inputs();  // Gather inputs for each expert (TIMED)

            status_t status = group_gemm_direct(
                layout, transA, transB,
                M_vec, N_vec, K_vec,
                alpha,
                src_ptrs, lda,
                weight_ptrs, ldb,
                bias_ptrs,
                beta,
                dst_ptrs, ldc,
                is_weights_const,
                params_vec
            );

            if (status != status_t::success) {
                fprintf(stderr, "error: ZenDNN group GEMM failed: status=%d\n",
                       static_cast<int>(status));
                exit(1);
            }

            scatter_outputs();
        }

        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / desc.repeats;
    double flops = 2.0 * M * K * n_used * N;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    BenchResult result;
    result.min_ms = min_ms;
    result.avg_ms = avg_ms;
    result.max_ms = max_ms;
    result.tflops = tflops;
    result.ctx_creation_ms = ctx_creation_ms;
    result.op_creation_ms = op_creation_ms;
    result.op_execution_ms = avg_ms;
    result.other_ms = 0.0;

    // Copy output for verification if requested
    if (desc.verify_output) {
        size_t output_size = M * n_used * N;
        result.output_data.resize(output_size);
        const float* output_f32 = reinterpret_cast<const float*>(output_ptr);
        std::copy(output_f32, output_f32 + output_size, result.output_data.begin());
    }

    return result;
}
