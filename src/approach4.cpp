#include "approach4.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "zendnnl.hpp"
#include "ggml_utils.h"
#include "routing_utils.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstring>

// Use the same namespace as zendnn_matmul_bench.cpp — all matmul API lives here.
using namespace zendnnl::lowoha::matmul;

// ---------------------------------------------------------------------------
// Internal context — scratch buffers reused across iterations.
// ---------------------------------------------------------------------------
struct ggml_backend_zendnn_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    std::unique_ptr<char[]> wdata_cur;
    size_t wdata_cur_size = 0;
    std::unique_ptr<char[]> dst_cur;
    size_t dst_cur_size   = 0;
    size_t work_size      = 0;
};

// ---------------------------------------------------------------------------
// Type helper
// ---------------------------------------------------------------------------
static zendnnl::common::data_type_t ggml_type_to_zendnn(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return zendnnl::common::data_type_t::f32;
        case GGML_TYPE_BF16: return zendnnl::common::data_type_t::bf16;
        default:
            fprintf(stderr, "error: approach4 unsupported ggml_type %d\n", t);
            exit(1);
    }
}

// ---------------------------------------------------------------------------
// Core typed matmul wrapper.
//
// Computes C[n, m] = B[n, k] x A^T[k, m]
// A is stored as [k, m] (column-major weights); ZenDNN transposes it via
// transB=true (second matrix in matmul_direct is the weight).
//
// Uses matmul_params / matmul_direct from zendnnl::lowoha::matmul —
// the same API used in zendnn_matmul_bench.cpp.
// ---------------------------------------------------------------------------
template <typename TA, typename TB, typename TC>
static bool approach4_matmul(ggml_backend_zendnn_context* ctx,
                             int64_t m, int64_t n, int64_t k,
                             const TA* A, int64_t lda,
                             const TB* B, int64_t ldb,
                             TC*       C, int64_t ldc) {
    matmul_data_types dtypes;
    dtypes.src     = ggml_type_to_zendnn(
                         std::is_same_v<TB, float> ? GGML_TYPE_F32 : GGML_TYPE_BF16);
    dtypes.wei     = ggml_type_to_zendnn(
                         std::is_same_v<TA, float> ? GGML_TYPE_F32 : GGML_TYPE_BF16);
    dtypes.dst     = ggml_type_to_zendnn(
                         std::is_same_v<TC, float> ? GGML_TYPE_F32 : GGML_TYPE_BF16);
    dtypes.bias    = zendnnl::common::data_type_t::none;
    dtypes.compute = zendnnl::common::data_type_t::f32;

    matmul_params params;
    params.dtypes      = dtypes;
    params.num_threads = ctx->n_threads;

    matmul_batch_params_t batch_params;
    batch_params.Batch_A = 1;
    batch_params.Batch_B = 1;

    // matmul_direct signature (matches zendnn_matmul_bench.cpp usage):
    //   layout, transA(src), transB(wei), M, N, K, alpha,
    //   src_ptr, lda, wei_ptr, ldb, bias, beta, dst_ptr, ldc,
    //   is_weights_const, batch_params, params
    //
    // Here: src=B[n,k], wei=A[k,m] with transB=true → A^T[m,k]
    //   M(rows of output) = n
    //   N(cols of output) = m
    //   K(inner)          = k
    status_t status = matmul_direct(
        'r',   false, true,   // row-major, no-transA (src B), transB (wei A)
        n,                    // M: rows of B and C
        m,                    // N: cols of A^T and C
        k,                    // K: inner dimension
        1.0f,
        B, ldb,               // src  B[n, k]
        A, lda,               // wei  A[k, m] (transposed by ZenDNN)
        nullptr, 0.0f,
        C, ldc,               // dst  C[n, m]
        true,                 // is_weights_const
        batch_params,
        params
    );

    if (status != status_t::success) {
        fprintf(stderr, "approach4: matmul_direct failed: status=%d\n",
                static_cast<int>(status));
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Type-dispatch sgemm (mirrors original ggml_zendnn_sgemm logic)
// ---------------------------------------------------------------------------
static bool approach4_sgemm(ggml_backend_zendnn_context* ctx,
                            int64_t m, int64_t n, int64_t k,
                            const void* A, int64_t lda,
                            const void* B, int64_t ldb,
                            void*       C, int64_t ldc,
                            int Atype, int Btype, int Ctype) {
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(lda >= k && ldb >= k && ldc >= m);

    switch (Atype) {
        case GGML_TYPE_F32:
            if (Btype != GGML_TYPE_F32 || Ctype != GGML_TYPE_F32) return false;
            return approach4_matmul<float, float, float>(
                ctx, m, n, k,
                (const float*)A, lda,
                (const float*)B, ldb,
                (float*)C,       ldc);

        case GGML_TYPE_BF16:
            if (Btype != GGML_TYPE_BF16) return false;
            if (Ctype == GGML_TYPE_BF16)
                return approach4_matmul<ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(
                    ctx, m, n, k,
                    (const ggml_bf16_t*)A, lda,
                    (const ggml_bf16_t*)B, ldb,
                    (ggml_bf16_t*)C,       ldc);
            if (Ctype == GGML_TYPE_F32)
                return approach4_matmul<ggml_bf16_t, ggml_bf16_t, float>(
                    ctx, m, n, k,
                    (const ggml_bf16_t*)A, lda,
                    (const ggml_bf16_t*)B, ldb,
                    (float*)C,             ldc);
            return false;

        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Row-mapping for matmul_id scatter/gather
// ---------------------------------------------------------------------------
struct mmid_row_mapping {
    int32_t i1;   // expert slot index
    int32_t i2;   // token index
};

// ---------------------------------------------------------------------------
// Core matmul_id compute (core logic preserved from original approach4.cpp,
// only the internal GEMM call is updated to use approach4_sgemm above).
//
// Algorithm:
//   1. Optionally convert src1 to vec_dot_type in a scratch buffer.
//   2. Group (slot, token) pairs by expert id.
//   3. For each expert with >0 routed tokens:
//        a. Gather those rows into wdata_cur.
//        b. approach4_sgemm for the expert weight slice.
//        c. Scatter results back into dst.
// ---------------------------------------------------------------------------
static void approach4_compute_forward_mul_mat_id(
        ggml_backend_zendnn_context* ctx,
        ggml_tensor* dst) {

    const ggml_tensor* src0 = dst->src[0];  // expert weights  [K, M, n_exp]
    const ggml_tensor* src1 = dst->src[1];  // token inputs    [K, n_used, N]
    const ggml_tensor* ids  = dst->src[2];  // routing ids     [n_used, N]

    GGML_TENSOR_BINARY_OP_LOCALS

    const ggml_type         vec_dot_type = src0->type;
    const ggml_from_float_t from_float   =
        ggml_get_type_traits(vec_dot_type)->from_float_ref;

    if (ne2 == 0 || ne11 == 0) return;

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(nb0 <= nb1 && nb1 <= nb2 && nb2 <= nb3);
    GGML_ASSERT(ne03 == 1 && ne13 == 1 && ne3 == 1);

    const int n_ids = ids->ne[0];   // n_experts_used
    const int n_as  = ne02;         // n_experts

    // --- optional type conversion of src1 → vec_dot_type ---
    void* work_data = ctx->work_data.get();
    if (src1->type != vec_dot_type) {
        const size_t nbw1    = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2    = nbw1 * ne11;
        const size_t nbw3    = nbw2 * ne12;
        const size_t desired = ne13 * nbw3;
        if (ctx->work_size < desired) {
            ctx->work_data.reset(new char[desired]);
            ctx->work_size = desired;
        }
        work_data = ctx->work_data.get();

        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        #pragma omp parallel for collapse(3) \
            num_threads(ctx->n_threads) schedule(static)
        for (int64_t i13 = 0; i13 < ne13; ++i13)
            for (int64_t i12 = 0; i12 < ne12; ++i12)
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    const float* s = (float*)(
                        (char*)src1->data + i11*nb11 + i12*nb12 + i13*nb13);
                    void* d = (char*)work_data + i11*nbw1 + i12*nbw2 + i13*nbw3;
                    from_float(s, d, ne10);
                }
    }

    const void*  wdata    = (src1->type == vec_dot_type) ? src1->data : work_data;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    // --- group rows by expert ---
    std::vector<int64_t>                       matrix_row_counts(n_as, 0);
    std::vector<std::vector<mmid_row_mapping>> matrix_rows(n_as);

    for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        for (int id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t*)(
                (const char*)ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(i02 >= 0 && i02 < n_as);
            matrix_rows[i02].push_back({id, (int32_t)iid1});
            matrix_row_counts[i02]++;
        }
    }

    // --- size scratch buffers ---
    const int64_t max_rows   = ids->ne[1] * n_ids;
    const size_t  need_wdata = max_rows * row_size;
    const size_t  need_dst   = max_rows * ggml_row_size(dst->type, ne01);

    if (ctx->wdata_cur_size < need_wdata) {
        ctx->wdata_cur.reset(new char[need_wdata]);
        ctx->wdata_cur_size = need_wdata;
    }
    if (ctx->dst_cur_size < need_dst) {
        ctx->dst_cur.reset(new char[need_dst]);
        ctx->dst_cur_size = need_dst;
    }

    void* wdata_cur = ctx->wdata_cur.get();
    void* dst_cur   = ctx->dst_cur.get();

    // --- per-expert: gather → GEMM → scatter ---
    for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1     = matrix_row_counts[cur_a];
        const char*   src0_cur =
            static_cast<const char*>(src0->data) + cur_a * nb02;

        if (cne1 == 0) continue;

        // gather input rows for this expert
        #pragma omp parallel for num_threads(ctx->n_threads) schedule(static)
        for (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            const mmid_row_mapping& mr  = matrix_rows[cur_a][ir1];
            const int64_t           i11 = mr.i1 % ne11;
            const int64_t           i12 = mr.i2;
            std::memcpy(
                (char*)wdata_cur + ir1 * row_size,
                (const char*)wdata + (i11 + i12 * ne11) * row_size,
                row_size);
        }

        // batched GEMM for this expert's token batch
        if (!approach4_sgemm(ctx,
                             ne01, cne1, ne10,
                             src0_cur, ne00,
                             wdata_cur, ne10,
                             dst_cur,   ne01,
                             src0->type, vec_dot_type, dst->type))
            GGML_ABORT("%s: approach4 sgemm failed for expert %ld\n",
                       __func__, cur_a);

        // scatter outputs back to dst
        #pragma omp parallel for num_threads(ctx->n_threads) schedule(static)
        for (int64_t ir1 = 0; ir1 < cne1; ++ir1) {
            const mmid_row_mapping& mr = matrix_rows[cur_a][ir1];
            std::memcpy(
                (char*)dst->data + mr.i1 * nb1 + mr.i2 * nb2,
                (char*)dst_cur   + ir1 * ggml_row_size(dst->type, ne01),
                ggml_row_size(dst->type, ne01));
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers used by bench_matmul_id_approach4 to set up ggml tensor scaffolding.
// bench_matmul_approach4 uses raw std::vector buffers instead.
// ---------------------------------------------------------------------------
static ggml_backend_t init_cpu_backend(int n_threads) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "error: CPU backend init failed\n"); exit(1); }
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    return backend;
}

static ggml_context* init_ggml_ctx() {
    const size_t ctx_size =
        8 * ggml_tensor_overhead() + ggml_graph_overhead() + 1024 * 1024;
    struct ggml_init_params p = {};
    p.mem_size = ctx_size;
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) { fprintf(stderr, "error: ggml_init failed\n"); exit(1); }
    return ctx;
}

// ---------------------------------------------------------------------------
// Public benchmark: matmul_id  (MoE expert-routed matmul)
//
// Tensor layout (GGML conventions):
//   as  [K, M, n_exp]   — expert weight stacks
//   b   [K, n_used, N]  — token inputs, n_used slots each
//   ids [n_used, N]     — i32 expert assignments
//   c   [M, n_used, N]  — f32 outputs
//
// FLOPs = 2 * M * K * n_used * N
// ---------------------------------------------------------------------------
BenchResult bench_matmul_id_approach4(const OpDesc& desc) {
    const int64_t M      = desc.m;
    const int64_t N      = desc.n;
    const int64_t K      = desc.k;
    const int64_t n_exp  = desc.n_experts;
    const int64_t n_used = desc.n_experts_used;
    const ggml_type src_dtype = desc.src_dtype;
    const ggml_type wei_dtype = desc.wei_dtype;

    if (wei_dtype != GGML_TYPE_F32 && wei_dtype != GGML_TYPE_BF16) {
        fprintf(stderr, "error: approach4 only supports f32/bf16 weights\n");
        exit(1);
    }

    auto t_ctx_start = std::chrono::steady_clock::now();
    ggml_backend_t backend = init_cpu_backend(desc.threads);
    ggml_context*  ctx     = init_ggml_ctx();
    ggml_backend_zendnn_context* zctx = new ggml_backend_zendnn_context;
    zctx->n_threads = desc.threads;
    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_creation_ms =
        std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    auto t_op_start = std::chrono::steady_clock::now();
    struct ggml_tensor* as  = ggml_new_tensor_3d(ctx, wei_dtype, K, M, n_exp);
    struct ggml_tensor* b   = ggml_new_tensor_3d(ctx, src_dtype, K, n_used, N);
    struct ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, N);
    struct ggml_tensor* c   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, M, n_used, N);
    c->src[0] = as;
    c->src[1] = b;
    c->src[2] = ids;
    c->op     = GGML_OP_MUL_MAT_ID;

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    ggml_gallocr_t allocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocr, graph)) {
        fprintf(stderr, "error: graph allocation failed\n"); exit(1);
    }

    fill_tensor_deterministic(as,  desc.data_seed,      false);
    fill_tensor_deterministic(b,   desc.data_seed + 95, false);
    std::vector<int32_t> routing_ids = generate_routing_ids(
        N, n_exp, n_used,
        desc.expert_token_counts,
        desc.routing_pattern,
        desc.routing_seed);
    memcpy(ids->data, routing_ids.data(), routing_ids.size() * sizeof(int32_t));
    auto t_op_end = std::chrono::steady_clock::now();
    double op_creation_ms =
        std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    for (int i = 0; i < desc.warmup; i++)
        approach4_compute_forward_mul_mat_id(zctx, c);

    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0, sum_ms = 0.0;
    for (int i = 0; i < desc.repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();
        approach4_compute_forward_mul_mat_id(zctx, c);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms  = std::min(min_ms, ms);
        max_ms  = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / desc.repeats;
    double flops  = 2.0 * M * K * n_used * N;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    BenchResult result;
    result.min_ms = min_ms; result.avg_ms = avg_ms; result.max_ms = max_ms;
    result.tflops = tflops;
    result.ctx_creation_ms = ctx_creation_ms;
    result.op_creation_ms  = op_creation_ms;
    result.op_execution_ms = avg_ms;
    result.other_ms        = 0.0;

    if (desc.verify_output) {
        size_t out_elems = M * n_used * N;
        result.output_data.resize(out_elems);
        memcpy(result.output_data.data(), c->data, out_elems * sizeof(float));
    }

    delete zctx;
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return result;
}

// ---------------------------------------------------------------------------
// Public benchmark: plain matmul (no routing)
//
// C[N, M] = B[N, K] x A^T[K, M]
// A stored as weights [K, M], transposed inside matmul_direct.
// FLOPs = 2 * M * N * K
//
// Uses raw std::vector buffers (same pattern as zendnn_matmul_bench.cpp)
// instead of ggml graph allocation — plain matmul has no ggml op to drive
// the allocator, so tensors would get null data pointers.
// ---------------------------------------------------------------------------
BenchResult bench_matmul_approach4(const OpDesc& desc) {
    const int64_t M = desc.m;
    const int64_t N = desc.n;
    const int64_t K = desc.k;
    const ggml_type src_dtype = desc.src_dtype;
    const ggml_type wei_dtype = desc.wei_dtype;

    if (wei_dtype != GGML_TYPE_F32 && wei_dtype != GGML_TYPE_BF16) {
        fprintf(stderr, "error: approach4 only supports f32/bf16 weights\n");
        exit(1);
    }
    if (src_dtype != wei_dtype) {
        fprintf(stderr,
                "error: approach4 matmul requires matching src and wei dtypes "
                "(got src=%d wei=%d)\n", src_dtype, wei_dtype);
        exit(1);
    }

    // ---- context setup ----
    auto t_ctx_start = std::chrono::steady_clock::now();

    ggml_backend_zendnn_context* zctx = new ggml_backend_zendnn_context;
    zctx->n_threads = desc.threads;

    auto t_ctx_end = std::chrono::steady_clock::now();
    double ctx_creation_ms =
        std::chrono::duration<double, std::milli>(t_ctx_end - t_ctx_start).count();

    // ---- allocate raw buffers (no ggml graph needed for plain matmul) ----
    auto t_op_start = std::chrono::steady_clock::now();

    const size_t elem_size = (wei_dtype == GGML_TYPE_F32) ? sizeof(float)
                                                          : sizeof(uint16_t);
    // A: weights [K, M]  →  K*M elements
    // B: input   [N, K]  →  N*K elements  (N rows of K)
    // C: output  [N, M]  →  N*M elements  (always f32)
    std::vector<char> a_buf(K * M * elem_size);
    std::vector<char> b_buf(N * K * elem_size);
    std::vector<float> c_buf(N * M, 0.0f);

    void* a_ptr = a_buf.data();
    void* b_ptr = b_buf.data();
    void* c_ptr = c_buf.data();

    // Fill with deterministic pseudo-random data matching ggml conventions.
    // We reuse the same xorshift pattern as zendnn_matmul_bench.cpp.
    auto xorshift32 = [](uint32_t& s) -> uint32_t {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
    };
    auto fill_f32 = [&](void* ptr, size_t n, uint32_t seed) {
        uint32_t s = seed;
        float* p = reinterpret_cast<float*>(ptr);
        for (size_t i = 0; i < n; i++) {
            uint32_t r = xorshift32(s);
            p[i] = static_cast<float>(static_cast<int32_t>(r)) /
                   static_cast<float>(INT32_MAX);
        }
    };
    auto fill_bf16 = [&](void* ptr, size_t n, uint32_t seed) {
        uint32_t s = seed;
        uint16_t* p = reinterpret_cast<uint16_t*>(ptr);
        for (size_t i = 0; i < n; i++) {
            uint32_t r = xorshift32(s);
            float val = static_cast<float>(static_cast<int32_t>(r)) /
                        static_cast<float>(INT32_MAX);
            uint32_t u = *reinterpret_cast<uint32_t*>(&val);
            if ((u & 0x7fffffff) > 0x7f800000)
                p[i] = (u >> 16) | 64;
            else
                p[i] = static_cast<uint16_t>(
                    (u + (0x7fff + ((u >> 16) & 1))) >> 16);
        }
    };

    if (wei_dtype == GGML_TYPE_F32) {
        fill_f32(a_ptr, K * M, desc.data_seed);
        fill_f32(b_ptr, N * K, desc.data_seed + 95);
    } else {
        fill_bf16(a_ptr, K * M, desc.data_seed);
        fill_bf16(b_ptr, N * K, desc.data_seed + 95);
    }

    auto t_op_end = std::chrono::steady_clock::now();
    double op_creation_ms =
        std::chrono::duration<double, std::milli>(t_op_end - t_op_start).count();

    // A[K,M]: lda=K  (K elements per row of A, M rows)
    // B[N,K]: ldb=K  (K elements per row of B, N rows)
    // C[N,M]: ldc=M  (M elements per row of C, N rows)
    auto run_once = [&]() {
        if (!approach4_sgemm(zctx,
                             M, N, K,
                             a_ptr, K,   // weights A[K,M]
                             b_ptr, K,   // input   B[N,K]
                             c_ptr, M,   // output  C[N,M]
                             wei_dtype, src_dtype, GGML_TYPE_F32))
            GGML_ABORT("%s: approach4 matmul sgemm failed\n", __func__);
    };

    // ---- warmup ----
    for (int i = 0; i < desc.warmup; i++) run_once();

    // ---- timed iterations ----
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0, sum_ms = 0.0;
    for (int i = 0; i < desc.repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();
        run_once();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms  = std::min(min_ms, ms);
        max_ms  = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / desc.repeats;
    double flops  = 2.0 * M * N * K;
    double tflops = flops / (avg_ms * 1e-3) / 1e12;

    BenchResult result;
    result.min_ms = min_ms; result.avg_ms = avg_ms; result.max_ms = max_ms;
    result.tflops = tflops;
    result.ctx_creation_ms = ctx_creation_ms;
    result.op_creation_ms  = op_creation_ms;
    result.op_execution_ms = avg_ms;
    result.other_ms        = 0.0;

    if (desc.verify_output) {
        result.output_data.assign(c_buf.begin(), c_buf.end());
    }

    delete zctx;
    return result;
}
