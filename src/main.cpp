#include "op_desc.h"
#include "benchmark.h"
#include "ggml_utils.h"
#include "layer_config.h"
#include "layer_bench.h"
#include "moe_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct Shape { int m, n, k; };

static bool parse_shape(const std::string& s, Shape& out) {
    int m, n, k; char x1, x2;
    std::istringstream ss(s);
    if (ss >> m >> x1 >> n >> x2 >> k && x1 == 'x' && x2 == 'x'
            && m > 0 && n > 0 && k > 0) {
        out = {m, n, k}; return true;
    }
    return false;
}

static std::vector<Shape> parse_shapes_list(const std::string& s) {
    std::vector<Shape> shapes;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        Shape sh;
        if (!parse_shape(token, sh)) {
            fprintf(stderr, "error: invalid shape '%s' (expected MxNxK)\n",
                    token.c_str());
            exit(1);
        }
        shapes.push_back(sh);
    }
    return shapes;
}

static std::vector<Shape> read_batch_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "error: cannot open batch file '%s'\n", path.c_str());
        exit(1);
    }
    std::vector<Shape> shapes;
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        lineno++;
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line[0] == '#') continue;
        Shape sh;
        if (parse_shape(line, sh)) { shapes.push_back(sh); continue; }
        std::istringstream ss(line);
        int m, n, k;
        if (ss >> m >> n >> k && m > 0 && n > 0 && k > 0) {
            shapes.push_back({m, n, k}); continue;
        }
        fprintf(stderr, "error: %s:%d: invalid line '%s'\n",
                path.c_str(), lineno, line.c_str());
        exit(1);
    }
    return shapes;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --op <matmul|matmul_id|layer>\n"
        "  --config <file>           Layer config file\n"
        "  --moe-config <file>       MoE config file\n"
        "  --m <int>                 M dimension (default: 512)\n"
        "  --n <int>                 N dimension (default: 512)\n"
        "  --k <int>                 K dimension (default: 512)\n"
        "  --shapes <list>           Comma-separated MxNxK shapes\n"
        "  --batch <file>            Read shapes from file\n"
        "  --src_dtype <f32|bf16>\n"
        "  --wei_dtype <f32|f16|bf16|q8_0|q4_0>\n"
        "  --backend <ggml|zendnn>\n"
        "  --threads <int>\n"
        "  --repeats <int>           (default: 100)\n"
        "  --warmup  <int>           (default: 10)\n"
        "  --n_experts <int>\n"
        "  --n_experts_used <int>\n"
        "  --routing_pattern <pat>\n"
        "  --expert_token_counts <c>\n"
        "  --routing_seed <int>\n"
        "  --seed <int>\n"
        "  --verify                  Three-way output verification\n"
        "  --help\n",
        prog);
}

// ===========================================================================
//  Verification helpers
// ===========================================================================

struct VerifyStats {
    double max_abs_err = 0.0;
    double sum_abs_err = 0.0;
    double max_rel_err = 0.0;
    double sum_rel_err = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref  = 0.0;
    int64_t n_fail_abs = 0;
    int64_t n_fail_rel = 0;
    int64_t count      = 0;

    // Derived — populated by finalise()
    double mae     = 0.0;
    double rmse    = 0.0;
    double nrmse   = 0.0;
    double avg_rel = 0.0;
};

static void finalise(VerifyStats& s) {
    if (s.count == 0) return;
    s.mae   = s.sum_abs_err / (double)s.count;
    s.rmse  = std::sqrt(s.sum_sq_diff / (double)s.count);
    double rms_ref = std::sqrt(s.sum_sq_ref / (double)s.count);
    s.nrmse = (rms_ref > 1e-10) ? (s.rmse / rms_ref) : 0.0;
    s.avg_rel = s.sum_rel_err / (double)s.count;
}

static void accumulate(VerifyStats& s,
                        double ref, double got,
                        double atol, double rtol) {
    double abs_err = std::abs(ref - got);
    double rel_err = (std::abs(ref) > 1e-6)
                   ? abs_err / std::abs(ref) : 0.0;
    s.sum_abs_err += abs_err;
    s.sum_rel_err += rel_err;
    s.sum_sq_diff += (ref - got) * (ref - got);
    s.sum_sq_ref  += ref * ref;
    s.max_abs_err  = std::max(s.max_abs_err, abs_err);
    s.max_rel_err  = std::max(s.max_rel_err, rel_err);
    s.count++;
    if (abs_err > atol) s.n_fail_abs++;
    if (rel_err > rtol) s.n_fail_rel++;
}

// ---------------------------------------------------------------------------
//  print_three_way_verification
//
//  ref     = out_ggml   from ggml BenchResult   (GGML backend = baseline)
//  custom  = out_custom from zendnn BenchResult (custom OMP kernel)
//  zendnn  = out_zendnn from zendnn BenchResult (ZenDNN kernel)
// ---------------------------------------------------------------------------
static bool print_three_way_verification(
    int64_t      n_rows,    // tokens
    int64_t      n_cols,    // output_features
    int64_t      ldc,       // row stride of C (== n_cols)
    const float* ref,       // GGML backend output  — baseline
    const float* custom,    // custom OMP output
    const float* zendnn,    // ZenDNN output
    double       atol,
    double       rtol)
{
    const int64_t total = n_rows * n_cols;

    VerifyStats sc, sz;
    double ref_min =  std::numeric_limits<double>::max();
    double ref_max = -std::numeric_limits<double>::max();

    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            double r  = static_cast<double>(ref   [i * ldc + j]);
            double vc = static_cast<double>(custom[i * ldc + j]);
            double vz = static_cast<double>(zendnn[i * ldc + j]);
            ref_min = std::min(ref_min, r);
            ref_max = std::max(ref_max, r);
            accumulate(sc, r, vc, atol, rtol);
            accumulate(sz, r, vz, atol, rtol);
        }
    }
    finalise(sc);
    finalise(sz);

    // ── banner ────────────────────────────────────────────────────────────
    printf("\n");
    printf("========================================================================\n");
    printf("THREE-WAY VERIFICATION RESULTS\n");
    printf("Baseline : ggml  (GGML backend output)\n");
    printf("Kernels  : custom_omp | zendnn_f32s8\n");
    printf("Shape    : tokens=%-6ld  out_features=%-6ld  total_cells=%ld\n",
           n_rows, n_cols, total);
    printf("Ref range: [%.6f, %.6f]\n", ref_min, ref_max);
    printf("Tolerances: atol=%.0e  rtol=%.0e\n", atol, rtol);
    printf("========================================================================\n");

    // ── metrics table ─────────────────────────────────────────────────────
    printf("\n%-18s %12s %12s %12s %12s %12s %10s %10s %10s\n",
           "Kernel",
           "MaxAbsErr", "MAE",
           "MaxRelErr%", "AvgRelErr%",
           "RMSE", "NRMSE%",
           "FailAbs", "FailRel");
    printf("%s\n", std::string(110, '-').c_str());

    auto print_row = [&](const char* label, const VerifyStats& s) {
        bool pass = (s.n_fail_abs == 0) && (s.n_fail_rel == 0);
        printf("%-18s %12.4e %12.4e %12.4f %12.4f %12.4e %10.4f %10ld %10ld   %s\n",
               label,
               s.max_abs_err, s.mae,
               s.max_rel_err * 100.0, s.avg_rel * 100.0,
               s.rmse, s.nrmse * 100.0,
               s.n_fail_abs, s.n_fail_rel,
               pass ? "✓ PASS" : "✗ FAIL");
    };

    print_row("custom_omp",   sc);
    print_row("zendnn_f32s8", sz);
    printf("%s\n", std::string(110, '-').c_str());

    // ── 3×3 spot-check values ─────────────────────────────────────────────
    int pr = (int)std::min((int64_t)3, n_rows);
    int pc = (int)std::min((int64_t)3, n_cols);

    printf("\nSpot-check — top-left %d×%d submatrix:\n", pr, pc);
    printf("  %-18s", "");
    for (int c = 0; c < pc; c++) printf("  col%-9d", c);
    printf("\n");

    auto print_grid = [&](const char* label, const float* data) {
        for (int r = 0; r < pr; r++) {
            printf("  %-18s", r == 0 ? label : "");
            for (int c = 0; c < pc; c++)
                printf("  %12.5f", data[r * ldc + c]);
            printf("\n");
        }
    };
    print_grid("ggml (baseline)", ref);    printf("\n");
    print_grid("custom_omp",      custom); printf("\n");
    print_grid("zendnn_f32s8",    zendnn);

    // ── per-cell relative error grids ─────────────────────────────────────
    printf("\nElement-wise relative error (%%) vs ggml baseline:\n");
    struct KP { const char* label; const float* data; };
    KP kps[2] = { {"custom_omp", custom}, {"zendnn_f32s8", zendnn} };
    for (auto& kp : kps) {
        printf("  %s:\n", kp.label);
        for (int r = 0; r < pr; r++) {
            printf("    [");
            for (int c = 0; c < pc; c++) {
                double rv  = std::abs(static_cast<double>(ref[r * ldc + c]));
                double err = std::abs(static_cast<double>(ref[r * ldc + c])
                                    - static_cast<double>(kp.data[r * ldc + c]));
                double pct = (rv > 1e-6) ? (err / rv * 100.0) : 0.0;
                printf("%9.4f%%", pct);
                if (c < pc - 1) printf(", ");
            }
            printf("]\n");
        }
    }

    // ── top-10 worst for any failing kernel ───────────────────────────────
    struct KS { const char* label; const VerifyStats& s; const float* data; };
    KS klist[2] = {
        { "custom_omp",   sc, custom },
        { "zendnn_f32s8", sz, zendnn }
    };
    for (auto& ks : klist) {
        if (ks.s.n_fail_abs == 0 && ks.s.n_fail_rel == 0) continue;
        printf("\nTop-10 worst relative errors — %s:\n", ks.label);
        printf("  %-10s %14s %14s %14s %14s\n",
               "flat_idx", "ggml_ref", ks.label, "abs_err", "rel_err%");
        std::vector<std::pair<double, int64_t>> errs;
        errs.reserve(total);
        for (int64_t j = 0; j < total; ++j) {
            double r_ = std::abs(static_cast<double>(ref[j]));
            double e  = std::abs(static_cast<double>(ref[j])
                               - static_cast<double>(ks.data[j]));
            errs.push_back({ (r_ > 1e-6) ? e / r_ : 0.0, j });
        }
        size_t top = std::min((int64_t)10, total);
        std::partial_sort(errs.begin(), errs.begin() + top, errs.end(),
                          std::greater<std::pair<double, int64_t>>());
        for (size_t i = 0; i < top; ++i) {
            int64_t j   = errs[i].second;
            double  rv  = static_cast<double>(ref[j]);
            double  kv  = static_cast<double>(ks.data[j]);
            printf("  %-10ld %14.6f %14.6f %14.6e %13.4f%%\n",
                   j, rv, kv, std::abs(rv - kv),
                   errs[i].first * 100.0);
        }
    }

    // ── verdict ───────────────────────────────────────────────────────────
    bool overall = (sc.n_fail_abs == 0) && (sc.n_fail_rel == 0)
                && (sz.n_fail_abs == 0) && (sz.n_fail_rel == 0);
    printf("\n========================================================================\n");
    printf(overall
        ? "✓  OVERALL PASS — all kernels within atol=%.0e rtol=%.0e\n"
        : "✗  OVERALL FAIL — see table above for failing cells\n",
        atol, rtol);
    printf("========================================================================\n\n");
    return overall;
}

// ---------------------------------------------------------------------------
//  two_way_compare — fallback for FP32 / BF16 weight types
//  Compares GGML backend vs ZenDNN backend output_data directly.
// ---------------------------------------------------------------------------
static bool two_way_compare(const OpDesc&      desc,
                             const BenchResult& ggml_r,
                             const BenchResult& zendnn_r)
{
    if (ggml_r.output_data.empty() || zendnn_r.output_data.empty()) {
        fprintf(stderr, "error: output_data missing for two-way compare\n");
        return false;
    }
    if (ggml_r.output_data.size() != zendnn_r.output_data.size()) {
        fprintf(stderr,
                "ERROR: output size mismatch — GGML=%zu  ZenDNN=%zu\n",
                ggml_r.output_data.size(), zendnn_r.output_data.size());
        return false;
    }

    double atol, rtol;
    if (desc.wei_dtype == GGML_TYPE_Q4_0 ||
        desc.wei_dtype == GGML_TYPE_Q8_0) {
        atol = 1e-2; rtol = 1e-2;
    } else if (desc.src_dtype == GGML_TYPE_BF16 ||
               desc.wei_dtype == GGML_TYPE_BF16) {
        atol = 1e-3; rtol = 5e-3;
    } else {
        atol = 1e-5; rtol = 5e-4;
    }

    size_t total = ggml_r.output_data.size();
    VerifyStats s;
    double global_min =  std::numeric_limits<double>::max();
    double global_max = -std::numeric_limits<double>::max();

    for (size_t j = 0; j < total; ++j) {
        double rv = static_cast<double>(ggml_r.output_data[j]);
        double zv = static_cast<double>(zendnn_r.output_data[j]);
        global_min = std::min(global_min, rv);
        global_max = std::max(global_max, rv);
        accumulate(s, rv, zv, atol, rtol);
    }
    finalise(s);

    bool pass = (s.n_fail_abs == 0) && (s.n_fail_rel == 0);

    int rows = desc.n, cols = desc.m;
    int pr = std::min(3, rows), pc = std::min(3, cols);

    printf("\n========================================================================\n");
    printf("TWO-WAY VERIFICATION  (GGML backend vs ZenDNN backend)\n");
    printf("Total cells: %zu    atol=%.0e    rtol=%.0e\n", total, atol, rtol);
    printf("GGML range : [%.6f, %.6f]\n", global_min, global_max);
    printf("------------------------------------------------------------------------\n");
    printf("%-18s %12s %12s %12s %12s %12s %10s\n",
           "Metric", "MaxAbsErr", "MAE",
           "MaxRelErr%", "AvgRelErr%", "RMSE", "NRMSE%");
    printf("%-18s %12.4e %12.4e %12.4f %12.4f %12.4e %10.4f\n",
           "zendnn vs ggml",
           s.max_abs_err, s.mae,
           s.max_rel_err * 100.0, s.avg_rel * 100.0,
           s.rmse, s.nrmse * 100.0);
    printf("FailAbs: %ld   FailRel: %ld   → %s\n",
           s.n_fail_abs, s.n_fail_rel, pass ? "✓ PASS" : "✗ FAIL");

    // 3×3 sample
    printf("\nSample values — top-left %d×%d:\n", pr, pc);
    printf("  %-16s", "");
    for (int c = 0; c < pc; c++) printf("  col%-9d", c);
    printf("\n");

    auto sg = [&](const char* lbl, const std::vector<float>& data) {
        for (int r = 0; r < pr; r++) {
            printf("  %-16s", r == 0 ? lbl : "");
            for (int c = 0; c < pc; c++)
                printf("  %12.5f", data[r * cols + c]);
            printf("\n");
        }
    };
    sg("ggml",   ggml_r.output_data); printf("\n");
    sg("zendnn", zendnn_r.output_data);

    printf("\n========================================================================\n");
    printf(pass ? "✓ PASS\n" : "✗ FAIL\n");
    printf("========================================================================\n\n");

    if (!pass) {
        std::vector<std::pair<double, size_t>> errs;
        errs.reserve(total);
        for (size_t j = 0; j < total; ++j) {
            double rv = std::abs(static_cast<double>(ggml_r.output_data[j]));
            double e  = std::abs(static_cast<double>(ggml_r.output_data[j])
                               - static_cast<double>(zendnn_r.output_data[j]));
            errs.push_back({ (rv > 1e-6) ? e / rv : 0.0, j });
        }
        size_t top = std::min(size_t(10), total);
        std::partial_sort(errs.begin(), errs.begin() + top, errs.end(),
                          std::greater<std::pair<double, size_t>>());
        printf("Top-10 worst relative errors:\n");
        printf("  %-10s %12s %12s %12s\n", "idx", "ggml", "zendnn", "rel_err%");
        for (size_t i = 0; i < top; ++i) {
            size_t j = errs[i].second;
            printf("  %-10zu %12.6f %12.6f %11.4f%%\n",
                   j,
                   (double)ggml_r.output_data[j],
                   (double)zendnn_r.output_data[j],
                   errs[i].first * 100.0);
        }
    }
    return pass;
}

// ===========================================================================
//  main
// ===========================================================================
int main(int argc, char** argv) {
    suppress_ggml_debug_logs();

    OpDesc base;
    base.op_name = "matmul";
    base.threads = static_cast<int>(std::thread::hardware_concurrency());
    if (base.threads < 1) base.threads = 4;

    std::vector<Shape> shapes;
    std::string batch_file, shapes_arg, config_file, moe_config_file;

    for (int i = 1; i < argc; i++) {
        auto arg  = [&](const char* n) { return strcmp(argv[i], n) == 0; };
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", argv[i]);
                exit(1);
            }
            return argv[++i];
        };

        if      (arg("--op"))       { base.op_name = next(); }
        else if (arg("--config"))   { config_file  = next(); }
        else if (arg("--moe-config")) { moe_config_file = next(); }
        else if (arg("--backend"))  { base.backend = next(); }
        else if (arg("--m"))        { base.m       = atoi(next()); }
        else if (arg("--n"))        { base.n       = atoi(next()); }
        else if (arg("--k"))        { base.k       = atoi(next()); }
        else if (arg("--shapes"))   { shapes_arg   = next(); }
        else if (arg("--batch"))    { batch_file   = next(); }
        else if (arg("--src_dtype")) {
            std::string s = next();
            base.src_dtype = parse_dtype(s);
            if (base.src_dtype == GGML_TYPE_COUNT) {
                fprintf(stderr, "error: unknown src_dtype '%s'\n", s.c_str());
                return 1;
            }
        }
        else if (arg("--wei_dtype")) {
            std::string s = next();
            base.wei_dtype = parse_dtype(s);
            if (base.wei_dtype == GGML_TYPE_COUNT) {
                fprintf(stderr, "error: unknown wei_dtype '%s'\n", s.c_str());
                return 1;
            }
        }
        else if (arg("--dtype")) {
            std::string s = next();
            ggml_type dt = parse_dtype(s);
            if (dt == GGML_TYPE_COUNT) {
                fprintf(stderr, "error: unknown dtype '%s'\n", s.c_str());
                return 1;
            }
            base.src_dtype = dt;
            base.wei_dtype = dt;
        }
        else if (arg("--threads"))  { base.threads  = atoi(next()); }
        else if (arg("--repeats"))  { base.repeats  = atoi(next()); }
        else if (arg("--warmup"))   { base.warmup   = atoi(next()); }
        else if (arg("--n_experts"))      { base.n_experts      = atoi(next()); }
        else if (arg("--n_experts_used")) { base.n_experts_used = atoi(next()); }
        else if (arg("--routing_pattern"))  { base.routing_pattern  = next(); }
        else if (arg("--routing_seed"))     { base.routing_seed     = atoi(next()); }
        else if (arg("--routing_ids_file")) { base.routing_ids_file = next(); }
        else if (arg("--seed")) { base.data_seed = atoi(next()); }
        else if (arg("--expert_token_counts")) {
            std::string cs = next();
            std::istringstream ss(cs);
            std::string tok;
            base.expert_token_counts.clear();
            while (std::getline(ss, tok, ','))
                base.expert_token_counts.push_back(atoi(tok.c_str()));
        }
        else if (arg("--verify")) { base.verify_output = true; }
        else if (arg("--help") || arg("-h")) { print_usage(argv[0]); return 0; }
        else {
            fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (base.threads <= 0)                                   return 1;
    if (base.backend != "ggml" && base.backend != "zendnn") return 1;
    if (base.src_dtype != GGML_TYPE_F32 &&
        base.src_dtype != GGML_TYPE_BF16)                   return 1;

    if (!moe_config_file.empty()) {
        OpDesc desc = parse_moe_config(moe_config_file);
        BenchResult result = run_benchmark(desc);
        print_results(desc, result);
        return 0;
    }

    if (base.op_name == "layer") {
        LayerConfig cfg = parse_layer_config(config_file);
        LayerBenchResult result;
        if (base.backend == "zendnn") {
#ifdef ENABLE_ZENDNN
            result = bench_layer_zendnn(cfg, base.wei_dtype, base.threads,
                                        base.warmup, base.repeats, base.src_dtype);
#endif
        } else {
            result = bench_layer_ggml(cfg, base.wei_dtype, base.threads,
                                      base.warmup, base.repeats, base.src_dtype);
        }
        print_layer_results(cfg, result, base.backend, base.wei_dtype,
                            base.threads, base.warmup, base.repeats);
        return 0;
    }

    if (!batch_file.empty()) {
        auto v = read_batch_file(batch_file);
        shapes.insert(shapes.end(), v.begin(), v.end());
    }
    if (!shapes_arg.empty()) {
        auto v = parse_shapes_list(shapes_arg);
        shapes.insert(shapes.end(), v.begin(), v.end());
    }
    if (shapes.empty())
        shapes.push_back({base.m, base.n, base.k});

    int overall_rc = 0;

    for (size_t si = 0; si < shapes.size(); si++) {
        if (si > 0) printf("\n");

        OpDesc desc = base;
        desc.m = shapes[si].m;
        desc.n = shapes[si].n;
        desc.k = shapes[si].k;

        // ===================================================================
        //  VERIFY MODE
        // ===================================================================
        if (desc.verify_output) {
            printf("========================================================================\n");
            printf("VERIFICATION MODE\n");
            printf("Shape M=%-4d N=%-4d K=%-4d   wei=%-6s src=%-6s   threads=%d\n",
                   desc.m, desc.n, desc.k,
                   dtype_to_string(desc.wei_dtype),
                   dtype_to_string(desc.src_dtype),
                   desc.threads);
            printf("========================================================================\n\n");

            // ── 1. GGML backend run ──────────────────────────────────────
            // Returns out_ggml + output_data (same data, both filled).
            desc.backend = "ggml";
            BenchResult ggml_result = run_benchmark(desc);
            printf("GGML:   "); print_results(desc, ggml_result);

            // ── 2. ZenDNN backend run ────────────────────────────────────
            // For Q8_0/Q4_0: returns out_custom + out_zendnn.
            // For FP32/BF16: returns only output_data.
            desc.backend = "zendnn";
            BenchResult zendnn_result = run_benchmark(desc);
            printf("ZenDNN: "); print_results(desc, zendnn_result);

            // ── 3. Choose verification path ──────────────────────────────
            //
            // Three-way path: quantised weight types (Q8_0, Q4_0).
            //   ref     = ggml_result.out_ggml    (GGML backend = baseline)
            //   custom  = zendnn_result.out_custom (custom OMP kernel)
            //   zendnn  = zendnn_result.out_zendnn (ZenDNN kernel)
            //
            // Two-way fallback: FP32 / BF16.
            //   ggml_result.output_data vs zendnn_result.output_data
            // ─────────────────────────────────────────────────────────────
            bool three_way = !ggml_result.out_ggml.empty()
                          && !zendnn_result.out_custom.empty()
                          && !zendnn_result.out_zendnn.empty();

            bool pass = false;

            if (three_way) {
                if (ggml_result.out_ggml.size() !=
                    zendnn_result.out_custom.size() ||
                    ggml_result.out_ggml.size() !=
                    zendnn_result.out_zendnn.size()) {
                    fprintf(stderr,
                            "ERROR: output size mismatch in three-way verify\n"
                            "  ggml=%zu  custom=%zu  zendnn=%zu\n",
                            ggml_result.out_ggml.size(),
                            zendnn_result.out_custom.size(),
                            zendnn_result.out_zendnn.size());
                    return 1;
                }

                // Q8_0 compound error ≈ 0.8% — use 1% tolerances.
                double atol = 1e-2, rtol = 1e-2;

                pass = print_three_way_verification(
                    /*n_rows=*/desc.n,
                    /*n_cols=*/desc.m,
                    /*ldc=*/  desc.m,
                    ggml_result.out_ggml.data(),
                    zendnn_result.out_custom.data(),
                    zendnn_result.out_zendnn.data(),
                    atol, rtol);

            } else {
                // FP32 / BF16 — original two-way comparison.
                pass = two_way_compare(desc, ggml_result, zendnn_result);
            }

            if (!pass) overall_rc = 1;

        } else {
            // ================================================================
            //  BENCHMARK MODE
            // ================================================================
            BenchResult result = run_benchmark(desc);
            print_results(desc, result);
        }
    }

    return overall_rc;
}