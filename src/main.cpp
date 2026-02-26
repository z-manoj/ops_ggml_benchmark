#include "op_desc.h"
#include "benchmark.h"
#include "ggml_utils.h"
#include "layer_config.h"
#include "layer_bench.h"
#include "moe_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct Shape { int m, n, k; };

// Parse "MxNxK" (e.g. "4096x4096x4096") into a Shape.
static bool parse_shape(const std::string& s, Shape& out) {
    int m, n, k;
    char x1, x2;
    std::istringstream ss(s);
    if (ss >> m >> x1 >> n >> x2 >> k && x1 == 'x' && x2 == 'x' &&
        m > 0 && n > 0 && k > 0) {
        out = {m, n, k};
        return true;
    }
    return false;
}

// Parse comma-separated shapes: "4096x4096x4096,1024x1024x1024"
static std::vector<Shape> parse_shapes_list(const std::string& s) {
    std::vector<Shape> shapes;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        Shape sh;
        if (!parse_shape(token, sh)) {
            fprintf(stderr, "error: invalid shape '%s' (expected MxNxK)\n", token.c_str());
            exit(1);
        }
        shapes.push_back(sh);
    }
    return shapes;
}

// Read shapes from a batch file.
// Each line is either "M N K" or "MxNxK". Blank lines and '#' comments are skipped.
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
        // Strip leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue; // blank line
        line = line.substr(start);
        if (line[0] == '#') continue; // comment

        Shape sh;
        // Try "MxNxK" first
        if (parse_shape(line, sh)) {
            shapes.push_back(sh);
            continue;
        }
        // Try "M N K" (whitespace separated)
        std::istringstream ss(line);
        int m, n, k;
        if (ss >> m >> n >> k && m > 0 && n > 0 && k > 0) {
            shapes.push_back({m, n, k});
            continue;
        }
        fprintf(stderr, "error: %s:%d: invalid line '%s'\n", path.c_str(), lineno, line.c_str());
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
        "                            Operator to benchmark (default: matmul)\n"
        "  --config <file>           Layer config file (required when --op layer)\n"
        "  --moe-config <file>       MoE config file (.moe format) for matmul_id\n"
        "  --m <int>                 M dimension (default: 512)\n"
        "  --n <int>                 N dimension (default: 512)\n"
        "  --k <int>                 K dimension (default: 512)\n"
        "  --shapes <list>           Comma-separated MxNxK shapes\n"
        "                            e.g. 4096x4096x4096,1024x1024x1024\n"
        "  --batch <file>            Read shapes from file (one per line)\n"
        "                            Lines: \"M N K\" or \"MxNxK\", # for comments\n"
        "  --src_dtype <f32|bf16>\n"
        "                            Source/input data type (default: f32)\n"
        "  --wei_dtype <f32|f16|bf16|q8_0|q4_0>\n"
        "                            Weight data type (default: f32)\n"
        "                            q8_0/q4_0: quantized weights (ggml only)\n"
        "  --dtype <f32|f16|bf16|q8_0|q4_0>\n"
        "                            Set both src and wei dtypes (deprecated)\n"
        "  --backend <ggml|zendnn>   Backend to use (default: ggml)\n"
        "  --threads <int>           Thread count (default: hw concurrency)\n"
        "  --repeats <int>           Timed iterations (default: 100)\n"
        "  --warmup <int>            Warmup iterations (default: 10)\n"
        "  --n_experts <int>         Total expert count for matmul_id (default: 1)\n"
        "  --n_experts_used <int>    Active experts per token for matmul_id (default: 1)\n"
        "  --routing_pattern <pattern>\n"
        "                            Routing pattern for matmul_id (default: uniform)\n"
        "                            Options: uniform, custom, random, skewed\n"
        "  --expert_token_counts <counts>\n"
        "                            Comma-separated per-expert token counts for custom routing\n"
        "                            e.g., 24,30,15,20,18,22,25,26\n"
        "  --routing_seed <int>      Seed for random/skewed routing (default: 42)\n"
        "  --routing_ids_file <file> Load pre-generated routing IDs from file\n"
        "  --help                    Show this message\n",
        prog);
}

int main(int argc, char** argv) {
    // Suppress GGML debug logs (repack messages, etc.)
    suppress_ggml_debug_logs();

    OpDesc base;
    base.op_name = "matmul";
    base.threads = static_cast<int>(std::thread::hardware_concurrency());
    if (base.threads < 1) base.threads = 4;

    std::vector<Shape> shapes;
    std::string batch_file;
    std::string shapes_arg;
    std::string config_file;
    std::string moe_config_file;

    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* name) {
            return strcmp(argv[i], name) == 0;
        };
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: missing value for %s\n", argv[i]);
                exit(1);
            }
            return argv[++i];
        };

        if (arg("--op"))      { base.op_name = next(); }
        else if (arg("--config")) { config_file = next(); }
        else if (arg("--moe-config")) { moe_config_file = next(); }
        else if (arg("--backend")) { base.backend = next(); }
        else if (arg("--m"))  { base.m = atoi(next()); }
        else if (arg("--n"))  { base.n = atoi(next()); }
        else if (arg("--k"))  { base.k = atoi(next()); }
        else if (arg("--shapes"))  { shapes_arg = next(); }
        else if (arg("--batch"))   { batch_file = next(); }
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
            // Backward compatibility: set both src and wei dtypes to the same value
            std::string s = next();
            ggml_type dt = parse_dtype(s);
            if (dt == GGML_TYPE_COUNT) {
                fprintf(stderr, "error: unknown dtype '%s'\n", s.c_str());
                return 1;
            }
            base.src_dtype = dt;
            base.wei_dtype = dt;
        }
        else if (arg("--threads")) { base.threads = atoi(next()); }
        else if (arg("--repeats")) { base.repeats = atoi(next()); }
        else if (arg("--warmup"))  { base.warmup  = atoi(next()); }
        else if (arg("--n_experts")) { base.n_experts = atoi(next()); }
        else if (arg("--n_experts_used")) { base.n_experts_used = atoi(next()); }
        else if (arg("--routing_pattern")) { base.routing_pattern = next(); }
        else if (arg("--routing_seed")) { base.routing_seed = atoi(next()); }
        else if (arg("--routing_ids_file")) { base.routing_ids_file = next(); }
        else if (arg("--expert_token_counts")) {
            std::string counts_str = next();
            // Parse comma-separated: "24,30,15,20,18,22,25,26"
            std::istringstream ss(counts_str);
            std::string token;
            base.expert_token_counts.clear();
            while (std::getline(ss, token, ',')) {
                base.expert_token_counts.push_back(atoi(token.c_str()));
            }
        }
        else if (arg("--verify")) { base.verify_output = true; }
        else if (arg("--help") || arg("-h")) {
            print_usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (base.threads <= 0) {
        fprintf(stderr, "error: thread count must be positive\n");
        return 1;
    }

    if (base.backend != "ggml" && base.backend != "zendnn") {
        fprintf(stderr, "error: unknown backend '%s' (must be 'ggml' or 'zendnn')\n", base.backend.c_str());
        return 1;
    }

    // Validate src_dtype (only f32 and bf16 supported)
    if (base.src_dtype != GGML_TYPE_F32 && base.src_dtype != GGML_TYPE_BF16) {
        fprintf(stderr, "error: src_dtype must be f32 or bf16\n");
        return 1;
    }

    // Validate dtypes for ZenDNN backend
    if (base.backend == "zendnn") {
        if (base.wei_dtype == GGML_TYPE_F16) {
            fprintf(stderr, "error: ZenDNN backend does not support f16 weights (only f32 and bf16 supported)\n");
            return 1;
        }
    }

    // Validate quantized dtypes (only supported on GGML backend for weights)
    if ((base.wei_dtype == GGML_TYPE_Q8_0 || base.wei_dtype == GGML_TYPE_Q4_0) && base.backend != "ggml") {
        fprintf(stderr, "error: quantized dtypes (q8_0, q4_0) are only supported on ggml backend\n");
        return 1;
    }

    // Validate that quantized types are only used for weights
    if (base.src_dtype == GGML_TYPE_Q8_0 || base.src_dtype == GGML_TYPE_Q4_0 ||
        base.src_dtype == GGML_TYPE_F16) {
        fprintf(stderr, "error: src_dtype can only be f32 or bf16 (not f16, q8_0, or q4_0)\n");
        return 1;
    }

    // GGML backend: For mixed data types, src must be F32
    if (base.backend == "ggml") {
        if (base.src_dtype != base.wei_dtype) {
            // Mixed data types - src must be F32
            if (base.src_dtype != GGML_TYPE_F32) {
                fprintf(stderr, "error: GGML backend with mixed data types requires src_dtype=f32\n");
                fprintf(stderr, "       Current: src=%s, wei=%s\n",
                        dtype_to_string(base.src_dtype), dtype_to_string(base.wei_dtype));
                fprintf(stderr, "       Supported combinations:\n");
                fprintf(stderr, "         - Same type: src=f32 wei=f32, OR src=bf16 wei=bf16\n");
                fprintf(stderr, "         - Mixed type: src=f32 wei=f16/bf16/q8_0/q4_0\n");
                return 1;
            }
        }
    }

    // GGML type compatibility: quantized weights require F32 input
    if (base.backend == "ggml") {
        if ((base.wei_dtype == GGML_TYPE_Q8_0 || base.wei_dtype == GGML_TYPE_Q4_0) &&
            base.src_dtype != GGML_TYPE_F32) {
            fprintf(stderr, "error: quantized weights (q8_0, q4_0) require f32 input for GGML backend\n");
            return 1;
        }
    }

    // --- MoE config mode (.moe file) ---
    if (!moe_config_file.empty()) {
        OpDesc desc = parse_moe_config(moe_config_file);

        // Run the matmul_id benchmark
        BenchResult result = run_benchmark(desc);
        print_results(desc, result);

        return 0;
    }

    // --- Layer benchmark mode ---
    if (base.op_name == "layer") {
        if (config_file.empty()) {
            fprintf(stderr, "error: --config <file> is required when --op layer\n");
            return 1;
        }

        // Type compatibility validation for layer mode
        if (base.backend == "ggml") {
            if (base.src_dtype != base.wei_dtype) {
                // Mixed data types - src must be F32
                if (base.src_dtype != GGML_TYPE_F32) {
                    fprintf(stderr, "error: GGML backend with mixed data types requires src_dtype=f32 for layer mode\n");
                    fprintf(stderr, "       Current: src=%s, wei=%s\n",
                            dtype_to_string(base.src_dtype), dtype_to_string(base.wei_dtype));
                    return 1;
                }
            }
        }

        // Generate timestamped CSV filename for layer benchmark (disabled for now)
        // char timestamp[32];
        // time_t now = time(nullptr);
        // struct tm* tm_info = localtime(&now);
        // strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);
        // std::string csv_filename = "layer_timings_" + std::string(timestamp) + ".csv";

        LayerConfig cfg = parse_layer_config(config_file);
        LayerBenchResult result;

        if (base.backend == "zendnn") {
#ifdef ENABLE_ZENDNN
            result = bench_layer_zendnn(cfg, base.wei_dtype, base.threads,
                                       base.warmup, base.repeats, base.src_dtype);
#else
            fprintf(stderr, "error: ZenDNN backend not enabled. Rebuild with -DENABLE_ZENDNN=ON\n");
            return 1;
#endif
        } else {
            result = bench_layer_ggml(cfg, base.wei_dtype, base.threads,
                                     base.warmup, base.repeats, base.src_dtype);
        }

        print_layer_results(cfg, result, base.backend, base.wei_dtype, base.threads,
                            base.warmup, base.repeats);

        // Write to CSV (disabled for now)
        // write_layer_csv_results(csv_filename, cfg, result, base.backend, base.dtype,
        //                        base.threads, base.warmup, base.repeats, true);
        // printf("Results saved to: %s\n", csv_filename.c_str());

        return 0;
    }

    // Collect shapes from --batch and --shapes (they can be combined)
    if (!batch_file.empty()) {
        auto batch_shapes = read_batch_file(batch_file);
        shapes.insert(shapes.end(), batch_shapes.begin(), batch_shapes.end());
    }
    if (!shapes_arg.empty()) {
        auto cli_shapes = parse_shapes_list(shapes_arg);
        shapes.insert(shapes.end(), cli_shapes.begin(), cli_shapes.end());
    }

    // If no --batch or --shapes given, use the single --m/--n/--k values
    if (shapes.empty()) {
        if (base.m <= 0 || base.n <= 0 || base.k <= 0) {
            fprintf(stderr, "error: dimensions must be positive\n");
            return 1;
        }
        shapes.push_back({base.m, base.n, base.k});
    }

    // Generate timestamped CSV filename (disabled for now)
    // char timestamp[32];
    // time_t now = time(nullptr);
    // struct tm* tm_info = localtime(&now);
    // strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);
    // std::string csv_filename = "timings_" + std::string(timestamp) + ".csv";
    // printf("Results will be saved to: %s\n\n", csv_filename.c_str());

    // Run each shape
    for (size_t i = 0; i < shapes.size(); i++) {
        if (i > 0) printf("\n");

        OpDesc desc = base;
        desc.m = shapes[i].m;
        desc.n = shapes[i].n;
        desc.k = shapes[i].k;

        // Verification mode: run both backends and compare outputs
        if (desc.verify_output) {
            printf("========================================================================\n");
            printf("VERIFICATION MODE: Comparing GGML vs ZenDNN outputs\n");
            printf("========================================================================\n\n");

            // Run GGML
            desc.backend = "ggml";
            BenchResult ggml_result = run_benchmark(desc);
            printf("GGML:   ");
            print_results(desc, ggml_result);

            // Run ZenDNN
            desc.backend = "zendnn";
            BenchResult zendnn_result = run_benchmark(desc);
            printf("ZenDNN: ");
            print_results(desc, zendnn_result);

            // Compare outputs
            if (ggml_result.output_data.empty() || zendnn_result.output_data.empty()) {
                fprintf(stderr, "\nERROR: Output data not captured for verification\n");
                return 1;
            }

            if (ggml_result.output_data.size() != zendnn_result.output_data.size()) {
                fprintf(stderr, "\nERROR: Output sizes don't match! GGML=%zu, ZenDNN=%zu\n",
                       ggml_result.output_data.size(), zendnn_result.output_data.size());
                return 1;
            }

            size_t n = ggml_result.output_data.size();
            double max_rel_diff = 0.0;
            double avg_rel_diff = 0.0;
            size_t num_mismatches = 0;
            // Relative tolerance based on data type: BF16 has only 7-bit mantissa
            // Accumulation errors in large matmuls can compound
            const double rel_tolerance = (desc.src_dtype == GGML_TYPE_BF16 || desc.wei_dtype == GGML_TYPE_BF16)
                                         ? 0.05  // 5% for BF16 (acceptable rounding + accumulation error)
                                         : 1e-4; // 0.01% for F32

            for (size_t j = 0; j < n; j++) {
                double abs_diff = std::abs(ggml_result.output_data[j] - zendnn_result.output_data[j]);
                double magnitude = std::max(std::abs(ggml_result.output_data[j]), std::abs(zendnn_result.output_data[j]));
                double rel_diff = (magnitude > 1e-6) ? (abs_diff / magnitude) : abs_diff;

                avg_rel_diff += rel_diff;
                if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;
                if (rel_diff > rel_tolerance) num_mismatches++;
            }
            avg_rel_diff /= n;

            printf("\n========================================================================\n");
            printf("VERIFICATION RESULTS:\n");
            printf("========================================================================\n");
            printf("Total elements:     %zu\n", n);
            printf("Max rel difference: %.6f%% \n", max_rel_diff * 100);
            printf("Avg rel difference: %.6f%%\n", avg_rel_diff * 100);
            printf("Rel tolerance:      %.6f%%\n", rel_tolerance * 100);
            printf("Mismatches:         %zu (%.2f%%)\n", num_mismatches, 100.0 * num_mismatches / n);

            if (num_mismatches == 0) {
                printf("\n✓ PASS: Outputs match within tolerance!\n");
            } else {
                printf("\n✗ FAIL: %zu elements exceed relative tolerance\n", num_mismatches);
                // Print first few mismatches
                size_t printed = 0;
                for (size_t j = 0; j < n && printed < 10; j++) {
                    double abs_diff = std::abs(ggml_result.output_data[j] - zendnn_result.output_data[j]);
                    double magnitude = std::max(std::abs(ggml_result.output_data[j]), std::abs(zendnn_result.output_data[j]));
                    double rel_diff = (magnitude > 1e-6) ? (abs_diff / magnitude) : abs_diff;

                    if (rel_diff > rel_tolerance) {
                        printf("  [%zu]: GGML=%.6f, ZenDNN=%.6f, rel_diff=%.2f%%\n",
                               j, ggml_result.output_data[j], zendnn_result.output_data[j], rel_diff * 100);
                        printed++;
                    }
                }
                if (num_mismatches > 10) {
                    printf("  ... (%zu more mismatches)\n", num_mismatches - 10);
                }
                return 1;
            }
            printf("========================================================================\n");
        } else {
            // Normal mode: just run the benchmark
            BenchResult result = run_benchmark(desc);
            print_results(desc, result);
        }

        // Write to CSV (write header only for first entry)
        // write_csv_results(csv_filename, desc, result, i == 0);
    }

    // printf("\nResults saved to: %s\n", csv_filename.c_str());
    return 0;
}
