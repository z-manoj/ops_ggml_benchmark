#include "op_desc.h"
#include "benchmark.h"
#include "ggml_utils.h"
#include "layer_config.h"
#include "layer_bench.h"
#include "custom_moe_bench.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        "  --op <matmul|matmul_id|layer|custom_moe|custom_layer>\n"
        "                            Operator to benchmark (default: matmul)\n"
        "  --config <file>           Layer config file (required for layer/custom_layer)\n"
        "  --m <int>                 M dimension (default: 512)\n"
        "  --n <int>                 N dimension (default: 512)\n"
        "  --k <int>                 K dimension (default: 512)\n"
        "  --shapes <list>           Comma-separated MxNxK shapes\n"
        "                            e.g. 4096x4096x4096,1024x1024x1024\n"
        "  --batch <file>            Read shapes from file (one per line)\n"
        "                            Lines: \"M N K\" or \"MxNxK\", # for comments\n"
        "  --dtype <f32|f16>         Data type (default: f32)\n"
        "  --threads <int>           Thread count (default: hw concurrency)\n"
        "  --repeats <int>           Timed iterations (default: 100)\n"
        "  --warmup <int>            Warmup iterations (default: 10)\n"
        "  --help                    Show this message\n",
        prog);
}

int main(int argc, char** argv) {
    OpDesc base;
    base.op_name = "matmul";
    base.threads = static_cast<int>(std::thread::hardware_concurrency());
    if (base.threads < 1) base.threads = 4;

    std::vector<Shape> shapes;
    std::string batch_file;
    std::string shapes_arg;
    std::string config_file;

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
        else if (arg("--m"))  { base.m = atoi(next()); }
        else if (arg("--n"))  { base.n = atoi(next()); }
        else if (arg("--k"))  { base.k = atoi(next()); }
        else if (arg("--shapes"))  { shapes_arg = next(); }
        else if (arg("--batch"))   { batch_file = next(); }
        else if (arg("--dtype")) {
            std::string s = next();
            base.dtype = parse_dtype(s);
            if (base.dtype == GGML_TYPE_COUNT) {
                fprintf(stderr, "error: unknown dtype '%s'\n", s.c_str());
                return 1;
            }
        }
        else if (arg("--threads")) { base.threads = atoi(next()); }
        else if (arg("--repeats")) { base.repeats = atoi(next()); }
        else if (arg("--warmup"))  { base.warmup  = atoi(next()); }
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

    // --- Layer benchmark mode ---
    if (base.op_name == "layer") {
        if (config_file.empty()) {
            fprintf(stderr, "error: --config <file> is required when --op layer\n");
            return 1;
        }
        LayerConfig cfg = parse_layer_config(config_file);
        LayerBenchResult result = bench_layer(cfg, base.dtype, base.threads,
                                              base.warmup, base.repeats);
        print_layer_results(cfg, result, base.dtype, base.threads,
                            base.warmup, base.repeats);
        return 0;
    }

    // --- Custom MoE standalone op ---
    if (base.op_name == "custom_moe") {
        if (shapes.empty()) shapes.push_back({base.m, base.n, base.k});
        OpDesc desc = base;
        desc.m = shapes[0].m;
        desc.n = shapes[0].n;
        desc.k = shapes[0].k;

        BenchResult result = bench_custom_moe(desc);
        printf("op: custom_moe (OpenMP)\n");
        printf("dtype: %s\n", dtype_to_string(desc.dtype));
        printf("shape: m=%d n=%d k=%d\n", desc.m, desc.n, desc.k);
        printf("experts: %d total, %d used\n", desc.n_experts, desc.n_experts_used);
        printf("threads: %d\n", desc.threads);
        printf("warmup: %d\n", desc.warmup);
        printf("repeats: %d\n", desc.repeats);
        printf("\n");
        printf("time(ms): min=%.2f avg=%.2f max=%.2f\n",
               result.min_ms, result.avg_ms, result.max_ms);
        printf("throughput: %.2f TFLOPS\n", result.tflops);
        return 0;
    }

    // --- Custom layer (all ops via custom OpenMP kernels) ---
    if (base.op_name == "custom_layer") {
        if (config_file.empty()) {
            fprintf(stderr, "error: --config <file> is required when --op custom_layer\n");
            return 1;
        }
        LayerConfig cfg = parse_layer_config(config_file);
        LayerBenchResult result = bench_custom_layer(cfg, base.dtype, base.threads,
                                                     base.warmup, base.repeats);
        printf("op: custom_layer (OpenMP)\n");
        print_layer_results(cfg, result, base.dtype, base.threads,
                            base.warmup, base.repeats);
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

    // Run each shape
    for (size_t i = 0; i < shapes.size(); i++) {
        if (i > 0) printf("\n---\n\n");

        OpDesc desc = base;
        desc.m = shapes[i].m;
        desc.n = shapes[i].n;
        desc.k = shapes[i].k;

        BenchResult result = run_benchmark(desc);
        print_results(desc, result);
    }

    return 0;
}
