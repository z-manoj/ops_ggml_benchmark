#include "op_desc.h"
#include "benchmark.h"
#include "ggml_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --op <matmul|matmul_id>   Operator to benchmark (default: matmul)\n"
        "  --m <int>                 M dimension (default: 512)\n"
        "  --n <int>                 N dimension (default: 512)\n"
        "  --k <int>                 K dimension (default: 512)\n"
        "  --dtype <f32|f16>         Data type (default: f32)\n"
        "  --threads <int>           Thread count (default: hw concurrency)\n"
        "  --repeats <int>           Timed iterations (default: 100)\n"
        "  --warmup <int>            Warmup iterations (default: 10)\n"
        "  --help                    Show this message\n",
        prog);
}

int main(int argc, char** argv) {
    OpDesc desc;
    desc.op_name = "matmul";
    desc.threads = static_cast<int>(std::thread::hardware_concurrency());
    if (desc.threads < 1) desc.threads = 4;

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

        if (arg("--op"))      { desc.op_name = next(); }
        else if (arg("--m"))  { desc.m = atoi(next()); }
        else if (arg("--n"))  { desc.n = atoi(next()); }
        else if (arg("--k"))  { desc.k = atoi(next()); }
        else if (arg("--dtype")) {
            std::string s = next();
            desc.dtype = parse_dtype(s);
            if (desc.dtype == GGML_TYPE_COUNT) {
                fprintf(stderr, "error: unknown dtype '%s'\n", s.c_str());
                return 1;
            }
        }
        else if (arg("--threads")) { desc.threads = atoi(next()); }
        else if (arg("--repeats")) { desc.repeats = atoi(next()); }
        else if (arg("--warmup"))  { desc.warmup  = atoi(next()); }
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

    if (desc.m <= 0 || desc.n <= 0 || desc.k <= 0) {
        fprintf(stderr, "error: dimensions must be positive\n");
        return 1;
    }
    if (desc.threads <= 0) {
        fprintf(stderr, "error: thread count must be positive\n");
        return 1;
    }

    BenchResult result = run_benchmark(desc);
    print_results(desc, result);
    return 0;
}
