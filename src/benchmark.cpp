#include "benchmark.h"
#include "ggml_matmul_bench.h"
#include "ggml_utils.h"

#ifdef ENABLE_ZENDNN
#include "zendnn_matmul_bench.h"
#endif

#include <cstdio>
#include <cstdlib>

BenchResult run_benchmark(const OpDesc& desc) {
    // Route to backend-specific implementation
    if (desc.backend == "zendnn") {
#ifdef ENABLE_ZENDNN
        if (desc.op_name == "matmul") {
            return bench_matmul_zendnn(desc);
        } else if (desc.op_name == "matmul_id") {
            fprintf(stderr, "error: matmul_id not yet implemented for ZenDNN backend\n");
            exit(1);
        } else {
            fprintf(stderr, "error: unknown operator '%s'\n", desc.op_name.c_str());
            exit(1);
        }
#else
        fprintf(stderr, "error: ZenDNN backend not enabled. Rebuild with -DENABLE_ZENDNN=ON\n");
        exit(1);
#endif
    } else if (desc.backend == "ggml") {
        if (desc.op_name == "matmul") {
            return bench_matmul_ggml(desc);
        } else if (desc.op_name == "matmul_id") {
            return bench_matmul_id_ggml(desc);
        } else {
            fprintf(stderr, "error: unknown operator '%s'\n", desc.op_name.c_str());
            exit(1);
        }
    } else {
        fprintf(stderr, "error: unknown backend '%s'\n", desc.backend.c_str());
        exit(1);
    }
}

void print_results(const OpDesc& desc, const BenchResult& result) {
    printf("op: %s\n", desc.op_name.c_str());
    printf("backend: %s\n", desc.backend.c_str());
    printf("dtype: %s\n", dtype_to_string(desc.dtype));
    printf("shape: m=%d n=%d k=%d\n", desc.m, desc.n, desc.k);
    printf("threads: %d\n", desc.threads);
    printf("warmup: %d\n", desc.warmup);
    printf("repeats: %d\n", desc.repeats);
    printf("\n");
    printf("time(ms): min=%.2f avg=%.2f max=%.2f\n",
           result.min_ms, result.avg_ms, result.max_ms);
}
