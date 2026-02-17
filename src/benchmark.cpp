#include "benchmark.h"
#include "matmul_bench.h"
#include "ggml_utils.h"
#include <cstdio>
#include <cstdlib>

BenchResult run_benchmark(const OpDesc& desc) {
    if (desc.op_name == "matmul") {
        return bench_matmul(desc);
    } else if (desc.op_name == "matmul_id") {
        return bench_matmul_id(desc);
    } else {
        fprintf(stderr, "error: unknown operator '%s'\n", desc.op_name.c_str());
        exit(1);
    }
}

void print_results(const OpDesc& desc, const BenchResult& result) {
    printf("op: %s\n", desc.op_name.c_str());
    printf("dtype: %s\n", dtype_to_string(desc.dtype));
    printf("shape: m=%d n=%d k=%d\n", desc.m, desc.n, desc.k);
    printf("threads: %d\n", desc.threads);
    printf("warmup: %d\n", desc.warmup);
    printf("repeats: %d\n", desc.repeats);
    printf("\n");
    printf("time(ms): min=%.2f avg=%.2f max=%.2f\n",
           result.min_ms, result.avg_ms, result.max_ms);
    printf("throughput: %.1f TFLOPS\n", result.tflops);
}
