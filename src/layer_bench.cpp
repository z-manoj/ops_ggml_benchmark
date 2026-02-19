#include "layer_bench.h"
#include "ggml_utils.h"

#include <cstdio>

void print_layer_results(const LayerConfig& cfg, const LayerBenchResult& result,
                         const std::string& backend, ggml_type dtype,
                         int threads, int warmup, int repeats) {
    printf("op: layer\n");
    printf("backend: %s\n", backend.c_str());
    printf("model: %s\n", cfg.model_name.c_str());
    printf("dtype: %s\n", dtype_to_string(dtype));
    printf("threads: %d\n", threads);
    printf("warmup: %d\n", warmup);
    printf("repeats: %d\n", repeats);
    printf("\n");

    printf("graph nodes: %zu\n", result.ops.size());
    for (size_t i = 0; i < result.ops.size(); i++) {
        const auto& op = result.ops[i];
        printf("  [%zu] %-12s %-16s m=%-5d n=%-5d k=%-5d",
               i, op.op_type.c_str(), op.label.c_str(),
               op.m, op.n, op.k);

        // Print per-op timing if available
        if (op.avg_ms > 0.0) {
            printf("  time(ms): min=%.2f avg=%.2f max=%.2f",
                   op.min_ms, op.avg_ms, op.max_ms);
        }
        printf("\n");
    }
    printf("\n");
    printf("total time(ms): min=%.2f avg=%.2f max=%.2f\n",
           result.min_ms, result.avg_ms, result.max_ms);
}
