#pragma once

#include "benchmark.h"
#include "layer_bench.h"
#include "layer_config.h"
#include "ggml.h"

// Benchmark custom MoE standalone op (same tensor layout as ggml_mul_mat_id).
BenchResult bench_custom_moe(const OpDesc& desc);

// Benchmark full layer using custom ops (custom matmul + custom MoE).
LayerBenchResult bench_custom_layer(const LayerConfig& cfg,
                                    ggml_type dtype, int threads,
                                    int warmup, int repeats);
