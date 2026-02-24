#pragma once

#include "layer_config.h"
#include "ggml.h"

#include <string>
#include <vector>

// Per-op detail for layer benchmark output.
struct LayerOpResult {
    std::string op_type;
    std::string label;
    int         m      = 0;
    int         n      = 0;
    int         k      = 0;
    double      gflops = 0.0;  // GFLOPs for this single op
    double      min_ms = 0.0;  // Min time for this op across repeats
    double      avg_ms = 0.0;  // Avg time for this op across repeats
    double      max_ms = 0.0;  // Max time for this op across repeats
    double      tflops = 0.0;  // Throughput for this op
};

// Aggregate results from a full-layer benchmark.
struct LayerBenchResult {
    std::vector<LayerOpResult> ops;
    double total_gflops = 0.0;
    double min_ms       = 0.0;
    double avg_ms       = 0.0;
    double max_ms       = 0.0;
    double tflops       = 0.0;

    // Timing breakdowns (ZenDNN benchdnn format)
    double ctx_creation_ms = 0.0;
    double op_creation_ms  = 0.0;
    double op_execution_ms = 0.0;
    double other_ms        = 0.0;
};

// Run the layer benchmark described by |cfg| using GGML backend.
// Supports both mul_mat and mul_mat_id operations.
LayerBenchResult bench_layer_ggml(const LayerConfig& cfg,
                                  ggml_type dtype, int threads,
                                  int warmup, int repeats);

#ifdef ENABLE_ZENDNN
// Run the layer benchmark described by |cfg| using ZenDNN backend.
// Only supports mul_mat operations (non-MoE models).
// Will error if config contains mul_mat_id ops.
LayerBenchResult bench_layer_zendnn(const LayerConfig& cfg,
                                    ggml_type dtype, int threads,
                                    int warmup, int repeats);
#endif

// Print layer benchmark results.
void print_layer_results(const LayerConfig& cfg, const LayerBenchResult& result,
                         const std::string& backend, ggml_type dtype,
                         int threads, int warmup, int repeats);

// Write layer results to CSV file.
void write_layer_csv_results(const std::string& csv_path, const LayerConfig& cfg,
                             const LayerBenchResult& result, const std::string& backend,
                             ggml_type dtype, int threads, int warmup, int repeats,
                             bool write_header);
