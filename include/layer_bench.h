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
};

// Aggregate results from a full-layer benchmark.
struct LayerBenchResult {
    std::vector<LayerOpResult> ops;
    double total_gflops = 0.0;
    double min_ms       = 0.0;
    double avg_ms       = 0.0;
    double max_ms       = 0.0;
    double tflops       = 0.0;
};

// Run the layer benchmark described by |cfg|.
// |dtype|, |threads|, |warmup|, |repeats| are execution parameters.
LayerBenchResult bench_layer(const LayerConfig& cfg,
                             ggml_type dtype, int threads,
                             int warmup, int repeats);

// Print layer benchmark results.
void print_layer_results(const LayerConfig& cfg, const LayerBenchResult& result,
                         ggml_type dtype, int threads, int warmup, int repeats);
