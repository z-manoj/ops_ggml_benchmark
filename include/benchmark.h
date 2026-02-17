#pragma once

#include "op_desc.h"

// Benchmark timing results.
struct BenchResult {
    double min_ms  = 0.0;
    double avg_ms  = 0.0;
    double max_ms  = 0.0;
    double tflops  = 0.0;
};

// Run the benchmark described by |desc| and return timing results.
// This is the main entry point called from main().
BenchResult run_benchmark(const OpDesc& desc);

// Print results in the benchdnn-inspired machine-parsable format.
void print_results(const OpDesc& desc, const BenchResult& result);
