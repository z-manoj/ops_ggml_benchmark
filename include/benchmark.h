#pragma once

#include "op_desc.h"
#include <string>

// Benchmark timing results with detailed breakdowns.
struct BenchResult {
    double min_ms  = 0.0;
    double avg_ms  = 0.0;
    double max_ms  = 0.0;
    double tflops  = 0.0;

    // Timing breakdowns (ZenDNN benchdnn format)
    double ctx_creation_ms = 0.0;     // Context/backend initialization
    double op_creation_ms  = 0.0;     // Operator/graph setup
    double op_execution_ms = 0.0;     // Actual compute time
    double other_ms        = 0.0;     // Other operations (memory allocation, etc.)
};

// Run the benchmark described by |desc| and return timing results.
// This is the main entry point called from main().
BenchResult run_benchmark(const OpDesc& desc);

// Print results in the benchdnn-inspired machine-parsable format.
void print_results(const OpDesc& desc, const BenchResult& result);

// Write results to CSV file (ZenDNN benchdnn format).
void write_csv_results(const std::string& csv_path, const OpDesc& desc, const BenchResult& result, bool write_header);
