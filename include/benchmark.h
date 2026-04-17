#pragma once
#include <vector>
#include <string>
#include "op_desc.h"

struct BenchResult {
    double min_ms          = 0.0;
    double avg_ms          = 0.0;
    double max_ms          = 0.0;
    double tflops          = 0.0;
    double ctx_creation_ms = 0.0;
    double op_creation_ms  = 0.0;
    double op_execution_ms = 0.0;
    double other_ms        = 0.0;

    // Primary output — always populated when verify_output=true.
    // For GGML backend  : holds the GGML compute result.
    // For ZenDNN backend: holds the ZenDNN result (backward compat).
    std::vector<float> output_data;

    // -----------------------------------------------------------------------
    // Three-way verification slots.
    // Populated ONLY by the ZenDNN backend when verify_output=true AND the
    // weight type is quantised (Q8_0 / Q4_0).
    //
    //   out_ggml   — GGML dequant reference  (computed internally, same data)
    //   out_custom — custom OMP scalar kernel
    //   out_zendnn — ZenDNN kernel result
    //
    // When these are non-empty, main.cpp switches to the three-way path.
    // -----------------------------------------------------------------------
    std::vector<float> out_ggml;
    std::vector<float> out_custom;
    std::vector<float> out_zendnn;
};

BenchResult run_benchmark(const OpDesc& desc);
void        print_results(const OpDesc& desc, const BenchResult& result);
void        write_csv_results(const std::string& csv_path,
                               const OpDesc&      desc,
                               const BenchResult& result,
                               bool               write_header);