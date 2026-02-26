#pragma once

#include "ggml.h"
#include <string>
#include <vector>

// Operator descriptor -- mirrors benchdnn problem descriptors.
// Carries the shape, data type, and execution parameters for a single
// operator benchmark invocation.
struct OpDesc {
    std::string op_name;       // "matmul" or "matmul_id"
    std::string backend = "ggml";  // "ggml" or "zendnn"
    int         m       = 512;
    int         n       = 512;
    int         k       = 512;

    // Separate data types for source and weight (output is always F32)
    ggml_type   src_dtype = GGML_TYPE_F32;  // Input/source data type (f32 or bf16)
    ggml_type   wei_dtype = GGML_TYPE_F32;  // Weight data type

    int         threads = 4;
    int         repeats = 100;
    int         warmup  = 10;

    // matmul_id specific
    int         n_experts    = 1;   // total expert count (default: 1 for simplicity)
    int         n_experts_used = 1; // experts selected per token (default: 1)

    // matmul_id routing configuration
    std::string routing_pattern = "uniform";      // "uniform", "custom", "random", "skewed"
    std::vector<int> expert_token_counts;         // explicit per-expert counts: [24, 30, 15, ...]
    int         routing_seed = 42;                // for reproducible random routing
    std::string routing_ids_file;                 // optional: load pre-generated routing

    // Verification mode
    bool        verify_output = false;            // save output data for comparison
};
