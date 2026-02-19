#pragma once

#include "ggml.h"
#include <string>

// Operator descriptor -- mirrors benchdnn problem descriptors.
// Carries the shape, data type, and execution parameters for a single
// operator benchmark invocation.
struct OpDesc {
    std::string op_name;       // "matmul" or "matmul_id"
    std::string backend = "ggml";  // "ggml" or "zendnn"
    int         m       = 512;
    int         n       = 512;
    int         k       = 512;
    ggml_type   dtype   = GGML_TYPE_F32;
    int         threads = 4;
    int         repeats = 100;
    int         warmup  = 10;

    // matmul_id specific
    int         n_experts    = 8;   // total expert count
    int         n_experts_used = 2; // experts selected per token
};
