#pragma once

#include "op_desc.h"
#include "benchmark.h"
#include "ggml.h"
#include "ggml-backend.h"

// Execute a ggml_mul_mat benchmark.
// Creates tensors, builds a single-node GGML graph, and runs
// warmup + timed iterations using the CPU backend.
BenchResult bench_matmul(const OpDesc& desc);

// Execute a ggml_mul_mat_id benchmark (MoE expert-routed matmul).
BenchResult bench_matmul_id(const OpDesc& desc);
