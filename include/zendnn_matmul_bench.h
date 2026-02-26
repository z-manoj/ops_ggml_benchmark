#pragma once

#include "op_desc.h"
#include "benchmark.h"

// ZenDNN-based matmul benchmark using oneDNN API.
// Uses dnnl::matmul primitive for C = A * B.
BenchResult bench_matmul_zendnn(const OpDesc& desc);

// ZenDNN-based matmul_id benchmark (MoE expert-routed matmul).
// Uses group_gemm_direct in parallel mode for expert routing.
BenchResult bench_matmul_id_zendnn(const OpDesc& desc);
