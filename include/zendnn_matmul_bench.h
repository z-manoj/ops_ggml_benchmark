#pragma once

#include "op_desc.h"
#include "benchmark.h"

// ZenDNN-based matmul benchmark using oneDNN API.
// Uses dnnl::matmul primitive for C = A * B.
BenchResult bench_matmul_zendnn(const OpDesc& desc);
