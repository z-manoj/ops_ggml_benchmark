#pragma once
#include "benchmark.h"

// Approach 4: ZenDNN-backed matmul_id using ggml tensor scaffolding +
// direct lowoha::matmul_direct calls grouped per-expert.
BenchResult bench_matmul_id_approach4(const OpDesc& desc);

// Approach 4: ZenDNN-backed plain matmul (no routing).
BenchResult bench_matmul_approach4(const OpDesc& desc);
