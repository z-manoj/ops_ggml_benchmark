#pragma once

#include "ggml.h"

// Standalone MoE (mul_mat_id) operation using OpenMP.
//
// Tensor layout (matches ggml_mul_mat_id):
//   experts : [K, N, n_experts]           weight matrices (f32 or f16)
//   input   : [K, n_experts_used, n_tok]  input vectors   (f32)
//   ids     : [n_experts_used, n_tok]     routing indices  (i32)
//   dst     : [N, n_experts_used, n_tok]  output           (f32)
//
// dst must be pre-allocated with the correct shape.
// n_threads controls OpenMP parallelism.
void custom_moe_compute(struct ggml_tensor* dst,
                        const struct ggml_tensor* experts,
                        const struct ggml_tensor* input,
                        const struct ggml_tensor* ids,
                        int n_threads);

// Standalone matmul using OpenMP (for comparison / layer mode).
//
// Tensor layout (matches ggml_mul_mat):
//   A   : [K, M]   weight (f32 or f16)
//   B   : [K, N]   input  (f32)
//   dst : [M, N]   output (f32)
void custom_matmul_compute(struct ggml_tensor* dst,
                           const struct ggml_tensor* A,
                           const struct ggml_tensor* B,
                           int n_threads);
