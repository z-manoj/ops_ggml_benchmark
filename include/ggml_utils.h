#pragma once

#include "ggml.h"
#include <string>

// Convert a CLI dtype string ("f32", "f16") to a ggml_type.
// Returns GGML_TYPE_COUNT on failure.
ggml_type parse_dtype(const std::string& s);

// Convert a ggml_type to a human-readable string.
const char* dtype_to_string(ggml_type t);

// Fill a tensor with deterministic pseudo-random float data.
// The tensor must already have its data buffer allocated.
void fill_tensor_deterministic(struct ggml_tensor* t, uint32_t seed);
