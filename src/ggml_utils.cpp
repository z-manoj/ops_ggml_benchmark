#include "ggml_utils.h"
#include <cstring>
#include <cstdint>

ggml_type parse_dtype(const std::string& s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    return GGML_TYPE_COUNT;
}

const char* dtype_to_string(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_F16: return "f16";
        default:            return "unknown";
    }
}

// Simple xorshift32 for deterministic, reproducible data.
static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void fill_tensor_deterministic(struct ggml_tensor* t, uint32_t seed) {
    const size_t n_bytes = ggml_nbytes(t);

    if (t->type == GGML_TYPE_F32) {
        float* data = reinterpret_cast<float*>(t->data);
        const size_t n = n_bytes / sizeof(float);
        uint32_t state = seed;
        for (size_t i = 0; i < n; i++) {
            // Values in [-1, 1] to keep magnitudes reasonable.
            uint32_t r = xorshift32(state);
            data[i] = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
        }
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t* data = reinterpret_cast<ggml_fp16_t*>(t->data);
        const size_t n = n_bytes / sizeof(ggml_fp16_t);
        uint32_t state = seed;
        for (size_t i = 0; i < n; i++) {
            uint32_t r = xorshift32(state);
            float val = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
            data[i] = ggml_fp32_to_fp16(val);
        }
    } else {
        // For other types, zero-fill as a safe fallback.
        memset(t->data, 0, n_bytes);
    }
}
