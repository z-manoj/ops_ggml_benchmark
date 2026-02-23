#include "ggml_utils.h"
#include "ggml-backend.h"
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>

ggml_type parse_dtype(const std::string& s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    return GGML_TYPE_COUNT;
}

const char* dtype_to_string(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_0: return "q4_0";
        default:             return "unknown";
    }
}

// Simple xorshift32 for deterministic, reproducible data.
static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void fill_tensor_deterministic(struct ggml_tensor* t, uint32_t seed, bool use_backend_set) {
    const size_t n_bytes = ggml_nbytes(t);

    // For backend set (repack buffer), we need temporary buffer
    void* temp_buffer = use_backend_set ? malloc(n_bytes) : t->data;
    if (!temp_buffer) {
        fprintf(stderr, "error: failed to allocate temporary buffer\n");
        return;
    }

    if (t->type == GGML_TYPE_F32) {
        float* data = reinterpret_cast<float*>(temp_buffer);
        const size_t n = n_bytes / sizeof(float);
        uint32_t state = seed;
        for (size_t i = 0; i < n; i++) {
            // Values in [-1, 1] to keep magnitudes reasonable.
            uint32_t r = xorshift32(state);
            data[i] = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
        }
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t* data = reinterpret_cast<ggml_fp16_t*>(temp_buffer);
        const size_t n = n_bytes / sizeof(ggml_fp16_t);
        uint32_t state = seed;
        for (size_t i = 0; i < n; i++) {
            uint32_t r = xorshift32(state);
            float val = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
            data[i] = ggml_fp32_to_fp16(val);
        }
    } else if (t->type == GGML_TYPE_BF16) {
        ggml_bf16_t* data = reinterpret_cast<ggml_bf16_t*>(temp_buffer);
        const size_t n = n_bytes / sizeof(ggml_bf16_t);
        uint32_t state = seed;
        for (size_t i = 0; i < n; i++) {
            uint32_t r = xorshift32(state);
            float val = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
            data[i] = ggml_fp32_to_bf16(val);
        }
    } else if (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_Q4_0) {
        // For quantized types, generate f32 data then quantize using GGML's API
        const int64_t n_elems = ggml_nelements(t);
        const int64_t n_per_row = t->ne[0];
        const int64_t nrows = n_elems / n_per_row;

        float* tmp_data = new float[n_elems];

        uint32_t state = seed;
        for (int64_t i = 0; i < n_elems; i++) {
            uint32_t r = xorshift32(state);
            tmp_data[i] = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
        }

        // Use GGML's public quantization API
        ggml_quantize_chunk(t->type, tmp_data, temp_buffer, 0, nrows, n_per_row, nullptr);

        delete[] tmp_data;
    } else {
        // For other types, zero-fill as a safe fallback.
        memset(temp_buffer, 0, n_bytes);
    }

    // If using backend set (for repack buffer), copy via ggml_backend_tensor_set
    // This triggers the q4_0 → q4_0x8 conversion in the repack buffer
    if (use_backend_set) {
        // Temporarily redirect stderr to suppress repack messages
        int stderr_fd = dup(STDERR_FILENO);
        int null_fd = open("/dev/null", O_WRONLY);
        if (null_fd >= 0) {
            dup2(null_fd, STDERR_FILENO);
            close(null_fd);
        }

        ggml_backend_tensor_set(t, temp_buffer, 0, n_bytes);

        // Restore stderr
        if (stderr_fd >= 0) {
            dup2(stderr_fd, STDERR_FILENO);
            close(stderr_fd);
        }

        free(temp_buffer);
    }
}

// Custom log callback that suppresses debug messages
static void ggml_log_callback_suppress_debug(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    // Only show warnings and errors, suppress info and debug
    if (level <= GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}

void suppress_ggml_debug_logs() {
    ggml_log_set(ggml_log_callback_suppress_debug, nullptr);
}
