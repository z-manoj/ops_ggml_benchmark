#include "ggml_unpack_weight_buffer.h"
#include "ggml_utils.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstring>
#include <cmath>
#include <omp.h>

#define QK8_0 32
#define QK4_0 32

typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;
typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0;
typedef struct { uint16_t d[8]; uint8_t qs[128]; } block_q4_0x8;

static uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

template <typename T>
static void fill_buffer(T* data, size_t n, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        uint32_t r = xorshift32(state);
        float val = static_cast<float>(static_cast<int32_t>(r)) / static_cast<float>(INT32_MAX);
        data[i] = static_cast<T>(val);
    }
}

static void generate_native_q4_0x8_superblocks(int64_t m, int64_t k, uint32_t seed, std::vector<block_q4_0x8>& weights) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "error: failed to init CPU backend\n");
        exit(1);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    bool using_repack = false;

    ggml_backend_dev_t cpu_dev = ggml_backend_get_device(backend);
    if (cpu_dev) {
        ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        if (cpu_reg) {
            typedef ggml_backend_buffer_type_t* (*get_extra_bufts_t)(ggml_backend_dev_t);
            auto get_extra_bufts = (get_extra_bufts_t)ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
            if (get_extra_bufts) {
                ggml_backend_buffer_type_t* extra_bufts = get_extra_bufts(cpu_dev);
                if (extra_bufts && *extra_bufts) {
                    buft = *extra_bufts;
                    using_repack = true;
                }
            }
        }
    }

    if (!using_repack) {
        fprintf(stderr, "error: GGML CPU backend does not support Q4_0x8 repacking\n");
        exit(1);
    }

    struct ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead() * 2 + 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    struct ggml_context* ctx = ggml_init(params);

    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, m);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        fprintf(stderr, "error: failed to allocate repack buffer\n");
        exit(1);
    }

    fill_tensor_deterministic(a, seed, using_repack);

    const int64_t num_superblocks = (m * k) / 256;
    weights.resize(num_superblocks);
    memcpy(weights.data(), a->data, num_superblocks * sizeof(block_q4_0x8));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <M> <K> <type> <repeats> [superblock] [bf16_scales] [unsigned_q4]\n", argv[0]);
        fprintf(stderr, "  type: 8 for Q8_0, 2 for Q4_0\n");
        fprintf(stderr, "  superblock: 1 for Q4_0x8, 0 otherwise\n");
        fprintf(stderr, "  bf16_scales: 1 for bf16 scales, 0 for fp32\n");
        fprintf(stderr, "  unsigned_q4: 1 for unsigned Q4, 0 for signed\n");
        return 1;
    }

    int64_t M = atoll(argv[1]);
    int64_t K = atoll(argv[2]);
    int ggml_type = atoi(argv[3]);
    int repeats = atoi(argv[4]);
    bool is_superblock = (argc > 5) ? atoi(argv[5]) : false;
    bool use_bf16_scales = (argc > 6) ? atoi(argv[6]) : false;
    bool use_unsigned_q4 = (argc > 7) ? atoi(argv[7]) : false;

    if (M <= 0 || K <= 0 || K % 32 != 0) {
        fprintf(stderr, "error: invalid M or K\n");
        return 1;
    }

    if (ggml_type != 8 && ggml_type != 2) {
        fprintf(stderr, "error: invalid type\n");
        return 1;
    }

    if (ggml_type == 2 && is_superblock && M % 8 != 0) {
        fprintf(stderr, "error: M must be divisible by 8 for superblock\n");
        return 1;
    }

    printf("Benchmarking ggml_unpack_weight_buffer\n");
    printf("M=%ld, K=%ld, type=%d, superblock=%d, bf16_scales=%d, unsigned_q4=%d, repeats=%d\n",
           M, K, ggml_type, is_superblock, use_bf16_scales, use_unsigned_q4, repeats);

    // Generate weights
    std::vector<char> weight_data;
    if (ggml_type == 8) {
        int64_t num_blocks = M * (K / QK8_0);
        weight_data.resize(num_blocks * sizeof(block_q8_0));
        std::vector<float> tmp_wei(M * K);
        fill_buffer(tmp_wei.data(), M * K, 42);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, tmp_wei.data(), (block_q8_0*)weight_data.data(), 0, M, K, nullptr);
    } else if (ggml_type == 2 && is_superblock) {
        std::vector<block_q4_0x8> weights;
        generate_native_q4_0x8_superblocks(M, K, 42, weights);
        weight_data.resize(weights.size() * sizeof(block_q4_0x8));
        memcpy(weight_data.data(), weights.data(), weight_data.size());
    } else {
        int64_t num_blocks = M * (K / QK4_0);
        weight_data.resize(num_blocks * sizeof(block_q4_0));
        std::vector<float> tmp_wei(M * K);
        fill_buffer(tmp_wei.data(), M * K, 42);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, tmp_wei.data(), (block_q4_0*)weight_data.data(), 0, M, K, nullptr);
    }

    // Allocate buffers
    int64_t weight_buffer_size = (ggml_type == 8) ? M * K : M * (K / 2);
    int64_t scale_buffer_size = (K / 32) * M;
    int64_t scale_elem_size = use_bf16_scales ? 2 : 4;

    std::vector<int8_t> weight_buffer(weight_buffer_size);
    std::vector<char> scale_buffer(scale_buffer_size * scale_elem_size);

    // Warmup
    for (int i = 0; i < 5; i++) {
        int ret = ggml_unpack_weight_buffer(
            weight_data.data(), ggml_type, is_superblock, use_bf16_scales, use_unsigned_q4,
            M, K, weight_buffer.data(), scale_buffer.data());
        if (ret != 0) {
            fprintf(stderr, "error: unpack failed\n");
            return 1;
        }
    }

    // Benchmark
    double min_ms = std::numeric_limits<double>::max();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    for (int i = 0; i < repeats; i++) {
        auto t0 = std::chrono::steady_clock::now();
        int ret = ggml_unpack_weight_buffer(
            weight_data.data(), ggml_type, is_superblock, use_bf16_scales, use_unsigned_q4,
            M, K, weight_buffer.data(), scale_buffer.data());
        auto t1 = std::chrono::steady_clock::now();
        if (ret != 0) {
            fprintf(stderr, "error: unpack failed\n");
            return 1;
        }
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    double avg_ms = sum_ms / repeats;
    double bytes_processed = weight_buffer_size + scale_buffer_size * scale_elem_size;
    double throughput_mb_s = (bytes_processed / (avg_ms / 1000.0)) / (1024 * 1024);

    printf("Results:\n");
    printf("  Min: %.3f ms\n", min_ms);
    printf("  Avg: %.3f ms\n", avg_ms);
    printf("  Max: %.3f ms\n", max_ms);
    printf("  Throughput: %.2f MB/s\n", throughput_mb_s);

    return 0;
}