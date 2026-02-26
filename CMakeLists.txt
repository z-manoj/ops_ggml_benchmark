cmake_minimum_required(VERSION 3.21)
project(ops_ggml_benchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Fetch llama.cpp (GGML only)
# ---------------------------------------------------------------------------
include(cmake/FetchLlamaCpp.cmake)

# ---------------------------------------------------------------------------
# Benchmark executable
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

add_executable(ops_ggml_benchmark
    src/main.cpp
    src/benchmark.cpp
    src/ggml_utils.cpp
    src/matmul_bench.cpp
    src/layer_config.cpp
    src/layer_bench.cpp
    src/custom/custom_moe.cpp
    src/custom/custom_moe_bench.cpp
)

target_include_directories(ops_ggml_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/custom
)

# AVX-512 + BF16 for custom SIMD kernels (AMD EPYC 9R14 optimized)
# Fallback to AVX2 at runtime if CPU doesn't support AVX-512
set_source_files_properties(src/custom/custom_moe.cpp PROPERTIES
    COMPILE_FLAGS "-mavx2 -mfma -mf16c -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512bf16"
)

# Link against GGML library targets provided by llama.cpp
target_link_libraries(ops_ggml_benchmark PRIVATE
    ggml
    ggml-cpu
    OpenMP::OpenMP_CXX
)
