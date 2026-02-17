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
add_executable(ops_ggml_benchmark
    src/main.cpp
    src/benchmark.cpp
    src/ggml_utils.cpp
    src/matmul_bench.cpp
    src/layer_config.cpp
    src/layer_bench.cpp
)

target_include_directories(ops_ggml_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against GGML library targets provided by llama.cpp
target_link_libraries(ops_ggml_benchmark PRIVATE
    ggml
    ggml-cpu
)
