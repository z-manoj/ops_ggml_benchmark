cmake_minimum_required(VERSION 3.21)
project(ops_ggml_benchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(ENABLE_ZENDNN "Enable ZenDNN backend support" OFF)

# ---------------------------------------------------------------------------
# Fetch llama.cpp (GGML only)
# ---------------------------------------------------------------------------
include(cmake/FetchLlamaCpp.cmake)

# ---------------------------------------------------------------------------
# Benchmark executable
# ---------------------------------------------------------------------------
set(BENCHMARK_SOURCES
    src/main.cpp
    src/benchmark.cpp
    src/ggml_utils.cpp
    src/ggml_matmul_bench.cpp
    src/layer_config.cpp
    src/layer_bench.cpp
    src/layer_bench_ggml.cpp
)

if (ENABLE_ZENDNN)
    list(APPEND BENCHMARK_SOURCES
        src/zendnn_matmul_bench.cpp
        src/layer_bench_zendnn.cpp
    )
endif()

add_executable(ops_ggml_benchmark ${BENCHMARK_SOURCES})

target_include_directories(ops_ggml_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against GGML library targets provided by llama.cpp
target_link_libraries(ops_ggml_benchmark PRIVATE
    ggml
    ggml-cpu
)

# ---------------------------------------------------------------------------
# ZenDNN support (optional)
# ---------------------------------------------------------------------------
if (ENABLE_ZENDNN)
    # Get ZenDNN path
    if (NOT DEFINED ZENDNN_ROOT OR ZENDNN_ROOT STREQUAL "")
        set(ZENDNN_ROOT "$ENV{ZENDNN_ROOT}")
    endif()

    if (NOT ZENDNN_ROOT OR ZENDNN_ROOT STREQUAL "" OR ZENDNN_ROOT STREQUAL "OFF")
        message(FATAL_ERROR "ZENDNN_ROOT is not set. Please pass -DZENDNN_ROOT=/path/to/zendnn or set ZENDNN_ROOT environment variable")
    else()
        message(STATUS "Using ZenDNN installation at: ${ZENDNN_ROOT}")
    endif()

    # Helper macro to find a library
    macro(find_zendnn_lib VAR NAME HINT_DIR)
        find_library(${VAR}
            NAMES ${NAME}
            HINTS ${HINT_DIR}
            NO_DEFAULT_PATH
        )
        if (${VAR})
            message(STATUS "Found ${NAME}: ${${VAR}}")
        else()
            find_library(${VAR} NAMES ${NAME} HINTS ${HINT_DIR})
            if (${VAR})
                message(STATUS "Found ${NAME} (fallback): ${${VAR}}")
            else()
                message(WARNING "Could not find library '${NAME}' in ${HINT_DIR}")
            endif()
        endif()
    endmacro()

    # Find Core ZenDNN Library
    find_library(LIB_ZENDNNL NAMES zendnnl zendnnl_archive HINTS ${ZENDNN_ROOT}/zendnnl/lib NO_DEFAULT_PATH)
    if (NOT LIB_ZENDNNL)
        message(FATAL_ERROR "Could not find libzendnnl in ${ZENDNN_ROOT}/zendnnl/lib")
    else()
        message(STATUS "Found ZenDNNL: ${LIB_ZENDNNL}")
    endif()

    # Find Dependencies
    find_zendnn_lib(LIB_AOCLDLP     aocl-dlp     ${ZENDNN_ROOT}/deps/aocldlp/lib)
    find_zendnn_lib(LIB_AOCLUTILS   aoclutils    ${ZENDNN_ROOT}/deps/aoclutils/lib)
    find_zendnn_lib(LIB_AU_CPUID    au_cpuid     ${ZENDNN_ROOT}/deps/aoclutils/lib)
    find_zendnn_lib(LIB_DNNL        dnnl         ${ZENDNN_ROOT}/deps/onednn/lib)
    find_zendnn_lib(LIB_XSMM        xsmm         ${ZENDNN_ROOT}/deps/libxsmm/lib)
    find_zendnn_lib(LIB_XSMMEXT     xsmmext      ${ZENDNN_ROOT}/deps/libxsmm/lib)
    find_zendnn_lib(LIB_XSMMNOBLAS  xsmmnoblas   ${ZENDNN_ROOT}/deps/libxsmm/lib)

    # Optional Dependencies
    find_library(LIB_FBGEMM NAMES fbgemm HINTS ${ZENDNN_ROOT}/deps/fbgemm/lib NO_DEFAULT_PATH)
    find_library(LIB_ASMJIT NAMES asmjit HINTS ${ZENDNN_ROOT}/deps/asmjit/lib NO_DEFAULT_PATH)
    find_library(LIB_CPUINFO NAMES cpuinfo HINTS ${ZENDNN_ROOT}/deps/cpuinfo/lib ${ZENDNN_ROOT}/deps/fbgemm/lib NO_DEFAULT_PATH)

    # Construct the list of libraries
    set(ZENDNN_LINK_LIBS
        ${LIB_ZENDNNL}
        ${LIB_DNNL}
        m
        pthread
    )

    # Add optional libs if found
    if (LIB_AOCLDLP)
        list(APPEND ZENDNN_LINK_LIBS ${LIB_AOCLDLP})
    endif()
    if (LIB_AOCLUTILS)
        list(APPEND ZENDNN_LINK_LIBS ${LIB_AOCLUTILS})
    endif()
    if (LIB_AU_CPUID)
        list(APPEND ZENDNN_LINK_LIBS ${LIB_AU_CPUID})
    endif()
    if (LIB_XSMM)
        list(APPEND ZENDNN_LINK_LIBS ${LIB_XSMM})
    endif()
    if (LIB_XSMMEXT)
        list(APPEND ZENDNN_LINK_LIBS ${LIB_XSMMEXT})
    endif()
    if (LIB_XSMMNOBLAS)
        list(APPEND ZENDNN_LINK_LIBS ${LIB_XSMMNOBLAS})
    endif()
    if (LIB_FBGEMM)
        list(APPEND ZENDNN_LINK_LIBS ${LIB_FBGEMM})
    endif()
    if (LIB_ASMJIT)
        list(APPEND ZENDNN_LINK_LIBS ${LIB_ASMJIT})
    endif()
    if (LIB_CPUINFO)
        list(APPEND ZENDNN_LINK_LIBS ${LIB_CPUINFO})
    endif()

    # Includes
    set(ZENDNN_INCLUDES
        ${ZENDNN_ROOT}/zendnnl/include
        ${ZENDNN_ROOT}/deps/aocldlp/include
        ${ZENDNN_ROOT}/deps/aoclutils/include
        ${ZENDNN_ROOT}/deps/json/include
        ${ZENDNN_ROOT}/deps/libxsmm/include
        ${ZENDNN_ROOT}/deps/onednn/include
        ${ZENDNN_ROOT}/deps/fbgemm/include
        ${ZENDNN_ROOT}/deps/asmjit/include
        ${ZENDNN_ROOT}/deps/cpuinfo/include
    )

    target_include_directories(ops_ggml_benchmark PRIVATE ${ZENDNN_INCLUDES})
    target_link_libraries(ops_ggml_benchmark PRIVATE ${ZENDNN_LINK_LIBS})
    target_compile_definitions(ops_ggml_benchmark PRIVATE ENABLE_ZENDNN)

    message(STATUS "ZenDNN backend enabled")
endif()
