include(FetchContent)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)

# Only build the GGML library targets we need.
# Disable everything else to keep the build minimal.
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

# CPU-only: disable accelerator backends
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_SYCL   OFF CACHE BOOL "" FORCE)
set(GGML_HIP    OFF CACHE BOOL "" FORCE)

# Enable native CPU optimizations
set(GGML_NATIVE ON CACHE BOOL "" FORCE)

# Explicitly enable AVX-512 with bf16 support for AMD EPYC Genoa / Intel Sapphire Rapids
# GGML_NATIVE should auto-detect these, but we force them on to ensure optimal performance
set(GGML_AVX ON CACHE BOOL "" FORCE)
set(GGML_AVX2 ON CACHE BOOL "" FORCE)
set(GGML_AVX512 ON CACHE BOOL "" FORCE)
set(GGML_AVX512_BF16 ON CACHE BOOL "" FORCE)
set(GGML_AVX512_VNNI ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)
