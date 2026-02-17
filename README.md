# ops_ggml_benchmark

Operator-level microbenchmark harness for [GGML](https://github.com/ggml-org/llama.cpp/tree/master/ggml) (llama.cpp).

Inspired by [oneDNN benchdnn](https://github.com/oneapi-src/oneDNN/tree/main/tests/benchdnn) -- this project benchmarks **individual GGML operators**, not full models.

## What it does

- Constructs GGML tensors and builds minimal single-operator compute graphs
- Executes them through GGML's existing CPU backend
- Measures per-iteration latency (min / avg / max) and throughput (TFLOPS)
- Produces stable, machine-parsable output

**This project does NOT:**
- Implement any new kernels or math routines
- Load models, run tokenizers, or build transformer graphs
- Modify llama.cpp source code in any way

All compute is performed by the existing GGML CPU kernels shipped with llama.cpp.

## Supported operators

| Operator       | GGML function      | Description                     |
|----------------|---------------------|---------------------------------|
| `matmul`       | `ggml_mul_mat`      | Dense matrix multiplication     |
| `matmul_id`    | `ggml_mul_mat_id`   | Expert-routed matmul (MoE)      |

## How to build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The first build will fetch llama.cpp via CMake FetchContent. Subsequent builds
reuse the cached source.

### Requirements

- CMake >= 3.21
- C++17 compiler (GCC >= 9, Clang >= 10, MSVC >= 2019)
- Git (for FetchContent)

## Example benchmark commands

```bash
# Default: 512x512x512 f32 matmul
./build/ops_ggml_benchmark

# Large f16 matmul on 32 threads
./build/ops_ggml_benchmark --op matmul --m 4096 --n 4096 --k 4096 \
    --dtype f16 --threads 32 --repeats 100 --warmup 10

# Smaller problem
./build/ops_ggml_benchmark --op matmul --m 1024 --n 1024 --k 1024 \
    --dtype f32 --threads 8

# Expert-routed matmul (MoE)
./build/ops_ggml_benchmark --op matmul_id --m 2048 --n 64 --k 2048 \
    --dtype f16 --threads 16
```

## Example output

```
op: matmul
dtype: f16
shape: m=4096 n=4096 k=4096
threads: 32
warmup: 10
repeats: 100

time(ms): min=3.12 avg=3.21 max=3.38
throughput: 21.3 TFLOPS
```

## Benchmark approach

Each benchmark invocation follows a fixed sequence of phases:

1. **Backend initialization** -- A GGML CPU backend is created and configured
   with the requested thread count (`--threads`). All subsequent compute is
   dispatched through this backend using GGML's built-in threadpool.

2. **Context & tensor creation** -- A lightweight GGML context is allocated
   with `no_alloc=true` (metadata only). Tensors for the operator's inputs
   and output are declared with the requested shape and dtype. For example,
   `matmul` creates tensors A `[K, M]` and B `[K, N]`; `matmul_id` also
   creates a 3D expert weight tensor and an integer routing-index tensor.

3. **Graph construction** -- A single-node GGML compute graph is built
   containing exactly one operator (e.g. `ggml_mul_mat`). No fusion, no
   optimization passes -- just the raw primitive.

4. **Buffer allocation** -- A GGML graph allocator (`ggml_gallocr`) reserves
   and assigns memory for all tensor data through the CPU backend's default
   buffer type.

5. **Deterministic data fill** -- Input tensors are populated with
   pseudo-random values from a seeded xorshift32 generator (values in
   `[-1, 1]`). Seeds are fixed so results are reproducible across runs.

6. **Warmup** -- The graph is executed `--warmup` times (default 10) to
   prime caches, JIT paths, and thread scheduling. These iterations are
   not timed.

7. **Timed iterations** -- The graph is executed `--repeats` times
   (default 100). Each iteration is individually timed with
   `std::chrono::steady_clock`. No timing happens inside GGML kernels --
   only wall-clock time around `ggml_backend_graph_compute()`.

8. **Result reporting** -- Min, avg, and max latency are computed from the
   per-iteration measurements. Throughput in TFLOPS is derived from the
   average latency and the theoretical FLOPs for the operation
   (`2 * M * N * K` for matmul).

9. **Cleanup** -- The graph allocator, GGML context, and backend are freed.
   No resources leak between runs.

## How it differs from model benchmarks

| Aspect           | ops_ggml_benchmark     | llama-bench / model benchmarks |
|------------------|------------------------|-------------------------------|
| Scope            | Single operator        | Full model inference          |
| Dependencies     | GGML only              | Full llama.cpp + model files  |
| Graph complexity | 1 compute node         | Hundreds of nodes             |
| Input data       | Synthetic / random     | Tokenized text                |
| Purpose          | Kernel characterization| End-to-end throughput         |

## Design philosophy

This project is a **measurement harness**, not a compute library.
It stays small, explicit, and deterministic. It exists to characterize
GGML operator performance in isolation, analogous to how benchdnn
characterizes oneDNN / ZenDNN primitives.
