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
| `layer`        | multi-node graph    | Full transformer layer GEMM workload |

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

# Single shape
./build/ops_ggml_benchmark --op matmul --m 4096 --n 4096 --k 4096 \
    --dtype f16 --threads 32 --repeats 100 --warmup 10

# Multiple shapes on the CLI (comma-separated MxNxK)
./build/ops_ggml_benchmark --op matmul --dtype f32 --threads 8 \
    --shapes 512x512x512,1024x1024x1024,4096x4096x4096

# Shapes from a batch file
./build/ops_ggml_benchmark --op matmul --dtype f16 --threads 16 \
    --batch configs/shapes.txt

# Combine both (batch file + CLI shapes run sequentially)
./build/ops_ggml_benchmark --op matmul --dtype f32 --threads 8 \
    --batch configs/shapes.txt --shapes 2048x2048x2048

# Expert-routed matmul (MoE)
./build/ops_ggml_benchmark --op matmul_id --m 2048 --n 64 --k 2048 \
    --dtype f16 --threads 16

# Layer-level benchmark (full transformer layer GEMM workload)
./build/ops_ggml_benchmark --op layer --config configs/mixtral-8x7b.cfg \
    --dtype f16 --threads 32 --repeats 50 --warmup 5

./build/ops_ggml_benchmark --op layer --config configs/qwen3-coder-30b-a3b.cfg \
    --dtype f16 --threads 32 --repeats 50 --warmup 5
```

### Batch file format

One shape per line. Supports `M N K` (whitespace-separated) or `MxNxK`.
Blank lines and `#` comments are ignored.

```
# shapes.txt -- LLM-typical matmul sizes
512 512 512
1024 1024 1024
4096x4096x4096
```

## Layer benchmark mode

The `layer` operator builds a multi-node GGML graph representing all GEMM
operations in a single transformer layer (attention projections + MoE expert
matmuls). This captures realistic memory pressure, cache contention, and
barrier overhead between sequential ops.

Two model configs are included:

| Config | Ops | Experts | Notes |
|--------|-----|---------|-------|
| `configs/mixtral-8x7b.cfg` | 8 | 8 total, 2 active | ~2.7 GB weights (f16) |
| `configs/qwen3-coder-30b-a3b.cfg` | 8 | 128 total, 8 active | ~3.4 GB weights (f16) |

### Config file format

One op per line. `model_name` and `seq_len` are metadata headers.

```
# configs/mixtral-8x7b.cfg
model_name  mixtral-8x7b-instruct
seq_len     512

mul_mat     attn_q        512  4096   4096
mul_mat     attn_k        512  1024   4096
mul_mat     attn_v        512  1024   4096
mul_mat     attn_output   512  4096   4096
mul_mat     ffn_gate_inp  512  8      4096
mul_mat_id  ffn_gate_exp  512  14336  4096   8  2
mul_mat_id  ffn_up_exp    512  14336  4096   8  2
mul_mat_id  ffn_down_exp  512  4096   14336  8  2
```

- `mul_mat <label> <M> <N> <K>` -- dense matmul
- `mul_mat_id <label> <M> <N> <K> <n_experts> <n_experts_used>` -- MoE expert-routed matmul

### Layer output format

```
op: layer
model: mixtral-8x7b-instruct
dtype: f16
threads: 32
warmup: 5
repeats: 50

graph nodes: 8
  [0] mul_mat      attn_q           m=512   n=4096  k=4096  (  17.18 GFLOPs)
  [1] mul_mat      attn_k           m=512   n=1024  k=4096  (   4.29 GFLOPs)
  ...
total: 403.76 GFLOPs

time(ms): min=12.34 avg=12.87 max=13.42
throughput: 8.68 TFLOPS
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

3. **Graph construction** -- A GGML compute graph is built containing the
   operator(s). For `matmul` and `matmul_id` this is a single node; for
   `layer` mode, all ops from the config file are expanded into one graph.
   No fusion, no optimization passes -- just the raw primitives.

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
| Scope            | Single op or layer     | Full model inference          |
| Dependencies     | GGML only              | Full llama.cpp + model files  |
| Graph complexity | 1-8 compute nodes      | Hundreds of nodes             |
| Input data       | Synthetic / random     | Tokenized text                |
| Purpose          | Kernel characterization| End-to-end throughput         |

## Design philosophy

This project is a **measurement harness**, not a compute library.
It stays small, explicit, and deterministic. It exists to characterize
GGML operator performance in isolation, analogous to how benchdnn
characterizes oneDNN / ZenDNN primitives.
