# ops_ggml_benchmark

Operator-level microbenchmark harness for [GGML](https://github.com/ggml-org/llama.cpp/tree/master/ggml) and [ZenDNN](https://github.com/amd/ZenDNN).

Inspired by [oneDNN benchdnn](https://github.com/oneapi-src/oneDNN/tree/main/tests/benchdnn) -- this project benchmarks **individual operators** from GGML and ZenDNN backends, not full models. Uses ZenDNN's LoWoHA (Low-Overhead High-Accuracy) matmul API for optimized AMD CPU performance.

## What it does

- Constructs minimal single-operator compute graphs for GGML or ZenDNN backends
- Executes operations through either:
  - GGML's CPU backend (llama.cpp)
  - ZenDNN/oneDNN matmul primitives
- Measures per-iteration latency (min / avg / max)
- Produces stable, machine-parsable output
- Supports multiple data types: **f32, f16, bf16** (backend-dependent)

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

### GGML backend only (default)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The first build will fetch llama.cpp via CMake FetchContent. Subsequent builds
reuse the cached source.

### With ZenDNN backend support

```bash
# Set ZENDNN_ROOT to your ZenDNN installation
export ZENDNN_ROOT=/path/to/zendnn/build/install

cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_ZENDNN=ON
cmake --build build -j$(nproc)
```

Or pass ZENDNN_ROOT directly to cmake:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_ZENDNN=ON \
  -DZENDNN_ROOT=/path/to/zendnn/build/install

cmake --build build -j$(nproc)
```

### Requirements

- CMake >= 3.21
- C++17 compiler (GCC >= 9, Clang >= 10, MSVC >= 2019)
- Git (for FetchContent)

## Example benchmark commands

```bash
# Default: 512x512x512 f32 matmul with GGML backend
./build/ops_ggml_benchmark

# Single shape with GGML backend (supports f32, f16, bf16)
./build/ops_ggml_benchmark --backend ggml --op matmul \
    --m 4096 --n 4096 --k 4096 --dtype f16 --threads 32 \
    --repeats 100 --warmup 10

# ZenDNN backend with f32 (ZenDNN supports f32 and bf16 only)
./build/ops_ggml_benchmark --backend zendnn --op matmul \
    --m 4096 --n 4096 --k 4096 --dtype f32 --threads 32

# ZenDNN backend with bf16 (brain float16)
./build/ops_ggml_benchmark --backend zendnn --op matmul \
    --m 2048 --n 2048 --k 2048 --dtype bf16 --threads 16

# Multiple shapes on the CLI (comma-separated MxNxK)
./build/ops_ggml_benchmark --op matmul --dtype f32 --threads 8 \
    --shapes 512x512x512,1024x1024x1024,4096x4096x4096

# Shapes from a batch file
./build/ops_ggml_benchmark --op matmul --dtype f16 --threads 16 \
    --batch shapes.txt

# Combine both (batch file + CLI shapes run sequentially)
./build/ops_ggml_benchmark --op matmul --dtype f32 --threads 8 \
    --batch shapes.txt --shapes 2048x2048x2048

# Expert-routed matmul (MoE)
./build/ops_ggml_benchmark --op matmul_id --m 2048 --n 64 --k 2048 \
    --dtype f16 --threads 16

# Layer-level benchmark (full transformer layer GEMM workload)
# Dense models - GGML backend (supports f32, f16, bf16)
./build/ops_ggml_benchmark --backend ggml --op layer \
    --config configs/llama-3.1-8b.cfg --dtype f16 --threads 32

./build/ops_ggml_benchmark --backend ggml --op layer \
    --config configs/qwen2-7b.cfg --dtype bf16 --threads 32

# Dense models - ZenDNN backend (supports f32 and bf16 only, NOT f16)
./build/ops_ggml_benchmark --backend zendnn --op layer \
    --config configs/llama-3.1-8b.cfg --dtype bf16 --threads 32

./build/ops_ggml_benchmark --backend zendnn --op layer \
    --config configs/qwen2-7b.cfg --dtype bf16 --threads 32

# MoE models (GGML backend only)
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

Model configs are included:

| Config | Ops | Type | Notes |
|--------|-----|------|-------|
| `configs/llama-3.1-8b.cfg` | 7 | Dense | Llama 3.1 8B, GQA (8 KV heads) |
| `configs/qwen2-7b.cfg` | 7 | Dense | Qwen2 7B, GQA (4 KV heads) |
| `configs/mixtral-8x7b.cfg` | 8 | MoE | 8 experts, 2 active, ~2.7 GB weights (f16) |
| `configs/qwen3-coder-30b-a3b.cfg` | 8 | MoE | 128 experts, 8 active, ~3.4 GB weights (f16) |

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

Shows per-operation timing and aggregate statistics:

```
op: layer
backend: zendnn
model: llama-3.1-8b
dtype: bf16
threads: 8
warmup: 2
repeats: 5

graph nodes: 7
  [0] mul_mat      attn_q           m=512   n=4096  k=4096  time(ms): min=9.90 avg=9.92 max=9.95
  [1] mul_mat      attn_k           m=512   n=1024  k=4096  time(ms): min=2.64 avg=2.65 max=2.66
  [2] mul_mat      attn_v           m=512   n=1024  k=4096  time(ms): min=2.64 avg=2.65 max=2.67
  [3] mul_mat      attn_output      m=512   n=4096  k=4096  time(ms): min=9.90 avg=9.92 max=9.94
  [4] mul_mat      ffn_gate         m=512   n=14336 k=4096  time(ms): min=34.10 avg=34.16 max=34.22
  [5] mul_mat      ffn_up           m=512   n=14336 k=4096  time(ms): min=34.12 avg=34.16 max=34.20
  [6] mul_mat      ffn_down         m=512   n=4096  k=14336 time(ms): min=34.32 avg=34.38 max=34.44

total time(ms): min=127.74 avg=127.86 max=128.02
```

**Note:** ZenDNN backend measures each operation individually, while GGML backend estimates per-operation times proportionally from the total graph execution time.

## Example output

### GGML backend example output:
```
op: matmul
backend: ggml
dtype: f16
shape: m=4096 n=4096 k=4096
threads: 32
warmup: 10
repeats: 100

time(ms): min=3.12 avg=3.21 max=3.38
```

### ZenDNN backend example output:
```
op: matmul
backend: zendnn
dtype: bf16
shape: m=4096 n=4096 k=4096
threads: 32
warmup: 10
repeats: 100

time(ms): min=2.87 avg=2.94 max=3.05
```

## Backends

| Backend | Description | Data Types | Operators |
|---------|-------------|------------|-----------|
| `ggml` | GGML CPU backend from llama.cpp | f32, f16, bf16 | matmul, matmul_id, layer (all models) |
| `zendnn` | ZenDNN LoWoHA matmul API (AMD optimized) | **f32, bf16** (NO f16) | matmul, layer (non-MoE only) |

Use `--backend <ggml|zendnn>` to select the backend (default: ggml).

**Important notes:**
- **ZenDNN backend does NOT support f16** - use f32 or bf16 instead
- ZenDNN backend does **not** support `matmul_id` (MoE expert routing)
- For `layer` benchmarks with **MoE models** (mixtral, qwen3-coder), use GGML backend
- For `layer` benchmarks with **dense models** (llama-3.1-8b, qwen2-7b), both backends are supported

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
   per-iteration measurements.

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

## Code organization

```
src/
├── main.cpp                    # CLI argument parsing and dispatch
├── benchmark.cpp               # Backend routing for single matmul ops
├── ggml_matmul_bench.cpp       # GGML matmul implementation
├── zendnn_matmul_bench.cpp     # ZenDNN matmul implementation (LoWoHA API)
├── layer_bench.cpp             # Common output formatting for layer benchmarks
├── layer_bench_ggml.cpp        # GGML layer benchmark (graph-based execution)
├── layer_bench_zendnn.cpp      # ZenDNN layer benchmark (per-op timing)
├── layer_config.cpp            # Layer config file parser
└── ggml_utils.cpp              # GGML type utilities
```
