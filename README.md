# ops_ggml_benchmark

Operator-level microbenchmark harness for [GGML](https://github.com/ggml-org/llama.cpp/tree/master/ggml) and [ZenDNN](https://github.com/amd/ZenDNN).

Inspired by [ZenDNN benchdnn](https://github.com/amd/ZenDNN/tree/main/benchdnn) and [oneDNN benchdnn](https://github.com/oneapi-src/oneDNN/tree/main/tests/benchdnn) -- this project benchmarks **individual operators** from GGML and ZenDNN backends, not full models. Uses ZenDNN's LoWoHA (Low-Overhead High-Accuracy) matmul API for optimized AMD CPU performance.

**Data Type Configuration:**
- **Source (input)**: Configurable via `--src_dtype` (f32 or bf16)
- **Weights**: Configurable via `--wei_dtype` (f32/f16/bf16/q8_0/q4_0 for GGML; f32/bf16 for ZenDNN)
- **Destination (output)**: Always F32

## Purpose and Audience

This tool is designed for **developers**, **researchers**, and **performance engineers** who need to:
- Characterize operator-level performance in isolation
- Compare GGML and ZenDNN backend performance
- Evaluate quantization impact (q8_0, q4_0) on inference latency
- Benchmark transformer layer GEMM workloads without full model overhead

## What it does

- Constructs minimal single-operator compute graphs for GGML or ZenDNN backends
- Executes operations through either:
  - GGML's CPU backend (llama.cpp) with quantization support
  - ZenDNN/oneDNN matmul primitives (AMD optimized)
- Measures per-iteration latency (min / avg / max) with detailed timing breakdowns
- Produces **ZenDNN benchdnn-style output**: tabular console display + timestamped CSV files
- Reports GFLOPS, timing breakdowns (context creation, operator setup, execution) with percentages
- Supports warmup iterations to eliminate cold-start effects

**This project does NOT:**
- Implement any new kernels or math routines
- Load models, run tokenizers, or build transformer graphs
- Modify llama.cpp or ZenDNN source code in any way

All compute is performed by existing GGML CPU kernels and ZenDNN primitives.

## Supported Features Overview

| Feature              | GGML Backend | ZenDNN Backend |
|----------------------|--------------|----------------|
| **Operators**        | matmul, matmul_id (MoE), layer | matmul, layer (dense only) |
| **Source Types**     | **f32, bf16** | **f32, bf16** |
| **Weight Types**     | **f32, f16, bf16, q8_0, q4_0** | **f32, bf16** |
| **Output Type**      | **f32 (fixed)** | **f32 (fixed)** |
| **Quantization**     | ✅ q8_0, q4_0 with repack | ❌ Not supported |
| **MoE Support**      | ✅ matmul_id, MoE layers | ❌ Dense only |
| **Batched Matmul**   | ✅ Via shape dimensions | ✅ Via shape dimensions |
| **Warmup Iterations**| ✅ Configurable | ✅ Configurable |

### Data Type Details

The benchmark supports separate data types for **source (input)** and **weights**, while output is always **f32**.

**Available Types:**
- `f32` - 32-bit float (4 bytes/element)
- `f16` - 16-bit float (2 bytes/element) - weights only
- `bf16` - BFloat16 (2 bytes/element)
- `q8_0` - 8-bit quantization (1.0625 bytes/element) - weights only, GGML only
- `q4_0` - 4-bit quantization (0.5625 bytes/element) - weights only, GGML only

**GGML Backend Type Compatibility:**

GGML supports two configurations:

**Option 1: F32 source with ANY weight type** (most flexible)
- ✅ `src=f32 wei=f32` - Standard F32 computation
- ✅ `src=f32 wei=f16` - F32 source + F16 weights
- ✅ `src=f32 wei=bf16` - F32 source + BF16 weights
- ✅ `src=f32 wei=q8_0` - F32 source + 8-bit quantized weights
- ✅ `src=f32 wei=q4_0` - F32 source + 4-bit quantized weights

**Option 2: BF16 source with BF16 weights ONLY** (best performance)
- ✅ `src=bf16 wei=bf16` - **~2x faster than F32+F32** (recommended for maximum performance)

**Invalid configurations** (will produce error):
- ❌ `src=bf16 wei=f32` - BF16 source requires BF16 weights
- ❌ `src=bf16 wei=f16` - BF16 source requires BF16 weights
- ❌ `src=bf16 wei=q8_0` - BF16 source requires BF16 weights
- ❌ `src=bf16 wei=q4_0` - BF16 source requires BF16 weights

**Rule:** If source is BF16, weights MUST be BF16. If source is F32, weights can be any supported type.

**ZenDNN Backend Type Compatibility:**

All combinations of f32 and bf16 are supported:
- ✅ `src=f32 wei=f32`
- ✅ `src=f32 wei=bf16`
- ✅ `src=bf16 wei=f32`
- ✅ `src=bf16 wei=bf16`
- ⚠️ **f16 is NOT supported** - use f32 or bf16
- ⚠️ **Quantized types (q8_0, q4_0) are NOT supported**

## Quantization Support (GGML Backend Only)

### Weight Quantization

Quantization is supported for **weights only** (not source inputs). Quantized weights require **F32 source input**.

**Supported quantized weight types:**
- `q8_0` - 8-bit quantization (1.0625 bytes/element)
- `q4_0` - 4-bit quantization with automatic repack (0.5625 bytes/element)

**Example usage:**
```bash
# 8-bit quantized weights with F32 source
./ops_ggml_benchmark --src_dtype f32 --wei_dtype q8_0 ...

# 4-bit quantized weights with F32 source
./ops_ggml_benchmark --src_dtype f32 --wei_dtype q4_0 ...
```

### q4_0 Automatic Repack Optimization

When using `--wei_dtype q4_0`, the benchmark automatically enables GGML's repack buffer optimization:

1. **Weight Loading**: q4_0 weights are automatically converted to interleaved q4_0x8 format
2. **Kernel Selection**: Uses optimized `ggml_gemm_q4_0_8x8` SIMD kernels (processes 8 blocks at once)
3. **Memory Efficiency**: 0.5625 bytes per element vs 1.0625 for q8_0

**Technical Details:**
- **Standard q4_0**: 18 bytes per 32 elements (block_q4_0)
- **Repacked q4_0x8**: 144 bytes per 256 elements (8 blocks interleaved)
- **Automatic**: Triggered via `ggml_backend_tensor_set()` during weight initialization
- **Transparent**: No user configuration needed
- **Requirement**: Must use F32 source input (`--src_dtype f32`)

## Supported operators

| Operator       | GGML function      | Description                     | GGML | ZenDNN |
|----------------|---------------------|---------------------------------|------|--------|
| `matmul`       | `ggml_mul_mat`      | Dense matrix multiplication     | ✅   | ✅     |
| `matmul_id`    | `ggml_mul_mat_id`   | Expert-routed matmul (MoE)      | ✅   | ❌     |
| `layer`        | multi-node graph    | Full transformer layer GEMM workload | ✅   | ✅ (dense only) |

### matmul_id (MoE) Implementation Details

The `matmul_id` operator benchmarks expert-routed matmul for Mixture-of-Experts models:

**Tensors created:**
- `as` (3D): Stacked expert weight matrices `[K, M, n_experts]` - all experts in one tensor
- `b` (3D): Input activations `[K, n_experts_used, N]` - N tokens, each routed to n_experts_used experts
- `ids` (2D, int32): Routing indices `[n_experts_used, N]` - which experts each token uses

**Routing logic:**
- Expert weights filled with deterministic random data (all experts)
- Routing IDs cycle through experts: token 0 → experts [0,1], token 1 → experts [1,2], etc.
- GGML's `ggml_mul_mat_id` extracts selected expert weights per token and computes matmuls

**Example:** For 8 experts with 2 active per token, the 3D `as` tensor contains all 8 expert weight matrices stacked. The `ids` tensor tells GGML which 2 experts each token should use.

## How to build

### Requirements

- CMake >= 3.21
- C++17 compiler (GCC >= 9, Clang >= 10, MSVC >= 2019)
- Git (for FetchContent)
- Optional: ZenDNN installation for ZenDNN backend

### GGML backend only (default)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The first build will fetch llama.cpp via CMake FetchContent. Subsequent builds reuse the cached source.

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

## Example benchmark commands

### Basic matmul benchmarks

```bash
# Default: 512x512x512 f32 src + f32 weights with GGML backend
./build/ops_ggml_benchmark

# F32 source with F32 weights (same type)
./build/ops_ggml_benchmark --backend ggml --op matmul \
    --m 4096 --n 4096 --k 4096 \
    --src_dtype f32 --wei_dtype f32 \
    --threads 16 --repeats 100 --warmup 10

# BF16 source with BF16 weights (same type - ~2x faster than F32!)
./build/ops_ggml_benchmark --backend ggml --op matmul \
    --m 4096 --n 4096 --k 4096 \
    --src_dtype bf16 --wei_dtype bf16 \
    --threads 16 --repeats 100 --warmup 10

# F32 source with BF16 weights (mixed type)
./build/ops_ggml_benchmark --backend ggml --op matmul \
    --m 4096 --n 4096 --k 4096 \
    --src_dtype f32 --wei_dtype bf16 \
    --threads 16 --repeats 100 --warmup 10

# ZenDNN backend with BF16 (supports all f32/bf16 combinations)
./build/ops_ggml_benchmark --backend zendnn --op matmul \
    --m 2048 --n 2048 --k 2048 \
    --src_dtype bf16 --wei_dtype bf16 --threads 16
```

### Quantization benchmarks (GGML backend only)

```bash
# q8_0: 8-bit quantized weights (requires F32 source)
./build/ops_ggml_benchmark --backend ggml --op matmul \
    --m 2048 --n 2048 --k 2048 \
    --src_dtype f32 --wei_dtype q8_0 --threads 16

# q4_0: 4-bit quantized weights with repack (requires F32 source)
./build/ops_ggml_benchmark --backend ggml --op matmul \
    --m 2048 --n 2048 --k 2048 \
    --src_dtype f32 --wei_dtype q4_0 --threads 16

# Compare weight types with F32 source
for wei in f32 bf16 q8_0 q4_0; do
  echo "=== F32 source + $wei weights ==="
  ./build/ops_ggml_benchmark --backend ggml \
    --src_dtype f32 --wei_dtype $wei \
    --m 2048 --n 2048 --k 2048 --threads 16 --repeats 10 --warmup 5
done

# Backward compatibility: --dtype sets both src and wei to same type
./build/ops_ggml_benchmark --backend ggml --op matmul \
    --m 2048 --n 2048 --k 2048 --dtype bf16 --threads 16
```

### Multiple shapes

```bash
# Multiple shapes on the CLI (comma-separated MxNxK)
./build/ops_ggml_benchmark --op matmul \
    --src_dtype f32 --wei_dtype q4_0 --threads 16 \
    --shapes 512x512x512,1024x1024x1024,4096x4096x4096

# Shapes from a batch file
./build/ops_ggml_benchmark --op matmul \
    --src_dtype f32 --wei_dtype q8_0 --threads 16 \
    --batch shapes.txt

# Combine both (batch file + CLI shapes run sequentially)
./build/ops_ggml_benchmark --op matmul \
    --src_dtype bf16 --wei_dtype bf16 --threads 16 \
    --batch shapes.txt --shapes 2048x2048x2048
```

### MoE (GGML backend only)

```bash
# Expert-routed matmul (MoE) - GGML backend only
# Default: 1 expert (simplest case, closest to regular matmul)
./build/ops_ggml_benchmark --backend ggml --op matmul_id \
    --m 2048 --n 64 --k 2048 \
    --src_dtype f32 --wei_dtype q4_0 --threads 16

# For realistic MoE testing: 8 experts, 2 active per token
./build/ops_ggml_benchmark --backend ggml --op matmul_id \
    --m 2048 --n 64 --k 2048 \
    --src_dtype f32 --wei_dtype q4_0 \
    --n_experts 8 --n_experts_used 2 --threads 16
```

### Layer-level benchmarks

Layer benchmarks support the same type compatibility rules as matmul operations:
- **GGML**: Use `src=f32` with any weight type, OR `src=bf16 wei=bf16` only
- **ZenDNN**: All f32/bf16 combinations supported

```bash
# Dense models - GGML backend
# Option 1: F32 source with any weight type
./build/ops_ggml_benchmark --backend ggml --op layer \
    --config configs/llama-3.1-8b.cfg --src_dtype f32 --wei_dtype q4_0 --threads 16

./build/ops_ggml_benchmark --backend ggml --op layer \
    --config configs/qwen2-7b.cfg --src_dtype f32 --wei_dtype bf16 --threads 16

# Option 2: BF16 source with BF16 weights (best performance)
./build/ops_ggml_benchmark --backend ggml --op layer \
    --config configs/qwen2-7b.cfg --src_dtype bf16 --wei_dtype bf16 --threads 16

# Dense models - ZenDNN backend (all f32/bf16 combinations supported)
./build/ops_ggml_benchmark --backend zendnn --op layer \
    --config configs/llama-3.1-8b.cfg --src_dtype bf16 --wei_dtype bf16 --threads 16

./build/ops_ggml_benchmark --backend zendnn --op layer \
    --config configs/qwen2-7b.cfg --src_dtype f32 --wei_dtype f32 --threads 16

# MoE models (GGML backend only)
./build/ops_ggml_benchmark --backend ggml --op layer \
    --config configs/mixtral-8x7b.cfg --src_dtype f32 --wei_dtype q4_0 --threads 16

./build/ops_ggml_benchmark --backend ggml --op layer \
    --config configs/qwen3-coder-30b-a3b.cfg --src_dtype f32 --wei_dtype q8_0 --threads 16

# Backward compatibility: --dtype sets both src and wei to same type
./build/ops_ggml_benchmark --backend ggml --op layer \
    --config configs/llama-3.1-8b.cfg --dtype bf16 --threads 16
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

The `layer` operator builds a multi-node GGML graph representing all GEMM operations in a single transformer layer (attention projections + FFN matmuls + MoE expert routing). This captures realistic memory pressure, cache contention, and barrier overhead between sequential ops.

Model configs are included:

| Config | Ops | Type | Notes | GGML | ZenDNN |
|--------|-----|------|-------|------|--------|
| `configs/llama-3.1-8b.cfg` | 7 | Dense | Llama 3.1 8B, GQA (8 KV heads) | ✅ | ✅ |
| `configs/qwen2-7b.cfg` | 7 | Dense | Qwen2 7B, GQA (4 KV heads) | ✅ | ✅ |
| `configs/mixtral-8x7b.cfg` | 8 | MoE | 8 experts, 2 active | ✅ | ❌ |
| `configs/qwen3-coder-30b-a3b.cfg` | 8 | MoE | 128 experts, 8 active | ✅ | ❌ |

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

### Layer Benchmark Output

Shows per-operation timing and aggregate statistics:

```bash
./build/ops_ggml_benchmark --backend ggml --op layer --config configs/llama-3.1-8b.cfg \
  --wei_dtype f32 --threads 8 --repeats 5 --warmup 2
```

```
[GGML] estimated weight memory: 0.81 GB

=== Layer Benchmark: llama-3.1-8b ===
Backend  Iters  Data_type  Threads  Avg_total(ms)  Avg_ctx_init(ms)  Avg_op_setup(ms)  Avg_exec(ms)  GFLOPS
ggml     5      f32        8        493.59         0.24              112.86            380.48        223.34

--- Per-Operation Details (7 ops) ---
Idx  Op_type  Label         M    N      K      GFLOPS  Min(ms)  Avg(ms)  Max(ms)
0    mul_mat  attn_q        512  4096   4096   17.18   28.81    29.27    29.69
1    mul_mat  attn_k        512  1024   4096   4.29    7.20     7.32     7.42
2    mul_mat  attn_v        512  1024   4096   4.29    7.20     7.32     7.42
3    mul_mat  attn_output   512  4096   4096   17.18   28.81    29.27    29.69
4    mul_mat  ffn_gate      512  14336  4096   60.13   100.82   102.44   103.91
5    mul_mat  ffn_up        512  14336  4096   60.13   100.82   102.44   103.91
6    mul_mat  ffn_down      512  4096   14336  60.13   100.82   102.44   103.91
```

**Note:**
- **GGML backend**: Executes entire graph in one call, per-operation times are estimated proportionally based on FLOPs
- **ZenDNN backend**: Executes and times each operation individually

## Output Format

The benchmark produces **ZenDNN benchdnn-inspired output** with tabular console display showing per-iteration average times for fair comparison.

### Console Output

Results are displayed in a tabular format with per-iteration averages:

```
Backend  M      N      K      Iters  Src_type        Wei_type        Threads  Avg_total(ms)  Avg_ctx_init(ms)  Avg_op_setup(ms)  Avg_exec(ms)  GFLOPS
ggml     2048   2048   2048   100    f32             f32             96       3.10           0.01              0.22              2.87          5993.75
ggml     2048   2048   2048   100    bf16            bf16            96       2.05           0.01              0.23              1.81          9481.22
ggml     2048   2048   2048   100    f32             q4_0            96       3.23           0.02              0.31              2.90          5924.14
```

**Columns explained:**
- **Backend**: ggml or zendnn
- **M, N, K**: Matrix dimensions
- **Iters**: Number of timed iterations
- **Src_type**: Source/input data type (f32 or bf16)
- **Wei_type**: Weight data type (f32, f16, bf16, q8_0, q4_0)
- **Threads**: Thread count
- **Avg_total(ms)**: Average total time per iteration (includes amortized setup)
- **Avg_ctx_init(ms)**: Context/backend initialization time (amortized per iteration)
- **Avg_op_setup(ms)**: Graph building + memory allocation time (amortized per iteration)
- **Avg_exec(ms)**: Pure matmul execution time per iteration (the key metric!)
- **GFLOPS**: Performance = (2*M*N*K) / (avg_exec_ms * 1e-3) / 1e9

**Performance Note:** BF16+BF16 delivers ~2x higher performance than F32+F32. Performance characteristics may vary with thread count - lower thread counts typically favor bf16+bf16, while higher thread counts may show similar performance between f32+bf16 and bf16+bf16 due to parallelization effects.

### Why Per-Iteration Averages?

One-time setup costs (context creation, graph building) are amortized across all iterations. This makes comparisons **fair** regardless of iteration count:

- 5 iterations: Setup overhead appears large
- 100 iterations: Setup overhead is negligible
- **Per-iteration average**: Directly comparable!

Example: Both show ~2.87ms execution time:
```
5 iters:   Avg_exec=2.85ms
100 iters: Avg_exec=2.87ms
```

### Example Outputs

#### Same Type vs Mixed Type Comparison (GGML backend, 2048x2048x2048, 96 threads):

```bash
# Same type: F32+F32
./build/ops_ggml_benchmark --src_dtype f32 --wei_dtype f32 \
  --m 2048 --n 2048 --k 2048 --threads 96 --repeats 100 --warmup 50

# Same type: BF16+BF16 (fastest!)
./build/ops_ggml_benchmark --src_dtype bf16 --wei_dtype bf16 \
  --m 2048 --n 2048 --k 2048 --threads 96 --repeats 100 --warmup 50

# Mixed type: F32 source + BF16 weights
./build/ops_ggml_benchmark --src_dtype f32 --wei_dtype bf16 \
  --m 2048 --n 2048 --k 2048 --threads 96 --repeats 100 --warmup 50

# Mixed type: F32 source + Q4_0 weights
./build/ops_ggml_benchmark --src_dtype f32 --wei_dtype q4_0 \
  --m 2048 --n 2048 --k 2048 --threads 96 --repeats 100 --warmup 50
```

```
=== Same Type: F32+F32 ===
Backend  M      N      K      Iters  Src_type  Wei_type  Threads  Avg_total(ms)  Avg_ctx_init(ms)  Avg_op_setup(ms)  Avg_exec(ms)  GFLOPS
ggml     2048   2048   2048   100    f32       f32       96       7.12           0.02              0.22              3.22          5331.60

=== Same Type: BF16+BF16 (FASTEST) ===
Backend  M      N      K      Iters  Src_type  Wei_type  Threads  Avg_total(ms)  Avg_ctx_init(ms)  Avg_op_setup(ms)  Avg_exec(ms)  GFLOPS
ggml     2048   2048   2048   100    bf16      bf16      96       6.11           0.02              0.26              1.65          10419.88

=== Mixed Type: F32+BF16 ===
Backend  M      N      K      Iters  Src_type  Wei_type  Threads  Avg_total(ms)  Avg_ctx_init(ms)  Avg_op_setup(ms)  Avg_exec(ms)  GFLOPS
ggml     2048   2048   2048   100    f32       bf16      96       7.71           0.02              0.24              3.40          5058.19

=== Mixed Type: F32+Q4_0 (Quantized) ===
Backend  M      N      K      Iters  Src_type  Wei_type  Threads  Avg_total(ms)  Avg_ctx_init(ms)  Avg_op_setup(ms)  Avg_exec(ms)  GFLOPS
ggml     2048   2048   2048   100    f32       q4_0      96       3.16           0.17              0.33              2.65          6486.80
```

**Key Findings:**
- **BF16+BF16 is ~2x faster than F32+F32** (1.65ms vs 3.22ms execution time)
- Mixed types (F32 source with any weight type) have similar performance to F32+F32
- Q4_0 quantized weights offer best memory efficiency with good performance

#### Backend Comparison (4096x4096x4096, BF16+BF16, 16 threads):

```bash
# GGML backend with BF16
./build/ops_ggml_benchmark --backend ggml --src_dtype bf16 --wei_dtype bf16 \
  --m 4096 --n 4096 --k 4096 --threads 16 --repeats 50 --warmup 10

# ZenDNN backend with BF16
./build/ops_ggml_benchmark --backend zendnn --src_dtype bf16 --wei_dtype bf16 \
  --m 4096 --n 4096 --k 4096 --threads 16 --repeats 50 --warmup 10
```

```
=== GGML BF16+BF16 ===
Backend  M      N      K      Iters  Src_type  Wei_type  Threads  Avg_total(ms)  Avg_ctx_init(ms)  Avg_op_setup(ms)  Avg_exec(ms)  GFLOPS
ggml     4096   4096   4096   50     bf16      bf16      16       62.73          0.02              1.87              60.84         2259.14

=== ZenDNN BF16+BF16 ===
Backend  M      N      K      Iters  Src_type  Wei_type  Threads  Avg_total(ms)  Avg_ctx_init(ms)  Avg_op_setup(ms)  Avg_exec(ms)  GFLOPS
zendnn   4096   4096   4096   50     bf16      bf16      16       41.21          0.74              1.28              39.19         3507.00
```

**Analysis**: ZenDNN is 1.55× faster than GGML (39.19ms vs 60.84ms execution time) for BF16 workloads.

## Backends

| Backend | Description | Source Types | Weight Types | Operators | MoE Support | Quantization |
|---------|-------------|--------------|--------------|-----------|-------------|--------------|
| `ggml` | GGML CPU backend from llama.cpp | **f32, bf16** | **f32, f16, bf16, q8_0, q4_0** | matmul, matmul_id, layer | ✅ | ✅ q8_0, q4_0 |
| `zendnn` | ZenDNN LoWoHA matmul API (AMD optimized) | **f32, bf16** | **f32, bf16** | matmul, layer (dense only) | ❌ | ❌ |

Use `--backend <ggml|zendnn>` to select the backend (default: ggml).

**Type Compatibility:**
- **GGML backend:**
  - `src=f32` with ANY weight type (f32/f16/bf16/q8_0/q4_0) - most flexible
  - `src=bf16 wei=bf16` ONLY - best performance (~2x faster than f32+f32)
  - Invalid: `src=bf16` with any non-bf16 weight (f32/f16/q8_0/q4_0)
- **ZenDNN backend:**
  - All combinations of f32 and bf16 are supported for both source and weights

**Feature Support:**
- **ZenDNN backend does NOT support:**
  - f16 weight type (use f32 or bf16)
  - Quantized types (q8_0, q4_0) - GGML backend only
  - `matmul_id` operator (MoE expert routing)
  - MoE layer configs (mixtral, qwen3-coder)
- **GGML backend supports all features:** All type combinations (with compatibility rules above)
- For MoE models, always use GGML backend
- For dense models, both backends supported

## Benchmark approach

Each benchmark invocation follows a fixed sequence of phases:

1. **Backend initialization** -- A GGML CPU backend is created and configured with the requested thread count (`--threads`). All subsequent compute is dispatched through this backend using GGML's built-in threadpool.

2. **Context & tensor creation** -- A lightweight GGML context is allocated with `no_alloc=true` (metadata only). Tensors for the operator's inputs and output are declared with the requested shape and dtype. For example, `matmul` creates tensors A `[K, M]` and B `[K, N]`; `matmul_id` also creates a 3D expert weight tensor and an integer routing-index tensor.

3. **Graph construction** -- A GGML compute graph is built containing the operator(s). For `matmul` and `matmul_id` this is a single node; for `layer` mode, all ops from the config file are expanded into one graph. No fusion, no optimization passes -- just the raw primitives.

4. **Buffer allocation** -- A GGML graph allocator (`ggml_gallocr`) reserves and assigns memory for all tensor data. For q4_0 dtype, the repack buffer type is automatically selected to enable q4_0x8 optimization.

5. **Deterministic data fill** -- Input tensors are populated with pseudo-random values from a seeded xorshift32 generator (values in `[-1, 1]`). For q4_0, weights are quantized then passed through `ggml_backend_tensor_set()` which triggers the q4_0 → q4_0x8 conversion. Seeds are fixed so results are reproducible across runs.

6. **Warmup** -- The graph is executed `--warmup` times (default 10) to prime caches, JIT paths, and thread scheduling. These iterations are not timed.

7. **Timed iterations** -- The graph is executed `--repeats` times (default 100). Each iteration is individually timed with `std::chrono::steady_clock`. No timing happens inside GGML kernels -- only wall-clock time around `ggml_backend_graph_compute()`.

8. **Result reporting** -- Min, avg, and max latency are computed from the per-iteration measurements.

9. **Cleanup** -- The graph allocator, GGML context, and backend are freed. No resources leak between runs.

## How it differs from model benchmarks

| Aspect           | ops_ggml_benchmark     | llama-bench / model benchmarks |
|------------------|------------------------|-------------------------------|
| Scope            | Single op or layer     | Full model inference          |
| Dependencies     | GGML + optional ZenDNN | Full llama.cpp + model files  |
| Graph complexity | 1-8 compute nodes      | Hundreds of nodes             |
| Input data       | Synthetic / random     | Tokenized text                |
| Purpose          | Kernel characterization| End-to-end throughput         |
| Quantization     | Explicit dtype selection| Model file determines         |

## Design philosophy

This project is a **measurement harness**, not a compute library. It stays small, explicit, and deterministic. It exists to characterize GGML operator performance in isolation, analogous to how benchdnn characterizes oneDNN / ZenDNN primitives.

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
└── ggml_utils.cpp              # GGML type utilities and quantization helpers

include/
├── benchmark.h                 # Benchmark result structure
├── ggml_matmul_bench.h         # GGML matmul function declarations
├── zendnn_matmul_bench.h       # ZenDNN matmul function declarations
├── layer_bench.h               # Layer benchmark interface
├── layer_config.h              # Layer config structures
├── op_desc.h                   # Operator descriptor (shape, dtype, etc.)
└── ggml_utils.h                # GGML utility functions

configs/
├── llama-3.1-8b.cfg           # Llama 3.1 8B layer config (dense)
├── qwen2-7b.cfg               # Qwen2 7B layer config (dense)
├── mixtral-8x7b.cfg           # Mixtral 8x7B layer config (MoE)
└── qwen3-coder-30b-a3b.cfg    # Qwen3 Coder 30B A3B layer config (MoE)

cmake/
└── FetchLlamaCpp.cmake        # CMake module to fetch llama.cpp
```
