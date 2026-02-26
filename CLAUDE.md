# ops_ggml_benchmark - Claude Code Instructions

## Project Overview

This is a microbenchmark harness for GGML operators from llama.cpp. It measures individual operator performance (matmul, matmul_id/MoE, and full transformer layers) in isolation, similar to how oneDNN's benchdnn works.

**Key principle**: This is a **measurement harness only**. It does NOT implement new kernels or modify llama.cpp. All compute is performed by GGML's existing CPU backend.

## Project Structure

```
ops_ggml_benchmark/
├── include/           # Public headers
│   ├── benchmark.h    # Core benchmark interface
│   ├── ggml_utils.h   # GGML helper functions
│   ├── matmul_bench.h # Single matmul/matmul_id benchmarks
│   ├── layer_bench.h  # Multi-op layer benchmarks
│   ├── layer_config.h # Config file parser
│   └── op_desc.h      # Operation descriptors
├── src/               # Implementation
│   ├── main.cpp       # CLI arg parsing & dispatch
│   ├── benchmark.cpp  # Core timing & statistics
│   ├── ggml_utils.cpp # Backend & context helpers
│   ├── matmul_bench.cpp
│   ├── layer_bench.cpp
│   ├── layer_config.cpp
│   └── custom/        # Custom kernels (AVX2/FMA)
│       ├── custom_moe.cpp
│       └── custom_moe_bench.cpp
├── configs/           # Layer config files
│   ├── mixtral-8x7b.cfg
│   └── qwen3-coder-30b-a3b.cfg
└── cmake/             # CMake modules
    └── FetchLlamaCpp.cmake
```

## Build System

```bash
# Standard build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# First build fetches llama.cpp via FetchContent
# Subsequent builds reuse cached source
```

**Requirements**:
- CMake >= 3.21
- C++17 compiler
- Git (for FetchContent)
- OpenMP (required for custom kernels)

**IMPORTANT**: Custom SIMD kernels in `src/custom/custom_moe.cpp` are compiled with `-mavx2 -mfma -mf16c` flags. Do not remove these flags.

## Development Guidelines

### Documentation Policy
**IMPORTANT**: Do NOT create summary documents, reports, or analysis files unless explicitly requested by the user.
- No `SUMMARY.md`, `ANALYSIS.md`, `REPORT.md`, etc. unless asked
- Focus on code changes and technical implementation
- Only create documentation when user specifically asks for it

### Code Style
- Use modern C++17 features where appropriate
- Keep functions focused and single-purpose
- Prefer explicit types over auto when it improves clarity
- Match existing formatting (4-space indentation, Allman-style braces in some places)

### GGML Integration
- Always use GGML's backend API (`ggml_backend_*`)
- Never bypass the backend and access tensors directly
- Use `ggml_gallocr` for all memory allocation
- Initialize backends before creating contexts
- Clean up in reverse order: graph allocator → context → backend

### Benchmark Invariants
1. **Deterministic inputs**: Use seeded xorshift32 for all random data
2. **Warmup before timing**: Default 10 warmup iterations
3. **Individual iteration timing**: Time each repeat separately for min/avg/max
4. **No GGML modifications**: Never patch or modify llama.cpp source
5. **Machine-parsable output**: Keep output format stable for scripting

### Adding New Operators
When adding a new operator benchmark:
1. Create header in `include/` with public interface
2. Implement in `src/` with these phases:
   - Backend initialization
   - Context & tensor creation (`no_alloc=true`)
   - Graph construction
   - Buffer allocation via `ggml_gallocr`
   - Deterministic data fill
   - Warmup iterations
   - Timed iterations
   - Result reporting
   - Cleanup
3. Add CLI argument parsing in `src/main.cpp`
4. Update README.md with examples
5. Add to operator table in README

### Custom Kernels (src/custom/)
The `src/custom/` directory contains experimental AVX2/FMA optimized kernels for MoE operations. These are NOT part of GGML and are project-specific.

- `custom_moe.cpp`: Custom matmul_id implementation with flattened parallel dispatch
- `custom_moe_bench.cpp`: Benchmark harness for custom kernels

These files must be compiled with SIMD flags: `-mavx2 -mfma -mf16c`

## Testing

### Manual Testing
```bash
# Basic matmul
./build/ops_ggml_benchmark --op matmul --m 512 --n 512 --k 512

# Multi-shape
./build/ops_ggml_benchmark --op matmul --shapes 512x512x512,1024x1024x1024

# MoE
./build/ops_ggml_benchmark --op matmul_id --m 2048 --n 64 --k 2048

# Layer benchmark
./build/ops_ggml_benchmark --op layer --config configs/mixtral-8x7b.cfg

# Custom MoE kernels
./build/ops_ggml_benchmark --op custom_moe --config configs/mixtral-8x7b.cfg
```

### Automated Testing
`test_all_features.sh` runs a comprehensive test suite covering all operators and features.

```bash
./test_all_features.sh
```

## Git Workflow

**Main branch**: `main`
**Current branch**: `mkumar/custom_moe`

### Commit Guidelines
- Write clear, imperative commit messages
- Focus on the "why" not the "what"
- Always include co-author line:
  ```
  Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
  ```

### Branch Strategy
- Feature branches: `username/feature-name`
- Bug fixes: `username/fix-description`
- Always branch from `main` for new work

## Common Pitfalls

1. **Memory leaks**: Always free contexts, backends, and graph allocators in reverse order
2. **Uninitialized tensors**: Use deterministic fills, never leave tensors with random memory
3. **Thread count**: Respect `--threads` argument, don't hardcode thread counts
4. **FLOPS calculation**: For matmul: `2*M*N*K` (not M*N*K)
5. **Timing**: Only time `ggml_backend_graph_compute()`, not setup or teardown
6. **Config files**: MoE ops require `n_experts` and `n_experts_used` parameters

## File Formats

### Batch file (--batch)
One shape per line, whitespace or 'x' delimited:
```
# Comment lines start with #
512 512 512
1024x1024x1024
```

### Layer config file (--config)
```
model_name  my-model
seq_len     512

mul_mat     label  M  N  K
mul_mat_id  label  M  N  K  n_experts  n_experts_used
```

## Performance Notes

- Results are highly sensitive to CPU frequency scaling (use performance governor)
- First run may be slower due to cold caches
- Thread count should typically match physical cores
- Large shapes (>4096) may show memory bandwidth bottlenecks
- MoE performance depends on routing pattern (currently uniformly random)

## When Making Changes

1. **Read before editing**: Always read existing files before modifying
2. **Test after changes**: Run `test_all_features.sh` or relevant manual tests
3. **Update docs**: If adding features, update README.md
4. **Preserve output format**: Many users parse benchmark output programmatically
5. **Ask before breaking changes**: Consult before changing CLI interface or output format
