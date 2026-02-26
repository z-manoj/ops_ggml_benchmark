# Memory Bandwidth Analysis - Why AVX-512 Only Gives 3% Improvement

## Thread Scaling Results (NUMA-Aware)

| Threads | NUMA Nodes | Throughput | TFLOPS/core | Efficiency |
|---------|------------|------------|-------------|------------|
| 24      | 1          | 1.21 TF    | 0.0504      | 100%       |
| 48      | 2          | 2.16 TF    | 0.0450      | 89%        |
| **96**  | **4**      | **3.13 TF**| **0.0326**  | **65%**    |
| 144     | 6          | 2.89 TF    | 0.0200      | 40%        |
| 192     | 8          | 3.09 TF    | 0.0160      | 32%        |

## Key Findings

### 1. **Memory Bandwidth Saturated at 96 Threads**
- Peak throughput: **3.13 TFLOPS at 96 threads (4 NUMA nodes)**
- Efficiency drops 68% from single NUMA node
- Beyond 96 threads, throughput **decreases** due to memory contention

### 2. **Why AVX-512 Only Gives 3% Improvement**
AVX-512 makes compute 2x faster, but we're **memory-bound**:
- **Bottleneck**: Loading FP16 weights from memory
- **Not bottleneck**: FP32 FMA compute
- **Result**: Faster ALUs wait on slower memory → minimal speedup

### 3. **Arithmetic Intensity Too Low**
For MoE operations (matmul_id):
```
Memory traffic = Load weights (K×N per expert) + inputs + outputs
Compute        = 2×M×N×K FLOPs
Intensity      = ~1-2 FLOPs/byte (memory-bound threshold)
```

### 4. **Why BF16 Native Compute Won't Help**
`_mm512_dpbf16_ps()` requires BF16-format data, but GGML stores FP16:
- Path: Load FP16 → convert FP32 → convert BF16 → dpbf16_ps
- **More expensive** than direct FP32 FMA!
- Would only help if GGML stored weights in BF16 natively

## Optimizations That Would Help

### High Impact:
1. ✅ **Use 96 threads (4 NUMA nodes)** - optimal bandwidth/efficiency  
2. **Increase weight reuse** - batch more tokens per expert (already done)
3. **Better prefetching** - hide memory latency with aggressive prefetch

### Medium Impact:
4. **Reduce memory footprint** - quantize to INT8/INT4 (requires GGML changes)
5. **Block for L3** - tile to fit multiple experts in shared L3 cache

### Low Impact (already saturated):
6. ❌ Wider SIMD (AVX-512) - compute not bottleneck
7. ❌ BF16 native - data format mismatch
8. ❌ More threads - makes it worse due to contention

## Recommendations

### For Current Implementation:
- **Use 96 threads on 4 NUMA nodes** (current default)
- AVX-512 provides 3-5% benefit (better than nothing)
- Accept memory bandwidth as fundamental limit

### For Future Work (requires GGML changes):
- Store weights in BF16 format → use `_mm512_dpbf16_ps()`
- INT8 quantization with VNNI (`_mm512_dpbusd_epi32`)
- Structured sparsity to reduce memory traffic

## Conclusion

The 3% AVX-512 improvement is **expected** given memory bandwidth saturation. 
Further gains require reducing memory traffic (quantization, sparsity) or 
increasing arithmetic intensity (larger batch sizes), not faster compute units.
