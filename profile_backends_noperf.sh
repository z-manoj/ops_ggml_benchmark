#!/bin/bash
#
# profile_backends_noperf.sh - Profiling without perf (no root required)
#
# Uses alternative profiling methods that don't require kernel access

set -e

# Configuration
OUTPUT_DIR="profile_results_$(date +%Y%m%d_%H%M%S)"
BENCHMARK_BIN="./build/ops_ggml_benchmark"
NUMA_CMD="numactl --physcpubind=0-95 --membind=0,1,2,3"
COMMON_ARGS="--op matmul_id --m 4096 --n 512 --k 4096 --n_experts 8 --n_experts_used 2 --routing_pattern custom --expert_token_counts 126,323,80,68,256,37,15,119 --src_dtype bf16 --wei_dtype bf16 --threads 96"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}=== Profiling GGML vs ZenDNN (No Root Required) ===${NC}"
echo -e "Output directory: ${BLUE}$OUTPUT_DIR${NC}"
echo ""

# Function to profile with gprof (if available)
profile_with_sampling() {
    local backend=$1
    echo -e "${GREEN}=== Profiling $backend Backend ===${NC}"

    # 1. Run benchmark and capture output
    echo -e "${BLUE}[1/3] Running benchmark (10 iterations for sampling)...${NC}"
    $NUMA_CMD $BENCHMARK_BIN --backend $backend $COMMON_ARGS \
        > "$OUTPUT_DIR/${backend}_benchmark_output.txt" 2>&1
    echo -e "${GREEN}✓ Benchmark output: $OUTPUT_DIR/${backend}_benchmark_output.txt${NC}"
    echo ""

    # 2. Run with time command for detailed timing
    echo -e "${BLUE}[2/3] Running with detailed timing...${NC}"
    /usr/bin/time -v $NUMA_CMD $BENCHMARK_BIN --backend $backend $COMMON_ARGS \
        > /dev/null 2> "$OUTPUT_DIR/${backend}_time_verbose.txt"
    echo -e "${GREEN}✓ Time report: $OUTPUT_DIR/${backend}_time_verbose.txt${NC}"
    echo ""

    # 3. Run with strace to see system calls (sample mode)
    echo -e "${BLUE}[3/3] Running with strace (system call analysis)...${NC}"
    strace -c -f -o "$OUTPUT_DIR/${backend}_strace.txt" \
        $NUMA_CMD $BENCHMARK_BIN --backend $backend $COMMON_ARGS > /dev/null 2>&1 || true
    echo -e "${GREEN}✓ Strace report: $OUTPUT_DIR/${backend}_strace.txt${NC}"
    echo ""
}

# Profile both backends
profile_with_sampling "ggml"
profile_with_sampling "zendnn"

# Generate summary
SUMMARY_FILE="$OUTPUT_DIR/SUMMARY.md"

cat > "$SUMMARY_FILE" <<'EOF'
# Profiling Summary: GGML vs ZenDNN

**Note**: This profile was run without kernel perf access. For detailed CPU profiling, run:
```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
./profile_backends.sh --skip-mem
```

---

## Files Generated

### GGML Backend
- `ggml_benchmark_output.txt` - Raw benchmark results
- `ggml_time_verbose.txt` - Resource usage (CPU%, memory, context switches)
- `ggml_strace.txt` - System call statistics

### ZenDNN Backend
- `zendnn_benchmark_output.txt` - Raw benchmark results
- `zendnn_time_verbose.txt` - Resource usage
- `zendnn_strace.txt` - System call statistics

---

## Analysis Commands

```bash
# Compare benchmark results
diff ggml_benchmark_output.txt zendnn_benchmark_output.txt

# Compare CPU usage
grep "Percent of CPU" *_time_verbose.txt

# Compare context switches (high = poor threading)
grep "context switches" *_time_verbose.txt

# Compare system calls
head -20 *_strace.txt
```

---

## Key Metrics

Extract from `*_time_verbose.txt`:
- **User time**: CPU time in user space (kernel execution)
- **System time**: CPU time in kernel (syscalls, scheduling)
- **Percent of CPU**: >9000% is good for 96 threads
- **Voluntary context switches**: Thread yielding (waiting on locks/IO)
- **Involuntary context switches**: Preemption (too many threads)

Extract from `*_strace.txt`:
- **futex calls**: High count = lock contention
- **mmap/munmap**: Memory allocation overhead
- **clone calls**: Thread creation

---

## To Enable Full Profiling

Run one of these commands (requires sudo):

```bash
# Temporary (until reboot)
sudo sysctl -w kernel.perf_event_paranoid=-1

# Permanent
echo "kernel.perf_event_paranoid = -1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

Then run the full profiling script:
```bash
./profile_backends.sh --skip-mem
```

EOF

echo -e "${GREEN}✓ Summary saved to: $OUTPUT_DIR/SUMMARY.md${NC}"
echo ""

# Quick comparison
echo -e "${GREEN}=== Quick Comparison ===${NC}"
echo ""

if [[ -f "$OUTPUT_DIR/ggml_time_verbose.txt" ]] && [[ -f "$OUTPUT_DIR/zendnn_time_verbose.txt" ]]; then
    echo -e "${BLUE}Resource Usage:${NC}"
    echo ""

    echo "GGML Backend:"
    grep -E "User time|System time|Percent of CPU|context switches" "$OUTPUT_DIR/ggml_time_verbose.txt"
    echo ""

    echo "ZenDNN Backend:"
    grep -E "User time|System time|Percent of CPU|context switches" "$OUTPUT_DIR/zendnn_time_verbose.txt"
    echo ""
fi

if [[ -f "$OUTPUT_DIR/ggml_strace.txt" ]] && [[ -f "$OUTPUT_DIR/zendnn_strace.txt" ]]; then
    echo -e "${BLUE}System Call Summary:${NC}"
    echo ""

    echo "GGML Backend (top 10 syscalls):"
    head -15 "$OUTPUT_DIR/ggml_strace.txt" | tail -10
    echo ""

    echo "ZenDNN Backend (top 10 syscalls):"
    head -15 "$OUTPUT_DIR/zendnn_strace.txt" | tail -10
    echo ""
fi

echo -e "${GREEN}=== Profiling Complete ===${NC}"
echo -e "Results saved to: ${BLUE}$OUTPUT_DIR/${NC}"
echo ""
echo -e "${YELLOW}For detailed CPU profiling, enable perf:${NC}"
echo "  sudo sysctl -w kernel.perf_event_paranoid=-1"
echo "  ./profile_backends.sh --skip-mem"
echo ""
