#!/bin/bash
#
# profile_backends.sh - Comprehensive profiling of GGML vs ZenDNN backends
#
# Usage: ./profile_backends.sh [options]
#
# Options:
#   --skip-ggml        Skip GGML profiling (profile ZenDNN only)
#   --skip-zendnn      Skip ZenDNN profiling (profile GGML only)
#   --skip-mem         Skip memory profiling (perf mem - requires root)
#   --skip-lock        Skip lock contention profiling
#   --skip-vtune       Skip Intel VTune profiling
#   --output-dir DIR   Output directory for results (default: profile_results_TIMESTAMP)
#   --help             Show this help message
#

set -e  # Exit on error

# Default configuration
PROFILE_GGML=1
PROFILE_ZENDNN=1
PROFILE_MEM=1
PROFILE_LOCK=1
PROFILE_VTUNE=1
OUTPUT_DIR="profile_results_$(date +%Y%m%d_%H%M%S)"

# Benchmark parameters
BENCHMARK_BIN="./build/ops_ggml_benchmark"
NUMA_CMD="numactl --physcpubind=0-95 --membind=0,1,2,3"
COMMON_ARGS="--op matmul_id --m 4096 --n 512 --k 4096 --n_experts 8 --n_experts_used 2 --routing_pattern custom --expert_token_counts 126,323,80,68,256,37,15,119 --src_dtype bf16 --wei_dtype bf16 --threads 96"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-ggml)
            PROFILE_GGML=0
            shift
            ;;
        --skip-zendnn)
            PROFILE_ZENDNN=0
            shift
            ;;
        --skip-mem)
            PROFILE_MEM=0
            shift
            ;;
        --skip-lock)
            PROFILE_LOCK=0
            shift
            ;;
        --skip-vtune)
            PROFILE_VTUNE=0
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if benchmark binary exists
if [[ ! -f "$BENCHMARK_BIN" ]]; then
    echo -e "${RED}Error: Benchmark binary not found: $BENCHMARK_BIN${NC}"
    echo "Please build the project first: cmake --build build -j\$(nproc)"
    exit 1
fi

# Check for required tools
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${YELLOW}Warning: $1 not found. Skipping $2 profiling.${NC}"
        return 1
    fi
    return 0
}

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}=== Profiling GGML vs ZenDNN Backends ===${NC}"
echo -e "Output directory: ${BLUE}$OUTPUT_DIR${NC}"
echo ""

# Function to run profiling for a specific backend
profile_backend() {
    local backend=$1
    local backend_upper=$(echo "$backend" | tr '[:lower:]' '[:upper:]')

    echo -e "${GREEN}=== Profiling $backend_upper Backend ===${NC}"

    # 1. Performance counters (perf stat)
    echo -e "${BLUE}[1/5] Running perf stat (performance counters)...${NC}"
    if check_tool perf "perf stat"; then
        $NUMA_CMD perf stat \
            -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,\
L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
            -o "$OUTPUT_DIR/${backend}_perf_stat.txt" \
            $BENCHMARK_BIN --backend $backend $COMMON_ARGS 2>&1 | tee "$OUTPUT_DIR/${backend}_benchmark_output.txt"

        echo -e "${GREEN}✓ Perf stat saved to: $OUTPUT_DIR/${backend}_perf_stat.txt${NC}"
    fi
    echo ""

    # 2. Hotspot analysis (perf record)
    echo -e "${BLUE}[2/5] Running perf record (hotspot analysis)...${NC}"
    if check_tool perf "perf record"; then
        $NUMA_CMD perf record -F 99 -g --call-graph dwarf \
            -o "$OUTPUT_DIR/${backend}_perf.data" \
            $BENCHMARK_BIN --backend $backend $COMMON_ARGS > /dev/null 2>&1

        # Generate text report
        perf report -i "$OUTPUT_DIR/${backend}_perf.data" -g --stdio \
            > "$OUTPUT_DIR/${backend}_perf_report.txt" 2>&1

        echo -e "${GREEN}✓ Perf data saved to: $OUTPUT_DIR/${backend}_perf.data${NC}"
        echo -e "${GREEN}✓ Perf report saved to: $OUTPUT_DIR/${backend}_perf_report.txt${NC}"
        echo -e "  View interactively: ${YELLOW}perf report -i $OUTPUT_DIR/${backend}_perf.data${NC}"
    fi
    echo ""

    # 3. Lock contention (perf lock) - only for the backend we're investigating
    if [[ $PROFILE_LOCK -eq 1 ]] && [[ "$backend" == "zendnn" ]]; then
        echo -e "${BLUE}[3/5] Running perf lock (synchronization analysis)...${NC}"
        if check_tool perf "perf lock"; then
            # Check kernel support for lock events
            if perf list | grep -q "lock:"; then
                $NUMA_CMD perf lock record \
                    -o "$OUTPUT_DIR/${backend}_lock.data" \
                    $BENCHMARK_BIN --backend $backend $COMMON_ARGS > /dev/null 2>&1

                perf lock report -i "$OUTPUT_DIR/${backend}_lock.data" \
                    > "$OUTPUT_DIR/${backend}_lock_report.txt" 2>&1

                echo -e "${GREEN}✓ Lock report saved to: $OUTPUT_DIR/${backend}_lock_report.txt${NC}"
            else
                echo -e "${YELLOW}Warning: Kernel lock events not available. Skipping lock profiling.${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}[3/5] Skipping lock profiling for $backend${NC}"
    fi
    echo ""

    # 4. Memory profiling (perf mem) - requires root
    if [[ $PROFILE_MEM -eq 1 ]]; then
        echo -e "${BLUE}[4/5] Running perf mem (memory access analysis)...${NC}"
        if check_tool perf "perf mem"; then
            if [[ $EUID -eq 0 ]]; then
                perf mem record -a -o "$OUTPUT_DIR/${backend}_mem.data" \
                    $NUMA_CMD $BENCHMARK_BIN --backend $backend $COMMON_ARGS > /dev/null 2>&1

                perf mem report -i "$OUTPUT_DIR/${backend}_mem.data" \
                    > "$OUTPUT_DIR/${backend}_mem_report.txt" 2>&1

                echo -e "${GREEN}✓ Memory report saved to: $OUTPUT_DIR/${backend}_mem_report.txt${NC}"
            else
                echo -e "${YELLOW}Warning: perf mem requires root. Skipping memory profiling.${NC}"
                echo -e "  Run with: ${YELLOW}sudo ./profile_backends.sh${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}[4/5] Skipping memory profiling${NC}"
    fi
    echo ""

    # 5. Intel VTune (if available) - only for the backend we're investigating
    if [[ $PROFILE_VTUNE -eq 1 ]] && [[ "$backend" == "zendnn" ]]; then
        echo -e "${BLUE}[5/5] Running Intel VTune (hotspots)...${NC}"
        if check_tool vtune "VTune"; then
            vtune -collect hotspots -knob sampling-mode=hw \
                -result-dir "$OUTPUT_DIR/${backend}_vtune" \
                -- $NUMA_CMD $BENCHMARK_BIN --backend $backend $COMMON_ARGS > /dev/null 2>&1

            # Generate summary report
            vtune -report summary -result-dir "$OUTPUT_DIR/${backend}_vtune" \
                -format text -report-output "$OUTPUT_DIR/${backend}_vtune_summary.txt" 2>&1

            echo -e "${GREEN}✓ VTune results saved to: $OUTPUT_DIR/${backend}_vtune/${NC}"
            echo -e "${GREEN}✓ VTune summary saved to: $OUTPUT_DIR/${backend}_vtune_summary.txt${NC}"
        fi
    else
        echo -e "${YELLOW}[5/5] Skipping VTune profiling for $backend${NC}"
    fi
    echo ""
}

# Profile GGML backend
if [[ $PROFILE_GGML -eq 1 ]]; then
    profile_backend "ggml"
fi

# Profile ZenDNN backend
if [[ $PROFILE_ZENDNN -eq 1 ]]; then
    profile_backend "zendnn"
fi

# Generate comparison summary
echo -e "${GREEN}=== Generating Comparison Summary ===${NC}"

SUMMARY_FILE="$OUTPUT_DIR/SUMMARY.md"

cat > "$SUMMARY_FILE" <<EOF
# Profiling Summary: GGML vs ZenDNN

**Date**: $(date)
**Workload**: matmul_id (MoE) - 4096x512x4096, bf16, 8 experts (2 active)
**Routing Pattern**: Custom imbalanced (126,323,80,68,256,37,15,119 tokens/expert)
**System**: 96 threads, NUMA nodes 0,1,2,3

---

## Files Generated

EOF

if [[ $PROFILE_GGML -eq 1 ]]; then
    cat >> "$SUMMARY_FILE" <<EOF
### GGML Backend
- \`ggml_benchmark_output.txt\` - Raw benchmark results
- \`ggml_perf_stat.txt\` - CPU performance counters
- \`ggml_perf_report.txt\` - Hotspot analysis (text)
- \`ggml_perf.data\` - Hotspot analysis (binary, use: \`perf report -i ggml_perf.data\`)

EOF
fi

if [[ $PROFILE_ZENDNN -eq 1 ]]; then
    cat >> "$SUMMARY_FILE" <<EOF
### ZenDNN Backend
- \`zendnn_benchmark_output.txt\` - Raw benchmark results
- \`zendnn_perf_stat.txt\` - CPU performance counters
- \`zendnn_perf_report.txt\` - Hotspot analysis (text)
- \`zendnn_perf.data\` - Hotspot analysis (binary, use: \`perf report -i zendnn_perf.data\`)
EOF

    if [[ $PROFILE_LOCK -eq 1 ]]; then
        cat >> "$SUMMARY_FILE" <<EOF
- \`zendnn_lock_report.txt\` - Lock contention analysis
EOF
    fi

    if [[ $PROFILE_MEM -eq 1 ]]; then
        cat >> "$SUMMARY_FILE" <<EOF
- \`zendnn_mem_report.txt\` - Memory access patterns
EOF
    fi

    if [[ $PROFILE_VTUNE -eq 1 ]]; then
        cat >> "$SUMMARY_FILE" <<EOF
- \`zendnn_vtune/\` - VTune results directory
- \`zendnn_vtune_summary.txt\` - VTune summary report
EOF
    fi
fi

cat >> "$SUMMARY_FILE" <<EOF

---

## Quick Analysis Commands

\`\`\`bash
# Compare IPC (instructions per cycle)
grep "instructions" ${OUTPUT_DIR}/*_perf_stat.txt

# Compare cache miss rates
grep "cache-misses" ${OUTPUT_DIR}/*_perf_stat.txt

# View top hotspots (GGML)
head -50 ${OUTPUT_DIR}/ggml_perf_report.txt

# View top hotspots (ZenDNN)
head -50 ${OUTPUT_DIR}/zendnn_perf_report.txt

# Interactive flamegraph
perf report -i ${OUTPUT_DIR}/zendnn_perf.data
\`\`\`

---

## Key Metrics to Compare

1. **IPC (Instructions Per Cycle)**: Higher is better
2. **Cache Miss Rate**: Lower is better
3. **LLC Miss Rate**: Lower is better (critical for memory-bound ops)
4. **Top Functions**: Where is CPU time spent?
5. **Lock Contention**: High contention = poor scaling
6. **Memory Bandwidth**: Close to peak = good utilization

---

## Expected Findings

Based on 3.57x performance gap (GGML faster), investigate:

- [ ] ZenDNN using FP32 conversion instead of native bf16
- [ ] Poor expert load balancing (idle threads)
- [ ] Cache thrashing due to memory layout
- [ ] Excessive synchronization barriers
- [ ] Missing AVX-512 optimizations

EOF

echo -e "${GREEN}✓ Summary saved to: $OUTPUT_DIR/SUMMARY.md${NC}"
echo ""

# Extract quick comparison metrics
echo -e "${GREEN}=== Quick Comparison ===${NC}"
echo ""

if [[ -f "$OUTPUT_DIR/ggml_perf_stat.txt" ]] && [[ -f "$OUTPUT_DIR/zendnn_perf_stat.txt" ]]; then
    echo -e "${BLUE}Performance Counters:${NC}"
    echo ""

    echo "GGML Backend:"
    grep -E "instructions|cycles|cache-misses" "$OUTPUT_DIR/ggml_perf_stat.txt" | head -6
    echo ""

    echo "ZenDNN Backend:"
    grep -E "instructions|cycles|cache-misses" "$OUTPUT_DIR/zendnn_perf_stat.txt" | head -6
    echo ""
fi

echo -e "${GREEN}=== Profiling Complete ===${NC}"
echo -e "All results saved to: ${BLUE}$OUTPUT_DIR/${NC}"
echo -e "Read summary: ${YELLOW}cat $OUTPUT_DIR/SUMMARY.md${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review $OUTPUT_DIR/SUMMARY.md for analysis commands"
echo "2. Compare hotspot reports: diff $OUTPUT_DIR/{ggml,zendnn}_perf_report.txt"
echo "3. View interactive flamegraph: perf report -i $OUTPUT_DIR/zendnn_perf.data"
echo ""
