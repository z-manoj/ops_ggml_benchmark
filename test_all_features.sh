#!/bin/bash
# Comprehensive test script for all operators and dtype combinations
# Tests: matmul, matmul_id, layer with all supported backends and dtypes
# DO NOT COMMIT TO GIT - For testing only

set -e

NUMA_CMD="numactl --physcpubind=0-7 --membind=0"
BENCHMARK="./build/ops_ggml_benchmark"

# Small problem size for quick validation
M=1024
N=512
K=1024
THREADS=8
REPEATS=3
WARMUP=1

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

total_tests=0
passed_tests=0
failed_tests=0

echo "========================================================================"
echo "COMPREHENSIVE FEATURE TEST - ALL OPERATORS AND COMBINATIONS"
echo "========================================================================"
echo "Problem size: M=$M, N=$N, K=$K, Threads=$THREADS"
echo "Repeats=$REPEATS, Warmup=$WARMUP"
echo "========================================================================"
echo ""

# Function to run test and capture result
run_test() {
    local test_name="$1"
    local cmd="$2"
    local expect_fail="${3:-false}"

    echo -n "[$test_name] ... "

    if eval "$cmd" > /tmp/bench_output.txt 2>&1; then
        if [ "$expect_fail" = "true" ]; then
            echo -e "${RED}✗ FAIL (expected to fail but passed)${NC}"
            failed_tests=$((failed_tests + 1))
            return 1
        else
            # Extract key metrics
            local backend=$(grep -E "^(ggml|zendnn)" /tmp/bench_output.txt | awk '{print $1}')
            local exec_time=$(grep -E "^(ggml|zendnn)" /tmp/bench_output.txt | awk '{print $11}')
            local gflops=$(grep -E "^(ggml|zendnn)" /tmp/bench_output.txt | awk '{print $12}')
            echo -e "${GREEN}✓ PASS${NC} (exec: ${exec_time}ms, ${gflops} GFLOPS)"
            passed_tests=$((passed_tests + 1))
            return 0
        fi
    else
        if [ "$expect_fail" = "true" ]; then
            echo -e "${GREEN}✓ PASS (correctly rejected)${NC}"
            passed_tests=$((passed_tests + 1))
            return 0
        else
            echo -e "${RED}✗ FAIL${NC}"
            echo "   Error:"
            grep -i "error" /tmp/bench_output.txt | head -3 | sed 's/^/   /'
            failed_tests=$((failed_tests + 1))
            return 1
        fi
    fi
}

# ===========================================================================
# 1. MATMUL OPERATOR - GGML BACKEND
# ===========================================================================
echo "========================================================================"
echo "1. MATMUL OPERATOR - GGML BACKEND"
echo "========================================================================"

# Valid combinations
total_tests=$((total_tests + 1))
run_test "ggml matmul f32+f32" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype f32 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul f32+f16" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype f16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul f32+bf16" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype bf16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul f32+q8_0" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype q8_0 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul f32+q4_0" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype q4_0 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul bf16+bf16" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul --m $M --n $N --k $K --src_dtype bf16 --wei_dtype bf16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

# Invalid combinations (should fail)
total_tests=$((total_tests + 1))
run_test "ggml matmul bf16+f32 (expect fail)" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul --m $M --n $N --k $K --src_dtype bf16 --wei_dtype f32 --threads $THREADS --repeats $REPEATS --warmup $WARMUP" \
    true

total_tests=$((total_tests + 1))
run_test "ggml matmul bf16+f16 (expect fail)" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul --m $M --n $N --k $K --src_dtype bf16 --wei_dtype f16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP" \
    true

total_tests=$((total_tests + 1))
run_test "ggml matmul bf16+q8_0 (expect fail)" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul --m $M --n $N --k $K --src_dtype bf16 --wei_dtype q8_0 --threads $THREADS --repeats $REPEATS --warmup $WARMUP" \
    true

echo ""

# ===========================================================================
# 2. MATMUL OPERATOR - ZENDNN BACKEND
# ===========================================================================
echo "========================================================================"
echo "2. MATMUL OPERATOR - ZENDNN BACKEND"
echo "========================================================================"

# All f32/bf16 combinations (ZenDNN supports all)
total_tests=$((total_tests + 1))
run_test "zendnn matmul f32+f32" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype f32 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "zendnn matmul f32+bf16" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype bf16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "zendnn matmul bf16+f32" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul --m $M --n $N --k $K --src_dtype bf16 --wei_dtype f32 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "zendnn matmul bf16+bf16" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul --m $M --n $N --k $K --src_dtype bf16 --wei_dtype bf16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

# Invalid combinations (should fail)
total_tests=$((total_tests + 1))
run_test "zendnn matmul f32+f16 (expect fail)" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype f16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP" \
    true

total_tests=$((total_tests + 1))
run_test "zendnn matmul f32+q8_0 (expect fail)" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype q8_0 --threads $THREADS --repeats $REPEATS --warmup $WARMUP" \
    true

total_tests=$((total_tests + 1))
run_test "zendnn matmul f32+q4_0 (expect fail)" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul --m $M --n $N --k $K --src_dtype f32 --wei_dtype q4_0 --threads $THREADS --repeats $REPEATS --warmup $WARMUP" \
    true

echo ""

# ===========================================================================
# 3. MATMUL_ID OPERATOR - GGML BACKEND
# ===========================================================================
echo "========================================================================"
echo "3. MATMUL_ID OPERATOR (MoE) - GGML BACKEND"
echo "========================================================================"

N_EXPERTS=8
N_EXPERTS_USED=2

# Valid combinations
total_tests=$((total_tests + 1))
run_test "ggml matmul_id f32+f32" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype f32 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul_id f32+f16" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype f16 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul_id f32+bf16" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype bf16 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul_id f32+q8_0" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype q8_0 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul_id f32+q4_0" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype q4_0 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul_id bf16+bf16" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype bf16 --wei_dtype bf16 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

# Invalid combinations (should fail)
total_tests=$((total_tests + 1))
run_test "ggml matmul_id bf16+f32 (expect fail)" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype bf16 --wei_dtype f32 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP" \
    true

echo ""

# ===========================================================================
# 4. MATMUL_ID OPERATOR - ZENDNN BACKEND
# ===========================================================================
echo "========================================================================"
echo "4. MATMUL_ID OPERATOR (MoE) - ZENDNN BACKEND"
echo "========================================================================"

# All f32/bf16 combinations
total_tests=$((total_tests + 1))
run_test "zendnn matmul_id f32+f32" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype f32 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "zendnn matmul_id f32+bf16" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype bf16 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "zendnn matmul_id bf16+f32" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul_id --m $M --n $N --k $K --src_dtype bf16 --wei_dtype f32 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "zendnn matmul_id bf16+bf16" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul_id --m $M --n $N --k $K --src_dtype bf16 --wei_dtype bf16 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

echo ""

# ===========================================================================
# 5. MATMUL_ID ROUTING PATTERNS - Test all routing options
# ===========================================================================
echo "========================================================================"
echo "5. MATMUL_ID ROUTING PATTERNS"
echo "========================================================================"

total_tests=$((total_tests + 1))
run_test "ggml matmul_id routing=uniform" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype f32 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --routing_pattern uniform --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul_id routing=random" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype f32 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --routing_pattern random --routing_seed 42 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul_id routing=skewed" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype f32 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --routing_pattern skewed --routing_seed 42 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "ggml matmul_id routing=custom" \
    "$NUMA_CMD $BENCHMARK --backend ggml --op matmul_id --m $M --n $N --k $K --src_dtype f32 --wei_dtype f32 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --routing_pattern custom --expert_token_counts 126,323,80,68,256,37,15,119 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

# Test ZenDNN routing patterns too
total_tests=$((total_tests + 1))
run_test "zendnn matmul_id routing=uniform" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul_id --m $M --n $N --k $K --src_dtype bf16 --wei_dtype bf16 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --routing_pattern uniform --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

total_tests=$((total_tests + 1))
run_test "zendnn matmul_id routing=custom" \
    "$NUMA_CMD $BENCHMARK --backend zendnn --op matmul_id --m $M --n $N --k $K --src_dtype bf16 --wei_dtype bf16 --n_experts $N_EXPERTS --n_experts_used $N_EXPERTS_USED --routing_pattern custom --expert_token_counts 126,323,80,68,256,37,15,119 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

echo ""

# ===========================================================================
# 6. LAYER OPERATOR - GGML BACKEND (Dense configs)
# ===========================================================================
echo "========================================================================"
echo "6. LAYER OPERATOR - GGML BACKEND (Dense Configs)"
echo "========================================================================"

if [ -f "configs/llama-3.1-8b.cfg" ]; then
    total_tests=$((total_tests + 1))
    run_test "ggml layer llama-3.1-8b f32+f32" \
        "$NUMA_CMD $BENCHMARK --backend ggml --op layer --config configs/llama-3.1-8b.cfg --src_dtype f32 --wei_dtype f32 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

    total_tests=$((total_tests + 1))
    run_test "ggml layer llama-3.1-8b f32+bf16" \
        "$NUMA_CMD $BENCHMARK --backend ggml --op layer --config configs/llama-3.1-8b.cfg --src_dtype f32 --wei_dtype bf16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

    total_tests=$((total_tests + 1))
    run_test "ggml layer llama-3.1-8b bf16+bf16" \
        "$NUMA_CMD $BENCHMARK --backend ggml --op layer --config configs/llama-3.1-8b.cfg --src_dtype bf16 --wei_dtype bf16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

    total_tests=$((total_tests + 1))
    run_test "ggml layer llama-3.1-8b f32+q4_0" \
        "$NUMA_CMD $BENCHMARK --backend ggml --op layer --config configs/llama-3.1-8b.cfg --src_dtype f32 --wei_dtype q4_0 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"
else
    echo -e "${YELLOW}⚠ Skipping llama-3.1-8b.cfg (file not found)${NC}"
fi

if [ -f "configs/qwen2-7b.cfg" ]; then
    total_tests=$((total_tests + 1))
    run_test "ggml layer qwen2-7b f32+f32" \
        "$NUMA_CMD $BENCHMARK --backend ggml --op layer --config configs/qwen2-7b.cfg --src_dtype f32 --wei_dtype f32 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

    total_tests=$((total_tests + 1))
    run_test "ggml layer qwen2-7b bf16+bf16" \
        "$NUMA_CMD $BENCHMARK --backend ggml --op layer --config configs/qwen2-7b.cfg --src_dtype bf16 --wei_dtype bf16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"
else
    echo -e "${YELLOW}⚠ Skipping qwen2-7b.cfg (file not found)${NC}"
fi

echo ""

# ===========================================================================
# 7. LAYER OPERATOR - ZENDNN BACKEND (Dense configs only)
# ===========================================================================
echo "========================================================================"
echo "7. LAYER OPERATOR - ZENDNN BACKEND (Dense Configs)"
echo "========================================================================"

if [ -f "configs/llama-3.1-8b.cfg" ]; then
    total_tests=$((total_tests + 1))
    run_test "zendnn layer llama-3.1-8b f32+f32" \
        "$NUMA_CMD $BENCHMARK --backend zendnn --op layer --config configs/llama-3.1-8b.cfg --src_dtype f32 --wei_dtype f32 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

    total_tests=$((total_tests + 1))
    run_test "zendnn layer llama-3.1-8b bf16+bf16" \
        "$NUMA_CMD $BENCHMARK --backend zendnn --op layer --config configs/llama-3.1-8b.cfg --src_dtype bf16 --wei_dtype bf16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"
else
    echo -e "${YELLOW}⚠ Skipping llama-3.1-8b.cfg (file not found)${NC}"
fi

if [ -f "configs/qwen2-7b.cfg" ]; then
    total_tests=$((total_tests + 1))
    run_test "zendnn layer qwen2-7b f32+bf16" \
        "$NUMA_CMD $BENCHMARK --backend zendnn --op layer --config configs/qwen2-7b.cfg --src_dtype f32 --wei_dtype bf16 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

    total_tests=$((total_tests + 1))
    run_test "zendnn layer qwen2-7b bf16+f32" \
        "$NUMA_CMD $BENCHMARK --backend zendnn --op layer --config configs/qwen2-7b.cfg --src_dtype bf16 --wei_dtype f32 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"
else
    echo -e "${YELLOW}⚠ Skipping qwen2-7b.cfg (file not found)${NC}"
fi

echo ""

# ===========================================================================
# 8. LAYER OPERATOR - MoE CONFIGS (GGML only)
# ===========================================================================
echo "========================================================================"
echo "8. LAYER OPERATOR - MoE CONFIGS (GGML Only)"
echo "========================================================================"

if [ -f "configs/mixtral-8x7b.cfg" ]; then
    total_tests=$((total_tests + 1))
    run_test "ggml layer mixtral-8x7b f32+f32" \
        "$NUMA_CMD $BENCHMARK --backend ggml --op layer --config configs/mixtral-8x7b.cfg --src_dtype f32 --wei_dtype f32 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"

    total_tests=$((total_tests + 1))
    run_test "ggml layer mixtral-8x7b f32+q4_0" \
        "$NUMA_CMD $BENCHMARK --backend ggml --op layer --config configs/mixtral-8x7b.cfg --src_dtype f32 --wei_dtype q4_0 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"
else
    echo -e "${YELLOW}⚠ Skipping mixtral-8x7b.cfg (file not found)${NC}"
fi

if [ -f "configs/qwen3-coder-30b-a3b.cfg" ]; then
    total_tests=$((total_tests + 1))
    run_test "ggml layer qwen3-coder-30b-a3b f32+q8_0" \
        "$NUMA_CMD $BENCHMARK --backend ggml --op layer --config configs/qwen3-coder-30b-a3b.cfg --src_dtype f32 --wei_dtype q8_0 --threads $THREADS --repeats $REPEATS --warmup $WARMUP"
else
    echo -e "${YELLOW}⚠ Skipping qwen3-coder-30b-a3b.cfg (file not found)${NC}"
fi

echo ""
echo "========================================================================"
echo "FINAL SUMMARY"
echo "========================================================================"
echo "Total tests run: $total_tests"
echo -e "${GREEN}Passed: $passed_tests${NC}"
echo -e "${RED}Failed: $failed_tests${NC}"
echo ""

if [ $failed_tests -eq 0 ]; then
    echo -e "${GREEN}✓✓✓ ALL TESTS PASSED! ✓✓✓${NC}"
    exit 0
else
    echo -e "${RED}✗✗✗ SOME TESTS FAILED ✗✗✗${NC}"
    exit 1
fi
