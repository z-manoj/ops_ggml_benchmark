
export ZENDNNL_MATMUL_ALGO=1

BINARY=./build_ggml/ops_ggml_benchmark
OUTDIR=results/bench/ggml_bf16_to_bf16
mkdir -p "$OUTDIR"

CONFIGS=(
   "gemm/qwen2-7b.gemm"
)

SEQ_LENS=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
THREADS=(16 32 64)
WEI_DTYPES=(bf16)

declare -A CPU_BIND
CPU_BIND[16]="96-111"
CPU_BIND[32]="96-127"
CPU_BIND[64]="96-159"
CPU_BIND[128]="96-191"

declare -A MEM_BIND
MEM_BIND[16]="4"
MEM_BIND[32]="4,5"
MEM_BIND[64]="4,5,6"
MEM_BIND[128]="4,5,6,7"

for cfg in "${CONFIGS[@]}"; do

    model_name=$(awk '/^model_name/{print $2}' "$cfg")

    mapfile -t MM_LABELS < <(awk '/^mul_mat/{print $2}' "$cfg")
    mapfile -t MM_NS     < <(awk '/^mul_mat/{print $3}' "$cfg")
    mapfile -t MM_KS     < <(awk '/^mul_mat/{print $4}' "$cfg")
    num_ops=${#MM_LABELS[@]}

    # Single output file per config
    outfile="${OUTDIR}/${model_name}.txt"
    # Truncate/create fresh for this run
    > "$outfile"

    total=$(( num_ops * ${#SEQ_LENS[@]} * ${#THREADS[@]} * ${#WEI_DTYPES[@]} ))
    count=0
    first_iter=true

    echo "=== Config: $cfg | Model: $model_name | ${num_ops} ops | Output: $outfile ==="

    for wei in "${WEI_DTYPES[@]}"; do
        for t in "${THREADS[@]}"; do

            cpubind="${CPU_BIND[$t]}"
            membind="${MEM_BIND[$t]}"

            for m in "${SEQ_LENS[@]}"; do
                for i in "${!MM_LABELS[@]}"; do

                    label="${MM_LABELS[$i]}"
                    N="${MM_NS[$i]}"
                    K="${MM_KS[$i]}"

                    count=$((count + 1))
                    echo "[${count}/${total}] model=${model_name} op=${label} M=${N} K=${K} N=${m} wei=${wei} threads=${t}"

                    # Capture output of binary
                    raw_output=$(numactl --physcpubind="${cpubind}" --membind="${membind}" \
                        $BINARY \
                            --backend ggml \
                            --op matmul \
                            --n      "$m" \
                            --k      "$K" \
                            --src_dtype bf16 \
                            --wei_dtype "$wei" \
                            --m "$N" \
                            --threads "$t" \
                            --warmup 100 \
                            --repeats 100 \
                        2>&1)

                    if $first_iter; then
                        # Write everything (header + data row) on first iteration
                        echo "$raw_output" >> "$outfile"
                        first_iter=false
                    else
                        # Skip the header line (first line), append only data row(s)
                        echo "$raw_output" | tail -n +2 >> "$outfile"
                    fi

                done
            done
        done
    done

    echo ""
    echo "Config done. Results in: $outfile"
    echo ""

done

echo "Bench 1 complete. Results in: $OUTDIR"
