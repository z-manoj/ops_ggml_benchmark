import pandas as pd
import sys

def load_file(path):
    # Read whitespace-separated table
    df = pd.read_csv(path, sep=r'\s+', engine='python')
    return df

def prepare_backend_df(df):
    backend = df["Backend"].iloc[0].lower()

    # Columns that define the GEMM config (common keys)
    key_cols = ["M", "N", "K", "Iters", "Src_type", "Wei_type", "Threads"]

    # Performance/stat columns
    stat_cols = [
        "Avg_total(ms)",
        "Avg_ctx_init(ms)",
        "Avg_op_setup(ms)",
        "Avg_exec(ms)",
        "GFLOPS"
    ]

    # Keep only relevant columns
    df = df[key_cols + stat_cols].copy()

    # Rename stat columns with backend prefix
    rename_map = {col: f"{backend}_{col}" for col in stat_cols}
    df.rename(columns=rename_map, inplace=True)

    return df, backend

def main(file1, file2, output_file):
    df1 = load_file(file1)
    df2 = load_file(file2)

    df1_prepared, backend1 = prepare_backend_df(df1)
    df2_prepared, backend2 = prepare_backend_df(df2)

    key_cols = ["M", "N", "K", "Iters", "Src_type", "Wei_type", "Threads"]

    # Merge on configuration keys
    merged = pd.merge(df1_prepared, df2_prepared, on=key_cols, how="inner")

    # ---- Reorder so metrics are adjacent ----
    stat_cols = [
        "Avg_total(ms)",
        "Avg_ctx_init(ms)",
        "Avg_op_setup(ms)",
        "Avg_exec(ms)",
        "GFLOPS"
    ]

    backends = sorted(
        set(col.split("_")[0] for col in merged.columns if "_" in col)
    )

    new_cols = key_cols.copy()

    for metric in stat_cols:
        for backend in backends:
            col_name = f"{backend}_{metric}"
            if col_name in merged.columns:
                new_cols.append(col_name)

    merged = merged[new_cols]
    # -----------------------------------------

    # ---- Speedup: file1 / file2 for Avg_total(ms) ----
    col1 = f"{backend1}_Avg_total(ms)"
    col2 = f"{backend2}_Avg_total(ms)"
    merged["Speedup(file1_vs_file2)"] = (merged[col2] / merged[col1]).map(lambda x: f"{x:.2f}x")
    # ---------------------------------------------------

    merged.to_csv(output_file, index=False)
    print(f"Combined file written to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python combine_benchmarks.py zendnn.txt ggml.txt combined.csv")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]

    main(file1, file2, output_file)