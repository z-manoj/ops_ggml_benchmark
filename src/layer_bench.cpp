#include "layer_bench.h"
#include "ggml_utils.h"

#include <cstdio>
#include <fstream>

void print_layer_results(const LayerConfig& cfg, const LayerBenchResult& result,
                         const std::string& backend, ggml_type dtype,
                         int threads, int warmup, int repeats) {
    // ZenDNN benchdnn-inspired tabular format with per-iteration averages

    // Amortize one-time costs across all iterations for fair comparison
    double avg_ctx_per_iter = result.ctx_creation_ms / repeats;
    double avg_setup_per_iter = result.op_creation_ms / repeats;
    double avg_total_per_iter = avg_ctx_per_iter + avg_setup_per_iter + result.avg_ms;

    // Print summary header
    printf("\n");
    printf("=== Layer Benchmark: %s ===\n", cfg.model_name.c_str());
    printf("%-8s %-6s %-15s %-8s %-18s %-20s %-20s %-18s %-10s\n",
           "Backend", "Iters", "Data_type", "Threads",
           "Avg_total(ms)", "Avg_ctx_init(ms)", "Avg_op_setup(ms)", "Avg_exec(ms)", "GFLOPS");

    printf("%-8s %-6d %-15s %-8d %-18.2f %-20.2f %-20.2f %-18.2f %-10.2f\n",
           backend.c_str(),
           repeats,
           dtype_to_string(dtype),
           threads,
           avg_total_per_iter,
           avg_ctx_per_iter,
           avg_setup_per_iter,
           result.avg_ms,
           result.total_gflops);

    printf("\n");

    // Print per-operation details
    printf("\n--- Per-Operation Details (%zu ops) ---\n", result.ops.size());
    printf("%-4s %-12s %-16s %-8s %-8s %-8s %-10s %-10s %-10s %-10s\n",
           "Idx", "Op_type", "Label", "M", "N", "K", "GFLOPS", "Min(ms)", "Avg(ms)", "Max(ms)");

    for (size_t i = 0; i < result.ops.size(); i++) {
        const auto& op = result.ops[i];
        printf("%-4zu %-12s %-16s %-8d %-8d %-8d %-10.2f %-10.2f %-10.2f %-10.2f\n",
               i, op.op_type.c_str(), op.label.c_str(),
               op.m, op.n, op.k,
               op.gflops, op.min_ms, op.avg_ms, op.max_ms);
    }
    printf("\n");
}

void write_layer_csv_results(const std::string& csv_path, const LayerConfig& cfg,
                             const LayerBenchResult& result, const std::string& backend,
                             ggml_type dtype, int threads, int warmup, int repeats,
                             bool write_header) {
    std::ofstream csv_file;
    csv_file.open(csv_path, write_header ? std::ios::out : std::ios::app);

    if (!csv_file.is_open()) {
        fprintf(stderr, "warning: failed to open CSV file '%s'\n", csv_path.c_str());
        return;
    }

    // Write header if first entry
    if (write_header) {
        csv_file << "Backend,Model,Iterations,Data_type,Threads,"
                 << "Avg_total_ms,GFLOPS,"
                 << "Avg_ctx_init_ms,Avg_op_setup_ms,Avg_exec_ms,"
                 << "Exec_min_ms,Exec_max_ms,Num_ops\n";
    }

    // Calculate metrics - amortize one-time costs for fair comparison
    double avg_ctx_per_iter = result.ctx_creation_ms / repeats;
    double avg_setup_per_iter = result.op_creation_ms / repeats;
    double avg_total_per_iter = avg_ctx_per_iter + avg_setup_per_iter + result.avg_ms;

    // Write data row
    csv_file << backend << ","
             << cfg.model_name << ","
             << repeats << ","
             << dtype_to_string(dtype) << ","
             << threads << ","
             << avg_total_per_iter << ","
             << result.total_gflops << ","
             << avg_ctx_per_iter << ","
             << avg_setup_per_iter << ","
             << result.avg_ms << ","
             << result.min_ms << "," << result.max_ms << ","
             << result.ops.size() << "\n";

    csv_file.close();
}
