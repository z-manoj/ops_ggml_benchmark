#include "benchmark.h"
#include "ggml_matmul_bench.h"
#include "ggml_utils.h"

#ifdef ENABLE_ZENDNN
#include "zendnn_matmul_bench.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sys/stat.h>

BenchResult run_benchmark(const OpDesc& desc) {
    // Route to backend-specific implementation
    if (desc.backend == "zendnn") {
#ifdef ENABLE_ZENDNN
        if (desc.op_name == "matmul") {
            return bench_matmul_zendnn(desc);
        } else if (desc.op_name == "matmul_id") {
            return bench_matmul_id_zendnn(desc);
        } else {
            fprintf(stderr, "error: unknown operator '%s'\n", desc.op_name.c_str());
            exit(1);
        }
#else
        fprintf(stderr, "error: ZenDNN backend not enabled. Rebuild with -DENABLE_ZENDNN=ON\n");
        exit(1);
#endif
    } else if (desc.backend == "ggml") {
        if (desc.op_name == "matmul") {
            return bench_matmul_ggml(desc);
        } else if (desc.op_name == "matmul_id") {
            return bench_matmul_id_ggml(desc);
        } else {
            fprintf(stderr, "error: unknown operator '%s'\n", desc.op_name.c_str());
            exit(1);
        }
    } else {
        fprintf(stderr, "error: unknown backend '%s'\n", desc.backend.c_str());
        exit(1);
    }
}

void print_results(const OpDesc& desc, const BenchResult& result) {
    // ZenDNN benchdnn-inspired tabular format with per-iteration averages
    // Header
    printf("%-8s %-6s %-6s %-6s %-6s %-15s %-15s %-8s %-18s %-20s %-20s %-18s %-10s\n",
           "Backend", "M", "N", "K", "Iters", "Src_type", "Wei_type", "Threads",
           "Avg_total(ms)", "Avg_ctx_init(ms)", "Avg_op_setup(ms)", "Avg_exec(ms)", "GFLOPS");

    // Calculate GFLOPS
    double flops;
    if (desc.op_name == "matmul_id") {
        // For matmul_id: FLOPs = 2 * M * K * n_experts_used * N
        flops = 2.0 * desc.m * desc.k * desc.n_experts_used * desc.n;
    } else {
        // For regular matmul: FLOPs = 2 * M * N * K
        flops = 2.0 * desc.m * desc.n * desc.k;
    }
    double gflops = flops / (result.avg_ms * 1e-3) / 1e9;

    // Amortize one-time costs across all iterations for fair comparison
    double avg_ctx_per_iter = result.ctx_creation_ms / desc.repeats;
    double avg_setup_per_iter = result.op_creation_ms / desc.repeats;
    double avg_total_per_iter = avg_ctx_per_iter + avg_setup_per_iter + result.avg_ms;

    // Data row
    printf("%-8s %-6d %-6d %-6d %-6d %-15s %-15s %-8d %-18.2f %-20.2f %-20.2f %-18.2f %-18.2f\n",
           desc.backend.c_str(),
           desc.m, desc.n, desc.k,
           desc.repeats,
           dtype_to_string(desc.src_dtype),
           dtype_to_string(desc.wei_dtype),
           desc.threads,
           avg_total_per_iter,
           avg_ctx_per_iter,
           avg_setup_per_iter,
           result.avg_ms,
           gflops);

    printf("\n");
}

void write_csv_results(const std::string& csv_path, const OpDesc& desc,
                       const BenchResult& result, bool write_header) {
    std::ofstream csv_file;
    csv_file.open(csv_path, write_header ? std::ios::out : std::ios::app);

    if (!csv_file.is_open()) {
        fprintf(stderr, "warning: failed to open CSV file '%s'\n", csv_path.c_str());
        return;
    }

    // Write header if this is the first entry
    if (write_header) {
        csv_file << "Backend,M,N,K,Iterations,Src_type,Wei_type,Threads,"
                 << "Avg_total_ms,GFLOPS,"
                 << "Avg_ctx_init_ms,Avg_op_setup_ms,Avg_exec_ms,"
                 << "Exec_min_ms,Exec_max_ms\n";
    }

    // Calculate metrics - amortize one-time costs for fair comparison
    double flops;
    if (desc.op_name == "matmul_id") {
        // For matmul_id: FLOPs = 2 * M * K * n_experts_used * N
        flops = 2.0 * desc.m * desc.k * desc.n_experts_used * desc.n;
    } else {
        // For regular matmul: FLOPs = 2 * M * N * K
        flops = 2.0 * desc.m * desc.n * desc.k;
    }
    double gflops = flops / (result.avg_ms * 1e-3) / 1e9;

    double avg_ctx_per_iter = result.ctx_creation_ms / desc.repeats;
    double avg_setup_per_iter = result.op_creation_ms / desc.repeats;
    double avg_total_per_iter = avg_ctx_per_iter + avg_setup_per_iter + result.avg_ms;

    // Write data row
    csv_file << desc.backend << ","
             << desc.m << "," << desc.n << "," << desc.k << ","
             << desc.repeats << ","
             << dtype_to_string(desc.src_dtype) << ","
             << dtype_to_string(desc.wei_dtype) << ","
             << desc.threads << ","
             << avg_total_per_iter << ","
             << gflops << ","
             << avg_ctx_per_iter << ","
             << avg_setup_per_iter << ","
             << result.avg_ms << ","
             << result.min_ms << "," << result.max_ms << "\n";

    csv_file.close();
}
