#include "moe_config.h"
#include "ggml_utils.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cstring>

// Trim whitespace from both ends of a string
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Parse comma-separated integers: "126,323,80,68,256,37,15,119"
static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> result;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        result.push_back(std::atoi(trim(token).c_str()));
    }
    return result;
}

OpDesc parse_moe_config(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        fprintf(stderr, "error: cannot open MoE config file '%s'\n", filename.c_str());
        exit(1);
    }

    OpDesc desc;
    // Set defaults
    desc.op_name = "matmul_id";
    desc.backend = "zendnn";
    desc.m = 4096;
    desc.n = 512;
    desc.k = 4096;
    desc.src_dtype = GGML_TYPE_F32;
    desc.wei_dtype = GGML_TYPE_F32;
    desc.n_experts = 8;
    desc.n_experts_used = 2;
    desc.routing_pattern = "uniform";
    desc.routing_seed = 42;
    desc.threads = 8;
    desc.repeats = 100;
    desc.warmup = 10;

    std::string line;
    int lineno = 0;

    while (std::getline(f, line)) {
        lineno++;

        // Trim whitespace
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Parse key=value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            fprintf(stderr, "warning: %s:%d: skipping invalid line (no '='): %s\n",
                    filename.c_str(), lineno, line.c_str());
            continue;
        }

        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));

        if (value.empty()) {
            fprintf(stderr, "warning: %s:%d: empty value for key '%s'\n",
                    filename.c_str(), lineno, key.c_str());
            continue;
        }

        // Parse each key
        if (key == "op_name") {
            desc.op_name = value;
        }
        else if (key == "backend") {
            desc.backend = value;
        }
        else if (key == "m") {
            desc.m = std::atoi(value.c_str());
        }
        else if (key == "n") {
            desc.n = std::atoi(value.c_str());
        }
        else if (key == "k") {
            desc.k = std::atoi(value.c_str());
        }
        else if (key == "n_experts") {
            desc.n_experts = std::atoi(value.c_str());
        }
        else if (key == "n_experts_used") {
            desc.n_experts_used = std::atoi(value.c_str());
        }
        else if (key == "src_dtype") {
            desc.src_dtype = parse_dtype(value);
            if (desc.src_dtype == GGML_TYPE_COUNT) {
                fprintf(stderr, "error: %s:%d: unknown src_dtype '%s'\n",
                        filename.c_str(), lineno, value.c_str());
                exit(1);
            }
        }
        else if (key == "wei_dtype") {
            desc.wei_dtype = parse_dtype(value);
            if (desc.wei_dtype == GGML_TYPE_COUNT) {
                fprintf(stderr, "error: %s:%d: unknown wei_dtype '%s'\n",
                        filename.c_str(), lineno, value.c_str());
                exit(1);
            }
        }
        else if (key == "routing_pattern") {
            desc.routing_pattern = value;
        }
        else if (key == "routing_seed") {
            desc.routing_seed = std::atoi(value.c_str());
        }
        else if (key == "expert_token_counts") {
            desc.expert_token_counts = parse_int_list(value);
        }
        else if (key == "threads") {
            desc.threads = std::atoi(value.c_str());
        }
        else if (key == "repeats") {
            desc.repeats = std::atoi(value.c_str());
        }
        else if (key == "warmup") {
            desc.warmup = std::atoi(value.c_str());
        }
        else if (key == "description") {
            // Metadata - store but don't use
            continue;
        }
        else {
            fprintf(stderr, "warning: %s:%d: unknown key '%s'\n",
                    filename.c_str(), lineno, key.c_str());
        }
    }

    // Validation
    if (desc.op_name != "matmul_id") {
        fprintf(stderr, "error: MoE config must have op_name=matmul_id (got '%s')\n",
                desc.op_name.c_str());
        exit(1);
    }

    if (desc.backend != "ggml" && desc.backend != "zendnn") {
        fprintf(stderr, "error: unknown backend '%s' (must be 'ggml' or 'zendnn')\n",
                desc.backend.c_str());
        exit(1);
    }

    if (desc.m <= 0 || desc.n <= 0 || desc.k <= 0) {
        fprintf(stderr, "error: dimensions must be positive (m=%d, n=%d, k=%d)\n",
                desc.m, desc.n, desc.k);
        exit(1);
    }

    if (desc.n_experts <= 0 || desc.n_experts_used <= 0) {
        fprintf(stderr, "error: n_experts and n_experts_used must be positive\n");
        exit(1);
    }

    if (desc.n_experts_used > desc.n_experts) {
        fprintf(stderr, "error: n_experts_used (%d) cannot exceed n_experts (%d)\n",
                desc.n_experts_used, desc.n_experts);
        exit(1);
    }

    // Validate src_dtype (only f32 and bf16 supported)
    if (desc.src_dtype != GGML_TYPE_F32 && desc.src_dtype != GGML_TYPE_BF16) {
        fprintf(stderr, "error: src_dtype must be f32 or bf16\n");
        exit(1);
    }

    // Validate dtypes for ZenDNN backend
    if (desc.backend == "zendnn") {
        if (desc.wei_dtype == GGML_TYPE_F16) {
            fprintf(stderr, "error: ZenDNN backend does not support f16 weights\n");
            exit(1);
        }
        if (desc.wei_dtype == GGML_TYPE_Q8_0 || desc.wei_dtype == GGML_TYPE_Q4_0) {
            fprintf(stderr, "error: ZenDNN backend does not support quantized weights\n");
            exit(1);
        }
    }

    // GGML backend type compatibility
    if (desc.backend == "ggml") {
        if (desc.src_dtype != desc.wei_dtype) {
            // Mixed data types - src must be F32
            if (desc.src_dtype != GGML_TYPE_F32) {
                fprintf(stderr, "error: GGML backend with mixed data types requires src_dtype=f32\n");
                fprintf(stderr, "       Current: src=%s, wei=%s\n",
                        dtype_to_string(desc.src_dtype), dtype_to_string(desc.wei_dtype));
                exit(1);
            }
        }
    }

    printf("Loaded MoE config from: %s\n", filename.c_str());
    printf("  Backend: %s, Dimensions: %dx%dx%d\n", desc.backend.c_str(), desc.m, desc.n, desc.k);
    printf("  Experts: %d total, %d active per token\n", desc.n_experts, desc.n_experts_used);
    printf("  Src: %s, Wei: %s\n", dtype_to_string(desc.src_dtype), dtype_to_string(desc.wei_dtype));
    printf("  Routing: %s", desc.routing_pattern.c_str());
    if (desc.routing_pattern == "custom" && !desc.expert_token_counts.empty()) {
        printf(" (");
        for (size_t i = 0; i < desc.expert_token_counts.size(); i++) {
            printf("%d", desc.expert_token_counts[i]);
            if (i < desc.expert_token_counts.size() - 1) printf(",");
        }
        printf(")");
    }
    printf("\n");
    printf("  Threads: %d, Repeats: %d, Warmup: %d\n", desc.threads, desc.repeats, desc.warmup);
    printf("\n");

    return desc;
}
