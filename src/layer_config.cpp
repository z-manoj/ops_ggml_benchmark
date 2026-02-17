#include "layer_config.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

LayerConfig parse_layer_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "error: cannot open config file '%s'\n", path.c_str());
        exit(1);
    }

    LayerConfig cfg;
    std::string line;
    int lineno = 0;

    while (std::getline(f, line)) {
        lineno++;

        // Strip leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line[0] == '#') continue;

        std::istringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "model_name") {
            if (!(ss >> cfg.model_name)) {
                fprintf(stderr, "error: %s:%d: missing value for model_name\n",
                        path.c_str(), lineno);
                exit(1);
            }
        } else if (keyword == "seq_len") {
            if (!(ss >> cfg.seq_len) || cfg.seq_len <= 0) {
                fprintf(stderr, "error: %s:%d: invalid seq_len\n",
                        path.c_str(), lineno);
                exit(1);
            }
        } else if (keyword == "mul_mat") {
            LayerOpDesc op;
            op.op_type = "mul_mat";
            if (!(ss >> op.label >> op.m >> op.n >> op.k) ||
                op.m <= 0 || op.n <= 0 || op.k <= 0) {
                fprintf(stderr, "error: %s:%d: invalid mul_mat line\n",
                        path.c_str(), lineno);
                exit(1);
            }
            cfg.ops.push_back(op);
        } else if (keyword == "mul_mat_id") {
            LayerOpDesc op;
            op.op_type = "mul_mat_id";
            if (!(ss >> op.label >> op.m >> op.n >> op.k
                     >> op.n_experts >> op.n_experts_used) ||
                op.m <= 0 || op.n <= 0 || op.k <= 0 ||
                op.n_experts <= 0 || op.n_experts_used <= 0) {
                fprintf(stderr, "error: %s:%d: invalid mul_mat_id line\n",
                        path.c_str(), lineno);
                exit(1);
            }
            cfg.ops.push_back(op);
        } else {
            fprintf(stderr, "error: %s:%d: unknown keyword '%s'\n",
                    path.c_str(), lineno, keyword.c_str());
            exit(1);
        }
    }

    if (cfg.ops.empty()) {
        fprintf(stderr, "error: %s: no ops defined\n", path.c_str());
        exit(1);
    }

    return cfg;
}
