#pragma once

#include <string>
#include <vector>

// Describes one GEMM operation inside a transformer layer.
struct LayerOpDesc {
    std::string op_type;       // "mul_mat" or "mul_mat_id"
    std::string label;         // e.g. "attn_q", "ffn_gate_exp"
    int         m = 0;
    int         n = 0;
    int         k = 0;

    // mul_mat_id specific
    int         n_experts      = 0;
    int         n_experts_used = 0;

    // Optional: per-expert token counts for realistic routing
    // If empty, uses uniform/random routing
    // If specified, must have n_experts elements that sum to M
    std::vector<int> expert_token_counts;
};

// A parsed layer config file.
struct LayerConfig {
    std::string model_name;
    int         seq_len = 0;
    std::vector<LayerOpDesc> ops;
};

// Parse a layer config file. Exits on error.
LayerConfig parse_layer_config(const std::string& path);
