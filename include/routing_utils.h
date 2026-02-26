#pragma once

#include <vector>
#include <string>
#include <cstdint>

// ---------------------------------------------------------------------------
// Routing utilities for matmul_id (MoE expert routing)
//
// Provides configurable routing patterns for fair comparison between backends
// and realistic load testing scenarios.
// ---------------------------------------------------------------------------

// Generate routing IDs based on configuration
// Returns: flat array of routing IDs with size = n_tokens * n_experts_used
//          ids[token * n_experts_used + slot] = expert_id
std::vector<int32_t> generate_routing_ids(
    int64_t n_tokens,                           // number of tokens (N)
    int64_t n_experts,                          // total expert count
    int64_t n_experts_used,                     // experts per token
    const std::vector<int>& expert_token_counts, // per-expert token counts (custom pattern)
    const std::string& pattern = "uniform",     // "uniform", "custom", "random", "skewed"
    int seed = 42                               // seed for random patterns
);

// Validate that expert_token_counts sums correctly
// Required sum: n_tokens * n_experts_used (total routing assignments)
bool validate_expert_counts(
    const std::vector<int>& counts,
    int64_t n_tokens,
    int64_t n_experts_used
);

// Generate uniform distribution (equal tokens per expert)
std::vector<int> generate_uniform_distribution(
    int64_t n_tokens,
    int64_t n_experts,
    int64_t n_experts_used
);

// Generate skewed distribution (80/20 rule: 80% of tokens to 20% of experts)
std::vector<int> generate_skewed_distribution(
    int64_t n_tokens,
    int64_t n_experts,
    int64_t n_experts_used,
    int seed
);

// Generate random distribution (random but balanced)
std::vector<int> generate_random_distribution(
    int64_t n_tokens,
    int64_t n_experts,
    int64_t n_experts_used,
    int seed
);
