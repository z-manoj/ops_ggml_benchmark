#include "routing_utils.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <numeric>

// ---------------------------------------------------------------------------
// Routing pattern generation
// ---------------------------------------------------------------------------

bool validate_expert_counts(
    const std::vector<int>& counts,
    int64_t n_tokens,
    int64_t n_experts_used
) {
    if (counts.empty()) return false;

    int64_t expected_sum = n_tokens * n_experts_used;
    int64_t actual_sum = 0;
    for (int c : counts) {
        if (c < 0) return false;
        actual_sum += c;
    }

    return actual_sum == expected_sum;
}

std::vector<int> generate_uniform_distribution(
    int64_t n_tokens,
    int64_t n_experts,
    int64_t n_experts_used
) {
    // Each expert gets approximately equal number of (token, slot) assignments
    std::vector<int> counts(n_experts, 0);
    int64_t total_assignments = n_tokens * n_experts_used;
    int64_t base_count = total_assignments / n_experts;
    int64_t remainder = total_assignments % n_experts;

    for (int64_t i = 0; i < n_experts; i++) {
        counts[i] = base_count + (i < remainder ? 1 : 0);
    }

    return counts;
}

std::vector<int> generate_skewed_distribution(
    int64_t n_tokens,
    int64_t n_experts,
    int64_t n_experts_used,
    int seed
) {
    // 80/20 rule: top 20% of experts get 80% of tokens
    std::vector<int> counts(n_experts, 0);
    int64_t total_assignments = n_tokens * n_experts_used;

    int64_t top_experts = std::max(int64_t(1), n_experts / 5);  // 20% of experts
    int64_t hot_assignments = static_cast<int64_t>(total_assignments * 0.8);  // 80% of work
    int64_t cold_assignments = total_assignments - hot_assignments;

    // Distribute hot assignments among top experts
    int64_t hot_per_expert = hot_assignments / top_experts;
    int64_t hot_remainder = hot_assignments % top_experts;

    for (int64_t i = 0; i < top_experts; i++) {
        counts[i] = hot_per_expert + (i < hot_remainder ? 1 : 0);
    }

    // Distribute cold assignments among remaining experts
    int64_t cold_experts = n_experts - top_experts;
    if (cold_experts > 0) {
        int64_t cold_per_expert = cold_assignments / cold_experts;
        int64_t cold_remainder = cold_assignments % cold_experts;

        for (int64_t i = top_experts; i < n_experts; i++) {
            counts[i] = cold_per_expert + ((i - top_experts) < cold_remainder ? 1 : 0);
        }
    }

    return counts;
}

std::vector<int> generate_random_distribution(
    int64_t n_tokens,
    int64_t n_experts,
    int64_t n_experts_used,
    int seed
) {
    // Generate random but balanced distribution
    std::vector<int> counts(n_experts, 0);
    int64_t total_assignments = n_tokens * n_experts_used;

    std::mt19937 rng(seed);

    // Start with uniform base
    int64_t base_count = total_assignments / n_experts;
    for (int64_t i = 0; i < n_experts; i++) {
        counts[i] = base_count;
    }

    // Distribute remainder randomly with some variance
    int64_t remaining = total_assignments - (base_count * n_experts);
    std::uniform_int_distribution<int64_t> dist(0, n_experts - 1);

    // Add some random variance (up to 20% of base count)
    int64_t variance_pool = static_cast<int64_t>(base_count * n_experts * 0.1);
    for (int64_t i = 0; i < variance_pool; i++) {
        int64_t donor = dist(rng);
        int64_t recipient = dist(rng);
        if (counts[donor] > 0 && donor != recipient) {
            counts[donor]--;
            counts[recipient]++;
        }
    }

    // Distribute original remainder
    for (int64_t i = 0; i < remaining; i++) {
        counts[dist(rng)]++;
    }

    return counts;
}

std::vector<int32_t> generate_routing_ids(
    int64_t n_tokens,
    int64_t n_experts,
    int64_t n_experts_used,
    const std::vector<int>& expert_token_counts,
    const std::string& pattern,
    int seed
) {
    std::vector<int> counts;

    // Determine which pattern to use
    if (pattern == "custom") {
        if (expert_token_counts.empty()) {
            fprintf(stderr, "error: custom routing pattern requires expert_token_counts\n");
            exit(1);
        }
        counts = expert_token_counts;
    } else if (pattern == "uniform") {
        counts = generate_uniform_distribution(n_tokens, n_experts, n_experts_used);
    } else if (pattern == "skewed") {
        counts = generate_skewed_distribution(n_tokens, n_experts, n_experts_used, seed);
    } else if (pattern == "random") {
        counts = generate_random_distribution(n_tokens, n_experts, n_experts_used, seed);
    } else {
        fprintf(stderr, "error: unknown routing pattern '%s'\n", pattern.c_str());
        fprintf(stderr, "       valid patterns: uniform, custom, random, skewed\n");
        exit(1);
    }

    // Validate counts
    if (!validate_expert_counts(counts, n_tokens, n_experts_used)) {
        fprintf(stderr, "error: invalid expert_token_counts\n");
        fprintf(stderr, "       expected sum: %lld (n_tokens=%lld * n_experts_used=%lld)\n",
                (long long)(n_tokens * n_experts_used), (long long)n_tokens, (long long)n_experts_used);
        int64_t actual_sum = 0;
        for (int c : counts) actual_sum += c;
        fprintf(stderr, "       actual sum: %lld\n", (long long)actual_sum);
        fprintf(stderr, "       counts: [");
        for (size_t i = 0; i < counts.size(); i++) {
            fprintf(stderr, "%d%s", counts[i], i + 1 < counts.size() ? "," : "");
        }
        fprintf(stderr, "]\n");
        exit(1);
    }

    // Build list of (token, slot, expert) assignments
    struct Assignment {
        int32_t token;
        int32_t slot;
        int32_t expert;
    };
    std::vector<Assignment> assignments;
    assignments.reserve(n_tokens * n_experts_used);

    // Assign (token, slot) pairs to experts based on counts
    int64_t assignment_idx = 0;
    for (int64_t expert_id = 0; expert_id < static_cast<int64_t>(counts.size()); expert_id++) {
        int count = counts[expert_id];
        for (int i = 0; i < count; i++) {
            int64_t token = assignment_idx / n_experts_used;
            int64_t slot = assignment_idx % n_experts_used;
            assignments.push_back({
                static_cast<int32_t>(token),
                static_cast<int32_t>(slot),
                static_cast<int32_t>(expert_id)
            });
            assignment_idx++;
        }
    }

    // Shuffle assignments to avoid sequential patterns
    // (but use deterministic seed for reproducibility)
    std::mt19937 rng(seed);
    std::shuffle(assignments.begin(), assignments.end(), rng);

    // Build output routing IDs array: ids[token * n_experts_used + slot] = expert_id
    std::vector<int32_t> ids(n_tokens * n_experts_used);
    for (const auto& a : assignments) {
        ids[a.token * n_experts_used + a.slot] = a.expert;
    }

    return ids;
}
