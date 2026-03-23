#pragma once

#include <vector>
#include <string>

namespace pipeline {

struct GenerationMetrics {
    int prompt_tokens = 0;
    int generated_tokens = 0;

    double prefill_time_ms = 0.0;
    double decode_time_ms = 0.0;
    double total_time_ms = 0.0;

    double prefill_tokens_per_sec = 0.0;
    double decode_tokens_per_sec = 0.0;
};

struct GenerationResult {
    std::vector<int> output_sequence;
    std::string decoded_text;
    GenerationMetrics metrics;
};

struct GenerationConfig {
    int max_new_tokens = 20;
    // We can add temperature, top_k, top_p etc. here later
};

} // namespace pipeline
