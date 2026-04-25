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

    // Pure GPU timings (excluding CPU overhead)
    double prefill_time_ms_gpu = 0.0;
    double decode_time_ms_gpu = 0.0;
    double total_time_ms_gpu = 0.0;
    double prefill_tokens_per_sec_gpu = 0.0;
    double decode_tokens_per_sec_gpu = 0.0;
};

struct GenerationResult {
    // Batched output: one sequence per item in the batch
    std::vector<std::vector<int>> output_sequences;
    std::vector<std::string> decoded_texts;
    GenerationMetrics metrics;

    // Convenience: number of sequences in the batch
    int batch_size() const { return output_sequences.size(); }
};

struct GenerationConfig {
    int batch_size = 1;
    int max_new_tokens = 20;
    // We can add temperature, top_k, top_p etc. here later
};

} // namespace pipeline
