#pragma once

#include "pipeline/metrics.hpp"
#include "memory.h"
#include <vector>
#include <cuda_runtime.h>

namespace pipeline {

// Abstract base class for execution strategies.
//
// Uses the Template Method pattern: generate() defines the execution skeleton
// and owns all timing/metrics computation. Subclasses implement the pure
// virtual hooks (run_prefill, run_decode, finalize) without any awareness
// of timing or metrics — guaranteeing consistent metric definitions across
// all strategies.
//
// Input is a batch of sequences (vector of vector<int>), each potentially
// a different length. Strategies are responsible for padding/packing.
class ExecutionStrategy {
public:
    virtual ~ExecutionStrategy() = default;

    // Template method — NOT virtual. This is the single entry point.
    // Defines what "prefill time" and "decode time" mean structurally.
    GenerationResult generate(const std::vector<std::vector<int>>& input_sequences, const GenerationConfig& config) {
        GenerationResult result;
        result.output_sequences = input_sequences;

        // Total prompt tokens across the batch
        int total_prompt_tokens = 0;
        for (const auto& seq : input_sequences) {
            total_prompt_tokens += seq.size();
        }
        result.metrics.prompt_tokens = total_prompt_tokens;

        // ---- PREFILL: base class defines and times this phase ----
        result.metrics.prefill_time_ms = time_cuda_execution([&]() {
            run_prefill(result, config);
        });

        // ---- DECODE: base class defines and times this phase ----
        result.metrics.decode_time_ms = time_cuda_execution([&]() {
            run_decode(result, config);
        });

        // ---- METRICS: computed once, consistently, for all strategies ----
        result.metrics.total_time_ms = result.metrics.prefill_time_ms 
                                     + result.metrics.decode_time_ms;

        result.metrics.prefill_tokens_per_sec = 
            result.metrics.prompt_tokens / (result.metrics.prefill_time_ms / 1000.0);

        double decode_time_sec = result.metrics.decode_time_ms / 1000.0;
        if (result.metrics.generated_tokens > 1) {
            // generated_tokens includes the 1 token from prefill
            result.metrics.decode_tokens_per_sec = 
                (result.metrics.generated_tokens - 1) / decode_time_sec;
        } else {
            result.metrics.decode_tokens_per_sec = 0.0;
        }

        // ---- FINALIZE: strategy-specific post-processing (e.g., detokenize) ----
        finalize(result);

        return result;
    }

protected:
    // Strategies implement ONLY these — no timing, no metric math.

    // Process the full prompt batch and produce the first new token for each sequence.
    virtual void run_prefill(GenerationResult& result, const GenerationConfig& config) = 0;

    // Generate remaining tokens autoregressively for all sequences.
    virtual void run_decode(GenerationResult& result, const GenerationConfig& config) = 0;

    // Post-processing (e.g., decode token IDs to text).
    virtual void finalize(GenerationResult& result) = 0;

    // Subclasses must provide access to their CUDA stream for timing.
    virtual cudaStream_t get_stream() const = 0;

private:
    // Precise GPU timing via CUDA events — used only by the base class.
    template <typename F>
    double time_cuda_execution(F&& func) {
        cudaStream_t s = get_stream();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, s);
        func();
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return static_cast<double>(ms);
    }
};

} // namespace pipeline
