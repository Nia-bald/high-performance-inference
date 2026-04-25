#pragma once

#include "pipeline/metrics.hpp"
#include "memory.h"
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

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
        auto prefill_times = time_execution([&]() {
            run_prefill(result, config);
        });
        result.metrics.prefill_time_ms = prefill_times.first;
        result.metrics.prefill_time_ms_gpu = prefill_times.second;

        // ---- DECODE: base class defines and times this phase ----
        auto decode_times = time_execution([&]() {
            run_decode(result, config);
        });
        result.metrics.decode_time_ms = decode_times.first;
        result.metrics.decode_time_ms_gpu = decode_times.second;

        // ---- METRICS: computed once, consistently, for all strategies ----
        result.metrics.total_time_ms = result.metrics.prefill_time_ms + result.metrics.decode_time_ms;
        result.metrics.prefill_tokens_per_sec = result.metrics.prompt_tokens / (result.metrics.prefill_time_ms / 1000.0);
        
        result.metrics.total_time_ms_gpu = result.metrics.prefill_time_ms_gpu + result.metrics.decode_time_ms_gpu;
        result.metrics.prefill_tokens_per_sec_gpu = result.metrics.prompt_tokens / (result.metrics.prefill_time_ms_gpu / 1000.0);

        double decode_time_sec = result.metrics.decode_time_ms / 1000.0;
        double decode_time_sec_gpu = result.metrics.decode_time_ms_gpu / 1000.0;
        size_t batch_size = input_sequences.size();

        if (result.metrics.generated_tokens > batch_size) {
            // generated_tokens includes the tokens from prefill
            result.metrics.decode_tokens_per_sec = 
                (result.metrics.generated_tokens - batch_size) / decode_time_sec;
            result.metrics.decode_tokens_per_sec_gpu = 
                (result.metrics.generated_tokens - batch_size) / decode_time_sec_gpu;
        } else {
            result.metrics.decode_tokens_per_sec = 0.0;
            result.metrics.decode_tokens_per_sec_gpu = 0.0;
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
    // Simultaneous Wall-clock and GPU timing. Returns {wall_ms, gpu_ms}
    template <typename F>
    std::pair<double, double> time_execution(F&& func) {
        cudaStream_t s = get_stream();

        // Ensure all previous CUDA work is completely finished before starting
        cudaStreamSynchronize(s);

        cudaEvent_t start_evt, stop_evt;
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);

        auto start_wall = std::chrono::high_resolution_clock::now();
        cudaEventRecord(start_evt, s);
        
        // Execute the function (which submits kernels to the stream)
        func();
        
        cudaEventRecord(stop_evt, s);
        // Block the CPU until the GPU finishes all submitted kernels
        cudaStreamSynchronize(s);
        
        auto end_wall = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> wall_ms = end_wall - start_wall;

        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, start_evt, stop_evt);
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);

        return {wall_ms.count(), static_cast<double>(gpu_ms)};
    }
};

} // namespace pipeline
