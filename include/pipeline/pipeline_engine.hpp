#pragma once

#include "pipeline/metrics.hpp"
#include "transformer.h"
#include "tokenizer.h"
#include "memory.h"
#include <vector>

namespace pipeline {

class PipelineEngine {
public:
    PipelineEngine(Transformer& model, GPT2Tokenizer& tokenizer, GPUMemoryArena& inference_arena, cudaStream_t stream = 0);

    // Main entry point for generating tokens.
    // Handles memory updates, loops, and accurately times execution.
    GenerationResult generate(const std::vector<int>& input_ids, const GenerationConfig& config);

private:
    Transformer& model;
    GPT2Tokenizer& tokenizer;
    GPUMemoryArena& inference_arena;
    cudaStream_t stream;

    // Persistent buffers
    int* d_input_ids;
    float* d_logits;
    int* d_next_token;
    size_t persistent_offset;

    // Helper to time a lambda execution via CUDA events
    template <typename F>
    double time_cuda_execution(F&& func);
};

} // namespace pipeline
