#include "pipeline/pipeline_engine.hpp"
#include <iostream>

namespace pipeline {

PipelineEngine::PipelineEngine(Transformer& model, GPT2Tokenizer& tokenizer, GPUMemoryArena& inference_arena, cudaStream_t stream)
    : model(model), tokenizer(tokenizer), inference_arena(inference_arena), stream(stream) {
    
    int vocab_size = model.get_vocab_size();
    int max_seq_len = model.get_max_seq_len();

    // Allocate persistent buffers (only once)
    d_input_ids = inference_arena.allocate<int>(max_seq_len);
    d_logits = inference_arena.allocate<float>(max_seq_len * vocab_size);
    d_next_token = inference_arena.allocate<int>(1);

    persistent_offset = inference_arena.get_user();
}

void PipelineEngine::run_prefill(GenerationResult& result, const GenerationConfig& config) {
    int vocab_size = model.get_vocab_size();
    int max_seq_len = model.get_max_seq_len();
    int current_seq_len = result.output_sequence.size();

    inference_arena.reset_to(persistent_offset);

    cudaMemcpyAsync(d_input_ids, result.output_sequence.data(), current_seq_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    model.forward(d_input_ids, d_logits, 1, current_seq_len, inference_arena, stream);

    float* last_logits = d_logits + (current_seq_len - 1) * vocab_size;
    kernels::launch_argmax(last_logits, d_next_token, 1, 1, vocab_size, stream);

    int next_token_id;
    cudaMemcpyAsync(&next_token_id, d_next_token, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    result.output_sequence.push_back(next_token_id);
    result.metrics.generated_tokens++;
}

void PipelineEngine::run_decode(GenerationResult& result, const GenerationConfig& config) {
    int vocab_size = model.get_vocab_size();
    int max_seq_len = model.get_max_seq_len();
    int max_new_tokens = config.max_new_tokens;

    // Cap generation length
    if (result.output_sequence.size() + (max_new_tokens - 1) > max_seq_len) {
        std::cerr << "Warning: Generation length exceeds maximum sequence length!" << std::endl;
        max_new_tokens = max_seq_len - result.output_sequence.size() + 1;
    }

    for (int step = 1; step < max_new_tokens; ++step) {
        int current_seq_len = result.output_sequence.size();

        // Re-use scratch memory for each token
        inference_arena.reset_to(persistent_offset);

        cudaMemcpyAsync(d_input_ids, result.output_sequence.data(), current_seq_len * sizeof(int), cudaMemcpyHostToDevice, stream);

        model.forward(d_input_ids, d_logits, 1, current_seq_len, inference_arena, stream);

        float* last_logits = d_logits + (current_seq_len - 1) * vocab_size;
        kernels::launch_argmax(last_logits, d_next_token, 1, 1, vocab_size, stream);

        int token;
        cudaMemcpyAsync(&token, d_next_token, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        result.output_sequence.push_back(token);
        result.metrics.generated_tokens++;
    }
}

void PipelineEngine::finalize(GenerationResult& result) {
    // Decode only the newly generated tokens
    std::vector<int> new_tokens(result.output_sequence.begin() + result.metrics.prompt_tokens, result.output_sequence.end());
    result.decoded_text = tokenizer.decode(new_tokens);
}

} // namespace pipeline
