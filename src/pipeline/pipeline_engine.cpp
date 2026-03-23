#include "pipeline/pipeline_engine.hpp"
#include <iostream>
#include <chrono>

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

template <typename F>
double PipelineEngine::time_cuda_execution(F&& func) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    func();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(ms);
}

GenerationResult PipelineEngine::generate(const std::vector<int>& input_ids, const GenerationConfig& config) {
    GenerationResult result;
    result.output_sequence = input_ids;
    result.metrics.prompt_tokens = input_ids.size();
    
    int vocab_size = model.get_vocab_size();
    int max_seq_len = model.get_max_seq_len();
    int max_new_tokens = config.max_new_tokens;

    // Fast fail check
    if (input_ids.size() + max_new_tokens > max_seq_len) {
        std::cerr << "Warning: Generation length exceeds maximum sequence length!" << std::endl;
        max_new_tokens = max_seq_len - input_ids.size();
    }

    // PHASE 1: PREFILL (First token)
    // Even without KV cache, we time the first forward pass separately as PREFILL
    result.metrics.prefill_time_ms = time_cuda_execution([&]() {
        int current_seq_len = result.output_sequence.size();
        inference_arena.reset_to(persistent_offset);
        
        cudaMemcpyAsync(d_input_ids, result.output_sequence.data(), current_seq_len * sizeof(int), cudaMemcpyHostToDevice, stream);
        
        model.forward(d_input_ids, d_logits, 1, current_seq_len, inference_arena, stream);
        
        float* last_logits = d_logits + (current_seq_len - 1) * vocab_size;
        kernels::launch_argmax(last_logits, d_next_token, 1, 1, vocab_size, stream);
    });

    int next_token_id;
    cudaMemcpyAsync(&next_token_id, d_next_token, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    result.output_sequence.push_back(next_token_id);
    result.metrics.generated_tokens++;

    // PHASE 2: DECODE
    result.metrics.decode_time_ms = time_cuda_execution([&]() {
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
    });

    // Populate Metrics
    result.metrics.total_time_ms = result.metrics.prefill_time_ms + result.metrics.decode_time_ms;
    
    // Throughput (assuming prefill processes N prompt tokens, and decode generates M new tokens)
    result.metrics.prefill_tokens_per_sec = (result.metrics.prompt_tokens / (result.metrics.prefill_time_ms / 1000.0));
    
    double decode_time_sec = result.metrics.decode_time_ms / 1000.0;
    // We generated (max_new_tokens - 1) tokens during the decode phase timed block
    if (result.metrics.generated_tokens > 1) {
        result.metrics.decode_tokens_per_sec = ((result.metrics.generated_tokens - 1) / decode_time_sec);
    } else {
        result.metrics.decode_tokens_per_sec = 0.0;
    }

    // Un-tokenize
    // We only extract the newly generated tokens
    std::vector<int> new_tokens(result.output_sequence.begin() + result.metrics.prompt_tokens, result.output_sequence.end());
    result.decoded_text = tokenizer.decode(new_tokens); // Returns the newly generated text
    
    return result;
}

} // namespace pipeline
