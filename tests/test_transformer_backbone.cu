#include "layers/transformer_backbone.h"
#include "memory.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <cassert>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            return -1; \
        } \
    } while (0)

// Helper to fill host vector with random floats in [-1.0, 1.0]
static void fill_random(std::vector<float>& vec, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : vec) {
        v = dist(gen);
    }
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  TransformerBackbone Unit & Regression Test Suite" << std::endl;
    std::cout << "==================================================" << std::endl;

    constexpr int D_MODEL = 64;
    constexpr int NUM_HEADS = 2;
    constexpr int NUM_LAYERS = 2;
    constexpr int D_FF = 256;
    constexpr int BATCH_SIZE = 2;
    constexpr int SEQ_LEN = 4;
    constexpr size_t TOTAL_ELEMENTS = BATCH_SIZE * SEQ_LEN * D_MODEL;

    // 1. Verify Static Estimators
    size_t expected_weight_mem = TransformerBackbone::estimate_weight_memory(D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF);
    size_t expected_scratch = TransformerBackbone::estimate_inference_scratch(BATCH_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF);
    std::cout << "[Test] Estimated Weight Memory: " << expected_weight_mem / 1024.0 << " KB" << std::endl;
    std::cout << "[Test] Estimated Inference Scratch: " << expected_scratch / 1024.0 << " KB" << std::endl;
    assert(expected_weight_mem > 0);
    assert(expected_scratch > 0);

    // 2. Setup Arenas
    GPUMemoryArena weights_arena(1024 * 1024 * 16);    // 16 MB
    GPUMemoryArena inference_arena(1024 * 1024 * 16);  // 16 MB

    // 3. Construct TransformerBackbone directly
    std::cout << "[Test] Initializing TransformerBackbone directly..." << std::endl;
    TransformerBackbone backbone(D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, weights_arena);

    // Verify getters
    assert(backbone.get_d_model() == D_MODEL);
    assert(backbone.get_num_heads() == NUM_HEADS);
    assert(backbone.get_num_layers() == NUM_LAYERS);
    assert(backbone.get_d_ff() == D_FF);

    // 4. Initialize random weights for each block
    for (int i = 0; i < NUM_LAYERS; ++i) {
        TransformerBlock* block = backbone.get_block(i);
        assert(block != nullptr);

        std::vector<float> gamma(D_MODEL, 1.0f);
        std::vector<float> beta(D_MODEL, 0.0f);
        block->get_attn_norm().load_weights(gamma.data(), beta.data());
        block->get_ffn_norm().load_weights(gamma.data(), beta.data());

        // Attention weights
        std::vector<float> W_q(D_MODEL * D_MODEL), W_k(D_MODEL * D_MODEL), W_v(D_MODEL * D_MODEL), W_o(D_MODEL * D_MODEL);
        std::vector<float> b_q(D_MODEL), b_k(D_MODEL), b_v(D_MODEL), b_o(D_MODEL);
        fill_random(W_q, 10 + i * 8);
        fill_random(W_k, 11 + i * 8);
        fill_random(W_v, 12 + i * 8);
        fill_random(W_o, 13 + i * 8);
        fill_random(b_q, 14 + i * 8);
        fill_random(b_k, 15 + i * 8);
        fill_random(b_v, 16 + i * 8);
        fill_random(b_o, 17 + i * 8);

        block->get_attention().load_weights(
            W_q.data(), W_k.data(), W_v.data(), W_o.data(),
            b_q.data(), b_k.data(), b_v.data(), b_o.data()
        );

        // FFN weights
        std::vector<float> W_up(D_MODEL * D_FF), W_down(D_FF * D_MODEL);
        std::vector<float> b_up(D_FF), b_down(D_MODEL);
        fill_random(W_up, 18 + i * 8);
        fill_random(W_down, 19 + i * 8);
        fill_random(b_up, 20 + i * 8);
        fill_random(b_down, 21 + i * 8);

        block->get_ffn().load_weights(W_up.data(), b_up.data(), W_down.data(), b_down.data());
    }

    // Load final LayerNorm weights
    std::vector<float> final_gamma(D_MODEL, 1.0f);
    std::vector<float> final_beta(D_MODEL, 0.0f);
    backbone.get_final_norm().load_weights(final_gamma.data(), final_beta.data());

    // 5. Create continuous float input tensor [B, S, D]
    std::vector<float> h_input(TOTAL_ELEMENTS);
    fill_random(h_input, 42);

    float *d_input = nullptr, *d_output = nullptr, *d_inplace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_inplace, TOTAL_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inplace, h_input.data(), TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));

    // 6. Test Out-of-Place Forward Pass
    std::cout << "[Test] Executing out-of-place backbone.forward()..." << std::endl;
    backbone.forward(d_input, d_output, BATCH_SIZE, SEQ_LEN, inference_arena, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out_outofplace(TOTAL_ELEMENTS);
    CUDA_CHECK(cudaMemcpy(h_out_outofplace.data(), d_output, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify output is not NaN / Inf
    for (size_t i = 0; i < TOTAL_ELEMENTS; ++i) {
        if (std::isnan(h_out_outofplace[i]) || std::isinf(h_out_outofplace[i])) {
            std::cerr << "FAIL: NaN or Inf found in out-of-place forward output at index " << i << std::endl;
            return -1;
        }
    }

    // 7. Test In-Place Forward Pass
    std::cout << "[Test] Executing in-place backbone.forward(d_inplace, d_inplace)..." << std::endl;
    inference_arena.reset();
    backbone.forward(d_inplace, d_inplace, BATCH_SIZE, SEQ_LEN, inference_arena, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out_inplace(TOTAL_ELEMENTS);
    CUDA_CHECK(cudaMemcpy(h_out_inplace.data(), d_inplace, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify in-place and out-of-place results match bit-identically
    float max_diff = 0.0f;
    for (size_t i = 0; i < TOTAL_ELEMENTS; ++i) {
        float diff = std::abs(h_out_outofplace[i] - h_out_inplace[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    std::cout << "[Test] In-place vs Out-of-place max diff: " << max_diff << std::endl;
    if (max_diff > 1e-5f) {
        std::cerr << "FAIL: In-place output did not match out-of-place output!" << std::endl;
        return -1;
    }

    // 8. Ground-Truth Step-by-Step Block Manual Forward Verification
    std::cout << "[Test] Verifying step-by-step block composition parity..." << std::endl;
    float *d_step_buf1 = nullptr, *d_step_buf2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_step_buf1, TOTAL_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_step_buf2, TOTAL_ELEMENTS * sizeof(float)));

    // Copy initial input
    CUDA_CHECK(cudaMemcpy(d_step_buf1, h_input.data(), TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));

    inference_arena.reset();
    // Block 0: reads d_step_buf1, writes d_step_buf2
    backbone.get_block(0)->forward(d_step_buf1, d_step_buf2, BATCH_SIZE, SEQ_LEN, &inference_arena, 0);
    // Block 1: reads d_step_buf2, writes d_step_buf1
    backbone.get_block(1)->forward(d_step_buf2, d_step_buf1, BATCH_SIZE, SEQ_LEN, &inference_arena, 0);
    // Final Norm: reads d_step_buf1, writes d_step_buf2
    backbone.get_final_norm().forward(d_step_buf1, d_step_buf2, BATCH_SIZE, SEQ_LEN, nullptr, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_manual(TOTAL_ELEMENTS);
    CUDA_CHECK(cudaMemcpy(h_manual.data(), d_step_buf2, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    float manual_max_diff = 0.0f;
    for (size_t i = 0; i < TOTAL_ELEMENTS; ++i) {
        float diff = std::abs(h_out_outofplace[i] - h_manual[i]);
        if (diff > manual_max_diff) {
            manual_max_diff = diff;
        }
    }
    std::cout << "[Test] Manual step-by-step vs backbone.forward() max diff: " << manual_max_diff << std::endl;
    if (manual_max_diff > 1e-5f) {
        std::cerr << "FAIL: Manual composition did not match backbone forward pass!" << std::endl;
        return -1;
    }

    // 9. Test Single-Layer Edge Case (num_layers = 1)
    std::cout << "[Test] Testing single-layer (num_layers=1) edge case..." << std::endl;
    {
        GPUMemoryArena single_w_arena(1024 * 1024 * 8);
        TransformerBackbone single_backbone(D_MODEL, NUM_HEADS, 1, D_FF, single_w_arena);
        single_backbone.get_final_norm().load_weights(final_gamma.data(), final_beta.data());
        
        inference_arena.reset();
        single_backbone.forward(d_input, d_output, BATCH_SIZE, SEQ_LEN, inference_arena, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 10. Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_inplace));
    CUDA_CHECK(cudaFree(d_step_buf1));
    CUDA_CHECK(cudaFree(d_step_buf2));

    std::cout << ">>> PASS: TransformerBackbone passed all validation tests!" << std::endl;
    return 0;
}
