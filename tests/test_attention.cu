#include "layers/attention.h"
#include "memory.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>

/// !important before testing make sure TILESIZE is a factor headsize before testing

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            return -1; \
        } \
    } while (0)

int main() {
    auto test_start = std::chrono::high_resolution_clock::now();

    const int D_MODEL = 4;
    const int NUM_HEADS = 2;
    const int BATCH_SIZE = 1;
    const int SEQ_LEN = 2;

    // Input: orthogonal vectors [[1,0,0,0], [0,1,0,0]]
    // With identity weight matrices, output should equal input
    std::vector<float> h_input(D_MODEL*2);

    for (int i = 0; i < D_MODEL * 2; i ++){
        h_input[i] = i + 1;
    }

    // Identity weight matrices (4x4 each)
    // W_q, W_k, W_v, W_o all set to identity
    std::vector<float> h_W_q(D_MODEL * D_MODEL, 0.0f);
    std::vector<float> h_W_k(D_MODEL * D_MODEL, 0.0f);
    std::vector<float> h_W_v(D_MODEL * D_MODEL, 0.0f);
    std::vector<float> h_W_o(D_MODEL * D_MODEL, 0.0f);

    for (int i = 0; i < D_MODEL * D_MODEL; ++i) {
        h_W_q[i] = i + 1.0f;
        h_W_k[i] = i + 1.0f;
        h_W_v[i] = i + 1.0f;
        h_W_o[i] = i + 1.0f;
    }

    std::vector<float> h_output(BATCH_SIZE * SEQ_LEN * D_MODEL);

    // Setup memory arenas
    GPUMemoryArena weights_arena(1024 * 1024);  // For layer weights
    GPUMemoryArena inference_arena(1024 * 1024); // For inference scratch

    // Construct SelfAttention layer (standard attention: qk_dim=0, v_dim=0)
    SelfAttention attention(D_MODEL, NUM_HEADS, weights_arena, 2, 2);

    // Load identity weight matrices
    attention.load_weights(h_W_q.data(), h_W_k.data(), h_W_v.data(), h_W_o.data());

    // Allocate input and output on GPU
    float* d_input = inference_arena.allocate<float>(h_input.size());
    float* d_output = inference_arena.allocate<float>(h_output.size());

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Launching SelfAttention forward pass..." << std::endl;
    
    // Time the forward pass
    auto start = std::chrono::high_resolution_clock::now();
    attention.forward(BATCH_SIZE, SEQ_LEN, d_input, d_output, inference_arena, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "⏱️  Time taken (forward pass): " << duration.count() << " ms" << std::endl;

    // Copy output back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Expected output (from PyTorch reference)
    std::vector<float> h_expected = {
        3140.f, 3560.f, 3980.f, 4400.f,  // Batch 0, Seq 0
        7268.f, 8232.f, 9196.f, 10160.f  // Batch 0, Seq 1
    };

    const float tolerance = 1e-3f;
    int error_count = 0;
    
    std::cout << "Input:  ";
    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; ++i) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output: ";
    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; ++i) {
        float diff = std::abs(h_output[i] - h_expected[i]);
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i 
                      << " | Expected " << h_expected[i] 
                      << " Got " << h_output[i] 
                      << " (diff: " << diff << ")" << std::endl;
            error_count++;
        }
    }

    auto test_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = test_end - test_start;
    std::cout << "⏱️  Time taken (total test): " << total_duration.count() << " ms" << std::endl;

    if (error_count == 0) {
        std::cout << ">>> PASS: All calculations correct!" << std::endl;
        return 0;
    } else {
        std::cout << ">>> FAIL: Found " << error_count << " errors." << std::endl;
        return -1;
    }
}
