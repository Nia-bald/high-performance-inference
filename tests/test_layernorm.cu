#include "kernels.cuh"
#include "memory.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            return -1; \
        } \
    } while (0)

int main() {
    const int BATCH_SIZE = 2;
    const int SEQ_LEN = 2;
    const int HIDDEN_DIM = 4; // Small for manual checking

    // Total tokens = 4
    // Token 0: [0, 10, 20, 30] -> Mean=15, Var=125, Std=11.18
    std::vector<float> h_input = {
        0.f, 10.f, 20.f, 30.f,  // Batch 0, Seq 0
        1.f, 2.f, 3.f, 4.f,     // Batch 0, Seq 1
        1.f, 1.f, 1.f, 1.f,     // Batch 1, Seq 0 (Var=0 test)
        -1.f, 1.f, -1.f, 1.f    // Batch 1, Seq 1 (Mean=0 test)
    };

    // Gamma = 1 (Identity), Beta = 0 (No shift)
    std::vector<float> h_gamma(HIDDEN_DIM, 1.0f);
    std::vector<float> h_beta(HIDDEN_DIM, 0.0f);
    std::vector<float> h_output(h_input.size());

    GPUMemoryArena arena(1024 * 1024);
    float* d_input = arena.allocate<float>(h_input.size());
    float* d_output = arena.allocate<float>(h_input.size());
    float* d_gamma = arena.allocate<float>(HIDDEN_DIM);
    float* d_beta = arena.allocate<float>(HIDDEN_DIM);

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Launching LayerNorm..." << std::endl;
    kernels::launch_layer_norm(
        d_input, d_output, d_gamma, d_beta, 
        BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, 
        0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Manual Verification for Token 0 [0, 10, 20, 30]
    // Mean = 15
    // Var = ((225 + 25 + 25 + 225) / 4) = 125
    // Std = sqrt(125) = 11.1803
    // Expected Output[0] = (0 - 15) / 11.1803 = -1.3416
    
    std::cout << "Token 0 Output: ";
    for(int i=0; i<4; i++) std::cout << h_output[i] << " ";
    std::cout << std::endl;

    if (std::abs(h_output[0] - (-1.3416f)) < 1e-3) {
        std::cout << ">>> PASS: Math looks correct!" << std::endl;
    } else {
        std::cout << ">>> FAIL: Check math logic." << std::endl;
        return -1;
    }

    return 0;
}