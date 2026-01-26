#include "kernels.cuh"
#include "memory.h"
#include <vector>
#include <iostream>
#include <cassert>

// Macro for checking CUDA errors
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
    const int SEQ_LEN = 4;
    const int HIDDEN_DIM = 8; // Small dim for easy debugging
    const int VOCAB_SIZE = 10;

    // 1. Setup Input Data (CPU)
    // Token IDs: [2, 4] vectors are chosen
    std::vector<int> h_tokens = { 
        1, 5, 0, 9,   // Batch 0
        2, 2, 8, 1    // Batch 1
    };

    // Embedding Table: Simple pattern to verify correctness
    // Row 0 fills with 0.0, Row 1 fills with 1.0, etc.
    std::vector<float> h_table(VOCAB_SIZE * HIDDEN_DIM);
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            h_table[i * HIDDEN_DIM + j] = (float)i; 
        }
    }

    std::vector<float> h_output(BATCH_SIZE * SEQ_LEN * HIDDEN_DIM);

    // 2. Allocate GPU Memory
    GPUMemoryArena arena(1024 * 1024); // 1MB is plenty
    int* d_tokens = arena.allocate<int>(BATCH_SIZE * SEQ_LEN);
    float* d_table = arena.allocate<float>(VOCAB_SIZE * HIDDEN_DIM);
    float* d_output = arena.allocate<float>(BATCH_SIZE * SEQ_LEN * HIDDEN_DIM);

    // 3. Copy Input to GPU
    CUDA_CHECK(cudaMemcpy(d_tokens, h_tokens.data(), h_tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), h_table.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Launch Kernel
    std::cout << "Launching Embedding Kernel..." << std::endl;
    kernels::launch_embedding_lookup(
        d_tokens, d_table, d_output, 
        BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, 
        0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Copy Result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. Verify Correctness
    // We expect: Output[Batch 0, Token 0] (which is ID 1) -> Should be all 1.0s
    //            Output[Batch 0, Token 1] (which is ID 5) -> Should be all 5.0s
    int error_count = 0;
    for (int i = 0; i < BATCH_SIZE * SEQ_LEN; ++i) {
        int token_id = h_tokens[i];
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float val = h_output[i * HIDDEN_DIM + j];
            if (val != (float)token_id) {
                std::cout << "Mismatch at Token Index " << i << " Dim " << j 
                          << " | Expected " << token_id << " Got " << val << std::endl;
                error_count++;
            }
        }
    }

    if (error_count == 0) {
        std::cout << ">>> PASS: All embeddings retrieved correctly!" << std::endl;
    } else {
        std::cout << ">>> FAIL: Found " << error_count << " errors." << std::endl;
        return -1;
    }

    return 0;
}