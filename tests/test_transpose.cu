#include "kernels.cuh"
#include "memory.h"
#include <vector>
#include <iostream>
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
    // We use a non-square matrix to catch "swapped rows/cols" bugs
    const int ROWS = 64; 
    const int COLS = 128; 

    // 1. Setup Input
    // Pattern: Value = Row * 1000 + Col (Easy to read: 5002 is Row 5, Col 2)
    std::vector<float> h_input(ROWS * COLS);
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            h_input[r * COLS + c] = (float)(r * 1000 + c);
        }
    }

    std::vector<float> h_output(ROWS * COLS); // Same size, just swapped shape interpretation

    GPUMemoryArena arena(1024 * 1024);
    float* d_input = arena.allocate<float>(ROWS * COLS);
    float* d_output = arena.allocate<float>(ROWS * COLS);

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 2. Launch
    std::cout << "Launching Transpose (64x128 -> 128x64)..." << std::endl;
    kernels::launch_transpose(d_input, d_output, ROWS, COLS, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Verify
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    int error_count = 0;
    for (int c = 0; c < COLS; ++c) {     // Iterate Output Rows (which are Input Cols)
        for (int r = 0; r < ROWS; ++r) { // Iterate Output Cols (which are Input Rows)
            
            // In Output Memory (128 x 64):
            // We are at Row 'c', Column 'r'
            float val = h_output[c * ROWS + r];

            // Expected Value from Input (Row 'r', Col 'c'):
            float expected = (float)(r * 1000 + c);

            if (val != expected) {
                if (error_count < 5) {
                    std::cout << "Mismatch at Out[" << c << "," << r << "]: "
                              << "Expected " << expected << " Got " << val << std::endl;
                }
                error_count++;
            }
        }
    }

    if (error_count == 0) {
        std::cout << ">>> PASS: Matrix Transposed Correctly!" << std::endl;
    } else {
        std::cout << ">>> FAIL: Found " << error_count << " errors." << std::endl;
        return -1;
    }

    return 0;
}