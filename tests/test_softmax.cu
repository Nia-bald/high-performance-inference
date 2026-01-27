#include "kernels.cuh"
#include "memory.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            return -1; \
        } \
    } while (0)

int main() {
    const int ROWS = 2; // e.g., Batch=1, Seq=2
    const int COLS = 10; // Hidden Dim

    // Row 0: Random numbers
    // Row 1: Large numbers to test stability (e.g., 1000)
    std::vector<float> h_input = {
        // Row 0: Simple
        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f,
        // Row 1: Stability Test (If you didn't subtract Max, this would explode)
        1000.0f, 1001.0f, 1002.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f, 1000.0f
    };
    
    // We copy h_input to h_output later, since softmax is in-place
    std::vector<float> h_output(ROWS * COLS);

    GPUMemoryArena arena(1024 * 1024);
    float* d_input = arena.allocate<float>(ROWS * COLS);

    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Launching Softmax..." << std::endl;
    kernels::launch_softmax(d_input, ROWS, COLS, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_input, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verification
    for (int r = 0; r < ROWS; ++r) {
        float row_sum = 0.0f;
        std::cout << "Row " << r << ": ";
        for (int c = 0; c < COLS; ++c) {
            float val = h_output[r * COLS + c];
            row_sum += val;
            std::cout << val << " ";
        }
        std::cout << "| Sum = " << row_sum << std::endl;

        if (std::abs(row_sum - 1.0f) > 1e-3) {
            std::cout << ">>> FAIL: Row " << r << " does not sum to 1.0" << std::endl;
            return -1;
        }
    }

    // Specific check for Row 1 (Numerical Stability)
    // 1000, 1001, 1002 -> Softmax(0, 1, 2) roughly
    // e^2 is much bigger than e^0. The argmax (index 2) should be largest.
    if (h_output[1 * COLS + 2] > h_output[1 * COLS + 1]) {
        std::cout << ">>> PASS: Numerical Stability check passed." << std::endl;
    } else {
        std::cout << ">>> FAIL: Max value logic seems wrong." << std::endl;
        return -1;
    }

    return 0;
}