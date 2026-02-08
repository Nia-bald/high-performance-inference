#include "kernels.cuh"
#include "memory.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <limits>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            return -1; \
        } \
    } while (0)

int main() {
    const int ROWS = 4;
    const int COLS = 4;
    const int STRIDE_ROW = 2;
    const int STRIDE_COL = 2;

    // Input matrix: 4x4 with sequential values for easy verification
    // Matrix layout (row-major):
    // [1,  2,  3,  4 ]
    // [5,  6,  7,  8 ]
    // [9,  10, 11, 12]
    // [13, 14, 15, 16]
    std::vector<float> h_input = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f,  10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    const float NEG_INF = -std::numeric_limits<float>::infinity();
    std::vector<float> h_expected = {
            1.0f,  2.0f,  3.0f,  4.0f,
            NEG_INF,  6.0f,  NEG_INF,  8.0f,
            9.0f,  10.0f, 11.0f, 12.0f,
            NEG_INF, 14.0f, NEG_INF, 16.0f
        };

    std::vector<float> h_output(ROWS * COLS);

    GPUMemoryArena arena(1024 * 1024);
    float* d_data = arena.allocate<float>(ROWS * COLS);

    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    std::cout << "Launching Batch Upper Triangulation (4x4, stride_row=2, stride_col=2)..." << std::endl;
    kernels::launch_batch_upper_triangulate(d_data, ROWS, COLS, STRIDE_ROW, STRIDE_COL, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_data, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    int error_count = 0;
    const float EPSILON = 1e-5f;

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int idx = r * COLS + c;
            float got = h_output[idx];
            float expected = h_expected[idx];

            // For -inf values, check if both are -inf
            bool is_neg_inf_expected = std::isinf(expected) && expected < 0;
            bool is_neg_inf_got = std::isinf(got) && got < 0;

            bool match = false;
            if (is_neg_inf_expected && is_neg_inf_got) {
                match = true;
            } else if (!is_neg_inf_expected && !is_neg_inf_got) {
                match = std::abs(got - expected) < EPSILON;
            }

            if (!match) {
                if (error_count < 10) {
                    std::cout.precision(10);
                    std::cout << "Mismatch at [" << r << "," << c << "]: "
                              << "Expected " << expected << ", Got " << got << std::endl;
                }
                error_count++;
            }
        }
    }

    if (error_count == 0) {
        std::cout << ">>> PASS: Batch Upper Triangulation test passed!" << std::endl;
        return 0;
    } else {
        std::cout << ">>> FAIL: Found " << error_count << " errors." << std::endl;
        return -1;
    }
}
