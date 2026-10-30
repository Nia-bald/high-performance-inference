#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "memory.h"
#include "kernels.cuh"

int main() {
    std::cout << "[Downstream Sample] Testing TransformerEngine import..." << std::endl;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "[Downstream Sample] No CUDA device available or driver not running; skipping GPU runtime execution." << std::endl;
        std::cout << "[Downstream Sample] PASS: Headers and linked symbols resolved successfully!" << std::endl;
        return 0;
    }

    try {
        constexpr int N = 64;
        GPUMemoryArena arena(1024 * 1024);
        std::cout << "[Downstream Sample] Successfully created GPUMemoryArena of size: "
                  << arena.get_total() << " bytes." << std::endl;

        float* d_buf = arena.allocate<float>(N);
        if (d_buf == nullptr) {
            std::cerr << "[Downstream Sample] Failed to allocate device memory from arena." << std::endl;
            return 1;
        }

        std::vector<float> h_input(N, 2.5f);
        cudaMemcpy(d_buf, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        // Execute exported CUDA kernel on device
        const float scale = 3.0f;
        kernels::launch_scale(d_buf, scale, N, 0);
        cudaDeviceSynchronize();

        std::vector<float> h_output(N, 0.0f);
        cudaMemcpy(h_output.data(), d_buf, N * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; ++i) {
            if (std::abs(h_output[i] - 7.5f) > 1e-4) {
                std::cerr << "[Downstream Sample] Verification failed at index " << i
                          << ": expected 7.5, got " << h_output[i] << std::endl;
                return 1;
            }
        }
        std::cout << "[Downstream Sample] Kernel execution verified on local GPU! (Output = 7.5)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Downstream Sample] Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[Downstream Sample] PASS: Successfully verified find_package(TransformerEngine REQUIRED)!" << std::endl;
    return 0;
}

