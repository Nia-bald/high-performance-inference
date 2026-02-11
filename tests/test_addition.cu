#include "kernels.cuh"
#include "memory.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << std::endl;              \
            return -1;                                                       \
        }                                                                    \
    } while (0)

static void cpu_addition(const float* A, const float* B, float* C, int length) {
    for (int i = 0; i < length; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Keep this reasonably large so timing is visible but not too slow on CPU.
    const int length = 1 << 24; // ~16M floats

    std::vector<float> h_A(length);
    std::vector<float> h_B(length);
    std::vector<float> h_C_cpu(length);
    std::vector<float> h_C_gpu(length);

    for (int i = 0; i < length; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    std::cout << "Starting CPU addition..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_addition(h_A.data(), h_B.data(), h_C_cpu.data(), length);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU addition end. Total time: " << cpu_duration.count() << " ms" << std::endl;

    // GPU path (wall-time measurement around launch + synchronize, like other repo tests)
    GPUMemoryArena arena(static_cast<size_t>(length) * sizeof(float) * 3 + 1024);
    float* d_A = arena.allocate<float>(length);
    float* d_B = arena.allocate<float>(length);
    float* d_C = arena.allocate<float>(length);

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), static_cast<size_t>(length) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), static_cast<size_t>(length) * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Starting GPU addition..." << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    kernels::launch_addition(d_A, d_B, d_C, length, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, static_cast<size_t>(length) * sizeof(float), cudaMemcpyDeviceToHost));

    std::chrono::duration<double, std::milli> gpu_duration = gpu_end - gpu_start;
    std::cout << "GPU addition end. Total time: " << gpu_duration.count() << " ms" << std::endl;

    // Verification
    const float max_error = 1e-6f;
    float worst = 0.0f;
    int worst_i = -1;
    for (int i = 0; i < length; ++i) {
        float diff = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > worst) {
            worst = diff;
            worst_i = i;
        }
        if (diff > max_error) {
            std::cout << "FAIL: diff " << diff << " > " << max_error
                      << " at i=" << i
                      << " gpu=" << h_C_gpu[i]
                      << " cpu=" << h_C_cpu[i]
                      << std::endl;
            return -1;
        }
    }

    std::cout << "PASS: value match. Worst diff=" << worst << " at i=" << worst_i << std::endl;
    return 0;
}

