#include <iostream>
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
        GPUMemoryArena arena(1024 * 1024);
        std::cout << "[Downstream Sample] Successfully created GPUMemoryArena of size: "
                  << arena.get_total() << " bytes." << std::endl;

        float* d_buf = arena.allocate<float>(64);
        if (d_buf != nullptr) {
            std::cout << "[Downstream Sample] Successfully allocated 64 floats on device from arena." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Downstream Sample] Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[Downstream Sample] PASS: Successfully verified find_package(TransformerEngine REQUIRED)!" << std::endl;
    return 0;
}
