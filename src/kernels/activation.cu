#include "kernels.cuh"
#include <cstdio>
#include <cmath>

#define TILE_SIZE 32

namespace kernels {

    // --- Helper: GELU Function ---
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    __device__ __forceinline__ float gelu(float x) {
        const float SQRT_2 = 1.41421356237f;
        return 0.5f * x * (1.0f + erff(x / SQRT_2));
    }

    // --- Kernel 1: Fused Bias Add + GELU ---
    // Grid: [Total_Elements / 256]
    // We treat the data as a flat array, but we need to map indices to bias columns.
    __global__ void bias_gelu_kernel(float* data, const float* bias, int total_elements, int stride_col) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < total_elements) {
            // Determine which column (feature) this thread is processing
            // to pick the correct bias value.
            int col = idx % stride_col;
            
            float val = data[idx] + bias[col];
            data[idx] = gelu(val);
        }
    }

    // --- Kernel 2: Simple Bias Add ---
    // Used for the final Down projection
    __global__ void bias_add_kernel(float* data, const float* bias, int total_elements, int stride_col) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < total_elements) {
            int col = idx % stride_col;
            data[idx] += bias[col];
        }
    }

    // --- Launchers ---

    void launch_bias_gelu(float* data, const float* bias, int rows, int cols, cudaStream_t stream) {
        int total_elements = rows * cols;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        bias_gelu_kernel<<<grid_size, block_size, 0, stream>>>(data, bias, total_elements, cols);
    }

    void launch_bias_add(float* data, const float* bias, int rows, int cols, cudaStream_t stream) {
        int total_elements = rows * cols;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        bias_add_kernel<<<grid_size, block_size, 0, stream>>>(data, bias, total_elements, cols);
    }
}