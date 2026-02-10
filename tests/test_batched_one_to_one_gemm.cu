#include "kernels.cuh"
#include "memory.h"
#include <vector>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

std::vector<float> cpu_batched_one_to_one_gemm_implementation(std::vector<float> A, std::vector<float> B, int M, int K, int N, int stride_A, int stride_B, int stride_K){
    int total_rows = M;
    int total_cols = stride_B*(K/stride_K);
    std::vector<float> C(total_rows*total_cols);

    for (int row = 0; row < total_rows; row += 1){
        for (int col = 0; col < total_cols; col += 1){
            int B_i = row/stride_A;
            int B_j = col/stride_B;
            float sum = 0.0f;
            for (int k = 0; k < stride_K; k += 1){
                sum += A[row*K + B_j*stride_K + k]*B[(B_i*stride_K + k)*N + col];
            }
            C[row*total_cols + col] = sum;
        }
    }
    return C;
}


int main() {
    // 1. Setup Architecture
    // We use small dimensions to verify quadrants easily
    // M=4, N=4, K=4 (Split point is 2)
    int M = 1024, N = 1024, K = 1024;
    
    std::cout << ">>> [TEST] Initializing GPU Arena..." << std::endl;
    GPUMemoryArena arena(10*1024*1024*64); // 1MB is plenty

    // // 2. Data Preparation
    size_t size = M * K * sizeof(float); // Since M=N=K, all same size
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    // Fill A: 
    // Top-Left (A11) = 1, Top-Right (A12) = 2
    // Bot-Left (A21) = 3, Bot-Right (A22) = 4
    #pragma unroll
    for (int i{}; i < M*K; ++i){
        h_A[i] = static_cast<float>(rand())/RAND_MAX;
        h_B[i] = static_cast<float>(rand())/RAND_MAX;
    }

    std::vector<float> h_C_test = {11.0f, 14.0f, 37.0f, 44.0f, 35.0f, 46.0f, 77.0f, 92.0f, 211.0f, 230.0f, 301.0f, 324.0f, 299.0f, 326.0f, 405.0f, 436.0f};

    // 3. GPU Allocations (Using Arena)
    float* d_A = arena.allocate<float>(M * K);
    float* d_B = arena.allocate<float>(K * N);
    float* d_C = arena.allocate<float>(M * N);

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // 4. Run Kernel
    std::cout << ">>> [TEST] Launching Masked GEMM..." << std::endl;
    kernels::launch_batched_one_to_one_gemm(d_A, d_B, d_C, M, N, K, 16, 16, 16);
    std::vector<float> h_C_CPU = cpu_batched_one_to_one_gemm_implementation(h_A, h_B, M, N, K, 16, 16, 16);

    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Verify Results
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    std::cout << "\n--- RESULTS ANALYSIS ---" << M*K << std::endl;
    float max_error = 1e-2f;

    for (int i = 0; i < M*K; i += 1){
        if (std::abs(h_C_CPU[i] - h_C[i]) > max_error){
            std::cout.precision(20);
            std::cout << ">>> FAIL: Calculation error. for i = " <<i<< std::endl;
            std::cout << ">>>   Calculation error for index i = " << i
                      << ": expected " << h_C_CPU[i]
                      << ", got " <<  h_C[i] << std::endl;

            return 0;
        }
    }
    std::cout << ">>> SUCCESS: test cases pased" << std::endl;

    return 0;
}