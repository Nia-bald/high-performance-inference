#include "kernels.cuh"
#include "memory.h"
#include <vector>
#include <chrono>
#include <iostream>


// super naive approach
void cpu_mm(float* A, float* B, float* C, int M, int N, int K){

    for (size_t k{}; k < K; ++k){
        for (size_t i{}; i < M; ++i){
            for (size_t j{}; j < N; ++j){
                C[i*N + j] += A[i*K + k]*B[ k*N + j]; 
            }
        }
    }
}

int main(){
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    std::vector<float> h_A(M*K);
    std::vector<float> h_B(K*N);
    std::vector<float> h_C_cpu(M*N);
    std::vector<float> h_C_gpu(M*N);

    #pragma unroll
    for (int i{}; i < M*K; ++i){
        h_A[i] = static_cast<float>(rand())/RAND_MAX;
        h_B[i] = static_cast<float>(rand())/RAND_MAX;
    }

    GPUMemoryArena arena(10*1024*1024*64);

    float* A_gpu = arena.allocate<float>(M*K);
    float* B_gpu = arena.allocate<float>(K*N);
    float* C_gpu = arena.allocate<float>(M*N);

    cudaMemcpy(A_gpu, h_A.data(), M * K * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, h_B.data(), K * N * sizeof(float),cudaMemcpyHostToDevice);


    auto gpu_start = std::chrono::high_resolution_clock::now();
    std::cout<< "starting GPU multiplication" << std::endl;
    kernels::launch_gemm_tiled(
        A_gpu,  // [M, K]
        B_gpu,  // [K, N]
        C_gpu, // [M, N]
        M, N, K,
        0
    );


    cudaDeviceSynchronize();

    cudaError_t copy_error = cudaMemcpy(h_C_gpu.data(),  C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (copy_error != cudaSuccess){
        printf("CUDA Error in COPY to HOST: %s\n", cudaGetErrorString(copy_error));
    }

    auto gpu_stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> gpu_duration = gpu_stop - gpu_start;

    std::cout<< "GPU multiplication end Total time: "<< gpu_duration.count() <<" ms" << std::endl;

    std::cout<< "starting CPU multiplication" << std::endl;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_mm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU multiplication end Total time: " << cpu_duration.count() << " ms";

    float max_error = 1e-3f;

    for (size_t i{}; i < M*N; ++i){
        if (std::abs(h_C_gpu[i] - h_C_cpu[i]) > max_error){
            std::cout<< "difference too high greater than: " << max_error <<"for i: " << i<< " with values "<< h_C_gpu[i] << " " << h_C_cpu[i] << std::endl;
            return -1;
        }
    }

    std::cout<< "pass value match" << std::endl;

    
    
    return 0;



}