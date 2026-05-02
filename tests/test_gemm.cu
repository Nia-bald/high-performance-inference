#include "kernels.cuh"
#include "memory.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

void cpu_mm(const float* A, const float* B, float* C, int M, int N, int K){
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i*N + j] += A[i*K + k] * B[k*N + j];
}

template<typename Fn>
static double measure_gpu_ms(Fn&& fn, int runs = 50) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    fn(); // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / runs;
}

void run_case(const char* label, int M, int N, int K, GPUMemoryArena& arena) {
    std::vector<float> h_A(M*K), h_B(K*N), h_C_cpu(M*N, 0.f), h_C_gpu(M*N);
    for (auto& v : h_A) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : h_B) v = static_cast<float>(rand()) / RAND_MAX;

    float* d_A = arena.allocate<float>(M*K);
    float* d_B = arena.allocate<float>(K*N);
    float* d_C = arena.allocate<float>(M*N);
    cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);

    double ms = measure_gpu_ms([&](){
        kernels::launch_gemm_tiled(d_A, d_B, d_C, M, N, K, 0);
    });

    // Validate
    cpu_mm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
    cudaMemcpy(h_C_gpu.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    float max_err = 0.f;
    for (int i = 0; i < M*N; ++i)
        max_err = std::max(max_err, std::abs(h_C_gpu[i] - h_C_cpu[i]));

    // GFLOP/s = (2 * M * N * K) / (ms * 1e-3) / 1e9
    double gflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e9;

    std::cout << std::left << std::setw(28) << label
              << " M=" << std::setw(5) << M
              << " N=" << std::setw(5) << N
              << " K=" << std::setw(5) << K
              << " | " << std::fixed << std::setprecision(4) << ms << " ms"
              << " | " << std::setprecision(2) << gflops << " GFLOP/s"
              << (max_err > 1e-2f ? "  !! MISMATCH" : "  OK") << "\n";
}

int main(){
    // 512 MB arena — enough for all test matrices
    GPUMemoryArena arena(512ULL * 1024 * 1024);

    std::cout << "=============================================================\n";
    std::cout << "  GEMM Benchmark — Real Engine Shapes (GTX 1050 Ti)\n";
    std::cout << "=============================================================\n\n";

    // ---- Prefill phase (prompt_len=79, batch=1) ----
    std::cout << "--- PREFILL phase (M = prompt_len = 79) ---\n";
    run_case("FF Up Projection",   79,  3072,  768, arena);
    run_case("FF Down Projection", 79,   768, 3072, arena);
    run_case("QKV Projection",     79,   768,  768, arena);
    run_case("Out Projection",     79,   768,  768, arena);

    // ---- Decode phase: first step (M = prompt_len + 1) ----
    std::cout << "\n--- DECODE first step (M = 80) ---\n";
    run_case("FF Up Projection",   80,  3072,  768, arena);
    run_case("FF Down Projection", 80,   768, 3072, arena);

    // ---- Decode phase: worst case (M = 79 + 50 = 129) ----
    std::cout << "\n--- DECODE worst case (M = 129, 50 new tokens generated) ---\n";
    run_case("FF Up Projection",  129,  3072,  768, arena);
    run_case("FF Down Projection",129,   768, 3072, arena);
    run_case("QKV Projection",    129,   768,  768, arena);
    run_case("Out Projection",    129,   768,  768, arena);

    // ---- Baseline: Square matrix (original test_gemm case) ----
    std::cout << "\n--- BASELINE: Square matrix (original test) ---\n";
    run_case("Square 1024x1024", 1024, 1024, 1024, arena);

    return 0;
}