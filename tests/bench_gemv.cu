#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels.cuh"
#include "memory.h"

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// ============================================================
// GEMV Benchmark: Custom Kernel vs cuBLAS
// ============================================================
// Tests: y[N] = x[K] * W[K, N]  (row-vector × weight matrix)
//
// Compares against both cublasSgemv AND cublasSgemm (M=1)
// to find the true cuBLAS baseline.
// ============================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// CPU reference: y[n] = sum_k x[k] * W[k*N + n]
static void cpu_gemv(const float* x, const float* W, float* y,
                     const float* bias, int N, int K) {
    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += x[k] * W[k * N + n];
        if (bias) sum += bias[n];
        y[n] = sum;
    }
}

// Timing: returns microseconds
template<typename Fn>
static double measure_us(Fn&& fn, int warmup = 10, int runs = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < runs; ++i) fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (ms * 1000.0) / runs;
}

struct BenchResult {
    const char* label;
    int N, K;
    bool has_bias;
    double custom_us;
    double cublas_gemv_us;
    double cublas_gemm_us;   // cublasSgemm with M=1
    double cutlass_gemm_us;  // CUTLASS with M=1
    double custom_gbps;
    double cublas_gemv_gbps;
    double cublas_gemm_gbps;
    double cutlass_gemm_gbps;
    double speedup_vs_gemv;
    double speedup_vs_gemm;
    double speedup_vs_cutlass;
    float max_err_custom;
    float max_err_cublas_gemv;
    float max_err_cublas_gemm;
    float max_err_cutlass;
};

static BenchResult run_benchmark(
    const char* label, int N, int K, bool use_bias,
    cublasHandle_t cublas_handle, GPUMemoryArena& arena
) {
    size_t arena_before = arena.get_used();

    std::vector<float> h_x(K), h_W(K * N), h_bias(N, 0.0f);
    std::vector<float> h_y_cpu(N, 0.0f);
    std::vector<float> h_y_custom(N), h_y_cublas_gemv(N), h_y_cublas_gemm(N);

    srand(42);
    for (auto& v : h_x) v = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
    for (auto& v : h_W) v = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
    if (use_bias)
        for (auto& v : h_bias) v = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.02f;

    float* d_x = arena.allocate<float>(K);
    float* d_W = arena.allocate<float>(K * N);
    float* d_y_custom = arena.allocate<float>(N);
    float* d_y_cublas_gemv = arena.allocate<float>(N);
    float* d_y_cublas_gemm = arena.allocate<float>(N);
    float* d_y_cutlass = arena.allocate<float>(N);
    float* d_bias = use_bias ? arena.allocate<float>(N) : nullptr;

    cudaMemcpy(d_x, h_x.data(), K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    if (use_bias)
        cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cpu_gemv(h_x.data(), h_W.data(), h_y_cpu.data(),
             use_bias ? h_bias.data() : nullptr, N, K);

    // ---- Custom GEMV ----
    double custom_us = measure_us([&]() {
        kernels::launch_gemv(d_x, d_W, d_y_custom, N, K, d_bias, 0);
    });
    cudaMemcpy(h_y_custom.data(), d_y_custom, N * sizeof(float), cudaMemcpyDeviceToHost);
    float max_err_custom = 0.0f;
    for (int i = 0; i < N; ++i)
        max_err_custom = std::max(max_err_custom, std::abs(h_y_custom[i] - h_y_cpu[i]));

    // ---- cuBLAS GEMV ----
    // Row-major W[K,N] → col-major A[N,K], lda=N
    // y = A * x → cublasSgemv(CUBLAS_OP_N, N, K, alpha, W, N, x, 1, beta, y, 1)
    float alpha = 1.0f, beta_val = 0.0f;

    double cublas_gemv_us = measure_us([&]() {
        cublasSgemv(cublas_handle, CUBLAS_OP_N, N, K,
                    &alpha, d_W, N, d_x, 1, &beta_val, d_y_cublas_gemv, 1);
        if (use_bias) {
            float one = 1.0f;
            cublasSaxpy(cublas_handle, N, &one, d_bias, 1, d_y_cublas_gemv, 1);
        }
    });
    cudaMemcpy(h_y_cublas_gemv.data(), d_y_cublas_gemv, N * sizeof(float), cudaMemcpyDeviceToHost);
    float max_err_cublas_gemv = 0.0f;
    for (int i = 0; i < N; ++i)
        max_err_cublas_gemv = std::max(max_err_cublas_gemv, std::abs(h_y_cublas_gemv[i] - h_y_cpu[i]));

    // ---- cuBLAS GEMM (M=1) ----
    // C[1,N] = A[1,K] * B[K,N]
    // In col-major: C^T[N,1] = B^T[N,K] * A^T[K,1]
    // cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K, alpha, B, N, A, K, beta, C, N)
    // where B=d_W (row-major [K,N] = col-major [N,K]), A=d_x (treated as [K,1])
    double cublas_gemm_us = measure_us([&]() {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, 1, K,
                    &alpha,
                    d_W, N,
                    d_x, K,
                    &beta_val,
                    d_y_cublas_gemm, N);
        if (use_bias) {
            float one = 1.0f;
            cublasSaxpy(cublas_handle, N, &one, d_bias, 1, d_y_cublas_gemm, 1);
        }
    });
    cudaMemcpy(h_y_cublas_gemm.data(), d_y_cublas_gemm, N * sizeof(float), cudaMemcpyDeviceToHost);
    float max_err_cublas_gemm = 0.0f;
    for (int i = 0; i < N; ++i)
        max_err_cublas_gemm = std::max(max_err_cublas_gemm, std::abs(h_y_cublas_gemm[i] - h_y_cpu[i]));

    // ---- CUTLASS GEMM (M=1) ----
    using CutlassGemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor
    >;
    CutlassGemm gemm_op;
    std::vector<float> h_y_cutlass(N);
    
    // Copy bias to output first if doing bias (since CUTLASS basic GEMM doesn't do epilogue bias add natively without specific epilogues)
    // To match cuBLAS test, we'll just saxpy it after or initialize d_y_cutlass to 0 and beta=0.
    // Actually, alpha=1, beta=0. 
    CutlassGemm::Arguments args(
        {1, N, K},
        {d_x, K},
        {d_W, N},
        {d_y_cutlass, N},
        {d_y_cutlass, N},
        {alpha, beta_val}
    );
    // Warmup & initialize workspace
    size_t workspace_size = CutlassGemm::get_workspace_size(args);
    void* workspace = arena.allocate<uint8_t>(workspace_size);
    gemm_op.initialize(args, workspace);
    
    double cutlass_gemm_us = measure_us([&]() {
        gemm_op(args, workspace);
        if (use_bias) {
            float one = 1.0f;
            cublasSaxpy(cublas_handle, N, &one, d_bias, 1, d_y_cutlass, 1);
        }
    });
    cudaMemcpy(h_y_cutlass.data(), d_y_cutlass, N * sizeof(float), cudaMemcpyDeviceToHost);
    float max_err_cutlass = 0.0f;
    for (int i = 0; i < N; ++i)
        max_err_cutlass = std::max(max_err_cutlass, std::abs(h_y_cutlass[i] - h_y_cpu[i]));

    // Bandwidth calc
    double bytes = (double)(K * N + K + N) * sizeof(float);
    if (use_bias) bytes += N * sizeof(float);
    double custom_gbps = bytes / (custom_us * 1e3);
    double cublas_gemv_gbps = bytes / (cublas_gemv_us * 1e3);
    double cublas_gemm_gbps = bytes / (cublas_gemm_us * 1e3);
    double cutlass_gemm_gbps = bytes / (cutlass_gemm_us * 1e3);

    // Speedup: compare vs the FASTER of gemv/gemm
    double best_cublas_us = std::min(cublas_gemv_us, cublas_gemm_us);
    double speedup_vs_gemv = cublas_gemv_us / custom_us;
    double speedup_vs_gemm = cublas_gemm_us / custom_us;
    double speedup_vs_cutlass = cutlass_gemm_us / custom_us;

    arena.reset_to(arena_before);

    return {
        label, N, K, use_bias,
        custom_us, cublas_gemv_us, cublas_gemm_us, cutlass_gemm_us,
        custom_gbps, cublas_gemv_gbps, cublas_gemm_gbps, cutlass_gemm_gbps,
        speedup_vs_gemv, speedup_vs_gemm, speedup_vs_cutlass,
        max_err_custom, max_err_cublas_gemv, max_err_cublas_gemm, max_err_cutlass
    };
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        GEMV BENCHMARK: Custom Kernel vs cuBLAS (gemv & gemm M=1)                   ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  GPU: " << std::left << std::setw(77) << prop.name << "║\n";
    std::cout << "║  SM:  " << prop.major << "." << prop.minor
              << "    SMs: " << prop.multiProcessorCount
              << "    VRAM: " << (prop.totalGlobalMem / (1024*1024)) << " MB"
              << std::setw(46) << "" << "║\n";
    std::cout << "║  Peak BW: 112.1 GB/s (theoretical)"
              << std::setw(49) << "" << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════════════════╝\n\n";

    GPUMemoryArena arena(256ULL * 1024 * 1024);
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    struct TestCase { const char* label; int N, K; bool bias; };
    std::vector<TestCase> cases = {
        {"QKV Projection",       2304,  768, true },
        {"Output Projection",     768,  768, true },
        {"FF Up Projection",     3072,  768, true },
        {"FF Down Projection",    768, 3072, true },
        {"QKV (no bias)",        2304,  768, false},
        {"Out (no bias)",         768,  768, false},
        {"FF Up (no bias)",      3072,  768, false},
        {"FF Down (no bias)",     768, 3072, false},
        {"Small (64x64)",          64,   64, false},
        {"Medium (512x512)",      512,  512, false},
        {"Large (4096x4096)",    4096, 4096, false},
        {"Wide (8192x768)",      8192,  768, false},
        {"Tall (768x8192)",       768, 8192, false},
    };

    std::vector<BenchResult> results;
    for (const auto& tc : cases)
        results.push_back(run_benchmark(tc.label, tc.N, tc.K, tc.bias, cublas_handle, arena));

    // Print results
    std::cout << "┌────────────────────────┬───────┬───────┬──────┬──────────┬──────────┬──────────┬──────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐\n";
    std::cout << "│ Test Case              │     N │     K │ Bias │ Cust  μs │ cGEMV μs │ cGEMM μs │ CTLSS μs │ C GB/s │ V GB/s │ M GB/s │ T GB/s │ vs GEV │ vs GEM │ vs CTL │\n";
    std::cout << "├────────────────────────┼───────┼───────┼──────┼──────────┼──────────┼──────────┼──────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤\n";

    for (const auto& r : results) {
        double best_cublas = std::min({r.cublas_gemv_us, r.cublas_gemm_us, r.cutlass_gemm_us});
        const char* ind_v = r.speedup_vs_gemv >= 1.0 ? " ✓" : " ✗";
        const char* ind_m = r.speedup_vs_gemm >= 1.0 ? " ✓" : " ✗";
        const char* ind_t = r.speedup_vs_cutlass >= 1.0 ? " ✓" : " ✗";

        std::cout << "│ " << std::left << std::setw(22) << r.label
                  << " │ " << std::right << std::setw(5) << r.N
                  << " │ " << std::setw(5) << r.K
                  << " │ " << std::setw(4) << (r.has_bias ? "yes" : "no")
                  << " │ " << std::fixed << std::setprecision(1) << std::setw(8) << r.custom_us
                  << " │ " << std::setw(8) << r.cublas_gemv_us
                  << " │ " << std::setw(8) << r.cublas_gemm_us
                  << " │ " << std::setw(8) << r.cutlass_gemm_us
                  << " │ " << std::setprecision(1) << std::setw(6) << r.custom_gbps
                  << " │ " << std::setw(6) << r.cublas_gemv_gbps
                  << " │ " << std::setw(6) << r.cublas_gemm_gbps
                  << " │ " << std::setw(6) << r.cutlass_gemm_gbps
                  << " │ " << std::setprecision(2) << std::setw(4) << r.speedup_vs_gemv << ind_v[0]
                  << " │ " << std::setw(4) << r.speedup_vs_gemm << ind_m[0]
                  << " │ " << std::setw(4) << r.speedup_vs_cutlass << ind_t[0]
                  << " │\n";
    }

    std::cout << "└────────────────────────┴───────┴───────┴──────┴──────────┴──────────┴──────────┴──────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘\n\n";

    // Summary
    int wins_gemv = 0, wins_gemm = 0, wins_cutlass = 0, wins_best = 0;
    for (const auto& r : results) {
        if (r.speedup_vs_gemv >= 1.0) wins_gemv++;
        if (r.speedup_vs_gemm >= 1.0) wins_gemm++;
        if (r.speedup_vs_cutlass >= 1.0) wins_cutlass++;
        double best = std::min({r.cublas_gemv_us, r.cublas_gemm_us, r.cutlass_gemm_us});
        if (best / r.custom_us >= 1.0) wins_best++;
    }

    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  SUMMARY\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  Custom vs cublasSgemv: " << wins_gemv << " / " << results.size() << " wins\n";
    std::cout << "  Custom vs cublasSgemm: " << wins_gemm << " / " << results.size() << " wins\n";
    std::cout << "  Custom vs CUTLASS Gemm: " << wins_cutlass << " / " << results.size() << " wins\n";
    std::cout << "  Custom vs BEST Vendor: " << wins_best << " / " << results.size() << " wins\n";

    std::cout << "\n  GPT-2 Decode Path (with bias):\n";
    for (const auto& r : results) {
        if (r.has_bias) {
            double best = std::min(r.cublas_gemv_us, r.cublas_gemm_us);
            double speedup = best / r.custom_us;
            std::cout << "    " << std::left << std::setw(22) << r.label
                      << " → " << std::fixed << std::setprecision(2) << speedup << "× "
                      << (speedup >= 1.0 ? "FASTER" : "SLOWER")
                      << " vs best cuBLAS (" << std::setprecision(1)
                      << r.custom_us << " vs " << best << " μs)\n";
        }
    }

    bool all_correct = true;
    for (const auto& r : results) {
        if (r.max_err_custom > 1e-2f) {
            std::cout << "  ⚠ FAIL: " << r.label << " (err: " << r.max_err_custom << ")\n";
            all_correct = false;
        }
    }
    if (all_correct)
        std::cout << "\n  ✓ All results verified correct (max error < 0.01)\n";

    std::cout << "═══════════════════════════════════════════════════════\n\n";

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}
