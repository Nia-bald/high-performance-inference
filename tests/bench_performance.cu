#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <numeric>
#include <cuda_runtime.h>

#include "transformer.h"
#include "tokenizer.h"
#include "memory.h"
#include "kernels.cuh"
#include "pipeline/pipeline_engine.hpp"
#include "batch_executor_orchestrator.hpp"
#include "benchmark_config.hpp"
namespace fs = std::filesystem;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

struct BenchmarkConfig {
    int batch_size;
    int seq_len;
    int max_new_tokens;
};

struct PromptConfig {
    std::string filename;
    std::string prompt_text;
    std::vector<int> input_ids;
    int max_new_tokens;
};

// ============================================================================
// Timing utilities
// ============================================================================

static double benchmarkKernel(cudaStream_t stream, auto &&kernel_fn) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    kernel_fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Measure (average of 10 runs for stability)
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < 10; ++i) {
        kernel_fn();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / 10.0;
}

static std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&time, &tm_buf);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
    return std::string(buf);
}

// ============================================================================
// BENCHMARK 1: Individual Kernels
// ============================================================================

struct KernelBenchResult {
    int batch_size;
    int seq_len;
    double embedding_ms;
    double qkv_proj_ms;
    double attn_qk_ms;
    double softmax_ms;
    double layernorm_ms;
};

static std::vector<KernelBenchResult> benchmarkKernels(const std::vector<BenchmarkConfig>& configs, int d_model, int num_heads, int d_ff) {
    std::cout << "\n========================================\n";
    std::cout << "  BENCHMARK 1: Individual Kernel Timing\n";
    std::cout << "========================================\n\n";

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<KernelBenchResult> results;

    for (const auto& cfg : configs) {
        int B = cfg.batch_size;
        int S = cfg.seq_len;
        
        int head_dim = d_model / num_heads;
        
        float *d_dummy_in, *d_dummy_out, *d_dummy_weights;
        int *d_token_ids;
        size_t in_size = B * S * d_model;
        size_t out_size = std::max((size_t)B * S * std::max(d_model, d_ff), (size_t)B * num_heads * S * S);
        size_t weights_size = d_model * std::max(d_model, d_ff);
        
        CUDA_CHECK(cudaMalloc(&d_dummy_in, in_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dummy_out, out_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dummy_weights, weights_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_token_ids, B * S * sizeof(int)));

        CUDA_CHECK(cudaMemset(d_dummy_in, 0, in_size * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_token_ids, 0, B * S * sizeof(int), stream));

        KernelBenchResult r;
        r.batch_size = B;
        r.seq_len = S;

        r.embedding_ms = benchmarkKernel(stream, [&]() {
            kernels::launch_embedding_lookup(d_token_ids, d_dummy_weights, d_dummy_weights, d_dummy_out, B, S, d_model, stream);
        });

        r.qkv_proj_ms = benchmarkKernel(stream, [&]() {
            kernels::launch_gemm_tiled(d_dummy_in, d_dummy_weights, d_dummy_out, B * S, d_model, d_model, stream);
        });

        float* d_dummy_Q = d_dummy_in;
        float* d_dummy_K_T = d_dummy_in;
        float* d_dummy_attn = d_dummy_out; // size B*H*S*S but dummy out is large enough for small S

        r.attn_qk_ms = benchmarkKernel(stream, [&]() {
            kernels::launch_batched_gemm_naive(d_dummy_Q, d_dummy_K_T, d_dummy_attn, B * S, B * S, d_model, S, S, head_dim, stream);
        });

        r.softmax_ms = benchmarkKernel(stream, [&]() {
            kernels::launch_softmax(d_dummy_attn, num_heads * B * S, S, stream);
        });

        r.layernorm_ms = benchmarkKernel(stream, [&]() {
            kernels::launch_layer_norm(d_dummy_in, d_dummy_out, d_dummy_weights, d_dummy_weights, B, S, d_model, stream);
        });

        std::cout << "  B=" << B << ", S=" << S 
                  << " | Emb: " << std::fixed << std::setprecision(3) << r.embedding_ms 
                  << " ms | Proj: " << r.qkv_proj_ms 
                  << " ms | Attn_QK: " << r.attn_qk_ms 
                  << " ms | Softmax: " << r.softmax_ms 
                  << " ms | LN: " << r.layernorm_ms << " ms\n";

        results.push_back(r);

        CUDA_CHECK(cudaFree(d_dummy_in));
        CUDA_CHECK(cudaFree(d_dummy_out));
        CUDA_CHECK(cudaFree(d_dummy_weights));
        CUDA_CHECK(cudaFree(d_token_ids));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return results;
}

// ============================================================================
// BENCHMARK 2: Full Pipeline Generation
// ============================================================================

struct PipelineBenchResult {
    int batch_size;
    int seq_len;
    int max_new_tokens;
    double prefill_ms;
    double decode_ms;
    double total_ms;
    double prefill_tok_sec;
    double decode_tok_sec;
};

// Load real weights definition
void load_gpt2_weights(Transformer& gpt, const std::string& path, int n_layers, int d_model, int vocab_size, int max_seq, int d_ff);

static std::vector<PipelineBenchResult> benchmarkPipeline(
    const std::vector<PromptConfig>& prompts, 
    const benchmark::ScenarioConfig& scenario,
    const std::string& output_dir, 
    int vocab_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff) {
    
    std::cout << "\n========================================\n";
    std::cout << "  BENCHMARK 2: Full Generation Pipeline \n";
    std::cout << "  Scenario: batch_size=" << scenario.batch_size 
              << ", max_new_tokens=" << scenario.max_new_tokens << "\n";
    std::cout << "========================================\n\n";

    ModelConfig config;
    config.vocab_size = vocab_size;
    config.max_seq_len = max_seq_len;
    config.d_model = d_model;
    config.num_heads = num_heads;
    config.num_layers = num_layers;
    config.d_ff = d_ff;

    std::cout << ">>> Initializing Transformer Engine with Orchestrator..." << std::endl;
    BatchExecutorOrchestrator orchestrator(
        config, 
        "/home/niare/Projects/transformer_inference_engine/vocab.json", 
        "/home/niare/Projects/transformer_inference_engine/merges.txt",
        scenario.batch_size
    );
    
    std::cout << ">>> Loading Weights for accurate generation..." << std::endl;
    orchestrator.load_weights("/home/niare/Projects/transformer_inference_engine/gpt2_weights.bin");

    // Build GenerationConfig from the scenario
    pipeline::GenerationConfig gen_cfg = benchmark::BenchmarkRunConfig::to_generation_config(scenario);

    // Build the batch: take all prompts, cycle if fewer than batch_size
    std::vector<std::vector<int>> batch_sequences;
    std::vector<std::string> batch_filenames;
    std::vector<std::string> batch_prompt_texts;

    for (int i = 0; i < scenario.batch_size; ++i) {
        const auto& p = prompts[i % prompts.size()];
        batch_sequences.push_back(p.input_ids);
        batch_filenames.push_back(p.filename);
        batch_prompt_texts.push_back(p.prompt_text);
    }

    std::cout << "  Batching " << batch_sequences.size() << " sequences";
    if (scenario.batch_size > (int)prompts.size()) {
        std::cout << " (" << prompts.size() << " unique prompts, cycled to fill batch)";
    }
    std::cout << "\n";
    for (int i = 0; i < (int)batch_sequences.size(); ++i) {
        std::cout << "    [" << i << "] '" << batch_filenames[i] << "' (len=" << batch_sequences[i].size() << ")\n";
    }
    std::cout << std::flush;

    // Submit the entire batch as one
    auto future_result = orchestrator.submit_batch(batch_sequences, gen_cfg, StrategyType::STANDARD);
    auto result = future_result.get();

    std::cout << "  Done.\n";

    // Record result
    // Find max prompt len for reporting
    int max_prompt_len = 0;
    for (const auto& seq : batch_sequences) {
        max_prompt_len = std::max(max_prompt_len, (int)seq.size());
    }

    PipelineBenchResult r;
    r.batch_size = scenario.batch_size;
    r.seq_len = max_prompt_len;
    r.max_new_tokens = gen_cfg.max_new_tokens;
    r.prefill_ms = result.metrics.prefill_time_ms;
    r.decode_ms = result.metrics.decode_time_ms;
    r.total_ms = result.metrics.total_time_ms;
    r.prefill_tok_sec = result.metrics.prefill_tokens_per_sec;
    r.decode_tok_sec = result.metrics.decode_tokens_per_sec;

    std::cout << "  Prefill Tok/s: " << r.prefill_tok_sec 
              << " | Decode Tok/s: " << r.decode_tok_sec << "\n";

    // Save generated text for each sequence in the batch
    for (int b = 0; b < (int)result.decoded_texts.size(); ++b) {
        std::string suffix = (scenario.batch_size > 1) ? ("_b" + std::to_string(b)) : "";
        std::string out_path = output_dir + "/" + batch_filenames[b] + suffix + ".out.txt";
        std::ofstream out_file(out_path);
        out_file << "Prompt:\n" << batch_prompt_texts[b] 
                 << "\n\nGenerated:\n" << result.decoded_texts[b] << "\n";
    }

    return { r };
}

// ============================================================================
// Report writers
// ============================================================================

static void writeKernelReport(const std::string& output_dir, const std::string& timestamp, const std::vector<KernelBenchResult>& results) {
    std::string path = output_dir + "/kernel_benchmark_" + timestamp + ".csv";
    std::ofstream csv(path);
    csv << "batch_size,seq_len,embedding_ms,qkv_proj_ms,attn_qk_ms,softmax_ms,layernorm_ms\n";

    for (const auto& r : results) {
        csv << r.batch_size << "," << r.seq_len << "," 
            << r.embedding_ms << "," << r.qkv_proj_ms << "," 
            << r.attn_qk_ms << "," << r.softmax_ms << "," 
            << r.layernorm_ms << "\n";
    }
    std::cout << "\n  Kernel report saved: " << path << "\n";
}

static void writePipelineReport(const std::string& output_dir, const std::string& timestamp, const std::vector<PipelineBenchResult>& results) {
    std::string path = output_dir + "/pipeline_benchmark_" + timestamp + ".csv";
    std::ofstream csv(path);
    csv << "batch_size,seq_len,max_new_tokens,prefill_ms,decode_ms,total_ms,prefill_tok_sec,decode_tok_sec\n";

    for (const auto& r : results) {
        csv << r.batch_size << "," << r.seq_len << "," << r.max_new_tokens << ","
            << r.prefill_ms << "," << r.decode_ms << "," << r.total_ms << "," 
            << r.prefill_tok_sec << "," << r.decode_tok_sec << "\n";
    }
    std::cout << "  Pipeline report saved: " << path << "\n";
}

static void writeSummaryReport(const std::string& output_dir, const std::string& timestamp, 
                               const std::vector<KernelBenchResult>& kernel_results,
                               const std::vector<PipelineBenchResult>& pipeline_results) {
    std::string path = output_dir + "/summary_" + timestamp + ".txt";
    std::ofstream out(path);

    out << "================================================================\n";
    out << "  TRANSFORMER INFERENCE PIPELINE — PERFORMANCE BENCHMARK REPORT\n";
    out << "  Generated: " << timestamp << "\n";
    out << "================================================================\n\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    out << "GPU: " << prop.name << "\n";
    out << "  SM: " << prop.major << "." << prop.minor << "\n";
    out << "  VRAM: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    out << "  SMs: " << prop.multiProcessorCount << "\n\n";

    out << "--- KERNEL BENCHMARKS ---\n\n";
    out << std::left << std::setw(10) << "B" << std::setw(10) << "S"
        << std::setw(15) << "Embedding" << std::setw(15) << "Proj"
        << std::setw(15) << "Attn QK" << std::setw(15) << "Softmax"
        << std::setw(15) << "LayerNorm" << "\n";
    
    for (const auto& r : kernel_results) {
        out << std::left << std::setw(10) << r.batch_size << std::setw(10) << r.seq_len
            << std::fixed << std::setprecision(3) 
            << std::setw(15) << r.embedding_ms
            << std::setw(15) << r.qkv_proj_ms
            << std::setw(15) << r.attn_qk_ms
            << std::setw(15) << r.softmax_ms
            << std::setw(15) << r.layernorm_ms << "\n";
    }

    out << "\n--- PIPELINE GENERATION BENCHMARKS ---\n\n";
    out << std::left << std::setw(10) << "B" << std::setw(10) << "S" << std::setw(15) << "NewTokens"
        << std::setw(15) << "Prefill ms" << std::setw(15) << "Decode ms" 
        << std::setw(20) << "Prefill Tok/s" << std::setw(20) << "Decode Tok/s" << "\n";
    
    for (const auto& r : pipeline_results) {
        out << std::left << std::setw(10) << r.batch_size << std::setw(10) << r.seq_len << std::setw(15) << r.max_new_tokens
            << std::fixed << std::setprecision(2) 
            << std::setw(15) << r.prefill_ms
            << std::setw(15) << r.decode_ms
            << std::setw(20) << r.prefill_tok_sec
            << std::setw(20) << r.decode_tok_sec << "\n";
    }

    std::cout << "  Summary report saved: " << path << "\n";
}

int main(int argc, char** argv) {
    std::cout << "================================================================\n";
    std::cout << "  Transformer Pipeline — Performance Benchmark Suite\n";
    std::cout << "================================================================\n";

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor 
              << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB)\n\n";

    std::string input_dir = "/home/niare/Projects/transformer_inference_engine/dataset/input";
    int batch_size = 1;
    int max_new_tokens = 50;
    std::string session_id = "";
    std::string output_dir = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset-dir" && i + 1 < argc) {
            input_dir = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--max-new-tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--session-id" && i + 1 < argc) {
            session_id = argv[++i];
        }
    }

    if (session_id.empty()) {
        session_id = getTimestamp();
    }
    std::string timestamp = session_id;
    
    if (output_dir.empty()) {
        output_dir = "./docs/performance_testing/run_" + timestamp;
    }
    std::string report_dir = output_dir;
    fs::create_directories(report_dir);

    int VOCAB_SIZE = 50257;
    int MAX_SEQ_LEN = 1024;
    int D_MODEL = 768;
    int NUM_HEADS = 12;
    int NUM_LAYERS = 12;
    int D_FF = 3072;

    // ---- Benchmark Run Configuration ----
    benchmark::BenchmarkRunConfig run_config;
    run_config.scenarios = {
        { .batch_size = batch_size, .max_new_tokens = max_new_tokens }
    };

    GPT2Tokenizer temp_tokenizer;
    temp_tokenizer.load("/home/niare/Projects/transformer_inference_engine/vocab.json", "/home/niare/Projects/transformer_inference_engine/merges.txt");
    
    std::vector<PromptConfig> prompts;
    
    if (fs::exists(input_dir)) {
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (entry.path().extension() == ".txt") {
                std::ifstream file(entry.path());
                std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                PromptConfig p;
                p.filename = entry.path().filename().string();
                p.prompt_text = content;
                p.input_ids = temp_tokenizer.encode(content);
                std::cout << "DEBUG FILE " << p.filename << " tokenized as: ";
                for (int id : p.input_ids) std::cout << id << " ";
                std::cout << "\n";
                p.max_new_tokens = 0; // 0 = use scenario default
                prompts.push_back(p);
            }
        }
    }

    if (prompts.empty()) {
        std::cerr << "No prompts found in " << input_dir << "!\n";
        return 1;
    }

    // Build kernel benchmark configs from first scenario's batch_size
    const auto& default_scenario = run_config.scenarios[0];
    std::vector<BenchmarkConfig> input_configs;
    for (const auto& p : prompts) {
        int s_len = std::min((int)p.input_ids.size(), MAX_SEQ_LEN);
        input_configs.push_back({default_scenario.batch_size, s_len, default_scenario.max_new_tokens});
    }

    // BENCHMARK 1: Kernel-level timing
    auto kernel_results = benchmarkKernels(input_configs, D_MODEL, NUM_HEADS, D_FF);
    writeKernelReport(report_dir, timestamp, kernel_results);

    // BENCHMARK 2: Full pipeline — run each scenario
    std::vector<PipelineBenchResult> all_pipeline_results;
    for (const auto& scenario : run_config.scenarios) {
        std::string dataset_output_dir = "/home/niare/Projects/transformer_inference_engine/dataset/output/run_" + timestamp;
        fs::create_directories(dataset_output_dir);

        auto pipeline_results = benchmarkPipeline(prompts, scenario, dataset_output_dir, VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF);
        all_pipeline_results.insert(all_pipeline_results.end(), pipeline_results.begin(), pipeline_results.end());
    }
    writePipelineReport(report_dir, timestamp, all_pipeline_results);

    writeSummaryReport(report_dir, timestamp, kernel_results, all_pipeline_results);
    
    std::cout << "\n================================================================\n";
    std::cout << "  Benchmark complete. Reports in: " << report_dir << "/\n";
    std::cout << "================================================================\n";

    return 0;
}
