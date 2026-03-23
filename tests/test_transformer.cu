#include "transformer.h"
#include "memory.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            return -1; \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// Helpers: run the Python reference script and parse its output
// ---------------------------------------------------------------------------

/// Parse a line of space-separated floats into `out`. Returns number parsed.
static int parse_floats(const std::string& line, std::vector<float>& out) {
    std::istringstream iss(line);
    float v;
    int count = 0;
    while (iss >> v) {
        out.push_back(v);
        ++count;
    }
    return count;
}

/// Parse a line of space-separated ints into `out`. Returns number parsed.
static int parse_ints(const std::string& line, std::vector<int>& out) {
    std::istringstream iss(line);
    int v;
    int count = 0;
    while (iss >> v) {
        out.push_back(v);
        ++count;
    }
    return count;
}

/// Struct to hold all per-layer weights
struct LayerWeights {
    std::vector<float> attn_norm_gamma;
    std::vector<float> attn_norm_beta;
    std::vector<float> W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o;
    std::vector<float> ffn_norm_gamma;
    std::vector<float> ffn_norm_beta;
    std::vector<float> W_up, b_up, W_down, b_down;
};

/// Run the Python reference, capture output.
/// Returns 0 on success.
static int run_python_reference(
        int vocab_size, int max_seq_len, int d_model, int num_heads,
        int num_layers, int d_ff, int batch_size, int seq_len, int seed,
        double& python_time_ms,
        std::vector<int>& h_token_ids,
        std::vector<float>& h_token_embed,
        std::vector<float>& h_pos_embed,
        std::vector<LayerWeights>& layer_weights,
        std::vector<float>& h_final_norm_gamma,
        std::vector<float>& h_final_norm_beta,
        std::vector<float>& h_lm_head,
        std::vector<float>& h_expected) {

    // Find the Python script
    const char* script_paths[] = {
        "../tools/python_get_transformer_output.py",
        "tools/python_get_transformer_output.py",
    };
    const char* script = nullptr;
    for (auto path : script_paths) {
        FILE* f = std::fopen(path, "r");
        if (f) { std::fclose(f); script = path; break; }
    }
    if (!script) {
        std::cerr << "ERROR: Cannot find python_get_transformer_output.py" << std::endl;
        return -1;
    }

    // Find a Python interpreter that has torch installed
    const char* python_paths[] = {
        "../venv/bin/python3",
        "venv/bin/python3",
        "python3",
    };
    const char* python = nullptr;
    for (auto p : python_paths) {
        FILE* f = std::fopen(p, "r");
        if (f) { std::fclose(f); python = p; break; }
    }
    if (!python) python = "python3";  // last resort

    char cmd[2048];
    std::snprintf(cmd, sizeof(cmd),
        "%s %s --generate "
        "--vocab_size %d --max_seq_len %d --d_model %d --num_heads %d "
        "--num_layers %d --d_ff %d --batch_size %d --seq_len %d --seed %d",
        python, script, vocab_size, max_seq_len, d_model, num_heads,
        num_layers, d_ff, batch_size, seq_len, seed);

    std::cout << "Running Python reference...\n  " << cmd << std::endl;

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "ERROR: popen failed for Python script" << std::endl;
        return -1;
    }

    // Read all output into a vector of lines
    std::vector<std::string> lines;
    {
        std::string current_line;
        char buf[65536];
        while (std::fgets(buf, sizeof(buf), pipe)) {
            current_line += buf;
            if (!current_line.empty() && current_line.back() == '\n') {
                while (!current_line.empty() && (current_line.back() == '\n' || current_line.back() == '\r'))
                    current_line.pop_back();
                if (!current_line.empty())
                    lines.push_back(std::move(current_line));
                current_line.clear();
            }
        }
        if (!current_line.empty()) {
            while (!current_line.empty() && (current_line.back() == '\n' || current_line.back() == '\r'))
                current_line.pop_back();
            if (!current_line.empty())
                lines.push_back(std::move(current_line));
        }
    }

    int status = pclose(pipe);
    if (status != 0) {
        std::cerr << "ERROR: Python script exited with status " << status << std::endl;
        return -1;
    }

    // Expected lines: 1 (time) + 1 (token_ids) + 1 (token_embed) + 1 (pos_embed)
    //                 + num_layers * 16 (per-layer weights)
    //                 + 2 (final norm gamma/beta) + 1 (lm_head) + 1 (expected logits)
    int expected_lines = 4 + num_layers * 16 + 2 + 1 + 1;
    if ((int)lines.size() < expected_lines) {
        std::cerr << "ERROR: Expected " << expected_lines << " output lines from Python, got " 
                  << lines.size() << std::endl;
        return -1;
    }

    int line_idx = 0;

    // Line 0: python_time_ms
    python_time_ms = std::stod(lines[line_idx++]);

    // Line 1: token_ids
    parse_ints(lines[line_idx++], h_token_ids);

    // Line 2: token_embedding_table
    parse_floats(lines[line_idx++], h_token_embed);

    // Line 3: pos_embedding_table
    parse_floats(lines[line_idx++], h_pos_embed);

    // Per-layer weights (16 lines each)
    layer_weights.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        parse_floats(lines[line_idx++], layer_weights[i].attn_norm_gamma);
        parse_floats(lines[line_idx++], layer_weights[i].attn_norm_beta);
        parse_floats(lines[line_idx++], layer_weights[i].W_q);
        parse_floats(lines[line_idx++], layer_weights[i].b_q);
        parse_floats(lines[line_idx++], layer_weights[i].W_k);
        parse_floats(lines[line_idx++], layer_weights[i].b_k);
        parse_floats(lines[line_idx++], layer_weights[i].W_v);
        parse_floats(lines[line_idx++], layer_weights[i].b_v);
        parse_floats(lines[line_idx++], layer_weights[i].W_o);
        parse_floats(lines[line_idx++], layer_weights[i].b_o);
        parse_floats(lines[line_idx++], layer_weights[i].ffn_norm_gamma);
        parse_floats(lines[line_idx++], layer_weights[i].ffn_norm_beta);
        parse_floats(lines[line_idx++], layer_weights[i].W_up);
        parse_floats(lines[line_idx++], layer_weights[i].b_up);
        parse_floats(lines[line_idx++], layer_weights[i].W_down);
        parse_floats(lines[line_idx++], layer_weights[i].b_down);
    }

    // Final norm
    parse_floats(lines[line_idx++], h_final_norm_gamma);
    parse_floats(lines[line_idx++], h_final_norm_beta);

    // LM Head
    parse_floats(lines[line_idx++], h_lm_head);

    // Expected logits
    parse_floats(lines[line_idx++], h_expected);

    // --- Sanity checks ---
    int BT  = batch_size * seq_len;
    int DD  = d_model * d_model;
    int DF  = d_model * d_ff;

    if ((int)h_token_ids.size()   != BT)                    { std::cerr << "token_ids size mismatch\n";   return -1; }
    if ((int)h_token_embed.size() != vocab_size * d_model)  { std::cerr << "token_embed size mismatch\n"; return -1; }
    if ((int)h_pos_embed.size()   != max_seq_len * d_model) { std::cerr << "pos_embed size mismatch\n";   return -1; }
    if ((int)h_lm_head.size()     != d_model * vocab_size)  { std::cerr << "lm_head size mismatch\n";     return -1; }
    if ((int)h_expected.size()    != BT * vocab_size)       { std::cerr << "expected size mismatch\n";    return -1; }

    for (int i = 0; i < num_layers; ++i) {
        auto& lw = layer_weights[i];
        if ((int)lw.attn_norm_gamma.size() != d_model) { std::cerr << "layer " << i << " attn_norm_gamma size mismatch\n"; return -1; }
        if ((int)lw.attn_norm_beta.size()  != d_model) { std::cerr << "layer " << i << " attn_norm_beta size mismatch\n";  return -1; }
        if ((int)lw.W_q.size()  != DD) { std::cerr << "layer " << i << " W_q size mismatch\n";  return -1; }
        if ((int)lw.b_q.size()  != d_model) { std::cerr << "layer " << i << " b_q size mismatch\n";  return -1; }
        if ((int)lw.W_k.size()  != DD) { std::cerr << "layer " << i << " W_k size mismatch\n";  return -1; }
        if ((int)lw.b_k.size()  != d_model) { std::cerr << "layer " << i << " b_k size mismatch\n";  return -1; }
        if ((int)lw.W_v.size()  != DD) { std::cerr << "layer " << i << " W_v size mismatch\n";  return -1; }
        if ((int)lw.b_v.size()  != d_model) { std::cerr << "layer " << i << " b_v size mismatch\n";  return -1; }
        if ((int)lw.W_o.size()  != DD) { std::cerr << "layer " << i << " W_o size mismatch\n";  return -1; }
        if ((int)lw.b_o.size()  != d_model) { std::cerr << "layer " << i << " b_o size mismatch\n";  return -1; }
        if ((int)lw.ffn_norm_gamma.size() != d_model) { std::cerr << "layer " << i << " ffn_norm_gamma size mismatch\n"; return -1; }
        if ((int)lw.ffn_norm_beta.size()  != d_model) { std::cerr << "layer " << i << " ffn_norm_beta size mismatch\n";  return -1; }
        if ((int)lw.W_up.size()    != DF)      { std::cerr << "layer " << i << " W_up size mismatch\n";   return -1; }
        if ((int)lw.b_up.size()    != d_ff)    { std::cerr << "layer " << i << " b_up size mismatch\n";   return -1; }
        if ((int)lw.W_down.size()  != DF)      { std::cerr << "layer " << i << " W_down size mismatch\n"; return -1; }
        if ((int)lw.b_down.size()  != d_model) { std::cerr << "layer " << i << " b_down size mismatch\n"; return -1; }
    }

    return 0;
}

// ---------------------------------------------------------------------------

int main() {
    auto test_start = std::chrono::high_resolution_clock::now();

    // ---- Test parameters (small for fast testing) ----
    const int VOCAB_SIZE   = 128;
    const int MAX_SEQ_LEN  = 16;
    const int D_MODEL      = 64;
    const int NUM_HEADS    = 2;
    const int NUM_LAYERS   = 2;
    const int D_FF         = 256;
    const int BATCH_SIZE   = 2;
    const int SEQ_LEN      = 4;
    const int SEED         = 42;

    std::cout << "Using fixed seed: " << SEED << std::endl;
    std::cout << "Config: Vocab=" << VOCAB_SIZE << " MaxSeq=" << MAX_SEQ_LEN
              << " D=" << D_MODEL << " H=" << NUM_HEADS << " L=" << NUM_LAYERS
              << " FF=" << D_FF << " B=" << BATCH_SIZE << " T=" << SEQ_LEN << std::endl;

    // ---- Step 1: Get Python ground truth ----
    double python_time_ms = 0.0;
    std::vector<int> h_token_ids;
    std::vector<float> h_token_embed, h_pos_embed;
    std::vector<LayerWeights> layer_weights;
    std::vector<float> h_final_norm_gamma, h_final_norm_beta;
    std::vector<float> h_lm_head;
    std::vector<float> h_expected;

    if (run_python_reference(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF,
                             BATCH_SIZE, SEQ_LEN, SEED,
                             python_time_ms,
                             h_token_ids, h_token_embed, h_pos_embed,
                             layer_weights,
                             h_final_norm_gamma, h_final_norm_beta,
                             h_lm_head, h_expected) != 0) {
        std::cerr << ">>> FAIL: Could not get Python reference output." << std::endl;
        return -1;
    }

    std::cout << "✅ Python reference completed in " << python_time_ms << " ms" << std::endl;

    // ---- Step 2: Run CUDA forward pass with the SAME random data ----
    const int total_logits = BATCH_SIZE * SEQ_LEN * VOCAB_SIZE;
    std::vector<float> h_output(total_logits);

    // Setup memory arenas (generous for all weights + inference scratch)
    size_t weights_arena_size = static_cast<size_t>(1024) * 1024 * 64;  // 64 MB
    size_t inference_arena_size = static_cast<size_t>(1024) * 1024 * 64; // 64 MB
    GPUMemoryArena weights_arena(weights_arena_size);
    GPUMemoryArena inference_arena(inference_arena_size);

    // Construct Transformer
    Transformer transformer(VOCAB_SIZE, MAX_SEQ_LEN, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, weights_arena);

    // Load embeddings
    transformer.load_embeddings(h_token_embed.data(), h_pos_embed.data());

    // Load per-layer weights
    for (int i = 0; i < NUM_LAYERS; ++i) {
        TransformerBlock* block = transformer.get_block(i);
        auto& lw = layer_weights[i];

        // Attention norm
        block->get_attn_norm().load_weights(lw.attn_norm_gamma.data(), lw.attn_norm_beta.data());

        // Attention projections
        block->get_attention().load_weights(lw.W_q.data(), lw.W_k.data(), lw.W_v.data(), lw.W_o.data(),
                                            lw.b_q.data(), lw.b_k.data(), lw.b_v.data(), lw.b_o.data());

        // FFN norm
        block->get_ffn_norm().load_weights(lw.ffn_norm_gamma.data(), lw.ffn_norm_beta.data());

        // FFN weights
        block->get_ffn().load_weights(lw.W_up.data(), lw.b_up.data(), lw.W_down.data(), lw.b_down.data());
    }

    // Load final norm
    transformer.get_final_norm().load_weights(h_final_norm_gamma.data(), h_final_norm_beta.data());

    // Load LM head
    transformer.load_head(h_lm_head.data());

    // Allocate token_ids and logits on GPU
    int* d_token_ids = nullptr;
    float* d_logits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_token_ids, h_token_ids.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_logits, total_logits * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_logits, 0, total_logits * sizeof(float)));

    // Copy token IDs to device
    CUDA_CHECK(cudaMemcpy(d_token_ids, h_token_ids.data(),
                          h_token_ids.size() * sizeof(int), cudaMemcpyHostToDevice));

    std::cout << "\nLaunching CUDA Transformer forward pass..." << std::endl;

    // Time the CUDA forward pass using CUDA events
    cudaEvent_t cuda_start, cuda_end;
    CUDA_CHECK(cudaEventCreate(&cuda_start));
    CUDA_CHECK(cudaEventCreate(&cuda_end));

    CUDA_CHECK(cudaEventRecord(cuda_start, 0));
    transformer.forward(d_token_ids, d_logits, BATCH_SIZE, SEQ_LEN, inference_arena, 0);
    CUDA_CHECK(cudaEventRecord(cuda_end, 0));
    CUDA_CHECK(cudaEventSynchronize(cuda_end));

    float cuda_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cuda_time_ms, cuda_start, cuda_end));

    CUDA_CHECK(cudaEventDestroy(cuda_start));
    CUDA_CHECK(cudaEventDestroy(cuda_end));

    // Copy logits back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_logits,
                          total_logits * sizeof(float), cudaMemcpyDeviceToHost));

    // Free manually allocated GPU memory
    CUDA_CHECK(cudaFree(d_token_ids));
    CUDA_CHECK(cudaFree(d_logits));

    // ---- Step 3: Compare results ----
    const float tolerance = 1e-2f;  // slightly relaxed for float32 accumulation diffs
    int error_count = 0;
    float worst_diff = 0.0f;
    int worst_idx = -1;

    for (int i = 0; i < total_logits; ++i) {
        float diff = std::abs(h_output[i] - h_expected[i]);
        if (diff > worst_diff) {
            worst_diff = diff;
            worst_idx = i;
        }
        if (diff > tolerance) {
            if (error_count < 10) {  // only print first 10 mismatches
                std::cout << "Mismatch at index " << i
                          << " | Expected " << h_expected[i]
                          << " Got " << h_output[i]
                          << " (diff: " << diff << ")" << std::endl;
            }
            error_count++;
        }
    }

    // ---- Step 4: Print timing comparison ----
    auto test_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = test_end - test_start;

    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "📊 TIMING COMPARISON" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "⏱️  Python (PyTorch) time:  " << python_time_ms    << " ms" << std::endl;
    std::cout << "⏱️  CUDA kernel time:       " << cuda_time_ms      << " ms" << std::endl;

    if (python_time_ms > 0.0) {
        double speedup = python_time_ms / cuda_time_ms;
        std::cout << "🚀 CUDA speedup:           " << speedup << "x" << std::endl;
    }

    std::cout << "⏱️  Total test time:        " << total_duration.count() << " ms" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // ---- Final verdict ----
    std::cout << "\nWorst diff: " << worst_diff << " at index " << worst_idx << std::endl;

    if (error_count == 0) {
        std::cout << ">>> PASS: All " << total_logits
                  << " logit values match within tolerance (" << tolerance << ")." << std::endl;
        return 0;
    } else {
        std::cout << ">>> FAIL: Found " << error_count << " mismatches out of "
                  << total_logits << " values." << std::endl;
        return -1;
    }
}
