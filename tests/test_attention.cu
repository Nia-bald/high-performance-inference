#include "layers/attention.h"
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

/// !important before testing make sure TILESIZE is a factor headsize before testing

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

/// Run the Python reference, capture its 7 output lines.
/// Returns 0 on success. Fills vectors and python_time_ms.
static int run_python_reference(
        int d_model, int num_heads, int batch_size, int seq_len, int seed,
        double& python_time_ms,
        std::vector<float>& h_input,
        std::vector<float>& h_W_q,
        std::vector<float>& h_W_k,
        std::vector<float>& h_W_v,
        std::vector<float>& h_W_o,
        std::vector<float>& h_expected) {

    // Build the command – look for the script relative to the executable or
    // at a well-known project-relative path.
    // Try both relative paths: from build/ subdir and from project root
    const char* script_paths[] = {
        "../tools/python_get_attention_output.py",
        "tools/python_get_attention_output.py",
    };
    const char* script = nullptr;
    for (auto path : script_paths) {
        FILE* f = std::fopen(path, "r");
        if (f) { std::fclose(f); script = path; break; }
    }
    if (!script) {
        std::cerr << "ERROR: Cannot find python_get_attention_output.py" << std::endl;
        return -1;
    }

    // Find a Python interpreter that has torch installed
    // Try the project venv first, then fall back to system python3
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

    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
        "%s %s --generate "
        "--d_model %d --num_heads %d --batch_size %d --seq_len %d --seed %d",
        python, script, d_model, num_heads, batch_size, seq_len, seed);

    std::cout << "Running Python reference...\n  " << cmd << std::endl;

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        std::cerr << "ERROR: popen failed for Python script" << std::endl;
        return -1;
    }

    // Read all output into a vector of lines
    std::vector<std::string> lines;
    {
        char buf[1 << 20]; // 1 MB buffer per line – weights can be large
        while (std::fgets(buf, sizeof(buf), pipe)) {
            std::string s(buf);
            // strip trailing newline
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r'))
                s.pop_back();
            if (!s.empty())
                lines.push_back(std::move(s));
        }
    }

    int status = pclose(pipe);
    if (status != 0) {
        std::cerr << "ERROR: Python script exited with status " << status << std::endl;
        return -1;
    }

    if (lines.size() < 7) {
        std::cerr << "ERROR: Expected 7 output lines from Python, got " << lines.size() << std::endl;
        return -1;
    }

    // Line 0: python_time_ms
    python_time_ms = std::stod(lines[0]);

    // Lines 1-6: input, W_q, W_k, W_v, W_o, expected output
    parse_floats(lines[1], h_input);
    parse_floats(lines[2], h_W_q);
    parse_floats(lines[3], h_W_k);
    parse_floats(lines[4], h_W_v);
    parse_floats(lines[5], h_W_o);
    parse_floats(lines[6], h_expected);

    // Sanity checks
    int BTD = batch_size * seq_len * d_model;
    int DD  = d_model * d_model;
    if ((int)h_input.size()    != BTD) { std::cerr << "input size mismatch\n";    return -1; }
    if ((int)h_W_q.size()      != DD)  { std::cerr << "W_q size mismatch\n";      return -1; }
    if ((int)h_W_k.size()      != DD)  { std::cerr << "W_k size mismatch\n";      return -1; }
    if ((int)h_W_v.size()      != DD)  { std::cerr << "W_v size mismatch\n";      return -1; }
    if ((int)h_W_o.size()      != DD)  { std::cerr << "W_o size mismatch\n";      return -1; }
    if ((int)h_expected.size() != BTD) { std::cerr << "expected size mismatch\n"; return -1; }

    return 0;
}

// ---------------------------------------------------------------------------

int main() {
    auto test_start = std::chrono::high_resolution_clock::now();

    // ---- Test parameters (feel free to change) ----
    const int D_MODEL    = 64;
    const int NUM_HEADS  = 2;
    const int BATCH_SIZE = 8;
    const int SEQ_LEN    = 2;

    // Random seed – use current time so every run is truly random
    const int SEED = static_cast<int>(
        std::chrono::system_clock::now().time_since_epoch().count() % 100000);
    std::cout << "Using random seed: " << SEED << std::endl;

    // ---- Step 1: Get Python ground truth (random weights + expected output) ----
    double python_time_ms = 0.0;
    std::vector<float> h_input, h_W_q, h_W_k, h_W_v, h_W_o, h_expected;

    if (run_python_reference(D_MODEL, NUM_HEADS, BATCH_SIZE, SEQ_LEN, SEED,
                             python_time_ms,
                             h_input, h_W_q, h_W_k, h_W_v, h_W_o, h_expected) != 0) {
        std::cerr << ">>> FAIL: Could not get Python reference output." << std::endl;
        return -1;
    }

    std::cout << "✅ Python reference completed in " << python_time_ms << " ms" << std::endl;

    // ---- Step 2: Run CUDA forward pass with the SAME random data ----
    const int total_elements = BATCH_SIZE * SEQ_LEN * D_MODEL;
    std::vector<float> h_output(total_elements);

    // Setup memory arenas
    size_t arena_size = static_cast<size_t>(1024) * 1024 * 16; // 16 MB
    GPUMemoryArena weights_arena(arena_size);
    GPUMemoryArena inference_arena(arena_size);

    // Construct SelfAttention layer
    SelfAttention attention(D_MODEL, NUM_HEADS, weights_arena,
                            D_MODEL / NUM_HEADS, D_MODEL / NUM_HEADS);

    // Load the randomly-generated weight matrices
    attention.load_weights(h_W_q.data(), h_W_k.data(), h_W_v.data(), h_W_o.data());

    // Allocate input and output on GPU
    float* d_input  = inference_arena.allocate<float>(h_input.size());
    float* d_output = inference_arena.allocate<float>(h_output.size());

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                          h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "\nLaunching CUDA SelfAttention forward pass..." << std::endl;

    // Time the CUDA forward pass
    auto cuda_start = std::chrono::high_resolution_clock::now();
    attention.forward(BATCH_SIZE, SEQ_LEN, d_input, d_output, inference_arena, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto cuda_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cuda_duration = cuda_end - cuda_start;
    double cuda_time_ms = cuda_duration.count();

    // Copy output back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Step 3: Compare results ----
    const float tolerance = 1e-2f;  // slightly relaxed for float32 accumulation diffs
    int error_count = 0;
    float worst_diff = 0.0f;
    int worst_idx = -1;

    for (int i = 0; i < total_elements; ++i) {
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
        std::cout << ">>> PASS: All " << total_elements
                  << " values match within tolerance (" << tolerance << ")." << std::endl;
        return 0;
    } else {
        std::cout << ">>> FAIL: Found " << error_count << " mismatches out of "
                  << total_elements << " values." << std::endl;
        return -1;
    }
}
