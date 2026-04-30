#include "layers/layer.h"
#include <fstream>
#include <mutex>

bool ENABLE_LAYER_PROFILING = true;
static std::ofstream csv_file;
static std::mutex csv_mutex;
static bool is_file_initialized = false;

void log_layer_profile_csv(const std::string& layer_name, int batch_size, int seq_len, float ms) {
    std::lock_guard<std::mutex> lock(csv_mutex);
    
    if (!is_file_initialized) {
        csv_file.open("layer_profile.csv", std::ios::out);
        if (csv_file.is_open()) {
            csv_file << "LayerName,BatchSize,SeqLen,Time_ms\n";
        }
        is_file_initialized = true;
    }
    
    if (csv_file.is_open()) {
        csv_file << layer_name << "," << batch_size << "," << seq_len << "," << ms << "\n";
        csv_file.flush();
    }
}
