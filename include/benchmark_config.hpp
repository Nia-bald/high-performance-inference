#pragma once

#include "pipeline/metrics.hpp"
#include <vector>
#include <string>

namespace benchmark {

// Defines a single benchmark scenario to run.
// These map directly to the parameters that control how
// a single generation benchmark is executed.
struct ScenarioConfig {
    int batch_size = 1;
    int max_new_tokens = 50;
    // Future: warmup_runs, measure_runs, sampling strategy, etc.
};

// Top-level benchmark configuration.
// Owns the list of scenarios and global settings for the benchmark suite.
struct BenchmarkRunConfig {
    // Global defaults
    std::string weights_path;
    std::string vocab_path;
    std::string merges_path;
    std::string input_dir;

    // Benchmark scenarios to sweep
    std::vector<ScenarioConfig> scenarios = { ScenarioConfig{} };

    // Helper: build a GenerationConfig from a scenario
    static pipeline::GenerationConfig to_generation_config(const ScenarioConfig& scenario) {
        pipeline::GenerationConfig gen_cfg;
        gen_cfg.batch_size = scenario.batch_size;
        gen_cfg.max_new_tokens = scenario.max_new_tokens;
        return gen_cfg;
    }
};

} // namespace benchmark
