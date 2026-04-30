#pragma once
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include "memory.h"

extern bool ENABLE_LAYER_PROFILING;

class Layer {
public:
    Layer(const std::string& name) : layer_name(name) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    virtual ~Layer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    const std::string& get_name() const { return layer_name; }

    void forward(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream) {
        if (ENABLE_LAYER_PROFILING) {
            cudaEventRecord(start_event, stream);
        }

        forward_impl(d_input, d_output, batch_size, seq_len, inference_arena, stream);

        if (ENABLE_LAYER_PROFILING) {
            cudaEventRecord(stop_event, stream);
            cudaEventSynchronize(stop_event);
            float ms = 0;
            cudaEventElapsedTime(&ms, start_event, stop_event);
            std::cout << "[Profiler] " << layer_name << " forward pass: " << ms << " ms" << std::endl;
        }
    }

protected:
    std::string layer_name;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    virtual void forward_impl(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream) = 0;
};
