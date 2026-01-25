#include "memory.h"
#include <cuda_runtime.h>
#include <iostream>


GPUMemoryArena::GPUMemoryArena(size_t size):total_size(size){
    cudaError_t err = cudaMalloc(&base_ptr, total_size);
    if (err != cudaSuccess){
        throw std::runtime_error("Failed ti allocate GPU Arena: " 
            + std::string(cudaGetErrorString(err)));
    }
    std::cout << "[GPU Arena] Allocated" << total_size/1024/1024 << "MB\n";
    offset = 0;
}

GPUMemoryArena::~GPUMemoryArena(){
    if (base_ptr) {
        cudaFree(base_ptr);
    }
}

void* GPUMemoryArena::allocate_bytes(size_t bytes){

    // round up to nearest multiple of allignment
    size_t aligned_bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    if (offset + aligned_bytes > total_size) {
        throw std::runtime_error("OOM: GPU Arena Out of memory!");
    }

    void* ptr = static_cast<char*>(base_ptr) + offset;

    offset += aligned_bytes;
    return ptr;
}

void GPUMemoryArena::reset(){
    offset = 0;
}
