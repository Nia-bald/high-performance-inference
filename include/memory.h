
#pragma once
#include <cstddef>
#include <stdexcept>
#include <cuda_runtime.h>

// primary use is to allocate all memory related to inference in same big chunk
// so the memory is not strided and loading of memory becomes easy

class GPUMemoryArena {
private:
    size_t offset; // how many bytes of memory in use starting from base pointer
    size_t total_size;
    void* base_ptr; // keeping it void as arena might hold multiple different data type

    // when GPU reads data it can read data from a block 256 bytes at once, 
    // we want to make sure all the data or variable we store with within this 256 bytes if possible
    // like we are storing 16 float we can't keep 8 of them in one block of 256 bytes and 8 another
    // we just move the pointer so that all 8 are in 256 bytes
    static const size_t ALIGNMENT = 256;  // keeping it size_t because this is used with bytes in allocate to round up to nearest multiple of alligment
public:
    GPUMemoryArena(size_t size);
    ~GPUMemoryArena();

    template <typename T>
    T* allocate(size_t count){
        return static_cast<T*>(allocate_bytes(count * sizeof(T)));
    } // allocates memory and returns the starting pointer of the data

    void* allocate_bytes(size_t bytes);
    void reset();

    size_t get_user() const {return offset;}
    size_t get_total() const {return total_size;}

};