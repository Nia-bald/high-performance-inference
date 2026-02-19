cmake_minimum_required(VERSION 3.18)
project(inference_engine LANGUAGES CXX CUDA)

# --- Project Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Default to Release for performance (O3 optimization)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# GPU Architecture: GTX 1050 Ti = Compute Capability 6.1
set(CMAKE_CUDA_ARCHITECTURES 61)

# --- Source Management ---

# 1. CUDA Kernels
set(KERNEL_SOURCES
    src/kernels/gemm.cu
    src/kernels/embedding.cu
    src/kernels/layernorm.cu
    src/kernels/softmax.cu
    src/kernels/transpose.cu
    src/kernels/batched_gemm.cu
    src/kernels/batched_upper_triangulation.cu
    src/kernels/batched_one_to_one_gemm.cu
    src/kernels/addition.cu
    src/kernels/activation.cu
    src/kernels/sampling.cu
)

# 2. C++ Layers & Core Logic (The Transformer Implementation)
set(LAYER_SOURCES
    src/transformer.cpp
    src/layers/attention.cpp
    src/layers/feed_forward.cpp
    src/layers/layer_norm.cpp
    src/layers/transformer_block.cpp
    src/memory.cpp
    src/utils/loader.cpp # Assuming you put the loader helper here
)

# --- Define the Shared Library ---
# Compiles all kernels and layers ONCE
add_library(inference_lib STATIC
    ${KERNEL_SOURCES}
    ${LAYER_SOURCES}
)

# Setup Include Directories for the library
target_include_directories(inference_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/layers  # <--- ADD THIS LINE
)

# --- Main Engine Executable ---
if(EXISTS "${CMAKE_SOURCE_DIR}/src/main.cpp")
    add_executable(gpt2_engine src/main.cpp)
    target_link_libraries(gpt2_engine PRIVATE inference_lib)
endif()

# --- Tests ---
# Macro to quickly add test executables
macro(add_inference_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE inference_lib)
endmacro()

# Register your tests
# add_inference_test(test_embedding tests/test_embedding.cu)
add_inference_test(test_softmax tests/test_softmax.cu)
add_inference_test(test_layernorm tests/test_layernorm.cu)
add_inference_test(test_gemm tests/test_gemm.cu)
add_inference_test(test_transpose tests/test_transpose.cu)
add_inference_test(test_batched_gemm tests/test_batched_gemm.cu)
add_inference_test(test_batched_upper_triangulation tests/test_batched_upper_triangulation.cu)
add_inference_test(test_batched_one_to_one_gemm tests/test_batched_one_to_one_gemm.cu)
add_inference_test(test_addition tests/test_addition.cu)

message(STATUS "Configured for GTX 1050 Ti (Compute Capability 6.1)")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")