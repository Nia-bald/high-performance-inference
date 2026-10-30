cmake_minimum_required(VERSION 3.18)

# --- GPU Architecture Configuration ---
# Default target: NVIDIA GeForce GTX 1050 Ti (Compute Capability 6.1).
# Can be overridden via -DCMAKE_CUDA_ARCHITECTURES="<archs>" (e.g. "80;89;90" for HPC targets)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()


project(TransformerEngine VERSION 0.1.0 LANGUAGES CXX CUDA)

# --- Project Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Release for performance (O3 optimization)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Build options
option(TRANSFORMER_ENGINE_BUILD_STANDALONE "Build standalone gpt2_engine executable" ON)
option(TRANSFORMER_ENGINE_BUILD_TESTS "Build test executables" ON)
option(TRANSFORMER_ENGINE_BUILD_BENCHMARKS "Build benchmark executables" ON)

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
    src/kernels/cache_append.cu
    src/kernels/cache_gather.cu
    src/kernels/gemv.cu
    src/kernels/gemv_fp16.cu
    src/kernels/gemm_fp16.cu
    src/kernels/deinterleave.cu
)

# 2. C++ Layers & Core Logic (The Transformer Implementation)
set(LAYER_SOURCES
    src/transformer.cpp
    src/layers/layer.cpp
    src/layers/attention.cpp
    src/layers/feed_forward.cpp
    src/layers/layer_norm.cpp
    src/layers/transformer_block.cpp
    src/memory.cpp
    src/utils/loader.cpp
    src/tokenizer.cpp
    src/pipeline/single_device_strategy.cpp
    src/kv_cache/contiguous_kv_cache.cu
    src/batch_executor.cpp
    src/batch_executor_orchestrator.cpp
)

# --- Define the Engine Library ---
add_library(inference_lib
    ${KERNEL_SOURCES}
    ${LAYER_SOURCES}
)

# Alias for consistent in-tree and external consumption
add_library(TransformerEngine::inference_lib ALIAS inference_lib)

set_target_properties(inference_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME "inference_lib"
)

# Setup Include Directories for build tree and install tree
target_include_directories(inference_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/layers>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/layers>
)

# Compile options: optimization & profiling
target_compile_options(inference_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--use_fast_math;-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Link Dependencies
find_package(CUDAToolkit REQUIRED)
target_link_libraries(inference_lib PUBLIC CUDA::nvtx3 CUDA::cublas)

# --- Main Engine Executable ---
if(TRANSFORMER_ENGINE_BUILD_STANDALONE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(gpt2_engine src/main.cpp)
    target_link_libraries(gpt2_engine PRIVATE TransformerEngine::inference_lib)
endif()

# --- Tests ---
if(TRANSFORMER_ENGINE_BUILD_TESTS)
    macro(add_inference_test test_name source_file)
        add_executable(${test_name} ${source_file})
        target_link_libraries(${test_name} PRIVATE TransformerEngine::inference_lib)
    endmacro()

    # Register tests
    # add_inference_test(test_embedding tests/test_embedding.cu)
    add_inference_test(test_softmax tests/test_softmax.cu)
    add_inference_test(test_layernorm tests/test_layernorm.cu)
    add_inference_test(test_gemm tests/test_gemm.cu)
    add_inference_test(test_transpose tests/test_transpose.cu)
    add_inference_test(test_batched_gemm tests/test_batched_gemm.cu)
    add_inference_test(test_batched_upper_triangulation tests/test_batched_upper_triangulation.cu)
    add_inference_test(test_batched_one_to_one_gemm tests/test_batched_one_to_one_gemm.cu)
    add_inference_test(test_addition tests/test_addition.cu)
    add_inference_test(test_attention tests/test_attention.cu)
    add_inference_test(test_feed_forward tests/test_feed_forward.cu)
    add_inference_test(test_transformer tests/test_transformer.cu)
endif()

# --- Benchmarks ---
if(TRANSFORMER_ENGINE_BUILD_BENCHMARKS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_performance.cu")
        add_executable(bench_performance tests/bench_performance.cu)
        target_link_libraries(bench_performance PRIVATE TransformerEngine::inference_lib)
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/bench_gemv.cu")
        add_executable(bench_gemv tests/bench_gemv.cu)
        target_link_libraries(bench_gemv PRIVATE TransformerEngine::inference_lib)
        target_compile_options(bench_gemv PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    endif()
endif()

# =====================================================================
# --- INSTALLATION & PACKAGE EXPORT ---
# =====================================================================

set(INSTALL_CONFIGDIR "${CMAKE_INSTALL_LIBDIR}/cmake/TransformerEngine")

# Install targets
install(TARGETS inference_lib
    EXPORT TransformerEngineTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install export target file
install(EXPORT TransformerEngineTargets
    FILE TransformerEngineTargets.cmake
    NAMESPACE TransformerEngine::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Install public headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.cuh"
)

# Generate and install package config files
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TransformerEngineConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TransformerEngineConfig.cmake"
    INSTALL_DESTINATION "${INSTALL_CONFIGDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TransformerEngineConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/TransformerEngineConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/TransformerEngineConfigVersion.cmake"
    DESTINATION "${INSTALL_CONFIGDIR}"
)

# Also export build tree targets for convenient subproject / out-of-install usage
export(EXPORT TransformerEngineTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/TransformerEngineTargets.cmake"
    NAMESPACE TransformerEngine::
)

message(STATUS "Configured for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")