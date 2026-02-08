cmake_minimum_required(VERSION 3.18)
project(inference_engine LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CUDA_ARCHITECTURES 61)


include_directories(${CMAKE_SOURCE_DIR}/include)

set(CUDA_SOURCES
    src/kernels/gemm.cu
    src/kernels/embedding.cu
    src/kernels/layernorm.cu
    src/kernels/softmax.cu
    src/kernels/transpose.cu
    src/kernels/batched_gemm.cu
    src/kernels/batched_upper_triangulation.cu
)

set(CPP_SOURCES
    src/memory.cpp
)


add_executable(test_embedding tests/test_embedding.cu src/memory.cpp src/kernels/embedding.cu)

add_executable(test_softmax
    tests/test_softmax.cu
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

add_executable(test_layernorm
    tests/test_layernorm.cu
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

add_executable(test_gemm
    tests/test_gemm.cu
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

add_executable(test_transpose
    tests/test_transpose.cu
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)


add_executable(test_batched_gemm
    tests/test_batched_gemm.cu
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)

add_executable(test_batched_upper_triangulation
    tests/test_batched_upper_triangulation.cu
    ${CUDA_SOURCES}
    ${CPP_SOURCES}
)


target_link_libraries(test_gemm PRIVATE cuda)
target_link_libraries(test_embedding PRIVATE cuda)
target_link_libraries(test_layernorm PRIVATE cuda)
target_link_libraries(test_softmax PRIVATE cuda)
target_link_libraries(test_transpose PRIVATE cuda)
target_link_libraries(test_batched_gemm PRIVATE cuda)
target_link_libraries(test_batched_upper_triangulation PRIVATE cuda)





message(STATUS "Configured for GTX 1050 Ti (Compute Capability 6.1)")