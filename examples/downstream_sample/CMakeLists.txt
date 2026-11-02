cmake_minimum_required(VERSION 3.18)
project(downstream_sample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the installed or exported TransformerEngine package
find_package(TransformerEngine REQUIRED)

add_executable(downstream_sample main.cpp)
target_link_libraries(downstream_sample PRIVATE TransformerEngine::inference_lib)
