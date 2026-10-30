# Downstream Sample Project

This sample demonstrates how external projects consume `TransformerEngine` using modern CMake `find_package(TransformerEngine REQUIRED)`.

## Prerequisites & Installation

First, build and install `TransformerEngine` to a local directory (e.g. `./dist`):

```bash
# From repository root
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target inference_lib -j$(nproc)
cmake --install build --prefix ./dist
```

## Building Downstream Sample

Configure and build pointing to the install prefix:

```bash
cmake -B build_sample -S examples/downstream_sample -DCMAKE_PREFIX_PATH="$(pwd)/dist"
cmake --build build_sample
./build_sample/downstream_sample
```
