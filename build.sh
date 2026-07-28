#!/usr/bin/env bash

set -euo pipefail

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="$REPO_DIR/build-amd"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
HIP_PATH=${HIP_PATH:-$ROCM_PATH}
HIPCXX=${HIPCXX:-$ROCM_PATH/lib/llvm/bin/clang}
HIP_DEVICE_LIB_PATH=${HIP_DEVICE_LIB_PATH:-$ROCM_PATH/amdgcn/bitcode}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}

if [[ ! -x "$HIPCXX" ]]; then
    echo "HIP compiler not found or not executable: $HIPCXX" >&2
    echo "Set HIPCXX to the ROCm clang executable." >&2
    exit 1
fi

if [[ ! -d "$HIP_PATH" ]]; then
    echo "HIP installation not found: $HIP_PATH" >&2
    echo "Set HIP_PATH and ROCM_PATH to the ROCm installation." >&2
    exit 1
fi

if [[ ! -f "$HIP_DEVICE_LIB_PATH/oclc_abi_version_400.bc" ]]; then
    echo "ROCm device libraries not found: $HIP_DEVICE_LIB_PATH" >&2
    echo "Set HIP_DEVICE_LIB_PATH to the directory containing oclc_abi_version_400.bc." >&2
    exit 1
fi

export ROCM_PATH
export HIP_PATH
export HIPCXX
export HIP_DEVICE_LIB_PATH

cmake \
    -S "$REPO_DIR" \
    -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER="$HIPCXX" \
    -DCMAKE_HIP_ARCHITECTURES="gfx1010;gfx1030;gfx1100" \
    -DGPU_TARGETS="gfx1010;gfx1030;gfx1100" \
    -DGGML_NATIVE=ON \
    -DGGML_OPENMP=ON \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_HIP_RCCL=OFF \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DGGML_VULKAN=ON \
    -DGGML_VULKAN_CHECK_RESULTS=OFF \
    -DGGML_VULKAN_DEBUG=OFF \
    -DGGML_VULKAN_MEMORY_DEBUG=OFF \
    -DGGML_VULKAN_VALIDATE=OFF \
    -DGGML_VULKAN_RUN_TESTS=OFF \
    -DGGML_CCACHE=OFF \
    -DGGML_LTO=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=ON

cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"

echo
echo "Build complete: $BUILD_DIR/bin"
"$BUILD_DIR/bin/llama-cli" --list-devices
