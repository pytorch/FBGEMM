# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CUDA Setup
################################################################################

BLOCK_PRINT(
  "CMAKE CUDA Flags"
  ""
  "CMAKE_CUDA_COMPILER_VERSION=${CMAKE_CUDA_COMPILER_VERSION}"
)

BLOCK_PRINT(
  "NCCL Flags"
  ""
  "NCCL_INCLUDE_DIRS=${NCCL_INCLUDE_DIRS}"
  "NCCL_LIBRARIES=${NCCL_LIBRARIES}"
)

# Set NVML_LIB_PATH if provided, or detect the default lib path
if(NOT NVML_LIB_PATH)
  set(DEFAULT_NVML_LIB_PATH
      "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so")

  if(EXISTS ${DEFAULT_NVML_LIB_PATH})
    message(STATUS "Setting NVML_LIB_PATH: \
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so")
    set(NVML_LIB_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so")
  endif()
endif()

if(NVML_LIB_PATH)
  message(STATUS "Found NVML_LIB_PATH: ${NVML_LIB_PATH}")
endif()

# The libcuda.so path was previously set by PyTorch CMake, but the setup has
# been removed from the PyTorch codebase, see:
# https://github.com/pytorch/pytorch/pull/128801
set(CUDA_DRIVER_LIBRARIES "${CUDA_cuda_driver_LIBRARY}" CACHE FILEPATH "")

BLOCK_PRINT(
  "CUDA Driver Path"
  ""
  "CUDA_DRIVER_LIBRARIES=${CUDA_DRIVER_LIBRARIES}"
)

# cuBLAS is called directly from src/sparse_ops/sparse_permute102.cu, but
# ${TORCH_LIBRARIES} does not put a libcublas DT_NEEDED entry on our targets.
# Consumers dlopen() our .SO files with RTLD_LOCAL, so the cuBLAS that PyTorch
# preloads is not visible in our lookup scope either, and the symbols come out
# undefined at import time.  Link it ourselves.
set(CUDA_CUBLAS_LIBRARIES "")
if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  if(NOT TARGET CUDA::cublas)
    find_package(CUDAToolkit QUIET)
  endif()

  if(TARGET CUDA::cublas)
    set(CUDA_CUBLAS_LIBRARIES CUDA::cublas)
  else()
    message(WARNING
      "CUDA::cublas target not found; fbgemm_gpu_py will be built without an "
      "explicit libcublas link and may fail to load with an undefined "
      "cublas* symbol.")
  endif()
endif()

BLOCK_PRINT(
  "CUDA cuBLAS"
  ""
  "CUDA_CUBLAS_LIBRARIES=${CUDA_CUBLAS_LIBRARIES}"
)
