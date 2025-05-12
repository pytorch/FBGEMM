# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/Utilities.cmake)


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
