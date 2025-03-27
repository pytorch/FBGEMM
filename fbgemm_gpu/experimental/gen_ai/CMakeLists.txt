# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################


file(GLOB experimental_gen_ai_cpp_source_files_gpu
  # Attention Ops
  src/attention/*.cpp
  src/attention/*.cu
  # Comm Ops
  src/comm/*.cpp
  src/comm/*.cu
  # Gather Scatter Ops
  src/gather_scatter/*.cpp
  src/gather_scatter/*.cu
  # Quantize Ops
  src/kv_cache/*.cpp
  src/kv_cache/*.cu
  # Quantize Ops
  src/quantize/*.cpp
  src/quantize/*.cu)

# Set the source file for FB only CPP
if(USE_FB_ONLY)
  file(GLOB fb_only_cxx_sources
      fb/src/moe/*.cu
      fb/src/moe/*.cpp
      fb/src/trt_llm/*.cu
      fb/src/trt_llm/*.cpp)
  list(APPEND experimental_gen_ai_cpp_source_files_gpu ${fb_only_cxx_sources})
endif()

# CUDA-specific sources
file(GLOB_RECURSE experimental_gen_ai_cpp_source_files_cuda
  src/quantize/cutlass_extensions/*.cu
  src/quantize/cutlass_extensions/**/*.cu
  src/quantize/fast_gemv/*.cu
  src/quantize/fast_gemv/**/*.cu
  src/quantize/fast_gemv/**/*.cuh)

# HIP-specific sources
file(GLOB_RECURSE experimental_gen_ai_cpp_source_files_hip
  src/gemm/gemm.cpp
  src/gemm/ck_extensions.hip
  src/quantize/ck_extensions/*.hip
  src/quantize/ck_extensions/**/*.hip)

# Python sources
file(GLOB_RECURSE experimental_gen_ai_python_source_files
  gen_ai/*.py)


################################################################################
# Build Shared Library
################################################################################

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_experimental_gen_ai
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantize
  GPU_SRCS
    ${experimental_gen_ai_cpp_source_files_gpu}
  CUDA_SPECIFIC_SRCS
    ${experimental_gen_ai_cpp_source_files_cuda}
  HIP_SPECIFIC_SRCS
    ${experimental_gen_ai_cpp_source_files_hip})


################################################################################
# Install Shared Library and Python Files
################################################################################

add_to_package(
  DESTINATION fbgemm_gpu/experimental/gen_ai
  TARGETS fbgemm_gpu_experimental_gen_ai
  FILES ${experimental_gen_ai_python_source_files})

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fb/gen_ai)
  install(
    DIRECTORY fb/gen_ai
    DESTINATION fbgemm_gpu/experimental)
endif()
