# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

set(experimental_gen_ai_cpp_source_files_gpu
    # Attention Ops
    src/attention/attention.cpp
    src/attention/gqa_attn_splitk.cu
    # Quantize Ops
    src/quantize/quantize.cu
    src/quantize/quantize.cpp
    # Comm Ops
    src/comm/car.cu
    src/comm/car.cpp
    # KV Cache Ops
    src/kv_cache/kv_cache.cu
    src/kv_cache/kv_cache.cpp)

# Set the source file for FB only CPP
if(USE_FB_ONLY)
  file(GLOB fb_only_ops_sources
      fb/src/*/*.cu
      fb/src/*/*.cpp)
  list(APPEND experimental_gen_ai_cpp_source_files_gpu ${fb_only_ops_sources})
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
