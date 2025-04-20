# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

file(GLOB experimental_gen_ai_cpp_source_files_cpu
  src/quantize/*.cpp)

file(GLOB experimental_gen_ai_cpp_source_files_gpu
  src/quantize/*.cu)

if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  file(GLOB tmp_list_cpu
    src/attention/*.cpp
    src/coalesce/*.cpp
    src/comm/*.cpp
    src/gather_scatter/*.cpp
    src/kv_cache/*.cpp)

  file(GLOB tmp_list_gpu
    src/attention/*.cu
    src/coalesce/*.cu
    src/comm/*.cu
    src/gather_scatter/*.cu
    src/kv_cache/*.cu)

  list(APPEND experimental_gen_ai_cpp_source_files_cpu ${tmp_list_cpu})
  list(APPEND experimental_gen_ai_cpp_source_files_gpu ${tmp_list_gpu})
endif()

# Set the source file for FB only CPP
if(USE_FB_ONLY AND (FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA))
  file(GLOB fb_only_sources_cpu
      fb/src/moe/*.cpp
      fb/src/trt_llm/*.cpp)

  file(GLOB fb_only_sources_gpu
      fb/src/moe/*.cu
      fb/src/trt_llm/*.cu)

  list(APPEND experimental_gen_ai_cpp_source_files_cpu ${fb_only_sources_cpu})
  list(APPEND experimental_gen_ai_cpp_source_files_gpu ${fb_only_sources_gpu})
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

# # Python sources
# file(GLOB_RECURSE experimental_gen_ai_python_source_files
#   bench/*.py
#   gen_ai/*.py
#   gen_ai/moe/*.py)


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
  CPU_SRCS
    ${experimental_gen_ai_cpp_source_files_cpu}
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
  TARGETS fbgemm_gpu_experimental_gen_ai)
  # FILES ${experimental_gen_ai_python_source_files})

install(
  DIRECTORY bench
  DESTINATION fbgemm_gpu/experimental)

install(
  DIRECTORY gen_ai
  DESTINATION fbgemm_gpu/experimental)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fb/gen_ai)
  install(
    DIRECTORY fb/gen_ai
    DESTINATION fbgemm_gpu/experimental)
endif()
