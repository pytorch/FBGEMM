# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

glob_files_nohip(experimental_gen_ai_cpp_source_files_cpu
  src/attention/*.cpp
  src/coalesce/*.cpp
  src/quantize/*.cpp)

glob_files_nohip(experimental_gen_ai_cpp_source_files_gpu
  src/attention/*.cu
  src/coalesce/*.cu
  src/quantize/*.cu)

if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  glob_files_nohip(tmp_list_cpu
    src/comm/*.cpp
    src/gather_scatter/*.cpp
    src/moe/*.cpp)

  glob_files_nohip(tmp_list_gpu
    src/comm/*.cu
    src/gather_scatter/*.cu
    src/moe/*.cu)

  # TODO: kv_cache sources need to be updated with proper CUDA_VERSION ifdef
  # wrapping before this if-clause can be removed
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    glob_files_nohip(tmp_list_cpu2
      src/kv_cache/*.cpp)

    glob_files_nohip(tmp_list_gpu2
      src/kv_cache/*.cu)
  endif()

  list(APPEND experimental_gen_ai_cpp_source_files_cpu ${tmp_list_cpu} ${tmp_list_cpu2})
  list(APPEND experimental_gen_ai_cpp_source_files_gpu ${tmp_list_gpu} ${tmp_list_gpu2})
endif()

# Set the source file for FB only CPP
if(USE_FB_ONLY
  AND (FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0))
  glob_files_nohip(fb_only_sources_cpu
      fb/src/*/*.cpp)

  glob_files_nohip(fb_only_sources_gpu
      fb/src/*/*.cu)

list(FILTER fb_only_sources_cpu
  EXCLUDE REGEX "fb/src/tensor_parallel(/.*)*\\.cpp$")

list(FILTER fb_only_sources_gpu
  EXCLUDE REGEX "fb/src/tensor_parallel(/.*)*\\.cu$")

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

# Filter out MoE sources for ROCm for now
list(FILTER experimental_gen_ai_cpp_source_files_hip
  EXCLUDE REGEX "src/quantize/ck_extensions/fused_moe(/.*)*\\.hip$")

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kv_cache
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
