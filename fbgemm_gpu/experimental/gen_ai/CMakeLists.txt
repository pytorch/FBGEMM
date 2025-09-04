# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake Prelude
################################################################################

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(CMAKEMODULES ${CMAKE_CURRENT_LIST_DIR}/../../../cmake/modules)
include(${CMAKEMODULES}/Utilities.cmake)
include(${CMAKEMODULES}/GpuCppLibrary.cmake)

# CUDA Setup
include(${CMAKEMODULES}/CudaSetup.cmake)

# ROCm and HIPify Setup
include(${CMAKEMODULES}/RocmSetup.cmake)

set(CMAKE_VERBOSE_MAKEFILE ON)

################################################################################
# Target Sources
################################################################################

glob_files_nohip(experimental_gen_ai_cpp_source_files_cpu
  src/attention/cuda/*.cpp
  src/coalesce/*.cpp
  src/comm/*.cpp
  src/kv_cache/*.cpp
  src/moe/*.cpp
  src/quantize/*.cpp
  src/quantize/common/*.cpp)

glob_files_nohip(experimental_gen_ai_cpp_source_files_gpu
  src/attention/cuda/*.cu
  src/attention/cuda/cutlass_blackwell_fmha/blackwell_*.cu
  src/coalesce/*.cu
  src/comm/*.cu
  src/kv_cache/*.cu
  src/moe/*.cu
  src/quantize/*.cu)

if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  glob_files_nohip(tmp_list_cpu
    src/gather_scatter/*.cpp)

  glob_files_nohip(tmp_list_gpu
    src/gather_scatter/*.cu)

  list(APPEND experimental_gen_ai_cpp_source_files_cpu
    ${tmp_list_cpu} ${tmp_list_cpu2})
  list(APPEND experimental_gen_ai_cpp_source_files_gpu
    ${tmp_list_gpu} ${tmp_list_gpu2})
endif()

# Include FB-internal code into the build
if(USE_FB_ONLY
  AND (FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  AND (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0))
  BLOCK_PRINT(
    "[FBPKG] USE_FB_ONLY is set, "
    "will be INCLUDING FB-internal sources.")

  glob_files_nohip(fb_only_sources_cpu
      fb/src/*/*.cpp)

  glob_files_nohip(fb_only_sources_gpu
      fb/src/*/*.cu)

  if(FBGEMM_FBPKG_BUILD)
    BLOCK_PRINT("[FBPKG] FBGEMM_FBPKG_BUILD is set.")

  else()
    BLOCK_PRINT(
      "[FBPKG] FBGEMM_FBPKG_BUILD is NOT set,"
      "certain FB-internal sources will be excluded from the build.")

    # NOTE: Some FB-internal code explicitly require an FB-internal
    # environment to build, such as code that depends on NCCLX
    list(FILTER fb_only_sources_cpu
      EXCLUDE REGEX "fb/src/tensor_parallel(/.*)*\\.cpp$")

    list(FILTER fb_only_sources_gpu
      EXCLUDE REGEX "fb/src/tensor_parallel(/.*)*\\.cu$")
  endif()

  list(APPEND experimental_gen_ai_cpp_source_files_cpu ${fb_only_sources_cpu})
  list(APPEND experimental_gen_ai_cpp_source_files_gpu ${fb_only_sources_gpu})

else()
  BLOCK_PRINT(
    "[FBPKG] USE_FB_ONLY is NOT set, "
    "will be EXCLUDING FB-internal sources.")
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

# Customizations for PyTorch build integration
IF(DEFINED CAFFE2_THIRD_PARTY_ROOT)
  # Remove "experimental" from the library name
  set(FBGEMM_GENAI_LIB_NAME fbgemm_gpu_gen_ai)
  # Keep lib prefix
  set(KEEP_PREFIX ON)
  # Install the library in the build directory
  set(DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
  set(FBGEMM_GENAI_LIB_NAME fbgemm_gpu_experimental_gen_ai)
  set(KEEP_PREFIX OFF)
  set(DESTINATION "")
endif()

gpu_cpp_library(
  PREFIX
    ${FBGEMM_GENAI_LIB_NAME}
  TYPE
    SHARED
  DESTINATION
    ${DESTINATION}
  KEEP_PREFIX
    ${KEEP_PREFIX}
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/attention/cuda/cutlass_blackwell_fmha
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantize
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantize/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quantize/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kv_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${FBGEMM_GENAI_INCLUDE_DIRS}
  CPU_SRCS
    ${experimental_gen_ai_cpp_source_files_cpu}
  GPU_SRCS
    ${experimental_gen_ai_cpp_source_files_gpu}
  CUDA_SPECIFIC_SRCS
    ${experimental_gen_ai_cpp_source_files_cuda}
  HIP_SPECIFIC_SRCS
    ${experimental_gen_ai_cpp_source_files_hip}
  TORCH_LIBS
    # Used when building as part of PyTorch
    ${FBGEMM_GENAI_TORCH_LIBS}
  HIPCC_FLAGS
    # Below flags are required for strong CK performance
    # on certain kernel instances
    -mllvm
    # Reduce register spillage on certain kernels
    -amdgpu-coerce-illegal-types=1
    -mllvm
    -enable-post-misched=0
    -mllvm
    -greedy-reverse-local-assignment=1
    -fhip-new-launch-api)



################################################################################
# Install Shared Library and Python Files
################################################################################

add_to_package(
  DESTINATION fbgemm_gpu/experimental/gen_ai
  TARGETS ${FBGEMM_GENAI_LIB_NAME})

install(
  DIRECTORY bench
  DESTINATION fbgemm_gpu/experimental/gen_ai)

install(
  DIRECTORY gen_ai
  DESTINATION fbgemm_gpu/experimental)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/fb/gen_ai)
  install(
    DIRECTORY fb/gen_ai
    DESTINATION fbgemm_gpu/experimental)
endif()
