# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake Prelude
################################################################################

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(CMAKEMODULES ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules)
set(FBGEMM_GPU ${CMAKE_CURRENT_SOURCE_DIR})
set(FBGEMM ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(THIRDPARTY ${FBGEMM}/external)

include(${CMAKEMODULES}/Utilities.cmake)

set(CMAKE_VERBOSE_MAKEFILE ON)

################################################################################
# Set Build Target
################################################################################

set(BUILD_TARGET_DEFAULT  "default")
set(BUILD_TARGET_GENAI    "genai")
set(BUILD_TARGET_VALUES   "${BUILD_TARGET_DEFAULT};${BUILD_TARGET_GENAI}")

if(NOT DEFINED FBGEMM_BUILD_TARGET)
  set(FBGEMM_BUILD_TARGET "${BUILD_TARGET_DEFAULT}")
elseif(NOT FBGEMM_BUILD_TARGET IN_LIST BUILD_TARGET_VALUES)
  message(FATAL_ERROR
    "Invalid FBGEMM_BUILD_TARGET value: ${FBGEMM_BUILD_TARGET}.
    Allowed values: ${BUILD_TARGET_VALUES}")
endif()

################################################################################
# Set Build Variant
################################################################################

set(BUILD_VARIANT_CPU     "cpu")
set(BUILD_VARIANT_CUDA    "cuda")
set(BUILD_VARIANT_ROCM    "rocm")
set(BUILD_VARIANT_VALUES
  "${BUILD_VARIANT_CPU};${BUILD_VARIANT_CUDA};${BUILD_VARIANT_ROCM}")

if (DEFINED FBGEMM_BUILD_VARIANT)
  # If FBGEMM_BUILD_VARIANT is set, validate it
  if(NOT FBGEMM_BUILD_VARIANT IN_LIST BUILD_VARIANT_VALUES)
    message(FATAL_ERROR
      "Invalid FBGEMM_BUILD_VARIANT value: ${FBGEMM_BUILD_VARIANT}.
      Allowed values: ${BUILD_VARIANT_VALUES}")
  endif()

elseif(((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) AND
  (NOT EXISTS "/bin/nvcc"))
  message(
    "AMD GPU has been detected; will default to ROCm build"
  )
  set(FBGEMM_BUILD_VARIANT "${BUILD_VARIANT_ROCM}")

else()
  set(FBGEMM_BUILD_VARIANT "${BUILD_VARIANT_CUDA}")

endif()

################################################################################
# FBGEMM_GPU Build Kickstart
################################################################################

# FBGEMM_GPU C++ Setup - must be set AFTER FBGEMM_BUILD_VARIANT declaration but
# BEFORE project declaration
include(${CMAKEMODULES}/CxxCompilerSetup.cmake)

if(SKBUILD)
  BLOCK_PRINT("The project is built using scikit-build")
endif()

BLOCK_PRINT(
  "Build Settings"
  ""
  "FBGEMM_BUILD_TARGET    : ${FBGEMM_BUILD_TARGET}"
  "FBGEMM_BUILD_VARIANT   : ${FBGEMM_BUILD_VARIANT}"
  ""
  "NVCC_VERBOSE           : ${NVCC_VERBOSE}"
  "CUDNN_INCLUDE_DIR      : ${CUDNN_INCLUDE_DIR}"
  "CUDNN_LIBRARY          : ${CUDNN_LIBRARY}"
  "NVML_LIB_PATH          : ${NVML_LIB_PATH}"
  "TORCH_CUDA_ARCH_LIST   : ${TORCH_CUDA_ARCH_LIST}"
  ""
  "HIP_ROOT_DIR           : ${HIP_ROOT_DIR}"
  "HIPCC_VERBOSE          : ${HIPCC_VERBOSE}"
  "AMDGPU_TARGETS         : ${AMDGPU_TARGETS}"
  "PYTORCH_ROCM_ARCH      : ${PYTORCH_ROCM_ARCH}")

set(project_languages CXX C)
if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  list(APPEND project_languages CUDA)
endif()

# Declare CMake project
project(
  fbgemm_gpu
  VERSION 1.2.0
  LANGUAGES ${project_languages})

# AVX Flags Setup - must be set AFTER project declaration
include(${CMAKEMODULES}/FindAVX.cmake)

# PyTorch Dependencies Setup
include(${CMAKEMODULES}/PyTorchSetup.cmake)

# CUDA Setup
include(${CMAKEMODULES}/CudaSetup.cmake)

# ROCm and HIPify Setup
include(${CMAKEMODULES}/RocmSetup.cmake)

# Load gpu_cpp_library()
include(${CMAKEMODULES}/GpuCppLibrary.cmake)


################################################################################
# Source Includes
################################################################################

set(fbgemm_sources_include_directories
  # FBGEMM
  ${FBGEMM}/include
  # FBGEMM_GPU
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  # PyTorch
  ${TORCH_INCLUDE_DIRS}
  # Third-party
  ${THIRDPARTY}/asmjit/src
  ${THIRDPARTY}/cpuinfo/include
  ${THIRDPARTY}/cutlass/include
  ${THIRDPARTY}/cutlass/tools/util/include
  ${THIRDPARTY}/composable_kernel/include
  ${THIRDPARTY}/composable_kernel/library/include
  ${THIRDPARTY}/json/include
  ${NCCL_INCLUDE_DIRS})


################################################################################
# Build Library Dependencies
################################################################################

# These dependencies should be declared and built before building FBGEMM_GPU

# Target: `asmjit`
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Asmjit.cmake)

# Target: `fbgemm`
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Fbgemm.cmake)


################################################################################
# TBE Code Generation
################################################################################

set(CMAKE_CODEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/codegen)

macro(RUN_GEN_SCRIPT SCRIPT)
  if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
    set(rocm_flag --is_rocm)
  endif()

  BLOCK_PRINT(
    "Running code generation script ..."
    "${PYTHON_EXECUTABLE} ${SCRIPT} --opensource ${rocm_flag}"
  )

  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" ${SCRIPT} "--opensource" ${rocm_flag})
endmacro()

foreach(script
    "${CMAKE_CODEGEN_DIR}/genscript/generate_backward_split.py"
    "${CMAKE_CODEGEN_DIR}/genscript/generate_embedding_optimizer.py"
    "${CMAKE_CODEGEN_DIR}/genscript/generate_forward_quantized.py"
    "${CMAKE_CODEGEN_DIR}/genscript/generate_forward_split.py"
    "${CMAKE_CODEGEN_DIR}/genscript/generate_index_select.py")
    RUN_GEN_SCRIPT(${script})
endforeach()


################################################################################
# HIP Code Generation
################################################################################

if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
  set(include_dirs_for_hipification
    # All directories need to be included for headers to be properly HIPified
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental/example
    ${CMAKE_CURRENT_SOURCE_DIR}/experimental/gen_ai)

  # HIPify all .CU and .CUH sources under the current directory (`/fbgemm_gpu`)
  #
  # Note that .H sources are not automatically HIPified, so if they reference
  # CUDA-specific code, e.g. `#include <c10/cuda/CUDAStream.h>`, they will need
  # to be updated with `#ifdef USE_ROCM` guards.
  hipify(
    CUDA_SOURCE_DIR
      ${PROJECT_SOURCE_DIR}
    HEADER_INCLUDE_DIR
      ${include_dirs_for_hipification})

  BLOCK_PRINT(
    "HIPify Sources"
    " "
    "CUDA_SOURCE_DIR:"
    "${PROJECT_SOURCE_DIR}"
    " "
    "HEADER_INCLUDE_DIR:"
    "${include_dirs_for_hipification}"
  )
endif()

function(get_tbe_sources_list variable_name)
  # Set the genfiles script path
  set(genfiles_script ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tbe_sources.py)

  # Execute the script to load the generated files list to memory,
  # and print the desired variable to stdout
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}"
      -c
      "exec(open('${genfiles_script}').read()); print(';'.join(${variable_name}))"
    WORKING_DIRECTORY
      "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE
      tempvar
    RESULT_VARIABLE
      resvar
    ERROR_VARIABLE
      errvar)

  # Exit if executing the script fails
  if(NOT "${resvar}" EQUAL "0")
    message(ERROR
      "Failed to execute Python (${PYTHON_EXECUTABLE})\n"
      "Result: ${resvar}\n"
      "Error: ${errvar}\n")
  endif()

  # Clean out the string
  string(REPLACE "\n" "" tempvar "${tempvar}")

  # Set the output variable
  set(${variable_name} ${tempvar} PARENT_SCOPE)
endfunction()


################################################################################
# Build Targets
################################################################################

if(FBGEMM_BUILD_TARGET STREQUAL BUILD_TARGET_GENAI)
  if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CPU)
    message(FATAL_ERROR
      "Unsupported (target, variant) combination:
      (${FBGEMM_BUILD_TARGET}, ${FBGEMM_BUILD_VARIANT})")
  endif()

  if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
    # Add experimental packaging example
    add_subdirectory(experimental/example)
  endif()

  # Build FBGEMM GenAI
  add_subdirectory(experimental/gen_ai)

  # Add Triton GEMM kernels
  add_subdirectory(experimental/gemm)

elseif(FBGEMM_BUILD_TARGET STREQUAL BUILD_TARGET_DEFAULT)
  # Build FBGEMM_GPU
  include(FbgemmGpu.cmake)
endif()
