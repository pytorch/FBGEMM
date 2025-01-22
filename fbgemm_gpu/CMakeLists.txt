# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake Prelude
################################################################################

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

set(CMAKEMODULES ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules)
set(FBGEMM_GPU ${CMAKE_CURRENT_SOURCE_DIR})
set(FBGEMM ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(THIRDPARTY ${FBGEMM}/external)

include(${CMAKEMODULES}/Utilities.cmake)

set(CMAKE_VERBOSE_MAKEFILE ON)

################################################################################
# FBGEMM_GPU Build Options
################################################################################

option(FBGEMM_CPU_ONLY   "Build FBGEMM_GPU without GPU support" OFF)
option(USE_ROCM          "Build FBGEMM_GPU for ROCm" OFF)
option(FBGEMM_GENAI_ONLY "Build FBGEMM_GPU with GEN AI only support" OFF)
option(USE_FB_ONLY       "Build FBGEMM_GPU FB only operators" OFF)

if((NOT FBGEMM_CPU_ONLY) AND
   ((EXISTS "/opt/rocm/") OR (EXISTS $ENV{ROCM_PATH})) AND
   (NOT EXISTS "/bin/nvcc"))
  message(
    "CMake has been set to build a non-CPU variant"
    "and AMD GPU has been detected; "
    "will default to ROCm build"
  )
  set(USE_ROCM ON)
endif()

if(FBGEMM_CPU_ONLY)
  BLOCK_PRINT("Building the CPU-only variant of FBGEMM-GPU")
elseif(USE_ROCM)
  BLOCK_PRINT("Building the ROCm variant of FBGEMM-GPU")
else()
  BLOCK_PRINT("Building the CUDA variant of FBGEMM-GPU")
endif()


################################################################################
# FBGEMM_GPU Build Kickstart
################################################################################

# FBGEMM_GPU C++ Setup - must be set BEFORE project declaration
include(${CMAKEMODULES}/CxxCompilerSetup.cmake)

if(SKBUILD)
  BLOCK_PRINT("The project is built using scikit-build")
endif()

if(FBGEMM_CPU_ONLY OR USE_ROCM)
  project(
    fbgemm_gpu
    VERSION 0.8.0
    LANGUAGES CXX C)
else()
  project(
    fbgemm_gpu
    VERSION 0.8.0
    LANGUAGES CXX C CUDA)
endif()

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
  if(USE_ROCM)
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

if(USE_ROCM)
  set(include_dirs_for_hipification
    # All directories need to be included for headers to be properly HIPified
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
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
# Build FBGEMM_GPU (Main) Module
################################################################################

if(NOT FBGEMM_GENAI_ONLY)
  include(FbgemmGpu.cmake)
endif()


################################################################################
# Build Experimental Modules
################################################################################

if(NOT FBGEMM_CPU_ONLY AND NOT USE_ROCM)
  add_subdirectory(experimental/example)
endif()

if(NOT FBGEMM_CPU_ONLY)
  # Package Triton GEMM (GenAI) kernels
  add_subdirectory(experimental/gemm)
endif()

if(NOT FBGEMM_CPU_ONLY AND NOT USE_ROCM)
  # TODO: Re-enable gen_ai for ROCm once ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp
  # lands into latest ROCm
  add_subdirectory(experimental/gen_ai)
endif()
