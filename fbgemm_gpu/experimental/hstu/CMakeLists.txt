# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

# Generate the CUDA source files
# Convert arch list to space-separated string for Python script argument
# NOTE: This line should be placed *before* the execute_process call.
string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST_STR "${TORCH_CUDA_ARCH_LIST}")

# Generate the CUDA source files and get the list of generated files
# Assumes generate_kernels.py accepts --arch-list and --print-files arguments,
# and prints a semicolon-separated list of relative paths (relative to WORKING_DIRECTORY) to stdout.
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" "generate_kernels.py"
          --arch-list="${TORCH_CUDA_ARCH_LIST_STR}" # Pass requested architectures
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
  RESULT_VARIABLE _resvar
  ERROR_VARIABLE _errvar
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY # Ensure build fails if script fails
)
# Check result (recommended)
if(NOT _resvar EQUAL 0)
  message(FATAL_ERROR "generate_kernels.py failed:\n${_errvar}")
endif()


# Collect HSTU Ampere source files
file(GLOB hstu_ampere_cpp_source_files
  src/hstu_ampere/*.cpp
  src/hstu_ampere/*.h)

# Collect HSTU Ampere GPU source files
file(GLOB hstu_ampere_cuda_source_files
  src/hstu_ampere/instantiations/*.cu)

# Collect HSTU Hopper source files
file(GLOB hstu_hopper_cpp_source_files
  src/hstu_hopper/*.cpp
  src/hstu_hopper/*.h)

# Collect HSTU Hopper GPU source files
file(GLOB hstu_hopper_cuda_source_files
  src/hstu_hopper/instantiations/*.cu)

# Initialize the combined source file lists
set(hstu_cpp_source_files "")
set(hstu_cpp_source_files_gpu "")

# Conditionally add Ampere sources if arch 8.0 is requested
# TORCH_CUDA_ARCH_LIST is expected to be set, e.g., by PyTorch's build system or CMake cache
if("8.0" IN_LIST TORCH_CUDA_ARCH_LIST)
  message(STATUS "HSTU: Selecting Ampere sources for arch 8.0")
  list(APPEND hstu_cpp_source_files ${hstu_ampere_cpp_source_files})
  list(APPEND hstu_cuda_source_files ${hstu_ampere_cuda_source_files})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
endif()

# Conditionally add Hopper sources if arch 9.0 is requested
if("9.0" IN_LIST TORCH_CUDA_ARCH_LIST OR "9.0a" IN_LIST TORCH_CUDA_ARCH_LIST)
  message(STATUS "HSTU: Selecting Hopper sources for arch 9.0")
  list(APPEND hstu_cpp_source_files ${hstu_hopper_cpp_source_files})
  list(APPEND hstu_cuda_source_files ${hstu_hopper_cuda_source_files})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90a,code=sm_90a")
endif()

# Check if any sources were selected
if(NOT hstu_cpp_source_files AND NOT hstu_cpp_source_files_gpu)
  message(WARNING "HSTU: Neither Ampere (8.0) nor Hopper (9.0) architectures found in TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}, would compile both")
  list(APPEND hstu_cpp_source_files ${hstu_ampere_cpp_source_files} ${hstu_hopper_cpp_source_files})
  list(APPEND hstu_cuda_source_files ${hstu_ampere_cuda_source_files} ${hstu_hopper_cuda_source_files})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90a,code=sm_90a")
endif()

# Add specific NVCC flags for HSTU compilation
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

################################################################################
# Build Shared Library
################################################################################

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_experimental_hstu
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hstu_ampere
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hstu_hopper
  CPU_SRCS
    ${hstu_cpp_source_files}
  GPU_SRCS
    ${hstu_cuda_source_files})

################################################################################
# Install Shared Library and Python Files
################################################################################

add_to_package(
  DESTINATION fbgemm_gpu/experimental/hstu
  TARGETS fbgemm_gpu_experimental_hstu)

install(
  DIRECTORY hstu
  DESTINATION fbgemm_gpu/experimental)
