# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${CMAKEMODULES}/Utilities.cmake)

################################################################################
# Target Sources
################################################################################

set(fbgemm_sources_include_directories
  # FBGEMM
  ${FBGEMM}/include
  # FBGEMM_GPU
  ${CMAKE_CURRENT_SOURCE_DIR}../..
  ${CMAKE_CURRENT_SOURCE_DIR}../../include
  ${CMAKE_CURRENT_SOURCE_DIR}../../../include
  # PyTorch
  ${TORCH_INCLUDE_DIRS}
  # Third-party
  ${THIRDPARTY}/asmjit/src
  ${THIRDPARTY}/cpuinfo/include
  ${THIRDPARTY}/cutlass/include
  ${THIRDPARTY}/cutlass/tools/util/include
  ${NCCL_INCLUDE_DIRS})

set(experimental_example_cpp_source_files
    src/cutlass_sgemm_nn.cu
    src/example_ops.cpp
    src/nccl_example.cpp)

set_source_files_properties(${experimental_example_cpp_source_files}
    PROPERTIES INCLUDE_DIRECTORIES
    "${fbgemm_sources_include_directories}")

set(experimental_example_python_source_files
    example/__init__.py
    example/utils.py)


################################################################################
# Build Shared Library
################################################################################

add_library(fbgemm_gpu_experimental_example_py MODULE
    ${experimental_example_cpp_source_files})

target_include_directories(fbgemm_gpu_experimental_example_py PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIRS})

target_link_libraries(fbgemm_gpu_experimental_example_py
    ${TORCH_LIBRARIES}
    ${NCCL_LIBRARIES}
    ${CUDA_DRIVER_LIBRARIES})

# Remove `lib` from the output artifact name
set_target_properties(fbgemm_gpu_experimental_example_py PROPERTIES PREFIX "")


################################################################################
# Install Shared Library and Python Files
################################################################################

install(TARGETS fbgemm_gpu_experimental_example_py
        DESTINATION fbgemm_gpu/experimental/example)

install(FILES ${experimental_example_python_source_files}
        DESTINATION fbgemm_gpu/experimental/example)
