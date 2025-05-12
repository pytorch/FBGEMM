# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

glob_files_nohip(experimental_example_cpp_source_files
  src/*.cpp)

if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_CUDA)
  # Only include .CU files if CUDA is enabled bc they use CUTLASS and HIPify
  # isn't able to handle them
  glob_files_nohip(tmp_files
    src/*.cu)

  list(APPEND experimental_example_cpp_source_files ${tmp_files})
endif()

glob_files_nohip(experimental_example_python_source_files
  example/*.py)


################################################################################
# Build Shared Library
################################################################################

gpu_cpp_library(
  PREFIX
    fbgemm_gpu_experimental_example_py
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  GPU_SRCS
    ${experimental_example_cpp_source_files})


################################################################################
# Install Shared Library and Python Files
################################################################################

add_to_package(DESTINATION fbgemm_gpu/experimental/example
  TARGETS fbgemm_gpu_experimental_example_py
  FILES ${experimental_example_python_source_files})
