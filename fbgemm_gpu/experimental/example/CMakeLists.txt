# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

file(GLOB_RECURSE experimental_example_cpp_source_files
  src/*.cu
  src/*.cpp)

file(GLOB_RECURSE experimental_example_python_source_files
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
