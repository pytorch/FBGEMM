# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Target Sources
################################################################################

# Python sources
file(GLOB_RECURSE experimental_triton_gemm_python_source_files
  triton_gemm/*.py)


################################################################################
# Install Python Files
################################################################################

add_to_package(
  DESTINATION fbgemm_gpu/experimental/gemm/triton_gemm
  FILES ${experimental_triton_gemm_python_source_files})
