# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Asmjit Sources
################################################################################

file(GLOB_RECURSE asmjit_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/../external/asmjit/src/asmjit/*/*.cpp")


################################################################################
# Build Intermediate Target (Static)
################################################################################

gpu_cpp_library(
  PREFIX
    asmjit
  TYPE
    SHARED
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  OTHER_SRCS
    ${asmjit_sources}
  # asmjit is third-party code, so do not apply this warning to its sources.
  EXCLUDED_WARNING_FLAGS
    -Wunused-const-variable
  DESTINATION
    fbgemm_gpu)
