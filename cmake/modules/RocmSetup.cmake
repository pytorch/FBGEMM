# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules/Utilities.cmake)


################################################################################
# ROCm and HIPify Setup
################################################################################

if(USE_ROCM)
  # Load CMake modules
  list(APPEND CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/cmake"
    "${THIRDPARTY}/hipify_torch/cmake")
  include(Hip)
  include(Hipify)

  # Configure compiler for HIP
  list(APPEND HIP_HCC_FLAGS
    " \"-Wno-#pragma-messages\" "
    " \"-Wno-#warnings\" "
    -Wno-cuda-compat
    -Wno-deprecated-declarations
    -Wno-format
    -Wno-ignored-attributes
    -Wno-unused-result)

  BLOCK_PRINT(
    "HIP found: ${HIP_FOUND}"
    "HIPCC compiler flags:"
    ""
    "${HIP_HCC_FLAGS}"
  )
endif()
