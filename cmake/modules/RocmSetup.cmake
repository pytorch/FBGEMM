# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# ROCm and HIPify Setup
################################################################################

if(FBGEMM_BUILD_VARIANT STREQUAL BUILD_VARIANT_ROCM)
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
    -fclang-abi-compat=17
    -Wno-cuda-compat
    -Wno-deprecated-declarations
    -Wno-format
    -Wno-ignored-attributes
    -Wno-unused-result)

  # is this hipify v2?
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "from torch.utils.hipify import __version__; print(__version__)"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar
    RESULT_VARIABLE _resvar
    ERROR_VARIABLE _errvar)
  if(NOT "${_resvar}" EQUAL "0")
    message(WARNING "Failed to execute Python (${Python_EXECUTABLE})\n"
      "Result: ${_resvar}\n"
      "Error: ${_errvar}\n")
  endif()
  string(FIND "${_tempvar}" "2" found_pos)
  if(found_pos GREATER_EQUAL 0)
    list(APPEND HIP_HCC_FLAGS -DHIPIFY_V2)
  endif()

  BLOCK_PRINT(
    "HIP found: ${HIP_FOUND}"
    "HIPCC compiler flags:"
    ""
    "${HIP_HCC_FLAGS}"
  )
endif()
