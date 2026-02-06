# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# PyTorch Dependencies Setup
################################################################################

find_package(Torch REQUIRED)

# Filter out specific flags that may have been inherited from PyTorch's
# CMAKE_CXX_FLAGS: - -Wno-duplicate-decl-specifier: C-only flag in GCC, valid
# for C++ in clang - -Wno-unused-command-line-argument: clang-specific,
# unrecognized by GCC
#
# Enhanced to strip flags from all CMake flag variables including configuration-
# specific ones (DEBUG, RELEASE, etc.) to prevent leakage through build variants
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  set(_clang_only_flags "-Wno-duplicate-decl-specifier"
                        "-Wno-unused-command-line-argument")

  # List of all CMake flag variables that might contain inherited flags
  set(_flag_variables
      CMAKE_CXX_FLAGS
      CMAKE_C_FLAGS
      CMAKE_CXX_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_DEBUG
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_RELWITHDEBINFO
      CMAKE_C_FLAGS_MINSIZEREL)

  # Strip clang-only flags from all variables
  foreach(_flag_var IN LISTS _flag_variables)
    foreach(_flag IN LISTS _clang_only_flags)
      if(DEFINED ${_flag_var})
        string(REPLACE "${_flag}" "" ${_flag_var} "${${_flag_var}}")
        # Also strip with leading/trailing spaces to handle various formats
        string(REPLACE " ${_flag} " " " ${_flag_var} "${${_flag_var}}")
        string(REGEX REPLACE "^${_flag} " "" ${_flag_var} "${${_flag_var}}")
        string(REGEX REPLACE " ${_flag}$" "" ${_flag_var} "${${_flag_var}}")
      endif()
    endforeach()
  endforeach()
  
  # Clean up extra spaces
  foreach(_flag_var IN LISTS _flag_variables)
    if(DEFINED ${_flag_var})
      string(REGEX REPLACE "  +" " " ${_flag_var} "${${_flag_var}}")
      string(STRIP "${${_flag_var}}" ${_flag_var})
    endif()
  endforeach()
endif()

#
# PyTorch CUDA Extensions are normally compiled with the flags below. However we
# disabled -D__CUDA_NO_HALF_CONVERSIONS__ here as it caused "error: no suitable
# constructor exists to convert from "int" to "__half" errors in
# gen_embedding_forward_quantized_split_[un]weighted_codegen_cuda.cu
#

set(TORCH_CUDA_OPTIONS
  --expt-relaxed-constexpr
  -D__CUDA_NO_HALF_OPERATORS__
  # -D__CUDA_NO_HALF_CONVERSIONS__
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__
  -D__CUDA_NO_HALF2_OPERATORS__)

BLOCK_PRINT(
  "PyTorch Flags:"
  " "
  "TORCH_INCLUDE_DIRS:"
  "${TORCH_INCLUDE_DIRS}"
  " "
  "TORCH_LIBRARIES:"
  "${TORCH_LIBRARIES}"
  " "
  "TORCH_CUDA_OPTIONS:"
  "${TORCH_CUDA_OPTIONS}"
)
