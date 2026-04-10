# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# PyTorch Dependencies Setup
################################################################################

find_package(Torch REQUIRED)

# Strip any -std=c++NN flags injected by PyTorch's find_package(Torch).
string(REGEX REPLACE "-std=c\\+\\+[0-9]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

foreach(_torch_target torch torch_cpu torch_cuda torch_hip)
  if(TARGET ${_torch_target})
    get_target_property(_torch_opts ${_torch_target} INTERFACE_COMPILE_OPTIONS)
    if(_torch_opts)
      # Remove standalone -std=c++NN list entries
      list(FILTER _torch_opts EXCLUDE REGEX "^-std=c\\+\\+[0-9]+$")
      # Also strip -std=c++NN embedded inside a compound string entry
      set(_torch_opts_cleaned "")
      foreach(_opt IN LISTS _torch_opts)
        string(REGEX REPLACE "-std=c\\+\\+[0-9]+" "" _opt "${_opt}")
        string(STRIP "${_opt}" _opt)
        if(_opt)
          list(APPEND _torch_opts_cleaned "${_opt}")
        endif()
      endforeach()
      set_target_properties(${_torch_target} PROPERTIES INTERFACE_COMPILE_OPTIONS "${_torch_opts_cleaned}")
    endif()
  endif()
endforeach()

# Filter out specific flags that may have been inherited from PyTorch's
# CMAKE_CXX_FLAGS: - -Wno-duplicate-decl-specifier: C-only flag in GCC, valid
# for C++ in clang - -Wno-unused-command-line-argument: clang-specific,
# unrecognized by GCC
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  set(clang_only_flags "-Wno-duplicate-decl-specifier"
                        "-Wno-unused-command-line-argument")

  foreach(_flag IN LISTS clang_only_flags)
    string(REPLACE "${_flag}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "${_flag}" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  endforeach()

  foreach(torch_target torch torch_cpu torch_cuda torch_hip)
    if(TARGET ${torch_target})
      get_target_property(options ${torch_target} INTERFACE_COMPILE_OPTIONS)
      if(options)
        foreach(_flag IN LISTS clang_only_flags)
          list(REMOVE_ITEM options "${_flag}")
        endforeach()
        set_target_properties(${torch_target} PROPERTIES INTERFACE_COMPILE_OPTIONS "${options}")
      endif()
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
