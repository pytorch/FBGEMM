# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FBGEMM_HAVE_HIP FALSE)

if(NOT DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH /opt/rocm)
else()
  set(ROCM_PATH $ENV{ROCM_PATH})
endif()

macro(torch_hip_get_arch_list store_var)
  if(DEFINED ENV{PYTORCH_ROCM_ARCH})
    set(_TMP $ENV{PYTORCH_ROCM_ARCH})
  else()
    # Use arch of installed GPUs as default
    execute_process(COMMAND "rocm_agent_enumerator" COMMAND bash "-c" "grep -v gfx000 | sort -u | xargs | tr -d '\n'"
                    RESULT_VARIABLE ROCM_AGENT_ENUMERATOR_RESULT
                    OUTPUT_VARIABLE ROCM_ARCH_INSTALLED)
    if(NOT ROCM_AGENT_ENUMERATOR_RESULT EQUAL 0)
      message(FATAL_ERROR " Could not detect ROCm arch for GPUs on machine. Result: '${ROCM_AGENT_ENUMERATOR_RESULT}'")
    endif()
    set(_TMP ${ROCM_ARCH_INSTALLED})
  endif()
  string(REPLACE " " ";" ${store_var} "${_TMP}")
endmacro()

torch_hip_get_arch_list(PYTORCH_ROCM_ARCH)
if(PYTORCH_ROCM_ARCH STREQUAL "")
  message(FATAL_ERROR "No GPU arch specified for ROCm build. Please use PYTORCH_ROCM_ARCH environment variable to specify GPU archs to build for.")
endif()
message("Building FBGEMM for GPU arch: ${PYTORCH_ROCM_ARCH}")

ADD_DEFINITIONS(-DNDEBUG)
# USE_ROCM flag is used inside FBGEMM_GPU C++ code
ADD_DEFINITIONS(-DUSE_ROCM)

# Add HIP to the CMAKE Module Path
set(CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip ${CMAKE_MODULE_PATH})

macro(find_package_and_print_version PACKAGE_NAME)
  find_package("${PACKAGE_NAME}" ${ARGN})
  message("${PACKAGE_NAME} VERSION: ${${PACKAGE_NAME}_VERSION}")
endmacro()

# Find the HIP Package
find_package_and_print_version(HIP 1.0)

if(HIP_FOUND)
  set(FBGEMM_HAVE_HIP TRUE)
  set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}")

  # Enabling HIP language support
  enable_language(HIP)

  if(HIP_COMPILER STREQUAL clang)
    set(hip_library_name amdhip64)
  else()
    set(hip_library_name hip_hcc)
  endif()
  message("HIP library name: ${hip_library_name}")

  # TODO: hip_hcc has an interface include flag "-hc" which is only
  # recognizable by hcc, but not gcc and clang. Right now in our
  # setup, hcc is only used for linking, but it should be used to
  # compile the *_hip.cc files as well.
  find_library(FBGEMM_HIP_HCC_LIBRARIES ${hip_library_name} HINTS ${ROCM_PATH}/lib)

  list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_OPERATORS__=1)
  # list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
  list(APPEND HIP_CXX_FLAGS -D__HIP_NO_BFLOAT16_CONVERSIONS__=1)
  list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF2_OPERATORS__=1)
  list(APPEND HIP_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  list(APPEND HIP_CXX_FLAGS -mavx2)
  list(APPEND HIP_CXX_FLAGS -mf16c)
  list(APPEND HIP_CXX_FLAGS -mfma)
  list(APPEND HIP_CXX_FLAGS -std=c++20)
  list(APPEND HIP_CXX_FLAGS -g)
  list(APPEND HIP_CXX_FLAGS -ggdb)

 # list(APPEND HIP_CXX_FLAGS -Wa,-adhln)
  #list(APPEND HIP_CXX_FLAGS -adhln)
  list(APPEND HIP_CXX_FLAGS -save-temps)
  list(APPEND HIP_CXX_FLAGS -fverbose-asm)


  set(HIP_HCC_FLAGS ${HIP_CXX_FLAGS})
  # Ask hcc to generate device code during compilation so we can use
  # host linker to link.
  list(APPEND HIP_HCC_FLAGS -fno-gpu-rdc)
  list(APPEND HIP_HCC_FLAGS -Wno-defaulted-function-deleted)
  foreach(fbgemm_rocm_arch ${PYTORCH_ROCM_ARCH})
    list(APPEND HIP_HCC_FLAGS --offload-arch=${fbgemm_rocm_arch})
  endforeach()

  set(FBGEMM_HIP_INCLUDE ${ROCM_PATH}/include ${FBGEMM_HIP_INCLUDE})
  set(FBGEMM_HIP_INCLUDE ${hip_INCLUDE_DIRS} $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> $<INSTALL_INTERFACE:include> ${FBGEMM_HIP_INCLUDE})

  hip_include_directories(${FBGEMM_HIP_INCLUDE} ${ROCRAND_INCLUDE} ${ROCM_SMI_INCLUDE})

  list (APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH})
endif()
