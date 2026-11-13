# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(fbgemm_platform_kernels_disabled_by_default)
if(WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
  set(fbgemm_platform_kernels_disabled_by_default neon sve128)
endif()
set(FBGEMM_DISABLE_PLATFORM_KERNELS
  "${fbgemm_platform_kernels_disabled_by_default}" CACHE STRING
  "Platform kernel ISAs not to build, e.g. \"sve128;avx512\"")

function(fbgemm_platform_kernels_flags isa outputvar)
  set(flags)
  if(isa STREQUAL "avx2")
    set(flags ${CXX_AVX2_FLAGS})
  elseif(isa STREQUAL "avx512")
    set(flags ${CXX_AVX512_FLAGS})
  else()
    get_filelist("get_fbgemm_platform_kernels_flags('${isa}')" flags)
  endif()
  set(${outputvar} ${flags} PARENT_SCOPE)
endfunction()

function(fbgemm_platform_kernels_isas outputvar)
  if("ref" IN_LIST FBGEMM_DISABLE_PLATFORM_KERNELS)
    message(FATAL_ERROR "The ref platform kernels cannot be disabled")
  endif()
  set(arch)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64|i386|i686")
    set(arch x86_64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(arch aarch64)
  endif()
  get_filelist("get_fbgemm_platform_kernels_isas('${arch}')" candidates)
  set(isas)
  foreach(isa ${candidates})
    if(isa IN_LIST FBGEMM_DISABLE_PLATFORM_KERNELS)
      message(STATUS "Not building ${isa} platform kernels")
      string(TOUPPER "${isa}" isa_upper)
      add_compile_definitions(FBGEMM_PLATFORM_KERNELS_DISABLE_${isa_upper})
    else()
      list(APPEND isas ${isa})
    endif()
  endforeach()
  set(${outputvar} ${isas} PARENT_SCOPE)
endfunction()

# Creates one OBJECT library per ISA from fbgemm_platform_kernels_isas(),
# compiling get_fbgemm_platform_kernels_srcs() with FBGEMM_SIMD_<ISA> and that
# ISA's flags.
function(fbgemm_add_platform_kernels_libraries outputvar)
  cmake_parse_arguments(args "" "" "INCLUDE_DIRS;DEPS" ${ARGN})
  set(targets)
  get_filelist("get_fbgemm_platform_kernels_srcs()" srcs)
  list(TRANSFORM srcs PREPEND "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../")
  fbgemm_platform_kernels_isas(isas)
  foreach(isa ${isas})
    fbgemm_platform_kernels_flags(${isa} isa_flags)
    string(TOUPPER "${isa}" isa_upper)
    cpp_library(
      PREFIX
        fbgemm_platform_kernels_${isa}
      TYPE
        OBJECT
      SRCS
        ${srcs}
      CC_FLAGS
        ${isa_flags}
      MSVC_FLAGS
        ${isa_flags}
      INCLUDE_DIRS
        ${args_INCLUDE_DIRS}
      DEPS
        ${args_DEPS}
    )
    target_compile_definitions(fbgemm_platform_kernels_${isa}
      PRIVATE FBGEMM_SIMD_${isa_upper})
    list(APPEND targets fbgemm_platform_kernels_${isa})
  endforeach()
  set(${outputvar} ${targets} PARENT_SCOPE)
endfunction()
