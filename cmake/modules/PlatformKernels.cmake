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
  "Platform kernel ISAs not to build, e.g. \"sve128;avx512\"; the platforms built on them are skipped too")
option(FBGEMM_DISABLE_MARCH_OVERRIDES
  "Don't add -march to the platform kernels' flags (set -march yourself)" OFF)

function(fbgemm_platform_kernels_isas outputvar)
  set(arch)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64|i386|i686")
    set(arch x86_64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(arch aarch64)
  endif()
  read_from_defs_bzl("get_fbgemm_platform_kernels_isas('${arch}', \
    disabled = [isa for isa in '${FBGEMM_DISABLE_PLATFORM_KERNELS}'.split(';') if isa])" isas)
  set(${outputvar} ${isas} PARENT_SCOPE)
endfunction()

# Creates one OBJECT library per ISA from fbgemm_platform_kernels_isas(),
# compiling get_fbgemm_platform_kernels_srcs() with FBGEMM_SIMD_<ISA> and that
# ISA's flags. PREPROCESSOR_MACROS_VAR receives the preprocessor macros that
# tell PlatformSpecificKernels.cc which platforms are built.
function(fbgemm_add_platform_kernels_libraries outputvar)
  cmake_parse_arguments(args "" "PREPROCESSOR_MACROS_VAR" "INCLUDE_DIRS;DEPS" ${ARGN})
  set(targets)
  read_from_defs_bzl("get_fbgemm_platform_kernels_srcs()" srcs)
  list(TRANSFORM srcs PREPEND "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../")
  fbgemm_platform_kernels_isas(isas)
  read_from_defs_bzl(
    "get_fbgemm_platform_kernels_defines('${isas}'.split(';'))" preprocessor_macros)
  set(msvc False)
  if(MSVC)
    set(msvc True)
  endif()
  set(disable_march_overrides False)
  if(FBGEMM_DISABLE_MARCH_OVERRIDES)
    set(disable_march_overrides True)
  endif()
  foreach(isa ${isas})
    read_from_defs_bzl(
      "get_fbgemm_platform_kernels_flags('${isa}', msvc = ${msvc}, disable_march_overrides = ${disable_march_overrides})"
      flags)
    string(TOUPPER "${isa}" isa_upper)
    cpp_library(
      PREFIX
        fbgemm_platform_kernels_${isa}
      TYPE
        OBJECT
      SRCS
        ${srcs}
      CC_FLAGS
        ${flags}
      MSVC_FLAGS
        ${flags}
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
  set(${args_PREPROCESSOR_MACROS_VAR} ${preprocessor_macros} PARENT_SCOPE)
endfunction()
