# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake Prelude
################################################################################

# A minimum CMake version of 3.21+ is needed to support the required C and C++
# standard versions
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Set up paths
set(MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${MODULE_PATH}")
set(FBGEMM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FBGEMM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

include(${MODULE_PATH}/Utilities.cmake)

# C++ Compiler Setup - must be set BEFORE project declaration
include(${MODULE_PATH}/CxxCompilerSetup.cmake)

# Load cpp_library()
include(${MODULE_PATH}/CppLibrary.cmake)

# Define function to extract filelists from defs.bzl file
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "exec(open('defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar
    RESULT_VARIABLE _resvar
    ERROR_VARIABLE _errvar)
  if(NOT "${_resvar}" EQUAL "0")
    message(WARNING "Failed to execute Python (${Python_EXECUTABLE})\n"
      "Result: ${_resvar}\n"
      "Error: ${_errvar}\n")
  endif()
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()


################################################################################
# FBGEMM Build Kickstart
################################################################################

# Declare CMake project
project(
  fbgemm
  VERSION 1.3.0
  LANGUAGES CXX)

# Add C++ compiler flag detection
include(CheckCXXCompilerFlag)

# Install libraries into correct locations on all platforms
include(GNUInstallDirs)

# Check AVX
set(CHECK_AVX_COMPILE ON)
include(${MODULE_PATH}/FindAVX.cmake)
# For some reason, AVX flag values end up being packed into a single quoted
# string when FBGEMM is built as part of PyTorch build, so we need to separate
# the flags here back into list-string format
separate_arguments(CXX_AVX2_FLAGS)
separate_arguments(CXX_AVX512_FLAGS)

# Load Python
find_package(Python)

set(FBGEMM_LIBRARY_TYPE "default"
  CACHE STRING
  "Type of library (shared, static, or default) to build")

option(FBGEMM_BUILD_TESTS "Build fbgemm unit tests" ON)
option(FBGEMM_BUILD_BENCHMARKS "Build fbgemm benchmarks" ON)
option(FBGEMM_BUILD_DOCS "Build fbgemm documentation" OFF)
option(FBGEMM_USE_IPO "Build fbgemm with interprocedural optimization" OFF)
option(DISABLE_FBGEMM_AUTOVEC "Disable FBGEMM Autovec" OFF)

set_property(CACHE FBGEMM_LIBRARY_TYPE PROPERTY STRINGS DEFAULT STATIC SHARED)
string(TOUPPER "${FBGEMM_LIBRARY_TYPE}" FBGEMM_LIBRARY_TYPE)
if(FBGEMM_LIBRARY_TYPE STREQUAL "DEFAULT")
  if(BUILD_SHARED_LIB)
    set(FBGEMM_LIBRARY_TYPE "SHARED")
  else()
    set(FBGEMM_LIBRARY_TYPE "STATIC")
  endif()
endif()

# Add address sanitizer
set(FBGEMM_USE_SANITIZER "" CACHE STRING "options include address, leak, ...")

# We should default to a Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if(DISABLE_FBGEMM_AUTOVEC)
  add_compile_definitions("DISABLE_FBGEMM_AUTOVEC")
endif()

if(MSVC)
  set(MSVC_BOOL True)
else(MSVC)
  set(MSVC_BOOL False)
  check_cxx_compiler_flag(-march=armv8-a+sve COMPILER_SUPPORTS_SVE)
  check_cxx_compiler_flag(-march=armv8-a+sve2 COMPILER_SUPPORTS_SVE2)
  check_cxx_compiler_flag(-march=armv8-a+fp16fml COMPILER_SUPPORTS_FP16FML)
endif(MSVC)

################################################################################
# GNU F2H_IEEE Support
################################################################################

# Check if compiler supports __gnu_f2h_ieee
include(${CMAKE_CURRENT_LIST_DIR}/cmake/modules/FindGnuH2fIeee.cmake)

if(NOT HAVE_GNU_F2H_IEEE)
  add_definitions(-DMISSING_GNU_F2H_IEEE)
endif()

################################################################################
# Setup OpenMP Flags
################################################################################

find_package(OpenMP)

if(OpenMP_FOUND)
  message(STATUS "OpenMP found: OpenMP_C_INCLUDE_DIRS = ${OpenMP_C_INCLUDE_DIRS}")
else()
  message(WARNING "OpenMP is not supported by the compiler")
endif()

if(FBGEMM_USE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported()
endif()

################################################################################
# Prepare Sources
################################################################################

if(NOT MSVC)
  set_source_files_properties(
    src/FbgemmFP16UKernelsAvx2.cc
    src/FbgemmFP16UKernelsAvx512.cc
    src/FbgemmFP16UKernelsAvx512_256.cc
    src/fp32/FbgemmFP32UKernelsAvx2.cc
    src/fp32/FbgemmFP32UKernelsAvx512.cc
    src/fp32/FbgemmFP32UKernelsAvx512_256.cc
    PROPERTIES COMPILE_FLAGS "-masm=intel")

  # Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
  if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3.0))
    set_source_files_properties(
      src/GenerateKernelU8S8S32ACC32.cc
      src/GenerateKernelDirectConvU8S8S32ACC32.cc
      PROPERTIES COMPILE_FLAGS "-Wno-attributes")
  endif()
endif()

set(fbgemm_arm_defs)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
  set(fbgemm_arm_defs
      FBGEMM_FP16_FALLBACK_TO_REF_KERNEL
      FBGEMM_FP32_FALLBACK_TO_REF_KERNEL)
  if(NOT APPLE)
    list(APPEND fbgemm_arm_defs FBGEMM_ENABLE_KLEIDIAI)
  endif()
endif()

################################################################################
# FBGEMM Generic Target
################################################################################

get_filelist("get_fbgemm_generic_srcs(with_base=True)" FBGEMM_GENERIC_SRCS)
if(MSVC)
  list(FILTER FBGEMM_GENERIC_SRCS EXCLUDE REGEX "src/fp32/.*\\.cc$")
endif()

set(fbgemm_generic_defs "${fbgemm_arm_defs}")
if(FBGEMM_LIBRARY_TYPE STREQUAL STATIC)
  list(APPEND fbgemm_generic_defs FBGEMM_STATIC)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
  set(fbgemm_generic_defs ${fbgemm_generic_defs} FBGEMM_ENABLE_KLEIDIAI)
endif()

cpp_library(
  PREFIX
    fbgemm_generic
  TYPE
    OBJECT
  SRCS
    ${FBGEMM_GENERIC_SRCS}
  INCLUDE_DIRS
    $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
    $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/bench>
  MSVC_FLAGS
    /wd4717 /bigobj
  SANITIZER_OPTIONS
    "${FBGEMM_USE_SANITIZER}"
  DEFINITIONS
    ${fbgemm_generic_defs}
  DEPS
    $<BUILD_INTERFACE:cpuinfo>
    $<BUILD_INTERFACE:asmjit>
)

set(fbgemm_targets)

################################################################################
# FBGEMM AVX2 Target
################################################################################

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686")

  message(STATUS "Processor is ${CMAKE_SYSTEM_PROCESSOR}; will build AVX2 target")

  get_filelist("get_fbgemm_avx2_srcs(msvc=${MSVC_BOOL})" FBGEMM_AVX2_SRCS)
  get_filelist("get_fbgemm_inline_avx2_srcs(msvc=${MSVC_BOOL})" FBGEMM_AVX2_INLINE_SRCS)

  cpp_library(
    PREFIX
      fbgemm_avx2
    TYPE
      OBJECT
    SRCS
      ${FBGEMM_AVX2_SRCS}
      ${FBGEMM_AVX2_INLINE_SRCS}
    CC_FLAGS
      -m64 -mf16c -mfma ${CXX_AVX2_FLAGS}
    MSVC_FLAGS
      ${CXX_AVX2_FLAGS}
    DEPS
      $<BUILD_INTERFACE:cpuinfo>
      $<BUILD_INTERFACE:asmjit>
      fbgemm_generic
  )

  list(APPEND fbgemm_targets fbgemm_avx2)

else()
  message(STATUS "Processor is ${CMAKE_SYSTEM_PROCESSOR}; will NOT build AVX2 target")
endif()

################################################################################
# FBGEMM AVX512 Target
################################################################################

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i386|i686" AND CXX_AVX512_FOUND)
  message(STATUS "Processor is ${CMAKE_SYSTEM_PROCESSOR}; will build AVX512 target")

  get_filelist("get_fbgemm_avx512_srcs(msvc=${MSVC_BOOL})" FBGEMM_AVX512_SRCS)
  get_filelist("get_fbgemm_inline_avx512_srcs(msvc=${MSVC_BOOL})" FBGEMM_AVX512_INLINE_SRCS)

  cpp_library(
    PREFIX
      fbgemm_avx512
    TYPE
      OBJECT
    SRCS
      ${FBGEMM_AVX512_SRCS}
      ${FBGEMM_AVX512_INLINE_SRCS}
    CC_FLAGS
      -m64 -mfma ${CXX_AVX512_FLAGS} ${CXX_AVX2_FLAGS}
    MSVC_FLAGS
      ${CXX_AVX512_FLAGS} ${CXX_AVX2_FLAGS}
    DEPS
      $<BUILD_INTERFACE:cpuinfo>
      $<BUILD_INTERFACE:asmjit>
      fbgemm_generic
  )

  list(APPEND fbgemm_targets fbgemm_avx512)
else()
  message(STATUS "Processor is ${CMAKE_SYSTEM_PROCESSOR}; will NOT build AVX512 target")
endif()

################################################################################
# FBGEMM Neon Target
################################################################################

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
  if(NOT COMPILER_SUPPORTS_FP16FML)
    message(FATAL_ERROR "The current compiler does not support building FP16FML code")
  endif()

  message(STATUS "Processor is ${CMAKE_SYSTEM_PROCESSOR}; will build Neon target")

  get_filelist("get_fbgemm_inline_neon_srcs(msvc=${MSVC_BOOL})" FBGEMM_NEON_SRCS)

  cpp_library(
    PREFIX
      fbgemm_neon
    TYPE
      OBJECT
    SRCS
      ${FBGEMM_NEON_SRCS}
    CC_FLAGS
      -ftree-vectorize
      -fno-trapping-math
      -Wignored-qualifiers
      -march=armv8-a+fp16fml
    DEFINITIONS
      ${fbgemm_arm_defs}
    DEPS
      $<BUILD_INTERFACE:cpuinfo>
      $<BUILD_INTERFACE:asmjit>
      fbgemm_generic
  )

  list(APPEND fbgemm_targets fbgemm_neon)

else()
  message(STATUS "Processor is ${CMAKE_SYSTEM_PROCESSOR}; will NOT build Neon target")
endif()

################################################################################
# FBGEMM SVE Target
################################################################################

set(FBGEMM_BUILD_SVE False)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
  if(COMPILER_SUPPORTS_SVE2)
    set(FBGEMM_BUILD_SVE True)
    set(fbgemm_sve_flags "-march=armv8-a+sve2")
  elseif(COMPILER_SUPPORTS_SVE)
    set(FBGEMM_BUILD_SVE True)
    set(fbgemm_sve_flags "-march=armv8-a+sve")
  endif()
endif()

if(FBGEMM_BUILD_SVE)
  message(STATUS "Processor is ${CMAKE_SYSTEM_PROCESSOR}; will build SVE target")

  get_filelist("get_fbgemm_inline_sve_srcs(msvc=${MSVC_BOOL})" FBGEMM_SVE_SRCS)

  cpp_library(
    PREFIX
      fbgemm_sve
    TYPE
      OBJECT
    SRCS
      ${FBGEMM_SVE_SRCS}
    CC_FLAGS
      -ftree-vectorize
      -fno-trapping-math
      -Wignored-qualifiers
      -Wno-inline-asm
      -Wno-asm-operand-widths
      ${fbgemm_sve_flags}
    DEFINITIONS
      ${fbgemm_arm_defs}
    DEPS
      $<BUILD_INTERFACE:cpuinfo>
      $<BUILD_INTERFACE:asmjit>
      fbgemm_generic
  )

  list(APPEND fbgemm_targets fbgemm_sve)

else()
  message(STATUS "Processor is ${CMAKE_SYSTEM_PROCESSOR}; will NOT build SVE target")
endif()

################################################################################
# FBGEMM Autovec Target
################################################################################

if(NOT DISABLE_FBGEMM_AUTOVEC)
  message(STATUS "Will build autovec target")

  get_filelist("get_fbgemm_autovec_srcs()" FBGEMM_AUTOVEC_SRCS)

  cpp_library(
    PREFIX
      fbgemm_autovec
    TYPE
      OBJECT
    SRCS
      ${FBGEMM_AUTOVEC_SRCS}
    ENABLE_IPO
      ${FBGEMM_USE_IPO}
    DEPS
      $<BUILD_INTERFACE:cpuinfo>
      $<BUILD_INTERFACE:asmjit>
      fbgemm_generic
  )

  list(APPEND fbgemm_targets fbgemm_autovec)
endif()

################################################################################
# FBGEMM Library Target
################################################################################

cpp_library(
  PREFIX
    fbgemm
  TYPE
    ${FBGEMM_LIBRARY_TYPE}
  DEPS
    fbgemm_generic
    ${fbgemm_targets}
)

get_filelist("get_fbgemm_public_headers()" FBGEMM_PUBLIC_HEADERS)

install(
  FILES ${FBGEMM_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fbgemm")

install(
  EXPORT fbgemmLibraryConfig
  DESTINATION share/cmake/fbgemm
  FILE fbgemmLibraryConfig.cmake)

################################################################################
# Asmjit Target
################################################################################

if(NOT TARGET asmjit)
  # Download asmjit from github if ASMJIT_SRC_DIR is not specified.
  if(NOT DEFINED ASMJIT_SRC_DIR)
    set(ASMJIT_SRC_DIR "${FBGEMM_SOURCE_DIR}/external/asmjit"
      CACHE STRING "asmjit source directory from submodules")
  endif()

  # Build asmjit
  set(ASMJIT_STATIC ON)
  set(ASMJIT_NO_INSTALL ON)

  add_subdirectory("${ASMJIT_SRC_DIR}" "${FBGEMM_BINARY_DIR}/asmjit")
  set_property(TARGET asmjit PROPERTY POSITION_INDEPENDENT_CODE ON)
  if(MSVC)
    target_compile_options(asmjit PRIVATE /wd5054)
  endif()
endif()

################################################################################
# Cpuinfo Target
################################################################################

if(NOT TARGET cpuinfo)
  #Download cpuinfo from github if CPUINFO_SOURCE_DIR is not specified.
  if(NOT DEFINED CPUINFO_SOURCE_DIR)
    set(CPUINFO_SOURCE_DIR "${FBGEMM_SOURCE_DIR}/external/cpuinfo"
      CACHE STRING "cpuinfo source directory from submodules")
  endif()

  #build cpuinfo
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Do not build cpuinfo unit tests")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Do not build cpuinfo mock tests")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Do not build cpuinfo benchmarks")
  set(CPUINFO_LIBRARY_TYPE static CACHE STRING "Set lib type to static")
  add_subdirectory("${CPUINFO_SOURCE_DIR}" "${FBGEMM_BINARY_DIR}/cpuinfo")
  set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

################################################################################
# Optional Targets
################################################################################

if(FBGEMM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(FBGEMM_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(FBGEMM_BUILD_DOCS)
  add_subdirectory(docs)
endif()
