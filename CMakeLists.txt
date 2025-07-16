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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

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
# FBGEMM C++ Setup
################################################################################

# Set the default C++ standard to C++20
# Individual targets can have this value overridden; see
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Set the default C standard to C17
# Individual targets can have this value overridden; see
# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Check if given flag is supported and append it to provided outputvar
# Also define HAS_UPPER_CASE_FLAG_NAME variable
# From: https://github.com/pytorch/pytorch/blob/62c8d30f9f6715d0b60d78fb5f5913a2f3bd185b/cmake/public/utils.cmake#L579
# Usage:
#   append_cxx_flag_if_supported("-Werror" CMAKE_CXX_FLAGS)
function(append_cxx_flag_if_supported flag outputvar)
    string(TOUPPER "HAS${flag}" _FLAG_NAME)
    string(REGEX REPLACE "[=-]" "_" _FLAG_NAME "${_FLAG_NAME}")

    # GCC silents unknown -Wno-XXX flags, so we detect the corresponding -WXXX.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      string(REGEX REPLACE "Wno-" "W" new_flag "${flag}")
    else()
      set(new_flag ${flag})
    endif()

    check_cxx_compiler_flag("${new_flag}" ${_FLAG_NAME})
    if(${_FLAG_NAME})
        string(APPEND ${outputvar} " ${flag}")
        set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
    endif()
endfunction()

function(target_compile_options_if_supported target flag)
  set(_compile_options "")
  append_cxx_flag_if_supported("${flag}" _compile_options)
  if(NOT "${_compile_options}" STREQUAL "")
    target_compile_options(${target} PRIVATE ${flag})
  endif()
endfunction()


################################################################################
# FBGEMM Build Kickstart
################################################################################

project(fbgemm VERSION 0.1 LANGUAGES CXX C)

# Install libraries into correct locations on all platforms
include(GNUInstallDirs)

# Load Python
find_package(Python)

set(FBGEMM_LIBRARY_TYPE "default"
  CACHE STRING
  "Type of library (shared, static, or default) to build")

set_property(CACHE FBGEMM_LIBRARY_TYPE PROPERTY STRINGS default static shared)
option(FBGEMM_BUILD_TESTS "Build fbgemm unit tests" ON)
option(FBGEMM_BUILD_BENCHMARKS "Build fbgemm benchmarks" ON)
option(FBGEMM_BUILD_DOCS "Build fbgemm documentation" OFF)
option(FBGEMM_BUILD_FBGEMM_GPU "Build fbgemm_gpu library" OFF)
option(FBGEMM_USE_IPO "Build fbgemm with interprocedural optimization" OFF)
option(DISABLE_FBGEMM_AUTOVEC "Disable FBGEMM Autovec" OFF)


if(FBGEMM_LIBRARY_TYPE STREQUAL "default")
  if(BUILD_SHARED_LIB)
    set(FBGEMM_LIBRARY_TYPE "shared")
  else()
    set(FBGEMM_LIBRARY_TYPE "static")
  endif()
endif()

if(FBGEMM_BUILD_TESTS)
  enable_testing()
endif()

set(FBGEMM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FBGEMM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FBGEMM_THIRDPARTY_DIR ${FBGEMM_BINARY_DIR}/external)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add address sanitizer
set(USE_SANITIZER "" CACHE STRING "options include address, leak, ...")

# Check if compiler supports AVX512
include(CheckCXXCompilerFlag)
if(MSVC)
  CHECK_CXX_COMPILER_FLAG(/arch:AVX512 COMPILER_SUPPORTS_AVX512)
  set(MSVC_BOOL True)
else(MSVC)
  CHECK_CXX_COMPILER_FLAG(-mavx512f COMPILER_SUPPORTS_AVX512)
  set(MSVC_BOOL False)
endif(MSVC)
if(NOT COMPILER_SUPPORTS_AVX512)
  message(FATAL_ERROR "A compiler with AVX512 support is required.")
endif()

# Check if compiler supports __gnu_f2h_ieee
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/FindGnuH2fIeee.cmake)

# We should default to a Release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if(DISABLE_FBGEMM_AUTOVEC)
  add_compile_definitions("DISABLE_FBGEMM_AUTOVEC")
endif()

# Check if compiler supports OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
  message(STATUS "OpenMP found! OpenMP_C_INCLUDE_DIRS = ${OpenMP_C_INCLUDE_DIRS}")
else()
  message(WARNING "OpenMP is not supported by the compiler")
endif()

# Define file lists
get_filelist("get_fbgemm_generic_srcs(with_base=True)" FBGEMM_GENERIC_SRCS)
get_filelist("get_fbgemm_avx2_srcs(msvc=${MSVC_BOOL})" FBGEMM_AVX2_SRCS)
get_filelist("get_fbgemm_inline_avx2_srcs(msvc=${MSVC_BOOL})"
  FBGEMM_AVX2_INLINE_SRCS)
get_filelist("get_fbgemm_avx512_srcs(msvc=${MSVC_BOOL})" FBGEMM_AVX512_SRCS)
get_filelist("get_fbgemm_inline_avx512_srcs(msvc=${MSVC_BOOL})"
  FBGEMM_AVX512_INLINE_SRCS)
if(NOT DISABLE_FBGEMM_AUTOVEC)
  get_filelist("get_fbgemm_autovec_srcs()" FBGEMM_AUTOVEC_SRCS)
endif()
get_filelist("get_fbgemm_public_headers()" FBGEMM_PUBLIC_HEADERS)

set(fbgemm_targets fbgemm_avx2 fbgemm_avx512)
add_library(fbgemm_generic OBJECT ${FBGEMM_GENERIC_SRCS})
add_library(fbgemm_avx2 OBJECT ${FBGEMM_AVX2_SRCS} ${FBGEMM_AVX2_INLINE_SRCS})
add_library(fbgemm_avx512 OBJECT
  ${FBGEMM_AVX512_SRCS} ${FBGEMM_AVX512_INLINE_SRCS})
if(NOT DISABLE_FBGEMM_AUTOVEC)
  add_library(fbgemm_autovec OBJECT ${FBGEMM_AUTOVEC_SRCS})
  list(APPEND fbgemm_targets fbgemm_autovec)
endif()


# Make libraries depend on defs.bzl
add_custom_target(defs.bzl DEPENDS defs.bzl)
add_dependencies(fbgemm_generic defs.bzl)
foreach(target IN LISTS fbgemm_targets)
  target_link_libraries(${target} PUBLIC fbgemm_generic)
endforeach()

target_include_directories(fbgemm_generic PUBLIC
      $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
      $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/bench>)

target_link_libraries(fbgemm_generic PUBLIC
  $<BUILD_INTERFACE:asmjit>
  $<BUILD_INTERFACE:cpuinfo>)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4305 /wd4309")
  target_compile_options(fbgemm_avx2 PRIVATE "/arch:AVX2")
  target_compile_options(fbgemm_avx512 PRIVATE "/arch:AVX512")
else(MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -Wall")
  string(APPEND CMAKE_CXX_FLAGS " -Wextra")
  string(APPEND CMAKE_CXX_FLAGS " -Werror")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-declarations")
  string(APPEND CMAKE_CXX_FLAGS " -Wunknown-pragmas")
  string(APPEND CMAKE_CXX_FLAGS " -Wimplicit-fallthrough")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-aliasing")
  string(APPEND CMAKE_CXX_FLAGS " -Wunused-variable")
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 17.0.0)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-vla-cxx-extension")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(APPEND CMAKE_CXX_FLAGS " -Wmaybe-uninitialized")
  endif()
  target_compile_options(fbgemm_avx2 PRIVATE
    "-m64" "-mavx2" "-mf16c" "-mfma")
  target_compile_options(fbgemm_avx512 PRIVATE
    "-m64" "-mavx2" "-mfma" "-mavx512f" "-mavx512bw" "-mavx512dq"
    "-mavx512vl")
  set_source_files_properties(
    src/FbgemmFP16UKernelsAvx2.cc
    src/FbgemmFP16UKernelsAvx512.cc
    src/FbgemmFP16UKernelsAvx512_256.cc
    PROPERTIES COMPILE_FLAGS "-masm=intel")
  set_source_files_properties(src/PackMatrix.cc
    PROPERTIES COMPILE_FLAGS "-Wno-infinite-recursion")
  # Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80947
  if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.3.0))
    set_source_files_properties(
      src/GenerateKernelU8S8S32ACC32.cc
      src/GenerateKernelDirectConvU8S8S32ACC32.cc
      PROPERTIES COMPILE_FLAGS "-Wno-attributes")
  endif()

endif(MSVC)

if(USE_SANITIZER)
  target_link_options(fbgemm_generic PUBLIC
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
  target_compile_options(fbgemm_generic PUBLIC
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
endif()

message(STATUS "==========")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "==========")

if(NOT TARGET asmjit)
  # Download asmjit from github if ASMJIT_SRC_DIR is not specified.
  if(NOT DEFINED ASMJIT_SRC_DIR)
    set(ASMJIT_SRC_DIR "${FBGEMM_SOURCE_DIR}/external/asmjit"
      CACHE STRING "asmjit source directory from submodules")
  endif()

  # Build asmjit
  # For MSVC shared build, asmjit needs to be shared also.
  if(MSVC AND (FBGEMM_LIBRARY_TYPE STREQUAL "shared"))
    set(ASMJIT_STATIC OFF)
  else()
    set(ASMJIT_STATIC ON)
  endif()
  set(ASMJIT_NO_INSTALL ON)

  add_subdirectory("${ASMJIT_SRC_DIR}" "${FBGEMM_BINARY_DIR}/asmjit")
  set_property(TARGET asmjit PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

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

file(WRITE "${CMAKE_BINARY_DIR}/fbgemm_lib_dummy.cc" "")
if(FBGEMM_LIBRARY_TYPE STREQUAL "shared")
  add_library(fbgemm SHARED "${CMAKE_BINARY_DIR}/fbgemm_lib_dummy.cc")
elseif(FBGEMM_LIBRARY_TYPE STREQUAL "static")
  add_library(fbgemm STATIC "${CMAKE_BINARY_DIR}/fbgemm_lib_dummy.cc")
  #MSVC need to define FBGEMM_STATIC for fbgemm_generic also to
  #avoid generating _dllimport functions.
  target_compile_definitions(fbgemm_generic PUBLIC FBGEMM_STATIC)
else()
  message(FATAL_ERROR "Unsupported library type ${FBGEMM_LIBRARY_TYPE}")
endif()
foreach(target IN ITEMS "${fbgemm_targets};fbgemm;fbgemm_generic")
  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
endforeach()

target_link_libraries(fbgemm PUBLIC fbgemm_generic ${fbgemm_targets})

if(OpenMP_FOUND)
  target_link_libraries(fbgemm_generic PUBLIC OpenMP::OpenMP_CXX)
endif()

install(
  TARGETS fbgemm fbgemm_generic ${fbgemm_targets}
  EXPORT fbgemmLibraryConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) #For windows

install(
  FILES ${FBGEMM_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fbgemm")

install(
  EXPORT fbgemmLibraryConfig
  DESTINATION share/cmake/fbgemm
  FILE fbgemmLibraryConfig.cmake)

#Make project importable from the build directory
#export(TARGETS fbgemm asmjit FILE fbgemmLibraryConfig.cmake)

if(FBGEMM_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(FBGEMM_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(FBGEMM_BUILD_DOCS)
  add_subdirectory(docs)
endif()

if(FBGEMM_BUILD_FBGEMM_GPU)
  add_subdirectory(fbgemm_gpu)
endif()

if(NOT HAVE_GNU_F2H_IEEE)
  add_definitions(-DMISSING_GNU_F2H_IEEE)
endif()

if(FBGEMM_USE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported()
  foreach(target IN LISTS fbgemm_targets "fbgemm_generic")
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE TRUE)
  endforeach()
  set_property(TARGET fbgemm_avx2 PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set_property(TARGET fbgemm_avx512 PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()
