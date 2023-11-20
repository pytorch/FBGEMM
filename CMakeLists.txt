# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# CMake Prelude
################################################################################

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Define function to extract filelists from defs.bzl file
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar
    RESULT_VARIABLE _resvar
    ERROR_VARIABLE _errvar)
  if (NOT "${_resvar}" EQUAL "0")
    message(WARNING "Failed to execute Python (${PYTHON_EXECUTABLE})\n"
      "Result: ${_resvar}\n"
      "Error: ${_errvar}\n")
  endif()
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()


################################################################################
# FBGEMM C++ Setup
################################################################################

# Set the default C++ standard to C++17
# Individual targets can have this value overridden; see
# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Set the default C standard to C11
# Individual targets can have this value overridden; see
# https://cmake.org/cmake/help/latest/prop_tgt/C_STANDARD.html
set(CMAKE_C_STANDARD 11)
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
find_package(PythonInterp)

set(FBGEMM_LIBRARY_TYPE "default"
  CACHE STRING
  "Type of library (shared, static, or default) to build")

set_property(CACHE FBGEMM_LIBRARY_TYPE PROPERTY STRINGS default static shared)
option(FBGEMM_BUILD_TESTS "Build fbgemm unit tests" ON)
option(FBGEMM_BUILD_BENCHMARKS "Build fbgemm benchmarks" ON)
option(FBGEMM_BUILD_DOCS "Build fbgemm documentation" OFF)
option(FBGEMM_BUILD_FBGEMM_GPU "Build fbgemm_gpu library" OFF)

if(FBGEMM_BUILD_TESTS)
  enable_testing()
endif()

set(FBGEMM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FBGEMM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FBGEMM_THIRDPARTY_DIR ${FBGEMM_BINARY_DIR}/third_party)
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

# We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Check if compiler supports OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
  message(WARNING "OpenMP found! OpenMP_C_INCLUDE_DIRS = ${OpenMP_C_INCLUDE_DIRS}")
  include_directories(${OpenMP_C_INCLUDE_DIRS})
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
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
get_filelist("get_fbgemm_public_headers()" FBGEMM_PUBLIC_HEADERS)

add_library(fbgemm_generic OBJECT ${FBGEMM_GENERIC_SRCS})
add_library(fbgemm_avx2 OBJECT ${FBGEMM_AVX2_SRCS} ${FBGEMM_AVX2_INLINE_SRCS})
add_library(fbgemm_avx512 OBJECT
  ${FBGEMM_AVX512_SRCS} ${FBGEMM_AVX512_INLINE_SRCS})

# Make libraries depend on defs.bzl
add_custom_target(defs.bzl DEPENDS defs.bzl)
add_dependencies(fbgemm_generic defs.bzl)
add_dependencies(fbgemm_avx2 defs.bzl)
add_dependencies(fbgemm_avx512 defs.bzl)

# On Windows:
# 1)  Adding definition of ASMJIT_STATIC to avoid generating asmjit function
#     calls with _dllimport attribute
# 2)  MSVC uses /MD in default cxx compiling flags,
# Need to change it to /MT in static case
if(MSVC)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4305 /wd4309")
  if(FBGEMM_LIBRARY_TYPE STREQUAL "static")
    target_compile_definitions(fbgemm_generic PRIVATE ASMJIT_STATIC)
    target_compile_definitions(fbgemm_avx2 PRIVATE ASMJIT_STATIC)
    target_compile_definitions(fbgemm_avx512 PRIVATE ASMJIT_STATIC)
    foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
  target_compile_options(fbgemm_avx2 PRIVATE "/arch:AVX2")
  target_compile_options(fbgemm_avx512 PRIVATE "/arch:AVX512")
else(MSVC)
  string(APPEND CMAKE_CXX_FLAGS " -Wall")
  string(APPEND CMAKE_CXX_FLAGS " -Wextra")
  string(APPEND CMAKE_CXX_FLAGS " -Werror")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated-declarations")
  string(APPEND CMAKE_CXX_FLAGS " -Wimplicit-fallthrough")
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
  set_source_files_properties(
    src/FbgemmI64.cc
    src/FbgemmI8Depthwise3DAvx2.cc
    src/FbgemmI8DepthwiseAvx2.cc
    src/UtilsAvx2.cc
    PROPERTIES COMPILE_FLAGS "-Wno-uninitialized")
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
  target_compile_options(fbgemm_generic PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
  target_compile_options(fbgemm_avx2 PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
  target_compile_options(fbgemm_avx512 PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
endif()

message(WARNING "==========")
message(WARNING "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(WARNING "CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message(WARNING "CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
message(WARNING "==========")

if(NOT TARGET asmjit)
  #Download asmjit from github if ASMJIT_SRC_DIR is not specified.
  if(NOT DEFINED ASMJIT_SRC_DIR)
    set(ASMJIT_SRC_DIR "${FBGEMM_SOURCE_DIR}/third_party/asmjit"
      CACHE STRING "asmjit source directory from submodules")
  endif()

  #build asmjit
  #For MSVC shared build, asmjit needs to be shared also.
  if (MSVC AND (FBGEMM_LIBRARY_TYPE STREQUAL "shared"))
    set(ASMJIT_STATIC OFF)
  else()
    set(ASMJIT_STATIC ON)
  endif()

  add_subdirectory("${ASMJIT_SRC_DIR}" "${FBGEMM_BINARY_DIR}/asmjit")
  set_property(TARGET asmjit PROPERTY POSITION_INDEPENDENT_CODE ON)
  # add a flag required for mac build
  if(NOT MSVC)
    target_compile_options(asmjit PRIVATE "-Wno-sign-conversion")
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 13.0.0)
    # See https://github.com/pytorch/pytorch/issues/74352, https://github.com/pytorch/FBGEMM/issues/1173
    target_compile_options_if_supported(asmjit -Wno-deprecated-copy)
    target_compile_options_if_supported(asmjit -Wno-unused-but-set-variable)
  endif()
endif()

if(NOT TARGET cpuinfo)
  #Download cpuinfo from github if CPUINFO_SOURCE_DIR is not specified.
  if(NOT DEFINED CPUINFO_SOURCE_DIR)
    set(CPUINFO_SOURCE_DIR "${FBGEMM_SOURCE_DIR}/third_party/cpuinfo"
      CACHE STRING "cpuinfo source directory from submodules")
  endif()

  #build cpuinfo
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Do not build cpuinfo unit tests")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Do not build cpuinfo mock tests")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Do not build cpuinfo benchmarks")
  set(CPUINFO_LIBRARY_TYPE static CACHE STRING "Set lib type to static")
  #Select static runtime, needed for static build for MSVC
  set(CPUINFO_RUNTIME_TYPE static CACHE STRING "Set runtime to static")
  add_subdirectory("${CPUINFO_SOURCE_DIR}" "${FBGEMM_BINARY_DIR}/cpuinfo")
  set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(fbgemm_generic BEFORE
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
      PRIVATE "${ASMJIT_SRC_DIR}/src"
      PRIVATE "${CPUINFO_SOURCE_DIR}/include")

target_include_directories(fbgemm_avx2 BEFORE
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
      PRIVATE "${ASMJIT_SRC_DIR}/src"
      PRIVATE "${CPUINFO_SOURCE_DIR}/include")

target_include_directories(fbgemm_avx512 BEFORE
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
      PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>
      PRIVATE "${ASMJIT_SRC_DIR}/src"
      PRIVATE "${CPUINFO_SOURCE_DIR}/include")

if(FBGEMM_LIBRARY_TYPE STREQUAL "default")
  add_library(fbgemm
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_avx2>
    $<TARGET_OBJECTS:fbgemm_avx512>)
elseif(FBGEMM_LIBRARY_TYPE STREQUAL "shared")
  add_library(fbgemm SHARED
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_avx2>
    $<TARGET_OBJECTS:fbgemm_avx512>)
  set_property(TARGET fbgemm_generic PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET fbgemm_avx2 PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET fbgemm_avx512 PROPERTY POSITION_INDEPENDENT_CODE ON)
elseif(FBGEMM_LIBRARY_TYPE STREQUAL "static")
  add_library(fbgemm STATIC
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_avx2>
    $<TARGET_OBJECTS:fbgemm_avx512>)
  #MSVC need to define FBGEMM_STATIC for fbgemm_generic also to
  #avoid generating _dllimport functions.
  target_compile_definitions(fbgemm_generic PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm_avx2 PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm_avx512 PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm PRIVATE FBGEMM_STATIC)
else()
  message(FATAL_ERROR "Unsupported library type ${FBGEMM_LIBRARY_TYPE}")
endif()

if(USE_SANITIZER)
  target_link_options(fbgemm PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
endif()

target_include_directories(fbgemm BEFORE
    PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
    PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>)

target_link_libraries(fbgemm
  $<BUILD_INTERFACE:asmjit>
  $<BUILD_INTERFACE:cpuinfo>)
add_dependencies(fbgemm
  asmjit
  cpuinfo)

if(OpenMP_FOUND)
  target_link_libraries(fbgemm OpenMP::OpenMP_CXX)
endif()

install(
  TARGETS fbgemm
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

if(MSVC)
  if(FBGEMM_LIBRARY_TYPE STREQUAL "shared")
    install(
      FILES $<TARGET_PDB_FILE:fbgemm> $<TARGET_PDB_FILE:asmjit>
      DESTINATION ${CMAKE_INSTALL_LIBDIR} OPTIONAL)
  endif()
  install(TARGETS fbgemm DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(TARGETS asmjit DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

#Make project importable from the build directory
#export(TARGETS fbgemm asmjit FILE fbgemmLibraryConfig.cmake)

if(FBGEMM_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(FBGEMM_BUILD_BENCHMARKS)
  add_subdirectory(bench)
  # add a flag to enable Clang 14
  set_source_files_properties(
    bench/GEMMsBenchmark.cc
    PROPERTIES COMPILE_FLAGS "-Wno-unused-variable")
endif()

if(FBGEMM_BUILD_DOCS)
  add_subdirectory(docs)
endif()

if(FBGEMM_BUILD_FBGEMM_GPU)
  add_subdirectory(fbgemm_gpu)
endif()
