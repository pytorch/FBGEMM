# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(FBGEMM_BUILD_TESTS AND NOT TARGET gtest)
  #Download Googletest framework from github if
  #GOOGLETEST_SOURCE_DIR is not specified.
  if(NOT DEFINED GOOGLETEST_SOURCE_DIR)
    set(GOOGLETEST_SOURCE_DIR "${FBGEMM_SOURCE_DIR}/third_party/googletest"
      CACHE STRING "googletest source directory from submodules")
  endif()

  #build Googletest framework
  #MSVC needs gtest_for_shared_crt to select right runtime lib
  if (MSVC AND FBGEMM_LIBRARY_TYPE STREQUAL "shared")
    message(WARNING "gtest_force_shared_crt is ON")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  else()
    message(WARNING "gtest_force_shared_crt is OFF")
  endif()
  add_subdirectory("${GOOGLETEST_SOURCE_DIR}" "${FBGEMM_BINARY_DIR}/googletest")
  # add flags required for mac build
  if(NOT MSVC)
    target_compile_options(gtest PRIVATE "-Wno-deprecated-copy")
    target_compile_options(gmock PRIVATE "-Wno-deprecated-copy")
  endif()
endif()

if (USE_SANITIZER)
  message(WARNING "USING SANITIZER IN TEST")
endif()

if(${OpenMP_FOUND})
  message(STATUS "OpenMP_LIBRARIES= ${OpenMP_CXX_LIBRARIES}")
endif()

macro(add_gtest TESTNAME)
  add_executable(${TESTNAME} ${ARGN}
    ../bench/BenchUtils.cc
    EmbeddingSpMDMTestUtils.cc
    QuantizationHelpers.cc
    TestUtils.cc)
  # To compile test files with AVX2 turned on
  # For static build, defining FBGEMM_STATIC to avoid generating
  # functions with _dllimport attributes.
  if(MSVC)
    target_compile_options(${TESTNAME} PRIVATE
      "/arch:AVX2" "/wd4244" "/wd4267" "/wd4305" "/wd4309")
    if (FBGEMM_LIBRARY_TYPE STREQUAL "static")
      target_compile_definitions(${TESTNAME} PRIVATE FBGEMM_STATIC)
    endif()
  else(MSVC)
    target_compile_options(${TESTNAME} PRIVATE
     "-m64" "-mavx2" "-mfma" "-masm=intel")
  endif(MSVC)
  if (USE_SANITIZER)
    target_compile_options(${TESTNAME} PRIVATE
      "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
    target_link_options(${TESTNAME} PRIVATE
      "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
  endif()

  target_link_libraries(${TESTNAME} gtest gmock gtest_main fbgemm)

  if(${OpenMP_FOUND})
    target_link_libraries(${TESTNAME} ${OpenMP_CXX_LIBRARIES})
  endif()

  add_dependencies(${TESTNAME} gtest fbgemm)
  add_test(${TESTNAME} ${TESTNAME})
  set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()


file(GLOB TEST_LIST "*Test.cc")

foreach(TEST_FILE ${TEST_LIST})
  get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
  get_filename_component(TEST_FILE_ONLY "${TEST_FILE}" NAME)
  add_gtest("${TEST_NAME}"
    "${TEST_FILE_ONLY}")
endforeach()
