# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# Find Packages
################################################################################

find_package(GTest)

if(NOT GTest_FOUND)
  # Download Googletest framework from github if
  # GOOGLETEST_SOURCE_DIR is not specified.
  set(INSTALL_GTEST OFF)

  if(NOT DEFINED GOOGLETEST_SOURCE_DIR)
    set(GOOGLETEST_SOURCE_DIR "${FBGEMM_SOURCE_DIR}/external/googletest"
      CACHE STRING "googletest source directory from submodules")
  endif()

  #build Googletest framework
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  add_subdirectory("${GOOGLETEST_SOURCE_DIR}" "${FBGEMM_BINARY_DIR}/googletest")
  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()

if (FBGEMM_USE_SANITIZER)
  message(WARNING "USING SANITIZER IN TEST")
endif()

if(${OpenMP_FOUND})
  message(STATUS "OpenMP_LIBRARIES= ${OpenMP_CXX_LIBRARIES}")
endif()

function(add_gtest TESTNAME)
  add_executable(${TESTNAME}
    ${ARGN}
    ../bench/BenchUtils.cc
    EmbeddingSpMDMTestUtils.cc
    QuantizationHelpers.cc
    TestUtils.cc)

  # To compile test files with AVX2 turned on
  # For static build, defining FBGEMM_STATIC to avoid generating
  # functions with _dllimport attributes.
  if(MSVC)
    target_compile_options(${TESTNAME} PRIVATE
      /arch:AVX2
      /wd4244
      /wd4267
      /wd4305
      /wd4309)

    if (FBGEMM_LIBRARY_TYPE STREQUAL STATIC)
      target_compile_definitions(${TESTNAME} PRIVATE
        FBGEMM_STATIC)
    endif()

  else()
    target_compile_options(${TESTNAME} PRIVATE
      -m64
      -mavx2
      -mfma
      -masm=intel)
  endif()

  if (FBGEMM_USE_SANITIZER)
    target_compile_options(${TESTNAME} PRIVATE
      "-fsanitize=${FBGEMM_USE_SANITIZER}" "-fno-omit-frame-pointer")
    target_link_options(${TESTNAME} PRIVATE
      "-fsanitize=${FBGEMM_USE_SANITIZER}" "-fno-omit-frame-pointer")
  endif()

  target_link_libraries(${TESTNAME}
    gtest
    gmock
    gtest_main
    fbgemm)

  if(OpenMP_FOUND)
    target_link_libraries(${TESTNAME} OpenMP::OpenMP_CXX)
  endif()

  add_test(${TESTNAME} ${TESTNAME})
  set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endfunction()


file(GLOB TEST_LIST "*Test.cc")
foreach(TEST_FILE ${TEST_LIST})
  # NOTE: Skip FP32 test until FP32 is properly integrated into OSS builds
  if(TEST_FILE MATCHES "FP32Test.cc$")
    continue()
  endif()

  get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
  get_filename_component(TEST_FILE_ONLY "${TEST_FILE}" NAME)
  add_gtest("${TEST_NAME}" "${TEST_FILE_ONLY}")
endforeach()
