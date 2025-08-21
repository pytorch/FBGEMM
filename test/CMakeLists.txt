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

if(FBGEMM_USE_SANITIZER)
  message(WARNING "USING SANITIZER IN TEST")
endif()

if(${OpenMP_FOUND})
  message(STATUS "OpenMP_LIBRARIES= ${OpenMP_CXX_LIBRARIES}")
endif()

function(add_gtest TESTNAME)
  set(test_sources
    ${ARGN}
    ../bench/BenchUtils.cc
    EmbeddingSpMDMTestUtils.cc
    QuantizationHelpers.cc
    TestUtils.cc)

  add_executable(${TESTNAME} ${test_sources})

  # To compile test files with AVX2 turned on
  if(MSVC)
    target_compile_options(${TESTNAME} PRIVATE
      /wd4244
      /wd4267
      /wd4305
      /wd4309)
  endif()

  if(CXX_AVX2_FOUND)
    if(MSVC)
      target_compile_options(${TESTNAME} PRIVATE ${CXX_AVX2_FLAGS})
    else()
      target_compile_options(${TESTNAME} PRIVATE
      -m64
      ${CXX_AVX2_FLAGS}
      -mfma
      -masm=intel)
    endif()
  endif()

  if(FBGEMM_USE_SANITIZER)
    target_compile_options(${TESTNAME} PRIVATE
      "-fsanitize=${FBGEMM_USE_SANITIZER}" "-fno-omit-frame-pointer")
    target_link_options(${TESTNAME} PRIVATE
      "-fsanitize=${FBGEMM_USE_SANITIZER}" "-fno-omit-frame-pointer")
  endif()

  target_link_libraries(${TESTNAME} fbgemm)
  if(TARGET GTest::gtest_main)
    target_link_libraries(${TESTNAME} GTest::gmock GTest::gtest_main)
  else()
    target_link_libraries(${TESTNAME} gmock gtest_main)
  endif()

  if(OpenMP_FOUND)
    target_link_libraries(${TESTNAME} OpenMP::OpenMP_CXX)
  endif()

  add_test(${TESTNAME} ${TESTNAME})
  set_target_properties(${TESTNAME} PROPERTIES FOLDER test)

  BLOCK_PRINT(
    "Test Target: ${TESTNAME}"
    " "
    "SRCS:"
    "${test_sources}")
endfunction()


file(GLOB TEST_LIST "*Test.cc")
foreach(TEST_FILE ${TEST_LIST})
  # NOTE: Skip tests on ARM for now until linking issues are fixed
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    if(TEST_FILE MATCHES "GConvTest"
    OR TEST_FILE MATCHES "UniConvTest"
    OR(TEST_FILE MATCHES "FP16Test.cc$" AND (NOT APPLE))
    OR TEST_FILE MATCHES "I8DepthwiseTest.cc$"
    OR TEST_FILE MATCHES "I8SpmdmTest.cc$"
    OR TEST_FILE MATCHES "I8DirectconvTest.cc$"
    OR TEST_FILE MATCHES "Requantize")
      continue()
    endif()
  endif()

  if(MSVC AND TEST_FILE MATCHES "FP32Test.cc$")
    # NOTE: Skip FP32 test for MSVC until intrinsic kernels are implemented
    continue()
  endif()

  message(STATUS "Processing: ${TEST_FILE}")

  get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
  get_filename_component(TEST_FILE_ONLY "${TEST_FILE}" NAME)
  add_gtest("${TEST_NAME}" "${TEST_FILE_ONLY}")
endforeach()
