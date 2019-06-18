cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(FBGEMM_BUILD_TESTS AND NOT TARGET gtest)
  #Download Googletest framework from github if
  #GOOGLETEST_SOURCE_DIR is not specified.
  if(NOT DEFINED GOOGLETEST_SOURCE_DIR)
    set(GOOGLETEST_SOURCE_DIR "${FBGEMM_SOURCE_DIR}/third_party/googletest"
      CACHE STRING "googletest source directory from submodules")
  endif()

  #build Googletest framework
  add_subdirectory("${GOOGLETEST_SOURCE_DIR}" "${FBGEMM_BINARY_DIR}/googletest")
endif()

macro(add_gtest TESTNAME)
  add_executable(${TESTNAME} ${ARGN}
    ../bench/BenchUtils.cc QuantizationHelpers.cc TestUtils.cc)
  set_target_properties(${TESTNAME} PROPERTIES
          CXX_STANDARD 11
          CXX_EXTENSIONS NO)
  if(MSVC)
    target_compile_options(${TESTNAME} PRIVATE "/DFBGEMM_STATIC")
  else(MSVC)
    target_compile_options(${TESTNAME} PRIVATE
      "-m64" "-mavx2" "-mfma" "-masm=intel")
  endif(MSVC)
  target_link_libraries(${TESTNAME} gtest gmock gtest_main fbgemm)
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
