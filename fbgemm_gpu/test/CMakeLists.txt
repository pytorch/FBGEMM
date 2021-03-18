cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(FBGEMMGPU_BUILD_TESTS AND NOT TARGET gtest)
  #Download Googletest framework from github if
  #GOOGLETEST_SOURCE_DIR is not specified.
  if(NOT DEFINED GOOGLETEST_SOURCE_DIR)
    set(GOOGLETEST_SOURCE_DIR
      "${FBGEMMGPU_SOURCE_DIR}/../third_party/googletest"
      CACHE STRING "googletest source directory from submodules")
  endif()

  #build Googletest framework
  #MSVC needs gtest_for_shared_crt to select right runtime lib
  if (MSVC AND FBGEMMGPU_LIBRARY_TYPE STREQUAL "shared")
    message(WARNING "gtest_force_shared_crt is ON")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  else()
    message(WARNING "gtest_force_shared_crt is OFF")
  endif()
  add_subdirectory("${GOOGLETEST_SOURCE_DIR}"
    "${FBGEMMGPU_BINARY_DIR}/googletest")
endif()

if (USE_SANITIZER)
  message(WARNING "USING SANITIZER IN TEST")
endif()

macro(add_gtest TESTNAME)
  add_executable(${TESTNAME} ${ARGN}
  )
  set_target_properties(${TESTNAME} PROPERTIES
          CUDA_SEPARABLE_COMPILATION OFF
          CXX_STANDARD 11
          CXX_EXTENSIONS NO)
  #To compile test files with AVX2 turned on
  #For static build, defining FBGEMMGPU_STATIC to avoid generating
  #functions with _dllimport attributes.
  if(MSVC)
    target_compile_options(${TESTNAME} PRIVATE
      "/arch:AVX2" "/wd4244" "/wd4267" "/wd4305" "/wd4309")
    if (FBGEMMGPU_LIBRARY_TYPE STREQUAL "static")
      target_compile_definitions(${TESTNAME} PRIVATE FBGEMMGPU_STATIC)
    endif()
  endif(MSVC)
  if (USE_SANITIZER)
    target_compile_options(${TESTNAME} PRIVATE
      "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
    target_link_options(${TESTNAME} PRIVATE
      "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
  endif()

  target_link_libraries(${TESTNAME} gtest gmock gtest_main fbgemm_gpu)

  add_dependencies(${TESTNAME} gtest fbgemm_gpu)
  add_test(${TESTNAME} ${TESTNAME})
  set_target_properties(${TESTNAME} PROPERTIES FOLDER test)
endmacro()


file(GLOB TEST_LIST "*_test.cu")

foreach(TEST_FILE ${TEST_LIST})
  get_filename_component(TEST_NAME "${TEST_FILE}" NAME_WE)
  get_filename_component(TEST_FILE_ONLY "${TEST_FILE}" NAME)

  message(STATUS "TEST_NAME: ${TEST_NAME}")
  message(STATUS "TEST_FILE_ONLY: ${TEST_FILE_ONLY}")

  add_gtest("${TEST_NAME}"
    "${TEST_FILE_ONLY}")
endforeach()
