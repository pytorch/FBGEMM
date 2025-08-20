# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

################################################################################
# Find Packages
################################################################################

find_package(MKL)
if(NOT ${MKL_FOUND})
  find_package(BLAS)
endif()

if(${MKL_FOUND})
  message(STATUS "MKL_LIBRARIES= ${MKL_LIBRARIES}")
endif()

if(${BLAS_FOUND})
  message(STATUS "BLAS_LIBRARIES= ${BLAS_LIBRARIES}")
endif()

################################################################################
# Function to Create Benchmark Targets
################################################################################

function(add_benchmark BENCHNAME)
  set(bench_sources
    ${ARGN}
    BenchUtils.cc
    ../test/QuantizationHelpers.cc
    ../test/EmbeddingSpMDMTestUtils.cc)

  add_executable(
    ${BENCHNAME}
    ${bench_sources})

  if(NOT MSVC)
    if(COMPILER_SUPPORTS_AVX2)
      target_compile_options(${BENCHNAME} PRIVATE
        -m64
        ${CXX_AVX2_FLAGS}
        -mfma
        -masm=intel)
    else()
      target_compile_options(${BENCHNAME} PRIVATE ${CXX_AVX2_FLAGS})
    endif()
  endif()
  if(APPLE)
    target_compile_options(${BENCHNAME} PRIVATE "-Wno-deprecated-declarations")
  endif()

  target_link_libraries(${BENCHNAME} fbgemm)

  if(${MKL_FOUND})
    target_include_directories(${BENCHNAME} PRIVATE
      ${MKL_INCLUDE_DIR})

    target_link_libraries(${BENCHNAME}
      ${MKL_LIBRARIES})

    target_compile_definitions(${BENCHNAME} PRIVATE
      -DUSE_MKL)
  endif()

  if(${BLAS_FOUND})
    target_link_libraries(${BENCHNAME}
      ${BLAS_LIBRARIES})

    target_compile_definitions(${BENCHNAME} PRIVATE
      -DUSE_BLAS)
  endif()

  set_target_properties(${BENCHNAME} PROPERTIES
    FOLDER test)

  ##############################################################################
  # Debug Summary
  ##############################################################################

  BLOCK_PRINT(
    "Benchmark Target: ${BENCHNAME}"
    " "
    "SRCS:"
    "${bench_sources}")

endfunction()

################################################################################
# Create Benchmark Targets
################################################################################

if(FBGEMM_BUILD_BENCHMARKS)
  file(GLOB BENCH_LIST "*Benchmark.cc")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    list(FILTER BENCH_LIST EXCLUDE REGEX "Depthwise")
    list(FILTER BENCH_LIST EXCLUDE REGEX "ConvUnifiedBenchmark")
    list(FILTER BENCH_LIST EXCLUDE REGEX "Quantize")
    list(FILTER BENCH_LIST EXCLUDE REGEX "GEMMsBenchmark")
    list(FILTER BENCH_LIST EXCLUDE REGEX "quantize")
  endif()

  if(MSVC)
    # NOTE: Skip FP32 benchmark for MSVC until intrinsic kernels are implemented
    list(FILTER BENCH_LIST EXCLUDE REGEX "FP32Benchmark\\.cc$")
  endif()

  foreach(BENCH_FILE ${BENCH_LIST})
    get_filename_component(BENCH_NAME ${BENCH_FILE} NAME_WE)
    get_filename_component(BENCH_FILENAME ${BENCH_FILE} NAME)

    # Create benchmark target
    add_benchmark(${BENCH_NAME} ${BENCH_FILENAME})
    # Add benchmark to list of benchmarks
    list(APPEND BENCHMARKS ${BENCH_NAME})
  endforeach()

  add_custom_target(run_benchmarks
    COMMAND ${BENCHMARKS})

  add_dependencies(run_benchmarks
    ${BENCHMARKS})
endif()
