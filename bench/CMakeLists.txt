cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

find_package(MKL)
if (NOT ${MKL_FOUND})
  find_package(BLAS)
endif()

#benchmarks
macro(add_benchmark BENCHNAME)
  add_executable(${BENCHNAME} ${ARGN}
    BenchUtils.cc ../test/QuantizationHelpers.cc)
  set_target_properties(${BENCHNAME} PROPERTIES
          CXX_STANDARD 11
          CXX_EXTENSIONS NO)
  if(MSVC)
    target_compile_options(${BENCHNAME} PRIVATE "/DFBGEMM_STATIC" "/MT")
  else(MSVC)
    target_compile_options(${BENCHNAME} PRIVATE
      "-m64" "-mavx2" "-mfma" "-masm=intel")
  endif(MSVC)
  target_link_libraries(${BENCHNAME} fbgemm)
  add_dependencies(${BENCHNAME} fbgemm)
  if(${MKL_FOUND})
    message(STATUS "MKL_LIBRARIES= ${MKL_LIBRARIES}")
    target_include_directories(${BENCHNAME} PRIVATE "${MKL_INCLUDE_DIR}")
    target_link_libraries(${BENCHNAME} "${MKL_LIBRARIES}")
    target_compile_options(${BENCHNAME} PRIVATE
      "-DUSE_MKL")
  endif()
  if (${BLAS_FOUND})
    message(STATUS "BLAS_LIBRARIES= ${BLAS_LIBRARIES}")
    target_compile_options(${BENCHNAME} PRIVATE "-DUSE_BLAS")
    target_link_libraries(${BENCHNAME} "${BLAS_LIBRARIES}")
  endif()

  set_target_properties(${BENCHNAME} PROPERTIES FOLDER test)
endmacro()

if(FBGEMM_BUILD_BENCHMARKS)

  set(BENCHMARKS "")

  file(GLOB BENCH_LIST "*Benchmark.cc")
  foreach(BENCH_FILE ${BENCH_LIST})
    get_filename_component(BENCH_NAME "${BENCH_FILE}" NAME_WE)
    get_filename_component(BENCH_FILE_ONLY "${BENCH_FILE}" NAME)
    add_benchmark("${BENCH_NAME}"
      "${BENCH_FILE_ONLY}")
    list(APPEND BENCHMARKS "${BENCH_NAME}")
  endforeach()

  add_custom_target(run_benchmarks
    COMMAND ${BENCHMARKS})

  add_dependencies(run_benchmarks
    ${BENCHMARKS})

endif()
