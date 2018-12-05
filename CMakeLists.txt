cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

project(fbgemm VERSION 0.1 LANGUAGES CXX C)

set(FBGEMM_LIBRARY_TYPE "default" CACHE STRING
  "Type of library (shared, static, or default) to build")
set_property(CACHE FBGEMM_LIBRARY_TYPE PROPERTY STRINGS default static shared)
option(FBGEMM_BUILD_TESTS "Build fbgemm unit tests" ON)
option(FBGEMM_BUILD_BENCHMARKS "Build fbgemm benchmarks" ON)

if(FBGEMM_BUILD_TESTS)
  enable_testing()
endif()

set(FBGEMM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FBGEMM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FBGEMM_THIRDPARTY_DIR ${FBGEMM_BINARY_DIR}/third_party)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#All the source files that either use avx2 instructions statically or JIT
#avx2/avx512 instructions.
set(FBGEMM_GENERIC_SRCS src/ExecuteKernel.cc
                src/ExecuteKernelU8S8.cc
                src/Fbgemm.cc
                src/FbgemmFP16.cc
                src/FbgemmI8Spmdm.cc
                src/GenerateKernelU8S8S32ACC16.cc
                src/GenerateKernelU8S8S32ACC16Avx512.cc
                src/GenerateKernelU8S8S32ACC32.cc
                src/GenerateKernelU8S8S32ACC32Avx512.cc
                src/PackAMatrix.cc
                src/PackAWithIm2Col.cc
                src/PackBMatrix.cc
                src/PackMatrix.cc
                src/PackAWithQuantRowOffset.cc
                src/PackAWithRowOffset.cc
                src/QuantUtils.cc
                src/RefImplementations.cc
                src/Utils.cc)

#check if compiler supports avx512
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-mavx512f COMPILER_SUPPORTS_AVX512)
if(NOT COMPILER_SUPPORTS_AVX512)
  message(FATAL_ERROR "A compiler with AVX512 support is required.")
endif()

#check if compiler supports openmp
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(WARNING "OpenMP is not supported by the compiler")
endif()

#All the source files that either use avx2 instructions statically
set(FBGEMM_AVX2_SRCS
  src/FbgemmFP16UKernelsAvx2.cc
  src/FbgemmI8DepthwiseAvx2.cc
  src/OptimizedKernelsAvx2.cc
  src/QuantUtilsAvx2.cc
  src/UtilsAvx2.cc)

#All the source files that use avx512 instructions statically
set(FBGEMM_AVX512_SRCS src/UtilsAvx512.cc)

set(FBGEMM_PUBLIC_HEADERS include/fbgemm/Fbgemm.h
                          include/fbgemm/OutputProcessing-inl.h
                          include/fbgemm/PackingTraits-inl.h
                          include/fbgemm/QuantUtils.h
                          include/fbgemm/QuantUtilsAvx2.h
                          include/fbgemm/Utils.h
                          include/fbgemm/UtilsAvx2.h
                          include/fbgemm/ConvUtils.h
                          include/fbgemm/Types.h
                          include/fbgemm/FbgemmI8Spmdm.h)


add_library(fbgemm_generic OBJECT ${FBGEMM_GENERIC_SRCS})
add_library(fbgemm_avx2 OBJECT ${FBGEMM_AVX2_SRCS})
add_library(fbgemm_avx512 OBJECT ${FBGEMM_AVX512_SRCS})

set_target_properties(fbgemm_generic fbgemm_avx2 fbgemm_avx512 PROPERTIES
      CXX_STANDARD 11
      CXX_EXTENSIONS NO
      CXX_VISIBILITY_PRESET hidden)

target_compile_options(fbgemm_avx2 PRIVATE
  "-m64" "-mavx2" "-mfma" "-masm=intel")
target_compile_options(fbgemm_avx512 PRIVATE
  "-m64" "-mavx2" "-mfma" "-mavx512f" "-masm=intel")

if(NOT TARGET asmjit)
  #Download asmjit from github if ASMJIT_SRC_DIR is not specified.
  if(NOT DEFINED ASMJIT_SRC_DIR)
    message(STATUS "Downloading asmjit to ${FBGEMM_THIRDPARTY_DIR}/asmjit
      (define ASMJIT_SRC_DIR to avoid it)")
    configure_file("${FBGEMM_SOURCE_DIR}/cmake/modules/DownloadASMJIT.cmake"
      "${FBGEMM_BINARY_DIR}/asmjit-download/CMakeLists.txt")
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${FBGEMM_BINARY_DIR}/asmjit-download")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${FBGEMM_BINARY_DIR}/asmjit-download")
    set(ASMJIT_SRC_DIR "${FBGEMM_THIRDPARTY_DIR}/asmjit" CACHE STRING
      "asmjit source directory")
  endif()

  #build asmjit
  set(ASMJIT_STATIC ON)
  add_subdirectory("${ASMJIT_SRC_DIR}" "${FBGEMM_BINARY_DIR}/asmjit")
  set_property(TARGET asmjit PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(NOT TARGET cpuinfo)
  #Download cpuinfo from github if CPUINFO_SOURCE_DIR is not specified.
  if(NOT DEFINED CPUINFO_SOURCE_DIR)
    message(STATUS "Downloading cpuinfo to ${FBGEMM_THIRDPARTY_DIR}/cpuinfo
    (define CPUINFO_SOURCE_DIR to avoid it)")
    configure_file("${FBGEMM_SOURCE_DIR}/cmake/modules/DownloadCPUINFO.cmake"
      "${FBGEMM_BINARY_DIR}/cpuinfo-download/CMakeLists.txt")
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${FBGEMM_BINARY_DIR}/cpuinfo-download")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${FBGEMM_BINARY_DIR}/cpuinfo-download")
    set(CPUINFO_SOURCE_DIR "${FBGEMM_THIRDPARTY_DIR}/cpuinfo" CACHE STRING
      "cpuinfo source directory")
  endif()

  #build cpuinfo
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Do not build cpuinfo unit tests")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Do not build cpuinfo mock tests")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Do not build cpuinfo benchmarks")
  set(CPUINFO_LIBRARY_TYPE static)
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
  set_target_properties(fbgemm PROPERTIES
    CXX_VISIBILITY_PRESET hidden)
elseif(FBGEMM_LIBRARY_TYPE STREQUAL "static")
  add_library(fbgemm STATIC
    $<TARGET_OBJECTS:fbgemm_generic>
    $<TARGET_OBJECTS:fbgemm_avx2>
    $<TARGET_OBJECTS:fbgemm_avx512>)
  target_compile_definitions(fbgemm_avx2 PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm_avx512 PRIVATE FBGEMM_STATIC)
  target_compile_definitions(fbgemm PRIVATE FBGEMM_STATIC)
else()
  message(FATAL_ERROR "Unsupported library type ${FBGEMM_LIBRARY_TYPE}")
endif()

target_include_directories(fbgemm BEFORE
    PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}>
    PUBLIC $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>)

target_link_libraries(fbgemm asmjit cpuinfo)
add_dependencies(fbgemm asmjit cpuinfo)

install(TARGETS fbgemm EXPORT fbgemmLibraryConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) #For windows

install(FILES ${FBGEMM_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fbgemm")

#Make project importable from the build directory
#export(TARGETS fbgemm asmjit FILE fbgemmLibraryConfig.cmake)

if(FBGEMM_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(FBGEMM_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
