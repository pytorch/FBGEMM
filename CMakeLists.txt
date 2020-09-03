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

# only static library is available for windows
if(MSVC)
  set(FBGEMM_LIBRARY_TYPE "static")
endif(MSVC)

#All the source files that either use avx2 instructions statically or JIT
#avx2/avx512 instructions.
set(FBGEMM_GENERIC_SRCS src/ExecuteKernel.cc
                src/ExecuteKernelU8S8.cc
                src/Fbgemm.cc
                src/FbgemmFP16.cc
                src/FbgemmConv.cc
                src/FbgemmI8Spmdm.cc
                src/GenerateKernelU8S8S32ACC16.cc
                src/GenerateKernelU8S8S32ACC16Avx512.cc
                src/GenerateKernelU8S8S32ACC16Avx512VNNI.cc
                src/GenerateKernelU8S8S32ACC32.cc
                src/GenerateKernelU8S8S32ACC32Avx512.cc
                src/GenerateKernelU8S8S32ACC32Avx512VNNI.cc
                src/GroupwiseConvAcc32Avx2.cc
                src/PackAMatrix.cc
                src/PackAWithIm2Col.cc
                src/PackBMatrix.cc
                src/PackMatrix.cc
                src/PackAWithQuantRowOffset.cc
                src/PackAWithRowOffset.cc
                src/PackWeightMatrixForGConv.cc
                src/PackWeightsForConv.cc
                src/QuantUtils.cc
                src/RefImplementations.cc
                src/Utils.cc)

#check if compiler supports avx512
if (MSVC)
  set(DISABLE_GLOBALLY "/wd\"4310\" /wd\"4324\"")
  set(INTRINSICS "/arch:AVX2")
  set(CMAKE_CXX_FLAGS           "/EHsc /DWIN32 /D_WINDOWS /DUNICODE /D_UNICODE /D_CRT_NONSTDC_NO_WARNINGS /D_CRT_SECURE_NO_WARNINGS ${DISABLE_GLOBALLY}")
  set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS} /MT /O2 ${INTRINSICS} /Zi /MP /GL /DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG     "${CMAKE_CXX_FLAGS} /MTd /Od /Ob0 ${INTRINSICS} /RTC1 /Zi /D_DEBUG")
  set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /LTCG:incremental /INCREMENTAL:NO /NODEFAULTLIB:MSVCRT /ignore:4049")
  set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /LTCG:incremental /NODEFAULTLIB:MSVCRT")
else()
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG(-mavx512f COMPILER_SUPPORTS_AVX512)
  if(NOT COMPILER_SUPPORTS_AVX512)
    message(FATAL_ERROR "A compiler with AVX512 support is required.")
  endif()
endif()

#We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
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
  src/FbgemmI8Depthwise3DAvx2.cc
  src/FbgemmI8DepthwiseAvx2.cc
  src/OptimizedKernelsAvx2.cc
  src/PackDepthwiseConvMatrixAvx2.cc
  src/QuantUtilsAvx2.cc
  src/UtilsAvx2.cc)

#All the source files that use avx512 instructions statically
set(FBGEMM_AVX512_SRCS src/UtilsAvx512.cc)

set(FBGEMM_PUBLIC_HEADERS include/fbgemm/Fbgemm.h
                          include/fbgemm/FbgemmBuild.h
                          include/fbgemm/FbgemmFP16.h
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

if (NOT MSVC)
  target_compile_options(fbgemm_avx2 PRIVATE
    "-m64" "-mavx2" "-mfma" "-masm=intel" "-mf16c")
  target_compile_options(fbgemm_avx512 PRIVATE
    "-m64" "-mavx2" "-mfma" "-mavx512f" "-mavx512bw" "-mavx512dq"
    "-mavx512vl" "-masm=intel" "-mf16c")
endif()

function(fixup_target TARGET TARGET_SOURCE_DIR)
  # Ensure target correctly prefixes its public header with ${TARGET_SOURCE_DIR},
  # which otherwise breaks installation targets including it as a submodule, so fix that
  
  # We cannot use list(TRANSFORM...) since it wasn't added until 3.12 (and the PREPEND operation added in 3.15)
  # and we need to maintain compatibility with 3.5.1, so implement it manually in a compatible way.
  # We need to maintain 3.5.1 compatibility since it's what ships with Ubuntu 16.04 LTS
  #list(TRANSFORM TARGET_PUBLIC_HEADERS PREPEND "${TARGET_SOURCE_DIR}/")
  get_property(TARGET_PUBLIC_HEADER_PROP TARGET ${TARGET} PROPERTY PUBLIC_HEADER)
  set(TARGET_PUBLIC_HEADERS "")
  foreach(_FILE IN ITEMS ${TARGET_PUBLIC_HEADER_PROP})
    # if the file is already in the target source dir, we can just use it as-is
    if("${_FILE}" MATCHES "${TARGET_SOURCE_DIR}.*")
	  list(APPEND TARGET_PUBLIC_HEADERS ${_FILE})
	else()
	  message(STATUS "Fixing ${TARGET} public header file path: ${_FILE} -> ${TARGET_SOURCE_DIR}/${_FILE}")
	  list(APPEND TARGET_PUBLIC_HEADERS "${TARGET_SOURCE_DIR}/${_FILE}")
	endif()
  endforeach()
  set_property(TARGET ${TARGET} PROPERTY PUBLIC_HEADER ${TARGET_PUBLIC_HEADERS})

  # ensure the target's interface include directories are only prefixed in the source directory
  # for the build interface, otherwise the install interface should strip the source directory
  get_property(TARGET_INCLUDE_DIRECTORIES_PROP TARGET ${TARGET} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  set(TARGET_INTERFACE_INCLUDE_DIRECTORIES "")
  foreach(_DIR IN ITEMS ${TARGET_INCLUDE_DIRECTORIES_PROP})
    # if the directory isn't in the source dir, we can just use it as-is
    if(NOT "${_DIR}" MATCHES "${CMAKE_SOURCE_DIR}")
	  list(APPEND TARGET_INTERFACE_INCLUDE_DIRECTORIES ${_DIR})
	else()
      string(REGEX REPLACE "${TARGET_SOURCE_DIR}/*" "" _INSTALL_DIR "${_DIR}")
	  message(STATUS "Fixing ${TARGET} install interface directory: ${_DIR} -> ${_INSTALL_DIR}")
	  list(APPEND TARGET_INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${_DIR}>)
	  list(APPEND TARGET_INTERFACE_INCLUDE_DIRECTORIES $<INSTALL_INTERFACE:${_INSTALL_DIR}>)
	endif()
  endforeach()
  set_property(TARGET ${TARGET} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${TARGET_INTERFACE_INCLUDE_DIRECTORIES})
endfunction()

if(NOT TARGET asmjit)
  #Download asmjit from github if ASMJIT_SRC_DIR is not specified.
  if(NOT DEFINED ASMJIT_SRC_DIR)
    set(ASMJIT_SRC_DIR "${FBGEMM_SOURCE_DIR}/third_party/asmjit"
      CACHE STRING "asmjit source directory from submodules")
  endif()

  #build asmjit
  set(ASMJIT_STATIC TRUE CACHE STRING "" FORCE)
  add_subdirectory("${ASMJIT_SRC_DIR}" "${FBGEMM_BINARY_DIR}/asmjit")
  set_property(TARGET asmjit PROPERTY POSITION_INDEPENDENT_CODE ON)

  # fixup third party target file paths for install targets
  fixup_target(asmjit "${ASMJIT_SRC_DIR}")
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
  set(CPUINFO_LIBRARY_TYPE static)
  if(MSVC)
    set(CPUINFO_RUNTIME_TYPE static)
  endif(MSVC)
  add_subdirectory("${CPUINFO_SOURCE_DIR}" "${FBGEMM_BINARY_DIR}/cpuinfo")
  set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)

  # fixup third party target file paths for install targets
  fixup_target(cpuinfo "${CPUINFO_SOURCE_DIR}")
  fixup_target(clog "${CPUINFO_SOURCE_DIR}/deps/clog")
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
  target_compile_definitions(fbgemm_generic PRIVATE FBGEMM_STATIC ASMJIT_STATIC)
  target_compile_definitions(fbgemm_avx2 PRIVATE FBGEMM_STATIC ASMJIT_STATIC)
  target_compile_definitions(fbgemm_avx512 PRIVATE FBGEMM_STATIC ASMJIT_STATIC)
  target_compile_definitions(fbgemm PRIVATE FBGEMM_STATIC ASMJIT_STATIC)
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
