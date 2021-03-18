cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")

set (CUB_INCLUDE
  "$ENV{CUB_DIR}")

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

# function to extract filelists from defs.bzl file
find_package(PythonInterp)
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('defs.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar)
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()

project(fbgemm_gpu VERSION 0.1 LANGUAGES CXX C CUDA)

set(FBGEMMGPU_LIBRARY_TYPE "default" CACHE STRING
  "Type of library (shared, static, or default) to build")
set_property(CACHE
  FBGEMMGPU_LIBRARY_TYPE PROPERTY STRINGS default static shared)
option(FBGEMMGPU_BUILD_TESTS "Build fbgemm_gpu unit tests" ON)
option(FBGEMMGPU_BUILD_BENCHMARKS "Build fbgemm_gpu benchmarks" ON)

if(FBGEMMGPU_BUILD_TESTS)
  enable_testing()
endif()

set(FBGEMMGPU_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FBGEMMGPU_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FBGEMMGPU_THIRDPARTY_DIR ${FBGEMMGPU_BINARY_DIR}/third_party)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#add address sanitizer
set(USE_SANITIZER "" CACHE STRING "options include address, leak, ...")

#We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Define file lists
get_filelist("get_fbgemm_gpu_wrapper_srcs()" FBGEMMGPU_GENERIC_SRCS)
get_filelist("get_fbgemm_gpu_public_headers()" FBGEMMGPU_PUBLIC_HEADERS)

message(STATUS "FBGEMMGPU_GENERIC_SRC: ${FBGEMMGPU_GENERIC_SRCS}")
message(STATUS "FBGEMMGPU_PUBLIC_HEADERS: ${FBGEMMGPU_PUBLIC_HEADERS}")

add_library(fbgemm_gpu_generic OBJECT ${FBGEMMGPU_GENERIC_SRCS})

# Make libraries depend on defs.bzl
add_custom_target(defs.bzl DEPENDS defs.bzl)
add_dependencies(fbgemm_gpu_generic defs.bzl)

set_target_properties(fbgemm_gpu_generic PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
      CXX_VISIBILITY_PRESET hidden)

#On Windows:
#  MSVC uses /MD in default cxx compiling flags,
#need to change it to /MT in static case
if(MSVC)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4305 /wd4309")
  if(FBGEMMGPU_LIBRARY_TYPE STREQUAL "static")
    foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
endif(MSVC)

if(USE_SANITIZER)
  target_compile_options(fbgemm_gpu_generic PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
endif()

message(WARNING "==========")
message(WARNING "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message(WARNING "CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message(WARNING "CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
message(WARNING "==========")

target_include_directories(fbgemm_gpu_generic BEFORE
      PUBLIC $<BUILD_INTERFACE:${FBGEMMGPU_SOURCE_DIR}>
      PUBLIC $<BUILD_INTERFACE:${FBGEMMGPU_SOURCE_DIR}/include>
      ${CUB_INCLUDE}
      )

message(STATUS "FBGEMMGPU_SOURCE_DIR: ${FBGEMMGPU_SOURCE_DIR}")
message(STATUS "$<TARGET_OBJECTS:fbgemm_gpu_generic: ${fbgemm_gpu_generic}")

if(FBGEMMGPU_LIBRARY_TYPE STREQUAL "default")
  add_library(fbgemm_gpu
    $<TARGET_OBJECTS:fbgemm_gpu_generic>)
elseif(FBGEMMGPU_LIBRARY_TYPE STREQUAL "shared")
  add_library(fbgemm_gpu SHARED
    $<TARGET_OBJECTS:fbgemm_gpu_generic>)
  set_property(TARGET fbgemm_gpu_generic PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(fbgemm_gpu PROPERTIES
    CXX_VISIBILITY_PRESET hidden)
elseif(FBGEMMGPU_LIBRARY_TYPE STREQUAL "static")
  add_library(fbgemm_gpu STATIC
    $<TARGET_OBJECTS:fbgemm_gpu_generic>)
  #MSVC need to define FBGEMMGPU_STATIC for fbgemm_gpu_generic also to
  #avoid generating _dllimport functions.
  target_compile_definitions(fbgemm_gpu_generic PRIVATE FBGEMMGPU_STATIC)
  target_compile_definitions(fbgemm_gpu PRIVATE FBGEMMGPU_STATIC)
else()
  message(FATAL_ERROR "Unsupported library type ${FBGEMMGPU_LIBRARY_TYPE}")
endif()

if(USE_SANITIZER)
  target_link_options(fbgemm_gpu PRIVATE
    "-fsanitize=${USE_SANITIZER}" "-fno-omit-frame-pointer")
endif()

target_include_directories(fbgemm_gpu BEFORE
    PUBLIC $<BUILD_INTERFACE:${FBGEMMGPU_SOURCE_DIR}>
    PUBLIC $<BUILD_INTERFACE:${FBGEMMGPU_SOURCE_DIR}/include>
    ${CUB_INCLUDE}
    )

install(TARGETS fbgemm_gpu EXPORT fbgemm_gpuLibraryConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}) #For windows

install(FILES ${FBGEMMGPU_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fbgemm_gpu")

install(EXPORT fbgemm_gpuLibraryConfig DESTINATION share/cmake/fbgemm_gpu
  FILE fbgemm_gpuLibraryConfig.cmake)

if(MSVC)
  if(FBGEMMGPU_LIBRARY_TYPE STREQUAL "shared")
    install(
      FILES $<TARGET_PDB_FILE:fbgemm_gpu>
      DESTINATION ${CMAKE_INSTALL_LIBDIR} OPTIONAL)
  endif()
  install(TARGETS fbgemm_gpu DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(FBGEMMGPU_BUILD_TESTS)
  add_subdirectory(test)
endif()

if(FBGEMMGPU_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
