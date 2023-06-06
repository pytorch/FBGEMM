# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FBGEMM_HAVE_HIP FALSE)

IF(NOT DEFINED ENV{ROCM_PATH})
  SET(ROCM_PATH /opt/rocm)
ELSE()
  SET(ROCM_PATH $ENV{ROCM_PATH})
ENDIF()
if(NOT DEFINED ENV{ROCM_INCLUDE_DIRS})
  set(ROCM_INCLUDE_DIRS ${ROCM_PATH}/include)
else()
  set(ROCM_INCLUDE_DIRS $ENV{ROCM_INCLUDE_DIRS})
endif()
# HIP_PATH
IF(NOT DEFINED ENV{HIP_PATH})
  SET(HIP_PATH ${ROCM_PATH}/hip)
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()

IF(NOT EXISTS ${HIP_PATH})
  return()
ENDIF()

# HCC_PATH
IF(NOT DEFINED ENV{HCC_PATH})
  SET(HCC_PATH ${ROCM_PATH}/hcc)
ELSE()
  SET(HCC_PATH $ENV{HCC_PATH})
ENDIF()

# HSA_PATH
IF(NOT DEFINED ENV{HSA_PATH})
  SET(HSA_PATH ${ROCM_PATH}/hsa)
ELSE()
  SET(HSA_PATH $ENV{HSA_PATH})
ENDIF()

# ROCBLAS_PATH
IF(NOT DEFINED ENV{ROCBLAS_PATH})
  SET(ROCBLAS_PATH ${ROCM_PATH}/rocblas)
ELSE()
  SET(ROCBLAS_PATH $ENV{ROCBLAS_PATH})
ENDIF()

# ROCSPARSE_PATH
IF(NOT DEFINED ENV{ROCSPARSE_PATH})
  SET(ROCSPARSE_PATH ${ROCM_PATH}/rocsparse)
ELSE()
  SET(ROCSPARSE_PATH $ENV{ROCSPARSE_PATH})
ENDIF()

# ROCFFT_PATH
IF(NOT DEFINED ENV{ROCFFT_PATH})
  SET(ROCFFT_PATH ${ROCM_PATH}/rocfft)
ELSE()
  SET(ROCFFT_PATH $ENV{ROCFFT_PATH})
ENDIF()

# HIPSPARSE_PATH
IF(NOT DEFINED ENV{HIPSPARSE_PATH})
  SET(HIPSPARSE_PATH ${ROCM_PATH}/hipsparse)
ELSE()
  SET(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH})
ENDIF()

# THRUST_PATH
IF(NOT DEFINED ENV{THRUST_PATH})
  SET(THRUST_PATH ${ROCM_PATH}/include)
ELSE()
  SET(THRUST_PATH $ENV{THRUST_PATH})
ENDIF()

# HIPRAND_PATH
IF(NOT DEFINED ENV{HIPRAND_PATH})
  SET(HIPRAND_PATH ${ROCM_PATH}/hiprand)
ELSE()
  SET(HIPRAND_PATH $ENV{HIPRAND_PATH})
ENDIF()

# ROCRAND_PATH
IF(NOT DEFINED ENV{ROCRAND_PATH})
  SET(ROCRAND_PATH ${ROCM_PATH}/rocrand)
ELSE()
  SET(ROCRAND_PATH $ENV{ROCRAND_PATH})
ENDIF()

# MIOPEN_PATH
IF(NOT DEFINED ENV{MIOPEN_PATH})
  SET(MIOPEN_PATH ${ROCM_PATH}/miopen)
ELSE()
  SET(MIOPEN_PATH $ENV{MIOPEN_PATH})
ENDIF()

# Add HIP to the CMAKE Module Path
set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})

# Disable Asserts In Code (Can't use asserts on HIP stack.)
ADD_DEFINITIONS(-DNDEBUG)
ADD_DEFINITIONS(-DUSE_ROCM)

IF(NOT DEFINED ENV{PYTORCH_ROCM_ARCH})
  SET(FBGEMM_ROCM_ARCH gfx900;gfx906;gfx908;gfx90a)
ELSE()
  SET(FBGEMM_ROCM_ARCH $ENV{PYTORCH_ROCM_ARCH})
ENDIF()

# Find the HIP Package
find_package(HIP)

IF(HIP_FOUND)
  set(FBGEMM_HAVE_HIP TRUE)

  # Find ROCM version for checks
  # ROCM 5.0 and later will have header api for version management
  if(EXISTS ${ROCM_INCLUDE_DIRS}/rocm_version.h)

    set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}")
    set(file "${PROJECT_BINARY_DIR}/detect_rocm_version.cc")
    file(WRITE ${file} ""
      "#include <rocm_version.h>\n"
      "#include <cstdio>\n"

      "#ifndef ROCM_VERSION_PATCH\n"
      "#define ROCM_VERSION_PATCH 0\n"
      "#endif\n"
      "#define STRINGIFYHELPER(x) #x\n"
      "#define STRINGIFY(x) STRINGIFYHELPER(x)\n"
      "int main() {\n"
      "  printf(\"%d.%d.%s\", ROCM_VERSION_MAJOR, ROCM_VERSION_MINOR, STRINGIFY(ROCM_VERSION_PATCH));\n"
      "  return 0;\n"
      "}\n"
      )

    try_run(run_result compile_result ${PROJECT_RANDOM_BINARY_DIR} ${file}
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${ROCM_INCLUDE_DIRS}"
      RUN_OUTPUT_VARIABLE rocm_version_from_header
      COMPILE_OUTPUT_VARIABLE output_var
      )
    # We expect the compile to be successful if the include directory exists.
    if(NOT compile_result)
      message(FATAL_ERROR "Caffe2: Couldn't determine version from header: " ${output_var})
    endif()
    message(STATUS "Caffe2: Header version is: " ${rocm_version_from_header})
    set(ROCM_VERSION_DEV_RAW ${rocm_version_from_header})
    message("\n***** ROCm version from rocm_version.h ****\n")
  endif()

  string(REGEX MATCH "^([0-9]+)\.([0-9]+)\.([0-9]+).*$" ROCM_VERSION_DEV_MATCH ${ROCM_VERSION_DEV_RAW})

  if(ROCM_VERSION_DEV_MATCH)
    set(ROCM_VERSION_DEV_MAJOR ${CMAKE_MATCH_1})
    set(ROCM_VERSION_DEV_MINOR ${CMAKE_MATCH_2})
    set(ROCM_VERSION_DEV_PATCH ${CMAKE_MATCH_3})
    set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
    math(EXPR ROCM_VERSION_DEV_INT "(${ROCM_VERSION_DEV_MAJOR}*10000) + (${ROCM_VERSION_DEV_MINOR}*100) + ${ROCM_VERSION_DEV_PATCH}")
  endif()

  message("ROCM_VERSION_DEV: ${ROCM_VERSION_DEV}")
  message("ROCM_VERSION_DEV_MAJOR: ${ROCM_VERSION_DEV_MAJOR}")
  message("ROCM_VERSION_DEV_MINOR: ${ROCM_VERSION_DEV_MINOR}")
  message("ROCM_VERSION_DEV_PATCH: ${ROCM_VERSION_DEV_PATCH}")
  message("ROCM_VERSION_DEV_INT:   ${ROCM_VERSION_DEV_INT}")

  math(EXPR TORCH_HIP_VERSION "(${HIP_VERSION_MAJOR} * 100) + ${HIP_VERSION_MINOR}")
  message("HIP_VERSION_MAJOR: ${HIP_VERSION_MAJOR}")
  message("HIP_VERSION_MINOR: ${HIP_VERSION_MINOR}")
  message("TORCH_HIP_VERSION: ${TORCH_HIP_VERSION}")

  message("\n***** Library versions from dpkg *****\n")
  execute_process(COMMAND dpkg -l COMMAND grep rocm-dev COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep rocm-libs COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hsakmt-roct COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep rocr-dev COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep -w hcc COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hip-base COMMAND awk "{print $2 \" VERSION: \" $3}")
  execute_process(COMMAND dpkg -l COMMAND grep hip_hcc COMMAND awk "{print $2 \" VERSION: \" $3}")

  message("\n***** Library versions from cmake find_package *****\n")

  # As of ROCm 5.1.x, all *.cmake files are under /opt/rocm/lib/cmake/<package>
  if(ROCM_VERSION_DEV VERSION_GREATER_EQUAL "5.1.0")
    set(hip_DIR ${HIP_PATH}/lib/cmake/hip)
    set(hsa-runtime64_DIR ${ROCM_PATH}/lib/cmake/hsa-runtime64)
    set(AMDDeviceLibs_DIR ${ROCM_PATH}/lib/cmake/AMDDeviceLibs)
    set(amd_comgr_DIR ${ROCM_PATH}/lib/cmake/amd_comgr)
    set(rocrand_DIR ${ROCM_PATH}/lib/cmake/rocrand)
    set(hiprand_DIR ${ROCM_PATH}/lib/cmake/hiprand)
    set(rocblas_DIR ${ROCM_PATH}/lib/cmake/rocblas)
    set(miopen_DIR ${ROCM_PATH}/lib/cmake/miopen)
    set(rocfft_DIR ${ROCM_PATH}/lib/cmake/rocfft)
    set(hipfft_DIR ${ROCM_PATH}/lib/cmake/hipfft)
    set(hipsparse_DIR ${ROCM_PATH}/lib/cmake/hipsparse)
    set(rccl_DIR ${ROCM_PATH}/lib/cmake/rccl)
    set(rocprim_DIR ${ROCM_PATH}/lib/cmake/rocprim)
    set(hipcub_DIR ${ROCM_PATH}/lib/cmake/hipcub)
    set(rocthrust_DIR ${ROCM_PATH}/lib/cmake/rocthrust)
    set(ROCclr_DIR ${ROCM_PATH}/rocclr/lib/cmake/rocclr)
    set(ROCRAND_INCLUDE ${ROCM_PATH}/include)
    set(ROCM_SMI_INCLUDE ${ROCM_PATH}/include)
  else()
    message(FATAL_ERROR "\n***** The minimal ROCm version is 5.1.0 but have ${ROCM_VERSION_DEV} installed *****\n")
  endif()

  find_package(hip REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(hipfft REQUIRED)
  find_package(hiprand REQUIRED)
  find_package(rocrand REQUIRED)
  find_package(hipsparse REQUIRED)
  find_package(rocprim REQUIRED)

  if(HIP_COMPILER STREQUAL clang)
    set(hip_library_name amdhip64)
  else()
    set(hip_library_name hip_hcc)
  endif()
  message("HIP library name: ${hip_library_name}")

  set(CMAKE_HCC_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  set(CMAKE_HCC_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  FIND_LIBRARY(FBGEMM_HIP_HCC_LIBRARIES ${hip_library_name} HINTS ${HIP_PATH}/lib)

  list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_OPERATORS__=1)
  # list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
  list(APPEND HIP_CXX_FLAGS -D__HIP_NO_BFLOAT16_CONVERSIONS__=1)
  list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF2_OPERATORS__=1)
  list(APPEND HIP_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  list(APPEND HIP_CXX_FLAGS -mavx2)
  list(APPEND HIP_CXX_FLAGS -mf16c)
  list(APPEND HIP_CXX_FLAGS -mfma)
  list(APPEND HIP_CXX_FLAGS -std=c++17)

  set(HIP_HCC_FLAGS ${HIP_CXX_FLAGS})
  # Ask hcc to generate device code during compilation so we can use
  # host linker to link.
  list(APPEND HIP_HCC_FLAGS -fno-gpu-rdc)
  list(APPEND HIP_HCC_FLAGS -Wno-defaulted-function-deleted)
  foreach(fbgemm_rocm_arch ${FBGEMM_ROCM_ARCH})
    list(APPEND HIP_HCC_FLAGS --offload-arch=${fbgemm_rocm_arch})
  endforeach()

  set(FBGEMM_HIP_INCLUDE ${ROCM_PATH}/include ${FBGEMM_HIP_INCLUDE})
  set(FBGEMM_HIP_INCLUDE ${hip_INCLUDE_DIRS} $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> $<INSTALL_INTERFACE:include> ${FBGEMM_HIP_INCLUDE})

  hip_include_directories(${FBGEMM_HIP_INCLUDE} ${ROCRAND_INCLUDE} ${ROCM_SMI_INCLUDE})

  list (APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH})
  set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})

ELSE()
  message("Not able to find HIP installation.")
ENDIF()
