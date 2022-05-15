set(FBGEMM_HAVE_HIP FALSE)

IF(NOT DEFINED ENV{ROCM_PATH})
  SET(ROCM_PATH /opt/rocm)
ELSE()
  SET(ROCM_PATH $ENV{ROCM_PATH})
ENDIF()

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

  if(HIP_COMPILER STREQUAL clang)
    set(hip_library_name amdhip64)
  else()
    set(hip_library_name hip_hcc)
  endif()
  message("HIP library name: ${hip_library_name}")

  find_package(hip REQUIRED)
  find_package(rocBLAS REQUIRED)
  find_package(hipFFT REQUIRED)
  find_package(hipRAND REQUIRED)
  find_package(rocRAND REQUIRED)
  find_package(hipSPARSE REQUIRED)
  find_package(OpenMP REQUIRED)
  find_package(rocPRIM REQUIRED)

  set(CMAKE_HCC_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  set(CMAKE_HCC_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  FIND_LIBRARY(FBGEMM_HIP_HCC_LIBRARIES ${hip_library_name} HINTS ${HIP_PATH}/lib)

  list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_OPERATORS__=1)
  # list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
  list(APPEND HIP_CXX_FLAGS -D__HIP_NO_BFLOAT16_CONVERSIONS__=1)
  list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF2_OPERATORS__=1)
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
    list(APPEND HIP_HCC_FLAGS --amdgpu-target=${fbgemm_rocm_arch})
  endforeach()

  set(hip_DIR ${HIP_PATH}/lib/cmake/hip)
  set(hsa-runtime64_DIR ${ROCM_PATH}/lib/cmake/hsa-runtime64)
  set(AMDDeviceLibs_DIR ${ROCM_PATH}/lib/cmake/AMDDeviceLibs)
  set(amd_comgr_DIR ${ROCM_PATH}/lib/cmake/amd_comgr)
  set(rocrand_DIR ${ROCRAND_PATH}/lib/cmake/rocrand)
  set(hiprand_DIR ${HIPRAND_PATH}/lib/cmake/hiprand)
  set(rocblas_DIR ${ROCBLAS_PATH}/lib/cmake/rocblas)
  set(miopen_DIR ${MIOPEN_PATH}/lib/cmake/miopen)
  set(rocfft_DIR ${ROCFFT_PATH}/lib/cmake/rocfft)
  set(hipfft_DIR ${HIPFFT_PATH}/lib/cmake/hipfft)
  set(hipsparse_DIR ${HIPSPARSE_PATH}/lib/cmake/hipsparse)
  set(rccl_DIR ${RCCL_PATH}/lib/cmake/rccl)
  set(rocprim_DIR ${ROCPRIM_PATH}/lib/cmake/rocprim)
  set(hipcub_DIR ${HIPCUB_PATH}/lib/cmake/hipcub)
  set(rocthrust_DIR ${ROCTHRUST_PATH}/lib/cmake/rocthrust)
  set(ROCclr_DIR ${ROCM_PATH}/rocclr/lib/cmake/rocclr)
  set(ROCRAND_INCLUDE ${ROCRAND_PATH}/include)
  set(ROCM_SMI_INCLUDE ${ROCM_PATH}/rocm_smi/include)

  set(FBGEMM_HIP_INCLUDE ${ROCM_PATH}/include ${FBGEMM_HIP_INCLUDE})
  set(FBGEMM_HIP_INCLUDE ${hip_INCLUDE_DIRS} $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> $<INSTALL_INTERFACE:include> ${FBGEMM_HIP_INCLUDE})

  hip_include_directories(${FBGEMM_HIP_INCLUDE} ${ROCRAND_INCLUDE} ${ROCM_SMI_INCLUDE})

  list (APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH})
  set(CMAKE_MODULE_PATH ${HIP_PATH}/cmake ${CMAKE_MODULE_PATH})

ELSE()
  message("Not able to find HIP installation.")
ENDIF()
