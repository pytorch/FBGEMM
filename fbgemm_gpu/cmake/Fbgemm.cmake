# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(CheckCXXCompilerFlag)

################################################################################
# FBGEMM (not FBGEMM_GPU) Sources
################################################################################

# Embedding / quantization sources from FBGEMM (CPU)
set(fbgemm_sources_normal
  "${FBGEMM}/src/EmbeddingSpMDM.cc"
  "${FBGEMM}/src/EmbeddingSpMDMNBit.cc"
  "${FBGEMM}/src/FbgemmBfloat16Convert.cc"
  "${FBGEMM}/src/FbgemmFloat16Convert.cc"
  "${FBGEMM}/src/JitPerfMap.cc"
  "${FBGEMM}/src/QuantUtils.cc"
  "${FBGEMM}/src/QuantUtilsNeon.cc"
  "${FBGEMM}/src/RefImplementations.cc"
  "${FBGEMM}/src/RowWiseSparseAdagradFused.cc"
  "${FBGEMM}/src/SparseAdagrad.cc"
  "${FBGEMM}/src/Utils.cc")
if(NOT DISABLE_FBGEMM_AUTOVEC)
  list(APPEND fbgemm_sources_normal
    "${FBGEMM}/src/EmbeddingSpMDMAutovec.cc"
    "${FBGEMM}/src/EmbeddingStatsTracker.cc")
endif()

set(fbgemm_sources_avx2
  "${FBGEMM}/src/EmbeddingSpMDMAvx2.cc"
  "${FBGEMM}/src/FbgemmBfloat16ConvertAvx2.cc"
  "${FBGEMM}/src/FbgemmFloat16ConvertAvx2.cc"
  "${FBGEMM}/src/QuantUtilsAvx2.cc")
set(fbgemm_sources_avx512
  "${FBGEMM}/src/EmbeddingSpMDMAvx512.cc"
  "${FBGEMM}/src/FbgemmBfloat16ConvertAvx512.cc"
  "${FBGEMM}/src/FbgemmFloat16ConvertAvx512.cc"
  "${FBGEMM}/src/QuantUtilsAvx512.cc")
set(fbgemm_sources_neon
  "${FBGEMM}/src/FbgemmFloat16ConvertNEON.cc")

# Assemble combined source list based on available ISA support
set(fbgemm_sources ${fbgemm_sources_normal})
if(CXX_AVX2_FOUND)
  list(APPEND fbgemm_sources ${fbgemm_sources_avx2})
  if(MSVC)
    set_source_files_properties(${fbgemm_sources_avx2}
      PROPERTIES COMPILE_OPTIONS "${CXX_AVX2_FLAGS}")
  else()
    set_source_files_properties(${fbgemm_sources_avx2}
      PROPERTIES COMPILE_OPTIONS "-mfma;${CXX_AVX2_FLAGS}")
  endif()
endif()
if(CXX_AVX512_FOUND)
  list(APPEND fbgemm_sources ${fbgemm_sources_avx512})
  if(MSVC)
    set_source_files_properties(${fbgemm_sources_avx512}
      PROPERTIES COMPILE_OPTIONS "${CXX_AVX512_FLAGS}")
  else()
    set_source_files_properties(${fbgemm_sources_avx512}
      PROPERTIES COMPILE_OPTIONS "-mfma;${CXX_AVX512_FLAGS}")
  endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
  check_cxx_compiler_flag(-march=armv8.2-a+fp16fml COMPILER_SUPPORTS_FP16FML)
  if(COMPILER_SUPPORTS_FP16FML)
    set(fbgemm_fml_flags "-march=armv8.2-a+fp16fml")
  endif()
  if(NOT COMPILER_SUPPORTS_FP16FML) # for clang
    check_cxx_compiler_flag(-mfp16fml COMPILER_SUPPORTS_FP16FML)
    if(COMPILER_SUPPORTS_FP16FML)
      set(fbgemm_fml_flags "-march=armv8.2;-mfp16fml")
    endif()
  endif()
  if(NOT COMPILER_SUPPORTS_FP16FML)
    message(FATAL_ERROR "The current compiler does not support building FP16FML code")
  endif()

  list(APPEND fbgemm_sources ${fbgemm_sources_neon})
  set_source_files_properties(${fbgemm_sources_neon}
    PROPERTIES COMPILE_OPTIONS "${fbgemm_fml_flags}")
endif()

set_source_files_properties(${fbgemm_sources} PROPERTIES
  INCLUDE_DIRECTORIES "${fbgemm_sources_include_directories}")


################################################################################
# Build Intermediate Target (Static)
################################################################################

gpu_cpp_library(
  PREFIX
    fbgemm
  TYPE
    SHARED
  CC_FLAGS
    ${FBGEMM_AARCH64_NATIVE_COMPILE_OPTIONS}
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  SYSTEM_INCLUDE_DIRS
    ${fbgemm_thirdparty_include_directories}
  OTHER_SRCS
    ${fbgemm_sources}
  DEPS
    asmjit
  DESTINATION
    fbgemm_gpu)
