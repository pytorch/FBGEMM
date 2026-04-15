# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# FBGEMM (not FBGEMM_GPU) Sources
################################################################################

# Embedding / quantization sources from FBGEMM (CPU)
set(fbgemm_sources_normal
  "${FBGEMM}/src/EmbeddingSpMDM.cc"
  "${FBGEMM}/src/EmbeddingSpMDMNBit.cc"
  "${FBGEMM}/src/QuantUtils.cc"
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
  "${FBGEMM}/src/QuantUtilsAvx2.cc")
set(fbgemm_sources_avx512
  "${FBGEMM}/src/EmbeddingSpMDMAvx512.cc"
  "${FBGEMM}/src/QuantUtilsAvx512.cc")

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
  INCLUDE_DIRS
    ${fbgemm_sources_include_directories}
  OTHER_SRCS
    ${fbgemm_sources}
  DEPS
    asmjit
  DESTINATION
    fbgemm_gpu)
