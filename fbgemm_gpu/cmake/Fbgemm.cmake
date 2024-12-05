# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################################################################
# FBGEMM (not FBGEMM_GPU) Sources
################################################################################

set(fbgemm_sources_normal
  "${FBGEMM}/src/EmbeddingSpMDM.cc"
  "${FBGEMM}/src/EmbeddingSpMDMAutovec.cc"
  "${FBGEMM}/src/EmbeddingSpMDMNBit.cc"
  "${FBGEMM}/src/QuantUtils.cc"
  "${FBGEMM}/src/RefImplementations.cc"
  "${FBGEMM}/src/RowWiseSparseAdagradFused.cc"
  "${FBGEMM}/src/SparseAdagrad.cc"
  "${FBGEMM}/src/Utils.cc")

set(fbgemm_sources_avx2
  "${FBGEMM}/src/EmbeddingSpMDMAvx2.cc"
  "${FBGEMM}/src/QuantUtilsAvx2.cc")

set(fbgemm_sources_avx512
  "${FBGEMM}/src/EmbeddingSpMDMAvx512.cc")

if(CXX_AVX2_FOUND)
  set_source_files_properties(${fbgemm_sources_avx2}
    PROPERTIES COMPILE_OPTIONS
    "${AVX2_FLAGS}")
endif()

if(CXX_AVX512_FOUND)
  set_source_files_properties(${fbgemm_sources_avx512}
    PROPERTIES COMPILE_OPTIONS
    "${AVX512_FLAGS}")
endif()

set(fbgemm_sources ${fbgemm_sources_normal})
if(CXX_AVX2_FOUND)
  set(fbgemm_sources
    ${fbgemm_sources}
    ${fbgemm_sources_avx2})
endif()
if(NOT USE_ROCM AND CXX_AVX512_FOUND)
  set(fbgemm_sources
    ${fbgemm_sources}
    ${fbgemm_sources_avx2}
    ${fbgemm_sources_avx512})
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_AVX512=1")
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
