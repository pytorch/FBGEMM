# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


################################################################################
# Finds and sets AVX compilation flags
# This file is copied over from the PyTorch repo
################################################################################

include(CheckCompilerFlag)
INCLUDE(CheckSourceRuns)
INCLUDE(CMakePushCheckState)

SET(AVX_CODE "
  #include <immintrin.h>

  int main()
  {
    __m256 a;
    a = _mm256_set1_ps(0);
    return 0;
  }
")

SET(AVX512_CODE "
  #include <immintrin.h>

  int main()
  {
    __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0);
    __m512i b = a;
    __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
    return 0;
  }
")

SET(AVX2_CODE "
  #include <immintrin.h>

  int main()
  {
    __m256i a = {0};
    a = _mm256_abs_epi16(a);
    __m256i x;
    _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
    return 0;
  }
")

MACRO(CHECK_SSE lang type flags)
  SET(__FLAG_I 1)
  FOREACH(__FLAG ${flags})
    IF(NOT ${lang}_${type}_FOUND)
      cmake_push_check_state(RESET)
      unset(${lang}_HAS_${type}_${__FLAG_I} CACHE)
      check_compiler_flag(${lang} ${__FLAG} ${lang}_HAS_${type}_${__FLAG_I})
      cmake_pop_check_state()
      IF(NOT ${lang}_HAS_${type}_${__FLAG_I})
        MATH(EXPR __FLAG_I "${__FLAG_I}+1")
        CONTINUE()
      ENDIF()
      IF(${CHECK_AVX_COMPILE})
        SET(${lang}_${type}_FOUND TRUE CACHE BOOL "${lang} ${type} support")
        string(REPLACE " " ";" __FLAG "${__FLAG}")
        SET(${lang}_${type}_FLAGS "${__FLAG}" CACHE STRING "${lang} ${type} flags")
        MATH(EXPR __FLAG_I "${__FLAG_I}+1")
        CONTINUE()
      ENDIF()

      cmake_push_check_state(RESET)
      unset(${lang}_HAS_${type}_${__FLAG_I} CACHE)
      SET(CMAKE_REQUIRED_FLAGS ${__FLAG})
      CHECK_SOURCE_RUNS(${lang} "${${type}_CODE}" ${lang}_HAS_${type}_${__FLAG_I})
      IF(${lang}_HAS_${type}_${__FLAG_I})
        SET(${lang}_${type}_FOUND TRUE CACHE BOOL "${lang} ${type} support")
        string(REPLACE " " ";" __FLAG "${__FLAG}")
        SET(${lang}_${type}_FLAGS "${__FLAG}" CACHE STRING "${lang} ${type} flags")
      ENDIF()
      MATH(EXPR __FLAG_I "${__FLAG_I}+1")
      cmake_pop_check_state()
    ENDIF()
  ENDFOREACH()

  SET(${lang}_${type}_FOUND "${${lang}_${type}_FOUND}" CACHE BOOL "${lang} ${type} support")
  SET(${lang}_${type}_FLAGS "${${lang}_${type}_FLAGS}" CACHE STRING "${lang} ${type} flags")

  MARK_AS_ADVANCED(${lang}_${type}_FOUND ${lang}_${type}_FLAGS)
ENDMACRO()


get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
foreach(lang C;CXX)
  if(lang IN_LIST languages)
    if(MSVC)
      CHECK_SSE(${lang} "AVX" "/arch:AVX;")
      CHECK_SSE(${lang} "AVX2" "/arch:AVX2;")
      CHECK_SSE(${lang} "AVX512" "/arch:AVX512;")
    else()
      CHECK_SSE(${lang} "AVX" "-mavx;")
      CHECK_SSE(${lang} "AVX2" "-mavx2 -mfma -mf16c;")
      CHECK_SSE(${lang} "AVX512" "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma;")
    endif()
  endif()
endforeach()

MESSAGE(STATUS "CXX_AVX_FLAGS: ${CXX_AVX_FLAGS}")
MESSAGE(STATUS "CXX_AVX2_FLAGS: ${CXX_AVX2_FLAGS}")
MESSAGE(STATUS "CXX_AVX512_FLAGS: ${CXX_AVX512_FLAGS}")
