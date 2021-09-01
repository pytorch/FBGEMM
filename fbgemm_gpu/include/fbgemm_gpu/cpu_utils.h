/*
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <omp.h>
#include <cstdint>
#include <utility>
#include <immintrin.h>
#include <ATen/ATen.h>
#include <fbgemm/Utils.h>

template <typename T>
using Radix_Sort_Pair = std::pair<T, T>;
// histogram size per thread
const int RDX_HIST_SIZE = 256;
template <typename T>
Radix_Sort_Pair<T>* radix_sort_parallel(
    Radix_Sort_Pair<T>* inp_buf,
    Radix_Sort_Pair<T>* tmp_buf,
    int64_t elements_count,
    int64_t max_value) {
  int maxthreads = omp_get_max_threads();
  alignas(64) int histogram[RDX_HIST_SIZE * maxthreads],
      histogram_ps[RDX_HIST_SIZE * maxthreads + 1];
  if (max_value == 0)
    return inp_buf;
  int num_bits = sizeof(T) * 8 - __builtin_clz(max_value);
  unsigned int num_passes = (num_bits + 7) / 8;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int* local_histogram = &histogram[RDX_HIST_SIZE * tid];
    int* local_histogram_ps = &histogram_ps[RDX_HIST_SIZE * tid];
    int elements_count_4 = elements_count / 4 * 4;
    Radix_Sort_Pair<T>* input = inp_buf;
    Radix_Sort_Pair<T>* output = tmp_buf;

    for (unsigned int pass = 0; pass < num_passes; pass++) {
      // Step 1: compute histogram
      for (int i = 0; i < RDX_HIST_SIZE; i++)
        local_histogram[i] = 0;

#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        T val_1 = input[i].first;
        T val_2 = input[i + 1].first;
        T val_3 = input[i + 2].first;
        T val_4 = input[i + 3].first;

        local_histogram[(val_1 >> (pass * 8)) & 0xFF]++;
        local_histogram[(val_2 >> (pass * 8)) & 0xFF]++;
        local_histogram[(val_3 >> (pass * 8)) & 0xFF]++;
        local_histogram[(val_4 >> (pass * 8)) & 0xFF]++;
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          T val = input[i].first;
          local_histogram[(val >> (pass * 8)) & 0xFF]++;
        }
      }
#pragma omp barrier
      // Step 2: prefix sum
      if (tid == 0) {
        int sum = 0, prev_sum = 0;
        for (int bins = 0; bins < RDX_HIST_SIZE; bins++)
          for (int t = 0; t < nthreads; t++) {
            sum += histogram[t * RDX_HIST_SIZE + bins];
            histogram_ps[t * RDX_HIST_SIZE + bins] = prev_sum;
            prev_sum = sum;
          }
        histogram_ps[RDX_HIST_SIZE * nthreads] = prev_sum;
        if (prev_sum != elements_count) {
        }
      }
#pragma omp barrier

      // Step 3: scatter
#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        T val_1 = input[i].first;
        T val_2 = input[i + 1].first;
        T val_3 = input[i + 2].first;
        T val_4 = input[i + 3].first;
        T bin_1 = (val_1 >> (pass * 8)) & 0xFF;
        T bin_2 = (val_2 >> (pass * 8)) & 0xFF;
        T bin_3 = (val_3 >> (pass * 8)) & 0xFF;
        T bin_4 = (val_4 >> (pass * 8)) & 0xFF;
        int pos;
        pos = local_histogram_ps[bin_1]++;
        output[pos] = input[i];
        pos = local_histogram_ps[bin_2]++;
        output[pos] = input[i + 1];
        pos = local_histogram_ps[bin_3]++;
        output[pos] = input[i + 2];
        pos = local_histogram_ps[bin_4]++;
        output[pos] = input[i + 3];
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          T val = input[i].first;
          int pos = local_histogram_ps[(val >> (pass * 8)) & 0xFF]++;
          output[pos] = input[i];
        }
      }

      Radix_Sort_Pair<T>* temp = input;
      input = output;
      output = temp;
#pragma omp barrier
    }
  }
  return (num_passes % 2 == 0 ? inp_buf : tmp_buf);
}

namespace fbgemm_gpu {

  template <typename T>
inline float toFloat(T val) {
  float ret = float(val);
  return ret;
}

template <>
inline float toFloat(at::Half val) {
  float ret = _cvtsh_ss(val);
  return ret;
}
template <typename T1, typename T2>
inline void madd_ker_ref(T1* inout, T2* in, int len, float alpha) {
#pragma omp simd
  for (long v = 0; v < len; v++) {
    inout[v] += toFloat(in[v]) * alpha;
  }
}

template <typename T1, typename T2>
inline void madd_ker(T1* inout, T2* in, int len, float alpha) {
  madd_ker_ref(inout, in, len, alpha);
}

template <>
inline void madd_ker(at::Half* inout, at::Half* in, int len, float alpha) {
  static bool a512 = fbgemm::fbgemmHasAvx512Support();
  if (!a512) madd_ker_ref(inout, in, len, alpha);
  __m512 vAlpha = _mm512_set1_ps(alpha);
  int i = 0;
  for (; i < len - 15; i += 16) {
    __m512 y1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(inout + i)));
    __m512 y2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(in + i)));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm256_storeu_si256(
        (__m256i*)(inout + i),
        _mm512_cvtps_ph(y1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  if (i < len) {
    int rem = len - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 y1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, inout + i));
    __m512 y2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, in + i));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm256_mask_storeu_epi16(
        inout + i,
        mask,
        _mm512_cvtps_ph(y1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
}

template <>
inline void madd_ker(float* inout, at::Half* in, int len, float alpha) {
  static bool a512 = fbgemm::fbgemmHasAvx512Support();
  if (!a512) madd_ker_ref(inout, in, len, alpha);
  __m512 vAlpha = _mm512_set1_ps(alpha);
  int i = 0;
  for (; i < len - 15; i += 16) {
    __m512 y1 = _mm512_loadu_ps(inout + i);
    __m512 y2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(in + i)));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_storeu_ps(inout + i, y1);
  }
  if (i < len) {
    int rem = len - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 y1 = _mm512_maskz_loadu_ps(mask, inout + i);
    __m512 y2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, in + i));
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_mask_storeu_ps(inout + i, mask, y1);
  }
}

template <>
inline void madd_ker(float* inout, float* in, int len, float alpha) {
  static bool a512 = fbgemm::fbgemmHasAvx512Support();
  if (!a512) madd_ker_ref(inout, in, len, alpha);
  __m512 vAlpha = _mm512_set1_ps(alpha);
  int i = 0;
  for (; i < len - 15; i += 16) {
    __m512 y1 = _mm512_loadu_ps(inout + i);
    __m512 y2 = _mm512_loadu_ps(in + i);
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_storeu_ps(inout + i, y1);
  }
  if (i < len) {
    int rem = len - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 y1 = _mm512_maskz_loadu_ps(mask, inout + i);
    __m512 y2 = _mm512_maskz_loadu_ps(mask, in + i);
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm512_mask_storeu_ps(inout + i, mask, y1);
  }
}

template <typename T>
static inline T sq_reduce_ker(T* in, int len) {
  T g_local_sum_square = 0.0;
#pragma omp simd reduction(+ : g_local_sum_square)
  for (int d = 0; d < len; d++) {
    g_local_sum_square += in[d] * in[d];
  }
  return g_local_sum_square;
}

static inline c10::Half sq_reduce_ker(c10::Half* in, int len) {
  c10::Half g_local_sum_square = 0.0;
  for (int d = 0; d < len; d++) {
    g_local_sum_square += in[d] * in[d];
  }
  return g_local_sum_square;
}

void init_threadlocal_rnd_state(unsigned int seed=0x42);
__m256i _mm512_cvtps_ph_stoc(__m512 src);

template <typename T1, typename T2>
inline void madd_ker_stochastic_ref(T1* inout, T2* in, int len, float alpha) {
#pragma omp simd
  for (long v = 0; v < len; v++) {
    inout[v] += toFloat(in[v]) * alpha;
  }
}
template <typename T1, typename T2>
inline void madd_ker_stochastic(T1* inout, T2* in, int len, float alpha) {
  madd_ker_stochastic_ref(inout, in, len, alpha);
}

template <>
inline void madd_ker_stochastic(
    at::Half* inout,
    float* in,
    int len,
    float alpha) {
  static bool a512 = fbgemm::fbgemmHasAvx512Support();
  if (!a512) madd_ker_stochastic_ref(inout, in, len, alpha);
  init_threadlocal_rnd_state();
  __m512 vAlpha = _mm512_set1_ps(alpha);
  int i = 0;
  for (; i < len - 15; i += 16) {
    __m512 y1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(inout + i)));
    __m512 y2 = _mm512_loadu_ps(in + i);
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm256_storeu_si256((__m256i*)(inout + i), _mm512_cvtps_ph_stoc(y1));
  }
  if (i < len) {
    int rem = len - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 y1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, inout + i));
    __m512 y2 = _mm512_maskz_loadu_ps(mask, in + i);
    y1 = _mm512_fmadd_ps(vAlpha, y2, y1);
    _mm256_mask_storeu_epi16(inout + i, mask, _mm512_cvtps_ph_stoc(y1));
  }
}
} // namespace fbgemm_gpu
