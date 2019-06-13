/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "BenchUtils.h"

#include <algorithm>
#include <random>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fbgemm {

std::default_random_engine eng;

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high, std::true_type) {
  std::uniform_int_distribution<T> dis(low, high);
  std::generate(vec.begin(), vec.end(), [&] { return dis(eng); });
}

// MSVC doesn't accept uint8_t and int8_t as a template argument.
#ifdef _MSC_VER
void randFill(aligned_vector<uint8_t>& vec, uint8_t low, uint8_t high, std::true_type) {
  std::uniform_int_distribution<unsigned short> dis(low, high);
  for (int i = 0; i < vec.size(); i++)
    vec[i] = (uint8_t)dis(eng);
}

void randFill(aligned_vector<int8_t>& vec, int8_t low, int8_t high, std::true_type) {
  std::uniform_int_distribution<short> dis(low, high);
  for (int i = 0; i < vec.size(); i++)
    vec[i] = (int8_t)dis(eng);
}
#endif

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high, std::false_type) {
  std::uniform_real_distribution<T> dis(low, high);
  std::generate(vec.begin(), vec.end(), [&] { return dis(eng); });
}

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high) {
  randFill(vec, low, high, std::is_integral<T>());
}

template void
randFill<float>(aligned_vector<float>& vec, float low, float high);
template void
randFill<uint8_t>(aligned_vector<uint8_t>& vec, uint8_t low, uint8_t high);
template void
randFill<int8_t>(aligned_vector<int8_t>& vec, int8_t low, int8_t high);
template void randFill<int>(aligned_vector<int>& vec, int low, int high);

void llc_flush(std::vector<char>& llc) {
  volatile char* data = llc.data();
  for (int i = 0; i < llc.size(); i++) {
    data[i]++;
  }
}

int fbgemm_get_num_threads() {
#if defined(FBGEMM_MEASURE_TIME_BREAKDOWN) || !defined(_OPENMP)
  return 1;
#else
  return omp_get_num_threads();
#endif
}

int fbgemm_get_thread_num() {
#if defined(FBGEMM_MEASURE_TIME_BREAKDOWN) || !defined(_OPENMP)
  return 0;
#else
  return omp_get_thread_num();
#endif
}

} // namespace fbgemm
