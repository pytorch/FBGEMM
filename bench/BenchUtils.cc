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
template void
randFill<int64_t>(aligned_vector<int64_t>& vec, int64_t low, int64_t high);

aligned_vector<float> getRandomSparseVector(
    unsigned size,
    float fractionNonZeros /*= 1.0*/) {
  aligned_vector<float> res(size);

  std::random_device rd;
  std::mt19937 gen(345);

  std::uniform_real_distribution<float> dis(0.0, 1.0);

  for (auto& f : res) {
    if (dis(gen) <= fractionNonZeros) {
      f = dis(gen);
    } else {
      f = 0;
    }
  }

  return res;
}

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
