/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "./BenchUtils.h"

#include <cstring>
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
  std::uniform_int_distribution<int> dis(low, high);
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
//template void
//randFill<int64_t>(aligned_vector<int64_t>& vec, int64_t low, int64_t high);
template <>
void randFill(aligned_vector<int64_t>& vec, int64_t low, int64_t high) {
  std::uniform_int_distribution<int64_t> dis(low, high);
  std::generate(vec.begin(), vec.end(), [&] { return dis(eng); });
}

void llc_flush(std::vector<char>& llc) {
  volatile char* data = llc.data();
  for (auto i = 0; i < llc.size(); i++) {
    data[i]++;
  }
}

int fbgemm_get_max_threads() {
#if defined(FBGEMM_MEASURE_TIME_BREAKDOWN) || !defined(_OPENMP)
  return 1;
#else
  return omp_get_max_threads();
#endif
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

int parseArgumentInt(
    int argc,
    const char* argv[],
    const char* arg,
    int non_exist_val,
    int def_val) {
  int val = non_exist_val;
  int arg_len = strlen(arg);
  for (auto i = 1; i < argc; ++i) {
    const char* ptr = strstr(argv[i], arg);
    if (ptr) {
      int res = atoi(ptr + arg_len);
      val = (*(ptr + arg_len - 1) == '=') ? res : def_val;
      break;
    }
  }
  return val;
}

bool parseArgumentBool(
    int argc,
    const char* argv[],
    const char* arg,
    bool def_val) {
  for (auto i = 1; i < argc; ++i) {
    const char* ptr = strstr(argv[i], arg);
    if (ptr) {
      return true;
    }
  }
  return def_val;
}

aligned_vector<float> getRandomSparseVector(
    unsigned size,
    float fractionNonZeros) {
  aligned_vector<float> res(size);

  std::mt19937 gen(345);

  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (auto& f : res) {
    f = dis(gen);
  }

  // Create exactly fractionNonZeros in result
  aligned_vector<float> sorted_res(res);
  std::sort(sorted_res.begin(), sorted_res.end());
  int32_t numZeros =
      size - static_cast<int32_t>(std::round(size * fractionNonZeros));
  float thr;
  if (numZeros) {
    thr = sorted_res[numZeros - 1];

    for (auto& f : res) {
      if (f <= thr) {
        f = 0.0f;
      }
    }
  }

  return res;
}

} // namespace fbgemm
