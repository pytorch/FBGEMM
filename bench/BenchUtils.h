/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <chrono>
#include <functional>
#include <vector>

#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include "./AlignedVec.h"

namespace fbgemm {

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high);

void llc_flush(std::vector<char>& llc);

int fbgemm_get_num_threads();
int fbgemm_get_thread_num();

template <typename T>
void cache_evict(const T& vec) {
  auto const size = vec.size();
  auto const elemSize = sizeof(typename T::value_type);
  auto const dataSize = size * elemSize;

  const char* data = reinterpret_cast<const char*>(vec.data());
  constexpr int CACHE_LINE_SIZE = 64;
  for (auto i = 0; i < dataSize; i += CACHE_LINE_SIZE) {
    _mm_clflush(&data[i]);
  }
}

/**
 * Parse application command line arguments
 *
 */
int parseArgumentInt(
    int argc,
    const char* argv[],
    const char* arg,
    int non_exist_val,
    int def_val);
bool parseArgumentBool(
    int argc,
    const char* argv[],
    const char* arg,
    bool def_val);

namespace {
struct empty_flush {
  void operator()() const {}
};
} // namespace
/**
 * @param Fn functor to execute
 * @param Fe data eviction functor
 */
template <class Fn, class Fe = std::function<void()>>
double measureWithWarmup(
    Fn&& fn,
    int warmupIterations,
    int measuredIterations,
    const Fe& fe = empty_flush(),
    bool useOpenMP = false) {
  for (int i = 0; i < warmupIterations; ++i) {
    // Evict data first
    fe();
    fn();
  }

  double ttot = 0.0;

#ifdef _OPENMP
#pragma omp parallel if (useOpenMP)
#endif
  for (int i = 0; i < measuredIterations; ++i) {
    int thread_id = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

#ifdef _OPENMP
    if (useOpenMP) {
      thread_id = omp_get_thread_num();
    }
#endif

    if (thread_id == 0) {
      fe();
    }

#ifdef _OPENMP
    if (useOpenMP) {
#pragma omp barrier
    }
#endif
    start = std::chrono::high_resolution_clock::now();

    fn();

    end = std::chrono::high_resolution_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    if (thread_id == 0) {
      // TODO: measure load imbalance
      ttot += dur.count();
    }
  }

  return ttot / 1e9 / measuredIterations;
}

} // namespace fbgemm
