/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <chrono>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "./AlignedVec.h"

namespace fbgemm {

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high);

aligned_vector<float> getRandomSparseVector(
    unsigned size,
    float fractionNonZeros = 1.0);

void llc_flush(std::vector<char>& llc);

int fbgemm_get_num_threads();
int fbgemm_get_thread_num();

/**
 * @param llc if not nullptr, flush llc
 */
template <class Fn>
double measureWithWarmup(
    Fn&& fn,
    int warmupIterations,
    int measuredIterations,
    std::vector<char>* llc = nullptr,
    bool useOpenMP = false) {
  for (int i = 0; i < warmupIterations; ++i) {
    if (llc) {
      llc_flush(*llc);
    }
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
    if (llc && thread_id == 0) {
      llc_flush(*llc);
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
