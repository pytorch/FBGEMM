/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <chrono>
#include <vector>
#include "AlignedVec.h"

namespace fbgemm {

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high);

void llc_flush(std::vector<char>& llc);

int fbgemm_get_num_threads();
int fbgemm_get_thread_num();

/**
 * @params llc if not nullptr, flush llc
 */
template <class Fn>
double measureWithWarmup(
    Fn&& fn,
    int warmupIterations,
    int measuredIterations,
    std::vector<char>* llc = nullptr) {
  for (int i = 0; i < warmupIterations; ++i) {
    if (llc) {
      llc_flush(*llc);
    }
    fn();
  }

  double ttot = 0.0;
  std::chrono::time_point<std::chrono::high_resolution_clock>
      start = std::chrono::high_resolution_clock::now(),
      end;

  for (int i = 0; i < measuredIterations; ++i) {
    if (llc) {
      llc_flush(*llc);
    }
    start = std::chrono::high_resolution_clock::now();
    fn();
    end = std::chrono::high_resolution_clock::now();

    auto dur =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    ttot += dur.count();
  }

  return ttot / 1e9 / measuredIterations;
}

} // namespace fbgemm
