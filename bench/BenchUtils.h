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
#include "fbgemm/FbgemmBuild.h"

namespace fbgemm {

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high);

void llc_flush(std::vector<char>& llc);

// Same as omp_get_max_threads() when OpenMP is available, otherwise 1
int fbgemm_get_max_threads();
// Same as omp_get_num_threads() when OpenMP is available, otherwise 1
int fbgemm_get_num_threads();
// Same as omp_get_thread_num() when OpenMP is available, otherwise 0
int fbgemm_get_thread_num();

template <typename T>
NOINLINE
float cache_evict(const T& vec) {
  auto const size = vec.size();
  auto const elemSize = sizeof(typename T::value_type);
  auto const dataSize = size * elemSize;

  const char* data = reinterpret_cast<const char*>(vec.data());
  constexpr int CACHE_LINE_SIZE = 64;
  // Not having this dummy computation significantly slows down the computation
  // that follows.
  float dummy = 0.0f;
  for (std::size_t i = 0; i < dataSize; i += CACHE_LINE_SIZE) {
    dummy += data[i] * 1.0f;
    _mm_mfence();
#ifndef _MSC_VER
    asm volatile("" ::: "memory");
#endif
    _mm_clflush(&data[i]);
  }

  return dummy;
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

/*
 * @brief Out-of-place transposition for M*N matrix ref.
 * @param M number of rows in input
 * @param K number of columns in input
 */
template <typename T>
void transpose_matrix(
    int M,
    int N,
    const T* src,
    int ld_src,
    T* dst,
    int ld_dst) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      dst[i * ld_dst + j] = src[i + j * ld_src];
    }
  } // for each output row
}

/*
 * @brief In-place transposition for nxk matrix ref.
 * @param n number of rows in input (number of columns in output)
 * @param k number of columns in input (number of rows in output)
 */
template <typename T>
void transpose_matrix(T* ref, int n, int k) {
  std::vector<T> local(n * k);
  transpose_matrix(n, k, ref, k, local.data(), n);
  memcpy(ref, local.data(), n * k * sizeof(T));
}

} // namespace fbgemm
