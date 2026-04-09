/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "./BenchUtils.h"
#include "fbgemm/QuantUtils.h"
#include "fbgemm/Types.h"

using namespace std;
using namespace fbgemm;

// T is the type of scale and bias
template <typename T>
static void performance_test() {
  constexpr int NWARMUP = 4;
  constexpr int NITER = 256;

  if constexpr (is_same_v<T, float16>) {
    cout << "With scale and bias as float16" << '\n';
  } else {
    cout << "With scale and bias as float" << '\n';
  }
  cout << setw(8) << "bit_rate" << ", " << setw(6) << "rows" << "," << setw(6)
       << "cols" << "," << setw(16) << "elems_per_usec" << "," << setw(10)
       << "GB/Sec" << '\n';
  std::vector<int> bit_rates;
  if constexpr (is_same_v<T, float16>) {
    bit_rates = {2, 4, 8};
  } else {
    // float
    bit_rates = {8};
  }
  for (int bit_rate : bit_rates) {
    for (int rowSize : {100, 120, 1000}) {
      for (int colSize : {16, 64, 128, 256, 512, 1024, 2048}) {
        aligned_vector<float> inpVec(rowSize * colSize);
        randFill<float>(inpVec, -10.0f, 10.0f);

        int out_emb_cols = colSize;

        if constexpr (is_same_v<T, float16>) {
          int elements_per_byte = 8 / bit_rate;
          out_emb_cols = (colSize + elements_per_byte - 1) / elements_per_byte;
        }
        int outVecSize = rowSize * (out_emb_cols + 2 * sizeof(float16));
        aligned_vector<uint8_t> outVec(outVecSize);

        double duration = 0.0f;

        duration = measureWithWarmup(
            [&]() {
              is_same_v<T, float16>
                  ? FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float>(
                        bit_rate,
                        inpVec.data(),
                        rowSize,
                        colSize,
                        outVec.data())
                  : FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float>(
                        inpVec.data(), rowSize, colSize, outVec.data());
            },
            NWARMUP,
            NITER,
            [&]() {
              cache_evict(inpVec);
              cache_evict(outVec);
            });

        float elements_per_usec = rowSize * colSize / (duration * 1e6);

        duration *= 1e9; // convert to ns
        long bytes_read = rowSize * colSize * sizeof(float);
        float gigabyes_per_sec = bytes_read / duration;

        cout << setw(8) << bit_rate << "," << setw(6) << rowSize << ", "
             << setw(6) << colSize << ",";
        cout << setw(16) << std::fixed << std::setprecision(2)
             << elements_per_usec << ", ";
        cout << setw(10) << std::fixed << std::setprecision(2)
             << gigabyes_per_sec << '\n';
      } // for each cols
    } // for each rows
  } // for each bit_rate
} // performance_test

// Benchmark float16 input with pre-supplied rowwise_min_max
static void performance_test_fp16_with_minmax() {
  constexpr int NWARMUP = 4;
  constexpr int NITER = 256;

  cout << "\nWith float16 input and rowwise_min_max" << '\n';
  cout << setw(8) << "bit_rate" << ", " << setw(6) << "rows" << "," << setw(6)
       << "cols" << "," << setw(16) << "elems_per_usec" << "," << setw(10)
       << "GB/Sec" << '\n';

  for (int bit_rate : {2, 4, 8}) {
    int num_elem_per_byte = 8 / bit_rate;
    for (int rowSize : {100, 1000}) {
      for (int colSize : {64, 128, 256, 512, 1024}) {
        int n = rowSize * colSize;
        aligned_vector<float> inpFloatVec(n);
        randFill<float>(inpFloatVec, -10.0f, 10.0f);

        aligned_vector<float16> inpVec(n);
        aligned_vector<float16> minMaxVec(rowSize * 2);
        ranges::transform(inpFloatVec, inpVec.begin(), cpu_float2half_rn);
        for (int r = 0; r < rowSize; ++r) {
          auto row = inpFloatVec.begin() + r * colSize;
          auto [mn, mx] = ranges::minmax_element(row, row + colSize);
          minMaxVec[r * 2] = cpu_float2half_rn(*mn);
          minMaxVec[r * 2 + 1] = cpu_float2half_rn(*mx);
        }

        int out_emb_cols =
            (colSize + num_elem_per_byte - 1) / num_elem_per_byte;
        aligned_vector<uint8_t> outVec(
            rowSize * (out_emb_cols + 2 * sizeof(float16)));

        double duration = measureWithWarmup(
            [&]() {
              FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float16>(
                  bit_rate,
                  inpVec.data(),
                  rowSize,
                  colSize,
                  outVec.data(),
                  minMaxVec.data());
            },
            NWARMUP,
            NITER,
            [&]() {
              cache_evict(inpVec);
              cache_evict(outVec);
            });

        cout << setw(8) << bit_rate << "," << setw(6) << rowSize << ", "
             << setw(6) << colSize << ",";
        cout << setw(16) << fixed << setprecision(2) << n / (duration * 1e6)
             << ", ";
        cout << setw(10) << fixed << setprecision(2)
             << n * sizeof(float16) / (duration * 1e9) << '\n';
      }
    }
  }
}

int main() {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif
  performance_test<float16>();
  performance_test<float>();
  performance_test_fp16_with_minmax();
  return 0;
}
