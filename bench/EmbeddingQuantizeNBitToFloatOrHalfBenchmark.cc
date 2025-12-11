/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
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
    cout << "With result as float16" << '\n';
  } else {
    cout << "With result as float" << '\n';
  }
  cout << setw(8) << "bit_rate" << ", " << setw(6) << "rows" << "," << setw(6)
       << "cols" << "," << setw(16) << "elems_per_usec" << "," << setw(10)
       << "GB/Sec" << '\n';
  std::vector<int> bit_rates = {2, 4, 8};
  for (int bit_rate : bit_rates) {
    for (int rowSize : {100, 120, 1000}) {
      for (int colSize : {16, 64, 128, 256, 512, 1024, 2048}) {
        int elem_per_byte = 8 / bit_rate;
        int bytes_per_row = colSize / elem_per_byte + 2 * sizeof(float16);

        aligned_vector<uint8_t> inpVec(rowSize * bytes_per_row);
        randFill<uint8_t>(inpVec, 0, 20);

        aligned_vector<T> outVec(rowSize * colSize);

        double duration = 0.0f;

        int constexpr kNumRepeats = is_same_v<T, float16> ? 4 : 8;

        duration = measureWithWarmup(
            [&]() {
              for (int i = 0; i < kNumRepeats; ++i) {
                FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
                    bit_rate,
                    inpVec.data(),
                    rowSize,
                    bytes_per_row,
                    outVec.data());
              }
            },
            NWARMUP,
            NITER,
            [&]() {
              cache_evict(inpVec);
              cache_evict(outVec);
            });

        float elements_per_usec =
            rowSize * colSize * kNumRepeats / (duration * 1e6);

        duration *= 1e9; // convert to ns
        long bytes_read = rowSize * colSize * sizeof(float) * kNumRepeats;
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
  return 0;
}
