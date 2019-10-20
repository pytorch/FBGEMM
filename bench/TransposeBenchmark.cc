/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "fbgemm/Utils.h"
#include "src/TransposeUtils.h"

using namespace std;
using namespace fbgemm;

void performance_test() {
  constexpr int NWARMUP = 4;
  constexpr int NITER = 256;

  normal_distribution<float> dist;
  default_random_engine engine;

  cout << setw(4) << "M" << setw(4) << "N" << " B_elements_per_sec" << endl;

  int dims[] = {1,  2,  3,  4,  5,  6,  8,   9,   10,  15,  16,
                17, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256};
  for (int M : dims) {
    for (int N : dims) {
      vector<float> a(M * N);
      vector<float> b(N * M), b_ref(N * M);

      generate(a.begin(), a.end(), [&dist, &engine] { return dist(engine); });
      transpose_ref(M, N, a.data(), N, b_ref.data(), M);

      chrono::time_point<chrono::high_resolution_clock> begin, end;
      for (int i = 0; i < NWARMUP + NITER; ++i) {
        if (i == NWARMUP) {
          begin = chrono::high_resolution_clock::now();
        }
        transpose_simd(M, N, a.data(), N, b.data(), M);
      }
      end = chrono::high_resolution_clock::now();

      auto duration = chrono::duration_cast<chrono::nanoseconds>(end - begin);

      cout << setw(4) << M << setw(4) << N << setw(10) << setprecision(3)
           << static_cast<double>(M * N) * NITER / duration.count() << endl;

      compare_buffers(b_ref.data(), b.data(), M, N, N, 5);
    } // N
  } // M
} // performance_test

int main() {
  performance_test();
  return 0;
}
