/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSpConv.h"

#include <iostream>

using namespace std;
using namespace fbgemm;

int main(int, char**) {
  vector<char> llc(128 * 1024 * 1024);

  // clang-format off
  vector<vector<unsigned>> shapes = {
    {128, 128, 28, 28}
  };
  // clang-format on

  // C is MxN -> CT is NxM
  // A is MxK -> BT is KxM
  // B is KxN -> AT is NxK

  for (auto const& s : shapes) {
    int Cout = s[0];
    int Cin = s[1];
    int IY = s[2];
    int IX = s[3];

    for (float fnz = 0.99; fnz >= 0.009999; fnz -= 0.01) {
      constexpr int KY = 3;
      constexpr int KX = 3;

      auto kData = getRandomSparseVector(KY * KX * Cout * Cin / 4, fnz);
      auto bData = getRandomSparseVector(Cin * IY * IX / 4);
      auto cData = getRandomSparseVector(Cout * IY * IX);

      auto kptr = reinterpret_cast<const int8_t*>(kData.data());
      auto bptr = reinterpret_cast<uint8_t*>(bData.data());

      for (int i = 0; i < bData.size() * 4; ++i) {
        bptr[i] &= 0x7F;
      }

      auto cptr = reinterpret_cast<int32_t*>(cData.data());

      auto fn = generateSpConv<int32_t>(Cin, Cout, IY, IX, kptr);

      double effective_flop = IY * IX * Cin * Cout * KY * KX * 2;

      auto secs = fbgemm::measureWithWarmup(
          [&]() { fn(bptr, cptr); }, 5, 10, [&]() { llc_flush(llc); });

      double effective_gflops = effective_flop / secs / 1e9;
      cout << fnz << "," << effective_gflops << "," << fnz * effective_gflops
           << endl;
    }
  }
}
