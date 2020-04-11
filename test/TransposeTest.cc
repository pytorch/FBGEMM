/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "fbgemm/Utils.h"

using namespace std;
using namespace fbgemm;

TEST(TransposeTest, TransposeTest) {
  // Generate shapes to test
  vector<tuple<int, int, int, int>> shapes;
  uniform_int_distribution<int> dist(0, 32);
  default_random_engine generator;
  for (int i = 0; i < 1024; ++i) {
    int m = dist(generator);
    int n = dist(generator);
    int ld_src = n + dist(generator);
    int ld_dst = m + dist(generator);
    shapes.push_back(make_tuple(m, n, ld_src, ld_dst));
  }

  for (const auto& shape : shapes) {
    int m, n, ld_src, ld_dst;
    tie(m, n, ld_src, ld_dst) = shape;

    vector<float> a(m * ld_src);
    vector<float> b(n * ld_dst);
    generate(
        a.begin(), a.end(), [&dist, &generator] { return dist(generator); });

    transpose_simd(m, n, a.data(), ld_src, b.data(), ld_dst);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        int expected = a[i * ld_src + j];
        int actual = b[i + j * ld_dst];
        EXPECT_EQ(actual, expected)
            << "Transpose results differ at (" << i << ", " << j << "). ref "
            << expected << " actual " << actual;
      }
    }
  }
}
