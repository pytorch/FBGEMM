/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include "./EmbeddingSpMDMTestUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/Utils.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // batch size, number of rows of table, emb dim , avg length
      {1, 8, 8, 4},
      {2, 8, 16, 4},
      {10, 4000, 32, 100},
      {100, 4000, 32, 100},
      {10, 4000, 64, 100},
      {10, 4000, 128, 100},
      {4, 400, 256, 10},
      {10, 4000, 48, 100},
      {10, 4000, 48, 100},
      {10, 4000, 40, 100},
      {10, 4000, 56, 100},
      {10, 4000, 1, 100},
      {10, 4000, 4, 100},
      // These were  from C2 tests
      {10, 40, 16, 10},
      {10, 40, 85, 10},
      {10, 40, 8, 10},
      {10, 40, 96, 10},
      {10, 40, 163, 10},
  };
  return input_dims;
}

vector<int> prefetch_distances{0, 16, 1000000};

namespace {

class RowWiseSparseAdagradFusedTest
    : public testing::TestWithParam<tuple<bool, int, EmbeddingSpMDMCornerCase>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    RowWiseSparseAdagradFusedTest,
    ::testing::Combine(
        ::testing::Bool(), // isIndex64b
        ::testing::ValuesIn(prefetch_distances),
        ::testing::Values(
            NONE,
            EMPTY_INDICES,
            OUT_OF_BOUND_INDICES,
            UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM)));

TEST_P(RowWiseSparseAdagradFusedTest, rowwiseTest) {
  vector<vector<int>> inputs(GetInputs_());
  bool isIndex64b;
  int prefetch;
  EmbeddingSpMDMCornerCase corner_case;
  tie(isIndex64b, prefetch, corner_case) = GetParam();

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    // Create embedding table
    vector<float> w(num_rows * embedding_dim), w_ref(num_rows * embedding_dim),
        h(num_rows), h_ref(num_rows), g(batch_size * embedding_dim);
    default_random_engine generator;
    uniform_real_distribution<float> values_gen(0, 2);
    for (int i = 0; i < w.size(); ++i) {
      w_ref[i] = w[i] = values_gen(generator);
    }
    for (int i = 0; i < h.size(); ++i) {
      h_ref[i] = h[i] = values_gen(generator);
    }
    for (int i = 0; i < g.size(); ++i) {
      g[i] = values_gen(generator);
    }

    vector<int> lengths;
    vector<int64_t> indices;
    vector<int32_t> indices_32;
    vector<float> weights;
    int lengths_sum = GenerateLengthsIndicesWeights(
        lengths,
        indices,
        indices_32,
        weights,
        batch_size,
        num_rows,
        embedding_dim,
        average_len,
        corner_case);

    float epsilon = 1e-5;
    float lr = 0.5;

    bool success, success_ref;
    if (isIndex64b) {
      success_ref = rowwise_sparse_adagrad_fused_ref(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          w_ref.data(),
          g.data(),
          h_ref.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices.data(),
          lengths.data(),
          epsilon,
          lr);

      auto kernel =
          GenerateRowWiseSparseAdaGradFused<int64_t>(embedding_dim, prefetch);
      success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          w.data(),
          g.data(),
          h.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices.data(),
          lengths.data(),
          epsilon,
          lr);
    } else { // 32 bit indices
      success_ref = rowwise_sparse_adagrad_fused_ref(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          w_ref.data(),
          g.data(),
          h_ref.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
          lengths.data(),
          epsilon,
          lr);

      auto kernel =
          GenerateRowWiseSparseAdaGradFused<int32_t>(embedding_dim, prefetch);
      success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          w.data(),
          g.data(),
          h.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
          lengths.data(),
          epsilon,
          lr);
    }

    EXPECT_EQ(success, success_ref)
        << "return vals differ, reference is: " << success_ref
        << " ,fbgemm is: " << success;
    if (success) {
      for (int i = 0; i < h.size(); ++i) {
        EXPECT_EQ(h[i], h_ref[i])
            << "results for h differ at (" << i << ") reference: " << h_ref[i]
            << ", FBGEMM: " << h[i] << " emb dim :" << embedding_dim;
      }
      for (int i = 0; i < w.size(); ++i) {
        EXPECT_EQ(w[i], w_ref[i])
            << "results for w differ at (" << i << ") reference: " << w_ref[i]
            << ", FBGEMM: " << w[i] << " emb dim :" << embedding_dim;
      }
    }
  }
}
