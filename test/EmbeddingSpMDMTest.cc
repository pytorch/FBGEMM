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

#include "fbgemm/Fbgemm.h"
#include "fbgemm/Utils.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

static vector<vector<int>> GetInputs_() {
  vector<vector<int>> input_dims = {
      // batch size, number of rows of table, emb dim , avg lengthl
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

namespace {

class EmbeddingSpMDMTest : public testing::TestWithParam<
                               tuple<bool, bool, int, bool, bool, bool, bool>> {
};
}; // namespace

vector<int> prefetch_distances = {0, 16, 1000000};

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    EmbeddingSpMDMTest,
    ::testing::Combine(
        ::testing::Bool(), // isIndex64b
        ::testing::Bool(), // is_wt_positional
        ::testing::ValuesIn(prefetch_distances),
        ::testing::Bool(), // use_weight
        ::testing::Bool(), // normalize_by_lengths
        ::testing::Bool(), // empty_indices
        ::testing::Bool())); // empty_indices

TEST_P(EmbeddingSpMDMTest, basicTest) {
  vector<vector<int>> inputs(GetInputs_());
  bool isIndex64b, is_wt_positional, use_weight, normalize_by_lengths,
      empty_indices, out_of_bounds;
  int prefetch;
  tie(isIndex64b,
      is_wt_positional,
      prefetch,
      use_weight,
      normalize_by_lengths,
      empty_indices,
      out_of_bounds) = GetParam();

  int batch_size, num_rows, embedding_dim, average_len;

  for (auto input : inputs) {
    batch_size = input[0];
    num_rows = input[1];
    embedding_dim = input[2];
    average_len = input[3];

    // Create embedding table
    vector<float> embedding_table(num_rows * embedding_dim);
    default_random_engine generator;
    normal_distribution<float> embedding_distribution;
    for (int i = 0; i < embedding_table.size(); ++i) {
      embedding_table[i] = embedding_distribution(generator);
    }

    // Generate lengths
    uniform_int_distribution<int> length_distribution(1, 2 * average_len + 1);
    vector<int> lengths(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      lengths[i] = empty_indices ? 0 : length_distribution(generator);
    }

    // Compute the number of indices
    int lengths_sum = accumulate(lengths.begin(), lengths.end(), 0);
    // cout << "lengths_sum " << lengths_sum << endl;

    // Generate indices
    vector<int64_t> indices;
    vector<int32_t> indices_32;

    // Generate indices
    vector<int> container(num_rows);
    for (int i = 0; i < batch_size; ++i) {
      iota(container.begin(), container.end(), 0);
      random_shuffle(container.begin(), container.end());
      copy(
          container.begin(),
          container.begin() + lengths[i],
          back_inserter(indices));
    }
    // use same indices for 32b and 64b
    copy(begin(indices), end(indices), back_inserter(indices_32));
    assert(indices.size() == lengths_sum);
    assert(indices_32.size() == lengths_sum);
    if (!empty_indices && out_of_bounds) {
      int idx = uniform_int_distribution<int>(0, lengths_sum - 1)(generator);
      indices_32[idx] = indices[idx] = num_rows;
    }
    if (!empty_indices) {
      // To make sure to exercise out-of-bound cases
      indices_32[0] = indices[0] = num_rows - 1;
    }

    // Generate weights
    vector<float> weights(lengths_sum);
    for (int i = 0; i < lengths_sum; ++i) {
      weights[i] = embedding_distribution(generator);
    }

    vector<float> output_sls_ref(batch_size * embedding_dim);
    vector<float> output_slws_ref(output_sls_ref.size()),
        output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

    vector<float>& output_ref = use_weight ? output_slws_ref : output_sls_ref;
    vector<float>& output = use_weight ? output_slws : output_sls;
    bool success, success_ref;

    if (isIndex64b) {
      success_ref = fbgemm::EmbeddingSpMDM_ref(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          embedding_table.data(),
          empty_indices ? nullptr : indices.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data(),
          is_wt_positional);

      success = fbgemm::EmbeddingSpMDM<float, int64_t>(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          embedding_table.data(),
          empty_indices ? nullptr : indices.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output.data(),
          prefetch,
          is_wt_positional);
    } else {
      success_ref = fbgemm::EmbeddingSpMDM_ref(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          embedding_table.data(),
          empty_indices ? nullptr : indices_32.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data(),
          is_wt_positional);

      success = fbgemm::EmbeddingSpMDM<float, int32_t>(
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          embedding_table.data(),
          empty_indices ? nullptr : indices_32.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output.data(),
          prefetch,
          is_wt_positional);
    }

    // Check correctness
    EXPECT_EQ(success, success_ref)
        << "Reference and JIT impl did not both succeed";
    if (success) {
      for (int i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], output_ref[i])
            << "results differ at (" << i << ") reference: " << output_ref[i]
            << ", FBGEMM: " << output[i] << " emb dim :" << embedding_dim;
      }
    }
  } // end for input
}
