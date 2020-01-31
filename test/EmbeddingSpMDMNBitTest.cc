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
#include "fbgemm/FbgemmConvert.h"
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
      {4, 400, 512, 10},
      {10, 4000, 48, 100},
      {10, 4000, 40, 100},
      {10, 4000, 56, 100},
      {10, 4000, 2, 100},
      {10, 4000, 4, 100},
      {10, 4000, 7, 100},
      // These were  from C2 tests
      {10, 40, 16, 10},
      {10, 40, 86, 10},
      {10, 40, 8, 10},
      {10, 40, 96, 10},
      {10, 40, 164, 10},
  };
  return input_dims;
}

vector<int> prefetch_distances{0, 16, 1000000};

namespace {

// tuple represents MB, IC, OC, IT, IH, IW, KH/KW, stride, pad
class FusedNBitRowwiseEmbeddingLookupTest
    : public testing::TestWithParam<
          tuple<int, bool, bool, int, bool, bool, bool, bool>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FusedNBitRowwiseEmbeddingLookupTest,
    ::testing::Combine(
        ::testing::Values(2, 4), // bit_rate
        ::testing::Bool(), // isIndex64b
        ::testing::Bool(), // is_wt_positional
        ::testing::ValuesIn(prefetch_distances),
        ::testing::Bool(), // use_weight
        ::testing::Bool(), // normalize_by_lengths
        ::testing::Bool(), // empty_indices
        ::testing::Bool())); // out of bounds

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, basicTest) {
  vector<vector<int>> inputs(GetInputs_());
  bool isIndex64b, is_wt_positional, use_weight, normalize_by_lengths,
      empty_indices, out_of_bounds;
  int bit_rate, prefetch;
  tie(bit_rate,
      isIndex64b,
      is_wt_positional,
      prefetch,
      use_weight,
      normalize_by_lengths,
      empty_indices,
      out_of_bounds) = GetParam();

  int num_elem_per_byte = 8 / bit_rate;

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    // Create embedding table
    default_random_engine generator;
    normal_distribution<float> embedding_distribution;
    uniform_int_distribution<int> entries(0, 16);

    int fused_embedding_dim =
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte +
        2 * sizeof(float16);
    uint8_t* fused_embedding_table =
        new uint8_t[num_rows * fused_embedding_dim];
    for (int i = 0; i < num_rows; i++) {
      for (int ii = 0;
           ii < (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
           ii++) {
        fused_embedding_table[i * fused_embedding_dim + ii] =
            entries(generator);
      }
      float16* scale_bias = reinterpret_cast<float16*>(
          fused_embedding_table + i * fused_embedding_dim +
          (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte);
      float scale = embedding_distribution(generator);
      float bias = embedding_distribution(generator);
      FloatToFloat16_ref(&scale, scale_bias, 1, true /* clip */);
      FloatToFloat16_ref(&bias, scale_bias + 1, 1, true /* clip */);
    }

    // Generate lengths
    uniform_int_distribution<int> length_distribution(1, 2 * average_len + 1);
    vector<int> lengths(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      lengths[i] = empty_indices ? 0 : length_distribution(generator);
    }

    // Compute the number of indices
    int lengths_sum = accumulate(lengths.begin(), lengths.end(), 0);

    // Generate indices
    vector<int64_t> indices(lengths_sum);
    vector<int32_t> indices_32(lengths_sum);

    uniform_int_distribution<int> index_distribution(0, num_rows - 1);
    for (int i = 0; i < lengths_sum; ++i) {
      indices_32[i] = indices[i] = index_distribution(generator);
    }
    if (!empty_indices && out_of_bounds) {
      int idx = uniform_int_distribution<int>(0, lengths_sum - 1)(generator);
      indices_32[idx] = indices[idx] = num_rows;
    }
    if (!empty_indices) {
      // to make sure to exercise out-of-bound cases
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
      success_ref = EmbeddingSpMDMNBit_ref<int64_t>(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table,
          empty_indices ? nullptr : indices.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data(),
          is_wt_positional);

      auto kernel = GenerateEmbeddingSpMDMNBit<int64_t>(
          bit_rate,
          embedding_dim,
          use_weight,
          normalize_by_lengths,
          prefetch,
          is_wt_positional);
      success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table,
          empty_indices ? nullptr : indices.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          output.data());
    } else {
      success_ref = EmbeddingSpMDMNBit_ref<int32_t>(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table,
          empty_indices ? nullptr : indices_32.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data(),
          is_wt_positional);

      auto kernel = GenerateEmbeddingSpMDMNBit<int32_t>(
          bit_rate,
          embedding_dim,
          use_weight,
          normalize_by_lengths,
          prefetch,
          is_wt_positional);
      success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table,
          empty_indices ? nullptr : indices_32.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          output.data());
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
    delete[] fused_embedding_table;
  } // end for input
}

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, rowwiseSparseTest) {
  vector<vector<int>> inputs(GetInputs_());
  bool isIndex64b, is_wt_positional, use_weight, normalize_by_lengths,
      empty_indices, out_of_bounds;
  int bit_rate, prefetch;
  tie(bit_rate,
      isIndex64b,
      is_wt_positional, // ignored
      prefetch,
      use_weight,
      normalize_by_lengths,
      empty_indices,
      out_of_bounds) = GetParam();

  int num_elem_per_byte = 8 / bit_rate;
  constexpr float sparsity = 0.7;

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    // Create embedding table
    default_random_engine generator;
    normal_distribution<float> embedding_distribution;
    uniform_int_distribution<int> entries(0, 16);

    vector<int64_t> mapping_table(num_rows);
    bernoulli_distribution row_prune_dist(sparsity);
    int num_compressed_rows = 0;
    for (int i = 0; i < num_rows; ++i) {
      if (row_prune_dist(generator)) {
        // pruned
        mapping_table[i] = -1;
      } else {
        mapping_table[i] = num_compressed_rows;
        ++num_compressed_rows;
      }
    }
    vector<int32_t> mapping_table_32;
    copy(
        mapping_table.begin(),
        mapping_table.end(),
        back_inserter(mapping_table_32));

    int fused_embedding_dim =
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte +
        2 * sizeof(float16);
    uint8_t* fused_embedding_table =
        new uint8_t[num_compressed_rows * fused_embedding_dim];
    for (int i = 0; i < num_compressed_rows; i++) {
      for (int ii = 0;
           ii < (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
           ii++) {
        fused_embedding_table[i * fused_embedding_dim + ii] =
            entries(generator);
      }
      float16* scale_bias = reinterpret_cast<float16*>(
          fused_embedding_table + i * fused_embedding_dim +
          (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte);
      float scale = embedding_distribution(generator);
      float bias = embedding_distribution(generator);
      FloatToFloat16_ref(&scale, scale_bias, 1, true /* clip */);
      FloatToFloat16_ref(&bias, scale_bias + 1, 1, true /* clip */);
    }

    // Generate lengths
    uniform_int_distribution<int> length_distribution(1, 2 * average_len + 1);
    vector<int> lengths(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      lengths[i] = empty_indices ? 0 : length_distribution(generator);
    }

    // Compute the number of indices
    int lengths_sum = accumulate(lengths.begin(), lengths.end(), 0);

    // Generate indices
    vector<int64_t> indices(lengths_sum);
    vector<int32_t> indices_32(lengths_sum);

    uniform_int_distribution<int> index_distribution(0, num_rows - 1);
    for (int i = 0; i < lengths_sum; ++i) {
      indices_32[i] = indices[i] = index_distribution(generator);
    }
    if (!empty_indices && out_of_bounds) {
      int idx = uniform_int_distribution<int>(0, lengths_sum - 1)(generator);
      indices_32[idx] = indices[idx] = num_rows;

      // idx = uniform_int_distribution<int>(0, num_rows - 1)(generator);
      // mapping_table_32[idx] = mapping_table[idx] = num_compressed_rows;
    }
    if (!empty_indices) {
      // to make sure to exercise out-of-bound cases
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
      success_ref = fbgemm::EmbeddingSpMDMNBitRowWiseSparse_ref<int64_t>(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table,
          empty_indices ? nullptr : indices.data(),
          mapping_table.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data());

      auto kernel = GenerateEmbeddingSpMDMNBitRowWiseSparse<int64_t>(
          bit_rate, embedding_dim, use_weight, normalize_by_lengths, prefetch);
      success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table,
          empty_indices ? nullptr : indices.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          output.data(),
          mapping_table.data());
    } else {
      success_ref = EmbeddingSpMDMNBitRowWiseSparse_ref<int32_t>(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table,
          empty_indices ? nullptr : indices_32.data(),
          mapping_table_32.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          normalize_by_lengths,
          output_ref.data());

      auto kernel = GenerateEmbeddingSpMDMNBitRowWiseSparse<int32_t>(
          bit_rate, embedding_dim, use_weight, normalize_by_lengths, prefetch);
      success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table,
          empty_indices ? nullptr : indices_32.data(),
          lengths.data(),
          use_weight ? weights.data() : nullptr,
          output.data(),
          mapping_table_32.data());
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
    delete[] fused_embedding_table;
  } // end for input
}
