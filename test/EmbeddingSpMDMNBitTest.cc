/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
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
#include "fbgemm/FbgemmConvert.h"
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

class FusedNBitRowwiseEmbeddingLookupTest : public testing::TestWithParam<tuple<
                                                int,
                                                int,
                                                EmbeddingSpMDMWeightChoice,
                                                EmbeddingSpMDMCornerCase>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FusedNBitRowwiseEmbeddingLookupTest,
    ::testing::Combine(
        ::testing::Values(2, 4), // bit_rate
        ::testing::ValuesIn(prefetch_distances),
        ::testing::Values(
            UNWEIGHTED,
            WEIGHTED,
            POSITIONAL_WEIGHTED), // use_weight
        ::testing::Values(
            NONE,
            EMPTY_INDICES,
            OUT_OF_BOUND_INDICES,
            UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM)));

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, basicTest) {
  vector<vector<int>> inputs(GetInputs_());

  default_random_engine generator;
  uniform_int_distribution<> bool_dist(0, 1);

  bool isIndex64b = bool_dist(generator);
  bool isOffset64b = bool_dist(generator);
  bool normalize_by_lengths = bool_dist(generator);
  bool use_offsets = bool_dist(generator);
  bool is_output_float = bool_dist(generator);
  bool scale_bias_last = bool_dist(generator);
  bool test_thread_local = bool_dist(generator);
  int bit_rate, prefetch;
  EmbeddingSpMDMWeightChoice weight_choice;
  EmbeddingSpMDMCornerCase corner_case;
  tie(bit_rate, prefetch, weight_choice, corner_case) = GetParam();
  bool is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  bool use_weight = weight_choice != UNWEIGHTED;

  if (corner_case != NONE || weight_choice == POSITIONAL_WEIGHTED) {
    // Check corner case only for subset of tests.
    if (normalize_by_lengths || !is_output_float || !scale_bias_last ||
        test_thread_local) {
      return;
    }
  }
  if (is_wt_positional && !use_weight) {
    // weight positional only makes sense when use_weight is true
    return;
  }

  int num_elem_per_byte = 8 / bit_rate;

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    // Create embedding table
    normal_distribution<float> embedding_distribution;
    uniform_int_distribution<int> entries(0, 16);

    int fused_embedding_dim =
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte +
        2 * sizeof(float16);
    vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim);
    for (int i = 0; i < num_rows; i++) {
      for (int ii = 0;
           ii < (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
           ii++) {
        fused_embedding_table
            [i * fused_embedding_dim + ii +
             (scale_bias_last ? 0 : 2 * sizeof(float16))] = entries(generator);
      }
      float16* scale_bias = reinterpret_cast<float16*>(
          fused_embedding_table.data() + i * fused_embedding_dim +
          (scale_bias_last
               ? (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte
               : 0));
      float scale = embedding_distribution(generator);
      float bias = embedding_distribution(generator);
      FloatToFloat16_ref(&scale, scale_bias, 1, true /* clip */);
      FloatToFloat16_ref(&bias, scale_bias + 1, 1, true /* clip */);
    }

    vector<int64_t> lengths, offsets, indices;
    vector<int32_t> lengths_32, offsets_32, indices_32;
    vector<float> weights;
    int lengths_sum = GenerateLengthsIndicesWeights(
        lengths,
        lengths_32,
        offsets,
        offsets_32,
        indices,
        indices_32,
        weights,
        batch_size,
        num_rows,
        average_len,
        corner_case);
    const int64_t* offsets_or_lengths =
        (use_offsets ? offsets : lengths).data();
    const int32_t* offsets_or_lengths_32 =
        (use_offsets ? offsets_32 : lengths_32).data();

    // Sentries at the end to make sure masking is done correctly not to write
    // out of bounds.
    constexpr int num_sentries = 10;
    const float sentry_value = 1.0f;
    int output_size_wo_sentries = batch_size * embedding_dim;
    vector<float> output_ref(output_size_wo_sentries + num_sentries);
    vector<float> output(output_ref.size());
    vector<float16> output_ref_fp16(output.size()), output_fp16(output.size());
    for (size_t i = output_size_wo_sentries; i < output.size(); ++i) {
      output_ref[i] = sentry_value;
      output[i] = sentry_value;
      output_ref_fp16[i] = cpu_float2half_rn(sentry_value);
      output_fp16[i] = cpu_float2half_rn(sentry_value);
    }

    bool success, success_ref;

#define TEST_BASE(                                                      \
    indices,                                                            \
    offsets_or_lengths,                                                 \
    output_ref,                                                         \
    output,                                                             \
    IndexType,                                                          \
    OffsetType,                                                         \
    OutType,                                                            \
    THREAD_LOCAL)                                                       \
  success_ref = EmbeddingSpMDMNBit_ref<IndexType, OffsetType, OutType>( \
      bit_rate,                                                         \
      embedding_dim,                                                    \
      batch_size,                                                       \
      lengths_sum,                                                      \
      num_rows,                                                         \
      fused_embedding_table.data(),                                     \
      corner_case == EMPTY_INDICES ? nullptr : indices.data(),          \
      offsets_or_lengths,                                               \
      use_weight ? weights.data() : nullptr,                            \
      normalize_by_lengths,                                             \
      output_ref.data(),                                                \
      is_wt_positional,                                                 \
      use_offsets,                                                      \
      /*output_stride=*/-1,                                             \
      /*input_stride=*/-1,                                              \
      scale_bias_last);                                                 \
                                                                        \
  auto kernel = GenerateEmbeddingSpMDMNBitWithStrides<                  \
      IndexType,                                                        \
      OffsetType,                                                       \
      OutType,                                                          \
      THREAD_LOCAL>(                                                    \
      bit_rate,                                                         \
      embedding_dim,                                                    \
      use_weight,                                                       \
      normalize_by_lengths,                                             \
      prefetch,                                                         \
      is_wt_positional,                                                 \
      use_offsets,                                                      \
      /*output_stride=*/-1,                                             \
      /*input_stride=*/-1,                                              \
      scale_bias_last);                                                 \
  success = kernel(                                                     \
      batch_size,                                                       \
      lengths_sum,                                                      \
      num_rows,                                                         \
      fused_embedding_table.data(),                                     \
      corner_case == EMPTY_INDICES ? nullptr : indices.data(),          \
      offsets_or_lengths,                                               \
      use_weight ? weights.data() : nullptr,                            \
      output.data());

#define TEST_THREAD_LOCAL(  \
    indices,                \
    offsets_or_lengths,     \
    output_ref,             \
    output,                 \
    IndexType,              \
    OffsetType,             \
    OutType)                \
  if (test_thread_local) {  \
    TEST_BASE(              \
        indices,            \
        offsets_or_lengths, \
        output_ref,         \
        output,             \
        IndexType,          \
        OffsetType,         \
        OutType,            \
        true);              \
  } else {                  \
    TEST_BASE(              \
        indices,            \
        offsets_or_lengths, \
        output_ref,         \
        output,             \
        IndexType,          \
        OffsetType,         \
        OutType,            \
        false);             \
  }

#define TEST_OUT_TYPE(indices, offsets_or_lengths, IndexType, OffsetType) \
  if (is_output_float) {                                                  \
    TEST_THREAD_LOCAL(                                                    \
        indices,                                                          \
        offsets_or_lengths,                                               \
        output_ref,                                                       \
        output,                                                           \
        IndexType,                                                        \
        OffsetType,                                                       \
        float);                                                           \
  } else {                                                                \
    TEST_THREAD_LOCAL(                                                    \
        indices,                                                          \
        offsets_or_lengths,                                               \
        output_ref_fp16,                                                  \
        output_fp16,                                                      \
        IndexType,                                                        \
        OffsetType,                                                       \
        float16);                                                         \
  }

#define TEST_OFFSET_TYPE(indices, IndexType)                           \
  if (isOffset64b) {                                                   \
    TEST_OUT_TYPE(indices, offsets_or_lengths, IndexType, int64_t);    \
  } else {                                                             \
    TEST_OUT_TYPE(indices, offsets_or_lengths_32, IndexType, int32_t); \
  }

    if (isIndex64b) {
      TEST_OFFSET_TYPE(indices, int64_t);
    } else {
      TEST_OFFSET_TYPE(indices_32, int32_t);
    }

#undef TEST_OFFSET_TYPE
#undef TEST_OUT_TYPE
#undef TEST_THREAD_LOCAL
#undef TEST_BASE

    // Check correctness
    EXPECT_EQ(success, success_ref)
        << "Reference and JIT impl did not both succeed";
    if (corner_case == OUT_OF_BOUND_INDICES ||
        corner_case == UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM) {
      EXPECT_EQ(success, false);
    }
    if (success) {
      for (size_t i = 0; i < output.size(); ++i) {
        float actual =
            is_output_float ? output[i] : cpu_half2float(output_fp16[i]);
        float expected = is_output_float ? output_ref[i]
                                         : cpu_half2float(output_ref_fp16[i]);
        EXPECT_EQ(actual, expected)
            << "results differ at (" << i << ") reference: " << expected
            << ", FBGEMM: " << actual << " emb dim :" << embedding_dim;
      }
      for (int offset = output_size_wo_sentries;
           offset < output_size_wo_sentries + num_sentries;
           ++offset) {
        float actual = is_output_float ? output[offset]
                                       : cpu_half2float(output_fp16[offset]);
        float expected = is_output_float
            ? output_ref[offset]
            : cpu_half2float(output_ref_fp16[offset]);
        EXPECT_EQ(actual, expected)
            << "results differ at (" << offset << ") reference: " << expected
            << ", FBGEMM: " << actual << " emb dim :" << embedding_dim;
      }
    }
  } // end for input
}

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, rowwiseSparseTest) {
  vector<vector<int>> inputs(GetInputs_());

  default_random_engine generator;
  uniform_int_distribution<> bool_dist(0, 1);

  bool isIndex64b = bool_dist(generator);
  bool isOffset64b = bool_dist(generator);
  bool normalize_by_lengths = bool_dist(generator);
  bool use_offsets = bool_dist(generator);
  bool is_output_float = bool_dist(generator);
  bool scale_bias_last = bool_dist(generator);

  int bit_rate, prefetch;
  EmbeddingSpMDMWeightChoice weight_choice;
  EmbeddingSpMDMCornerCase corner_case;
  tie(bit_rate, prefetch, weight_choice, corner_case) = GetParam();
  bool is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  bool use_weight = weight_choice != UNWEIGHTED;

  if (!is_output_float || !scale_bias_last) {
    return;
  }

  int num_elem_per_byte = 8 / bit_rate;
  constexpr float sparsity = 0.7;

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    // Create mapping table for rowwise sparsity
    vector<int32_t> mapping_table;
    int num_compressed_rows =
        CreateMappingTableForRowWiseSparsity(mapping_table, num_rows, sparsity);

    // Create embedding table
    normal_distribution<float> embedding_distribution;
    uniform_int_distribution<int> entries(0, 16);

    int fused_embedding_dim =
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte +
        2 * sizeof(float16);
    vector<uint8_t> fused_embedding_table(
        num_compressed_rows * fused_embedding_dim);
    for (int i = 0; i < num_compressed_rows; i++) {
      for (int ii = 0;
           ii < (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
           ii++) {
        fused_embedding_table[i * fused_embedding_dim + ii] =
            entries(generator);
      }
      float16* scale_bias = reinterpret_cast<float16*>(
          fused_embedding_table.data() + i * fused_embedding_dim +
          (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte);
      float scale = embedding_distribution(generator);
      float bias = embedding_distribution(generator);
      FloatToFloat16_ref(&scale, scale_bias, 1, true /* clip */);
      FloatToFloat16_ref(&bias, scale_bias + 1, 1, true /* clip */);
    }

    vector<int64_t> lengths, offsets, indices;
    vector<int32_t> lengths_32, offsets_32, indices_32;
    vector<float> weights;
    int lengths_sum = GenerateLengthsIndicesWeights(
        lengths,
        lengths_32,
        offsets,
        offsets_32,
        indices,
        indices_32,
        weights,
        batch_size,
        num_rows,
        average_len,
        corner_case);
    const int64_t* offsets_or_lengths =
        (use_offsets ? offsets : lengths).data();
    const int32_t* offsets_or_lengths_32 =
        (use_offsets ? offsets_32 : lengths_32).data();

    vector<float> output_sls_ref(batch_size * embedding_dim);
    vector<float> output_slws_ref(output_sls_ref.size()),
        output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

    vector<float>& output_ref = use_weight ? output_slws_ref : output_sls_ref;
    vector<float>& output = use_weight ? output_slws : output_sls;
    bool success, success_ref;

    if (isOffset64b) {
      if (isIndex64b) {
        success_ref = fbgemm::EmbeddingSpMDMNBitRowWiseSparse_ref<int64_t>(
            bit_rate,
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            mapping_table.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDMNBitRowWiseSparse<int64_t, int64_t>(
            bit_rate,
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            offsets_or_lengths,
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
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            mapping_table.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDMNBitRowWiseSparse<int32_t, int64_t>(
            bit_rate,
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            output.data(),
            mapping_table.data());
      }
    } else {
      if (isIndex64b) {
        success_ref = fbgemm::EmbeddingSpMDMNBitRowWiseSparse_ref<int64_t>(
            bit_rate,
            embedding_dim,
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            mapping_table.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDMNBitRowWiseSparse<int64_t>(
            bit_rate,
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices.data(),
            offsets_or_lengths_32,
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
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            mapping_table.data(),
            offsets_or_lengths,
            use_weight ? weights.data() : nullptr,
            normalize_by_lengths,
            output_ref.data(),
            is_wt_positional,
            use_offsets);

        auto kernel = GenerateEmbeddingSpMDMNBitRowWiseSparse<int32_t>(
            bit_rate,
            embedding_dim,
            use_weight,
            normalize_by_lengths,
            prefetch,
            is_wt_positional,
            use_offsets);
        success = kernel(
            batch_size,
            lengths_sum,
            num_rows,
            fused_embedding_table.data(),
            corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
            offsets_or_lengths_32,
            use_weight ? weights.data() : nullptr,
            output.data(),
            mapping_table.data());
      }
    }

    // Check correctness
    EXPECT_EQ(success, success_ref)
        << "Reference and JIT impl did not both succeed";
    if (corner_case == OUT_OF_BOUND_INDICES ||
        corner_case == UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM) {
      EXPECT_EQ(success, false);
    }
    if (success) {
      for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], output_ref[i])
            << "results differ at (" << i << ") reference: " << output_ref[i]
            << ", FBGEMM: " << output[i] << " emb dim :" << embedding_dim;
      }
    }
  } // end for input
}
