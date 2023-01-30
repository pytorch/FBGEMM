/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <numeric> // for accumulate and iota
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

class EmbeddingSpMDMTest : public testing::TestWithParam<tuple<
                               int,
                               EmbeddingSpMDMWeightChoice,
                               EmbeddingSpMDMCornerCase,
                               EmbeddingSpMDMDtypeChoice,
                               EmbeddingSpMDMDtypeChoice>> {};

class rowwiseSparseEmbeddingSpMDMTest
    : public testing::TestWithParam<
          tuple<int, EmbeddingSpMDMWeightChoice, EmbeddingSpMDMCornerCase>> {};

class IndexRemapTest
    : public testing::TestWithParam<tuple<int, int, int, bool, bool>> {};
}; // namespace

vector<int> prefetch_distances = {0, 16, 1000000};

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    EmbeddingSpMDMTest,
    ::testing::Combine(
        ::testing::ValuesIn(prefetch_distances),
        ::testing::Values(
            UNWEIGHTED,
            WEIGHTED,
            POSITIONAL_WEIGHTED), // use_weight
        ::testing::Values(
            NONE,
            EMPTY_INDICES,
            OUT_OF_BOUND_INDICES,
            UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM),
        ::testing::Values(FLOAT, FLOAT16, BFLOAT16),
        ::testing::Values(FLOAT, FLOAT16, BFLOAT16)));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    rowwiseSparseEmbeddingSpMDMTest,
    ::testing::Combine(
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

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    IndexRemapTest,
    ::testing::Combine(
        ::testing::ValuesIn({1, 2, 5, 10}), // batch size
        ::testing::ValuesIn({1, 50, 100, 1000}), // number of rows
        ::testing::ValuesIn({1, 5, 16}), // avg len
        ::testing::Bool(), // is index 64 bit?
        ::testing::Bool())); // per sample weights?

TEST_P(EmbeddingSpMDMTest, basicTest) {
  vector<vector<int>> inputs(GetInputs_());

  default_random_engine generator;
  uniform_int_distribution<> bool_dist(0, 1);

  bool isIndex64b = bool_dist(generator);
  bool isOffset64b = bool_dist(generator);
  bool normalize_by_lengths = bool_dist(generator);
  bool use_offsets = bool_dist(generator);
  bool use_output_input_stride = bool_dist(generator);
  bool test_thread_local = bool_dist(generator);
  int prefetch;
  EmbeddingSpMDMWeightChoice weight_choice;
  EmbeddingSpMDMCornerCase corner_case;
  EmbeddingSpMDMDtypeChoice in_type;
  EmbeddingSpMDMDtypeChoice out_type;
  tie(prefetch, weight_choice, corner_case, in_type, out_type) = GetParam();
  bool is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  bool use_weight = weight_choice != UNWEIGHTED;
  bool isFp16 = in_type == FLOAT16;
  bool isBf16 = in_type == BFLOAT16;
  bool is_output_float = out_type == FLOAT;
  bool is_output_bfloat16 = out_type == BFLOAT16;

  if (isBf16 ^ is_output_bfloat16) {
    // only support both in and out are bf16 now
    return;
  }
  if (corner_case != NONE || is_wt_positional) {
    // Check corner case only for subset of tests.
    if (isFp16 || normalize_by_lengths || use_output_input_stride ||
        !is_output_float || test_thread_local) {
      return;
    }
  }
  if (is_wt_positional && !use_weight) {
    // weight positional only makes sense when use_weight is true
    return;
  }

  for (auto input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];
    int output_stride = use_output_input_stride ? embedding_dim * 2 + 3 : -1;
    int input_stride = use_output_input_stride ? embedding_dim * 2 + 3 : -1;

    // Create embedding table
    vector<float> embedding_table(
        num_rows * (use_output_input_stride ? input_stride : embedding_dim));
    normal_distribution<float> embedding_distribution;
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < embedding_dim; ++j) {
        embedding_table
            [i * (use_output_input_stride ? input_stride : embedding_dim) + j] =
                embedding_distribution(generator);
      }
    }
    vector<float16> embedding_table_fp16;
    if (isFp16) {
      embedding_table_fp16.resize(embedding_table.size());
      FloatToFloat16_simd(
          embedding_table.data(),
          embedding_table_fp16.data(),
          embedding_table.size());
    }

    vector<bfloat16> embedding_table_bf16;
    if (isBf16) {
      embedding_table_bf16.resize(embedding_table.size());
      FloatToBfloat16_simd(
          embedding_table.data(),
          embedding_table_bf16.data(),
          embedding_table.size());
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
    int output_size_wo_sentries =
        batch_size * (use_output_input_stride ? output_stride : embedding_dim);
    vector<float> output_ref(output_size_wo_sentries + num_sentries);
    vector<float> output(output_ref.size());
    vector<float16> output_ref_fp16(output.size()), output_fp16(output.size());
    vector<bfloat16> output_ref_bf16(output.size()), output_bf16(output.size());
    for (size_t i = output_size_wo_sentries; i < output.size(); ++i) {
      output_ref[i] = sentry_value;
      output[i] = sentry_value;
      output_ref_fp16[i] = cpu_float2half_rn(sentry_value);
      output_fp16[i] = cpu_float2half_rn(sentry_value);
      FloatToBfloat16_ref(&sentry_value, &output_ref_bf16[i], 1);
      FloatToBfloat16_ref(&sentry_value, &output_bf16[i], 1);
    }

    bool success, success_ref;

#define TEST_BASE(                                             \
    table,                                                     \
    indices,                                                   \
    offsets_or_lengths,                                        \
    output_ref,                                                \
    output,                                                    \
    InType,                                                    \
    IndexType,                                                 \
    OffsetType,                                                \
    OutType,                                                   \
    THREAD_LOCAL)                                              \
  success_ref = EmbeddingSpMDM_ref(                            \
      embedding_dim,                                           \
      batch_size,                                              \
      lengths_sum,                                             \
      num_rows,                                                \
      table.data(),                                            \
      corner_case == EMPTY_INDICES ? nullptr : indices.data(), \
      offsets_or_lengths,                                      \
      use_weight ? weights.data() : nullptr,                   \
      normalize_by_lengths,                                    \
      output_ref.data(),                                       \
      is_wt_positional,                                        \
      use_offsets,                                             \
      output_stride,                                           \
      input_stride,                                            \
      true,                                                    \
      false,                                                   \
      isBf16);                                                 \
                                                               \
  auto kernel = GenerateEmbeddingSpMDMWithStrides<             \
      InType,                                                  \
      IndexType,                                               \
      OffsetType,                                              \
      OutType,                                                 \
      THREAD_LOCAL>(                                           \
      embedding_dim,                                           \
      use_weight,                                              \
      normalize_by_lengths,                                    \
      prefetch,                                                \
      is_wt_positional,                                        \
      use_offsets,                                             \
      output_stride,                                           \
      input_stride,                                            \
      true,                                                    \
      false,                                                   \
      isBf16);                                                 \
  success = kernel(                                            \
      batch_size,                                              \
      lengths_sum,                                             \
      num_rows,                                                \
      table.data(),                                            \
      corner_case == EMPTY_INDICES ? nullptr : indices.data(), \
      offsets_or_lengths,                                      \
      use_weight ? weights.data() : nullptr,                   \
      output.data());

#define TEST_THREAD_LOCAL(  \
    table,                  \
    indices,                \
    offsets_or_lengths,     \
    output_ref,             \
    output,                 \
    InType,                 \
    IndexType,              \
    OffsetType,             \
    OutType)                \
  if (test_thread_local) {  \
    TEST_BASE(              \
        table,              \
        indices,            \
        offsets_or_lengths, \
        output_ref,         \
        output,             \
        InType,             \
        IndexType,          \
        OffsetType,         \
        OutType,            \
        true);              \
  } else {                  \
    TEST_BASE(              \
        table,              \
        indices,            \
        offsets_or_lengths, \
        output_ref,         \
        output,             \
        InType,             \
        IndexType,          \
        OffsetType,         \
        OutType,            \
        false);             \
  }

#define TEST_OUT_TYPE(                                                 \
    table, indices, offsets_or_lengths, InType, IndexType, OffsetType) \
  if (is_output_float) {                                               \
    TEST_THREAD_LOCAL(                                                 \
        table,                                                         \
        indices,                                                       \
        offsets_or_lengths,                                            \
        output_ref,                                                    \
        output,                                                        \
        InType,                                                        \
        IndexType,                                                     \
        OffsetType,                                                    \
        float);                                                        \
  } else if (is_output_bfloat16) {                                     \
    TEST_THREAD_LOCAL(                                                 \
        table,                                                         \
        indices,                                                       \
        offsets_or_lengths,                                            \
        output_ref_bf16,                                               \
        output_bf16,                                                   \
        InType,                                                        \
        IndexType,                                                     \
        OffsetType,                                                    \
        bfloat16);                                                     \
  } else {                                                             \
    TEST_THREAD_LOCAL(                                                 \
        table,                                                         \
        indices,                                                       \
        offsets_or_lengths,                                            \
        output_ref_fp16,                                               \
        output_fp16,                                                   \
        InType,                                                        \
        IndexType,                                                     \
        OffsetType,                                                    \
        float16);                                                      \
  }

#define TEST_OFFSET_TYPE(table, indices, InType, IndexType)                 \
  if (isOffset64b) {                                                        \
    TEST_OUT_TYPE(                                                          \
        table, indices, offsets_or_lengths, InType, IndexType, int64_t);    \
  } else {                                                                  \
    TEST_OUT_TYPE(                                                          \
        table, indices, offsets_or_lengths_32, InType, IndexType, int32_t); \
  }

#define TEST_INDEX_TYPE(table, InType)                    \
  if (isIndex64b) {                                       \
    TEST_OFFSET_TYPE(table, indices, InType, int64_t);    \
  } else {                                                \
    TEST_OFFSET_TYPE(table, indices_32, InType, int32_t); \
  }

    if (isFp16) {
      TEST_INDEX_TYPE(embedding_table_fp16, float16);
    } else if (isBf16) {
      TEST_INDEX_TYPE(embedding_table_bf16, bfloat16);
    } else {
      TEST_INDEX_TYPE(embedding_table, float);
    }

#undef TEST_INDEX_TYPE
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

    auto get_actual = [&](int offset) {
      if (is_output_float)
        return output[offset];
      else if (is_output_bfloat16) {
        float v;
        Bfloat16ToFloat_ref(&output_bf16[offset], &v, 1);
        return v;
      } else
        return cpu_half2float(output_fp16[offset]);
    };

    auto get_expected = [&](int offset) {
      if (is_output_float)
        return output_ref[offset];
      else if (is_output_bfloat16) {
        float v;
        Bfloat16ToFloat_ref(&output_ref_bf16[offset], &v, 1);
        return v;
      } else
        return cpu_half2float(output_ref_fp16[offset]);
    };

    if (success) {
      for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
          int offset =
              i * (use_output_input_stride ? output_stride : embedding_dim) + j;
          float actual = get_actual(offset);
          float expected = get_expected(offset);
          EXPECT_EQ(actual, expected)
              << "results differ at (" << i << ") reference: " << expected
              << ", FBGEMM: " << actual << " emb dim :" << embedding_dim;
        }
      }
      for (int offset = output_size_wo_sentries;
           offset < output_size_wo_sentries + num_sentries;
           ++offset) {
        float actual = get_actual(offset);
        float expected = get_expected(offset);
        EXPECT_EQ(actual, expected)
            << "results differ at (" << offset << ") reference: " << expected
            << ", FBGEMM: " << actual << " emb dim :" << embedding_dim;
      }
    }
  } // end for input
}

TEST_P(rowwiseSparseEmbeddingSpMDMTest, rowwiseSparseTest) {
  vector<vector<int>> inputs(GetInputs_());

  default_random_engine generator;
  uniform_int_distribution<> bool_dist(0, 1);

  bool isFp16 = bool_dist(generator);
  bool isIndex64b = bool_dist(generator);
  bool isOffset64b = bool_dist(generator);
  bool normalize_by_lengths = bool_dist(generator);
  bool use_offsets = bool_dist(generator);
  bool is_output_float = bool_dist(generator);
  int prefetch;
  EmbeddingSpMDMWeightChoice weight_choice;
  EmbeddingSpMDMCornerCase corner_case;
  tie(prefetch, weight_choice, corner_case) = GetParam();
  bool is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  bool use_weight = weight_choice != UNWEIGHTED;

  if (!is_output_float) {
    // Don't test is_output_float for row-wise sparse embedding spmdm
    return;
  }

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
    vector<float> embedding_table(num_compressed_rows * embedding_dim);
    normal_distribution<float> embedding_distribution;
    for (size_t i = 0; i < embedding_table.size(); ++i) {
      embedding_table[i] = embedding_distribution(generator);
    }
    vector<float16> embedding_table_fp16;
    if (isFp16) {
      embedding_table_fp16.resize(embedding_table.size());
      FloatToFloat16_simd(
          embedding_table.data(),
          embedding_table_fp16.data(),
          embedding_table.size());
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
        if (isFp16) {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel =
              GenerateEmbeddingSpMDMRowWiseSparse<float16, int64_t, int64_t>(
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
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        } else {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel =
              GenerateEmbeddingSpMDMRowWiseSparse<float, int64_t, int64_t>(
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
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        }
      } else {
        if (isFp16) {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel =
              GenerateEmbeddingSpMDMRowWiseSparse<float16, int32_t, int64_t>(
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
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        } else {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel =
              GenerateEmbeddingSpMDMRowWiseSparse<float, int32_t, int64_t>(
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
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        }
      }
    } else {
      if (isIndex64b) {
        if (isFp16) {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<float16, int64_t>(
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
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        } else {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<float, int64_t>(
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
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        }
      } else {
        if (isFp16) {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<float16, int32_t>(
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
              embedding_table_fp16.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        } else {
          success_ref = EmbeddingSpMDMRowWiseSparse_ref(
              embedding_dim,
              batch_size,
              lengths_sum,
              num_rows,
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              mapping_table.data(),
              offsets_or_lengths,
              use_weight ? weights.data() : nullptr,
              normalize_by_lengths,
              output_ref.data(),
              is_wt_positional,
              use_offsets);

          auto kernel = GenerateEmbeddingSpMDMRowWiseSparse<float, int32_t>(
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
              embedding_table.data(),
              corner_case == EMPTY_INDICES ? nullptr : indices_32.data(),
              offsets_or_lengths_32,
              use_weight ? weights.data() : nullptr,
              output.data(),
              mapping_table.data());
        }
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

TEST_P(IndexRemapTest, basicTest) {
  int batch_size, num_rows, avg_len;
  bool isIndex64b, per_sample_weights;
  tie(batch_size, num_rows, avg_len, isIndex64b, per_sample_weights) =
      GetParam();
  constexpr float sparsity = 0.5;

  vector<int64_t> lengths, offsets, indices;
  vector<int32_t> lengths_32, offsets_32, indices_32;
  vector<float> weights;
  GenerateLengthsIndicesWeights(
      lengths,
      lengths_32,
      offsets,
      offsets_32,
      indices,
      indices_32,
      weights,
      batch_size,
      num_rows,
      avg_len, // average number of indices in a batch
      EmbeddingSpMDMCornerCase::NONE);

  // Create mapping table for rowwise sparsity
  vector<int32_t> mapping_table;
  CreateMappingTableForRowWiseSparsity(mapping_table, num_rows, sparsity);

  // outputs
  vector<int32_t> out_indices_32(indices_32.size(), 0);
  vector<int32_t> out_offsets_32(offsets_32.size(), 0);
  vector<float> out_weights(weights.size(), 0);

  vector<int64_t> out_indices(indices.size(), 0);
  vector<int64_t> out_offsets(offsets.size(), 0);

  // reference outputs
  vector<int32_t> out_indices_32_ref(indices_32.size(), 0);
  vector<int32_t> out_offsets_32_ref(offsets_32.size(), 0);
  vector<float> out_weights_ref(weights.size(), 0);

  vector<int64_t> out_indices_ref(indices.size(), 0);
  vector<int64_t> out_offsets_ref(offsets.size(), 0);

  // number of elements in the offset array ( it's equal to batch_size + 1)
  int offset_numel = offsets_32.size();

  if (isIndex64b) {
    if (per_sample_weights) {
      compressed_indices_remap<int64_t>(
          offset_numel,
          indices.data(),
          mapping_table.data(),
          offsets.data(),
          weights.data(),
          out_indices.data(),
          out_offsets.data(),
          out_weights.data());

      compressed_indices_remap_ref<int64_t>(
          offset_numel,
          indices.data(),
          mapping_table.data(),
          offsets.data(),
          weights.data(),
          out_indices_ref.data(),
          out_offsets_ref.data(),
          out_weights_ref.data());
    } else {
      compressed_indices_remap<int64_t>(
          offset_numel,
          indices.data(),
          mapping_table.data(),
          offsets.data(),
          nullptr,
          out_indices.data(),
          out_offsets.data(),
          nullptr);

      compressed_indices_remap_ref<int64_t>(
          offset_numel,
          indices.data(),
          mapping_table.data(),
          offsets.data(),
          nullptr,
          out_indices_ref.data(),
          out_offsets_ref.data(),
          nullptr);
    }
  } else {
    if (per_sample_weights) {
      compressed_indices_remap<int32_t>(
          offset_numel,
          indices_32.data(),
          mapping_table.data(),
          offsets_32.data(),
          weights.data(),
          out_indices_32.data(),
          out_offsets_32.data(),
          out_weights.data());

      compressed_indices_remap_ref<int32_t>(
          offset_numel,
          indices_32.data(),
          mapping_table.data(),
          offsets_32.data(),
          weights.data(),
          out_indices_32_ref.data(),
          out_offsets_32_ref.data(),
          out_weights_ref.data());
    } else {
      compressed_indices_remap<int32_t>(
          offset_numel,
          indices_32.data(),
          mapping_table.data(),
          offsets_32.data(),
          nullptr,
          out_indices_32.data(),
          out_offsets_32.data(),
          nullptr);

      compressed_indices_remap_ref<int32_t>(
          offset_numel,
          indices_32.data(),
          mapping_table.data(),
          offsets_32.data(),
          nullptr,
          out_indices_32_ref.data(),
          out_offsets_32_ref.data(),
          nullptr);
    }
  }

  if (isIndex64b) {
    EXPECT_EQ(out_offsets, out_offsets_ref) << "offsets don't match";
    for (int i = 0; i < out_offsets[offset_numel - 1]; ++i) {
      EXPECT_EQ(out_indices[i], out_indices_ref[i])
          << "indices don't match at " << i;
    }
  } else {
    EXPECT_EQ(out_offsets_32, out_offsets_32_ref) << "offsets don't match";
    for (int i = 0; i < out_offsets_32[offset_numel - 1]; ++i) {
      EXPECT_EQ(out_indices_32[i], out_indices_32_ref[i])
          << "indices don't match at " << i;
    }
  }

  if (per_sample_weights) {
    size_t len = isIndex64b ? out_offsets[offset_numel - 1]
                            : out_offsets_32[offset_numel - 1];

    for (size_t i = 0; i < len; ++i) {
      EXPECT_EQ(out_weights[i], out_weights_ref[i])
          << "weights don't match at" << i;
    }
  }
}
