/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <random>

#include <gtest/gtest.h>

#include "./EmbeddingSpMDMTestUtils.h"
#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmConvert.h"
#include "src/RefImplementations.h" // @manual

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

static vector<int> prefetch_distances{0, 16, 1000000};

static float fp16_tolerance(int average_len, float expected) {
  constexpr float eps = 9.77e-4f;
  // NBit scale/bias are natively fp16 (no fp32 narrowing), so error
  // is dominated by fp16 accumulation rounding. With random data,
  // errors follow a random walk: O(sqrt(avg_len) * eps * |value|).
  // 2x safety margin for tail distribution.
  return eps + 2.0f * sqrtf(average_len) * eps * abs(expected);
}

static float clamp_for_fp16(float val) {
  return std::min(std::abs(val), 1.0f);
}

namespace {

class FusedNBitRowwiseEmbeddingLookupTest : public testing::TestWithParam<tuple<
                                                int,
                                                int,
                                                EmbeddingSpMDMWeightChoice,
                                                EmbeddingSpMDMCornerCase,
                                                EmbeddingSpMDMDtypeChoice,
                                                EmbeddingSpMDMKernelChoice>> {};
}; // namespace

INSTANTIATE_TEST_SUITE_P(
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
            UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM),
        ::testing::Values(FLOAT, FLOAT16, BFLOAT16),
        ::testing::Values(DISPATCH_DEFAULT, DISPATCH_AUTOVEC)));

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, basicTest) {
  vector<vector<int>> inputs(GetInputs_());

  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<> bool_dist(0, 1);

  bool isIndex64b = bool_dist(generator);
  bool isOffset64b = bool_dist(generator);
  bool normalize_by_lengths = bool_dist(generator);
  bool use_offsets = bool_dist(generator);
  bool scale_bias_last = bool_dist(generator);
  bool test_thread_local = bool_dist(generator);
  int bit_rate = 0, prefetch = 0;
  EmbeddingSpMDMWeightChoice weight_choice{};
  EmbeddingSpMDMCornerCase corner_case{};
  EmbeddingSpMDMDtypeChoice out_type{};
  EmbeddingSpMDMKernelChoice kernel_choice{};
  tie(bit_rate, prefetch, weight_choice, corner_case, out_type, kernel_choice) =
      GetParam();
  ScopedKernelOverride kernel_override(kernel_choice);
  bool is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  bool use_weight = weight_choice != UNWEIGHTED;
  bool is_bf16_out = out_type == BFLOAT16;

  if (corner_case != NONE || weight_choice == POSITIONAL_WEIGHTED) {
    // Check corner case only for subset of tests.
    if (normalize_by_lengths || out_type != FLOAT || !scale_bias_last ||
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
      if (is_sve_fp16_enabled()) {
        scale = clamp_for_fp16(scale);
        bias = clamp_for_fp16(bias);
      }
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

    if (is_sve_fp16_enabled()) {
      for (auto& w : weights) {
        w = abs(w);
      }
    }

    if (!scale_bias_last && use_weight) {
      // When scale_bias_last == false, assume this is for table batched
      // embedding (TBE) that can get -1 for pruned rows.
      uniform_int_distribution<int> pruned_indices_distribution(
          0, indices.size() - 1);
      constexpr float PRUNED_INDICES_PROPORTION = 0.1;
      for (int i = 0; i < indices.size() * PRUNED_INDICES_PROPORTION; ++i) {
        auto idx = pruned_indices_distribution(generator);
        indices[idx] = -1;
        indices_32[idx] = -1;
      }
    }

    // Sentries at the end to make sure masking is done correctly not to write
    // out of bounds.
    constexpr int num_sentries = 10;
    const float sentry_value = 1.0f;
    int output_size_wo_sentries = batch_size * embedding_dim;
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

    bool success = false, success_ref = false;

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
      scale_bias_last,                                                  \
      is_bf16_out);                                                     \
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
      scale_bias_last,                                                  \
      is_bf16_out);                                                     \
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
  if (out_type == FLOAT) {                                                \
    TEST_THREAD_LOCAL(                                                    \
        indices,                                                          \
        offsets_or_lengths,                                               \
        output_ref,                                                       \
        output,                                                           \
        IndexType,                                                        \
        OffsetType,                                                       \
        float);                                                           \
  } else if (out_type == BFLOAT16) {                                      \
    TEST_THREAD_LOCAL(                                                    \
        indices,                                                          \
        offsets_or_lengths,                                               \
        output_ref_bf16,                                                  \
        output_bf16,                                                      \
        IndexType,                                                        \
        OffsetType,                                                       \
        bfloat16);                                                        \
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

    auto get_actual = [&](int offset) {
      if (out_type == FLOAT) {
        return output[offset];
      } else if (out_type == BFLOAT16) {
        return cpu_bf162float(output[offset]);
      } else {
        return cpu_half2float(output[offset]);
      }
    };

    auto get_expected = [&](int offset) {
      if (out_type == FLOAT) {
        return output_ref[offset];
      } else if (out_type == BFLOAT16) {
        return cpu_bf162float(output_ref[offset]);
      } else {
        return cpu_half2float(output_ref[offset]);
      }
    };

    if (success) {
      for (size_t i = 0; i < output.size(); ++i) {
        float actual = get_actual(i);
        float expected = get_expected(i);
        if (is_sve_fp16_enabled() && out_type == FLOAT16) {
          EXPECT_NEAR(actual, expected, fp16_tolerance(average_len, expected))
              << "results differ at (" << i << ") reference: " << expected
              << ", FBGEMM: " << actual << " emb dim :" << embedding_dim;
        } else {
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
        if (is_sve_fp16_enabled() && out_type == FLOAT16) {
          EXPECT_NEAR(actual, expected, fp16_tolerance(average_len, expected))
              << "results differ at (" << offset << ") reference: " << expected
              << ", FBGEMM: " << actual << " emb dim :" << embedding_dim;
        } else {
          EXPECT_EQ(actual, expected)
              << "results differ at (" << offset << ") reference: " << expected
              << ", FBGEMM: " << actual << " emb dim :" << embedding_dim;
        }
      }
    }
  } // end for input
}

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, fp16CorrectnessTest) {
  auto
      [bit_rate,
       prefetch,
       weight_choice,
       corner_case,
       out_type,
       kernel_choice] = GetParam();
  if (!is_sve_fp16_enabled() || out_type != FLOAT16) {
    return;
  }
  ScopedKernelOverride kernel_override(kernel_choice);
  bool use_weight = weight_choice != UNWEIGHTED;
  bool is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  int num_elem_per_byte = 8 / bit_rate;

  vector<vector<int>> inputs = {
      // {batch, rows, dim, avg_len}
      {4, 100, 2, 4},
      {4, 100, 4, 10},
      {4, 100, 7, 4},
      {4, 100, 8, 4},
      {4, 100, 9, 4},
      {4, 100, 10, 4},
      {4, 100, 16, 4},
      {4, 100, 17, 4},
      {4, 100, 19, 4},
      {4, 100, 24, 10},
      {4, 100, 32, 10},
      {4, 100, 33, 10},
      {4, 100, 48, 10},
      {4, 100, 64, 10},
      {4, 100, 65, 10},
      {10, 40, 93, 10},
      {10, 40, 95, 10},
      {4, 100, 128, 10},
      {4, 100, 256, 4},
      {4, 100, 500, 4},
      {4, 100, 512, 4},
      {4, 100, 900, 4},
      {4, 100, 1024, 4},
      {4, 100, 40, 10}, // ITERS=5, no tail — Phase 1 fill-in
      {4, 100, 56, 10}, // ITERS=7, no tail — Phase 1 fill-in
      {4, 100, 72, 10}, // ITERS=9, no tail — Phase 1b lower boundary
      {4, 100, 79, 10}, // ITERS=9, tail=7 — Phase 1b lower boundary + max tail
      {4, 100, 136, 10}, // ITERS=17, no tail — Tiled lower boundary,
      // LAST_TILE_ITERS=1
      {4, 100, 143, 10}, // ITERS=17, tail=7 — Tiled lower boundary + tail
      {4, 100, 192, 10}, // ITERS=24, no tail — Tiled interior,
      // LAST_TILE_ITERS=8
      {4, 100, 200, 4}, // ITERS=25, tail=0 — additional Tiled interior
      {4, 100, 263, 4}, // ITERS=32, tail=7 — Tiled upper boundary with tail
      {10, 4000, 32, 100}, // production-scale avg_len
      {10, 4000, 64, 100}, // production-scale avg_len
  };

  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<int> entries(0, 16);

  for (auto& input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    int fused_embedding_dim =
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte +
        2 * sizeof(float16);

    // ---- Test 1: Integer scale/bias (should be exact) ----
    {
      vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim);
      for (int i = 0; i < num_rows; i++) {
        for (int ii = 0;
             ii < (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
             ii++) {
          fused_embedding_table[i * fused_embedding_dim + ii] =
              entries(generator);
        }
        float16* scale_bias = reinterpret_cast<float16*>(
            fused_embedding_table.data() + i * fused_embedding_dim +
            (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte);
        float scale = 1.0f;
        float bias = 0.0f;
        FloatToFloat16_ref(&scale, scale_bias, 1, true);
        FloatToFloat16_ref(&bias, scale_bias + 1, 1, true);
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

      if (use_weight) {
        for (auto& w : weights) {
          w = 1.0f;
        }
      }

      int output_size = batch_size;
      vector<float16> output_ref_fp16(output_size * embedding_dim);
      vector<float16> output_fp16(output_size * embedding_dim);

      bool success_ref = EmbeddingSpMDMNBit_ref<int64_t, int64_t, float16>(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          false,
          output_ref_fp16.data(),
          is_wt_positional,
          true,
          -1,
          -1,
          true,
          false);

      auto kernel =
          GenerateEmbeddingSpMDMNBitWithStrides<int64_t, int64_t, float16>(
              bit_rate,
              embedding_dim,
              use_weight,
              false,
              prefetch,
              is_wt_positional,
              true,
              -1,
              -1,
              true,
              false);

      bool success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          output_fp16.data());

      ASSERT_EQ(success, success_ref)
          << "Integer test: ref and kernel disagree"
          << " bit_rate=" << bit_rate << " dim=" << embedding_dim;
      if (corner_case == OUT_OF_BOUND_INDICES ||
          corner_case == UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM) {
        EXPECT_FALSE(success);
      }

      if (success) {
        for (int i = 0; i < output_size * embedding_dim; ++i) {
          float actual = cpu_half2float(output_fp16[i]);
          float expected = cpu_half2float(output_ref_fp16[i]);
          EXPECT_EQ(actual, expected)
              << "Integer test MISMATCH at i=" << i << " bit_rate=" << bit_rate
              << " dim=" << embedding_dim << " expected=" << expected
              << " actual=" << actual;
        }
      }
    }

    // ---- Test 2: fp16-representable float scale/bias ----
    {
      vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim);
      for (int i = 0; i < num_rows; i++) {
        for (int ii = 0;
             ii < (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
             ii++) {
          fused_embedding_table[i * fused_embedding_dim + ii] =
              entries(generator);
        }
        float16* scale_bias = reinterpret_cast<float16*>(
            fused_embedding_table.data() + i * fused_embedding_dim +
            (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte);
        float scale = 0.25f;
        float bias = 0.5f;
        FloatToFloat16_ref(&scale, scale_bias, 1, true);
        FloatToFloat16_ref(&bias, scale_bias + 1, 1, true);
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

      if (use_weight) {
        for (auto& w : weights) {
          w = 1.0f;
        }
      }

      int output_size = batch_size;
      vector<float16> output_ref_fp16(output_size * embedding_dim);
      vector<float16> output_fp16(output_size * embedding_dim);

      bool success_ref = EmbeddingSpMDMNBit_ref<int64_t, int64_t, float16>(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          false,
          output_ref_fp16.data(),
          is_wt_positional,
          true,
          -1,
          -1,
          true,
          false);

      auto kernel =
          GenerateEmbeddingSpMDMNBitWithStrides<int64_t, int64_t, float16>(
              bit_rate,
              embedding_dim,
              use_weight,
              false,
              prefetch,
              is_wt_positional,
              true,
              -1,
              -1,
              true,
              false);

      bool success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          output_fp16.data());

      ASSERT_EQ(success, success_ref);
      if (corner_case == OUT_OF_BOUND_INDICES ||
          corner_case == UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM) {
        EXPECT_FALSE(success);
      }

      if (success) {
        for (int i = 0; i < output_size * embedding_dim; ++i) {
          float actual = cpu_half2float(output_fp16[i]);
          float expected = cpu_half2float(output_ref_fp16[i]);
          EXPECT_EQ(actual, expected)
              << "FP16 representable test at i=" << i
              << " bit_rate=" << bit_rate << " dim=" << embedding_dim
              << " expected=" << expected << " actual=" << actual;
        }
      }
    }

    // ---- Test 3: Random scale/bias with normalization ----
    for (int norm = 0; norm <= 1; ++norm) {
      bool normalize = (norm == 1);
      normal_distribution<float> embedding_distribution;

      vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim);
      for (int i = 0; i < num_rows; i++) {
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
        if (is_sve_fp16_enabled()) {
          scale = clamp_for_fp16(scale);
          bias = clamp_for_fp16(bias);
        }
        FloatToFloat16_ref(&scale, scale_bias, 1, true);
        FloatToFloat16_ref(&bias, scale_bias + 1, 1, true);
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

      if (is_sve_fp16_enabled()) {
        for (auto& w : weights) {
          w = abs(w);
        }
      }

      int output_size = batch_size;
      constexpr int num_sentries = 10;
      const float16 sentry_value = cpu_float2half_rn(1.0f);
      vector<float16> output_ref_fp16(
          output_size * embedding_dim + num_sentries, sentry_value);
      vector<float16> output_fp16(output_ref_fp16.size(), sentry_value);

      bool success_ref = EmbeddingSpMDMNBit_ref<int64_t, int64_t, float16>(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          normalize,
          output_ref_fp16.data(),
          is_wt_positional,
          true,
          -1,
          -1,
          true,
          false);

      auto kernel =
          GenerateEmbeddingSpMDMNBitWithStrides<int64_t, int64_t, float16>(
              bit_rate,
              embedding_dim,
              use_weight,
              normalize,
              prefetch,
              is_wt_positional,
              true,
              -1,
              -1,
              true,
              false);

      bool success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          corner_case == EMPTY_INDICES ? nullptr : indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          output_fp16.data());

      ASSERT_EQ(success, success_ref);
      if (corner_case == OUT_OF_BOUND_INDICES ||
          corner_case == UNMATCHED_NUM_INDICES_AND_LENGTHS_SUM) {
        EXPECT_FALSE(success);
      }

      if (success) {
        for (int i = 0; i < output_size * embedding_dim; ++i) {
          float actual = cpu_half2float(output_fp16[i]);
          float expected = cpu_half2float(output_ref_fp16[i]);
          EXPECT_NEAR(actual, expected, fp16_tolerance(average_len, expected))
              << "Random test at i=" << i << " bit_rate=" << bit_rate
              << " dim=" << embedding_dim << " avg_len=" << average_len
              << " norm=" << normalize << " wt=" << use_weight;
        }
        for (int i = output_size * embedding_dim;
             i < output_size * embedding_dim + num_sentries;
             ++i) {
          EXPECT_EQ(output_fp16[i], sentry_value)
              << "Sentry overwritten at i=" << i << " bit_rate=" << bit_rate
              << " dim=" << embedding_dim;
        }
      }
    }
  }
}

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, fp16ScaleBiasFirstTest) {
  auto
      [bit_rate,
       prefetch,
       weight_choice,
       corner_case,
       out_type,
       kernel_choice] = GetParam();
  if (!is_sve_fp16_enabled() || out_type != FLOAT16 || corner_case != NONE) {
    return;
  }
  ScopedKernelOverride kernel_override(kernel_choice);
  bool use_weight = weight_choice != UNWEIGHTED;
  bool is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  int num_elem_per_byte = 8 / bit_rate;

  vector<vector<int>> inputs = {
      // {batch, rows, dim, avg_len}
      {4, 100, 2, 4},
      {4, 100, 4, 10},
      {4, 100, 7, 4},
      {4, 100, 8, 4},
      {4, 100, 9, 4},
      {4, 100, 10, 4},
      {4, 100, 16, 4},
      {4, 100, 17, 4},
      {4, 100, 19, 4},
      {4, 100, 24, 10},
      {4, 100, 32, 10},
      {4, 100, 33, 10},
      {4, 100, 48, 10},
      {4, 100, 64, 10},
      {4, 100, 65, 10},
      {10, 40, 93, 10},
      {10, 40, 95, 10},
      {4, 100, 128, 10},
      {4, 100, 256, 4},
      {4, 100, 500, 4},
      {4, 100, 512, 4},
      {4, 100, 900, 4},
      {4, 100, 1024, 4},
      {4, 100, 40, 10}, // ITERS=5, no tail — Phase 1 fill-in
      {4, 100, 56, 10}, // ITERS=7, no tail — Phase 1 fill-in
      {4, 100, 72, 10}, // ITERS=9, no tail — Phase 1b lower boundary
      {4, 100, 79, 10}, // ITERS=9, tail=7 — Phase 1b lower boundary + max tail
      {4, 100, 136, 10}, // ITERS=17, no tail — Tiled lower boundary,
      // LAST_TILE_ITERS=1
      {4, 100, 143, 10}, // ITERS=17, tail=7 — Tiled lower boundary + tail
      {4, 100, 192, 10}, // ITERS=24, no tail — Tiled interior,
      // LAST_TILE_ITERS=8
      {4, 100, 200, 4}, // ITERS=25, tail=0 — additional Tiled interior
      {4, 100, 263, 4}, // ITERS=32, tail=7 — Tiled upper boundary with tail
      {10, 4000, 32, 100}, // production-scale avg_len
      {10, 4000, 64, 100}, // production-scale avg_len
  };

  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<int> entries(0, 16);
  normal_distribution<float> embedding_distribution;

  for (auto& input : inputs) {
    int batch_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];
    int average_len = input[3];

    int packed_dim =
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
    int fused_embedding_dim = packed_dim + 2 * sizeof(float16);

    for (int norm = 0; norm <= 1; ++norm) {
      bool normalize = (norm == 1);

      vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim);
      for (int i = 0; i < num_rows; i++) {
        for (int ii = 0; ii < packed_dim; ii++) {
          fused_embedding_table
              [i * fused_embedding_dim + 2 * sizeof(float16) + ii] =
                  entries(generator);
        }
        float16* scale_bias = reinterpret_cast<float16*>(
            fused_embedding_table.data() + i * fused_embedding_dim);
        float scale = clamp_for_fp16(embedding_distribution(generator));
        float bias = clamp_for_fp16(embedding_distribution(generator));
        FloatToFloat16_ref(&scale, scale_bias, 1, true);
        FloatToFloat16_ref(&bias, scale_bias + 1, 1, true);
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

      if (use_weight) {
        for (auto& w : weights) {
          w = abs(w);
        }
      }

      // Inject pruned indices (idx=-1) for TBE format
      uniform_int_distribution<int> pruned_dist(0, indices.size() - 1);
      for (size_t i = 0; i < indices.size() * 0.1; ++i) {
        auto idx = pruned_dist(generator);
        indices[idx] = -1;
      }

      int output_size = batch_size;
      vector<float16> output_ref_fp16(output_size * embedding_dim);
      vector<float16> output_fp16(output_size * embedding_dim);

      bool success_ref = EmbeddingSpMDMNBit_ref<int64_t, int64_t, float16>(
          bit_rate,
          embedding_dim,
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          normalize,
          output_ref_fp16.data(),
          is_wt_positional,
          true,
          -1,
          -1,
          false,
          false);

      auto kernel =
          GenerateEmbeddingSpMDMNBitWithStrides<int64_t, int64_t, float16>(
              bit_rate,
              embedding_dim,
              use_weight,
              normalize,
              prefetch,
              is_wt_positional,
              true,
              -1,
              -1,
              false,
              false);

      bool success = kernel(
          batch_size,
          lengths_sum,
          num_rows,
          fused_embedding_table.data(),
          indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          output_fp16.data());

      ASSERT_EQ(success, success_ref)
          << "scale_bias_first test: ref and kernel disagree"
          << " bit_rate=" << bit_rate << " dim=" << embedding_dim
          << " norm=" << normalize;

      if (success) {
        for (int i = 0; i < output_size * embedding_dim; ++i) {
          float actual = cpu_half2float(output_fp16[i]);
          float expected = cpu_half2float(output_ref_fp16[i]);
          EXPECT_NEAR(actual, expected, fp16_tolerance(average_len, expected))
              << "scale_bias_first test at i=" << i << " bit_rate=" << bit_rate
              << " dim=" << embedding_dim << " norm=" << normalize
              << " wt=" << use_weight;
        }
      }
    }
  }
}

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, fp16NoBagTest) {
  auto
      [bit_rate,
       prefetch,
       weight_choice,
       corner_case,
       out_type,
       kernel_choice] = GetParam();
  if (!is_sve_fp16_enabled() || out_type != FLOAT16) {
    return;
  }
  if (corner_case != NONE) {
    return;
  }
  ScopedKernelOverride kernel_override(kernel_choice);
  bool use_weight = weight_choice != UNWEIGHTED;
  int num_elem_per_byte = 8 / bit_rate;

  vector<vector<int>> inputs = {
      // {output_size, rows, dim}
      {10, 100, 8},
      {10, 100, 9},
      {10, 100, 16},
      {10, 100, 17},
      {10, 100, 32},
      {10, 100, 33},
      {10, 100, 48},
      {10, 100, 64},
      {10, 100, 65},
      {10, 100, 128},
      {10, 100, 256},
  };

  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<int> entries(0, 16);

  for (auto& input : inputs) {
    int output_size = input[0];
    int num_rows = input[1];
    int embedding_dim = input[2];

    int packed_dim =
        (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
    int fused_embedding_dim = packed_dim + 2 * sizeof(float16);

    for (bool scale_bias_last : {true, false}) {
      vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim);
      for (int i = 0; i < num_rows; i++) {
        int data_offset = scale_bias_last ? 0 : 2 * sizeof(float16);
        for (int ii = 0; ii < packed_dim; ii++) {
          fused_embedding_table[i * fused_embedding_dim + data_offset + ii] =
              entries(generator);
        }
        int sb_offset = scale_bias_last ? packed_dim : 0;
        float16* scale_bias = reinterpret_cast<float16*>(
            fused_embedding_table.data() + i * fused_embedding_dim + sb_offset);
        float scale = 0.25f;
        float bias = 0.5f;
        FloatToFloat16_ref(&scale, scale_bias, 1, true);
        FloatToFloat16_ref(&bias, scale_bias + 1, 1, true);
      }

      uniform_int_distribution<int> idx_dist(0, num_rows - 1);
      vector<int64_t> indices(output_size);
      for (int i = 0; i < output_size; ++i) {
        indices[i] = idx_dist(generator);
      }

      vector<float> weights(output_size);
      for (int i = 0; i < output_size; ++i) {
        weights[i] = use_weight ? 1.0f : 0.0f;
      }

      auto kernel_nobag =
          GenerateEmbeddingSpMDMNBitWithStrides<int64_t, int64_t, float16>(
              bit_rate,
              embedding_dim,
              use_weight,
              false,
              prefetch,
              false,
              true,
              -1,
              -1,
              scale_bias_last,
              false,
              /*no_bag=*/true);

      auto kernel_bag =
          GenerateEmbeddingSpMDMNBitWithStrides<int64_t, int64_t, float16>(
              bit_rate,
              embedding_dim,
              use_weight,
              false,
              prefetch,
              false,
              true,
              -1,
              -1,
              scale_bias_last,
              false,
              /*no_bag=*/false);

      vector<int64_t> offsets(output_size + 1);
      for (int i = 0; i <= output_size; ++i) {
        offsets[i] = i;
      }

      vector<float16> output_nobag(output_size * embedding_dim);
      vector<float16> output_bag(output_size * embedding_dim);

      bool success_nobag = kernel_nobag(
          output_size,
          output_size,
          num_rows,
          fused_embedding_table.data(),
          indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          output_nobag.data());

      bool success_bag = kernel_bag(
          output_size,
          output_size,
          num_rows,
          fused_embedding_table.data(),
          indices.data(),
          offsets.data(),
          use_weight ? weights.data() : nullptr,
          output_bag.data());

      ASSERT_TRUE(success_nobag)
          << "NoBag kernel failed bit_rate=" << bit_rate
          << " dim=" << embedding_dim << " scale_bias_last=" << scale_bias_last;
      ASSERT_TRUE(success_bag)
          << "Bag-of-1 kernel failed bit_rate=" << bit_rate
          << " dim=" << embedding_dim << " scale_bias_last=" << scale_bias_last;

      for (int i = 0; i < output_size * embedding_dim; ++i) {
        float actual = cpu_half2float(output_nobag[i]);
        float expected = cpu_half2float(output_bag[i]);
        EXPECT_EQ(actual, expected)
            << "NoBag vs bag-of-1 mismatch at i=" << i
            << " bit_rate=" << bit_rate << " dim=" << embedding_dim
            << " scale_bias_last=" << scale_bias_last;
      }
    }
  }
}

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, fp16NoBagNegativeIndexTest) {
  auto
      [bit_rate,
       prefetch,
       weight_choice,
       corner_case,
       out_type,
       kernel_choice] = GetParam();
  if (!is_sve_fp16_enabled() || out_type != FLOAT16 || corner_case != NONE ||
      weight_choice != UNWEIGHTED) {
    return;
  }
  ScopedKernelOverride kernel_override(kernel_choice);
  int num_elem_per_byte = 8 / bit_rate;

  int output_size = 10;
  int num_rows = 100;
  int embedding_dim = 32;
  int packed_dim = (embedding_dim + num_elem_per_byte - 1) / num_elem_per_byte;
  int fused_embedding_dim = packed_dim + 2 * sizeof(float16);

  vector<uint8_t> fused_embedding_table(num_rows * fused_embedding_dim, 1);
  for (int i = 0; i < num_rows; i++) {
    float16* scale_bias = reinterpret_cast<float16*>(
        fused_embedding_table.data() + i * fused_embedding_dim + packed_dim);
    float scale = 1.0f, bias = 0.0f;
    FloatToFloat16_ref(&scale, scale_bias, 1, true);
    FloatToFloat16_ref(&bias, scale_bias + 1, 1, true);
  }

  vector<int64_t> indices(output_size, 0);
  indices[5] = -1;
  vector<int64_t> offsets(output_size + 1);
  iota(offsets.begin(), offsets.end(), 0);

  vector<float16> output(output_size * embedding_dim);

  auto kernel =
      GenerateEmbeddingSpMDMNBitWithStrides<int64_t, int64_t, float16>(
          bit_rate,
          embedding_dim,
          false,
          false,
          prefetch,
          false,
          true,
          -1,
          -1,
          true,
          false,
          /*no_bag=*/true);

  bool success = kernel(
      output_size,
      output_size,
      num_rows,
      fused_embedding_table.data(),
      indices.data(),
      offsets.data(),
      nullptr,
      output.data());

  EXPECT_FALSE(success) << "NoBag kernel should return false for negative index"
                        << " bit_rate=" << bit_rate;
}

TEST_P(FusedNBitRowwiseEmbeddingLookupTest, rowwiseSparseTest) {
  vector<vector<int>> inputs(GetInputs_());

  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<> bool_dist(0, 1);

  bool isIndex64b = bool_dist(generator);
  bool isOffset64b = bool_dist(generator);
  bool normalize_by_lengths = bool_dist(generator);
  bool use_offsets = bool_dist(generator);
  bool scale_bias_last = bool_dist(generator);

  int bit_rate = 0, prefetch = 0;
  EmbeddingSpMDMWeightChoice weight_choice{};
  EmbeddingSpMDMCornerCase corner_case{};
  EmbeddingSpMDMDtypeChoice out_type{};
  EmbeddingSpMDMKernelChoice kernel_choice{};
  tie(bit_rate, prefetch, weight_choice, corner_case, out_type, kernel_choice) =
      GetParam();
  ScopedKernelOverride kernel_override(kernel_choice);
  bool is_wt_positional = weight_choice == POSITIONAL_WEIGHTED;
  bool use_weight = weight_choice != UNWEIGHTED;

  if (out_type != FLOAT || !scale_bias_last) {
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
      if (is_sve_fp16_enabled()) {
        scale = clamp_for_fp16(scale);
        bias = clamp_for_fp16(bias);
      }
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

    if (is_sve_fp16_enabled()) {
      for (auto& w : weights) {
        w = abs(w);
      }
    }

    vector<float> output_sls_ref(batch_size * embedding_dim);
    vector<float> output_slws_ref(output_sls_ref.size()),
        output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

    vector<float>& output_ref = use_weight ? output_slws_ref : output_sls_ref;
    vector<float>& output = use_weight ? output_slws : output_sls;
    bool success = false, success_ref = false;

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
