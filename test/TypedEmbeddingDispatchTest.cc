/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "fbgemm/Fbgemm.h"
#include "fbgemm/FbgemmConvert.h"
#include "src/RefImplementations.h" // @manual

namespace fbgemm {
namespace {

template <typename InType>
void verifyBlockSizeOneDispatch(bool has_weight) {
  constexpr int64_t kRows = 3;
  const std::vector<InType> input{
      from_float<InType>(1.5f),
      from_float<InType>(-2.0f),
      from_float<InType>(4.0f)};
  const std::vector<int32_t> indices{2, 0, 1};
  const std::vector<int32_t> offsets{0, 3};
  const std::vector<float> weights{0.5f, 2.0f, -1.0f};
  std::vector<float> expected(1);
  ASSERT_TRUE((EmbeddingSpMDM_ref<InType, int32_t, int32_t, float>(
      1,
      1,
      indices.size(),
      kRows,
      input.data(),
      indices.data(),
      offsets.data(),
      has_weight ? weights.data() : nullptr,
      false,
      expected.data())));

  auto kernel = GenerateEmbeddingSpMDM<InType, int32_t, int32_t, float>(
      1, has_weight, false, 16, false, true);
  std::vector<float> actual(1);
  ASSERT_TRUE(kernel(
      1,
      indices.size(),
      kRows,
      input.data(),
      indices.data(),
      offsets.data(),
      has_weight ? weights.data() : nullptr,
      actual.data()));
  EXPECT_FLOAT_EQ(actual[0], expected[0]);
}

template <typename InType, typename OutType>
void verifyAvx2RemainderDispatch() {
  constexpr int64_t kBlockSize = 3;
  constexpr int64_t kRows = 2;
  const std::vector<InType> input{
      from_float<InType>(1.0f),
      from_float<InType>(2.0f),
      from_float<InType>(3.0f),
      from_float<InType>(4.0f),
      from_float<InType>(5.0f),
      from_float<InType>(6.0f)};
  const std::vector<int32_t> indices{0, 1};
  const std::vector<int32_t> offsets{0, 2};
  std::vector<OutType> expected(kBlockSize);
  ASSERT_TRUE((EmbeddingSpMDM_ref<InType, int32_t, int32_t, OutType>(
      kBlockSize,
      1,
      indices.size(),
      kRows,
      input.data(),
      indices.data(),
      offsets.data(),
      nullptr,
      false,
      expected.data())));

  auto kernel = GenerateEmbeddingSpMDM<InType, int32_t, int32_t, OutType>(
      kBlockSize, false, false, 16, false, true);
  std::vector<OutType> actual(kBlockSize);
  ASSERT_TRUE(kernel(
      1,
      indices.size(),
      kRows,
      input.data(),
      indices.data(),
      offsets.data(),
      nullptr,
      actual.data()));
  EXPECT_EQ(actual, expected);
}

TEST(TypedEmbeddingDispatchTest, BlockSizeOneHalfInputs) {
  for (const bool has_weight : {false, true}) {
    verifyBlockSizeOneDispatch<float16>(has_weight);
    verifyBlockSizeOneDispatch<bfloat16>(has_weight);
  }
}

TEST(TypedEmbeddingDispatchTest, Avx2HalfRemainders) {
  verifyAvx2RemainderDispatch<float16, float16>();
  verifyAvx2RemainderDispatch<float16, bfloat16>();
  verifyAvx2RemainderDispatch<bfloat16, float16>();
  verifyAvx2RemainderDispatch<bfloat16, bfloat16>();
}

} // namespace
} // namespace fbgemm
