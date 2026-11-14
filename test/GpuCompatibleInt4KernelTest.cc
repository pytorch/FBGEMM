/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "fbgemm/FbgemmEmbedding.h"
#include "src/GpuCompatibleInt4.h" // @manual

namespace fbgemm {
namespace {

std::array<uint8_t, 32> cancellationTable() {
  std::array<uint8_t, 32> table{};
  const uint16_t scale_bias0[]{0x2e66, 0};
  const uint16_t scale_bias1[]{0x2e66, 0xb4cc};
  std::memcpy(table.data(), scale_bias0, sizeof(scale_bias0));
  std::memcpy(table.data() + 16, scale_bias1, sizeof(scale_bias1));
  std::fill(table.begin() + 4, table.begin() + 8, 0x33);
  return table;
}

TEST(GpuCompatibleInt4Kernel, MatchesGpuRowsBeforeAccumulating) {
  const auto table = cancellationTable();
  const int32_t indices[]{0, 1};
  const int32_t offsets[]{0, 2};
  std::array<float16, 8> output;
  const auto gpu =
      GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t, int32_t, float16>(
          8, false, true, 8, 16, false);
  ASSERT_TRUE(gpu(1, 2, 2, table.data(), indices, offsets, output.data()));
  for (const auto value : output) {
    EXPECT_EQ(value.val, 0);
  }
  const auto cpu =
      GenerateEmbeddingSpMDMNBitWithStrides<false, int32_t, int32_t, float16>(
          4, 8, false, 0, false, true, 8, 16, false);
  ASSERT_TRUE(cpu(1, 2, 2, table.data(), indices, offsets, output.data()));
  for (const auto value : output) {
    EXPECT_EQ(value.val, 0x0800);
  }
}

template <typename I, typename O>
void checkLayoutsAndBounds() {
  const auto table = cancellationTable();
  const I indices[]{0, 1, -1, 0};
  const O offsets[]{7, 9, 9, 11};
  std::array<float16, 30> output;
  output.fill(float16{0x7bff});
  const auto kernel = GenerateEmbeddingSpMDMInt4WithGpuRounding<I, O, float16>(
      7, false, true, 10, 16, false);
  ASSERT_TRUE(kernel(3, 4, 2, table.data(), indices, offsets, output.data()));
  for (int bag = 0; bag < 3; ++bag) {
    for (int d = 0; d < 7; ++d) {
      EXPECT_EQ(output[bag * 10 + d].val, bag == 2 ? 0x34cc : 0);
    }
    for (int d = 7; d < 10; ++d) {
      EXPECT_EQ(output[bag * 10 + d].val, 0x7bff);
    }
  }
  const O lengths[]{2, 0, 2};
  const auto by_length =
      GenerateEmbeddingSpMDMInt4WithGpuRounding<I, O, float16>(
          7, false, false, 10, 16, false);
  ASSERT_TRUE(
      by_length(3, 4, 2, table.data(), indices, lengths, output.data()));
  const I invalid[]{2};
  const O one[]{0, 1};
  EXPECT_FALSE(kernel(1, 1, 2, table.data(), invalid, one, output.data()));
  const O decreasing[]{2, 1};
  EXPECT_FALSE(
      kernel(1, 1, 2, table.data(), indices, decreasing, output.data()));
  EXPECT_FALSE(kernel(1, 0, 2, table.data(), indices, one, output.data()));
  const O extreme[]{
      std::numeric_limits<O>::min(), std::numeric_limits<O>::max()};
  EXPECT_FALSE(kernel(1, 0, 2, table.data(), indices, extreme, output.data()));
}

TEST(GpuCompatibleInt4Kernel, IndexAndOffsetWidths) {
  checkLayoutsAndBounds<int32_t, int32_t>();
  checkLayoutsAndBounds<int32_t, int64_t>();
  checkLayoutsAndBounds<int64_t, int32_t>();
  checkLayoutsAndBounds<int64_t, int64_t>();
}

TEST(GpuCompatibleInt4Kernel, PrunedIndexMatchesLegacyLayoutPolicy) {
  const auto table = cancellationTable();
  const int64_t indices[]{-1};
  const int64_t offsets[]{0, 1};
  for (bool scaleBiasLast : {false, true}) {
    std::array<float16, 8> gpuOutput{};
    std::array<float16, 8> legacyOutput{};
    const auto gpu =
        GenerateEmbeddingSpMDMInt4WithGpuRounding<int64_t, int64_t, float16>(
            8, false, true, 8, 16, scaleBiasLast);
    const auto legacy =
        GenerateEmbeddingSpMDMNBitWithStrides<false, int64_t, int64_t, float16>(
            4, 8, false, 0, false, true, 8, 16, scaleBiasLast);
    const bool gpuValid =
        gpu(1, 1, 2, table.data(), indices, offsets, gpuOutput.data());
    const bool legacyValid =
        legacy(1, 1, 2, table.data(), indices, offsets, legacyOutput.data());
    EXPECT_EQ(gpuValid, legacyValid);
    EXPECT_EQ(gpuValid, !scaleBiasLast);
    if (gpuValid) {
      for (int d = 0; d < 8; ++d) {
        EXPECT_EQ(gpuOutput[d].val, legacyOutput[d].val);
      }
    }
  }
}

TEST(GpuCompatibleInt4Kernel, MeanAndTailScaleBias) {
  const auto front = cancellationTable();
  auto tail = front;
  for (int row = 0; row < 2; ++row) {
    std::memcpy(tail.data() + row * 16, front.data() + row * 16 + 4, 4);
    std::memcpy(tail.data() + row * 16 + 4, front.data() + row * 16, 4);
  }
  const int64_t indices[]{0, 0, 0};
  const int64_t lengths[]{3};
  std::array<float, 8> output{};
  const auto kernel =
      GenerateEmbeddingSpMDMInt4WithGpuRounding<int64_t, int64_t, float>(
          8, true, false, 8, 16, true);
  ASSERT_TRUE(kernel(1, 3, 2, tail.data(), indices, lengths, output.data()));
  for (const auto value : output) {
    EXPECT_EQ(value, detail::halfToFloat(0x34cc));
  }
}

TEST(GpuCompatibleInt4Kernel, RejectsInvalidConfiguration) {
  EXPECT_THROW(
      (GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t>(
          0, false, true, -1, -1, false)),
      std::invalid_argument);
  EXPECT_THROW(
      (GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t>(
          8, false, true, 7, 16, false)),
      std::invalid_argument);
  EXPECT_THROW(
      (GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t>(
          8, false, true, 8, 7, false)),
      std::invalid_argument);
  EXPECT_THROW(
      (GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t>(
          8, false, true, std::numeric_limits<int64_t>::max(), 16, false)),
      std::invalid_argument);
}

} // namespace
} // namespace fbgemm
