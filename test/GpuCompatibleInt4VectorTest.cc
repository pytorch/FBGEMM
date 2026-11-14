/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include "fbgemm/FbgemmEmbedding.h"
#include "fbgemm/Utils.h"
#include "src/GpuCompatibleInt4.h" // @manual

namespace fbgemm {
namespace {

uint32_t floatBits(float value) {
  uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename OutType>
void checkOrderedNibbles(
    int64_t dimension,
    bool scaleBiasLast,
    bool normalize,
    bool overflow) {
  const int64_t packedBytes = (dimension + 1) / 2;
  const int64_t inputStride = packedBytes + 7;
  const int64_t outputStride = dimension + 5;
  const std::array<std::array<uint16_t, 2>, 4> parameters = overflow
      ? std::array<
            std::array<uint16_t, 2>,
            4>{{{0x5800, 0}, {0xd800, 0}, {0x6800, 0}, {0x57ff, 0}}}
      : std::array<std::array<uint16_t, 2>, 4>{
            {{0x2363, 0xae96}, {0x302a, 0xbc3b}, {0x3d55, 0x8001}, {0, 0}}};
  std::vector<uint8_t> table(4 * inputStride, 0);
  for (int row = 0; row < 4; ++row) {
    auto* start = table.data() + row * inputStride;
    std::memcpy(
        start + (scaleBiasLast ? packedBytes : 0), parameters[row].data(), 4);
    auto* packed = start + (scaleBiasLast ? 0 : 4);
    for (int64_t byte = 0; byte < packedBytes; ++byte) {
      const uint8_t low = (byte * 7 + row * 3) & 15;
      const uint8_t high = (byte * 11 + row * 5 + 1) & 15;
      packed[byte] = low | (high << 4);
    }
  }
  const std::vector<int32_t> indices = overflow
      ? std::vector<int32_t>{0, 1, 2, 3}
      : std::vector<int32_t>{2, 0, 1, 2, 1, 0, 2};
  const std::vector<int32_t> lengths = overflow
      ? std::vector<int32_t>{1, 1, 1, 1}
      : std::vector<int32_t>{3, 0, 4};
  const OutType sentinel = [] {
    if constexpr (std::is_same_v<OutType, float>) {
      return 123.0f;
    } else {
      return float16{0x7bff};
    }
  }();
  std::vector<OutType> output(lengths.size() * outputStride, sentinel);
  const auto kernel =
      GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t, int32_t, OutType>(
          dimension,
          normalize,
          false,
          outputStride,
          inputStride,
          scaleBiasLast);
  ASSERT_TRUE(kernel(
      lengths.size(),
      indices.size(),
      4,
      table.data(),
      indices.data(),
      lengths.data(),
      output.data()));
  size_t position = 0;
  for (size_t bag = 0; bag < lengths.size(); ++bag) {
    std::vector<float> expected(dimension, 0.0f);
    for (int n = 0; n < lengths[bag]; ++n) {
      const int row = indices[position++];
      const auto lut = detail::makeGpuCompatibleInt4Lut(
          parameters[row][0], parameters[row][1]);
      const uint8_t* packed =
          table.data() + row * inputStride + (scaleBiasLast ? 0 : 4);
      for (int64_t d = 0; d < dimension; ++d) {
        const uint8_t q = (d & 1) ? packed[d / 2] >> 4 : packed[d / 2] & 15;
        expected[d] += lut[q + ((d & 1) ? 16 : 0)];
      }
    }
    const float reciprocal = normalize && lengths[bag]
        ? 1.0f / static_cast<float>(lengths[bag])
        : 1.0f;
    for (int64_t d = 0; d < outputStride; ++d) {
      SCOPED_TRACE(bag);
      SCOPED_TRACE(d);
      const auto actual = output[bag * outputStride + d];
      if (d >= dimension) {
        if constexpr (std::is_same_v<OutType, float>) {
          EXPECT_EQ(actual, sentinel);
        } else {
          EXPECT_EQ(actual.val, sentinel.val);
        }
      } else {
        const float value = expected[d] * reciprocal;
        if constexpr (std::is_same_v<OutType, float>) {
          if (std::isnan(value)) {
            EXPECT_TRUE(std::isnan(actual));
          } else {
            EXPECT_EQ(floatBits(actual), floatBits(value));
          }
        } else {
          if (std::isnan(value)) {
            EXPECT_TRUE(std::isnan(detail::halfToFloat(actual.val)));
          } else {
            EXPECT_EQ(actual.val, detail::doubleToHalfRne(value));
          }
        }
      }
    }
  }
}

void checkVectorLayouts() {
  for (int64_t dimension : {16, 31, 32, 33}) {
    SCOPED_TRACE(dimension);
    for (bool tail : {false, true}) {
      for (bool normalize : {false, true}) {
        checkOrderedNibbles<float>(dimension, tail, normalize, false);
        checkOrderedNibbles<float16>(dimension, tail, normalize, false);
      }
      checkOrderedNibbles<float>(dimension, tail, false, true);
      checkOrderedNibbles<float16>(dimension, tail, false, true);
    }
  }
}

void checkLutProofBoundaries() {
  constexpr int64_t dimension = 33;
  constexpr int64_t stride = (dimension + 1) / 2 + 4;
  const std::vector<std::array<uint16_t, 2>> parameters{
      {0x3d55, 0x8001}, {0x2363, 0xae96}, {0x302a, 0xbc3b}, {0x0001, 0x0001},
      {0x8001, 0x8001}, {0x03ff, 0x0400}, {0x0400, 0x83ff}, {0x0000, 0x0000},
      {0x0000, 0x8000}, {0x8000, 0x0000}, {0x8000, 0x8000}, {0xbc00, 0x3c00},
      {0x3c00, 0xbc00}, {0x57ff, 0x7bff}, {0x57ff, 0xfbff}, {0x0001, 0x7bff},
      {0x4c00, 0x7bf8}, {0x5800, 0x0000}, {0xd800, 0x8000}, {0x6800, 0x0000},
      {0x3c00, 0x7c00}, {0x3c00, 0xfc00}, {0x3c00, 0x7e00}, {0x7e00, 0x3c00}};
  std::vector<uint8_t> table(parameters.size() * stride, 0);
  std::vector<int32_t> indices(parameters.size());
  std::vector<int32_t> lengths(parameters.size(), 1);
  for (size_t row = 0; row < parameters.size(); ++row) {
    auto* start = table.data() + row * stride;
    std::memcpy(start, parameters[row].data(), 4);
    for (int64_t byte = 0; byte < (dimension + 1) / 2; ++byte) {
      start[4 + byte] = (byte & 15) * 17;
    }
    indices[row] = row;
  }
  std::vector<float> floats(parameters.size() * dimension);
  std::vector<float16> halves(parameters.size() * dimension);
  struct RestoreRounding {
    const int original = std::fegetround();
    ~RestoreRounding() {
      std::fesetround(original);
    }
  } restore;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    SCOPED_TRACE(mode);
    ASSERT_EQ(std::fesetround(mode), 0);
    const auto floatKernel =
        GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t, int32_t, float>(
            dimension, false, false, dimension, stride, false);
    const auto halfKernel =
        GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t, int32_t, float16>(
            dimension, false, false, dimension, stride, false);
    ASSERT_TRUE(floatKernel(
        parameters.size(),
        indices.size(),
        parameters.size(),
        table.data(),
        indices.data(),
        lengths.data(),
        floats.data()));
    ASSERT_TRUE(halfKernel(
        parameters.size(),
        indices.size(),
        parameters.size(),
        table.data(),
        indices.data(),
        lengths.data(),
        halves.data()));
    for (size_t row = 0; row < parameters.size(); ++row) {
      const auto lut = detail::makeGpuCompatibleInt4Lut(
          parameters[row][0], parameters[row][1]);
      for (int64_t d = 0; d < dimension; ++d) {
        SCOPED_TRACE(row);
        SCOPED_TRACE(d);
        volatile float zero = 0.0f;
        const float expected = zero + lut[(d / 2) % 16 + ((d & 1) ? 16 : 0)];
        const size_t index = row * dimension + d;
        if (std::isnan(expected)) {
          EXPECT_TRUE(std::isnan(floats[index]));
          EXPECT_TRUE(std::isnan(detail::halfToFloat(halves[index].val)));
        } else {
          EXPECT_EQ(floatBits(floats[index]), floatBits(expected));
          EXPECT_EQ(halves[index].val, detail::doubleToHalfRne(expected));
        }
      }
    }
    EXPECT_EQ(floatBits(floats[6]), floatBits(detail::halfToFloat(0x43ff)));
    EXPECT_EQ(halves[6].val, 0x43ff);
  }
}

struct RestoreIsa {
  const inst_set_t original = fbgemmInstructionSet();
  explicit RestoreIsa(inst_set_t requested) {
    fbgemmForceIsa(requested);
  }
  ~RestoreIsa() {
    fbgemmForceIsa(original);
  }
};

TEST(GpuCompatibleInt4Kernel, NativeDispatchOrderedNibblesAndTails) {
  checkVectorLayouts();
}

TEST(GpuCompatibleInt4Kernel, BoundedLutMatchesScalarAcrossRoundingModes) {
  checkLutProofBoundaries();
}

TEST(GpuCompatibleInt4Kernel, LutOverflowUsesScalarFallback) {
#if (CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64) && \
    (defined(__clang__) || defined(__GNUC__))
  if (!cpuinfo_initialize() || !fbgemmHasAvx2Support() ||
      !cpuinfo_has_x86_f16c()) {
    GTEST_SKIP() << "AVX2/F16C unavailable";
  }
  RestoreIsa restoreIsa(inst_set_t::avx2);
  if (fbgemmInstructionSet() != inst_set_t::avx2) {
    GTEST_SKIP() << "AVX2 cannot be forced on this host";
  }
  struct RestoreEnvironment {
    std::fenv_t original;
    RestoreEnvironment() {
      std::fegetenv(&original);
    }
    ~RestoreEnvironment() {
      std::fesetenv(&original);
    }
  } restoreEnvironment;
  constexpr int64_t dimension = 33;
  std::array<uint8_t, (dimension + 1) / 2 + 4> table;
  const uint16_t scaleBias[]{0x4c00, 0x7bf8};
  std::memcpy(table.data(), scaleBias, sizeof(scaleBias));
  std::fill(table.begin() + 4, table.end(), 0xff);
  const int32_t indices[]{0};
  const int32_t lengths[]{1};
  std::array<float, dimension> output;
  const auto kernel =
      GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t, int32_t, float>(
          dimension, false, false, dimension, table.size(), false);
  ASSERT_EQ(std::feclearexcept(FE_OVERFLOW), 0);
  ASSERT_TRUE(kernel(1, 1, 1, table.data(), indices, lengths, output.data()));
  EXPECT_EQ(std::fetestexcept(FE_OVERFLOW), 0);
  for (float value : output) {
    EXPECT_EQ(value, std::numeric_limits<float>::infinity());
  }
#else
  GTEST_SKIP() << "Native x86 SIMD helpers unavailable";
#endif
}

TEST(GpuCompatibleInt4Kernel, Avx2DispatchOrderedNibblesAndTails) {
#if (CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64) && \
    (defined(__clang__) || defined(__GNUC__))
  if (!cpuinfo_initialize() || !fbgemmHasAvx2Support()) {
    GTEST_SKIP() << "AVX2 unavailable";
  }
  RestoreIsa restore(inst_set_t::avx2);
  if (fbgemmInstructionSet() != inst_set_t::avx2) {
    GTEST_SKIP() << "AVX2 cannot be forced on this host";
  }
  checkVectorLayouts();
  checkLutProofBoundaries();
#else
  GTEST_SKIP() << "Native x86 SIMD helpers unavailable";
#endif
}

TEST(GpuCompatibleInt4Kernel, Avx512DispatchOrderedNibblesAndTails) {
#if (CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64) && \
    (defined(__clang__) || defined(__GNUC__))
  if (!cpuinfo_initialize() || !fbgemmHasAvx2Support() ||
      !fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 unavailable";
  }
  RestoreIsa restore(inst_set_t::avx512);
  if (fbgemmInstructionSet() != inst_set_t::avx512) {
    GTEST_SKIP() << "AVX512 cannot be forced on this host";
  }
  checkVectorLayouts();
  checkLutProofBoundaries();
#else
  GTEST_SKIP() << "Native x86 SIMD helpers unavailable";
#endif
}

TEST(GpuCompatibleInt4Kernel, HalfOutputUsesFixedNearestEven) {
  constexpr int64_t dimension = 33;
  constexpr int64_t stride = (dimension + 1) / 2 + 4;
  const uint16_t biases[]{0x3c00, 0x3c01, 0x1000, 0x0001, 0x8001, 0};
  std::array<uint8_t, stride * 6> table{};
  for (int row = 0; row < 6; ++row) {
    const uint16_t scaleBias[]{0, biases[row]};
    std::memcpy(table.data() + row * stride, scaleBias, sizeof(scaleBias));
  }
  const int32_t ties[]{0, 2, 1, 2};
  const int32_t underflows[]{3, 5, 4, 5};
  const int32_t lengths[]{2, 2};
  std::array<float16, dimension * 2> output;
  struct RestoreRounding {
    const int original = std::fegetround();
    ~RestoreRounding() {
      std::fesetround(original);
    }
  } restore;
  for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    const auto sum =
        GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t, int32_t, float16>(
            dimension, false, false, dimension, stride, false);
    ASSERT_TRUE(sum(2, 4, 6, table.data(), ties, lengths, output.data()));
    for (int64_t d = 0; d < dimension; ++d) {
      EXPECT_EQ(output[d].val, 0x3c00);
      EXPECT_EQ(output[dimension + d].val, 0x3c02);
    }
    const auto mean =
        GenerateEmbeddingSpMDMInt4WithGpuRounding<int32_t, int32_t, float16>(
            dimension, true, false, dimension, stride, false);
    ASSERT_TRUE(
        mean(2, 4, 6, table.data(), underflows, lengths, output.data()));
    for (int64_t d = 0; d < dimension; ++d) {
      EXPECT_EQ(output[d].val, 0x0000);
      EXPECT_EQ(output[dimension + d].val, 0x8000);
    }
  }
}

} // namespace
} // namespace fbgemm
