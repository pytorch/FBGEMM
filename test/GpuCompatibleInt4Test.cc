/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include <gtest/gtest.h>

#include "src/GpuCompatibleInt4.h" // @manual

namespace fbgemm {
namespace {

uint32_t floatBits(float value) {
  uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

double referenceHalfValue(uint16_t bits) {
  const int exponent = (bits >> 10) & 0x1f;
  const int fraction = bits & 0x3ff;
  double magnitude{};
  if (exponent == 31) {
    magnitude = fraction ? std::numeric_limits<double>::quiet_NaN()
                         : std::numeric_limits<double>::infinity();
  } else if (exponent == 0) {
    magnitude = std::ldexp(static_cast<double>(fraction), -24);
  } else {
    magnitude = std::ldexp(static_cast<double>(1024 + fraction), exponent - 25);
  }
  return std::copysign(magnitude, bits & 0x8000 ? -1.0 : 1.0);
}

// Find the nearest representable value without inspecting binary64 bits.
uint16_t referenceRoundHalf(double value) {
  const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  const double magnitude = std::abs(value);
  if (std::isnan(value)) {
    return sign | 0x7e00;
  }
  if (magnitude >= 65520.0) {
    return sign | 0x7c00;
  }
  uint16_t lower = 0;
  uint16_t upper = 0x7bff;
  while (lower < upper) {
    const uint16_t middle = lower + (upper - lower + 1) / 2;
    if (referenceHalfValue(middle) <= magnitude) {
      lower = middle;
    } else {
      upper = middle - 1;
    }
  }
  if (lower == 0x7bff) {
    return sign | lower;
  }
  const double midpoint =
      (referenceHalfValue(lower) + referenceHalfValue(lower + 1)) / 2.0;
  if (magnitude > midpoint || (magnitude == midpoint && (lower & 1))) {
    ++lower;
  }
  return sign | lower;
}

uint16_t referenceGpuInt4(uint16_t scale, uint16_t bias, int q, bool high) {
  const uint16_t packedHalf = high ? q << 4 : q;
  const uint16_t operand =
      referenceRoundHalf(referenceHalfValue(packedHalf) * 32768.0);
  const uint16_t coefficient =
      referenceRoundHalf(referenceHalfValue(scale) * (high ? 32.0 : 512.0));
  return referenceRoundHalf(
      std::fma(
          referenceHalfValue(operand),
          referenceHalfValue(coefficient),
          referenceHalfValue(bias)));
}

void expectHalf(float actual, uint16_t expected) {
  if ((expected & 0x7fff) > 0x7c00) {
    EXPECT_TRUE(std::isnan(actual));
  } else {
    EXPECT_EQ(
        floatBits(actual),
        floatBits(static_cast<float>(referenceHalfValue(expected))));
  }
}

TEST(GpuCompatibleInt4Test, HalfRoundTripAndValues) {
  for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
    SCOPED_TRACE(bits);
    const float value = detail::halfToFloat(bits);
    const uint16_t rounded = detail::doubleToHalfRne(value);
    if ((bits & 0x7fff) > 0x7c00) {
      EXPECT_TRUE(std::isnan(value));
      EXPECT_GT(rounded & 0x7fff, 0x7c00);
    } else {
      expectHalf(value, bits);
      EXPECT_EQ(rounded, bits);
    }
  }
}

TEST(GpuCompatibleInt4Test, DoubleConversionEdges) {
  const double infinity = std::numeric_limits<double>::infinity();
  const double underflowTie = std::ldexp(1.0, -25);
  const double normalTie = std::ldexp(1023.5, -24);
  const std::array<std::pair<double, uint16_t>, 18> cases{{
      {0.0, 0x0000},
      {-0.0, 0x8000},
      {std::numeric_limits<double>::denorm_min(), 0x0000},
      {-std::numeric_limits<double>::denorm_min(), 0x8000},
      {underflowTie, 0x0000},
      {-underflowTie, 0x8000},
      {std::nextafter(underflowTie, infinity), 0x0001},
      {-std::nextafter(underflowTie, infinity), 0x8001},
      {std::nextafter(underflowTie, 0.0), 0x0000},
      {std::ldexp(1.0, -24), 0x0001},
      {normalTie, 0x0400},
      {std::nextafter(normalTie, 0.0), 0x03ff},
      {65504.0, 0x7bff},
      {std::nextafter(65520.0, 0.0), 0x7bff},
      {65520.0, 0x7c00},
      {-65520.0, 0xfc00},
      {infinity, 0x7c00},
      {-infinity, 0xfc00},
  }};
  for (const auto& [value, expected] : cases) {
    EXPECT_EQ(detail::doubleToHalfRne(value), expected) << value;
  }
  EXPECT_GT(
      detail::doubleToHalfRne(std::numeric_limits<double>::quiet_NaN()) &
          0x7fff,
      0x7c00);
}

TEST(GpuCompatibleInt4Test, EveryFiniteHalfMidpoint) {
  const double infinity = std::numeric_limits<double>::infinity();
  for (uint16_t lower = 0; lower < 0x7bff; ++lower) {
    SCOPED_TRACE(lower);
    const double midpoint =
        (referenceHalfValue(lower) + referenceHalfValue(lower + 1)) / 2.0;
    const uint16_t tie = (lower & 1) ? lower + 1 : lower;
    EXPECT_EQ(detail::doubleToHalfRne(midpoint), tie);
    EXPECT_EQ(detail::doubleToHalfRne(-midpoint), tie | 0x8000);
    EXPECT_EQ(detail::doubleToHalfRne(std::nextafter(midpoint, 0.0)), lower);
    EXPECT_EQ(
        detail::doubleToHalfRne(std::nextafter(midpoint, infinity)), lower + 1);
  }
}

TEST(GpuCompatibleInt4Test, AvoidsFloatDoubleRounding) {
  const auto lut = detail::makeGpuCompatibleInt4Lut(0x3d55, 0x8001);
  expectHalf(lut[3], 0x43ff);
  expectHalf(lut[16 + 3], 0x43ff);
  const double exact =
      3.0 * referenceHalfValue(0x3d55) + referenceHalfValue(0x8001);
  EXPECT_EQ(detail::doubleToHalfRne(exact), 0x43ff);
  EXPECT_EQ(detail::doubleToHalfRne(static_cast<float>(exact)), 0x4400);
}

TEST(GpuCompatibleInt4Test, SignedZerosAndCoefficientOverflow) {
  const auto negativeZeros = detail::makeGpuCompatibleInt4Lut(0x8000, 0x8000);
  const auto mixedZeros = detail::makeGpuCompatibleInt4Lut(0x0000, 0x8000);
  for (int i = 0; i < 32; ++i) {
    expectHalf(negativeZeros[i], 0x8000);
    expectHalf(mixedZeros[i], 0x0000);
  }
  const auto cancellation = detail::makeGpuCompatibleInt4Lut(0x3c00, 0xbc00);
  expectHalf(cancellation[1], 0x0000);
  expectHalf(cancellation[17], 0x0000);
  const auto negative = detail::makeGpuCompatibleInt4Lut(0xbc00, 0x8000);
  expectHalf(negative[0], 0x8000);
  expectHalf(negative[16], 0x8000);

  const auto lowOverflow = detail::makeGpuCompatibleInt4Lut(0x5800, 0x0000);
  EXPECT_TRUE(std::isnan(lowOverflow[0]));
  for (int q = 1; q < 16; ++q) {
    expectHalf(lowOverflow[q], 0x7c00);
  }
  expectHalf(lowOverflow[16], 0x0000);
  expectHalf(lowOverflow[17], 0x5800);
  expectHalf(lowOverflow[31], 0x6780);
  const auto bothOverflow = detail::makeGpuCompatibleInt4Lut(0x6800, 0xfc00);
  for (float value : bothOverflow) {
    EXPECT_TRUE(std::isnan(value));
  }
}

TEST(GpuCompatibleInt4Test, MatchesLiteralGpuHalfStages) {
  const std::array<uint16_t, 18> scales{
      0x0000,
      0x8000,
      0x0001,
      0x8001,
      0x03ff,
      0x0400,
      0x3c00,
      0xbc00,
      0x3d55,
      0x57ff,
      0x5800,
      0x67ff,
      0x6800,
      0x7bff,
      0xfbff,
      0x7c00,
      0xfc00,
      0x7e00};
  const std::array<uint16_t, 13> biases{
      0x0000,
      0x8000,
      0x0001,
      0x8001,
      0x3555,
      0xb555,
      0x3c00,
      0xbc00,
      0x7bff,
      0xfbff,
      0x7c00,
      0xfc00,
      0x7e00};
  for (uint16_t scale : scales) {
    for (uint16_t bias : biases) {
      SCOPED_TRACE(scale);
      SCOPED_TRACE(bias);
      const auto lut = detail::makeGpuCompatibleInt4Lut(scale, bias);
      for (int high = 0; high < 2; ++high) {
        for (int q = 0; q < 16; ++q) {
          SCOPED_TRACE(high * 16 + q);
          expectHalf(
              lut[high * 16 + q], referenceGpuInt4(scale, bias, q, high));
        }
      }
    }
  }
}

TEST(GpuCompatibleInt4Test, FiniteCoefficientsShareBothNibbles) {
  const std::array<uint16_t, 12> scales{
      0x0000,
      0x8000,
      0x0001,
      0x8001,
      0x03ff,
      0x0400,
      0x3c00,
      0xbc00,
      0x3d55,
      0xbd55,
      0x57ff,
      0xd7ff};
  const std::array<uint16_t, 11> biases{
      0x0000,
      0x8000,
      0x0001,
      0x8001,
      0x3c00,
      0xbc00,
      0x7bff,
      0xfbff,
      0x7c00,
      0xfc00,
      0x7e00};
  for (uint16_t scale : scales) {
    for (uint16_t bias : biases) {
      SCOPED_TRACE(scale);
      SCOPED_TRACE(bias);
      const auto lut = detail::makeGpuCompatibleInt4Lut(scale, bias);
      for (int q = 0; q < 16; ++q) {
        SCOPED_TRACE(q);
        EXPECT_EQ(floatBits(lut[q]), floatBits(lut[16 + q]));
        expectHalf(lut[q], referenceGpuInt4(scale, bias, q, false));
        expectHalf(lut[16 + q], referenceGpuInt4(scale, bias, q, true));
      }
    }
  }
}

TEST(GpuCompatibleInt4Test, RecordedRowQuantizationParameters) {
  struct Row {
    uint16_t scale;
    uint16_t bias;
    std::array<uint16_t, 16> values;
  };
  const std::array<Row, 2> rows{{
      {0x2363,
       0xae96,
       {0xae96,
        0xadaa,
        0xacbd,
        0xaba2,
        0xa9c9,
        0xa7e0,
        0xa42f,
        0x97d8,
        0x2268,
        0x26e6,
        0x294c,
        0x2b24,
        0x2c7e,
        0x2d6b,
        0x2e57,
        0x2f44}},
      {0x302a,
       0xbc3b,
       {0xbc3b,
        0xbb6c,
        0xba61,
        0xb956,
        0xb84c,
        0xb683,
        0xb46e,
        0xb0b2,
        0xa440,
        0x2f44,
        0x33cc,
        0x35fb,
        0x3808,
        0x3912,
        0x3a1d,
        0x3b28}},
  }};
  for (const auto& row : rows) {
    const auto lut = detail::makeGpuCompatibleInt4Lut(row.scale, row.bias);
    for (int q = 0; q < 16; ++q) {
      expectHalf(lut[q], row.values[q]);
      expectHalf(lut[16 + q], row.values[q]);
    }
  }
}

TEST(GpuCompatibleInt4Test, IndependentOfHostRoundingMode) {
  struct RestoreRounding {
    int original = std::fegetround();
    ~RestoreRounding() {
      std::fesetround(original);
    }
  } restore;
  for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    ASSERT_EQ(std::fesetround(mode), 0);
    EXPECT_EQ(detail::doubleToHalfRne(1.00048828125), 0x3c00);
    const auto lut = detail::makeGpuCompatibleInt4Lut(0x3d55, 0x8001);
    expectHalf(lut[3], 0x43ff);
    expectHalf(lut[19], 0x43ff);
    const auto cancellation = detail::makeGpuCompatibleInt4Lut(0x3c00, 0xbc00);
    expectHalf(cancellation[1], 0x0000);
    expectHalf(cancellation[17], 0x0000);
  }
}

} // namespace
} // namespace fbgemm
