/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include "test/PlatformSpecificKernelsTests/Float16ConvertTestUtils.h" // @manual
#include "test/PlatformSpecificKernelsTests/PlatformSpecificKernelsTestUtils.h" // @manual

namespace fbgemm {
namespace {

std::vector<float16> allFloat16() {
  std::vector<float16> res(size_t{1} << 16);
  for (size_t i = 0; i != res.size(); ++i) {
    res[i].val = static_cast<uint16_t>(i);
  }
  return res;
}

std::vector<float> floatsAroundEveryFloat16() {
  constexpr std::array<uint32_t, 6> lowBits{
      0x0000, 0x0001, 0x0fff, 0x1000, 0x1001, 0x1fff};
  std::vector<float> res;
  for (uint32_t high = 0; high != (uint32_t{1} << 19); ++high) {
    for (uint32_t low : lowBits) {
      res.push_back(std::bit_cast<float>(high << 13 | low));
    }
  }
  return res;
}

TEST(Float16ConvertKernelsTest, Smoke) {
  const Float16ConvertTestUtils utils;

  const std::vector<float16> f16In{
      {0x0000}, {0x3c00}, {0xbc00}, {0x4240}, {0x7c00}};
  const std::vector<float> expectedF32{
      0.0f, 1.0f, -1.0f, 3.125f, std::numeric_limits<float>::infinity()};
  std::vector<float> f32Out(f16In.size());
  ASSERT_NO_THROW(utils.f16ToF32(f16In, f32Out));
  EXPECT_EQ(f32Out, expectedF32);

  const std::vector<float> f32In{
      1.0f, -2.5f, std::numbers::pi_v<float>, 1.0009765625f, 65520.f};
  std::vector<float16> f16Out(f32In.size());

  const std::vector<float16> expectedF16{
      {0x3c00}, {0xc100}, {0x4248}, {0x3c01}, {0x7c00}};
  ASSERT_NO_THROW(utils.f32ToF16(f32In, f16Out, /*doClip=*/false));
  EXPECT_EQ(f16Out, expectedF16);

  const std::vector<float16> expectedClippedF16{
      {0x3c00}, {0xc100}, {0x4248}, {0x3c01}, {0x7bff}};
  ASSERT_NO_THROW(utils.f32ToF16(f32In, f16Out, /*doClip=*/true));
  EXPECT_EQ(f16Out, expectedClippedF16);
}

TEST(Float16ConvertKernelsTest, Float16ToFloatAllValues) {
  const Float16ConvertTestUtils utils;
  const std::vector<float16> in = allFloat16();
  std::vector<float> out(in.size());
  EXPECT_NO_THROW(forEachTestSubspan(
      [&](auto inSub, auto outSub) { utils.f16ToF32Test(inSub, outSub); },
      std::span(in),
      std::span(out)));
}

TEST(Float16ConvertKernelsTest, RoundTripAllValues) {
  const Float16ConvertTestUtils utils;
  const std::vector<float16> in = allFloat16();
  std::vector<float> f32Out(in.size());
  std::vector<float16> f16Out(in.size());
  EXPECT_NO_THROW(forEachTestSubspan(
      [&](auto inSub, auto f32Sub, auto f16Sub) {
        utils.f16ToF32ToF16Test(inSub, f32Sub, f16Sub);
      },
      std::span(in),
      std::span(f32Out),
      std::span(f16Out)));
}

TEST(Float16ConvertKernelsTest, FloatToFloat16AroundEveryFloat16) {
  const Float16ConvertTestUtils utils;
  const std::vector<float> in = floatsAroundEveryFloat16();
  std::vector<float16> out(in.size());
  for (const bool doClip : {false, true}) {
    EXPECT_NO_THROW(forEachTestSubspan(
        [&](auto inSub, auto outSub) {
          utils.f32ToF16Test(inSub, outSub, doClip);
        },
        std::span(in),
        std::span(out)))
        << "do_clip " << doClip;
  }
}

} // namespace
} // namespace fbgemm
