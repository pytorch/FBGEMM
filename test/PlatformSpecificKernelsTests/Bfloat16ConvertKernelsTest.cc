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

#include "test/PlatformSpecificKernelsTests/Bfloat16ConvertTestUtils.h" // @manual
#include "test/PlatformSpecificKernelsTests/PlatformSpecificKernelsTestUtils.h" // @manual

namespace fbgemm {
namespace {

std::vector<bfloat16> allBfloat16() {
  std::vector<bfloat16> res(size_t{1} << 16);
  for (size_t i = 0; i != res.size(); ++i) {
    res[i].val = static_cast<uint16_t>(i);
  }
  return res;
}

std::vector<float> floatsAroundEveryBfloat16() {
  constexpr std::array<uint32_t, 5> lowBits{
      0x0000, 0x7fff, 0x8000, 0x8001, 0xffff};
  std::vector<float> res;
  for (uint32_t high = 0; high != (uint32_t{1} << 16); ++high) {
    for (uint32_t low : lowBits) {
      res.push_back(std::bit_cast<float>(high << 16 | low));
    }
  }
  return res;
}

TEST(Bfloat16ConvertKernelsTest, Smoke) {
  const Bfloat16ConvertTestUtils utils;

  const std::vector<bfloat16> b16In{
      {0x0000}, {0x3f80}, {0xbf80}, {0x4048}, {0x7f80}};
  const std::vector<float> expectedF32{
      0.0f, 1.0f, -1.0f, 3.125f, std::numeric_limits<float>::infinity()};
  std::vector<float> f32Out(b16In.size());
  ASSERT_NO_THROW(utils.b16ToF32(b16In, f32Out));
  EXPECT_EQ(f32Out, expectedF32);

  const std::vector<float> f32In{
      1.0f,
      -2.5f,
      std::numbers::pi_v<float>,
      1.00390625f,
      std::numeric_limits<float>::max()};
  const std::vector<bfloat16> expectedB16{
      {0x3f80}, {0xc020}, {0x4049}, {0x3f81}, {0x7f80}};
  std::vector<bfloat16> b16Out(f32In.size());
  ASSERT_NO_THROW(utils.f32ToB16(f32In, b16Out));
  EXPECT_EQ(b16Out, expectedB16);
}

TEST(Bfloat16ConvertKernelsTest, Bfloat16ToFloatAllValues) {
  const Bfloat16ConvertTestUtils utils;
  const std::vector<bfloat16> in = allBfloat16();
  std::vector<float> out(in.size());
  EXPECT_NO_THROW(forEachTestSubspan(
      [&](auto inSub, auto outSub) { utils.b16ToF32Test(inSub, outSub); },
      std::span(in),
      std::span(out)));
}

TEST(Bfloat16ConvertKernelsTest, RoundTripAllValues) {
  const Bfloat16ConvertTestUtils utils;
  const std::vector<bfloat16> in = allBfloat16();
  std::vector<float> f32Out(in.size());
  std::vector<bfloat16> b16Out(in.size());
  EXPECT_NO_THROW(forEachTestSubspan(
      [&](auto inSub, auto f32Sub, auto b16Sub) {
        utils.b16ToF32ToB16Test(inSub, f32Sub, b16Sub);
      },
      std::span(in),
      std::span(f32Out),
      std::span(b16Out)));
}

TEST(Bfloat16ConvertKernelsTest, FloatToBfloat16AroundEveryBfloat16) {
  const Bfloat16ConvertTestUtils utils;
  const std::vector<float> in = floatsAroundEveryBfloat16();
  std::vector<bfloat16> out(in.size());
  EXPECT_NO_THROW(forEachTestSubspan(
      [&](auto inSub, auto outSub) { utils.f32ToB16Test(inSub, outSub); },
      std::span(in),
      std::span(out)));
}

} // namespace
} // namespace fbgemm
