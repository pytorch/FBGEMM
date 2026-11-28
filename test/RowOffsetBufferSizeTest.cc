/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include "fbgemm/Fbgemm.h"

namespace fbgemm {

namespace {

// More rows than any ISA's MCB, so the first packed block is a full one.
constexpr int kRows = 1024;
constexpr int kCols = 64;

// A caller sizes the row offset buffer it hands a packA with
// rowOffsetBufferSize(), as fbgemmConv does, and packing fills one entry per
// row of a block. The size must be the block rows the constructor chose for
// the forced ISA: smaller overflows the buffer, and a size taken from another
// ISA is only safe by luck of that host's MCB.
template <typename PackA>
void expectSizedForPackedRows(PackA& packA, int bufferSize) {
  packA.pack(
      block_type_t{
          0,
          packA.blockRowSize(),
          0,
          std::min(packA.blockColSize(), packA.numCols())});
  EXPECT_GE(bufferSize, packA.numPackedRows());
  EXPECT_EQ(bufferSize, packA.blockRowSize());
}

// The buffer the packA writes into is larger than the size under test, so a
// wrong size fails the expectations instead of overflowing.
std::vector<std::int32_t> rowOffsets(int bufferSize) {
  return std::vector<std::int32_t>(std::max(bufferSize, kRows));
}

class RowOffsetBufferSizeTest : public ::testing::TestWithParam<inst_set_t> {
 protected:
  void SetUp() override {
#if !defined(__x86_64__) && !defined(_M_X64)
    GTEST_SKIP() << "ISA forcing is x86 only";
#endif
    ASSERT_TRUE(cpuinfo_initialize());
    fbgemmForceIsa(GetParam());
    if (fbgemmInstructionSet() != GetParam()) {
      GTEST_SKIP() << "The host can't run ISA " << static_cast<int>(GetParam());
    }
  }

  void TearDown() override {
    fbgemmForceIsa(inst_set_t::anyarch);
  }
};

template <typename accT>
void testRowOffset() {
  const std::vector<std::uint8_t> a(kRows * kCols);
  const int bufferSize =
      PackAWithRowOffset<std::uint8_t, accT>::rowOffsetBufferSize();
  auto buffer = rowOffsets(bufferSize);
  PackAWithRowOffset<std::uint8_t, accT> packA(
      matrix_op_t::NoTranspose,
      kRows,
      kCols,
      a.data(),
      kCols,
      nullptr,
      1,
      buffer.data());
  expectSizedForPackedRows(packA, bufferSize);
}

template <typename accT, int SPATIAL_DIM>
void testIm2Col(const conv_param_t<SPATIAL_DIM>& conv_p) {
  int inputSize = conv_p.MB * conv_p.IC;
  for (const int dim : conv_p.IN_DIM) {
    inputSize *= dim;
  }
  const std::vector<std::uint8_t> a(inputSize);
  const int bufferSize =
      PackAWithIm2Col<std::uint8_t, accT, SPATIAL_DIM>::rowOffsetBufferSize();
  auto buffer = rowOffsets(bufferSize);
  PackAWithIm2Col<std::uint8_t, accT, SPATIAL_DIM> packA(
      conv_p, a.data(), nullptr, 0, buffer.data());
  expectSizedForPackedRows(packA, bufferSize);
}

} // namespace

TEST_P(RowOffsetBufferSizeTest, PackAWithRowOffset) {
  testRowOffset<std::int32_t>();
  testRowOffset<std::int16_t>();
}

TEST_P(RowOffsetBufferSizeTest, PackAWithQuantRowOffset) {
  const std::vector<float> a(kRows * kCols);
  const int bufferSize =
      PackAWithQuantRowOffset<std::uint8_t>::rowOffsetBufferSize();
  auto buffer = rowOffsets(bufferSize);
  PackAWithQuantRowOffset<std::uint8_t> packA(
      matrix_op_t::NoTranspose,
      kRows,
      kCols,
      a.data(),
      kCols,
      nullptr,
      1.0f,
      0,
      1,
      buffer.data());
  expectSizedForPackedRows(packA, bufferSize);
}

TEST_P(RowOffsetBufferSizeTest, PackAWithIm2Col) {
  // Output rows (MB * OUT_DIM) exceed every ISA's MCB.
  const conv_param_t<1> conv1d(1, 8, 8, {kRows}, 1, {3}, {1}, {1, 1});
  testIm2Col<std::int32_t>(conv1d);
  testIm2Col<std::int16_t>(conv1d);
  const conv_param_t<2> conv2d(
      1, 8, 8, {32, 32}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1});
  testIm2Col<std::int32_t>(conv2d);
  testIm2Col<std::int16_t>(conv2d);
}

INSTANTIATE_TEST_SUITE_P(
    ForcedIsas,
    RowOffsetBufferSizeTest,
    ::testing::Values(
        inst_set_t::avx2,
        inst_set_t::avx512,
        inst_set_t::avx512_ymm,
        inst_set_t::avx512_vnni,
        inst_set_t::avx512_vnni_ymm),
    [](const ::testing::TestParamInfo<inst_set_t>& info) -> std::string {
      switch (info.param) {
        case inst_set_t::avx2:
          return "avx2";
        case inst_set_t::avx512:
          return "avx512";
        case inst_set_t::avx512_ymm:
          return "avx512_ymm";
        case inst_set_t::avx512_vnni:
          return "avx512_vnni";
        case inst_set_t::avx512_vnni_ymm:
          return "avx512_vnni_ymm";
        default:
          return "isa" + std::to_string(static_cast<int>(info.param));
      }
    });

} // namespace fbgemm
