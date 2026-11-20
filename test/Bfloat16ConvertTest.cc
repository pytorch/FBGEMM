/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cpuinfo.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <random>

#include "bench/BenchUtils.h" // @manual
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/FloatConversion.h"
#include "fbgemm/Utils.h"

#if defined(__x86_64__) || defined(_M_X64)
#include "src/CodeGenHelpers.h" // @manual
#endif

using namespace std;
using namespace fbgemm;

namespace {

template <typename F>
void checkLegacyFiniteRounding(F convert) {
  constexpr std::array<uint32_t, 16> bits{
      0x3f808000,
      0x3f818000,
      0xbf808000,
      0xbf818000,
      0x3f807fff,
      0x3f808001,
      0x00008000,
      0x00018000,
      0x80008000,
      0x80018000,
      0x00000000,
      0x80000000,
      0x7f800000,
      0xff800000,
      0x7f7fffff,
      0xff7fffff};
  constexpr std::array<uint16_t, 16> words{
      0x3f81,
      0x3f82,
      0xbf81,
      0xbf82,
      0x3f80,
      0x3f81,
      0x0001,
      0x0002,
      0x8001,
      0x8002,
      0x0000,
      0x8000,
      0x7f80,
      0xff80,
      0x7f80,
      0xff80};
  std::array<float, 33> input{};
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = std::bit_cast<float>(bits[i % bits.size()]);
    EXPECT_EQ(cpu_float2bfloat16(input[i]).val, words[i % bits.size()]);
  }
  for (size_t count : {1, 7, 8, 15, 16, 17, 31, 32, 33}) {
    SCOPED_TRACE(count);
    std::array<bfloat16, 35> output{};
    std::array<bfloat16, 35> expected{};
    output.fill(bfloat16{0xbeef});
    expected.fill(bfloat16{0xbeef});
    for (size_t i = 0; i < count; ++i) {
      expected[i + 1] = bfloat16{words[i % bits.size()]};
    }
    convert(input.data(), output.data() + 1, count);
    EXPECT_EQ(output, expected);
  }
}

} // namespace

template <typename F>
static void checkNaNPreservation(F convert) {
  constexpr std::array<uint32_t, 16> bits{
      0x7f800001,
      0xff800001,
      0x7f803fff,
      0xff803fff,
      0x7fbfffff,
      0xffbfffff,
      0x7fc00000,
      0xffc00000,
      0x7fff8000,
      0xffff8000,
      0x7fffffff,
      0xffffffff,
      0x3f808000,
      0xbf808000,
      0x7f800000,
      0xff800000};
  constexpr std::array<uint16_t, 16> words{
      0x7fc0,
      0xffc0,
      0x7fc0,
      0xffc0,
      0x7fff,
      0xffff,
      0x7fc0,
      0xffc0,
      0x7fff,
      0xffff,
      0x7fff,
      0xffff,
      0x3f81,
      0xbf81,
      0x7f80,
      0xff80};
  std::array<float, 65> input{};
  std::array<bfloat16, 16> scalar{};
  std::array<bfloat16, 16> expected_scalar{};
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = std::bit_cast<float>(bits[i % bits.size()]);
  }
  for (size_t i = 0; i < bits.size(); ++i) {
    scalar[i] = cpu_float2bfloat16(input[i]);
    expected_scalar[i] = bfloat16{words[i]};
  }
  EXPECT_EQ(scalar, expected_scalar);
  for (size_t count : {0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65}) {
    SCOPED_TRACE(count);
    std::array<bfloat16, 67> output{};
    output.fill(bfloat16{0xbeef});
    auto expected = output;
    for (size_t i = 0; i < count; ++i) {
      expected[i + 1] = bfloat16{words[i % words.size()]};
    }
    convert(input.data(), output.data() + 1, count);
    EXPECT_EQ(output, expected);
  }
}

TEST(FBGemmBfloat16Test, NaNPreservation) {
  checkNaNPreservation(FloatToBfloat16_ref);
  checkNaNPreservation(FloatToBfloat16_simd);
}

TEST(FBGemmBfloat16Test, LegacyFiniteRounding) {
  checkLegacyFiniteRounding(FloatToBfloat16_ref);
  checkLegacyFiniteRounding(FloatToBfloat16_simd);
}

#if defined(__x86_64__) || defined(_M_X64)
template <inst_set_t instSet>
static void checkNaNPreservationJit() {
  using Vec = typename simd_info<instSet>::vec_reg_t;
  constexpr int width = simd_info<instSet>::WIDTH_32BIT_ELEMS;
  asmjit::JitRuntime runtime;
  asmjit::CodeHolder code;
  ASSERT_EQ(code.init(runtime.environment(), runtime.cpuFeatures()), 0);
  asmjit::x86::Assembler a(&code);
  asmjit::FuncDetail function;
  function.init(
      asmjit::FuncSignature::build<void, const float*, uint32_t*>(),
      a.environment());
  asmjit::FuncFrame frame;
  frame.init(function);
  frame.setDirtyRegs(
      asmjit::RegGroup::kVec, asmjit::Support::bitMask(0, 1, 2, 3, 4));
  frame.setDirtyRegs(asmjit::RegGroup::kGp, asmjit::Support::bitMask(0, 1, 2));
  asmjit::FuncArgsAssignment args(&function);
  args.assignAll(asmjit::x86::rcx, asmjit::x86::rdx);
  args.updateFuncFrame(frame);
  frame.finalize();
  a.emitProlog(frame);
  a.emitArgsAssignment(frame, args);
  const Vec value(0), nan_mask(1), nan_value(2), scratch(3), bias(4);
  a.vmovups(value, asmjit::x86::ptr(asmjit::x86::rcx));
  a.mov(asmjit::x86::eax, 0x8000);
  a.vmovd(bias.xmm(), asmjit::x86::eax);
  a.vpbroadcastd(bias, bias.xmm());
  emitFp32ToBf16<instSet>(
      a.as<asmjit::x86::Emitter>(),
      value,
      nan_mask,
      nan_value,
      scratch,
      bias,
      asmjit::x86::rax);
  a.vmovups(asmjit::x86::ptr(asmjit::x86::rdx), value);
  a.emitEpilog(frame);
  using Kernel = void (*)(const float*, uint32_t*);
  Kernel kernel = nullptr;
  ASSERT_EQ(runtime.add(&kernel, &code), 0);
  const auto convert = [kernel](
                           const float* input, bfloat16* output, size_t count) {
    for (size_t i = 0; i < count; i += width) {
      std::array<float, width> values{};
      std::array<uint32_t, width> words{};
      const size_t size = std::min(size_t{width}, count - i);
      std::copy_n(input + i, size, values.data());
      kernel(values.data(), words.data());
      for (size_t j = 0; j < size; ++j) {
        output[i + j] = bfloat16{static_cast<uint16_t>(words[j])};
      }
    }
  };
  checkNaNPreservation(convert);
  checkLegacyFiniteRounding(convert);
  ASSERT_EQ(runtime.release(kernel), 0);
}

TEST(FBGemmBfloat16Test, NaNPreservationJitAvx2) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx2Support()) {
    GTEST_SKIP() << "AVX2 is unavailable";
  }
  checkNaNPreservationJit<inst_set_t::avx2>();
}

TEST(FBGemmBfloat16Test, NaNPreservationJitAvx512) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 is unavailable";
  }
  checkNaNPreservationJit<inst_set_t::avx512>();
}

TEST(FBGemmBfloat16Test, NaNPreservationAvx2) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx2Support()) {
    GTEST_SKIP() << "AVX2 is unavailable";
  }
  checkNaNPreservation(FloatToBfloat16_avx2);
}

TEST(FBGemmBfloat16Test, NaNPreservationAvx512) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 is unavailable";
  }
  checkNaNPreservation(FloatToBfloat16_avx512);
}

TEST(FBGemmBfloat16Test, LegacyFiniteRoundingAvx2) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx2Support()) {
    GTEST_SKIP() << "AVX2 is unavailable";
  }
  checkLegacyFiniteRounding(FloatToBfloat16_avx2);
}

TEST(FBGemmBfloat16Test, LegacyFiniteRoundingAvx512) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 is unavailable";
  }
  checkLegacyFiniteRounding(FloatToBfloat16_avx512);
}
#endif

TEST(FBGemmBfloat16Test, Conversion) {
  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b[100]; // bfloat16 type
  float c[100]; // fp32 type
  FloatToBfloat16_ref(a, b, 100);
  Bfloat16ToFloat_ref(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128);
  }
}

TEST(FBGemmBfloat16Test, Conversion_simd) {
  float a[100]; // fp32 type
  for (int i = 0; i < 100; ++i) {
    a[i] = i + 1.25;
  }
  bfloat16 b[100]; // bfloat16 type
  float c[100]; // fp32 type
  FloatToBfloat16_simd(a, b, 100);
  Bfloat16ToFloat_simd(b, c, 100);
  for (int i = 0; i < 100; ++i) {
    // The relative error should be less than 1/(2^7) since bfloat16
    // has 7 bits mantissa.
    EXPECT_LE(fabs(c[i] - a[i]) / a[i], 1.0 / 128)
        << "Conversion results differ at (" << i << " ). ref: " << a[i]
        << " conversion: " << c[i];
  }
}

TEST(FBGemmBfloat16Test, Conversion_simd2) {
  vector<vector<int>> shapes;
  random_device r;
  default_random_engine generator(r());
  uniform_int_distribution<int> dm(1, 256);
  uniform_int_distribution<int> dn(1, 1024);

  for (int i = 0; i < 10; i++) {
    int m = dm(generator);
    int n = dn(generator);
    shapes.push_back({m, n});
  }

  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];

    cerr << "m = " << m << " n = " << n << '\n';
    aligned_vector<float> A_fp32_ref(m * n); // fp32 type
    aligned_vector<bfloat16> A_bfloat16(m * n); // bfloat16 type
    aligned_vector<float> A_fp32_final(m * n); // fp32 type
    // randFill(A_fp32_ref, 0.0f, 4.0f);
    for (int i = 0; i < m * n; ++i) {
      A_fp32_ref[i] = i + 1.25;
    }

    FloatToBfloat16_simd(A_fp32_ref.data(), A_bfloat16.data(), m * n);
    Bfloat16ToFloat_simd(A_bfloat16.data(), A_fp32_final.data(), m * n);
    for (int i = 0; i < m * n; ++i) {
      // The relative error should be less than 1/(2^7) since bfloat16
      // has 7 bits mantissa.
      // printf( "A_fp32_final[%d]: %f; A_fp32_ref[%d]: %f\n", i,
      // A_fp32_final[i], i, A_fp32_ref[i]);
      EXPECT_LE(
          fabs(A_fp32_final[i] - A_fp32_ref[i]) / A_fp32_ref[i], 1.0 / 128);
    }
  }
}
