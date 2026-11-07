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
#include <chrono>
#include <cmath>
#include <random>

#include "bench/BenchUtils.h" // @manual
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/FloatConversion.h"
#include "src/CodeGenHelpers.h" // @manual

using namespace std;
using namespace fbgemm;

namespace {

using RoundingKernel = void (*)(const float*, uint16_t*);

template <typename Vec>
RoundingKernel makeRoundingKernel(
    asmjit::JitRuntime& runtime,
    bool native,
    int live_lanes = 0) {
  asmjit::CodeHolder code;
  code.init(runtime.environment());
  asmjit::x86::Assembler a(&code);
  asmjit::FuncDetail func;
  func.init(
      asmjit::FuncSignature::build<void, const float*, uint16_t*>(),
      a.environment());
  asmjit::FuncFrame frame;
  frame.init(func);
  frame.setDirtyRegs(asmjit::RegGroup::kGp, asmjit::Support::bitMask(0, 1, 2));
  frame.setDirtyRegs(asmjit::RegGroup::kVec, asmjit::Support::bitMask(0, 1, 2));
  asmjit::FuncArgsAssignment args(&func);
  args.assignAll(x86::rax, x86::rdx);
  args.updateFuncFrame(frame);
  frame.finalize();
  a.emitProlog(frame);
  a.emitArgsAssignment(frame, args);
  const Vec value(0), scratch(1), bias(2);
  a.vmovups(value, x86::ptr(x86::rax));
  a.mov(x86::ecx, 0x7fff);
  a.vmovd(bias.xmm(), x86::ecx);
  a.vpbroadcastd(bias, bias.xmm());
  if constexpr (std::is_same_v<Vec, Zmm>) {
    x86::KReg mask;
    if (live_lanes) {
      a.mov(x86::ecx, (1 << live_lanes) - 1);
      a.kmovw(x86::k(1), x86::ecx);
      mask = x86::k(1);
    }
    genBfloat16Store(
        a.as<x86::Emitter>(),
        value,
        scratch,
        bias,
        x86::ptr(x86::rdx),
        mask,
        native);
  } else {
    genBfloat16Rounding(a.as<x86::Emitter>(), value, scratch, bias);
    a.vpackusdw(value, value, value);
    a.vpermq(value, value, 0xd8);
    a.vmovups(x86::ptr(x86::rdx), value.xmm());
  }
  a.emitEpilog(frame);
  RoundingKernel kernel = nullptr;
  EXPECT_EQ(runtime.add(&kernel, &code), asmjit::kErrorOk);
  return kernel;
}

template <typename Vec>
void checkJitRounding(bool native) {
  asmjit::JitRuntime runtime;
  const auto kernel = makeRoundingKernel<Vec>(runtime, native);
  ASSERT_NE(kernel, nullptr);
  constexpr int lanes = std::is_same_v<Vec, Zmm> ? 16 : 8;
  // The first group exercises hardware conversion; the second includes
  // subnormals, which require the software fallback on AVX512-BF16 hosts.
  const uint32_t cases[] = {
      0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000, 0x3f807fff, 0x3f808001,
      0x7f800000, 0xff800000, 0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff,
      0x7f7fffff, 0xff7fffff, 0x00000000, 0x80000000, 0x00008000, 0x00018000,
      0x80008000, 0x80018000, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
      0x3f800000, 0x40000000, 0x3f818000, 0xbf818000, 0x7fc00000, 0xffc00000,
      0x00000001, 0x80000001};
  for (int base = 0; base < 32; base += lanes) {
    std::array<float, 16> input{};
    std::array<uint16_t, 18> output;
    output.fill(0xbeef);
    for (int i = 0; i < lanes; ++i) {
      input[i] = std::bit_cast<float>(cases[base + i]);
    }
    kernel(input.data(), output.data() + 1);
    EXPECT_EQ(output[0], 0xbeef);
    EXPECT_EQ(output[lanes + 1], 0xbeef);
    for (int i = 0; i < lanes; ++i) {
      EXPECT_EQ(output[i + 1], cpu_float2bfloat16(input[i]).val)
          << "input bits " << cases[base + i];
    }
  }
  if constexpr (std::is_same_v<Vec, Zmm>) {
    for (int live = 1; live < lanes; ++live) {
      const auto tail = makeRoundingKernel<Vec>(runtime, native, live);
      ASSERT_NE(tail, nullptr);
      for (int base : {0, 16}) {
        std::array<float, 16> input;
        std::array<uint16_t, 18> output;
        output.fill(0xbeef);
        for (int i = 0; i < lanes; ++i) {
          input[i] = std::bit_cast<float>(cases[base + i]);
        }
        tail(input.data(), output.data() + 1);
        EXPECT_EQ(output[0], 0xbeef);
        for (int i = 0; i < live; ++i) {
          EXPECT_EQ(output[i + 1], cpu_float2bfloat16(input[i]).val);
        }
        for (int i = live + 1; i < 18; ++i) {
          EXPECT_EQ(output[i], 0xbeef) << "tail length " << live;
        }
      }
    }
  }
}

} // namespace

TEST(FBGemmBfloat16Test, JitAvx2Rounding) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx2Support()) {
    GTEST_SKIP() << "AVX2 is unavailable";
  }
  checkJitRounding<Ymm>(false);
}

TEST(FBGemmBfloat16Test, JitAvx512Rounding) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 is unavailable";
  }
  checkJitRounding<Zmm>(false);
}

TEST(FBGemmBfloat16Test, JitAvx512Bf16Rounding) {
  ASSERT_TRUE(cpuinfo_initialize());
  if (!fbgemmHasAvx512Support() || !fbgemmHasAvx512Bf16Support()) {
    GTEST_SKIP() << "AVX512-BF16 is unavailable";
  }
  checkJitRounding<Zmm>(true);
  asmjit::JitRuntime runtime;
  const auto software = makeRoundingKernel<Zmm>(runtime, false);
  const auto native = makeRoundingKernel<Zmm>(runtime, true);
  ASSERT_NE(software, nullptr);
  ASSERT_NE(native, nullptr);
  std::array<float, 16> input;
  std::array<uint16_t, 16> output;
  input.fill(1.25f);
  auto measure = [&](RoundingKernel kernel) {
    constexpr int iterations = 500000;
    for (int i = 0; i < 10000; ++i) {
      kernel(input.data(), output.data());
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      kernel(input.data(), output.data());
    }
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - start)
               .count() /
        iterations;
  };
  std::array<double, 7> software_ns, native_ns;
  for (int i = 0; i < 7; ++i) {
    if (i % 2 == 0) {
      software_ns[i] = measure(software);
      native_ns[i] = measure(native);
    } else {
      native_ns[i] = measure(native);
      software_ns[i] = measure(software);
    }
  }
  std::sort(software_ns.begin(), software_ns.end());
  std::sort(native_ns.begin(), native_ns.end());
  std::cout
      << "BF16 conversion and packed store, 16 lanes, median of 7 alternating warm runs: "
      << "software " << software_ns[3] << " ns; native " << native_ns[3]
      << " ns; CPU " << cpuinfo_get_package(0)->name << '\n';
  EXPECT_EQ(output[0], cpu_float2bfloat16(input[0]).val);
}

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

TEST(FBGemmBfloat16Test, ScalarRoundingAndNaN) {
  const std::pair<uint32_t, uint16_t> cases[] = {
      {0x3F808000, 0x3F80},
      {0x3F818000, 0x3F82},
      {0xBF808000, 0xBF80},
      {0xBF818000, 0xBF82},
      {0x00008000, 0x0000},
      {0x00018000, 0x0002},
      {0x7F800000, 0x7F80},
      {0xFF800000, 0xFF80},
      {0x7F800001, 0x7FC0},
      {0xFFFFFFFF, 0xFFFF}};
  for (const auto& [bits, expected] : cases) {
    const float input = std::bit_cast<float>(bits);
    EXPECT_EQ(cpu_float2bfloat16(input).val, expected);
    bfloat16 output;
    FloatToBfloat16_ref(&input, &output, 1);
    EXPECT_EQ(output.val, expected);
  }
}
