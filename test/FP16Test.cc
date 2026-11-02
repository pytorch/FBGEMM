/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "./FBGemmFPTest.h"
#include "fbgemm/FbgemmFP16.h"
#include "fbgemm/FbgemmFPCommon.h"
#include "fbgemm/Utils.h"

using FBGemmFP16Test = fbgemm::FBGemmFPTest<fbgemm::float16>;

INSTANTIATE_TEST_SUITE_P(
    InstantiationName,
    FBGemmFP16Test,
    ::testing::Values(
        std::pair{
            fbgemm::matrix_op_t::NoTranspose,
            fbgemm::matrix_op_t::NoTranspose},
        std::pair{
            fbgemm::matrix_op_t::NoTranspose,
            fbgemm::matrix_op_t::Transpose}));

TEST_P(FBGemmFP16Test, Test) {
  TestRun();
}

TEST_P(FBGemmFP16Test, TestAvx2) {
  TestRunWithIsa(fbgemm::inst_set_t::avx2);
}

TEST_P(FBGemmFP16Test, TestAvx512) {
  TestRunWithIsa(fbgemm::inst_set_t::avx512);
}

TEST_P(FBGemmFP16Test, Unpack) {
  UnpackTestRun();
}

// Covers every AVX-512 kernel row count, K packing boundaries, beta
// accumulation and output partitioning. Operands are chosen so every product
// and sum is exact in FP32, which makes the expected output exact.
TEST_P(FBGemmFP16Test, KernelBoundaries) {
  const auto btrans = GetParam().second;
  constexpr int numThreads = 3;
  constexpr float initialC = 0.25f;
  for (int m : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 121}) {
    for (const auto& [n, k] :
         {std::pair{1, 1}, {17, 31}, {35, 513}, {64, 1025}}) {
      std::vector<float> a(m * k), b(k * n);
      for (size_t i = 0; i < a.size(); ++i) {
        a[i] = (static_cast<int>(i % 13) - 6) * 0.03125f;
      }
      for (size_t i = 0; i < b.size(); ++i) {
        b[i] = (static_cast<int>(i % 11) - 5) * 0.015625f;
      }
      const fbgemm::PackedGemmMatrixFP16 packed(btrans, k, n, 1.0f, b.data());
      for (float beta : {0.0f, 1.0f, -0.5f}) {
        std::vector<float> expected(m * n, beta * initialC);
        for (int row = 0; row < m; ++row) {
          for (int col = 0; col < n; ++col) {
            for (int i = 0; i < k; ++i) {
              expected[row * n + col] += a[row * k + i] *
                  b[btrans == fbgemm::matrix_op_t::Transpose ? col * k + i
                                                             : i * n + col];
            }
          }
        }

        std::vector<float> actual(m * n, initialC);
        for (int thread = 0; thread < numThreads; ++thread) {
          fbgemm::cblas_gemm_compute<fbgemm::float16>(
              fbgemm::matrix_op_t::NoTranspose,
              m,
              a.data(),
              packed,
              beta,
              actual.data(),
              thread,
              numThreads);
        }

        EXPECT_EQ(expected, actual)
            << "m=" << m << " n=" << n << " k=" << k << " beta=" << beta;
      }
    }
  }
}

// x86 test builds compile with -mavx2 -masm=intel, so the trampoline below is
// written in Intel syntax.
#if defined(__x86_64__) && defined(__AVX2__) && !defined(_MSC_VER)
namespace {

// Register and control state around one kernel call. Offsets are hardcoded in
// callKernelWithAbiProbe.
struct AbiState {
  std::array<uint64_t, 6> calleeSaved; // rbx, rbp, r12-r15 set before the call
  std::array<uint64_t, 6> calleeSavedAfter;
  uint64_t scratchPoison; // rcx, rdx, rsi, r8-r11 before the call
  uint64_t rspBefore;
  uint64_t rspAfter;
  uint64_t rflagsAfter;
  uint32_t mxcsrBefore;
  uint32_t mxcsrAfter;
};
static_assert(offsetof(AbiState, calleeSavedAfter) == 48);
static_assert(offsetof(AbiState, scratchPoison) == 96);
static_assert(offsetof(AbiState, rspBefore) == 104);
static_assert(offsetof(AbiState, rspAfter) == 112);
static_assert(offsetof(AbiState, rflagsAfter) == 120);
static_assert(offsetof(AbiState, mxcsrBefore) == 128);
static_assert(offsetof(AbiState, mxcsrAfter) == 132);

// Calls kernel(gp) with sentinels in every callee-saved GPR and a
// non-canonical pointer in every scratch GPR (so a load or store through an
// uninitialized one faults), then records the callee-saved GPRs, rsp, RFLAGS
// and MXCSR seen on return.
__attribute__((naked, noinline)) void callKernelWithAbiProbe(
    fbgemm::funcptr_t<fbgemm::float16> /*kernel*/,
    fbgemm::GemmParams<fbgemm::float16>* /*gp*/,
    AbiState* /*state*/) {
  asm volatile(
      "push rbx\n"
      "push rbp\n"
      "push r12\n"
      "push r13\n"
      "push r14\n"
      "push r15\n"
      "push rdx\n"
      "stmxcsr DWORD PTR [rdx + 128]\n"
      "mov [rdx + 104], rsp\n"
      "mov rax, rdi\n"
      "mov rdi, rsi\n"
      "mov rbx, [rdx]\n"
      "mov rbp, [rdx + 8]\n"
      "mov r12, [rdx + 16]\n"
      "mov r13, [rdx + 24]\n"
      "mov r14, [rdx + 32]\n"
      "mov r15, [rdx + 40]\n"
      "mov rcx, [rdx + 96]\n"
      "mov rsi, rcx\n"
      "mov r8, rcx\n"
      "mov r9, rcx\n"
      "mov r10, rcx\n"
      "mov r11, rcx\n"
      "mov rdx, rcx\n"
      "call rax\n"
      "mov rax, [rsp]\n"
      "mov [rax + 48], rbx\n"
      "mov [rax + 56], rbp\n"
      "mov [rax + 64], r12\n"
      "mov [rax + 72], r13\n"
      "mov [rax + 80], r14\n"
      "mov [rax + 88], r15\n"
      "mov [rax + 112], rsp\n"
      "pushfq\n"
      "pop rcx\n"
      "mov [rax + 120], rcx\n"
      "stmxcsr DWORD PTR [rax + 132]\n"
      "pop rdx\n"
      "pop r15\n"
      "pop r14\n"
      "pop r13\n"
      "pop r12\n"
      "pop rbp\n"
      "pop rbx\n"
      "ret\n");
}

} // namespace

// Calls every AVX-512 FP16 kernel directly and checks that it preserves the
// caller's callee-saved state, writes only its own C tile, and leaves its
// parameters, A and packed B untouched.
TEST(FP16KernelAbiTest, Avx512KernelsPreserveCallerStateAndMemory) {
  if (!fbgemm::isZmm(fbgemm::fbgemmInstructionSet())) {
    GTEST_SKIP() << "Requires full-width AVX-512";
  }
  const auto& kernels = std::get<0>(
      fbgemm::getIsaHandlers<fbgemm::float16>(fbgemm::inst_set_t::avx512));
  constexpr int k = 67;
  constexpr int n = 64; // two 32-column blocks
  constexpr int ldc = n + 16; // padding columns that must stay untouched
  constexpr int guard = 64; // floats before and after the C tile
  constexpr float initialC = 0.25f;
  constexpr float untouched = -12345.0f;

  std::vector<float> b(k * n);
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = (static_cast<int>(i % 11) - 5) * 0.015625f;
  }
  const fbgemm::PackedGemmMatrixFP16 packed(
      fbgemm::matrix_op_t::NoTranspose, k, n, 1.0f, b.data());
  ASSERT_EQ(packed.blockColSize(), 32);
  ASSERT_LE(k, packed.blockRowSize());
  const std::vector<fbgemm::float16> packedBefore(
      packed.pmat(), packed.pmat() + packed.matSize());
  const auto fields = [](const fbgemm::GemmParams<fbgemm::float16>& p) {
    return std::make_tuple(
        p.k, p.A, p.B, p.beta, p.C, p.ldc, p.b_block_cols, p.b_block_size);
  };

  for (int rows = 1; rows <= fbgemm::MAX_KERNEL_NROWS; ++rows) {
    // Packed A layout: the `rows` values of each k step are contiguous.
    std::vector<float> a(k * rows);
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < k; ++i) {
        a[i * rows + r] = (((r * k + i) % 13) - 6) * 0.03125f;
      }
    }
    const std::vector<float> aBefore = a;

    for (float beta : {0.0f, 1.0f}) {
      std::vector<float> expected(guard + rows * ldc + guard, untouched);
      std::vector<float> c = expected;
      for (int r = 0; r < rows; ++r) {
        for (int col = 0; col < n; ++col) {
          float sum = beta * initialC;
          for (int i = 0; i < k; ++i) {
            sum += a[i * rows + r] * b[i * n + col];
          }
          expected[guard + r * ldc + col] = sum;
          c[guard + r * ldc + col] = initialC;
        }
      }

      fbgemm::GemmParams<fbgemm::float16> gp{};
      gp.k = k;
      gp.A = a.data();
      gp.B = packed.pmat();
      gp.beta = beta;
      gp.C = c.data() + guard;
      gp.ldc = ldc * sizeof(float);
      gp.b_block_cols = n / packed.blockColSize();
      gp.b_block_size = k * packed.blockColSize() * sizeof(fbgemm::float16);
      const auto gpBefore = gp;

      AbiState state{};
      state.calleeSaved = {
          0x1111111111111111,
          0x2222222222222222,
          0x3333333333333333,
          0x4444444444444444,
          0x5555555555555555,
          0x6666666666666666};
      state.scratchPoison = 0xdead0000deadbeef;
      callKernelWithAbiProbe(kernels[rows], &gp, &state);

      const auto where = ::testing::Message()
          << "rows=" << rows << " beta=" << beta;
      EXPECT_EQ(state.calleeSaved, state.calleeSavedAfter) << where;
      EXPECT_EQ(state.rspBefore, state.rspAfter) << where;
      EXPECT_EQ(state.rflagsAfter & (1 << 10), 0u) << where << " DF set";
      EXPECT_EQ(state.mxcsrBefore, state.mxcsrAfter) << where;
      EXPECT_EQ(expected, c) << where;
      EXPECT_EQ(fields(gpBefore), fields(gp)) << where;
      EXPECT_EQ(aBefore, a) << where;
      EXPECT_TRUE(
          std::equal(packedBefore.begin(), packedBefore.end(), packed.pmat()))
          << where;
    }
  }
}
#endif
