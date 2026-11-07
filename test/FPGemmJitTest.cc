/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuinfo.h>

#include <cmath>
#include <type_traits>
#include <vector>

#include "fbgemm/FloatConversion.h"
#include "fbgemm/SimdUtils.h"
#include "src/GenerateKernelFP16FP32.h"  // @manual

namespace fbgemm {
namespace {
template <typename T, inst_set_t Isa>
void checkKernels(int max_rows) {
  constexpr int lanes = simd_info<Isa>::WIDTH_32BIT_ELEMS;
  for (int rows = 1; rows <= max_rows; ++rows) {
    auto kernel = generateGemmKernel<T, Isa>(rows);
    ASSERT_NE(kernel, nullptr);
    for (int k : {1, 2, 5, 9}) {
      for (int blocks : {1, 2}) {
        for (float beta : {0.f, -0.f, 2.f, NAN, -NAN}) {
          SCOPED_TRACE(::testing::Message()
                       << "rows=" << rows << " k=" << k << " blocks=" << blocks
                       << " beta=" << beta);
          const int n = 2 * lanes * blocks;
          std::vector<float> a(rows * k);
          std::vector<T> b(n * k);
          std::vector<float> c(rows * n + 8, 12345.f);
          for (int i = 0; i < rows * k; ++i) {
            a[i] = float(i % 7 - 3);
          }
          for (int i = 0; i < n * k; ++i) {
            float value = float(i % 13 - 6) * 0.25f;
            if constexpr (std::is_same_v<T, float16>) {
              b[i] = cpu_float2half_rn(value);
            } else {
              b[i] = value;
            }
          }
          for (int i = 0; i < rows * n; ++i) {
            c[i] = beta == 0 ? NAN : 4.f;
          }
          GemmParams<T> params{
              uint64_t(k),      a.data(),
              b.data(),         beta,
              c.data(),         uint64_t(n * sizeof(float)),
              uint64_t(blocks), uint64_t(k * 2 * lanes * sizeof(T))};
          kernel(&params);
          for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < n; ++col) {
              float expected = beta == 0 ? 0.f : beta * 4.f;
              for (int kk = 0; kk < k; ++kk) {
                int index = (col / (2 * lanes)) * k * 2 * lanes +
                            kk * 2 * lanes + col % (2 * lanes);
                expected = std::fma(a[kk * rows + row],
                                    float(index % 13 - 6) * 0.25f, expected);
              }
              if (std::isnan(expected)) {
                EXPECT_TRUE(std::isnan(c[row * n + col]));
              } else {
                EXPECT_FLOAT_EQ(c[row * n + col], expected);
              }
            }
          }
          for (int i = rows * n; i < rows * n + 8; ++i) {
            EXPECT_EQ(c[i], 12345.f);
          }
        }
      }
    }
  }
}

template <typename T>
class FPGemmJitTest : public ::testing::Test {};
using InputTypes = ::testing::Types<float16, float>;
TYPED_TEST_SUITE(FPGemmJitTest, InputTypes);

TYPED_TEST(FPGemmJitTest, Avx2) {
  if (!cpuinfo_initialize() || !fbgemmHasAvx2Support()) {
    GTEST_SKIP() << "AVX2 not supported";
  }
  checkKernels<TypeParam, inst_set_t::avx2>(6);
}

TYPED_TEST(FPGemmJitTest, Avx512) {
  if (!cpuinfo_initialize() || !fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 not supported";
  }
  checkKernels<TypeParam, inst_set_t::avx512>(14);
}

TYPED_TEST(FPGemmJitTest, Avx512Ymm) {
  if (!cpuinfo_initialize() || !fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "AVX512 not supported";
  }
  checkKernels<TypeParam, inst_set_t::avx512_ymm>(14);
}
}  // namespace
}  // namespace fbgemm
#endif
