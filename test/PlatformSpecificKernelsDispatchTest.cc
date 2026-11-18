/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include "fbgemm/Utils.h"
#include "src/PlatformSpecificKernels.h" // @manual

namespace fbgemm {
namespace {

class PlatformSpecificKernelsDispatchTest : public ::testing::Test {
 protected:
  void TearDown() override {
    fbgemmForceIsa(inst_set_t::anyarch);
  }
};

TEST_F(PlatformSpecificKernelsDispatchTest, ForcedAvx2) {
  if (!cpuinfo_initialize() || !fbgemmHasAvx2Support()) {
    GTEST_SKIP() << "No AVX2";
  }
  fbgemmForceIsa(inst_set_t::avx2);
  EXPECT_EQ(platformKernels().name(), "avx2");
}

TEST_F(PlatformSpecificKernelsDispatchTest, ForcedAvx512Ymm) {
  if (!cpuinfo_initialize() || !fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "No AVX-512";
  }
  fbgemmForceIsa(inst_set_t::avx512_ymm);
  EXPECT_EQ(platformKernels().name(), "avx512_256");
}

TEST_F(PlatformSpecificKernelsDispatchTest, ForcedAvx512VnniYmm) {
  if (!cpuinfo_initialize() || !fbgemmHasAvx512VnniSupport()) {
    GTEST_SKIP() << "No AVX-512 VNNI";
  }
  fbgemmForceIsa(inst_set_t::avx512_vnni_ymm);
  EXPECT_EQ(
      platformKernels().name(),
      fbgemmHasAvx512Bf16Support() ? "avx512_256_bf16" : "avx512_256");
}

TEST_F(PlatformSpecificKernelsDispatchTest, ForcedAvx512) {
  if (!cpuinfo_initialize() || !fbgemmHasAvx512Support()) {
    GTEST_SKIP() << "No AVX-512";
  }
  fbgemmForceIsa(inst_set_t::avx512);
  EXPECT_EQ(platformKernels().name(), "avx512");
}

TEST_F(PlatformSpecificKernelsDispatchTest, NotForcedAvx512Bf16) {
  if (!cpuinfo_initialize() || !fbgemmHasAvx512Support() ||
      !fbgemmHasAvx512Bf16Support()) {
    GTEST_SKIP() << "No AVX-512 BF16";
  }
  fbgemmForceIsa(inst_set_t::anyarch);
  EXPECT_EQ(platformKernels().name(), "avx512_bf16");
}

} // namespace
} // namespace fbgemm
