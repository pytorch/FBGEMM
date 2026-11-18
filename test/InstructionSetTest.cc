/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include "fbgemm/Utils.h"

namespace fbgemm {

namespace {

inst_set_t expectedDetectedIsa() {
  using enum inst_set_t;
  const bool ymm = fbgemmIsIntelXeonD();
  if (fbgemmHasAvx512VnniSupport()) {
    return ymm ? avx512_vnni_ymm : avx512_vnni;
  }
  if (fbgemmHasAvx512Support()) {
    return ymm ? avx512_ymm : avx512;
  }
  if (fbgemmHasAvx2Support()) {
    return avx2;
  }
  return anyarch;
}

void testIsaEnforcement(
    inst_set_t expectedIsa,
    const std::vector<inst_set_t>& supportedIsas,
    const std::vector<inst_set_t>& unsupportedIsas) {
  EXPECT_EQ(fbgemmInstructionSet(), expectedIsa);
  fbgemmForceIsa(inst_set_t::anyarch);
  EXPECT_EQ(fbgemmInstructionSet(), expectedIsa);
  for (const inst_set_t isa : supportedIsas) {
    fbgemmForceIsa(isa);
    EXPECT_EQ(fbgemmInstructionSet(), isa) << static_cast<int>(isa);
  }
  for (const inst_set_t isa : unsupportedIsas) {
    fbgemmForceIsa(isa);
    EXPECT_EQ(fbgemmInstructionSet(), expectedIsa) << static_cast<int>(isa);
  }
}

class InstructionSetTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    fbgemmEnableAvx512Ymm(true);
  }

  void SetUp() override {
#if !defined(__x86_64__) && !defined(_M_X64)
    GTEST_SKIP() << "ISA forcing is x86 only";
#endif
    ASSERT_TRUE(cpuinfo_initialize());
  }

  void TearDown() override {
    fbgemmForceIsa(inst_set_t::anyarch);
  }
};

} // namespace

TEST_F(InstructionSetTest, ForceIsa) {
  using enum inst_set_t;
  const inst_set_t expected = expectedDetectedIsa();
  switch (expected) {
    case avx512_vnni:
    case avx512_vnni_ymm:
      testIsaEnforcement(
          expected,
          {avx512_vnni, avx512_vnni_ymm, avx512, avx512_ymm, avx2},
          {sve});
      break;
    case avx512:
    case avx512_ymm:
      testIsaEnforcement(
          expected,
          {avx512, avx512_ymm, avx2},
          {avx512_vnni, avx512_vnni_ymm, sve});
      break;
    case avx2:
      testIsaEnforcement(
          expected,
          {avx2},
          {avx512, avx512_ymm, avx512_vnni, avx512_vnni_ymm, sve});
      break;
    default:
      testIsaEnforcement(
          expected,
          {},
          {avx2, avx512, avx512_ymm, avx512_vnni, avx512_vnni_ymm, sve});
      break;
  }
}

} // namespace fbgemm
