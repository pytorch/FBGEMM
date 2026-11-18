/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "src/PlatformSpecificKernels.h" // @manual

namespace fbgemm {
namespace {

const PlatformSpecificKernels& initialKernels = platformKernels();

TEST(PlatformSpecificKernelsFirstCallTest, SelectsOnFirstCall) {
  const std::vector<const PlatformSpecificKernels*> supported =
      supportedPlatformKernels();
  ASSERT_EQ(std::ranges::find(supported, &initialKernels), supported.end());
  EXPECT_EQ(initialKernels.name(), supported.back()->name());
  EXPECT_EQ(&platformKernels(), supported.back());
}

} // namespace
} // namespace fbgemm
