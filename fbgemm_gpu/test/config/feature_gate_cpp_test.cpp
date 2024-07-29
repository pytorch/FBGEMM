/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include "fbgemm_gpu/config/feature_gates.h"

namespace config = fbgemm_gpu::config;

TEST(FeatureGateTest, feature_gates) {
  // Enumerate all enum values
#define X(name) config::FeatureGateName::name,
  const config::FeatureGateName flags[] = {ENUMERATE_ALL_FEATURE_FLAGS};
#undef X

  for (const auto flag : flags) {
    EXPECT_NO_THROW([&] {
      const auto flag_val = config::to_string(flag);
      std::cout << "Checking feature flag: " << flag_val << " ..." << std::endl;

      const auto enabled = config::is_feature_enabled(flag);

      std::cout << "Feature " << flag_val
                << " enabled: " << (enabled ? "true" : "false") << std::endl;
    }());
  }
}
