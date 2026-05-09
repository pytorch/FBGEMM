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

// These tests run in a binary launched with the following environment
// (set by the BUCK target):
//
//   FBGEMM_NO_JK=2                          -> EnvFirstThenJk policy (FBCODE)
//   FBGEMM_TBE_V2=1                         -> feature is "enabled" via env
//   FBGEMM_BOUNDS_CHECK_INDICES_V2=0        -> feature is "set but disabled"
//
// In FBCODE builds, these exercise the new EnvFirstThenJk path: env_has_key
// returning true must short-circuit the JustKnobs lookup, so the value comes
// from ev_check_key alone.
//
// In OSS builds, the same env values flow through the unconditional env-only
// path (FBGEMM_NO_JK is ignored), and the assertions still hold by
// construction.
//
// Each test uses a distinct FeatureGateName because the FeatureGate
// singleton's per-key cache is process-lifetime: the first lookup for a
// given key freezes its result.

TEST(FeatureGateNoJkTest, env_set_to_one_returns_enabled) {
  EXPECT_TRUE(config::is_feature_enabled(config::FeatureGateName::TBE_V2));
}

TEST(FeatureGateNoJkTest, env_set_to_zero_returns_disabled) {
  EXPECT_FALSE(
      config::is_feature_enabled(
          config::FeatureGateName::BOUNDS_CHECK_INDICES_V2));
}
