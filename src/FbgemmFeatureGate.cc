/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm/FbgemmFeatureGate.h"
#include "fbgemm/FbgemmBuild.h"

#ifdef FBGEMM_FBCODE
#include "fb/FbgemmFeatureGateFb.h"
#endif

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

namespace fbgemm::config {

const std::string to_string(const FeatureGateName& value) {
  switch (value) {
#define X(value)               \
  case FeatureGateName::value: \
    return #value;
    ENUMERATE_ALL_FEATURE_FLAGS
#undef X
  }
  return "UNKNOWN";
}

bool ev_check_key(const std::string& key) {
  const auto env_var = "FBGEMM_" + key;

  const auto value = std::getenv(env_var.c_str());
  if (!value) {
    return false;
  }

  try {
    return std::stoi(value) == 1;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

FBGEMM_API bool check_feature_gate_key(const std::string& key) {
  // Cache feature flags to avoid repeated JK and env var checks
  static std::map<std::string, bool> feature_flags_cache;

  if (const auto search = feature_flags_cache.find(key);
      search != feature_flags_cache.end()) {
    return search->second;

  } else {
#ifdef FBGEMM_FBCODE
    const auto value = jk_check_key(key);
#else
    const auto value = ev_check_key(key);
#endif

    feature_flags_cache.insert({key, value});
    return value;
  }
}

FBGEMM_API bool is_feature_enabled(const FeatureGateName& feature) {
  return check_feature_gate_key(to_string(feature));
}

#ifdef FBGEMM_FBCODE
FBGEMM_API bool is_feature_enabled(const FbFeatureGateName& feature) {
  return check_feature_gate_key(to_string(feature));
}
#endif

} // namespace fbgemm::config
