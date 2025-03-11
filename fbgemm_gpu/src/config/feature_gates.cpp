/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/config/feature_gates.h"
#include "fbgemm_gpu/utils/ops_utils.h"

#ifdef FBGEMM_FBCODE
#include "fbgemm_gpu/config/feature_gates_fb.h"
#endif

#include <cstdlib>
#include <map>
#include <string>

namespace fbgemm_gpu::config {

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

DLL_PUBLIC bool check_feature_gate_key(const std::string& key) {
  // Cache feature flags to avoid repeated JK and env var checks
  static std::map<std::string, bool> feature_flags_cache;
#ifdef FBGEMM_FBCODE
  static const auto no_jk = ev_check_key("NO_JK");
#endif

  if (const auto search = feature_flags_cache.find(key);
      search != feature_flags_cache.end()) {
    return search->second;

  } else {
    const auto value =
#ifdef FBGEMM_FBCODE
        (no_jk) ? ev_check_key(key) : jk_check_key(key);
#else
        ev_check_key(key);
#endif

    feature_flags_cache.insert({key, value});
    return value;
  }
}

DLL_PUBLIC bool is_feature_enabled(const FeatureGateName& feature) {
  return check_feature_gate_key(to_string(feature));
}

#ifdef FBGEMM_FBCODE
DLL_PUBLIC bool is_feature_enabled(const FbFeatureGateName& feature) {
  return check_feature_gate_key(to_string(feature));
}
#endif

} // namespace fbgemm_gpu::config

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "check_feature_gate_key(str key) -> bool",
      fbgemm_gpu::config::check_feature_gate_key);
}
