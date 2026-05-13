/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/config/feature_gates.h"
#include "fbgemm_gpu/utils/function_types.h"

#ifdef FBGEMM_FBCODE
#include "fbgemm_gpu/config/feature_gates_fb.h"
#endif

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fbgemm_gpu::config {

std::string to_string(const FeatureGateName& value) {
  switch (value) {
#define X(value)               \
  case FeatureGateName::value: \
    return #value;
    ENUMERATE_ALL_FEATURE_FLAGS
#undef X
  }
  return "UNKNOWN";
}

namespace {

// Returns true iff the env var "FBGEMM_<key>" is set (regardless of value).
[[maybe_unused]] bool env_has_key(const std::string& key) {
  const auto env_var = "FBGEMM_" + key;
  return std::getenv(env_var.c_str()) != nullptr;
}

// Reads the env var "FBGEMM_<key>" and returns true iff it is set to "1".
bool env_check_key(const std::string& key) {
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

#ifdef FBGEMM_FBCODE
// Lookup policy controlled by the FBGEMM_NO_JK env var.
//   unset or "0" -> JkOnly
//   "1"          -> EnvOnly
//   "2"          -> EnvFirstThenJk
class NoJkMode {
 public:
  enum Value : uint8_t { JkOnly, EnvOnly, EnvFirstThenJk };

  constexpr NoJkMode(Value value) : value_(value) {}

  constexpr operator Value() const {
    return value_;
  }

  explicit operator bool() const = delete;

  static NoJkMode from_env() {
    const auto value = std::getenv("FBGEMM_NO_JK");
    if (!value) {
      return NoJkMode::JkOnly;
    }

    try {
      const auto parsed = std::stoi(value);
      if (parsed == 1) {
        return NoJkMode::EnvOnly;
      } else if (parsed == 2) {
        return NoJkMode::EnvFirstThenJk;
      } else {
        return NoJkMode::JkOnly;
      }
    } catch (const std::exception&) {
      // Best-effort parse: fall back to JkOnly on any parse failure
      // (std::invalid_argument for non-numeric input,
      //  std::out_of_range for values that overflow int).
      return NoJkMode::JkOnly;
    }
  }

 private:
  Value value_;
};
#endif // FBGEMM_FBCODE

class FeatureGate {
 public:
  static FeatureGate& instance() {
    static FeatureGate gate;
    return gate;
  }

  FeatureGate(const FeatureGate&) = delete;
  FeatureGate& operator=(const FeatureGate&) = delete;
  FeatureGate(FeatureGate&&) = delete;
  FeatureGate& operator=(FeatureGate&&) = delete;

  bool lookup(const std::string& key) {
    if (const auto search = cache_.find(key); search != cache_.end()) {
      return search->second;
    }

    bool value = false;
#ifdef FBGEMM_FBCODE
    static const NoJkMode mode = NoJkMode::from_env();
    if (mode == NoJkMode::EnvOnly) {
      value = env_check_key(key);
    } else if (mode == NoJkMode::EnvFirstThenJk) {
      value = env_has_key(key) ? env_check_key(key) : jk_check_key(key);
    } else /* NoJkMode::JkOnly */ {
      value = jk_check_key(key);
    }
#else
    value = env_check_key(key);
#endif

    cache_.insert({key, value});
    return value;
  }

 private:
  FeatureGate() = default;

  std::unordered_map<std::string, bool> cache_;
};

} // namespace

DLL_PUBLIC bool check_feature_gate_key(const std::string& key) {
  return FeatureGate::instance().lookup(key);
}

DLL_PUBLIC bool is_feature_enabled(const FeatureGateName& feature) {
  return FeatureGate::instance().lookup(to_string(feature));
}

#ifdef FBGEMM_FBCODE
DLL_PUBLIC bool is_feature_enabled(const FbFeatureGateName& feature) {
  return FeatureGate::instance().lookup(to_string(feature));
}
#endif // FBGEMM_FBCODE

} // namespace fbgemm_gpu::config
