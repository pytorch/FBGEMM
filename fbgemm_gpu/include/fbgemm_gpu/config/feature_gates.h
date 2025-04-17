/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#ifdef FBGEMM_FBCODE
#include "deeplearning/fbgemm/fbgemm_gpu/fb/include/fbgemm_gpu/config/feature_gates_fb.h"
#endif

/// @defgroup fbgemm-gpu-config FBGEMM_GPU Configuration
/// FBGEMM_GPU runtime configuration and settings

namespace fbgemm_gpu::config {

/// @ingroup fbgemm-gpu-config
///
/// @brief FBGEMM_GPU feature gates enum (C++).
///
///   Feature gates are used to enable/disable experimental features based on
///   environment settings.
///
///   ENUMs are defined using the X-macro pattern.  To add a feature gate,
///   simply append `X(FEATURE_NAME)` to the `ENUMERATE_ALL_FEATURE_FLAGS`
///   macro. Then, to use the feature gate, see example below.
///
///   **Example:**
///   ```c++
///     namespace config = fbgemm_gpu::config;
///
///     void foo() {
///       if (config::is_feature_enabled(config::FeatureGateName::FEATURE_NAME))
///       {
///         // Do something if feature is enabled
///         ...
///       } else {
///         // Do something different if feature is disabled
///         ...
///       }
///     }
///   ```
///
/// @note
///
/// While not required, it is best to mirror the enum values in Python,
/// in `fbgemm_gpu.config.FeatureGateName`
///
/// For fbcode: The ENUM name must match EXACTLY with the JK knob name in the
/// UI.
///
/// For OSS: The environment variable will be evaluated as f"FBGEMM_{ENUM}"
#define ENUMERATE_ALL_FEATURE_FLAGS \
  X(TBE_V2)                         \
  X(TBE_ENSEMBLE_ROWWISE_ADAGRAD)   \
  X(TBE_ANNOTATE_KINETO_TRACE)      \
  X(TBE_ROCM_INFERENCE_PACKED_BAGS) \
  X(TBE_ROCM_HIP_BACKWARD_KERNEL)   \
  X(BOUNDS_CHECK_INDICES_V2)
// X(EXAMPLE_FEATURE_FLAG)

/// @ingroup fbgemm-gpu-config
///
/// @brief Enum class definition for feature gates, generated using the X-macro
/// pattern
enum class FeatureGateName {
#define X(value) value,
  ENUMERATE_ALL_FEATURE_FLAGS
#undef X
};

/// @ingroup fbgemm-gpu-config
///
/// @brief Get the string value of the `FeatureGateName` enum.
const std::string to_string(const FeatureGateName& value);

/// @ingroup fbgemm-gpu-config
///
/// @brief Look up the feature gate value for the given key.
bool check_feature_gate_key(const std::string& key);

/// @ingroup fbgemm-gpu-config
///
/// @brief For the given `FeatureGateName`, check if the corresponding feature
/// is enabled.
bool is_feature_enabled(const FeatureGateName& feature);

#ifdef FBGEMM_FBCODE
bool is_feature_enabled(const FbFeatureGateName& feature);
#endif

} // namespace fbgemm_gpu::config
