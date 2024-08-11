/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cutlass_extensions/include/utils.h>

namespace cutlass_extensions {

static struct {
  char const* text;
  char const* pretty;
  FusionKind enumerant;
} FusionKind_enumerants[] = {
    {"none", "None", FusionKind::kNone},
    {"tensorwise scaling", "TensorWiseScaling", FusionKind::kTensorwiseScaling},
    {"rowwise scaling", "RowWiseScaling", FusionKind::kRowwiseScaling},
    {"blockwise scaling", "BlockWiseScaling", FusionKind::kBlockwiseScaling},
    {"invalid", "Invalid", FusionKind::kInvalid}};

/// Converts a FusionKind enumerant to a string
char const* to_string(FusionKind fusion_kind, bool pretty) {
  for (auto const& possible : FusionKind_enumerants) {
    if (fusion_kind == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      } else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}
/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const* text;
  char const* pretty;
  MainloopSchedule enumerant;
} MainloopSchedule_enumerants[] = {
    {"unknown", "Unknown", MainloopSchedule::kUnknown},
    {"multistage", "Multistage", MainloopSchedule::kMultistage},
    {"warpspecialized", "Warpspecialized", MainloopSchedule::kWarpspecialized},
    {"warpspecialized pingpong",
     "WarpspecializedPingpong",
     MainloopSchedule::kWarpspecializedPingpong},
    {"warpspecialized cooperative",
     "WarpspecializedCooperative",
     MainloopSchedule::kWarpspecializedCooperative},
    {"invalid", "Invalid", MainloopSchedule::kInvalid},
};

/// Converts a MainloopSchedule enumerant to a string
char const* to_string(MainloopSchedule mainloop_schedule, bool pretty) {
  for (auto const& possible : MainloopSchedule_enumerants) {
    if (mainloop_schedule == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      } else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}
/////////////////////////////////////////////////////////////////////////////////////////////////

static struct {
  char const* text;
  char const* pretty;
  AccumKind enumerant;
} AccumKind_enumerants[] = {
    {"default", "Default", AccumKind::kDefault},
    {"fast accum", "FastAccum", AccumKind::kFastAccum},
    {"invalid", "Invalid", AccumKind::kInvalid},
};

/// Converts a MainloopSchedule enumerant to a string
char const* to_string(AccumKind accum_kind, bool pretty) {
  for (auto const& possible : AccumKind_enumerants) {
    if (accum_kind == possible.enumerant) {
      if (pretty) {
        return possible.pretty;
      } else {
        return possible.text;
      }
    }
  }

  return pretty ? "Invalid" : "invalid";
}
/////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace cutlass_extensions
