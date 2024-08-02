/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/library/library.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_extensions {

/////////////////////////////////////////////////////////////////////////////////////////////////
//                                 Enumerations for reflection
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Fusion type identifier
enum class FusionKind {
  kNone,
  kTensorwiseScaling,
  kRowwiseScaling,
  kBlockwiseScaling,
  kInvalid,
};

/// Mainloop schedule identifier
enum class MainloopSchedule {
  kUnknown,
  kMultistage,
  kWarpspecialized,
  kWarpspecializedPingpong,
  kWarpspecializedCooperative,
  kInvalid,
};

/// Epilogue schedule identifier
enum class EpilogueSchedule {
  kUnknown,
  kNoSmem,
  kTma,
  kInvalid,
};

/// Accumulation algorithm kind
enum class AccumKind {
  kDefault, // For non-fp8 types, Default accumulation is in-place regular
            // accumulation. For fp8 types, Default accumulation is slow with
            // staging into f32 registers.
  kFastAccum, // Fast accumulation is for fp8 is regular accumulation.
  kInvalid,
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                                Enumerations to string conversion
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts a FusionKind enumerant to a string
char const* to_string(FusionKind fusion_kind, bool pretty = false);

/// Converts a Mainloop enumerant to a string
char const* to_string(MainloopSchedule mainloop, bool pretty = false);

/// Converts a Epilogue enumerant to a string
char const* to_string(EpilogueSchedule epilogue, bool pretty = false);

/// Converts a Accumulation kind enumerant to a string
char const* to_string(AccumKind fp8acc, bool pretty = false);

/////////////////////////////////////////////////////////////////////////////////////////////////
//                           Compile-time tags to runtime enumerants
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct KernelScheduleMap;

template <>
struct KernelScheduleMap<cutlass::gemm::KernelMultistage> {
  static MainloopSchedule const kMainloopSchedule =
      MainloopSchedule::kMultistage;
  static AccumKind const kAccumKind = AccumKind::kDefault;
  static EpilogueSchedule const kEpilogueSchedule = EpilogueSchedule::kUnknown;
};

// Default accumulation
template <>
struct KernelScheduleMap<cutlass::gemm::KernelTmaWarpSpecialized> {
  static MainloopSchedule const kMainloopSchedule =
      MainloopSchedule::kWarpspecialized;
  static AccumKind const kAccumKind = AccumKind::kDefault;
  static EpilogueSchedule const kEpilogueSchedule = EpilogueSchedule::kUnknown;
};

template <>
struct KernelScheduleMap<cutlass::gemm::KernelTmaWarpSpecializedPingpong> {
  static MainloopSchedule const kMainloopSchedule =
      MainloopSchedule::kWarpspecializedPingpong;
  static AccumKind const kAccumKind = AccumKind::kDefault;
  static EpilogueSchedule const kEpilogueSchedule = EpilogueSchedule::kUnknown;
};

template <>
struct KernelScheduleMap<cutlass::gemm::KernelTmaWarpSpecializedCooperative> {
  static MainloopSchedule const kMainloopSchedule =
      MainloopSchedule::kWarpspecializedCooperative;
  static AccumKind const kAccumKind = AccumKind::kDefault;
  static EpilogueSchedule const kEpilogueSchedule = EpilogueSchedule::kUnknown;
};

// Fast accumulation for FP8
template <>
struct KernelScheduleMap<cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum> {
  static MainloopSchedule const kMainloopSchedule =
      MainloopSchedule::kWarpspecialized;
  static AccumKind const kAccumKind = AccumKind::kFastAccum;
  static EpilogueSchedule const kEpilogueSchedule = EpilogueSchedule::kUnknown;
};

template <>
struct KernelScheduleMap<
    cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum> {
  static MainloopSchedule const kMainloopSchedule =
      MainloopSchedule::kWarpspecializedPingpong;
  static AccumKind const kAccumKind = AccumKind::kFastAccum;
  static EpilogueSchedule const kEpilogueSchedule = EpilogueSchedule::kUnknown;
};

template <>
struct KernelScheduleMap<
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum> {
  static MainloopSchedule const kMainloopSchedule =
      MainloopSchedule::kWarpspecializedCooperative;
  static AccumKind const kAccumKind = AccumKind::kFastAccum;
  static EpilogueSchedule const kEpilogueSchedule = EpilogueSchedule::kUnknown;
};

} // namespace cutlass_extensions

/////////////////////////////////////////////////////////////////////////////////////////////////
