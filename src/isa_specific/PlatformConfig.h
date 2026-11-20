/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "fbgemm/FbgemmBuild.h"

// Arm compilers don't support enabling a single feature (there is no -msve2).
// You can only override -march as a whole, which can set it to an older
// architecture instead of adding capabilities. So the build system can opt out
// of our -march, and SVE2 may end up not enabled through no fault of the build.
// We accommodate that instead of reporting a misconfiguration: sve128 overrides
// nothing and gets neon's kernels.
#if defined(FBGEMM_SIMD_SVE128) && !defined(__ARM_FEATURE_SVE2)
#define FBGEMM_SIMD_DISABLED sve128
#endif

#ifndef FBGEMM_SIMD_DISABLED
#include "./PlatformConfigValidation.h"
#endif

#ifndef FBGEMM_SIMD_NAMESPACE
#error "Define FBGEMM_SIMD_<ISA> before including"
#endif
