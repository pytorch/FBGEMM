/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "../PlatformSpecificKernels.h"
#include "./PlatformConfig.h"
#include "./PlatformSpecificKernelsMacros.h"

#pragma push_macro("FBGEMM_SIMD_NAMESPACE")
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE ref
#define FBGEMM_SIMD_BASE ::fbgemm::PlatformSpecificKernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx2
#define FBGEMM_SIMD_BASE ::fbgemm::ref::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx512_256
#define FBGEMM_SIMD_BASE ::fbgemm::avx2::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx512_256_bf16
#define FBGEMM_SIMD_BASE ::fbgemm::avx512_256::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx512
#define FBGEMM_SIMD_BASE ::fbgemm::avx512_256::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE avx512_bf16
#define FBGEMM_SIMD_BASE ::fbgemm::avx512::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE neon
#define FBGEMM_SIMD_BASE ::fbgemm::ref::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#define FBGEMM_SIMD_NAMESPACE sve128
#define FBGEMM_SIMD_BASE ::fbgemm::neon::Kernels
#include "./PlatformSpecificKernelsClass.h"
#undef FBGEMM_SIMD_BASE
#undef FBGEMM_SIMD_NAMESPACE

#pragma pop_macro("FBGEMM_SIMD_NAMESPACE")
