/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <cstdint>
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FbgemmFPCommon.h"
#include "fbgemm/Types.h"

namespace fbgemm {

using GemmParamsFP32 = GemmParams<float>;

void NOINLINE gemmkernel_1x2_Avx2_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_2x2_Avx2_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_3x2_Avx2_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_4x2_Avx2_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_5x2_Avx2_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_6x2_Avx2_fp32_fA0fB0fC0(GemmParamsFP32* gp);

} // namespace fbgemm
