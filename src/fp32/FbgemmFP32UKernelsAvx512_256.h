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

void NOINLINE gemmkernel_7x2_Avx512_256_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_8x2_Avx512_256_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_9x2_Avx512_256_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_10x2_Avx512_256_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_11x2_Avx512_256_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_12x2_Avx512_256_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_13x2_Avx512_256_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_14x2_Avx512_256_fp32_fA0fB0fC0(GemmParamsFP32* gp);

} // namespace fbgemm
