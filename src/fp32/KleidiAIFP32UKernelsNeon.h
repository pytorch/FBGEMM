/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliate
 * <open-source-office@arm.com> SPDX-License-Identifier: BSD-3-Clause
 */

#ifdef FBGEMM_ENABLE_KLEIDIAI

#pragma once
#include <cstdint>
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FbgemmFPCommon.h"
#include "fbgemm/Types.h"

namespace kleidiai {

using GemmParamsFP32 = fbgemm::GemmParams<float>;

void NOINLINE gemmkernel_1x2_Neon_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_2x2_Neon_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_3x2_Neon_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_4x2_Neon_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_5x2_Neon_fp32_fA0fB0fC0(GemmParamsFP32* gp);
void NOINLINE gemmkernel_6x2_Neon_fp32_fA0fB0fC0(GemmParamsFP32* gp);

} // namespace kleidiai

#endif
