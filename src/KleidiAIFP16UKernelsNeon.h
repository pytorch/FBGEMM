/*
 * @lint-ignore-every LICENSELINT
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliate
 * <open-source-office@arm.com> SPDX-License-Identifier: BSD-3-Clause
 */
#ifdef FBGEMM_ENABLE_KLEIDIAI

#pragma once
#include <cstdint>
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FbgemmFPCommon.h"
#include "fbgemm/Types.h"

namespace kleidiai {

using GemmParamsFP16 = fbgemm::GemmParams<fbgemm::float16>;

void NOINLINE gemmkernel_1x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_2x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_3x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_4x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_5x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_6x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_7x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp);
void NOINLINE gemmkernel_8x1_Neon_fp16_fA0fB0fC0(GemmParamsFP16* gp);

} // namespace kleidiai

#endif
