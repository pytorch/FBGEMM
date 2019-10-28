/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>
#include "FbgemmFP16UKernelsAvx2.h"
#include "fbgemm/Types.h"

namespace fbgemm {

void __attribute__((noinline)) gemmkernel_1x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_2x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_3x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_4x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_5x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_6x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_7x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_8x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_9x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_10x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_11x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_12x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_13x2_Avx512_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_14x2_Avx512_fA0fB0fC0(GemmParams* gp);
typedef void (*funcptr_fp16)(GemmParams* gp);
;

} // namespace fbgemm

