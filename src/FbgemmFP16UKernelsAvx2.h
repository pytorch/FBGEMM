/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef FBGEMM_UKERNELS
#define FBGEMM_UKERNELS
#include <cstdint>
#include "fbgemm/Types.h"

namespace fbgemm {

using fp16 = float16;
using fp32 = float;
#ifdef _MSC_VER
 #define NOINLINE_ATTR __declspec(noinline)
#else
 #define NOINLINE_ATTR __attribute__((noinline))
#endif
struct GemmParams {
  uint64_t k;
  float* A;
  const fp16* B;
  float* beta;
  uint64_t accum;
  float* C;
  uint64_t ldc;
  uint64_t b_block_cols;
  uint64_t b_block_size;
};
void NOINLINE_ATTR gemmkernel_1x2_AVX2_fA0fB0fC0(GemmParams* gp);
void NOINLINE_ATTR gemmkernel_2x2_AVX2_fA0fB0fC0(GemmParams* gp);
void NOINLINE_ATTR gemmkernel_3x2_AVX2_fA0fB0fC0(GemmParams* gp);
void NOINLINE_ATTR gemmkernel_4x2_AVX2_fA0fB0fC0(GemmParams* gp);
void NOINLINE_ATTR gemmkernel_5x2_AVX2_fA0fB0fC0(GemmParams* gp);
void NOINLINE_ATTR gemmkernel_6x2_AVX2_fA0fB0fC0(GemmParams* gp);
typedef void (*funcptr_fp16)(GemmParams* gp);
;

} // namespace fbgemm

#endif
