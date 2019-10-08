/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cstdint>
#include "fbgemm/Types.h"

namespace fbgemm {

using fp16 = float16;
using fp32 = float;
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
void __attribute__((noinline)) gemmkernel_1x2_AVX2_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_2x2_AVX2_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_3x2_AVX2_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_4x2_AVX2_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_5x2_AVX2_fA0fB0fC0(GemmParams* gp);
void __attribute__((noinline)) gemmkernel_6x2_AVX2_fA0fB0fC0(GemmParams* gp);
typedef void (*funcptr_fp16)(GemmParams* gp);
;

} // namespace fbgemm

