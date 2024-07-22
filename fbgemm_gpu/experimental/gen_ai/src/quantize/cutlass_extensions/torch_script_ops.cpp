/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

#include "c10/core/ScalarType.h"

#include <ATen/cuda/CUDAEvent.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <vector>
#include "c10/util/Exception.h"

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
#define torch_fp8_e4m3 at::kFloat8_e4m3fnuz
#else
#define torch_fp8_e4m3 at::kFloat8_e4m3fn
#endif

namespace fbgemm_gpu {

// CUTLASS kernel v2
at::Tensor f8f8bf16_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor scale,
    bool use_fast_accum = true);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
#ifndef USE_ROCM
  // CUTLASS kernel v2
  m.def(
      "f8f8bf16_v2(Tensor XQ, Tensor WQ, Tensor scale, bool use_fast_accum=True) -> Tensor");

#endif
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
#ifndef USE_ROCM
  // CUTLASS kernel v2
  m.impl("f8f8bf16_v2", f8f8bf16_v2);
#endif
}

} // namespace fbgemm_gpu
