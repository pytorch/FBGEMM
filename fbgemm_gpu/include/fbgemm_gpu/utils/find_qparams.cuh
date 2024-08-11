/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/vec4.cuh"

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// Find Quantization Parameters (Thrust)
////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__device__ float2 thrust_find_qparams(scalar_t* input_row, int D) {
  float2 qparams;

  scalar_t scalar_minimum = *(input_row++);
  scalar_t scalar_maximum = scalar_minimum;

  while (--D > 0) {
    scalar_t next = *(input_row++);
    scalar_minimum = (scalar_minimum <= next) ? scalar_minimum : next;
    scalar_maximum = (scalar_maximum >= next) ? scalar_maximum : next;
  }
  float minimum_element = scalar_minimum;
  float maximum_element = scalar_maximum;

  float range = maximum_element - minimum_element;
  qparams.x = range / 255.0f;
  qparams.y = minimum_element;
  return qparams;
}

template <typename scalar_t>
__device__ float2
thrust_find_qparams(fbgemm_gpu::Vec4T<scalar_t>* input_row, int D) {
  // TODO: replace uses in backward kernels with warp find qparams
  float2 qparams;
  float min_val = input_row[0].vmin();
  float max_val = input_row[0].vmax();
  for (int i = 0; i < D / 4; ++i) {
    min_val = min(min_val, input_row[i].vmin());
    max_val = max(max_val, input_row[i].vmax());
  }
  qparams.x = (max_val - min_val) / 255.0f;
  qparams.y = min_val;
  return qparams;
}

////////////////////////////////////////////////////////////////////////////////
// Find Quantization Parameters (Warp)
////////////////////////////////////////////////////////////////////////////////

// Min a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warp_reduce_min(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val = std::min(val, shfl_xor(val, mask));
  }
  return val;
}

// Max a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val = std::max(val, shfl_xor(val, mask));
  }
  return val;
}

template <typename scalar_t>
DEVICE_INLINE float2 warp_find_qparams(scalar_t local_min, scalar_t local_max) {
  float2 qparams;
  qparams.x = 0.0f;
  qparams.y = 0.0f;
  local_min = warp_reduce_min<scalar_t>(local_min);
  local_max = warp_reduce_max<scalar_t>(local_max);
  if (threadIdx.x == 0) {
    qparams.x = (local_max - local_min) / 255.0f;
    qparams.y = local_min;
  }
  qparams.x = shfl_sync(qparams.x, 0);
  qparams.y = shfl_sync(qparams.y, 0);
  return qparams;
}

} // namespace fbgemm_gpu
