/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "fbgemm_gpu/utils/float.cuh"

namespace fbgemm_gpu {

template <typename T>
__device__ inline T min(const T* from, const T* to) {
  T result = *(from++);
  while (from < to) {
    T next = *(from++);
    result = (result <= next) ? result : next;
  }
  return result;
}

template <typename T>
__device__ inline T max(const T* from, const T* to) {
  T result = *(from++);
  while (from < to) {
    T next = *(from++);
    result = (result >= next) ? result : next;
  }
  return result;
}

// Helper functions for storing float in quantized storage
static DEVICE_INLINE void quantize_float_store(
    at::BFloat16* output,
    const float input) {
  *reinterpret_cast<__nv_bfloat16*>(output) = __float2bfloat16(input);
}

static DEVICE_INLINE void quantize_float_store(
    at::Half* output,
    const float input) {
  *output = __float2half(input);
}

static DEVICE_INLINE void quantize_float_store(
    float* output,
    const float input) {
  *output = input;
}

} // namespace fbgemm_gpu
