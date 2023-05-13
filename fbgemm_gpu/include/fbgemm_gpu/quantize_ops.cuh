/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

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

} // namespace fbgemm_gpu
