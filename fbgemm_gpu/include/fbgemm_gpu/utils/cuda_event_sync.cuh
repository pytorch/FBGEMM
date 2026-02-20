/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>

namespace fbgemm_gpu {

inline void maybe_synchronize_cuda_event(
    std::optional<int64_t> event_ptr_to_wait,
    const at::Tensor& tensor) {
  if (!event_ptr_to_wait.has_value() || event_ptr_to_wait.value() == 0) {
    return;
  }
  if (tensor.is_cuda()) {
    C10_CUDA_CHECK(cudaSetDevice(tensor.device().index()));
  }
  cudaEvent_t cuda_event =
      reinterpret_cast<cudaEvent_t>(event_ptr_to_wait.value());
  C10_CUDA_CHECK(cudaEventSynchronize(cuda_event));
}

} // namespace fbgemm_gpu
