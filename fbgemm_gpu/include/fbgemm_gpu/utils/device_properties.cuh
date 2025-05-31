/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

namespace fbgemm_gpu::utils::cuda {

// Based on the empirical study, max grid size that is 64x larger than the
// number of SMs gives good performance across the board
constexpr int32_t MAX_THREAD_BLOCKS_FACTOR = 64;

inline auto get_max_thread_blocks(const c10::cuda::CUDAStream& stream) {
  const auto device = stream.device_index();
  return MAX_THREAD_BLOCKS_FACTOR *
      at::cuda::getDeviceProperties(device)->multiProcessorCount;
}

inline auto get_compute_versions() {
  static const auto versions = [] {
    int runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);

    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);

    return std::make_tuple(runtime_version, driver_version);
  }();

  return versions;
}

} // namespace fbgemm_gpu::utils::cuda
