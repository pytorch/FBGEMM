/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <unordered_map>

namespace fbgemm_gpu::utils {

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

////////////////////////////////////////////////////////////////////////////////
// Get CUDA Device Properties
//
// Given a device by its ID, fetch the device properties.  This function is
// memoized since cudaGetDeviceProperties is a very expensive operation.
////////////////////////////////////////////////////////////////////////////////

inline auto get_device_properties(const int device) {
  // Keep as thread local to avoid race conditions (cudaGetDeviceProperties is
  // known to be thread-safe)
  static thread_local std::unordered_map<int, cudaDeviceProp> table;

  if (const auto search = table.find(device); search != table.end()) {
    return search->second;

  } else {
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    table.insert({device, prop});
    return prop;
  }
}

} // namespace fbgemm_gpu::utils
