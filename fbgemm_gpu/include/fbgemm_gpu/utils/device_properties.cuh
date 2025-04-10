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

////////////////////////////////////////////////////////////////////////////////
// Get CUDA Device From Stream
//
// Given a CUDA stream, fetch the device ID that the stream is associated with.
// This function is memoized since the operation may be expensive
////////////////////////////////////////////////////////////////////////////////

inline auto get_device_for_stream(const cudaStream_t stream) {
  // Keep as thread local to avoid race conditions
  static thread_local std::unordered_map<cudaStream_t, int> table;

  if (const auto search = table.find(stream); search != table.end()) {
    return search->second;

  } else {
    int device = 0;

#if CUDA_VERSION >= 12080
    // CUDA 12.8+ introduced a new API to straightforwardly fetch the device
    // from a given stream
    cudaStreamGetDevice(stream, &device);

#else
    // In the absence of cudaStreamGetDevice() API, the device ID can be fetched
    // by context capture.

    // Save current device
    int current_device;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));

    // Force stream association by capturing dummy work
    cudaStreamCaptureStatus status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));

    // Save the device associated with the stream, and revert back to the
    // current device
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaSetDevice(current_device));
#endif

    table.insert({stream, device});
    return device;
  }
}

} // namespace fbgemm_gpu::utils
