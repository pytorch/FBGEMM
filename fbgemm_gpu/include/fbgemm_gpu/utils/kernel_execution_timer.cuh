/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <stdexcept>

namespace fbgemm_gpu::utils {

class KernelExecutionTimer {
 public:
  explicit KernelExecutionTimer(const c10::cuda::CUDAStream stream)
      : stream_(stream) {
    C10_CUDA_CHECK(cudaEventCreate(&start_));
    C10_CUDA_CHECK(cudaEventCreate(&stop_));
  }

  KernelExecutionTimer(const KernelExecutionTimer&) = delete;
  KernelExecutionTimer& operator=(const KernelExecutionTimer&) = delete;
  KernelExecutionTimer(KernelExecutionTimer&&) = delete;
  KernelExecutionTimer& operator=(KernelExecutionTimer&&) = delete;

  ~KernelExecutionTimer() {
    C10_CUDA_CHECK(cudaEventDestroy(start_));
    C10_CUDA_CHECK(cudaEventDestroy(stop_));
  }

  void start() {
    if (started_) {
      throw std::logic_error("Cannot call start() more than once.");
    }
    C10_CUDA_CHECK(cudaEventRecord(start_, stream_));
    started_ = true;
  }

  void stop() {
    if (!started_) {
      throw std::logic_error("Must call start() before stop().");
    }
    if (stopped_) {
      throw std::logic_error("Cannot call stop() more than once.");
    }
    C10_CUDA_CHECK(cudaEventRecord(stop_, stream_));
    stopped_ = true;
  }

  float elapsedMillis() const {
    if (!stopped_) {
      throw std::logic_error(
          "Must call stop() before retrieving elapsed time.");
    }
    float milliseconds = 0;
    C10_CUDA_CHECK(cudaEventSynchronize(stop_)); // Ensure timing is complete
    C10_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  const c10::cuda::CUDAStream stream_;
  bool started_ = false;
  bool stopped_ = false;
};

} // namespace fbgemm_gpu::utils
