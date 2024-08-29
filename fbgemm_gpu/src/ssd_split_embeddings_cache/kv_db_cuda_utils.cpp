/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kv_db_cuda_utils.h"
#include <ATen/cuda/CUDAContext.h>

namespace kv_db_utils {

namespace {

class CudaExecutor {
 public:
  static folly::CPUThreadPoolExecutor* get_executor() {
    static auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    return executor.get();
  }
};

void host_async_threadpool_executor(void (*f)(void*), void* userData) {
  CudaExecutor::get_executor()->add([f, userData]() { f(userData); });
}

}; // namespace

void cuda_callback_func(
    cudaStream_t /*stream*/,
    cudaError_t status,
    void* functor) {
  AT_CUDA_CHECK(status);
  auto* f = reinterpret_cast<std::function<void()>*>(functor);
  AT_CUDA_CHECK(cudaGetLastError());
  (*f)();
  // delete f; // unfortunately, this invoke destructors that call CUDA
  // API functions (e.g. caching host allocators issue cudaGetDevice(..),
  // etc)
  host_async_threadpool_executor(
      [](void* functor) {
        auto* fn = reinterpret_cast<std::function<void()>*>(functor);
        delete fn;
      },
      functor);
}

}; // namespace kv_db_utils
