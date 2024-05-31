/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kv_db_table_batched_embeddings.h"

namespace kv_db {

folly::CPUThreadPoolExecutor* CudaExecutor::get_executor() {
  static auto executor = std::make_unique<folly::CPUThreadPoolExecutor>(1);
  return executor.get();
}

void hostAsynchronousThreadPoolExecutor(void (*f)(void*), void* userData) {
  CudaExecutor::get_executor()->add([f, userData]() { f(userData); });
}

void EmbeddingKVDB::get_cuda(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count) {
  // take reference to self to avoid lifetime issues.
  auto self = shared_from_this();
  std::function<void()>* functor =
      new std::function<void()>([=]() { self->get(indices, weights, count); });
  auto callFunctor =
      [](cudaStream_t /*stream*/, cudaError_t status, void* userData) -> void {
    AT_CUDA_CHECK(status);
    auto* f = reinterpret_cast<std::function<void()>*>(userData);
    AT_CUDA_CHECK(cudaGetLastError());
    (*f)();
    // delete f; // unfortunately, this invoke destructors that call CUDA
    // API functions (e.g. caching host allocators issue cudaGetDevice(..),
    // etc)
    hostAsynchronousThreadPoolExecutor(
        [](void* userData) {
          auto* fn = reinterpret_cast<std::function<void()>*>(userData);
          delete fn;
        },
        userData);
  };
  AT_CUDA_CHECK(cudaStreamAddCallback(
      at::cuda::getCurrentCUDAStream(), callFunctor, functor, 0));
}

void EmbeddingKVDB::set_cuda(
    const at::Tensor& indices,
    const at::Tensor& weights,
    const at::Tensor& count,
    const int64_t timestep) {
  // take reference to self to avoid lifetime issues.
  auto self = shared_from_this();
  std::function<void()>* functor = new std::function<void()>([=]() {
    self->set(indices, weights, count);
    self->flush_or_compact(timestep);
  });
  auto callFunctor =
      [](cudaStream_t /*stream*/, cudaError_t status, void* userData) -> void {
    AT_CUDA_CHECK(status);
    auto* f = reinterpret_cast<std::function<void()>*>(userData);
    AT_CUDA_CHECK(cudaGetLastError());
    (*f)();
    // delete f; // unfortunately, this invoke destructors that call CUDA
    // API functions (e.g. caching host allocators issue cudaGetDevice(..),
    // etc)
    hostAsynchronousThreadPoolExecutor(
        [](void* userData) {
          auto* fn = reinterpret_cast<std::function<void()>*>(userData);
          delete fn;
        },
        userData);
  };
  AT_CUDA_CHECK(cudaStreamAddCallback(
      at::cuda::getCurrentCUDAStream(), callFunctor, functor, 0));
}

} // namespace kv_db
