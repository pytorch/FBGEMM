/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <mkl.h>
#endif
#include <random>

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <folly/container/F14Map.h>
#include <glog/logging.h>

#include <folly/Random.h>
#include <folly/concurrency/UnboundedQueue.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/hash/Hash.h>

#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/rate_limiter.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <rocksdb/table_properties.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "fbgemm_gpu/dispatch_macros.h"

namespace kv_db {

class CudaExecutor {
 public:
  static folly::CPUThreadPoolExecutor* get_executor();
};

class EmbeddingKVDB : public std::enable_shared_from_this<EmbeddingKVDB> {
 public:
  virtual ~EmbeddingKVDB() {}

  virtual void set(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) = 0;

  virtual void get(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) = 0;

  virtual void compact() = 0;

  virtual void flush() = 0;

  // The function attaches the CUDA callback logic to the compute
  // stream to ensure that the data retrieval is carried out properly.
  // It internally invokes get to fetch values from the KV database.
  void get_cuda(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count);

  void set_cuda(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const int64_t timestep);

 private:
  virtual void flush_or_compact(const int64_t timestep) = 0;
}; // class EmbeddingKVDB

} // namespace kv_db
