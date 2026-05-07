/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <ATen/core/ivalue.h>
#include <folly/coro/Task.h>
#include <folly/futures/Future.h>
#include <torch/script.h>

#include "feature_evict.h"

namespace kv_mem {

/// @ingroup KVMemEmbedding
///
/// @brief Interface for KV Inference Embedding implementations
///
/// This interface defines the core API that all KV embedding implementations
/// must provide, enabling different backend implementations (DRAM, SSD, etc.)
/// to be used interchangeably.
///
template <typename weight_type>
class KVInferenceEmbeddingInterface {
 public:
  virtual ~KVInferenceEmbeddingInterface() = default;

  /// Initialize the initializers for weight initialization
  ///
  /// @param num_shards number of shards for the kvstore
  /// @param max_D the maximum dimension of embedding tensor
  /// @param uniform_init_lower the lower bound of the uniform distribution
  /// @param uniform_init_upper the upper bound of the uniform distribution
  /// @param row_storage_bitwidth storage bitwidth for each row
  /// @param disable_random_init whether to disable random initialization
  virtual void initialize_initializers(
      int64_t num_shards,
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t row_storage_bitwidth,
      bool disable_random_init) = 0;

  /// Set embeddings in the KV store (sync version)
  ///
  /// @param indices The 1D embedding index tensor
  /// @param weights The 2D tensor containing embeddings
  /// @param count A single element tensor with number of indices to process
  /// @param inplace_update_ts Optional timestamp for inplace update
  virtual void set_kv_db_sync(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      std::optional<uint32_t> inplace_update_ts) = 0;

  /// Get embeddings from KV store (sync version)
  ///
  /// @param indices The 1D embedding index tensor
  /// @param weights The 2D tensor to be filled with embeddings
  /// @param count A single element tensor with number of indices to process
  virtual void get_kv_db_sync(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) = 0;

  /// Set embeddings in the KV store (async inference version)
  ///
  /// @param indices The 1D embedding index tensor
  /// @param weights The 2D tensor containing embeddings
  /// @param count A single element tensor with number of indices to process
  /// @param inplace_update_ts Optional timestamp for inplace update
  /// @return SemiFuture for async completion
  virtual folly::SemiFuture<std::vector<folly::Unit>> inference_set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      std::optional<uint32_t> inplace_update_ts) = 0;

  /// Get embeddings from KV store (async)
  ///
  /// @param indices The 1D embedding index tensor
  /// @param weights The 2D tensor to be filled with embeddings
  /// @param count A single element tensor with number of indices to process
  /// @return SemiFuture for async completion
  virtual folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) = 0;

  /// Compact the KV store (placeholder for future implementations)
  virtual void compact() = 0;

  /// Trigger feature eviction
  ///
  /// @param inplace_update_ts Optional timestamp for eviction threshold
  virtual void trigger_feature_evict(
      std::optional<uint32_t> inplace_update_ts = std::nullopt) = 0;

  /// Maybe trigger eviction based on configured trigger mode
  virtual void maybe_evict() = 0;

  /// Wait until ongoing eviction completes
  virtual void wait_until_eviction_done() = 0;

  /// Get the total memory used by the KV store
  ///
  /// @return Memory size in bytes
  virtual size_t get_map_used_memsize_in_bytes() const = 0;

  /// Get the actual memory used by allocated chunks
  ///
  /// @return Memory size in bytes
  virtual size_t get_map_actual_used_chunk_in_bytes() const = 0;

  /// Get the number of rows in the KV store
  ///
  /// @return Number of rows
  virtual size_t get_num_rows() const = 0;

  /// Resume ongoing eviction
  ///
  /// @param force_resume Force resume even if not paused
  virtual void resume_ongoing_eviction(bool force_resume = false) = 0;

  /// Pause ongoing eviction
  ///
  /// @param force_pause Force pause even if not running
  virtual void pause_ongoing_eviction(bool force_pause = false) = 0;

  /// Log statistics for inplace update (inference only)
  virtual void log_inplace_update_stats() = 0;

  /// Get feature eviction metrics
  ///
  /// @return Optional metrics tensors
  virtual std::optional<FeatureEvictMetricTensors> get_feature_evict_metric()
      const = 0;

  /// Get performance metrics
  ///
  /// @param step Current step/iteration
  /// @param interval Reporting interval
  /// @return Vector of performance metrics
  virtual std::vector<double> get_dram_kv_perf(
      const int64_t step,
      const int64_t interval) = 0;

  /// Flush or compact at a specific timestep
  ///
  /// @param timestep The timestep for flush/compact
  virtual void flush_or_compact(const int64_t timestep) = 0;
};

} // namespace kv_mem
