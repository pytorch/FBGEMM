/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "../ssd_split_embeddings_cache/initializer.h"
#include "SynchronizedShardedMap.h"
#include "dram_kv_embedding_base.h"
#include "feature_evict.h"

namespace kv_mem {
#define TORCH_CHECK_TENSOR_PROPERTIES(tensor, scalar_type) \
  do {                                                     \
    TORCH_CHECK((tensor).has_value());                     \
    TORCH_CHECK((tensor)->dim() == 1);                     \
    TORCH_CHECK((tensor)->dtype() == (scalar_type));       \
    TORCH_CHECK((tensor)->is_contiguous());                \
    TORCH_CHECK((tensor)->device().is_cpu());              \
  } while (0)

#define TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(                         \
    source_tensor, target_container, data_type)                         \
  do {                                                                  \
    TORCH_CHECK(                                                        \
        (source_tensor)->numel() + 1 == hash_size_cumsum->numel(),      \
        "hash_size_cumsum length must be one more than " #source_tensor \
        " length, but got ",                                            \
        hash_size_cumsum->numel(),                                      \
        " and ",                                                        \
        (source_tensor)->numel());                                      \
    (target_container)                                                  \
        .assign(                                                        \
            (source_tensor)->data_ptr<data_type>(),                     \
            (source_tensor)->data_ptr<data_type>() +                    \
                (source_tensor)->numel());                              \
  } while (0)

/// @ingroup KVMemEmbedding
///
/// @brief An implementation of EmbeddingKVDB for ZCH v.Next
///
template <typename weight_type>
class DramKVEmbeddingCache : public DramKVEmbeddingBase {
 public:
  /// DramKVEmbeddingCache constructor
  ///
  /// @param max_D the maximum dimension of of embedding tensor
  /// @param uniform_init_lower the lower bound of the uniform distribution
  /// @param uniform_init_upper the upper bound of the uniform distribution
  /// @param evict_trigger_mode  evict trigger mode, reference EvictTriggerMode
  /// @param trigger_step_interval trigger step interval
  /// (EvictTriggerMode::ITERATION used)
  /// @param mem_util_threshold_in_GB mem util threshold (.GB
  /// EvictTriggerMode::MEM_UTIL used)
  /// @param evict_trigger_strategy evict trigger strategy, reference
  /// EvictTriggerStrategy
  /// @param counter_thresholds count threshold for each table,
  /// at::ScalarType::UInt32
  /// @param ttls_in_mins the time to feature live for each table,(.minutes)
  /// at::ScalarType::UInt32
  /// @param counter_decay_rates count decay rate for each table,
  /// at::ScalarType::Float
  /// @param l2_weight_thresholds l2 weight threshold for each table,
  /// at::ScalarType::Double
  /// @param num_shards number of shards for the kvstore. This is to improve
  /// parallelization. Each key value pair will be sharded into one shard.
  /// @param num_threads num of threads that kvstore needs to be run upon for
  /// parallelization. This is to improve read and write performance.
  /// @param row_storage_bitwidth storage bitwidth for each row of embedding for
  /// initializers. 32 kFloat || 16 kHalf || 8 kByte
  /// @param enable_async_update whether to enable async update for the cache
  /// @param table_dims the table dimension for each table
  /// @param hash_size_cumsum the hash size cumulative sum for each table
  /// @return None
  explicit DramKVEmbeddingCache(
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t evict_trigger_mode = 0,
      int64_t trigger_step_interval = 0,
      int64_t mem_util_threshold_in_GB = 0.0,
      int64_t evict_trigger_strategy = 1,
      const std::optional<at::Tensor>& counter_thresholds = std::nullopt,
      const std::optional<at::Tensor>& ttls_in_mins = std::nullopt,
      const std::optional<at::Tensor>& counter_decay_rates = std::nullopt,
      const std::optional<at::Tensor>& l2_weight_thresholds = std::nullopt,
      int64_t num_shards = 8,
      int64_t num_threads = 32,
      int64_t row_storage_bitwidth = 32,
      bool enable_async_update = false,
      std::optional<at::Tensor> table_dims = std::nullopt,
      std::optional<at::Tensor> hash_size_cumsum = std::nullopt);

  void initialize_initializers(
      int64_t num_shards,
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t row_storage_bitwidth);

  /// get all ids in the kvstore
  ///
  /// @return a Tensor contained ids
  at::Tensor get_keys_in_range_impl(
      int64_t start,
      int64_t end,
      std::optional<int64_t> offset = std::nullopt) override;

  /// insert embeddings into kvstore.
  /// current underlying memory management is done through F14FastMap
  /// key value pair will be sharded into multiple shards to increase
  /// parallelism.
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights The 2D tensor that each row(embeddings) is paired up with
  /// relative element in <indices>
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return None
  folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const kv_db::RocksdbWriteMode w_mode =
          kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ /*unused*/) override;

  /// Get embeddings from kvstore.
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights The 2D tensor that each row(embeddings) is paired up with
  /// relative element in <indices>. This tensor will be filled up with the
  /// returned embeddings from KVstore.
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return None
  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async_impl(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt);

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override;

  void set_range_to_storage(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length);

  void get_range_from_snapshot(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length,
      const ssd::SnapshotHandle* snapshot_handle, // should be nullptr for dram
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override;

  void get_kv_from_storage_by_snapshot(
      const at::Tensor& ids,
      const at::Tensor& weights,
      const ssd::SnapshotHandle* snapshot_handle, // should be nullptr for dram
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override;

  void compact() override;

  void trigger_feature_evict();

  void maybe_evict() override;

  size_t get_map_used_memsize() const override;

  FeatureEvictMetricTensors get_feature_evict_metric() const override;

 private:
  int64_t get_dim_from_index(int64_t weight_idx) const;

  int64_t get_width_for_weights(
      int64_t weight_idx,
      int64_t width_offset,
      int64_t row_width) const;

  void fill_from_row_storage(
      int shard_id,
      unsigned char* weights_data_ptr,
      int64_t weights_row_index,
      unsigned char* row_storage_data_ptr,
      int64_t width_offset,
      int64_t row_width,
      int64_t copied_width);

  /// shard input ids into multiple shards based on hash function
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return a map of shard id to a vector of id indexes
  folly::F14FastMap<int, std::vector<int64_t>> shard_input(
      const at::Tensor& indices,
      const at::Tensor& count);

  void flush_or_compact(const int64_t timestep) override;

  void feature_evict_resume();

  void feature_evict_pause();

  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  // background thread
  folly::FunctionScheduler scheduler_;
  int64_t max_D_;
  int64_t num_shards_;
  // mempool params
  size_t block_size_;
  size_t block_alignment_;
  SynchronizedShardedMap<int64_t, weight_type*> kv_store_;
  std::atomic_bool is_eviction_ongoing_ = false;
  std::vector<std::unique_ptr<ssd::Initializer>> initializers_;
  int64_t elem_size_;
  std::vector<int64_t> sub_table_dims_;
  std::vector<int64_t> sub_table_hash_cumsum_;
  FeatureEvictConfig feature_evict_config_;
  std::unique_ptr<FeatureEvict<weight_type>> feature_evict_;
  int current_iter_ = 0;
}; // class DramKVEmbeddingCache

} // namespace kv_mem
