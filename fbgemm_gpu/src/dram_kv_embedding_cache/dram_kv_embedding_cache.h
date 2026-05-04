/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <ATen/core/ivalue.h>
#include <caffe2/torch/fb/distributed/wireSerializer/WireSerializer.h>
#include <common/base/Proc.h>
#include <common/stats/Stats.h>
#include <folly/SocketAddress.h>
#include <folly/coro/BlockingWait.h>
#include <folly/coro/Collect.h>
#include <folly/coro/Invoke.h>
#include <folly/coro/Task.h>
#include <folly/executors/FunctionScheduler.h>
#include <folly/gen/Base.h>
#include <folly/logging/xlog.h>
#include <servicerouter/client/cpp2/ServiceRouter.h>
#include <thrift/lib/cpp2/protocol/Serializer.h>
#include <torch/script.h>
#include <cmath>
#include <random>
#include <string_view>
#include "common/time/Time.h"

#include "../ssd_split_embeddings_cache/initializer.h"
#include "../ssd_split_embeddings_cache/kv_db_table_batched_embeddings.h"
#include "SynchronizedShardedMap.h"
#include "deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/feature_store_enrichment.h"
#include "deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/igr_enrichment.h"
#include "deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/oneflow_enrichment.h"
#include "enrichment_config.h"
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "feature_evict.h"
#include "fixed_block_pool.h"

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
            (source_tensor)->const_data_ptr<data_type>(),               \
            (source_tensor)->const_data_ptr<data_type>() +              \
                (source_tensor)->numel());                              \
  } while (0)

/// @ingroup KVMemEmbedding
///
/// @brief An implementation of EmbeddingKVDB for ZCH v.Next
///
template <typename weight_type>
class DramKVEmbeddingCache : public kv_db::EmbeddingKVDB {
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
  /// at::ScalarType::Int32
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
      std::optional<c10::intrusive_ptr<kv_mem::FeatureEvictConfig>>
          feature_evict_config,
      int64_t num_shards = 8,
      int64_t num_threads = 32,
      int64_t row_storage_bitwidth = 32,
      bool backend_return_whole_row = false,
      bool enable_async_update = false,
      std::optional<at::Tensor> table_dims = std::nullopt,
      std::optional<at::Tensor> hash_size_cumsum = std::nullopt,
      bool is_training = true,
      bool disable_random_init = false,
      bool enable_raw_embedding_streaming = false,
      int64_t res_store_shards = 0,
      int64_t res_server_port = 0,
      std::vector<std::string> table_names = {},
      std::vector<int64_t> table_offsets = {},
      std::vector<int64_t> table_sizes = {},
      std::optional<c10::intrusive_ptr<kv_mem::EnrichmentConfig>>
          enrichment_config = std::nullopt)
      : kv_db::EmbeddingKVDB(
            num_shards,
            max_D,
            0, // l2_cache_size_gb =0 to disable l2 cache
            0, // tbe_unqiue_id
            2, // ele_size_bytes
            enable_async_update,
            enable_raw_embedding_streaming,
            res_store_shards,
            res_server_port,
            std::move(table_names),
            std::move(table_offsets),
            table_sizes),
        max_D_(max_D),
        num_shards_(num_shards),
        block_size_(FixedBlockPool::calculate_block_size<weight_type>(max_D)),
        block_alignment_(
            FixedBlockPool::calculate_block_alignment<weight_type>()),
        kv_store_(
            SynchronizedShardedMap<int64_t, weight_type*>(
                num_shards_,
                block_size_,
                block_alignment_,
                /*blocks_per_chunk=*/8192)),
        elem_size_(row_storage_bitwidth / 8),
        backend_return_whole_row_(backend_return_whole_row),
        feature_evict_config_(std::move(feature_evict_config)),
        is_training_(is_training),
        enable_raw_embedding_streaming_(enable_raw_embedding_streaming) {
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(std::max<size_t>(
        num_threads, facebook::Proc::getCpuInfo().numCpuCores));
    // Dedicated executor for enrichment (low priority, won't affect
    // forward/backward). Only created when enrichment is configured.
    if (enrichment_config.has_value()) {
      enrichment_config_ = enrichment_config;
      enrichment_executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(4);
      // Pre-initialize LaserClient for IGR enrichment types
      if (enrichment_config_.value()->enrichment_type_ ==
              kv_mem::EnrichmentType::IGR_LASER_EMBEDDING ||
          enrichment_config_.value()->enrichment_type_ ==
              kv_mem::EnrichmentType::IGR_LASER_SID) {
        laser_client_ =
            igr_enrichment::initializeLaserClient(*enrichment_config_.value());
      }

      // Initialize OpenTab reader if type is ONEFLOW_OPENTAB_SID
      if (enrichment_config_.value()->enrichment_type_ ==
          kv_mem::EnrichmentType::ONEFLOW_OPENTAB_SID) {
        open_tab_reader_ = oneflow_enrichment::initializeOpenTabReader(
            *enrichment_config_.value());
      }
    }
    initialize_initializers(
        num_shards,
        max_D,
        uniform_init_lower,
        uniform_init_upper,
        row_storage_bitwidth,
        disable_random_init);
    if (table_dims.has_value()) {
      TORCH_CHECK_TENSOR_PROPERTIES(table_dims, at::ScalarType::Long);
      TORCH_NUM_CHECK_AND_ASSIGN_TENSOR_DATA(
          table_dims, sub_table_dims_, int64_t);
    }
    if (hash_size_cumsum.has_value()) {
      TORCH_CHECK_TENSOR_PROPERTIES(hash_size_cumsum, at::ScalarType::Long);
      sub_table_hash_cumsum_.assign(
          hash_size_cumsum->const_data_ptr<int64_t>() + 1, // skip the first 0
          hash_size_cumsum->const_data_ptr<int64_t>() +
              hash_size_cumsum->numel());
    }
    if (feature_evict_config_.has_value() &&
        feature_evict_config_.value()->trigger_mode_ !=
            EvictTriggerMode::DISABLED) {
      TORCH_CHECK(hash_size_cumsum.has_value());
      feature_evict_ = create_feature_evict(
          feature_evict_config_.value(),
          kv_store_,
          sub_table_hash_cumsum_,
          is_training_);
    }
  }

  void initialize_initializers(
      int64_t num_shards,
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t row_storage_bitwidth,
      bool disable_random_init) {
    for (auto i = 0; i < num_shards; ++i) {
      auto* gen = at::check_generator<at::CPUGeneratorImpl>(
          at::detail::getDefaultCPUGenerator());
      {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        initializers_.push_back(
            std::make_unique<ssd::Initializer>(
                gen->random64(),
                max_D,
                uniform_init_lower,
                uniform_init_upper,
                row_storage_bitwidth));
      }
    }
    disable_random_init_ = disable_random_init;
  }

  /// get all ids in the kvstore
  ///
  /// @return a Tensor contained ids
  at::Tensor get_keys_in_range_impl(
      int64_t start,
      int64_t end,
      std::optional<int64_t> offset = std::nullopt) override {
    std::vector<std::vector<int64_t>> ids(num_shards_);
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(num_shards_);
    for (int shard_id = 0; shard_id < num_shards_; shard_id++) {
      auto f = folly::via(executor_.get())
                   .thenValue([this, shard_id, start, end, offset, &ids](
                                  folly::Unit) {
                     auto rlmap = kv_store_.by(shard_id).rlock();
                     for (const auto& [key, value] : *rlmap) {
                       if (key >= start && key < end) {
                         ids[shard_id].push_back(offset ? key - *offset : key);
                       }
                     }
                   });
      futures.emplace_back(std::move(f));
    }
    folly::collect(futures).get();

    auto all_ids_ptr = std::make_shared<std::vector<int64_t>>();
    for (auto& sub : ids) {
      all_ids_ptr->insert(
          all_ids_ptr->end(),
          std::make_move_iterator(sub.begin()),
          std::make_move_iterator(sub.end()));
    }

    return torch::from_blob(
               all_ids_ptr->data(),
               {int64_t(all_ids_ptr->size())},
               [all_ids_ptr](void* p) mutable { all_ids_ptr.reset(); },
               torch::kInt64 // data type
               )
        .view({-1, 1});
  }

  /// Get eviction metadata given indices
  at::Tensor get_kv_zch_eviction_metadata_impl(
      const at::Tensor& indices,
      const at::Tensor& count) override {
    auto start_ts = facebook::WallClockUtil::NowInUsecFast();
    std::vector<folly::Future<std::tuple<int64_t, int64_t, int64_t>>> futures;
    auto numel = indices.size(0);
    read_metadata_load_size_ = numel;
    auto metadata_tensor = at::zeros({numel}, at::kLong);
    auto before_shard_ts = facebook::WallClockUtil::NowInUsecFast();
    auto shardid_to_indexes = shard_input(indices, count);
    read_metadata_sharding_total_duration_ +=
        facebook::WallClockUtil::NowInUsecFast() - before_shard_ts;
    for (const auto& [shard_id, indexes] : shardid_to_indexes) {
      futures.emplace_back(
          folly::via(executor_.get())
              .thenValue([this, shard_id, indexes, &indices, &metadata_tensor](
                             folly::Unit) {
                int64_t local_read_cache_hit_copy_total_duration = 0;
                int64_t local_read_lookup_cache_total_duration = 0;
                int64_t local_read_aquire_lock_duration = 0;
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kv_zch_eviction_metadata",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &metadata_tensor,
                     &local_read_cache_hit_copy_total_duration,
                     &local_read_lookup_cache_total_duration,
                     &local_read_aquire_lock_duration] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      auto indices_data_ptr = indices.data_ptr<index_t>();
                      auto* metadata =
                          metadata_tensor.mutable_data_ptr<int64_t>();
                      {
                        auto before_read_lock_ts =
                            facebook::WallClockUtil::NowInUsecFast();
                        auto rlmap = kv_store_.by(shard_id).rlock();
                        local_read_aquire_lock_duration =
                            facebook::WallClockUtil::NowInUsecFast() -
                            before_read_lock_ts;
                        for (const auto& id_index : indexes) {
                          auto id = int64_t(indices_data_ptr[id_index]);

                          // use mempool
                          weight_type* block = nullptr;
                          auto before_lookup_cache_ts =
                              facebook::WallClockUtil::NowInUsecFast();
                          auto it = rlmap->find(id);
                          local_read_lookup_cache_total_duration +=
                              facebook::WallClockUtil::NowInUsecFast() -
                              before_lookup_cache_ts;
                          // All ids should be found in backend to get metadata
                          CHECK(it != rlmap->end());
                          block = it->second;
                          auto before_cache_hit_copy_ts =
                              facebook::WallClockUtil::NowInUsecFast();
                          metadata[id_index] =
                              FixedBlockPool::get_metaheader_raw(block);
                          local_read_cache_hit_copy_total_duration +=
                              facebook::WallClockUtil::NowInUsecFast() -
                              before_cache_hit_copy_ts;
                        }
                      }
                    });
                return std::tuple{
                    local_read_lookup_cache_total_duration,
                    local_read_cache_hit_copy_total_duration,
                    local_read_aquire_lock_duration};
              }));
    }
    auto results = folly::collectAll(futures).get();

    for (const auto& result : results) {
      if (result.hasValue()) {
        auto [lookup_cache_dur, cache_hit_copy_dur, acquire_lock_dur] =
            result.value();
        read_metadata_lookup_cache_total_avg_duration_ +=
            lookup_cache_dur / num_shards_;
        read_metadata_cache_hit_copy_avg_duration_ +=
            cache_hit_copy_dur / num_shards_;
        read_metadata_acquire_lock_avg_duration_ +=
            acquire_lock_dur / num_shards_;
      }
    }
    read_metadata_total_duration_ +=
        facebook::WallClockUtil::NowInUsecFast() - start_ts;
    return metadata_tensor;
  }

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
          kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ /*unused*/) override {
    auto start_ts = facebook::WallClockUtil::NowInUsecFast();
    pause_ongoing_eviction();
    std::vector<
        folly::Future<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>>>
        futures;
    auto before_shard_ts = facebook::WallClockUtil::NowInUsecFast();
    auto shardid_to_indexes = shard_input(indices, count);
    write_sharding_total_duration_ +=
        facebook::WallClockUtil::NowInUsecFast() - before_shard_ts;

    for (const auto& [shard_id, indexes] : shardid_to_indexes) {
      futures.emplace_back(
          folly::via(executor_.get())
              .thenValue([this, shard_id, indexes, &indices, &weights](
                             folly::Unit) {
                int64_t local_write_allocate_total_duration = 0;
                int64_t local_write_cache_copy_total_duration = 0;
                int64_t local_write_lookup_cache_total_duration = 0;
                int64_t local_write_acquire_lock_duration = 0;
                int64_t local_write_missing_load = 0;
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kv_set",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &weights,
                     &local_write_allocate_total_duration,
                     &local_write_cache_copy_total_duration,
                     &local_write_lookup_cache_total_duration,
                     &local_write_acquire_lock_duration,
                     &local_write_missing_load] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      CHECK(weights.is_contiguous());
                      CHECK_EQ(indices.size(0), weights.size(0));
                      int64_t stride = weights.size(1);
                      auto indices_data_ptr = indices.const_data_ptr<index_t>();
                      auto weights_data_ptr = weights.data_ptr<weight_type>();
                      {
                        auto before_write_lock_ts =
                            facebook::WallClockUtil::NowInUsecFast();
                        auto wlmap = kv_store_.by(shard_id).wlock();
                        local_write_acquire_lock_duration =
                            facebook::WallClockUtil::NowInUsecFast() -
                            before_write_lock_ts;
                        auto* pool = kv_store_.pool_by(shard_id);
                        for (const auto& id_index : indexes) {
                          auto id = int64_t(indices_data_ptr[id_index]);
                          // use mempool
                          weight_type* block = nullptr;
                          // First check if the key already exists
                          auto before_lookup_cache_ts =
                              facebook::WallClockUtil::NowInUsecFast();
                          auto it = wlmap->find(id);
                          local_write_lookup_cache_total_duration +=
                              facebook::WallClockUtil::NowInUsecFast() -
                              before_lookup_cache_ts;
                          if (it != wlmap->end()) {
                            block = it->second;
                          } else {
                            // Key doesn't exist, allocate new block and insert.
                            local_write_missing_load++;
                            auto before_alloc_ts =
                                facebook::WallClockUtil::NowInUsecFast();
                            block = pool->template allocate_t<weight_type>();
                            FixedBlockPool::set_key(block, id);
                            wlmap->insert({id, block});
                            local_write_allocate_total_duration +=
                                facebook::WallClockUtil::NowInUsecFast() -
                                before_alloc_ts;
                            if (feature_evict_config_.has_value() &&
                                feature_evict_config_.value()->trigger_mode_ !=
                                    EvictTriggerMode::DISABLED &&
                                feature_evict_) {
                              auto* feature_score_evict = dynamic_cast<
                                  FeatureScoreBasedEvict<weight_type>*>(
                                  feature_evict_.get());
                              if (feature_score_evict) {
                                feature_score_evict
                                    ->update_feature_score_statistics(
                                        block, 0, shard_id, true);
                              }
                            }
                          }
                          if (feature_evict_config_.has_value() &&
                              feature_evict_config_.value()->trigger_mode_ !=
                                  EvictTriggerMode::DISABLED &&
                              feature_evict_) {
                            feature_evict_->update_feature_statistics(block);
                          }
                          auto before_copy_ts =
                              facebook::WallClockUtil::NowInUsecFast();
                          auto* data_ptr =
                              FixedBlockPool::data_ptr<weight_type>(block);
                          std::copy(
                              weights_data_ptr + id_index * stride,
                              weights_data_ptr + (id_index + 1) * stride,
                              data_ptr);
                          local_write_cache_copy_total_duration +=
                              facebook::WallClockUtil::NowInUsecFast() -
                              before_copy_ts;
                        }
                      }
                    });
                return std::tuple{
                    local_write_allocate_total_duration,
                    local_write_cache_copy_total_duration,
                    local_write_lookup_cache_total_duration,
                    local_write_acquire_lock_duration,
                    local_write_missing_load};
              }));
    }
    return folly::collect(std::move(futures))
        .via(executor_.get())
        .thenValue([this, start_ts, w_mode](
                       const std::vector<std::tuple<
                           int64_t,
                           int64_t,
                           int64_t,
                           int64_t,
                           int64_t>>& results) {
          int64_t write_allocate_total_duration = 0;
          int64_t write_cache_copy_total_duration = 0;
          int64_t write_lookup_cache_total_duration = 0;
          int64_t write_acquire_lock_total_duration = 0;
          int64_t write_missing_load = 0;
          for (
              const auto& [allocate_dur, cache_copy_dur, lookup_cache_dur, acquire_lock_dur, missing_load] :
              results) {
            write_allocate_total_duration += allocate_dur;
            write_cache_copy_total_duration += cache_copy_dur;
            write_lookup_cache_total_duration += lookup_cache_dur;
            write_acquire_lock_total_duration += acquire_lock_dur;
            write_missing_load += missing_load;
          }
          auto duration = facebook::WallClockUtil::NowInUsecFast() - start_ts;
          switch (w_mode) {
            case kv_db::RocksdbWriteMode::BWD_L1_CNFLCT_MISS_WRITE_BACK:
              bwd_l1_cnflct_miss_write_total_duration_ += duration;
              bwd_l1_cnflct_miss_write_allocate_avg_duration_ +=
                  write_allocate_total_duration / num_shards_;
              bwd_l1_cnflct_miss_write_cache_copy_avg_duration_ +=
                  write_cache_copy_total_duration / num_shards_;
              bwd_l1_cnflct_miss_write_lookup_cache_avg_duration_ +=
                  write_lookup_cache_total_duration / num_shards_;
              bwd_l1_cnflct_miss_write_acquire_lock_avg_duration_ +=
                  write_acquire_lock_total_duration / num_shards_;
              bwd_l1_cnflct_miss_write_missing_load_avg_ +=
                  write_missing_load / num_shards_;
              break;
            case kv_db::RocksdbWriteMode::FWD_L1_EVICTION:
              fwd_l1_eviction_write_total_duration_ += duration;
              fwd_l1_eviction_write_allocate_avg_duration_ +=
                  write_allocate_total_duration / num_shards_;
              fwd_l1_eviction_write_cache_copy_avg_duration_ +=
                  write_cache_copy_total_duration / num_shards_;
              fwd_l1_eviction_write_lookup_cache_avg_duration_ +=
                  write_lookup_cache_total_duration / num_shards_;
              fwd_l1_eviction_write_acquire_lock_avg_duration_ +=
                  write_acquire_lock_total_duration / num_shards_;
              fwd_l1_eviction_write_missing_load_avg_ +=
                  write_missing_load / num_shards_;
              break;
            case kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ:
              break;
            case kv_db::RocksdbWriteMode::FLUSH:
              break;
            case kv_db::RocksdbWriteMode::STREAM:
              break;
          }
          return std::vector<folly::Unit>(results.size());
        });
  }

  /// Separate write path for enrichment that runs on enrichment_executor_
  /// instead of the main executor_. This avoids competing with forward/backward
  /// for thread pool resources. Uses pause/resume to yield to fwd/bwd —
  /// removing this and reusing the main write path causes QPS regression.
  folly::SemiFuture<std::vector<folly::Unit>>
  set_kv_db_async_on_enrichment_executor(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) {
    auto start_ts = facebook::WallClockUtil::NowInUsecFast();
    std::vector<folly::Future<folly::Unit>> futures;
    auto shardid_to_indexes = shard_input(indices, count);

    for (auto iter = shardid_to_indexes.begin();
         iter != shardid_to_indexes.end();
         iter++) {
      const auto shard_id = iter->first;
      const auto indexes = iter->second;
      futures.emplace_back(
          folly::via(enrichment_executor_.get())
              .thenValue([this, shard_id, indexes, indices, weights](
                             folly::Unit) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kv_set_laser",
                    [this, shard_id, indexes, &indices, &weights] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      CHECK(weights.is_contiguous());
                      CHECK_EQ(indices.size(0), weights.size(0));
                      int64_t stride = weights.size(1);
                      auto indices_data_ptr = indices.data_ptr<index_t>();
                      auto weights_data_ptr = weights.data_ptr<weight_type>();

                      // Two-level loop like eviction pattern:
                      // Outer loop: check pause/resume
                      // Inner loop: process batch while holding lock
                      size_t cursor = 0;
                      size_t total = indexes.size();
                      while (cursor < total) {
                        // Wait until resume if paused (like eviction pattern)
                        wait_until_laser_write_resume();

                        // Process batch while holding lock
                        {
                          auto wlmap = kv_store_.by(shard_id).wlock();
                          auto* pool = kv_store_.pool_by(shard_id);
                          while (cursor < total) {
                            const auto& id_index = indexes[cursor];
                            auto id = int64_t(indices_data_ptr[id_index]);
                            weight_type* block = nullptr;
                            auto it = wlmap->find(id);
                            if (it != wlmap->end()) {
                              block = it->second;
                            } else {
                              // Key doesn't exist, allocate new block and
                              // insert
                              block = pool->template allocate_t<weight_type>();
                              if (block == nullptr) {
                                cursor++;
                                continue; // Skip if allocation fails
                              }
                              FixedBlockPool::set_key(block, id);
                              wlmap->insert({id, block});
                            }
                            auto* data_ptr =
                                FixedBlockPool::data_ptr<weight_type>(block);
                            std::copy(
                                weights_data_ptr + id_index * stride,
                                weights_data_ptr + (id_index + 1) * stride,
                                data_ptr);
                            cursor++;
                            // Check if we should pause and yield lock
                            if (is_laser_write_interrupted()) {
                              break;
                            }
                          }
                        }
                        // Lock released here, forward/backward can proceed
                      }
                    });
                return folly::Unit{};
              }));
    }
    return folly::collect(std::move(futures))
        .via(enrichment_executor_.get())
        .thenValue([start_ts](const std::vector<folly::Unit>& results) {
          auto latency_ms =
              (facebook::WallClockUtil::NowInUsecFast() - start_ts) / 1000;
          XLOG(INFO)
              << "[EmbeddingCacheEnrich] set_kv_db_async_on_enrichment_executor "
              << "completed, latency_ms=" << latency_ms;
          return results;
        });
  }

  folly::SemiFuture<std::vector<folly::Unit>> inference_set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      std::optional<uint32_t> inplace_update_ts) {
    auto shardid_to_indexes = shard_input(indices, count);
    std::vector<folly::Future<std::tuple<int64_t, int64_t>>> futures;
    futures.reserve(shardid_to_indexes.size());

    auto* tt_evict = dynamic_cast<TimeThresholdBasedEvict<weight_type>*>(
        feature_evict_.get());
    CHECK(tt_evict != nullptr);
    for (const auto& [shard_id, indexes] : shardid_to_indexes) {
      auto f =
          folly::via(executor_.get())
              .thenValue([this,
                          shard_id,
                          indexes,
                          &indices,
                          &weights,
                          tt_evict,
                          inplace_update_ts](folly::Unit) {
                int64_t hit_cnt = 0;
                int64_t miss_cnt = 0;
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "inference_dram_kv_set",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &weights,
                     tt_evict,
                     inplace_update_ts,
                     &hit_cnt,
                     &miss_cnt] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      CHECK(weights.is_contiguous());
                      CHECK_EQ(indices.size(0), weights.size(0));
                      int64_t stride = weights.size(1);
                      auto indices_data_ptr = indices.data_ptr<index_t>();
                      auto weights_data_ptr = weights.data_ptr<weight_type>();
                      {
                        // 1st step, collect hit/miss per inplace update chunk
                        // [tensor_offset, weight_addr]
                        std::vector<std::tuple<int64_t, weight_type*>> hit_info;
                        // [id, tensor_offset]
                        std::vector<std::tuple<int64_t, int64_t>> miss_info;
                        hit_info.reserve(indexes.size() / 2);
                        miss_info.reserve(indexes.size() / 10);
                        auto rlmap = kv_store_.by(shard_id).rlock();
                        for (const auto& idx : indexes) {
                          auto id = int64_t(indices_data_ptr[idx]);
                          auto it = rlmap->find(id);
                          if (it != rlmap->end()) {
                            hit_info.emplace_back(idx, it->second);
                          } else {
                            miss_info.emplace_back(id, idx);
                          }
                        }
                        rlmap.unlock();
                        hit_cnt = hit_info.size();
                        miss_cnt = miss_info.size();
                        // 2nd step, no lock on update hits, it is possible that
                        // inference read is accessing a weight being updated,
                        // we assume it is fine for now, will iterate on it if
                        // we find QE regress during inplace update
                        for (auto& [tensor_offset, block] : hit_info) {
                          auto* data_ptr =
                              FixedBlockPool::data_ptr<weight_type>(block);
                          std::copy(
                              weights_data_ptr + tensor_offset * stride,
                              weights_data_ptr + (tensor_offset + 1) * stride,
                              data_ptr);
                          // update provided ts for existing blocks
                          if (feature_evict_config_.has_value() &&
                              feature_evict_config_.value()->trigger_mode_ !=
                                  EvictTriggerMode::DISABLED &&
                              feature_evict_ && inplace_update_ts.has_value()) {
                            FixedBlockPool::set_timestamp(
                                block, inplace_update_ts.value());
                          }
                        }

                        // 3rd step, update misses in fixed block pool, we only
                        // need mempool lock at this stage to avoid race
                        // condition with eviction, if any, this helps reduce
                        // the blocking time for inference read

                        std::unordered_map<int64_t, weight_type*> temp_kv;
                        auto* pool = kv_store_.pool_by(shard_id);
                        auto mem_pool_lock = pool->acquire_lock();
                        for (auto& [id, tensor_offset] : miss_info) {
                          auto block = pool->template allocate_t<weight_type>();
                          FixedBlockPool::set_key(block, id);
                          temp_kv.insert({id, block});

                          auto* data_ptr =
                              FixedBlockPool::data_ptr<weight_type>(block);
                          std::copy(
                              weights_data_ptr + tensor_offset * stride,
                              weights_data_ptr + (tensor_offset + 1) * stride,
                              data_ptr);

                          // update provided ts for new allocated blocks
                          if (feature_evict_config_.has_value() &&
                              feature_evict_config_.value()->trigger_mode_ !=
                                  EvictTriggerMode::DISABLED &&
                              feature_evict_) {
                            if (inplace_update_ts.has_value()) {
                              FixedBlockPool::set_timestamp(
                                  block, inplace_update_ts.value());
                            } else {
                              // inplace_update_ts is nullopt for delta publish
                              // update
                              tt_evict->update_feature_statistics(block);
                            }
                          }
                        }
                        mem_pool_lock.unlock();

                        // 4th step, update shard hmap with newly allocated info
                        // this is blocking read, by separating it out from
                        // original set_kv_db_async, we can reduce the blocking
                        // time for inference read significantly
                        auto wlmap = kv_store_.by(shard_id).wlock();
                        for (auto& [id, block] : temp_kv) {
                          wlmap->insert({id, block});
                        }
                        wlmap.unlock();
                      }
                    });
                return std::tuple{hit_cnt, miss_cnt};
              });
      futures.emplace_back(std::move(f));
    }
    return folly::collect(std::move(futures))
        .via(executor_.get())
        .thenValue(
            [this](const std::vector<std::tuple<int64_t, int64_t>>& tuples) {
              for (const auto& [hit_cnt, miss_cnt] : tuples) {
                inplace_update_hit_cnt_ += hit_cnt;
                inplace_update_miss_cnt_ += miss_cnt;
              }
              return std::vector<folly::Unit>(tuples.size());
            });
  }

  /// Helper: schedule a fetch+prepare+write coroutine on enrichment_executor_.
  /// Eliminates boilerplate duplication across enrichment type branches.
  /// FetchFn: (const vector<int64_t>&) -> Task<PayloadMap>
  /// PrepareFn: (const vector<int64_t>&, const vector<int64_t>&,
  ///             const PayloadMap&) -> optional<EnrichmentResult>
  template <typename FetchFn, typename PrepareFn>
  void dispatchEnrichmentAsync(
      std::vector<int64_t> hashed_ids,
      std::vector<int64_t> unhashed_ids,
      const char* log_prefix,
      FetchFn fetchFn,
      PrepareFn prepareFn) {
    folly::coro::co_invoke(
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-capturing-lambda-coroutines)
        [this,
         hashed_ids = std::move(hashed_ids),
         unhashed_ids = std::move(unhashed_ids),
         log_prefix,
         fetchFn = std::move(fetchFn),
         prepareFn =
             std::move(prepareFn)]() mutable -> folly::coro::Task<void> {
          auto start_time = facebook::WallClockUtil::NowInUsecFast();
          auto payloads = co_await fetchFn(unhashed_ids);
          auto latency_ms =
              (facebook::WallClockUtil::NowInUsecFast() - start_time) / 1000;
          XLOG(INFO) << "[EmbeddingCacheEnrich] " << log_prefix
                     << payloads.size() << "/" << unhashed_ids.size()
                     << ", latency_ms: " << latency_ms;
          enrichment_query_count_.fetch_add(unhashed_ids.size());
          if (unhashed_ids.size() >= payloads.size()) {
            enrichment_empty_count_.fetch_add(
                unhashed_ids.size() - payloads.size());
          } else {
            XLOG(WARN) << "[EmbeddingCacheEnrich] " << log_prefix
                       << "payloads.size() (" << payloads.size()
                       << ") > unhashed_ids.size() (" << unhashed_ids.size()
                       << ")";
          }
          if (!payloads.empty()) {
            auto result = prepareFn(hashed_ids, unhashed_ids, payloads);
            if (result.has_value()) {
              set_kv_db_async_on_enrichment_executor(
                  result->indices, result->weights, result->count)
                  .via(enrichment_executor_.get())
                  .thenValue([this, result](const std::vector<folly::Unit>&) {
                    // Stream enriched embeddings to TrainingPsProcess for
                    // delta publishing.
                    if (enable_raw_embedding_streaming_) {
                      raw_embedding_streamer_->stream(
                          result->indices,
                          result->weights,
                          std::nullopt, /*identities*/
                          std::nullopt, /*runtime_meta*/
                          result->count,
                          false, /*require_tensor_copy - tensors already on
                                    CPU*/
                          false /*blocking_tensor_copy*/);
                    }
                  });
            }
          }
          pending_laser_requests_.fetch_sub(1);
          laser_write_in_progress_.store(false);
        })
        // NOLINTNEXTLINE(readability-redundant-smartptr-get)
        .scheduleOn(enrichment_executor_.get())
        .start();
  }

  /// Decode int64 SID from cache weights.
  /// Reverse of prepareInt64PayloadTensors dibit encoding:
  /// 32 dibits -> uint64 -> int64
  /// dibit d encodes as float: {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0}
  /// Decode: dibit = round(fval * 2 - 1)
  static int64_t decodeSIDFromWeights(const weight_type* weights_ptr) {
    uint64_t uval = 0;
    for (int i = 0; i < 32; ++i) {
      float fval = static_cast<float>(weights_ptr[i]);
      uint8_t dibit =
          static_cast<uint8_t>(std::lround(fval * 2.0f - 1.0f)) & 0x3;
      uval |= (static_cast<uint64_t>(dibit) << (i * 2));
    }
    return static_cast<int64_t>(uval);
  }

  /// Sync fetch SIDs for publish (OneFlow).
  /// Cache-first: reads SIDs from kv_store if available,
  /// only calls remote (FeatureStore/OpenTab) for cache misses.
  /// No CAS/drop mechanism, blocks until complete.
  /// @return (vids_tensor, sids_tensor) both int64
  std::tuple<at::Tensor, at::Tensor> fetch_sids_sync(
      at::Tensor hashed_indices,
      at::Tensor unhashed_indices,
      at::Tensor count) override {
    if (!enrichment_config_.has_value()) {
      return {
          torch::empty({0}, torch::kInt64), torch::empty({0}, torch::kInt64)};
    }

    const int64_t num_ids = count.item<int64_t>();
    if (num_ids == 0) {
      return {
          torch::empty({0}, torch::kInt64), torch::empty({0}, torch::kInt64)};
    }

    CHECK(hashed_indices.is_contiguous());
    CHECK(unhashed_indices.is_contiguous());
    CHECK_EQ(hashed_indices.size(0), unhashed_indices.size(0));

    const auto* hashed_ptr = hashed_indices.data_ptr<int64_t>();
    const auto* unhashed_ptr = unhashed_indices.data_ptr<int64_t>();

    // ─── Step 1: Shard scan ───
    // Cache hit + weights non-zero → decode SID from weights
    // Cache miss or zero-weight → collect for remote fetch
    std::vector<int64_t> result_vids, result_sids;
    std::vector<int64_t> need_fetch_unhashed;

    auto shardid_to_indexes = shard_input(hashed_indices, count);
    for (auto iter = shardid_to_indexes.begin();
         iter != shardid_to_indexes.end();
         iter++) {
      const auto shard_id = iter->first;
      const auto& indexes = iter->second;
      if (indexes.empty()) {
        continue;
      }
      auto rlmap = kv_store_.by(shard_id).rlock();
      for (const auto& id_index : indexes) {
        auto hashed_id = hashed_ptr[id_index];
        auto unhashed_id = unhashed_ptr[id_index];
        auto it = rlmap->find(hashed_id);
        if (it != rlmap->end()) {
          weight_type* block = it->second;
          if (!FixedBlockPool::is_weights_all_zero(block, block_size_)) {
            // Cache hit — decode SID
            auto* data_ptr = FixedBlockPool::data_ptr<weight_type>(block);
            int64_t sid = decodeSIDFromWeights(data_ptr);
            if (sid != 0 && sid != -1) {
              result_vids.push_back(unhashed_id);
              result_sids.push_back(sid);
              continue;
            }
          }
        }
        // Cache miss or zero-weight or invalid SID → need remote fetch
        need_fetch_unhashed.push_back(unhashed_id);
      }
    }

    XLOG(INFO) << "[fetch_sids_sync] cache hits: " << result_vids.size()
               << ", need fetch: " << need_fetch_unhashed.size();

    // ─── Step 2: Remote fetch for misses ───
    if (!need_fetch_unhashed.empty()) {
      const auto enrichment_type = enrichment_config_.value()->enrichment_type_;
      folly::F14FastMap<int64_t, int64_t> payloads;

      if (enrichment_type == kv_mem::EnrichmentType::ONEFLOW_OPENTAB_SID) {
        payloads = folly::coro::blockingWait(
            oneflow_enrichment::fetchFromOpenTab(
                open_tab_reader_,
                *enrichment_config_.value(),
                need_fetch_unhashed)
                .scheduleOn(enrichment_executor_.get()));
      } else if (
          enrichment_type ==
          kv_mem::EnrichmentType::ONEFLOW_FEATURE_STORE_SID) {
        payloads = folly::coro::blockingWait(
            feature_store_enrichment::fetchSIDFromFeatureStore(
                *enrichment_config_.value(), need_fetch_unhashed)
                .scheduleOn(enrichment_executor_.get()));
      } else {
        XLOG(WARNING) << "[fetch_sids_sync] unsupported enrichment_type: "
                      << static_cast<int64_t>(enrichment_type);
      }

      XLOG(INFO) << "[fetch_sids_sync] remote fetched: " << payloads.size()
                 << "/" << need_fetch_unhashed.size();

      for (const auto& [vid, sid] : payloads) {
        if (sid != 0 && sid != -1) {
          result_vids.push_back(vid);
          result_sids.push_back(sid);
        }
      }
    }

    // ─── Step 3: Convert to tensors ───
    if (result_vids.empty()) {
      return {
          torch::empty({0}, torch::kInt64), torch::empty({0}, torch::kInt64)};
    }

    auto vids_tensor = torch::from_blob(
                           result_vids.data(),
                           {static_cast<int64_t>(result_vids.size())},
                           torch::kInt64)
                           .clone();
    auto sids_tensor = torch::from_blob(
                           result_sids.data(),
                           {static_cast<int64_t>(result_sids.size())},
                           torch::kInt64)
                           .clone();

    XLOG(INFO) << "[fetch_sids_sync] returning " << result_vids.size()
               << " VID->SID pairs";

    return {vids_tensor, sids_tensor};
  }

  void set_embedding_cache_enrich_query_id_async(
      at::Tensor hashed_indices,
      at::Tensor unhashed_indices,
      at::Tensor count) override {
    if (!enrichment_config_.has_value()) {
      return;
    }
    // Drop mechanism: atomically check-and-set to ensure only one request
    // proceeds. This prevents task accumulation and keeps QPS stable.
    // Using compare_exchange_strong to atomically set flag at function entry,
    // not inside lambda (which would be too late and cause race condition).
    bool expected = false;
    if (!laser_write_in_progress_.compare_exchange_strong(expected, true)) {
      XLOG(INFO)
          << "[EmbeddingCacheEnrich] skipping - laser write already in progress";
      return;
    }
    // Now laser_write_in_progress_ = true, other requests will be dropped

    // Fire and forget - run entire operation in background using dedicated
    // enrichment_executor_ This ensures Laser enrichment doesn't compete with
    // forward/backward for threads
    folly::via(
        enrichment_executor_.get(),
        [this,
         hashed_indices = std::move(hashed_indices),
         unhashed_indices = std::move(unhashed_indices),
         count = std::move(count)]() {
          std::vector<folly::Future<
              std::tuple<int64_t, std::vector<int64_t>, std::vector<int64_t>>>>
              futures;
          auto shardid_to_indexes = shard_input(hashed_indices, count);
          for (auto iter = shardid_to_indexes.begin();
               iter != shardid_to_indexes.end();
               iter++) {
            const auto shard_id = iter->first;
            const auto indexes = iter->second;
            auto f =
                folly::via(enrichment_executor_.get())
                    .thenValue([this,
                                shard_id,
                                indexes,
                                hashed_indices,
                                unhashed_indices](folly::Unit) {
                      int64_t zero_id_count = 0;
                      std::vector<int64_t> zero_weight_hashed_ids;
                      std::vector<int64_t> zero_weight_unhashed_ids;
                      FBGEMM_DISPATCH_INTEGRAL_TYPES(
                          hashed_indices.scalar_type(),
                          "dram_set_embedding_cache_enrich_query_id",
                          [this,
                           shard_id,
                           indexes,
                           hashed_indices,
                           unhashed_indices,
                           &zero_id_count,
                           &zero_weight_hashed_ids,
                           &zero_weight_unhashed_ids] {
                            using index_t = scalar_t;
                            CHECK(hashed_indices.is_contiguous());
                            CHECK(unhashed_indices.is_contiguous());
                            CHECK_EQ(
                                hashed_indices.size(0),
                                unhashed_indices.size(0));
                            auto hashed_indices_data_ptr =
                                hashed_indices.data_ptr<index_t>();
                            auto unhashed_indices_data_ptr =
                                unhashed_indices.data_ptr<index_t>();
                            int64_t in_cache_zero_count = 0;
                            int64_t not_in_cache_count = 0;
                            {
                              auto rlmap = kv_store_.by(shard_id).rlock();

                              for (auto index_iter = indexes.begin();
                                   index_iter != indexes.end();
                                   index_iter++) {
                                const auto& id_index = *index_iter;
                                auto hashed_id =
                                    int64_t(hashed_indices_data_ptr[id_index]);
                                auto unhashed_id = int64_t(
                                    unhashed_indices_data_ptr[id_index]);
                                auto it = rlmap->find(hashed_id);
                                if (it != rlmap->end()) {
                                  weight_type* block = it->second;
                                  if (FixedBlockPool::is_weights_all_zero(
                                          block, block_size_)) {
                                    zero_weight_hashed_ids.push_back(hashed_id);
                                    zero_weight_unhashed_ids.push_back(
                                        unhashed_id);
                                    zero_id_count++;
                                    in_cache_zero_count++;
                                  }
                                } else {
                                  // Id not in kv_store (was filled from
                                  // row_storage in get_kv_db_async)
                                  zero_weight_hashed_ids.push_back(hashed_id);
                                  zero_weight_unhashed_ids.push_back(
                                      unhashed_id);
                                  zero_id_count++;
                                  not_in_cache_count++;
                                }
                              }
                            }
                            if (shard_id == 0) {
                              XLOG(INFO)
                                  << "[EmbeddingCacheEnrich] shard_0: "
                                  << "total=" << indexes.size()
                                  << ", in_cache_zero=" << in_cache_zero_count
                                  << ", not_in_cache=" << not_in_cache_count;
                            }
                          });
                      return std::make_tuple(
                          zero_id_count,
                          std::move(zero_weight_hashed_ids),
                          std::move(zero_weight_unhashed_ids));
                    });
            futures.push_back(std::move(f));
          }
          folly::collect(std::move(futures))
              .via(enrichment_executor_.get())
              .thenValue([this](
                             const std::vector<std::tuple<
                                 int64_t,
                                 std::vector<int64_t>,
                                 std::vector<int64_t>>>& results) {
                // Aggregate results from all shards
                std::vector<int64_t> all_zero_weight_hashed_ids;
                std::vector<int64_t> all_zero_weight_unhashed_ids;
                for (const auto& result : results) {
                  const auto& hashed_ids = std::get<1>(result);
                  const auto& unhashed_ids = std::get<2>(result);
                  all_zero_weight_hashed_ids.insert(
                      all_zero_weight_hashed_ids.end(),
                      hashed_ids.begin(),
                      hashed_ids.end());
                  all_zero_weight_unhashed_ids.insert(
                      all_zero_weight_unhashed_ids.end(),
                      unhashed_ids.begin(),
                      unhashed_ids.end());
                }

                XLOG(INFO) << "[EmbeddingCacheEnrich] found "
                           << all_zero_weight_unhashed_ids.size()
                           << " zero_weight_ids, enrichment_type: "
                           << static_cast<int64_t>(
                                  enrichment_config_.value()->enrichment_type_)
                           << ", pending_laser_requests: "
                           << pending_laser_requests_.load();

                // Skip if too many pending requests (avoid overwhelming
                // the external source)
                constexpr int64_t kMaxPendingLaserRequests = 2;
                if (pending_laser_requests_.load() >=
                    kMaxPendingLaserRequests) {
                  XLOG(INFO)
                      << "[EmbeddingCacheEnrich] skipping fetch, too many pending requests: "
                      << pending_laser_requests_.load();
                  laser_write_in_progress_.store(false);
                  return;
                }

                // Fetch from external source for zero-weight IDs
                if (!all_zero_weight_unhashed_ids.empty() &&
                    enrichment_config_.has_value()) {
                  XLOG(INFO) << "[EmbeddingCacheEnrich] starting fetch for "
                             << all_zero_weight_unhashed_ids.size() << " IDs";

                  pending_laser_requests_.fetch_add(1);

                  // Dispatch to model-specific enrichment handler
                  const auto enrichment_type =
                      enrichment_config_.value()->enrichment_type_;

                  if (enrichment_type ==
                      kv_mem::EnrichmentType::IGR_LASER_EMBEDDING) {
                    dispatchEnrichmentAsync(
                        std::move(all_zero_weight_hashed_ids),
                        std::move(all_zero_weight_unhashed_ids),
                        "laser_hit: ",
                        [this](const std::vector<int64_t>& ids) {
                          return igr_enrichment::fetchEmbeddingsFromLaser(
                              laser_client_, *enrichment_config_.value(), ids);
                        },
                        [this](
                            const std::vector<int64_t>& h,
                            const std::vector<int64_t>& u,
                            const auto& data) {
                          return igr_enrichment::prepareCacheWriteTensors<
                              weight_type>(h, u, data, max_D_);
                        });
                  } else if (
                      enrichment_type ==
                      kv_mem::EnrichmentType::IGR_LASER_SID) {
                    dispatchEnrichmentAsync(
                        std::move(all_zero_weight_hashed_ids),
                        std::move(all_zero_weight_unhashed_ids),
                        "sid_hit: ",
                        [this](const std::vector<int64_t>& ids) {
                          return igr_enrichment::fetchSIDsFromLaser(
                              laser_client_, *enrichment_config_.value(), ids);
                        },
                        [this](
                            const std::vector<int64_t>& h,
                            const std::vector<int64_t>& u,
                            const auto& data) {
                          return igr_enrichment::prepareSIDCacheWriteTensors(
                              h, u, data, max_D_);
                        });
                  } else if (
                      enrichment_type ==
                      kv_mem::EnrichmentType::ONEFLOW_OPENTAB_SID) {
                    dispatchEnrichmentAsync(
                        std::move(all_zero_weight_hashed_ids),
                        std::move(all_zero_weight_unhashed_ids),
                        "opentab_hit: ",
                        [this](const std::vector<int64_t>& ids) {
                          return oneflow_enrichment::fetchFromOpenTab(
                              open_tab_reader_,
                              *enrichment_config_.value(),
                              ids);
                        },
                        [this](
                            const std::vector<int64_t>& h,
                            const std::vector<int64_t>& u,
                            const auto& data) {
                          return oneflow_enrichment::prepareInt64PayloadTensors<
                              weight_type>(h, u, data, max_D_);
                        });
                  } else if (
                      enrichment_type ==
                      kv_mem::EnrichmentType::ONEFLOW_FEATURE_STORE_SID) {
                    dispatchEnrichmentAsync(
                        std::move(all_zero_weight_hashed_ids),
                        std::move(all_zero_weight_unhashed_ids),
                        "feature_store_hit: ",
                        [this](const std::vector<int64_t>& ids) {
                          return feature_store_enrichment::
                              fetchSIDFromFeatureStore(
                                  *enrichment_config_.value(), ids);
                        },
                        [this](
                            const std::vector<int64_t>& h,
                            const std::vector<int64_t>& u,
                            const auto& data) {
                          return oneflow_enrichment::prepareInt64PayloadTensors<
                              weight_type>(h, u, data, max_D_);
                        });
                  } else {
                    XLOG(WARN)
                        << "[EmbeddingCacheEnrich] unknown enrichment_type: "
                        << static_cast<int64_t>(enrichment_type);
                    pending_laser_requests_.fetch_sub(1);
                    laser_write_in_progress_.store(false);
                  }
                } else {
                  // No zero-weight IDs or no laser providers, clear flag
                  laser_write_in_progress_.store(false);
                }
              });
        });
  }

  /// Update feature scores metadata into kvstore.
  folly::SemiFuture<std::vector<folly::Unit>>
  set_kv_zch_eviction_metadata_async(
      at::Tensor indices,
      at::Tensor count,
      at::Tensor engege_rates) override {
    auto start_ts = facebook::WallClockUtil::NowInUsecFast();
    if (!feature_evict_ || !feature_evict_config_.has_value() ||
        feature_evict_config_.value()->trigger_mode_ ==
            EvictTriggerMode::DISABLED) {
      // featre eviction is disabled
      auto duration = facebook::WallClockUtil::NowInUsecFast() - start_ts;
      metadata_write_total_duration_ += duration;
      return folly::makeSemiFuture(std::vector<folly::Unit>());
    }

    CHECK_EQ(engege_rates.scalar_type(), at::ScalarType::Float);
    auto* feature_score_evict =
        dynamic_cast<FeatureScoreBasedEvict<weight_type>*>(
            feature_evict_.get());

    if (feature_score_evict == nullptr) {
      auto duration = facebook::WallClockUtil::NowInUsecFast() - start_ts;
      metadata_write_total_duration_ += duration;
      // Not a feature score based eviction
      return folly::makeSemiFuture(std::vector<folly::Unit>());
    }
    pause_ongoing_eviction();
    auto before_shard_ts = facebook::WallClockUtil::NowInUsecFast();
    auto shardid_to_indexes = shard_input(indices, count);
    metadata_write_sharding_total_duration_ +=
        facebook::WallClockUtil::NowInUsecFast() - before_shard_ts;
    std::vector<
        folly::Future<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>>>
        futures;

    for (const auto& [shard_id, indexes] : shardid_to_indexes) {
      futures.emplace_back(
          folly::via(executor_.get())
              .thenValue([this,
                          shard_id,
                          indexes,
                          indices,
                          engege_rates,
                          feature_score_evict](folly::Unit) {
                int64_t updated_id_count = 0;
                int64_t local_write_allocate_total_duration = 0;
                int64_t local_write_lookup_cache_total_duration = 0;
                int64_t local_write_acquire_lock_duration = 0;
                int64_t local_write_cache_miss = 0;
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_set_kv_feature_score_metadata",
                    [this,
                     shard_id,
                     indexes,
                     indices,
                     engege_rates,
                     feature_score_evict,
                     &updated_id_count,
                     &local_write_allocate_total_duration,
                     &local_write_lookup_cache_total_duration,
                     &local_write_acquire_lock_duration,
                     &local_write_cache_miss] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      CHECK(engege_rates.is_contiguous());
                      CHECK_EQ(indices.size(0), engege_rates.size(0));
                      auto indices_data_ptr = indices.const_data_ptr<index_t>();
                      auto engage_rate_ptr =
                          engege_rates.const_data_ptr<float>();
                      {
                        auto before_write_lock_ts =
                            facebook::WallClockUtil::NowInUsecFast();
                        auto wlmap = kv_store_.by(shard_id).wlock();
                        local_write_acquire_lock_duration =
                            facebook::WallClockUtil::NowInUsecFast() -
                            before_write_lock_ts;
                        auto* pool = kv_store_.pool_by(shard_id);

                        for (const auto& id_index : indexes) {
                          auto id = int64_t(indices_data_ptr[id_index]);
                          float engege_rate = float(engage_rate_ptr[id_index]);
                          // use mempool
                          weight_type* block = nullptr;
                          auto before_lookup_cache_ts =
                              facebook::WallClockUtil::NowInUsecFast();
                          auto it = wlmap->find(id);
                          local_write_lookup_cache_total_duration +=
                              facebook::WallClockUtil::NowInUsecFast() -
                              before_lookup_cache_ts;
                          if (it != wlmap->end()) {
                            block = it->second;
                            feature_score_evict
                                ->update_feature_score_statistics(
                                    block, engege_rate, shard_id, false);
                            local_write_cache_miss += 1;
                          } else {
                            // Key doesn't exist, allocate new block and
                            // insert.
                            auto before_alloc_ts =
                                facebook::WallClockUtil::NowInUsecFast();
                            block = pool->template allocate_t<weight_type>();
                            FixedBlockPool::set_key(block, id);
                            FixedBlockPool::set_feature_score_rate(
                                block, engege_rate);
                            wlmap->insert({id, block});
                            feature_score_evict
                                ->update_feature_score_statistics(
                                    block, 0, shard_id, true);
                            local_write_allocate_total_duration +=
                                facebook::WallClockUtil::NowInUsecFast() -
                                before_alloc_ts;
                          }
                          updated_id_count++;
                        }
                      }
                    });
                return std::tuple{
                    updated_id_count,
                    local_write_allocate_total_duration,
                    local_write_lookup_cache_total_duration,
                    local_write_acquire_lock_duration,
                    local_write_cache_miss};
              }));
    }
    return folly::collect(std::move(futures))
        .via(executor_.get())
        .thenValue([this, start_ts](
                       const std::vector<std::tuple<
                           int64_t,
                           int64_t,
                           int64_t,
                           int64_t,
                           int64_t>>& results) {
          resume_ongoing_eviction();
          int total_updated_ids = 0;
          int64_t write_allocate_total_duration = 0;
          int64_t write_lookup_cache_total_duration = 0;
          int64_t write_acquire_lock_total_duration = 0;
          int64_t write_cache_miss = 0;
          for (
              const auto& [updated_ids, allocate_dur, lookup_cache_dur, acquire_lock_dur, cache_miss] :
              results) {
            total_updated_ids += updated_ids;
            write_allocate_total_duration += allocate_dur;
            write_lookup_cache_total_duration += lookup_cache_dur;
            write_acquire_lock_total_duration += acquire_lock_dur;
            write_cache_miss += cache_miss;
          }
          auto duration = facebook::WallClockUtil::NowInUsecFast() - start_ts;
          metadata_write_total_duration_ += duration;
          metadata_write_allocate_avg_duration_ +=
              write_allocate_total_duration / num_shards_;
          metadata_write_lookup_cache_avg_duration_ +=
              write_lookup_cache_total_duration / num_shards_;
          metadata_write_acquire_lock_avg_duration_ +=
              write_acquire_lock_total_duration / num_shards_;
          metadata_write_cache_miss_avg_count_ +=
              write_cache_miss / num_shards_;

          LOG(INFO)
              << "[DRAM KV][Feature Score Eviction]Total updated IDs across all shards: "
              << total_updated_ids;
          return std::vector<folly::Unit>(results.size());
        });
  }

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
      std::optional<int64_t> width_length = std::nullopt) {
    auto read_count = count.scalar_type() == at::ScalarType::Long
        ? *(count.const_data_ptr<int64_t>())
        : *(count.const_data_ptr<int32_t>());
    read_num_counts_ += read_count;
    // assuming get is called once each iteration and only by train
    // iteration(excluding state_dict)
    auto start_ts = facebook::WallClockUtil::NowInUsecFast();
    pause_ongoing_eviction(); // noop calls, no impact if called multiple times
    std::vector<folly::Future<
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>>>
        futures;
    auto row_width = weights.size(1);
    auto copy_width = width_length.value_or(row_width);
    CHECK_LE(row_width, max_D_);
    CHECK_EQ(copy_width, row_width);
    auto before_shard_ts = facebook::WallClockUtil::NowInUsecFast();
    auto shardid_to_indexes = shard_input(indices, count);
    read_sharding_total_duration_ +=
        facebook::WallClockUtil::NowInUsecFast() - before_shard_ts;

    for (const auto& [shard_id, indexes] : shardid_to_indexes) {
      futures.emplace_back(
          folly::via(executor_.get())
              .thenValue([this,
                          shard_id,
                          indexes,
                          &indices,
                          &weights,
                          width_offset,
                          row_width](folly::Unit) {
                int64_t local_read_cache_hit_copy_total_duration = 0;
                int64_t local_read_fill_row_storage_total_duration = 0;
                int64_t local_read_lookup_cache_total_duration = 0;
                int64_t local_read_aquire_lock_duration = 0;
                int64_t local_read_missing_load = 0;
                int64_t local_read_hit_load = 0;
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kvstore_set",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &weights,
                     width_offset,
                     row_width,
                     &local_read_cache_hit_copy_total_duration,
                     &local_read_fill_row_storage_total_duration,
                     &local_read_lookup_cache_total_duration,
                     &local_read_aquire_lock_duration,
                     &local_read_missing_load,
                     &local_read_hit_load] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      CHECK(weights.is_contiguous());
                      CHECK_EQ(indices.size(0), weights.size(0));

                      weight_type* row_storage_data_ptr = nullptr;
                      auto indices_data_ptr = indices.data_ptr<index_t>();
                      auto weights_data_ptr = weights.data_ptr<weight_type>();
                      auto before_read_lock_ts =
                          facebook::WallClockUtil::NowInUsecFast();
                      auto wlmap = kv_store_.by(shard_id).rlock();
                      local_read_aquire_lock_duration =
                          facebook::WallClockUtil::NowInUsecFast() -
                          before_read_lock_ts;

                      if (!wlmap->empty() && !is_training_) {
                        row_storage_data_ptr =
                            FixedBlockPool::data_ptr<weight_type>(
                                wlmap->begin()->second);
                      } else {
                        const auto& init_storage =
                            initializers_[shard_id]->row_storage_;
                        TORCH_CHECK(
                            init_storage.scalar_type() == weights.scalar_type(),
                            "init_storage (",
                            toString(init_storage.scalar_type()),
                            ") and weights scalar (",
                            toString(weights.scalar_type()),
                            ") types mismatch");
                        row_storage_data_ptr =
                            init_storage.template data_ptr<weight_type>();
                      }
                      {
                        for (const auto& weights_row_index : indexes) {
                          auto weight_idx =
                              int64_t(indices_data_ptr[weights_row_index]);
                          auto before_lookup_cache_ts =
                              facebook::WallClockUtil::NowInUsecFast();
                          const auto cached_iter = wlmap->find(weight_idx);
                          local_read_lookup_cache_total_duration +=
                              facebook::WallClockUtil::NowInUsecFast() -
                              before_lookup_cache_ts;
                          if (cached_iter == wlmap->end()) {
                            local_read_missing_load++;
                            auto weight_width = get_width_for_weights(
                                weight_idx, width_offset, row_width);
                            auto before_fill_from_row_storage_ts =
                                facebook::WallClockUtil::NowInUsecFast();
                            fill_from_row_storage(
                                shard_id,
                                reinterpret_cast<unsigned char*>(
                                    weights_data_ptr),
                                weights_row_index,
                                reinterpret_cast<unsigned char*>(
                                    row_storage_data_ptr),
                                width_offset,
                                row_width,
                                weight_width);
                            local_read_fill_row_storage_total_duration +=
                                facebook::WallClockUtil::NowInUsecFast() -
                                before_fill_from_row_storage_ts;
                            continue;
                          }
                          // use mempool
                          const auto* data_ptr =
                              FixedBlockPool::data_ptr<weight_type>(
                                  cached_iter->second);
                          auto before_cache_hit_copy_ts =
                              facebook::WallClockUtil::NowInUsecFast();
                          std::copy(
                              data_ptr + width_offset,
                              data_ptr + width_offset + row_width,
                              &(weights_data_ptr
                                    [weights_row_index *
                                     row_width])); // dst_start
                          local_read_cache_hit_copy_total_duration +=
                              facebook::WallClockUtil::NowInUsecFast() -
                              before_cache_hit_copy_ts;
                          local_read_hit_load++;
                        }
                      }
                    });
                return std::tuple{
                    local_read_lookup_cache_total_duration,
                    local_read_fill_row_storage_total_duration,
                    local_read_cache_hit_copy_total_duration,
                    local_read_aquire_lock_duration,
                    local_read_missing_load,
                    local_read_hit_load};
              }));
    }

    return folly::collect(std::move(futures))
        .via(executor_.get())
        .thenValue([this, start_ts](
                       const std::vector<std::tuple<
                           int64_t,
                           int64_t,
                           int64_t,
                           int64_t,
                           int64_t,
                           int64_t>>& results) {
          resume_laser_write();
          int64_t read_lookup_cache_total_duration = 0;
          int64_t read_fill_row_storage_total_duration = 0;
          int64_t read_cache_hit_copy_total_duration = 0;
          int64_t read_acquire_lock_total_duration = 0;
          int64_t read_missing_load = 0;
          int64_t read_hit_load = 0;
          for (
              const auto& [lookup_cache_dur, fill_row_storage_dur, cache_hit_copy_dur, acquire_lock_dur, missing_load, hit_load] :
              results) {
            read_lookup_cache_total_duration += lookup_cache_dur;
            read_fill_row_storage_total_duration += fill_row_storage_dur;
            read_cache_hit_copy_total_duration += cache_hit_copy_dur;
            read_acquire_lock_total_duration += acquire_lock_dur;
            read_missing_load += missing_load;
            read_hit_load += hit_load;
          }
          auto duration = facebook::WallClockUtil::NowInUsecFast() - start_ts;
          read_total_duration_ += duration;
          read_cache_hit_copy_avg_duration_ +=
              read_cache_hit_copy_total_duration / num_shards_;
          read_fill_row_storage_avg_duration_ +=
              read_fill_row_storage_total_duration / num_shards_;
          read_lookup_cache_total_avg_duration_ +=
              read_lookup_cache_total_duration / num_shards_;
          read_acquire_lock_avg_duration_ +=
              read_acquire_lock_total_duration / num_shards_;
          read_missing_load_avg_ += read_missing_load / num_shards_;
          read_hit_count_ += read_hit_load;
          read_miss_count_ += read_missing_load;
          return std::vector<folly::Unit>(results.size());
        });
  };

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    current_iter_++;
    return get_kv_db_async_impl(indices, weights, count);
  }

  void set_range_to_storage(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length) override {
    if (backend_return_whole_row_) {
      set_kv_with_metaheader_to_storage(weights);
    } else {
      const auto seq_indices = at::arange(
          start, start + length, at::TensorOptions().dtype(at::kLong));
      const auto count = at::tensor({length}, at::ScalarType::Long);
      folly::coro::blockingWait(set_kv_db_async(seq_indices, weights, count));
    }
  }

  void get_range_from_snapshot(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length,
      const ssd::SnapshotHandle* snapshot_handle, // should be nullptr for dram
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override {
    CHECK(snapshot_handle == nullptr);
    const auto seq_indices =
        at::arange(start, start + length, at::TensorOptions().dtype(at::kLong));

    if (backend_return_whole_row_) {
      get_kv_with_metaheader_from_storage(seq_indices, weights);
    } else {
      const auto count = at::tensor({length}, at::ScalarType::Long);
      get_kv_db_async_impl(
          seq_indices, weights, count, width_offset, width_length)
          .wait();
    }

    // this is called by checkpoint mostly, and checkpoint should wait until
    // eviction finishes so that we could reacha consistent state before/after
    // state_dict() calls
    // TODO: assert there isn't any eviction including paused
  }

  void set_kv_to_storage(const at::Tensor& ids, const at::Tensor& weights) {
    if (backend_return_whole_row_) {
      set_kv_with_metaheader_to_storage(weights);
    } else {
      const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
      set_kv_db_async(ids, weights, count).wait();
    }
  }

  void get_kv_from_storage_by_snapshot(
      const at::Tensor& ids,
      const at::Tensor& weights,
      const ssd::SnapshotHandle* snapshot_handle, // should be nullptr for dram
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) override {
    CHECK(snapshot_handle == nullptr);

    if (backend_return_whole_row_) {
      get_kv_with_metaheader_from_storage(
          ids, weights, width_offset, width_length);
    } else {
      const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
      get_kv_db_async_impl(ids, weights, count, width_offset, width_length)
          .wait();
    }
  }

  // used for ckpt, get kv with metaheader from storage
  void get_kv_with_metaheader_from_storage(
      const at::Tensor& ids,
      const at::Tensor& weights_with_metaheader,
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) {
    const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
    get_kv_db_with_metaheader_async_impl(
        ids, weights_with_metaheader, count, width_offset, width_length)
        .wait();
  }

  void set_kv_with_metaheader_to_storage(
      const at::Tensor& weights_with_metaheader) {
    std::vector<int64_t> keys(weights_with_metaheader.size(0), 0);
    for (int64_t i = 0; i < weights_with_metaheader.size(0); ++i) {
      keys[i] = FixedBlockPool::get_key(weights_with_metaheader[i].data_ptr());
    }
    auto indices =
        torch::from_blob(keys.data(), {int64_t(keys.size())}, torch::kInt64);
    const auto count =
        at::tensor({weights_with_metaheader.size(0)}, at::ScalarType::Long);
    set_kv_db_with_metaheader_async_impl(
        indices, weights_with_metaheader, count)
        .wait();
    // this is called by checkpoint mostly, and checkpoint should wait until
    // eviction finishes so that we could reacha consistent state before/after
    // state_dict() calls
    // TODO: assert there isn't any eviction including paused
  }

  void compact() override {}

  void trigger_feature_evict() {
    if (feature_evict_) {
      feature_evict_->trigger_evict();
    }
  }

  void maybe_evict() override {
    if (!feature_evict_config_.has_value()) {
      return;
    }
    switch (feature_evict_config_.value()->trigger_mode_) {
      case EvictTriggerMode::ITERATION: {
        if (feature_evict_config_.value()->trigger_step_interval_.value() > 0 &&
            current_iter_ %
                    feature_evict_config_.value()
                        ->trigger_step_interval_.value() ==
                0) {
          trigger_feature_evict();
        }
        break;
      }
      case EvictTriggerMode::MEM_UTIL: {
        auto mem_util = get_map_used_memsize_in_bytes() / (1024 * 1024 * 1024);
        if (mem_util >
            feature_evict_config_.value()->mem_util_threshold_in_GB_.value()) {
          trigger_feature_evict();
        }
        break;
      }
      case EvictTriggerMode::ID_COUNT: {
        auto used_id_count = get_num_rows();
        if (used_id_count > feature_evict_config_.value()
                                ->total_id_eviction_trigger_count_.value()) {
          trigger_feature_evict();
        }
        break;
      }
      case EvictTriggerMode::FREE_MEM: {
        // For free mem eviction, all conditions checked in frontend, no check
        // option in backend
        return;
      }
      default:
        break;
    }
  }

  // wait until eviction finishes, if any
  void wait_until_eviction_done() override {
    if (feature_evict_) {
      feature_evict_->wait_until_eviction_done();
    }
  }

  size_t get_map_used_memsize_in_bytes() const override {
    return kv_store_.getUsedMemSizeInBytes();
  }

  size_t get_map_actual_used_chunk_in_bytes() const {
    return kv_store_.getActualUsedChunkInBytes();
  }

  size_t get_num_rows() const {
    return kv_store_.getNumRows();
  }

  void resume_ongoing_eviction(bool force_resume = false) override {
    if (!is_training_ && !force_resume) {
      return;
    }
    resume_laser_write(); // Also resume laser write
    if (feature_evict_) {
      feature_evict_->resume();
    }
  }

  void pause_ongoing_eviction(bool force_pause = false) override {
    if (!is_training_ && !force_pause) {
      return;
    }
    if (!feature_evict_config_.has_value()) {
      return;
    }
    pause_laser_write(); // Also pause laser write
    if (feature_evict_config_.value()->trigger_mode_ !=
        EvictTriggerMode::DISABLED) {
      if (feature_evict_) {
        feature_evict_->pause();
      }
    }
  }

  bool is_evicting() override {
    if (feature_evict_) {
      return feature_evict_->is_evicting();
    }
    return false;
  }

  /// Pause laser write operations (called before forward/backward)
  void pause_laser_write() {
    std::unique_lock<std::mutex> lock(laser_write_mutex_);
    laser_write_interrupt_.store(true);
  }

  /// Resume laser write operations (called after forward/backward)
  void resume_laser_write() {
    std::unique_lock<std::mutex> lock(laser_write_mutex_);
    laser_write_interrupt_.store(false);
    laser_write_cv_.notify_all();
  }

  /// Wait until laser write is resumed if paused (like eviction pattern)
  /// Returns immediately if not interrupted, otherwise waits for resume
  void wait_until_laser_write_resume() {
    std::unique_lock<std::mutex> lock(laser_write_mutex_);
    if (!laser_write_interrupt_.load()) {
      return; // Not interrupted, continue immediately
    }
    laser_write_cv_.wait(
        lock, [this] { return !laser_write_interrupt_.load(); });
  }

  /// Check if laser write is interrupted (used in inner loop)
  bool is_laser_write_interrupted() const {
    return laser_write_interrupt_.load();
  }

  // for inference only, this logs the total hit/miss count
  // this should be called at the end of full/delta snapshot chunk by chunk
  // update
  void log_inplace_update_stats() {
    if (!is_training_) {
      int reset_val = 0;

      auto inplace_update_hit_cnt = inplace_update_hit_cnt_.exchange(reset_val);
      auto inplace_update_miss_cnt =
          inplace_update_miss_cnt_.exchange(reset_val);
      LOG(INFO) << "inplace update stats: hit count: " << inplace_update_hit_cnt
                << ", miss count: " << inplace_update_miss_cnt
                << ", total count: "
                << inplace_update_hit_cnt + inplace_update_miss_cnt
                << ", hit ratio: "
                << (double)inplace_update_hit_cnt /
              (inplace_update_hit_cnt + inplace_update_miss_cnt);
    }
  }

  std::optional<FeatureEvictMetricTensors> get_feature_evict_metric()
      const override {
    if (!feature_evict_config_.has_value()) {
      return std::nullopt;
    }
    if (feature_evict_config_.value()->trigger_mode_ ==
        EvictTriggerMode::DISABLED) {
      throw std::runtime_error("feature evict is disabled");
    }
    return feature_evict_->get_feature_evict_metric();
  }

  void set_backend_return_whole_row(bool backend_return_whole_row) override {
    backend_return_whole_row_ = backend_return_whole_row;
  }

 private:
  int64_t get_dim_from_index(int64_t weight_idx) const {
    if (sub_table_dims_.empty()) {
      return max_D_;
    }
    auto it = std::upper_bound(
        sub_table_hash_cumsum_.begin(),
        sub_table_hash_cumsum_.end(),
        weight_idx);
    if (it != sub_table_hash_cumsum_.end()) {
      int index = std::distance(sub_table_hash_cumsum_.begin(), it);
      return sub_table_dims_[index];
    }
    CHECK(false) << "weight_idx " << weight_idx
                 << " doesn't belong to any feature";
    return max_D_;
  }

  int64_t get_width_for_weights(
      int64_t weight_idx,
      int64_t width_offset,
      int64_t row_width) const {
    // when init an untouch embedding, we only want to init the weights part
    // and set the optimizer part to 0. This function helps us to get the dim
    // for each sub table, calculate the max bytes we should copy to the passed
    // in weights_data_ptr before optimizer section
    auto feature_dim = get_dim_from_index(weight_idx);
    CHECK_GT(feature_dim, width_offset);
    auto feature_width = feature_dim - width_offset;
    return std::min(feature_width, row_width);
  }

  void fill_from_row_storage(
      int shard_id,
      unsigned char* weights_data_ptr,
      int64_t weights_row_index,
      unsigned char* row_storage_data_ptr,
      int64_t width_offset,
      int64_t row_width,
      int64_t copied_width) {
    CHECK_GE(row_width, copied_width);
    CHECK_GE(max_D_, row_width);
    int64_t row_bytes = row_width * elem_size_;

    if (disable_random_init_) {
      // Skip data copy and leave values empty (zero-initialized)
      std::memset(
          &(weights_data_ptr[weights_row_index * row_bytes]), 0, row_bytes);
      return;
    }

    int64_t storage_row_bytes = elem_size_ * max_D_;
    auto copied_bytes = elem_size_ * copied_width;
    int64_t start_offset_bytes = elem_size_ * width_offset;
    int64_t row_index = 0;
    if (is_training_) {
      initializers_[shard_id]->producer_queue_.dequeue(row_index);
    }
    // TODO: fill the opt state as zeros for init value?
    std::copy(
        &(row_storage_data_ptr
              [row_index * storage_row_bytes + start_offset_bytes]),
        &(row_storage_data_ptr
              [row_index * storage_row_bytes + start_offset_bytes +
               copied_bytes]),
        &(weights_data_ptr[weights_row_index * row_bytes]));
    if (row_bytes > copied_bytes) {
      std::memset(
          &(weights_data_ptr[weights_row_index * row_bytes + copied_bytes]),
          0,
          row_bytes - copied_bytes);
    }
    if (is_training_) {
      initializers_[shard_id]->consumer_queue_.enqueue(row_index);
    }
  }

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
      const at::Tensor& count) {
    folly::F14FastMap<int, std::vector<int64_t>> shardid_to_indexes;

    FBGEMM_DISPATCH_INTEGRAL_TYPES(
        indices.scalar_type(),
        "dram_shard_input",
        [this, &indices, &shardid_to_indexes, &count] {
          using index_t = scalar_t;
          // Due to duplicate indicies, we only need to get/set the first count
          // of
          // entries.
          auto conv_count = count.scalar_type() == at::ScalarType::Long
              ? *(count.const_data_ptr<int64_t>())
              : *(count.const_data_ptr<int32_t>());
          auto indices_data_ptr = indices.data_ptr<index_t>();
          // There could be negative indices, which we should skipp
          for (int i = 0; i < conv_count; i++) {
            auto index = int64_t(indices_data_ptr[i]);
            if (index < 0) {
              continue;
            }

            const auto shard_id = kv_db_utils::hash_shard(index, num_shards_);
            shardid_to_indexes[shard_id].push_back(i);
          }
        });

    // chunk request based on bucket sharding
    return shardid_to_indexes;
  }

  void flush_or_compact(const int64_t timestep) override {}

  bool get_backend_return_whole_row() override {
    return backend_return_whole_row_;
  }

  int64_t get_metaheader_width_in_front() override {
    return backend_return_whole_row_
        ? FixedBlockPool::get_metaheader_dim<weight_type>()
        : 0;
  }

  std::vector<double> get_dram_kv_perf(
      const int64_t step,
      const int64_t interval) {
    std::vector<double> ret(40, 0); // num metrics
    if (step > 0 && step % interval == 0) {
      const double d_interval = static_cast<double>(interval);
      int reset_val = 0;

      auto dram_read_total_duration = read_total_duration_.exchange(reset_val);
      auto dram_read_sharding_total_duration =
          read_sharding_total_duration_.exchange(reset_val);
      auto dram_read_cache_hit_copy_duration =
          read_cache_hit_copy_avg_duration_.exchange(reset_val);
      auto dram_read_fill_row_storage_duration =
          read_fill_row_storage_avg_duration_.exchange(reset_val);
      auto dram_read_lookup_cache_duration =
          read_lookup_cache_total_avg_duration_.exchange(reset_val);
      auto dram_read_acquire_lock_duration =
          read_acquire_lock_avg_duration_.exchange(reset_val);
      auto dram_read_missing_load = read_missing_load_avg_.exchange(reset_val);
      auto dram_write_sharding_total_duration =
          write_sharding_total_duration_.exchange(reset_val);

      auto metadata_write_sharding_total_duration =
          metadata_write_sharding_total_duration_.exchange(reset_val);
      auto metadata_write_total_duration =
          metadata_write_total_duration_.exchange(reset_val);
      auto metadata_write_allocate_avg_duration =
          metadata_write_allocate_avg_duration_.exchange(reset_val);
      auto metadata_write_lookup_cache_avg_duration =
          metadata_write_lookup_cache_avg_duration_.exchange(reset_val);
      auto metadata_write_acquire_lock_avg_duration =
          metadata_write_acquire_lock_avg_duration_.exchange(reset_val);
      auto metadata_write_cache_miss_avg_count =
          metadata_write_cache_miss_avg_count_.exchange(reset_val);
      auto read_metadata_total_duration =
          read_metadata_total_duration_.exchange(reset_val);
      auto read_metadata_sharding_total_duration =
          read_metadata_sharding_total_duration_.exchange(reset_val);
      auto read_metadata_cache_hit_copy_avg_duration =
          read_metadata_cache_hit_copy_avg_duration_.exchange(reset_val);
      auto read_metadata_lookup_cache_total_avg_duration =
          read_metadata_lookup_cache_total_avg_duration_.exchange(reset_val);
      auto read_metadata_acquire_lock_avg_duration =
          read_metadata_acquire_lock_avg_duration_.exchange(reset_val);
      auto read_metadata_load_size =
          read_metadata_load_size_.exchange(reset_val);

      auto dram_fwd_l1_eviction_write_total_duration =
          fwd_l1_eviction_write_total_duration_.exchange(reset_val);
      auto dram_fwd_l1_eviction_write_allocate_duration =
          fwd_l1_eviction_write_allocate_avg_duration_.exchange(reset_val);
      auto dram_fwd_l1_eviction_write_cache_copy_duration =
          fwd_l1_eviction_write_cache_copy_avg_duration_.exchange(reset_val);
      auto dram_fwd_l1_eviction_write_lookup_cache_duration =
          fwd_l1_eviction_write_lookup_cache_avg_duration_.exchange(reset_val);
      auto dram_fwd_l1_eviction_write_acquire_lock_duration_ =
          fwd_l1_eviction_write_acquire_lock_avg_duration_.exchange(reset_val);
      auto dram_fwd_l1_eviction_write_missing_load_ =
          fwd_l1_eviction_write_missing_load_avg_.exchange(reset_val);

      auto dram_bwd_l1_cnflct_miss_write_total_duration =
          bwd_l1_cnflct_miss_write_total_duration_.exchange(reset_val);
      auto dram_bwd_l1_cnflct_miss_write_allocate_duration =
          bwd_l1_cnflct_miss_write_allocate_avg_duration_.exchange(reset_val);
      auto dram_bwd_l1_cnflct_miss_write_cache_copy_duration =
          bwd_l1_cnflct_miss_write_cache_copy_avg_duration_.exchange(reset_val);
      auto dram_bwd_l1_cnflct_miss_write_lookup_cache_duration =
          bwd_l1_cnflct_miss_write_lookup_cache_avg_duration_.exchange(
              reset_val);
      auto dram_bwd_l1_cnflct_miss_write_acquire_lock_duration_ =
          bwd_l1_cnflct_miss_write_acquire_lock_avg_duration_.exchange(
              reset_val);
      auto dram_bwd_l1_cnflct_miss_write_missing_load_ =
          bwd_l1_cnflct_miss_write_missing_load_avg_.exchange(reset_val);

      auto read_num_counts = read_num_counts_.exchange(reset_val);
      auto read_hit_count = read_hit_count_.exchange(reset_val);
      auto read_miss_count = read_miss_count_.exchange(reset_val);
      auto enrichment_query_count = enrichment_query_count_.exchange(reset_val);
      auto enrichment_empty_count = enrichment_empty_count_.exchange(reset_val);

      ret[0] = dram_read_total_duration / d_interval;
      ret[1] = dram_read_sharding_total_duration / d_interval;
      ret[2] = dram_read_cache_hit_copy_duration / d_interval;
      ret[3] = dram_read_fill_row_storage_duration / d_interval;
      ret[4] = dram_read_lookup_cache_duration / d_interval;
      ret[5] = dram_read_acquire_lock_duration / d_interval;
      ret[6] = dram_read_missing_load / d_interval;
      ret[7] = dram_write_sharding_total_duration / d_interval;

      ret[8] = dram_fwd_l1_eviction_write_total_duration / d_interval;
      ret[9] = dram_fwd_l1_eviction_write_allocate_duration / d_interval;
      ret[10] = dram_fwd_l1_eviction_write_cache_copy_duration / d_interval;
      ret[11] = dram_fwd_l1_eviction_write_lookup_cache_duration / d_interval;
      ret[12] = dram_fwd_l1_eviction_write_acquire_lock_duration_ / d_interval;
      ret[13] = dram_fwd_l1_eviction_write_missing_load_ / d_interval;

      ret[14] = dram_bwd_l1_cnflct_miss_write_total_duration / d_interval;
      ret[15] = dram_bwd_l1_cnflct_miss_write_allocate_duration / d_interval;
      ret[16] = dram_bwd_l1_cnflct_miss_write_cache_copy_duration / d_interval;
      ret[17] =
          dram_bwd_l1_cnflct_miss_write_lookup_cache_duration / d_interval;
      ret[18] =
          dram_bwd_l1_cnflct_miss_write_acquire_lock_duration_ / d_interval;
      ret[19] = dram_bwd_l1_cnflct_miss_write_missing_load_ / d_interval;

      ret[20] = get_map_used_memsize_in_bytes();
      ret[21] = get_map_actual_used_chunk_in_bytes();

      ret[22] = get_num_rows();
      ret[23] = read_num_counts / d_interval;

      ret[24] = metadata_write_sharding_total_duration / d_interval;
      ret[25] = metadata_write_total_duration / d_interval;
      ret[26] = metadata_write_allocate_avg_duration / d_interval;
      ret[27] = metadata_write_lookup_cache_avg_duration / d_interval;
      ret[28] = metadata_write_acquire_lock_avg_duration / d_interval;
      ret[29] = metadata_write_cache_miss_avg_count / d_interval;

      ret[30] = read_metadata_total_duration / d_interval;
      ret[31] = read_metadata_sharding_total_duration / d_interval;
      ret[32] = read_metadata_cache_hit_copy_avg_duration / d_interval;
      ret[33] = read_metadata_lookup_cache_total_avg_duration / d_interval;
      ret[34] = read_metadata_acquire_lock_avg_duration / d_interval;
      ret[35] = read_metadata_load_size / d_interval;

      ret[36] = read_hit_count;
      ret[37] = read_miss_count;

      ret[38] = enrichment_query_count;
      ret[39] = enrichment_empty_count;
    }
    return ret;
  }

  /// Get embeddings and metaheader from kvstore.
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights_with_metaheader The 2D tensor that each row(embeddings) is
  /// paired up with relative element in <indices>. This tensor will be
  /// filled up with the returned embeddings from KVstore.
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return None
  folly::SemiFuture<std::vector<folly::Unit>>
  get_kv_db_with_metaheader_async_impl(
      const at::Tensor& indices,
      const at::Tensor& weights_with_metaheader,
      const at::Tensor& count,
      int64_t width_offset = 0,
      std::optional<int64_t> width_length = std::nullopt) {
    auto row_width = weights_with_metaheader.size(1);
    auto copy_width = width_length.value_or(row_width);
    CHECK_LE(row_width, block_size_);
    CHECK_EQ(copy_width, row_width);
    auto shardid_to_indexes = shard_input(indices, count);
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(shardid_to_indexes.size());

    for (const auto& [shard_id, indexes] : shardid_to_indexes) {
      auto f =
          folly::via(executor_.get())
              .thenValue([this,
                          shard_id,
                          indexes,
                          &indices,
                          &weights_with_metaheader,
                          width_offset,
                          row_width](folly::Unit) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kvstore_get_with_metaheader",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &weights_with_metaheader,
                     width_offset,
                     row_width] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      CHECK(weights_with_metaheader.is_contiguous());
                      CHECK_EQ(
                          indices.size(0), weights_with_metaheader.size(0));
                      auto wlmap = kv_store_.by(shard_id).wlock();
                      auto indices_data_ptr = indices.data_ptr<index_t>();
                      auto weights_data_ptr =
                          weights_with_metaheader.data_ptr<weight_type>();
                      {
                        for (const auto& weights_row_index : indexes) {
                          auto weight_idx =
                              int64_t(indices_data_ptr[weights_row_index]);
                          const auto cached_iter = wlmap->find(weight_idx);
                          // Defensive programming
                          // it shouldn't occur under normal circumstances
                          if (cached_iter == wlmap->end()) {
                            std::memset(
                                &(weights_data_ptr
                                      [weights_row_index * row_width]),
                                0,
                                row_width);
                            continue;
                          }

                          // For weight KVT, offset=0 and it will read the whole
                          // row. For optimizer, offset=dim(metaheader) +
                          // emb_dim so it will only read the optimizer part
                          const auto* ptr_offset_from_front =
                              FixedBlockPool::ptr_offset_from_front<
                                  weight_type>(
                                  cached_iter->second, width_offset);
                          std::copy(
                              ptr_offset_from_front,
                              ptr_offset_from_front + row_width,
                              &(weights_data_ptr
                                    [weights_row_index * row_width]));
                        }
                      }
                    });
              });
      futures.emplace_back(std::move(f));
    }
    return folly::collect(futures);
  }

  /// insert embeddings and metaheader into kvstore.
  /// current underlying memory management is done through F14FastMap
  /// key value pair will be sharded into multiple shards to increase
  /// parallelism.
  ///
  /// @param indices The 1D embedding index tensor, should skip on negative
  /// value
  /// @param weights_with_metaheader The 2D tensor that each row(embeddings with
  /// metaheader) is paired up with relative element in <indices>
  /// @param count A single element tensor that contains the number of indices
  /// to be processed
  ///
  /// @return None
  folly::SemiFuture<std::vector<folly::Unit>>
  set_kv_db_with_metaheader_async_impl(
      const at::Tensor& indices,
      const at::Tensor& weights_with_metaheader,
      const at::Tensor& count) {
    auto shardid_to_indexes = shard_input(indices, count);
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(shardid_to_indexes.size());
    for (const auto& [shard_id, indexes] : shardid_to_indexes) {
      auto f =
          folly::via(executor_.get())
              .thenValue([this,
                          shard_id,
                          indexes,
                          &indices,
                          &weights_with_metaheader](folly::Unit) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kv_set_with_metaheader",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &weights_with_metaheader] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      CHECK(weights_with_metaheader.is_contiguous());
                      CHECK_EQ(
                          indices.size(0), weights_with_metaheader.size(0));
                      {
                        auto wlmap = kv_store_.by(shard_id).wlock();
                        auto* pool = kv_store_.pool_by(shard_id);
                        int64_t stride = weights_with_metaheader.size(1);
                        auto indices_data_ptr = indices.data_ptr<index_t>();
                        auto weights_data_ptr =
                            weights_with_metaheader.data_ptr<weight_type>();
                        for (const auto& id_index : indexes) {
                          auto id = int64_t(indices_data_ptr[id_index]);
                          // Defensive programming
                          // used is false shouldn't occur under normal
                          // circumstances
                          FixedBlockPool::set_used(
                              weights_data_ptr + id_index * stride, true);

                          // use mempool
                          weight_type* block = nullptr;
                          // First check if the key already exists
                          auto it = wlmap->find(id);
                          bool new_block = false;
                          if (it != wlmap->end()) {
                            block = it->second;
                          } else {
                            // Key doesn't exist, allocate new block and
                            // insert.
                            block = pool->template allocate_t<weight_type>();
                            wlmap->insert({id, block});
                            new_block = true;
                          }
                          std::copy(
                              weights_data_ptr + id_index * stride,
                              weights_data_ptr + (id_index + 1) * stride,
                              block);

                          if (new_block) {
                            if (feature_evict_config_.has_value() &&
                                feature_evict_config_.value()->trigger_mode_ !=
                                    EvictTriggerMode::DISABLED &&
                                feature_evict_) {
                              auto* feature_score_evict = dynamic_cast<
                                  FeatureScoreBasedEvict<weight_type>*>(
                                  feature_evict_.get());
                              if (feature_score_evict) {
                                feature_score_evict
                                    ->update_feature_score_statistics(
                                        block, 0, shard_id, true);
                              }
                            }
                          }
                        }
                      }
                    });
              });
      futures.emplace_back(std::move(f));
    }
    return folly::collect(futures);
  }

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
  bool backend_return_whole_row_;
  std::vector<int64_t> sub_table_dims_;
  std::vector<int64_t> sub_table_hash_cumsum_;
  std::optional<c10::intrusive_ptr<FeatureEvictConfig>> feature_evict_config_;
  std::unique_ptr<FeatureEvict<weight_type>> feature_evict_;
  std::atomic<int> current_iter_{0};
  const bool is_training_;

  // perf stats
  std::atomic<int64_t> read_total_duration_{0};
  std::atomic<int64_t> read_sharding_total_duration_{0};
  std::atomic<int64_t> read_cache_hit_copy_avg_duration_{0};
  std::atomic<int64_t> read_fill_row_storage_avg_duration_{0};
  std::atomic<int64_t> read_lookup_cache_total_avg_duration_{0};
  std::atomic<int64_t> read_acquire_lock_avg_duration_{0};
  std::atomic<int64_t> read_missing_load_avg_{0};
  std::atomic<int64_t> write_sharding_total_duration_{0};

  std::atomic<int64_t> read_metadata_total_duration_{0};
  std::atomic<int64_t> read_metadata_sharding_total_duration_{0};
  std::atomic<int64_t> read_metadata_cache_hit_copy_avg_duration_{0};
  std::atomic<int64_t> read_metadata_lookup_cache_total_avg_duration_{0};
  std::atomic<int64_t> read_metadata_acquire_lock_avg_duration_{0};
  std::atomic<int64_t> read_metadata_load_size_{0};

  std::atomic<int64_t> metadata_write_total_duration_{0};
  std::atomic<int64_t> metadata_write_sharding_total_duration_{0};
  std::atomic<int64_t> metadata_write_allocate_avg_duration_{0};
  std::atomic<int64_t> metadata_write_lookup_cache_avg_duration_{0};
  std::atomic<int64_t> metadata_write_acquire_lock_avg_duration_{0};
  std::atomic<int64_t> metadata_write_cache_miss_avg_count_{0};

  std::atomic<int64_t> bwd_l1_cnflct_miss_write_total_duration_{0};
  std::atomic<int64_t> bwd_l1_cnflct_miss_write_allocate_avg_duration_{0};
  std::atomic<int64_t> bwd_l1_cnflct_miss_write_cache_copy_avg_duration_{0};
  std::atomic<int64_t> bwd_l1_cnflct_miss_write_lookup_cache_avg_duration_{0};
  std::atomic<int64_t> bwd_l1_cnflct_miss_write_acquire_lock_avg_duration_{0};
  std::atomic<int64_t> bwd_l1_cnflct_miss_write_missing_load_avg_{0};

  std::atomic<int64_t> fwd_l1_eviction_write_total_duration_{0};
  std::atomic<int64_t> fwd_l1_eviction_write_allocate_avg_duration_{0};
  std::atomic<int64_t> fwd_l1_eviction_write_cache_copy_avg_duration_{0};
  std::atomic<int64_t> fwd_l1_eviction_write_lookup_cache_avg_duration_{0};
  std::atomic<int64_t> fwd_l1_eviction_write_acquire_lock_avg_duration_{0};
  std::atomic<int64_t> fwd_l1_eviction_write_missing_load_avg_{0};

  std::atomic<int64_t> inplace_update_hit_cnt_{0};
  std::atomic<int64_t> inplace_update_miss_cnt_{0};

  std::atomic<int64_t> read_num_counts_{0};

  // DRAM KV cache hit/miss raw counters (not averaged per shard)
  std::atomic<int64_t> read_hit_count_{0};
  std::atomic<int64_t> read_miss_count_{0};

  // Enrichment (laser) query counters: total keys queried and empty results
  std::atomic<int64_t> enrichment_query_count_{0};
  std::atomic<int64_t> enrichment_empty_count_{0};

  bool disable_random_init_;

  // Whether raw embedding streaming (RES) is enabled for this cache
  bool enable_raw_embedding_streaming_ = false;

  // Enrichment configuration (passed from Python, replaces hardcoded laser
  // config)
  std::optional<c10::intrusive_ptr<kv_mem::EnrichmentConfig>>
      enrichment_config_;

  // Enrichment rate limiting
  std::atomic<int64_t> pending_laser_requests_{0};
  std::atomic<bool> laser_write_in_progress_{false};

  // Enrichment write pause/resume mechanism (similar to eviction)
  std::atomic<bool> laser_write_interrupt_{false};
  std::mutex laser_write_mutex_;
  std::condition_variable laser_write_cv_;

  // Dedicated executor for enrichment (separate from main executor)
  std::unique_ptr<folly::CPUThreadPoolExecutor> enrichment_executor_;

  // Pre-initialized LaserClient for IGR enrichment (reused across fetches)
  std::shared_ptr<facebook::laser::LaserClient> laser_client_;

  // OpenTab/Maple reader for ONEFLOW_OPENTAB_SID enrichment (type-erased)
  oneflow_enrichment::ReaderPtr open_tab_reader_;
}; // class DramKVEmbeddingCache

} // namespace kv_mem
