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
#include <folly/ExceptionString.h>
#include <folly/ScopeGuard.h>
#include <folly/SocketAddress.h>
#include <folly/Synchronized.h>
#include <folly/container/small_vector.h>
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
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include "common/time/Time.h"

#include <c10/cuda/CUDAGuard.h>
#include "../ssd_split_embeddings_cache/initializer.h"
#include "../ssd_split_embeddings_cache/kv_db_cuda_utils.h"
#include "../ssd_split_embeddings_cache/kv_db_table_batched_embeddings.h"
#include "SynchronizedShardedMap.h"
#include "deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/fake_enrichment.h"
#include "deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/feature_store_enrichment.h"
#include "deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/igr_enrichment.h"
#include "deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/oneflow_enrichment.h"
#include "enrichment_config.h"
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "feature_evict.h"
#include "fixed_block_pool.h"
#include "serving_qdq.h"

namespace kv_db {
template <typename W>
class DramSsdKVEmbeddingCache;
} // namespace kv_db

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
  template <typename W>
  friend class kv_db::DramSsdKVEmbeddingCache;

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
  /// @param enable_ssd_backend whether an SSD backend is wired up, enabling
  /// dirty-bit tracking so dirty DRAM blocks can be written back to SSD. The
  /// dirty bit denotes a change to the embedding value itself, as opposed to
  /// its metadata; in the embedding-cache use case this only happens when a new
  /// embedding is inserted, since the cache never updates an existing
  /// embedding's value. Defaults to false.
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
          enrichment_config = std::nullopt,
      bool enable_ssd_backend = false)
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
        packed_(packed_layout(max_D, enrichment_config)),
        int8_(int8_layout(max_D, enrichment_config)),
        block_size_(row_block_size(max_D, packed_, int8_)),
        block_alignment_(
            FixedBlockPool::calculate_block_alignment<weight_type>()),
        kv_store_(
            SynchronizedShardedMap<int64_t, weight_type*>(
                num_shards_,
                block_size_,
                block_alignment_,
                /*blocks_per_chunk=*/8192,
                /*enable_dirty_tracking=*/enable_ssd_backend)),
        elem_size_(row_storage_bitwidth / 8),
        backend_return_whole_row_(backend_return_whole_row),
        feature_evict_config_(std::move(feature_evict_config)),
        is_training_(is_training),
        disable_random_init_(disable_random_init),
        enable_raw_embedding_streaming_(enable_raw_embedding_streaming),
        enable_ssd_backend_(enable_ssd_backend) {
    // Per-row perf timestamps are reporting-only, and nothing reads them unless
    // a stats interval is set. The read/write loops call the clock four times
    // PER ROW (around a single F14 find and a pointer deref), which on a table
    // serving ~320k DRAM reads per step is ~1.3M clock calls per step for
    // numbers nobody collects. Default keeps today's behaviour; set
    // KVZCH_SKIP_DRAM_PERF_TIMING=1 to skip.
    collect_perf_stats_ =
        std::getenv("KVZCH_SKIP_DRAM_PERF_TIMING") == nullptr ||
        std::string(std::getenv("KVZCH_SKIP_DRAM_PERF_TIMING")) != "1";
    const char* const parallel_shard_input =
        std::getenv("KVZCH_PARALLEL_DRAM_SHARD_INPUT");
    parallel_shard_input_ = parallel_shard_input != nullptr &&
        std::string_view(parallel_shard_input) == "1";
    if (parallel_shard_input_) {
      LOG(INFO) << "KVZCH parallel DRAM input sharding enabled";
    }
    // Enrichment-cache rows are inserted once and then only overwritten by the
    // enrichment write, which leaves the timestamp alone, so TTL eviction ages
    // rows from insertion rather than last use. These opt-ins keep the
    // timestamp tracking enrichment-query hits and checkpoint load, not all
    // DRAM/HBM accesses. is_training_ describes the C++ cache instance; the
    // production wrapper always passes true, including when reused for
    // publish/eval. Keep these env flags off in jobs that must preserve saved
    // timestamps. The DRAM_SSD cache's restore path is not covered.
    refresh_timestamp_on_enrich_hit_ =
        is_env_flag_set("KVZCH_REFRESH_TIMESTAMP_ON_ENRICH_HIT");
    restamp_on_restore_ = is_env_flag_set("KVZCH_RESTAMP_ON_RESTORE");
    if (refresh_timestamp_on_enrich_hit_ || restamp_on_restore_) {
      LOG(INFO) << "KVZCH timestamp refresh on enrich hit: "
                << refresh_timestamp_on_enrich_hit_
                << ", restamp on restore: " << restamp_on_restore_;
    }
    executor_thread_count_ = get_executor_thread_count(num_threads);
    // Parallel sharding synchronously joins chunk work submitted to executor_.
    // Keep a one-thread debug executor on the serial path so an executor
    // worker can never wait for work queued behind itself.
    if (parallel_shard_input_ && executor_thread_count_ == 1) {
      LOG(WARNING)
          << "KVZCH parallel DRAM input sharding disabled: executor has one "
             "thread";
      parallel_shard_input_ = false;
    }
    executor_ =
        std::make_unique<folly::CPUThreadPoolExecutor>(executor_thread_count_);
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

      // Initialize OpenTab reader for scalar and ID-list payload enrichment.
      if (enrichment_config_.value()->enrichment_type_ ==
              kv_mem::EnrichmentType::ONEFLOW_OPENTAB_SID ||
          enrichment_config_.value()->enrichment_type_ ==
              kv_mem::EnrichmentType::ONEFLOW_OPENTAB_ID_LIST) {
        open_tab_reader_ = oneflow_enrichment::initializeOpenTabReader(
            *enrichment_config_.value());
      }
    }
    initialize_initializers(
        num_shards,
        max_D,
        uniform_init_lower,
        uniform_init_upper,
        row_storage_bitwidth);
    const int requested_lookup_prefetch_distance =
        get_requested_lookup_prefetch_distance();
    if (requested_lookup_prefetch_distance > 0 && !disable_random_init_) {
      LOG(WARNING)
          << "KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE requested but refused: "
             "disable_random_init must be enabled";
    } else if (requested_lookup_prefetch_distance > 0 && collect_perf_stats_) {
      LOG(WARNING)
          << "KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE requested but refused: "
             "KVZCH_SKIP_DRAM_PERF_TIMING must be enabled";
    } else {
      dram_lookup_prefetch_distance_ = requested_lookup_prefetch_distance;
      if (dram_lookup_prefetch_distance_ > 0) {
        LOG(INFO) << "KVZCH DRAM lookup prefetch distance: "
                  << dram_lookup_prefetch_distance_;
      }
    }
    const int requested_read_splits = get_requested_read_splits_per_shard();
    if (requested_read_splits > 1 && !disable_random_init_) {
      LOG(WARNING) << "KVZCH_DRAM_READ_SPLITS_PER_SHARD requested but refused: "
                      "disable_random_init must be enabled";
    } else if (requested_read_splits > 1 && collect_perf_stats_) {
      LOG(WARNING) << "KVZCH_DRAM_READ_SPLITS_PER_SHARD requested but refused: "
                      "KVZCH_SKIP_DRAM_PERF_TIMING must be enabled";
    } else {
      dram_read_splits_per_shard_ = requested_read_splits;
      if (dram_read_splits_per_shard_ > 1) {
        LOG(INFO) << "KVZCH DRAM read splits per shard: "
                  << dram_read_splits_per_shard_;
      }
    }
    const char* const split_zero_copy =
        std::getenv("KVZCH_DRAM_READ_SPLIT_ZERO_COPY");
    const bool requested_split_zero_copy =
        split_zero_copy != nullptr && std::string_view(split_zero_copy) == "1";
    if (requested_split_zero_copy && dram_read_splits_per_shard_ == 1) {
      LOG(WARNING) << "KVZCH_DRAM_READ_SPLIT_ZERO_COPY requested but refused: "
                      "KVZCH_DRAM_READ_SPLITS_PER_SHARD must be greater than 1";
    } else {
      dram_read_split_zero_copy_ = requested_split_zero_copy;
      if (dram_read_split_zero_copy_) {
        LOG(INFO) << "KVZCH zero-copy DRAM read split ranges enabled";
      }
    }
    const char* const segmented_shard_reads =
        std::getenv("KVZCH_DRAM_SEGMENTED_SHARD_READS");
    const bool requested_segmented_shard_reads =
        segmented_shard_reads != nullptr &&
        std::string_view(segmented_shard_reads) == "1";
    if (requested_segmented_shard_reads &&
        (!parallel_shard_input_ || num_shards_ <= 1 || !disable_random_init_ ||
         collect_perf_stats_ || dram_read_splits_per_shard_ != 1 ||
         std::cmp_greater(num_shards_, std::numeric_limits<uint16_t>::max()))) {
      LOG(WARNING)
          << "KVZCH_DRAM_SEGMENTED_SHARD_READS requested but refused: "
             "multiple shards, parallel input sharding, disabled random "
             "initialization and perf timing, and one read task per shard "
             "are required";
    } else {
      segmented_shard_reads_ = requested_segmented_shard_reads;
      if (segmented_shard_reads_) {
        LOG(INFO) << "KVZCH segmented DRAM shard reads enabled";
      }
    }
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
    if (packed_) {
      check_packed_storage_supported();
      fallback_rows_.resize(num_shards_);
      LOG(INFO) << "KVZCH packed DRAM rows: dim=" << packed_->dim
                << " group_size=" << packed_->group_size
                << " tail=" << packed_->tail << " block_size=" << block_size_
                << " (fp16 block_size="
                << FixedBlockPool::calculate_block_size<weight_type>(max_D_)
                << ")";
    }
    if (int8_) {
      check_int8_storage_supported();
      LOG(INFO) << "KVZCH INT8 DRAM rows: dim=" << int8_->dim
                << " tail=" << int8_->tail << " block_size=" << block_size_
                << " (fp16 block_size="
                << FixedBlockPool::calculate_block_size<weight_type>(max_D_)
                << ")";
    }
  }

  void initialize_initializers(
      int64_t num_shards,
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t row_storage_bitwidth) {
    const char* const skip_disabled_initializers =
        std::getenv("KVZCH_SKIP_DISABLED_INITIALIZERS");
    skip_disabled_initializers_ = disable_random_init_ &&
        skip_disabled_initializers != nullptr &&
        std::string_view(skip_disabled_initializers) == "1";
    if (skip_disabled_initializers_) {
      LOG(INFO) << "KVZCH skipping disabled DRAM initializers";
      return;
    }

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
      auto f =
          folly::via(executor_.get())
              .thenValue(
                  [this, shard_id, start, end, offset, &ids](folly::Unit) {
                    auto rlmap = kv_store_.by(shard_id).rlock();
                    for (const auto& [key, value] : *rlmap) {
                      if (key >= start && key < end) {
                        ids[shard_id].push_back(offset ? key - *offset : key);
                      }
                    }
                    if (hbm_) {
                      for (const auto& [key, meta] : hbm_rows_[shard_id]) {
                        if (key >= start && key < end) {
                          ids[shard_id].push_back(offset ? key - *offset : key);
                        }
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
               [all_ids_ptr](void* /*p*/) mutable { all_ids_ptr.reset(); },
               torch::kInt64 // data type
               )
        .view({-1, 1});
  }

  /// Get eviction metadata given indices
  at::Tensor get_kv_zch_eviction_metadata_impl(
      const at::Tensor& indices,
      const at::Tensor& count) override {
    auto start_ts = perf_now_usec();
    std::vector<folly::Future<std::tuple<int64_t, int64_t, int64_t>>> futures;
    auto numel = indices.size(0);
    read_metadata_load_size_ = numel;
    auto metadata_tensor = at::zeros({numel}, at::kLong);
    auto before_shard_ts = perf_now_usec();
    auto shardid_to_indexes = shard_input(indices, count);
    read_metadata_sharding_total_duration_ += perf_now_usec() - before_shard_ts;
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
                        auto before_read_lock_ts = perf_now_usec();
                        auto rlmap = kv_store_.by(shard_id).rlock();
                        local_read_aquire_lock_duration =
                            perf_now_usec() - before_read_lock_ts;
                        for (const auto& id_index : indexes) {
                          auto id = int64_t(indices_data_ptr[id_index]);

                          // use mempool
                          weight_type* block = nullptr;
                          auto before_lookup_cache_ts = perf_now_usec();
                          auto it = rlmap->find(id);
                          local_read_lookup_cache_total_duration +=
                              perf_now_usec() - before_lookup_cache_ts;
                          // All ids should be found in backend to get metadata
                          if (it == rlmap->end()) {
                            // With an HBM tier, filled by hbm_fill_metadata
                            // below, or left 0 for a key that left both tiers
                            // after it was listed.
                            CHECK(hbm_);
                            continue;
                          }
                          block = it->second;
                          auto before_cache_hit_copy_ts = perf_now_usec();
                          metadata[id_index] =
                              FixedBlockPool::get_metaheader_raw(block);
                          local_read_cache_hit_copy_total_duration +=
                              perf_now_usec() - before_cache_hit_copy_ts;
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
    if (hbm_) {
      hbm_fill_metadata(indices, metadata_tensor);
    }
    read_metadata_total_duration_ += perf_now_usec() - start_ts;
    return metadata_tensor;
  }

  at::Tensor get_kv_metadata_rows(
      const at::Tensor& indices,
      const at::Tensor& count) {
    auto numel = indices.size(0);
    const int64_t metadata_dim =
        static_cast<int64_t>(FixedBlockPool::get_metaheader_dim<weight_type>());
    auto metadata_tensor = at::zeros(
        {numel, metadata_dim},
        at::TensorOptions().dtype(
            c10::CppTypeToScalarType<weight_type>::value));
    auto shardid_to_indexes = shard_input(indices, count);
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(shardid_to_indexes.size());
    const size_t metadata_bytes = metadata_dim * sizeof(weight_type);
    for (const auto& [shard_id, indexes] : shardid_to_indexes) {
      futures.emplace_back(
          folly::via(executor_.get())
              .thenValue([this,
                          shard_id,
                          indexes,
                          &indices,
                          &metadata_tensor,
                          metadata_bytes](folly::Unit) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kv_metadata_rows",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &metadata_tensor,
                     metadata_bytes] {
                      using index_t = scalar_t;
                      CHECK(indices.is_contiguous());
                      auto* idx_ptr = indices.const_data_ptr<index_t>();
                      auto* md_ptr =
                          metadata_tensor
                              .template mutable_data_ptr<weight_type>();
                      const int64_t md_stride = metadata_tensor.size(1);
                      auto rlmap = kv_store_.by(shard_id).rlock();
                      for (const auto& id_index : indexes) {
                        auto id = int64_t(idx_ptr[id_index]);
                        auto it = rlmap->find(id);
                        // Eviction can remove a row after shard_input() and
                        // before this shard lock is acquired. The output is
                        // zero-initialized, so preserve the missing-row
                        // contract instead of aborting the process.
                        if (it == rlmap->end()) {
                          continue;
                        }
                        std::memcpy(
                            md_ptr + id_index * md_stride,
                            reinterpret_cast<const char*>(it->second),
                            metadata_bytes);
                      }
                    });
              }));
    }
    folly::collect(futures).wait();
    return metadata_tensor;
  }

  at::Tensor get_missing_mask(
      const at::Tensor& indices,
      const at::Tensor& count) {
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    auto missing = at::zeros(
        {indices.size(0)}, at::TensorOptions().dtype(at::ScalarType::Bool));
    auto shardid_to_indexes = shard_input(indices, count);
    FBGEMM_DISPATCH_INTEGRAL_TYPES(
        indices.scalar_type(), "dram_kv_missing_mask", [&] {
          using index_t = scalar_t;
          const auto* indices_ptr = indices.const_data_ptr<index_t>();
          auto* missing_ptr = missing.mutable_data_ptr<bool>();
          for (const auto& [shard_id, indexes] : shardid_to_indexes) {
            auto rlmap = kv_store_.by(shard_id).rlock();
            for (const auto index : indexes) {
              missing_ptr[index] =
                  rlmap->find(int64_t(indices_ptr[index])) == rlmap->end();
            }
          }
        });
    return missing;
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
    if (w_mode == kv_db::RocksdbWriteMode::STREAM) {
      auto pause_state = pause_for_stream_write();
      SCOPE_EXIT {
        resume_after_stream_write(pause_state);
      };
      set_kv_db_async_impl(indices, weights, count, w_mode).get();
      return folly::makeSemiFuture(std::vector<folly::Unit>());
    }
    return set_kv_db_async_impl(indices, weights, count, w_mode);
  }

  struct StreamPauseState {
    bool laser_suspended{false};
    bool eviction_suspended{false};
  };

  StreamPauseState pause_for_stream_write() {
    StreamPauseState state;
    try {
      if (is_training_) {
        std::unique_lock<std::mutex> lock(laser_write_mutex_);
        if (stream_laser_suspensions_++ == 0) {
          laser_write_epoch_.fetch_add(1);
        }
        laser_write_interrupt_.store(true);
        state.laser_suspended = true;
        laser_write_cv_.wait(
            lock, [this] { return active_laser_writers_ == 0; });
      }
      if (feature_evict_config_.has_value() &&
          feature_evict_config_.value()->trigger_mode_ !=
              EvictTriggerMode::DISABLED &&
          feature_evict_) {
        feature_evict_->suspend_triggers();
        state.eviction_suspended = true;
      }
    } catch (...) {
      resume_after_stream_write(state);
      throw;
    }
    return state;
  }

  void resume_after_stream_write(const StreamPauseState& state) {
    if (state.laser_suspended) {
      std::unique_lock<std::mutex> lock(laser_write_mutex_);
      CHECK_GT(stream_laser_suspensions_, 0);
      if (--stream_laser_suspensions_ == 0) {
        laser_write_epoch_.fetch_add(1);
        if (!laser_paused_for_forward_) {
          laser_write_interrupt_.store(false);
          laser_write_cv_.notify_all();
        }
      }
    }
    if (state.eviction_suspended && feature_evict_) {
      feature_evict_->resume_triggers();
    }
  }

  folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async_with_metadata(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& metadata,
      const at::Tensor& count,
      const kv_db::RocksdbWriteMode w_mode,
      bool manage_pause = true) {
    TORCH_CHECK(metadata.is_contiguous(), "metadata must be contiguous");
    TORCH_CHECK(
        metadata.size(0) == indices.size(0),
        "metadata and indices must have the same number of rows");
    TORCH_CHECK(
        metadata.size(1) == FixedBlockPool::get_metaheader_dim<weight_type>(),
        "metadata must contain one complete metaheader per row");
    if (!manage_pause) {
      return set_kv_db_async_impl(indices, weights, count, w_mode, &metadata);
    }
    auto pause_state = pause_for_stream_write();
    SCOPE_EXIT {
      resume_after_stream_write(pause_state);
    };
    set_kv_db_async_impl(indices, weights, count, w_mode, &metadata).get();
    return folly::makeSemiFuture(std::vector<folly::Unit>());
  }

  folly::SemiFuture<std::vector<folly::Unit>> set_kv_db_async_impl(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const kv_db::RocksdbWriteMode w_mode,
      const at::Tensor* metadata = nullptr) {
    auto start_ts = perf_now_usec();
    if (w_mode != kv_db::RocksdbWriteMode::STREAM) {
      pause_ongoing_eviction();
    }
    std::vector<
        folly::Future<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>>>
        futures;
    auto before_shard_ts = perf_now_usec();
    auto shardid_to_indexes = shard_input(indices, count);
    const auto* metadata_ptr =
        metadata == nullptr ? nullptr : metadata->const_data_ptr<weight_type>();
    const auto metadata_width = metadata == nullptr ? 0 : metadata->size(1);
    write_sharding_total_duration_ += perf_now_usec() - before_shard_ts;

    for (const auto& [shard_id, indexes] : shardid_to_indexes) {
      futures.emplace_back(
          folly::via(executor_.get())
              .thenValue([this,
                          shard_id,
                          indexes,
                          &indices,
                          &weights,
                          w_mode,
                          metadata_ptr,
                          metadata_width](folly::Unit) {
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
                     w_mode,
                     metadata_ptr,
                     metadata_width,
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
                        auto before_write_lock_ts = perf_now_usec();
                        auto wlmap = kv_store_.by(shard_id).wlock();
                        local_write_acquire_lock_duration =
                            perf_now_usec() - before_write_lock_ts;
                        auto* pool = kv_store_.pool_by(shard_id);
                        for (const auto& id_index : indexes) {
                          auto id = int64_t(indices_data_ptr[id_index]);
                          // use mempool
                          weight_type* block = nullptr;
                          bool newly_allocated = false;
                          bool restored_eviction_metadata = false;
                          // First check if the key already exists
                          auto before_lookup_cache_ts = perf_now_usec();
                          auto it = wlmap->find(id);
                          local_write_lookup_cache_total_duration +=
                              perf_now_usec() - before_lookup_cache_ts;
                          if (it != wlmap->end()) {
                            block = it->second;
                          } else {
                            // Key doesn't exist, allocate new block and insert.
                            local_write_missing_load++;
                            newly_allocated = true;
                            auto before_alloc_ts = perf_now_usec();
                            block = pool->template allocate_t<weight_type>();
                            const auto* stored_metadata =
                                metadata_ptr == nullptr
                                ? nullptr
                                : metadata_ptr + id_index * metadata_width;
                            if (stored_metadata != nullptr &&
                                FixedBlockPool::get_used(stored_metadata)) {
                              restored_eviction_metadata = true;
                              std::copy(
                                  stored_metadata,
                                  stored_metadata + metadata_width,
                                  block);
                            }
                            FixedBlockPool::set_key(block, id);
                            FixedBlockPool::set_used(block, true);
                            wlmap->insert({id, block});
                            hbm_take_tombstone_locked(shard_id, id);
                            local_write_allocate_total_duration +=
                                perf_now_usec() - before_alloc_ts;
                            if (is_eviction_enabled()) {
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
                          if (is_eviction_enabled() &&
                              (w_mode != kv_db::RocksdbWriteMode::STREAM ||
                               (newly_allocated &&
                                !restored_eviction_metadata))) {
                            feature_evict_->update_feature_statistics(block);
                          }
                          auto before_copy_ts = perf_now_usec();
                          write_payload(
                              shard_id,
                              FixedBlockPool::data_ptr<weight_type>(block),
                              weights_data_ptr + id_index * stride,
                              stride);
                          if (enable_ssd_backend_ &&
                              w_mode !=
                                  kv_db::RocksdbWriteMode::FWD_ROCKSDB_READ) {
                            pool->set_dirty(block, true);
                          }
                          local_write_cache_copy_total_duration +=
                              perf_now_usec() - before_copy_ts;
                        }
                        hbm_post_invalidations_locked(shard_id);
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
          auto duration = perf_now_usec() - start_ts;
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

  /// The write epoch an enrichment write captures before it is queued.
  uint64_t laser_write_epoch() const {
    return laser_write_epoch_.load();
  }

  /// Waits until no set_embedding_cache_rows_async write is in flight.
  void wait_row_writes_done() const {
    while (row_write_in_progress_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  /// Separate write path for enrichment that runs on enrichment_executor_
  /// instead of the main executor_. This avoids competing with forward/backward
  /// for thread pool resources. Uses pause/resume to yield to fwd/bwd —
  /// removing this and reusing the main write path causes QPS regression.
  /// With skip_nonzero_rows, a key whose row is already non-zero keeps it.
  folly::SemiFuture<std::vector<folly::Unit>>
  set_kv_db_async_on_enrichment_executor(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      uint64_t write_epoch,
      bool skip_nonzero_rows = false) {
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
              .thenValue([this,
                          shard_id,
                          indexes,
                          indices,
                          weights,
                          write_epoch,
                          skip_nonzero_rows](folly::Unit) {
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "dram_kv_set_laser",
                    [this,
                     shard_id,
                     indexes,
                     &indices,
                     &weights,
                     write_epoch,
                     skip_nonzero_rows] {
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
                        if (!begin_laser_write(write_epoch)) {
                          return;
                        }
                        SCOPE_EXIT {
                          end_laser_write();
                        };

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
                              if (skip_nonzero_rows &&
                                  !FixedBlockPool::is_weights_all_zero(
                                      block, payload_end_bytes())) {
                                cursor++;
                                continue;
                              }
                            } else {
                              if (skip_nonzero_rows && hbm_ &&
                                  hbm_rows_[shard_id].contains(id)) {
                                cursor++;
                                continue;
                              }
                              // Key doesn't exist, allocate new block and
                              // insert
                              block = pool->template allocate_t<weight_type>();
                              if (block == nullptr) {
                                cursor++;
                                continue; // Skip if allocation fails
                              }
                              FixedBlockPool::set_key(block, id);
                              wlmap->insert({id, block});
                              hbm_take_tombstone_locked(shard_id, id);
                            }
                            write_payload(
                                shard_id,
                                FixedBlockPool::data_ptr<weight_type>(block),
                                weights_data_ptr + id_index * stride,
                                stride);
                            if (enable_ssd_backend_) {
                              pool->set_dirty(block, true);
                            }
                            cursor++;
                            // Check if we should pause and yield lock
                            if (is_laser_write_interrupted()) {
                              break;
                            }
                          }
                          hbm_post_invalidations_locked(shard_id);
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
          XLOG_EVERY_MS(INFO, 60000)
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
    // Writes rows under the shard read lock, which packed rows' fallback
    // bookkeeping cannot share.
    TORCH_CHECK(
        !packed_, "inference_set_kv_db_async does not support packed rows");
    TORCH_CHECK(!int8_, "inference_set_kv_db_async does not support INT8 rows");
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
                        // [tensor_offset, weight_addr]
                        std::vector<std::tuple<int64_t, weight_type*>> hit_info;
                        // [id, tensor_offset]
                        std::vector<std::tuple<int64_t, int64_t>> miss_info;
                        hit_info.reserve(indexes.size() / 2);
                        miss_info.reserve(indexes.size() / 10);
                        // feature evict configured and active
                        const bool evict_enabled = is_eviction_enabled();
                        {
                          // 1st step, collect hit/miss per inplace update chunk
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
                          hit_cnt = hit_info.size();
                          miss_cnt = miss_info.size();
                          // 2nd step, update hits under the shard rlock; the
                          // evict loop erases keys before freeing blocks, so
                          // the held rlock keeps these pointers valid.
                          for (auto& [tensor_offset, block] : hit_info) {
                            auto* data_ptr =
                                FixedBlockPool::data_ptr<weight_type>(block);
                            std::copy(
                                weights_data_ptr + tensor_offset * stride,
                                weights_data_ptr + (tensor_offset + 1) * stride,
                                data_ptr);
                            // update provided ts for existing blocks
                            if (evict_enabled &&
                                inplace_update_ts.has_value()) {
                              FixedBlockPool::set_timestamp(
                                  block, inplace_update_ts.value());
                            }
                          }
                        } // rlmap released here

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

                          // TODO: skip FixedBlockPool set_dirty here. This
                          // DRAM_SSD embedding cache path only handles
                          // backfill, where data already exists in SSD, so
                          // marking dirty would trigger a redundant flush.

                          // update provided ts for new allocated blocks
                          if (evict_enabled) {
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
    const auto write_epoch = laser_write_epoch_.load();
    auto enrichTask = folly::coro::co_invoke(
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-capturing-lambda-coroutines)
        [this,
         hashed_ids = std::move(hashed_ids),
         unhashed_ids = std::move(unhashed_ids),
         log_prefix,
         write_epoch,
         fetchFn = std::move(fetchFn),
         prepareFn =
             std::move(prepareFn)]() mutable -> folly::coro::Task<void> {
          auto start_time = facebook::WallClockUtil::NowInUsecFast();
          auto payloads = co_await fetchFn(unhashed_ids);
          auto latency_ms =
              (facebook::WallClockUtil::NowInUsecFast() - start_time) / 1000;
          const size_t non_empty_payloads =
              oneflow_enrichment::countNonEmptyPayloads(payloads);
          XLOG_EVERY_MS(INFO, 60000)
              << "[EmbeddingCacheEnrich] " << log_prefix << non_empty_payloads
              << "/" << unhashed_ids.size() << ", latency_ms: " << latency_ms;
          enrichment_query_count_.fetch_add(unhashed_ids.size());
          if (unhashed_ids.size() >= non_empty_payloads) {
            enrichment_empty_count_.fetch_add(
                unhashed_ids.size() - non_empty_payloads);
          } else {
            XLOG(WARN) << "[EmbeddingCacheEnrich] " << log_prefix
                       << "payloads.size() (" << non_empty_payloads
                       << ") > unhashed_ids.size() (" << unhashed_ids.size()
                       << ")";
          }
          if (!payloads.empty()) {
            auto result = prepareFn(hashed_ids, unhashed_ids, payloads);
            if (result.has_value()) {
              apply_serving_qdq(
                  result->weights, result->count.template item<int64_t>());
              set_kv_db_async_on_enrichment_executor(
                  result->indices, result->weights, result->count, write_epoch)
                  .via(enrichment_executor_.get())
                  .thenValue([this, result, write_epoch](
                                 const std::vector<folly::Unit>&) {
                    // Stream enriched embeddings to TrainingPsProcess for
                    // delta publishing.
                    if (enable_raw_embedding_streaming_ &&
                        begin_laser_write(write_epoch)) {
                      SCOPE_EXIT {
                        end_laser_write();
                      };
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
        });
    // NOLINTNEXTLINE(readability-redundant-smartptr-get)
    co_withExecutor(enrichment_executor_.get(), std::move(enrichTask)).start();
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
          if (!FixedBlockPool::is_weights_all_zero(
                  block, payload_end_bytes())) {
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

    XLOG_EVERY_MS(INFO, 60000)
        << "[fetch_sids_sync] cache hits: " << result_vids.size()
        << ", need fetch: " << need_fetch_unhashed.size();

    // ─── Step 2: Remote fetch for misses ───
    if (!need_fetch_unhashed.empty()) {
      const auto enrichment_type = enrichment_config_.value()->enrichment_type_;
      folly::F14FastMap<int64_t, int64_t> payloads;

      if (enrichment_type == kv_mem::EnrichmentType::ONEFLOW_OPENTAB_SID) {
        payloads = folly::coro::blockingWait(co_withExecutor(
            enrichment_executor_.get(),
            oneflow_enrichment::fetchFromOpenTab(
                open_tab_reader_,
                *enrichment_config_.value(),
                need_fetch_unhashed)));
      } else if (
          enrichment_type ==
          kv_mem::EnrichmentType::ONEFLOW_FEATURE_STORE_SID) {
        payloads = folly::coro::blockingWait(co_withExecutor(
            enrichment_executor_.get(),
            feature_store_enrichment::fetchSIDFromFeatureStore(
                *enrichment_config_.value(), need_fetch_unhashed)));
      } else {
        XLOG(WARNING) << "[fetch_sids_sync] unsupported enrichment_type: "
                      << static_cast<int64_t>(enrichment_type);
      }

      XLOG_EVERY_MS(INFO, 60000)
          << "[fetch_sids_sync] remote fetched: " << payloads.size() << "/"
          << need_fetch_unhashed.size();

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

    XLOG_EVERY_MS(INFO, 60000) << "[fetch_sids_sync] returning "
                               << result_vids.size() << " VID->SID pairs";

    return {vids_tensor, sids_tensor};
  }

  bool should_refresh_timestamp_on_enrich_hit() const {
    return refresh_timestamp_on_enrich_hit_ && is_training_ &&
        is_eviction_enabled();
  }

  /// Sets the timestamps of `ids` that are still present and non-zero to now.
  /// Every training timestamp write happens under the shard write lock, so
  /// readers holding the read lock never see a concurrent write. Rows are
  /// re-found under the lock, so one evicted since the caller's lookup is
  /// skipped. `now` is read after taking the lock, so no earlier write carries
  /// a later timestamp. A row stamped ahead of `now` (clock stepped back, or a
  /// skewed checkpoint restored without restamping) is pulled back to `now`:
  /// the evictor's unsigned age check would otherwise evict it at once.
  /// Yields exactly like the enrichment write: the lock is held across rows and
  /// released only when a training lookup or write raises the interrupt flag,
  /// which is checked after every row.
  void refresh_hit_timestamps_under_wlock(
      int shard_id,
      const std::vector<int64_t>& ids) {
    if (ids.empty()) {
      return;
    }
    int64_t refreshed = 0;
    int64_t interrupts = 0;
    int64_t wlock_us = 0;
    size_t cursor = 0;
    const auto write_epoch = laser_write_epoch_.load();
    while (cursor < ids.size()) {
      if (!begin_laser_write(write_epoch)) {
        return;
      }
      SCOPE_EXIT {
        end_laser_write();
      };
      const auto lock_start = facebook::WallClockUtil::NowInUsecFast();
      {
        auto wlmap = kv_store_.by(shard_id).wlock();
        const auto now = FixedBlockPool::current_timestamp();
        while (cursor < ids.size()) {
          const auto id = ids[cursor++];
          const auto it = wlmap->find(id);
          if (it != wlmap->end() &&
              !FixedBlockPool::is_weights_all_zero(
                  it->second, payload_end_bytes())) {
            FixedBlockPool::set_timestamp(it->second, now);
            refreshed++;
          } else if (hbm_) {
            const auto hbm_it = hbm_rows_[shard_id].find(id);
            if (hbm_it != hbm_rows_[shard_id].end()) {
              hbm_it->second = with_timestamp(hbm_it->second, now);
              refreshed++;
            }
          }
          if (is_laser_write_interrupted() && cursor < ids.size()) {
            interrupts++;
            break;
          }
        }
      }
      wlock_us += facebook::WallClockUtil::NowInUsecFast() - lock_start;
    }
    timestamp_refresh_ids_ += refreshed;
    timestamp_refresh_wlock_us_ += wlock_us;
    timestamp_refresh_interrupts_ += interrupts;
    XLOG_EVERY_MS(INFO, 60000)
        << "[EmbeddingCacheEnrich] timestamp refresh shard " << shard_id
        << ": ids=" << ids.size() << ", refreshed=" << refreshed
        << ", wlock_us=" << wlock_us << ", interrupts=" << interrupts
        << "; totals refreshed=" << timestamp_refresh_ids_.load()
        << ", wlock_us=" << timestamp_refresh_wlock_us_.load()
        << ", interrupts=" << timestamp_refresh_interrupts_.load();
  }

  /// Refresh-only pass for a query dropped while an enrichment write is in
  /// flight: queues no fetch, only marks the non-zero rows it hits as used.
  void refresh_hit_timestamps_async(
      at::Tensor hashed_indices,
      at::Tensor count) {
    if (!should_refresh_timestamp_on_enrich_hit()) {
      return;
    }
    bool expected = false;
    if (!timestamp_refresh_in_progress_.compare_exchange_strong(
            expected, true)) {
      return;
    }
    auto reset_flag_on_error = folly::makeGuard(
        [this] { timestamp_refresh_in_progress_.store(false); });
    folly::via(
        folly::getKeepAliveToken(*enrichment_executor_),
        [this,
         hashed_indices = std::move(hashed_indices),
         count = std::move(count)]() {
          for (const auto& [shard_id, indexes] :
               shard_input(hashed_indices, count)) {
            std::vector<int64_t> ids;
            ids.reserve(indexes.size());
            FBGEMM_DISPATCH_INTEGRAL_TYPES(
                hashed_indices.scalar_type(),
                "dram_refresh_enrich_hit_timestamps",
                [&] {
                  using index_t = scalar_t;
                  const auto* data = hashed_indices.const_data_ptr<index_t>();
                  for (const auto id_index : indexes) {
                    ids.push_back(int64_t(data[id_index]));
                  }
                });
            refresh_hit_timestamps_under_wlock(shard_id, ids);
          }
        })
        .thenError([](const folly::exception_wrapper& ew) {
          XLOG_EVERY_MS(ERR, 60000)
              << "[EmbeddingCacheEnrich] refresh-only timestamp pass failed: "
              << ew.what();
        })
        .ensure([this] { timestamp_refresh_in_progress_.store(false); });
    reset_flag_on_error.dismiss();
  }

  /// Writes rows the trainer already has (e.g. an item embedding feature)
  /// through the same enrichment-executor write as Laser responses, into keys
  /// that are absent or all-zero. Like a Laser request, a write that arrives
  /// while the previous one is still running is dropped.
  void set_embedding_cache_rows_async(
      at::Tensor indices,
      at::Tensor weights,
      at::Tensor count) override {
    if (!enrichment_config_.has_value()) {
      return;
    }
    bool expected = false;
    if (!row_write_in_progress_.compare_exchange_strong(expected, true)) {
      XLOG(INFO)
          << "[EmbeddingCacheEnrich] skipping row write - previous row write in progress";
      return;
    }
    const auto write_epoch = laser_write_epoch_.load();
    folly::via(enrichment_executor_.get())
        .thenValue([this,
                    write_epoch,
                    indices = std::move(indices),
                    weights = std::move(weights),
                    count = std::move(count)](folly::Unit) mutable {
          // Snapped to the serving fixed point like Laser fills, so training
          // reads what serving decodes; a no-op without serving QDQ.
          if (serving_qdq_format().bit_rate != 0) {
            weights = weights.clone();
            apply_serving_qdq(weights, count.item<int64_t>());
          }
          XLOG(INFO) << "[EmbeddingCacheEnrich] row write: "
                     << count.item<int64_t>() << " rows";
          return set_kv_db_async_on_enrichment_executor(
                     indices,
                     weights,
                     count,
                     write_epoch,
                     /*skip_nonzero_rows=*/true)
              .via(enrichment_executor_.get());
        })
        .ensure([this]() { row_write_in_progress_.store(false); });
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
      XLOG_EVERY_MS(INFO, 60000)
          << "[EmbeddingCacheEnrich] skipping - laser write already in progress";
      // A dropped query still saw these ids in use, so keep their rows alive.
      refresh_hit_timestamps_async(std::move(hashed_indices), std::move(count));
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
                            const bool refresh_hits =
                                should_refresh_timestamp_on_enrich_hit();
                            std::vector<int64_t> hit_ids;
                            if (refresh_hits) {
                              hit_ids.reserve(indexes.size());
                            }
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
                                          block, payload_end_bytes())) {
                                    zero_weight_hashed_ids.push_back(hashed_id);
                                    zero_weight_unhashed_ids.push_back(
                                        unhashed_id);
                                    zero_id_count++;
                                    in_cache_zero_count++;
                                  } else if (refresh_hits) {
                                    hit_ids.push_back(hashed_id);
                                  }
                                } else if (
                                    hbm_ &&
                                    hbm_rows_[shard_id].contains(hashed_id)) {
                                  // Its only copy is in HBM.
                                  if (refresh_hits) {
                                    hit_ids.push_back(hashed_id);
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
                            refresh_hit_timestamps_under_wlock(
                                shard_id, hit_ids);
                            if (shard_id == 0) {
                              XLOG_EVERY_MS(INFO, 60000)
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

                XLOG_EVERY_MS(INFO, 60000)
                    << "[EmbeddingCacheEnrich] found "
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
                  XLOG_EVERY_MS(INFO, 60000)
                      << "[EmbeddingCacheEnrich] skipping fetch, too many pending requests: "
                      << pending_laser_requests_.load();
                  laser_write_in_progress_.store(false);
                  return;
                }

                // Fetch from external source for zero-weight IDs
                if (!all_zero_weight_unhashed_ids.empty() &&
                    enrichment_config_.has_value()) {
                  XLOG_EVERY_MS(INFO, 60000)
                      << "[EmbeddingCacheEnrich] starting fetch for "
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
                      kv_mem::EnrichmentType::ONEFLOW_OPENTAB_ID_LIST) {
                    dispatchEnrichmentAsync(
                        std::move(all_zero_weight_hashed_ids),
                        std::move(all_zero_weight_unhashed_ids),
                        "opentab_id_list_hit: ",
                        [this](const std::vector<int64_t>& ids) {
                          return oneflow_enrichment::fetchIdListsFromOpenTab(
                              open_tab_reader_,
                              *enrichment_config_.value(),
                              ids);
                        },
                        [this](
                            const std::vector<int64_t>& h,
                            const std::vector<int64_t>& u,
                            const auto& data) {
                          return oneflow_enrichment::
                              prepareInt64ListPayloadTensors<weight_type>(
                                  h, u, data, max_D_);
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
                  } else if (
                      enrichment_type ==
                      kv_mem::EnrichmentType::IN_MEMORY_TEST_ONLY) {
                    dispatchEnrichmentAsync(
                        std::move(all_zero_weight_hashed_ids),
                        std::move(all_zero_weight_unhashed_ids),
                        "fake_hit: ",
                        [this](const std::vector<int64_t>& ids) {
                          return fake_enrichment::fetchFromFake(
                              *enrichment_config_.value(), ids);
                        },
                        [this](
                            const std::vector<int64_t>& h,
                            const std::vector<int64_t>& u,
                            const auto& data) {
                          return fake_enrichment::prepareCacheWriteTensors<
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
    auto start_ts = perf_now_usec();
    if (!is_eviction_enabled()) {
      // featre eviction is disabled
      auto duration = perf_now_usec() - start_ts;
      metadata_write_total_duration_ += duration;
      return folly::makeSemiFuture(std::vector<folly::Unit>());
    }

    CHECK_EQ(engege_rates.scalar_type(), at::ScalarType::Float);
    auto* feature_score_evict =
        dynamic_cast<FeatureScoreBasedEvict<weight_type>*>(
            feature_evict_.get());

    if (feature_score_evict == nullptr) {
      auto duration = perf_now_usec() - start_ts;
      metadata_write_total_duration_ += duration;
      // Not a feature score based eviction
      return folly::makeSemiFuture(std::vector<folly::Unit>());
    }
    pause_ongoing_eviction();
    auto before_shard_ts = perf_now_usec();
    auto shardid_to_indexes = shard_input(indices, count);
    metadata_write_sharding_total_duration_ +=
        perf_now_usec() - before_shard_ts;
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
                        auto before_write_lock_ts = perf_now_usec();
                        auto wlmap = kv_store_.by(shard_id).wlock();
                        local_write_acquire_lock_duration =
                            perf_now_usec() - before_write_lock_ts;
                        auto* pool = kv_store_.pool_by(shard_id);

                        for (const auto& id_index : indexes) {
                          auto id = int64_t(indices_data_ptr[id_index]);
                          float engege_rate = float(engage_rate_ptr[id_index]);
                          // use mempool
                          weight_type* block = nullptr;
                          auto before_lookup_cache_ts = perf_now_usec();
                          auto it = wlmap->find(id);
                          local_write_lookup_cache_total_duration +=
                              perf_now_usec() - before_lookup_cache_ts;
                          if (it != wlmap->end()) {
                            block = it->second;
                            feature_score_evict
                                ->update_feature_score_statistics(
                                    block, engege_rate, shard_id, false);
                            local_write_cache_miss += 1;
                          } else {
                            // Key doesn't exist, allocate new block and
                            // insert.
                            auto before_alloc_ts = perf_now_usec();
                            block = pool->template allocate_t<weight_type>();
                            FixedBlockPool::set_key(block, id);
                            FixedBlockPool::set_feature_score_rate(
                                block, engege_rate);
                            wlmap->insert({id, block});
                            feature_score_evict
                                ->update_feature_score_statistics(
                                    block, 0, shard_id, true);
                            local_write_allocate_total_duration +=
                                perf_now_usec() - before_alloc_ts;
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
          auto duration = perf_now_usec() - start_ts;
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
      std::optional<int64_t> width_length = std::nullopt,
      const std::optional<at::Tensor>& found_mask = std::nullopt,
      bool count_tombstone_reads = false) {
    auto read_count = count.scalar_type() == at::ScalarType::Long
        ? *(count.const_data_ptr<int64_t>())
        : *(count.const_data_ptr<int32_t>());
    uint8_t* found_mask_data_ptr = nullptr;
    if (found_mask.has_value()) {
      TORCH_CHECK(
          disable_random_init_,
          "Found-mask reads require random initialization to be disabled");
      TORCH_CHECK(found_mask->device().is_cpu());
      TORCH_CHECK_EQ(found_mask->scalar_type(), at::ScalarType::Byte);
      TORCH_CHECK_EQ(found_mask->dim(), 1);
      TORCH_CHECK_GE(found_mask->numel(), indices.numel());
      TORCH_CHECK(found_mask->is_contiguous());
      found_mask_data_ptr = found_mask->mutable_data_ptr<uint8_t>();
    }
    read_num_counts_ += read_count;
    // assuming get is called once each iteration and only by train
    // iteration(excluding state_dict)
    auto start_ts = perf_now_usec();
    pause_ongoing_eviction(); // noop calls, no impact if called multiple times
    std::vector<folly::Future<
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>>>
        futures;
    auto row_width = weights.size(1);
    auto copy_width = width_length.value_or(row_width);
    CHECK_LE(row_width, max_D_);
    CHECK_EQ(copy_width, row_width);
    auto before_shard_ts = perf_now_usec();
    constexpr int64_t kMinIndicesPerParallelChunk = 4096;
    std::shared_ptr<FlatShardInput> flat_shard_input;
    folly::F14FastMap<int, std::vector<int64_t>> shardid_to_indexes;
    if (segmented_shard_reads_ &&
        read_count >= 2 * kMinIndicesPerParallelChunk) {
      flat_shard_input = shard_input_for_read(indices, count);
    } else {
      shardid_to_indexes = shard_input(indices, count);
    }
    read_sharding_total_duration_ += perf_now_usec() - before_shard_ts;
    futures.reserve(
        flat_shard_input
            ? num_shards_
            : shardid_to_indexes.size() * dram_read_splits_per_shard_);

    struct ShardReadIndices {
      explicit ShardReadIndices(std::vector<int64_t> indexes)
          : owned(std::move(indexes)), end(owned.size()) {}

      ShardReadIndices(
          std::shared_ptr<const std::vector<int64_t>> shared,
          size_t range_begin,
          size_t range_end)
          : shared(std::move(shared)), begin(range_begin), end(range_end) {}

      ShardReadIndices(
          std::shared_ptr<const FlatPositions> flat,
          size_t range_begin,
          size_t range_end)
          : flat(std::move(flat)), begin(range_begin), end(range_end) {}

      const int64_t* data() const {
        if (flat) {
          return flat->data() + begin;
        }
        return shared ? shared->data() + begin : owned.data();
      }

      size_t size() const {
        return end - begin;
      }

      std::vector<int64_t> owned;
      std::shared_ptr<const std::vector<int64_t>> shared;
      std::shared_ptr<const FlatPositions> flat;
      size_t begin{0};
      size_t end{0};
    };

    const auto schedule_shard_read = [this,
                                      &futures,
                                      &indices,
                                      &weights,
                                      width_offset,
                                      row_width,
                                      found_mask_data_ptr,
                                      count_tombstone_reads](
                                         int shard_id,
                                         ShardReadIndices indexes) {
      futures.emplace_back(
          folly::via(executor_.get())
              .thenValue([this,
                          shard_id,
                          indexes = std::move(indexes),
                          &indices,
                          &weights,
                          width_offset,
                          row_width,
                          found_mask_data_ptr,
                          count_tombstone_reads](folly::Unit) {
                const auto* const index_positions = indexes.data();
                const auto num_indexes = indexes.size();
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
                     &indices,
                     &weights,
                     width_offset,
                     row_width,
                     index_positions,
                     num_indexes,
                     found_mask_data_ptr,
                     count_tombstone_reads,
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
                      auto before_read_lock_ts = perf_now_usec();
                      auto wlmap = kv_store_.by(shard_id).rlock();
                      local_read_aquire_lock_duration =
                          perf_now_usec() - before_read_lock_ts;

                      if (!skip_disabled_initializers_) {
                        if (!wlmap->empty() && !is_training_) {
                          row_storage_data_ptr =
                              FixedBlockPool::data_ptr<weight_type>(
                                  wlmap->begin()->second);
                        } else {
                          const auto& init_storage =
                              initializers_[shard_id]->row_storage_;
                          TORCH_CHECK(
                              init_storage.scalar_type() ==
                                  weights.scalar_type(),
                              "init_storage (",
                              toString(init_storage.scalar_type()),
                              ") and weights scalar (",
                              toString(weights.scalar_type()),
                              ") types mismatch");
                          row_storage_data_ptr =
                              init_storage.template data_ptr<weight_type>();
                        }
                      }
                      const bool use_lookup_prefetch =
                          dram_lookup_prefetch_distance_ > 0 && !wlmap->empty();
                      const auto copy_or_fill_row =
                          [this,
                           shard_id,
                           weights_data_ptr,
                           row_storage_data_ptr,
                           width_offset,
                           row_width,
                           use_lookup_prefetch,
                           found_mask_data_ptr,
                           count_tombstone_reads,
                           &local_read_cache_hit_copy_total_duration,
                           &local_read_fill_row_storage_total_duration,
                           &local_read_missing_load,
                           &local_read_hit_load](
                              const int64_t weights_row_index,
                              const int64_t weight_idx,
                              const weight_type* const cached_data_ptr) {
                            if (found_mask_data_ptr != nullptr) {
                              found_mask_data_ptr[weights_row_index] =
                                  cached_data_ptr != nullptr;
                            }
                            if (cached_data_ptr == nullptr) {
                              local_read_missing_load++;
                              if (count_tombstone_reads &&
                                  hbm_rows_[shard_id].contains(weight_idx)) {
                                if (found_mask_data_ptr != nullptr) {
                                  // Present, but only in HBM: a found-mask
                                  // reader (the write-side cache fill) skips it
                                  // instead of retrying it.
                                  found_mask_data_ptr[weights_row_index] =
                                      kFoundOnlyInHbm;
                                  hbm_tombstone_mask_reads_++;
                                } else {
                                  hbm_tombstone_reads_++;
                                }
                              }
                              if (found_mask_data_ptr != nullptr) {
                                return;
                              }
                              const auto weight_width = use_lookup_prefetch
                                  ? row_width
                                  : get_width_for_weights(
                                        weight_idx, width_offset, row_width);
                              const auto before_fill_from_row_storage_ts =
                                  perf_now_usec();
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
                                  perf_now_usec() -
                                  before_fill_from_row_storage_ts;
                              return;
                            }

                            const auto before_cache_hit_copy_ts =
                                perf_now_usec();
                            read_payload(
                                shard_id,
                                cached_data_ptr,
                                width_offset,
                                row_width,
                                &(weights_data_ptr
                                      [weights_row_index * row_width]));
                            local_read_cache_hit_copy_total_duration +=
                                perf_now_usec() - before_cache_hit_copy_ts;
                            local_read_hit_load++;
                          };

                      if (!use_lookup_prefetch) {
                        for (size_t i = 0; i < num_indexes; ++i) {
                          const auto weights_row_index = index_positions[i];
                          const auto weight_idx =
                              int64_t(indices_data_ptr[weights_row_index]);
                          const auto before_lookup_cache_ts = perf_now_usec();
                          const auto cached_iter = wlmap->find(weight_idx);
                          local_read_lookup_cache_total_duration +=
                              perf_now_usec() - before_lookup_cache_ts;
                          copy_or_fill_row(
                              weights_row_index,
                              weight_idx,
                              cached_iter == wlmap->end()
                                  ? nullptr
                                  : FixedBlockPool::data_ptr<weight_type>(
                                        cached_iter->second));
                        }
                        return;
                      }

                      constexpr size_t kMaxLookupPrefetchDistance = 64;
                      using LookupTokens = std::array<
                          folly::F14HashToken,
                          kMaxLookupPrefetchDistance>;
                      LookupTokens tokens;
                      std::array<const weight_type*, kMaxLookupPrefetchDistance>
                          prefetched_rows;
                      std::array<int64_t, kMaxLookupPrefetchDistance>
                          prefetched_row_indices;
                      std::array<int64_t, kMaxLookupPrefetchDistance>
                          prefetched_weight_indices;
                      const auto prefetch_distance = std::min(
                          static_cast<size_t>(dram_lookup_prefetch_distance_),
                          num_indexes);
                      for (size_t i = 0; i < prefetch_distance; ++i) {
                        const auto weight_idx =
                            int64_t(indices_data_ptr[index_positions[i]]);
                        tokens[i] = wlmap->prehash(weight_idx);
                        wlmap->prefetch(tokens[i]);
                      }
                      for (size_t i = 0; i < num_indexes + prefetch_distance;
                           ++i) {
                        const auto token_index = i % prefetch_distance;
                        if (i >= prefetch_distance) {
                          copy_or_fill_row(
                              prefetched_row_indices[token_index],
                              prefetched_weight_indices[token_index],
                              prefetched_rows[token_index]);
                        }
                        if (i >= num_indexes) {
                          continue;
                        }

                        const auto weights_row_index = index_positions[i];
                        const auto weight_idx =
                            int64_t(indices_data_ptr[weights_row_index]);
                        const auto before_lookup_cache_ts = perf_now_usec();
                        const auto cached_iter =
                            wlmap->find(tokens[token_index], weight_idx);
                        local_read_lookup_cache_total_duration +=
                            perf_now_usec() - before_lookup_cache_ts;
                        const weight_type* cached_data_ptr = nullptr;
                        if (cached_iter != wlmap->end()) {
                          cached_data_ptr =
                              FixedBlockPool::data_ptr<weight_type>(
                                  cached_iter->second);
                          constexpr size_t kCacheLineBytes = 64;
                          const auto* const row_bytes =
                              reinterpret_cast<const char*>(
                                  cached_data_ptr + width_offset);
                          if (packed_ || int8_) {
                            // Every line the payload touches: a payload
                            // shorter than its block often straddles one
                            // more line than its length suggests.
                            const auto first =
                                reinterpret_cast<uintptr_t>(row_bytes) &
                                ~(kCacheLineBytes - 1);
                            const auto end =
                                reinterpret_cast<uintptr_t>(row_bytes) +
                                payload_end_bytes() -
                                sizeof(FixedBlockPool::MetaHeader);
                            for (auto line = first; line < end;
                                 line += kCacheLineBytes) {
                              __builtin_prefetch(
                                  // NOLINTNEXTLINE(performance-no-int-to-ptr)
                                  reinterpret_cast<const char*>(line),
                                  /*rw=*/0,
                                  /*locality=*/1);
                            }
                          } else {
                            const auto bytes_to_copy =
                                static_cast<size_t>(row_width) *
                                sizeof(weight_type);
                            for (size_t byte = 0; byte < bytes_to_copy;
                                 byte += kCacheLineBytes) {
                              __builtin_prefetch(
                                  row_bytes + byte, /*rw=*/0, /*locality=*/1);
                            }
                          }
                        }
                        prefetched_row_indices[token_index] = weights_row_index;
                        prefetched_weight_indices[token_index] = weight_idx;
                        prefetched_rows[token_index] = cached_data_ptr;
                        const auto next = i + prefetch_distance;
                        if (next < num_indexes) {
                          const auto next_weight_idx =
                              int64_t(indices_data_ptr[index_positions[next]]);
                          tokens[token_index] = wlmap->prehash(next_weight_idx);
                          wlmap->prefetch(tokens[token_index]);
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
    };

    if (flat_shard_input) {
      for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
        const auto begin = flat_shard_input->shard_offsets[shard_id];
        const auto end = flat_shard_input->shard_offsets[shard_id + 1];
        if (begin != end) {
          schedule_shard_read(
              shard_id,
              ShardReadIndices(flat_shard_input->positions, begin, end));
        }
      }
    } else {
      constexpr size_t kMinReadIndicesPerSplit = 2048;
      for (auto& [shard_id, indexes] : shardid_to_indexes) {
        const auto max_useful_splits =
            std::max<size_t>(1, indexes.size() / kMinReadIndicesPerSplit);
        const auto num_splits = std::min(
            static_cast<size_t>(dram_read_splits_per_shard_),
            max_useful_splits);
        if (num_splits == 1) {
          schedule_shard_read(shard_id, ShardReadIndices(std::move(indexes)));
          continue;
        }

        const auto indices_per_split =
            (indexes.size() + num_splits - 1) / num_splits;
        std::shared_ptr<const std::vector<int64_t>> shared_indexes;
        if (dram_read_split_zero_copy_) {
          shared_indexes =
              std::make_shared<const std::vector<int64_t>>(std::move(indexes));
        }
        for (size_t split = 0; split < num_splits; ++split) {
          const auto begin = split * indices_per_split;
          const auto indexes_size =
              shared_indexes ? shared_indexes->size() : indexes.size();
          const auto end = std::min(indexes_size, begin + indices_per_split);
          if (begin == end) {
            break;
          }
          if (shared_indexes) {
            schedule_shard_read(
                shard_id, ShardReadIndices(shared_indexes, begin, end));
          } else {
            schedule_shard_read(
                shard_id,
                ShardReadIndices(
                    std::vector<int64_t>(
                        indexes.begin() + begin, indexes.begin() + end)));
          }
        }
      }
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
          auto duration = perf_now_usec() - start_ts;
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
    return get_kv_db_async_impl(
        indices,
        weights,
        count,
        0,
        std::nullopt,
        std::nullopt,
        /*count_tombstone_reads=*/hbm_.has_value());
  }

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async_with_found_mask(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      const at::Tensor& found_mask) override {
    // A zero mask entry makes the corresponding weights row invalid. Leave
    // that row untouched so callers can reuse scratch storage without a full
    // clear; consumers must gate every read of weights on found_mask. Entries
    // at positions >= count remain untouched in both tensors.
    current_iter_++;
    return get_kv_db_async_impl(
        indices,
        weights,
        count,
        0,
        std::nullopt,
        found_mask,
        /*count_tombstone_reads=*/hbm_.has_value());
  }

  bool supports_found_mask_reads() const override {
    return disable_random_init_;
  }

  void set_range_to_storage(
      const at::Tensor& weights,
      const int64_t start,
      const int64_t length) override {
    if (int8_ && weights.scalar_type() == at::kByte) {
      set_int8_rows_with_metaheader(weights);
      return;
    }
    if (backend_return_whole_row_) {
      auto pause_state = pause_for_stream_write();
      SCOPE_EXIT {
        resume_after_stream_write(pause_state);
      };
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

    if (int8_ && weights.scalar_type() == at::kByte) {
      CHECK_EQ(width_offset, 0);
      get_int8_rows_with_metaheader(seq_indices, weights);
      return;
    }
    auto tombstoned = hbm_tombstoned_positions(seq_indices);
    if (backend_return_whole_row_) {
      get_kv_with_metaheader_from_storage(seq_indices, weights);
      hbm_fill_rows(
          seq_indices, weights, 0, /*whole_row=*/true, std::move(tombstoned));
    } else {
      const auto count = at::tensor({length}, at::ScalarType::Long);
      get_kv_db_async_impl(
          seq_indices, weights, count, width_offset, width_length)
          .wait();
      hbm_fill_rows(
          seq_indices,
          weights,
          width_offset,
          /*whole_row=*/false,
          std::move(tombstoned));
    }

    // this is called by checkpoint mostly, and checkpoint should wait until
    // eviction finishes so that we could reacha consistent state before/after
    // state_dict() calls
    // TODO: assert there isn't any eviction including paused
  }

  void set_kv_to_storage(const at::Tensor& ids, const at::Tensor& weights)
      override {
    if (int8_ && weights.scalar_type() == at::kByte) {
      set_int8_rows_with_metaheader(weights, ids);
      return;
    }
    if (backend_return_whole_row_) {
      auto pause_state = pause_for_stream_write();
      SCOPE_EXIT {
        resume_after_stream_write(pause_state);
      };
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

    if (int8_ && weights.scalar_type() == at::kByte) {
      CHECK_EQ(width_offset, 0);
      get_int8_rows_with_metaheader(ids, weights);
      return;
    }
    auto tombstoned = hbm_tombstoned_positions(ids);
    if (backend_return_whole_row_) {
      get_kv_with_metaheader_from_storage(
          ids, weights, width_offset, width_length);
    } else {
      const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
      get_kv_db_async_impl(ids, weights, count, width_offset, width_length)
          .wait();
    }
    hbm_fill_rows(
        ids,
        weights,
        width_offset,
        backend_return_whole_row_,
        std::move(tombstoned));
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

  void update_kv_with_metaheader_if_exists(
      const at::Tensor& ids,
      const at::Tensor& weights_with_metaheader) {
    const auto count = at::tensor({ids.size(0)}, at::ScalarType::Long);
    set_kv_db_with_metaheader_async_impl<false>(
        ids, weights_with_metaheader, count)
        .get();
  }

  /// Raw INT8 rows for the checkpoint: row i of `rows` ([n,
  /// get_int8_row_bytes()] uint8) gets the bytes of ids[i]'s block (MetaHeader,
  /// codes, fp32 scale and bias, fp16 tail columns), the HBM copy of a row
  /// that lives only in HBM, or zeros for a missing id.
  void get_int8_rows_with_metaheader(
      const at::Tensor& ids,
      const at::Tensor& rows) {
    check_int8_rows(rows);
    const auto id_values = ids.to(at::kLong).contiguous();
    CHECK_EQ(id_values.numel(), rows.size(0));
    auto tombstoned = hbm_tombstoned_positions(id_values);
    read_int8_blocks(id_values, rows);
    hbm_fill_int8_rows(id_values, rows, std::move(tombstoned));
  }

  /// Writes raw INT8 rows (as read by get_int8_rows_with_metaheader, the key
  /// in each row's first 8 bytes) without re-quantizing them.
  void set_int8_rows_with_metaheader(
      const at::Tensor& rows,
      std::optional<at::Tensor> ids = std::nullopt) {
    TORCH_CHECK(int8_, "uint8 rows need int8_storage");
    TORCH_CHECK(
        rows.dim() == 2 && rows.scalar_type() == at::kByte &&
            rows.is_contiguous() && rows.device().is_cpu(),
        "INT8 rows must be a contiguous 2D uint8 CPU tensor");
    const int64_t required_bytes =
        sizeof(FixedBlockPool::MetaHeader) + int8_->row_bytes();
    TORCH_CHECK(
        rows.size(1) >= required_bytes &&
            (rows.size(1) - required_bytes) % sizeof(weight_type) == 0,
        "INT8 rows need a MetaHeader, codes, fp32 qparams and an fp16 tail");
    // A complete-row restore, like set_kv_with_metaheader_to_storage.
    auto pause_state = pause_for_stream_write();
    SCOPE_EXIT {
      resume_after_stream_write(pause_state);
    };
    const int64_t n = rows.size(0);
    const int64_t row_bytes = rows.size(1);
    const auto* src = rows.const_data_ptr<uint8_t>();
    std::vector<int64_t> keys(n);
    if (ids.has_value()) {
      const auto local_ids = ids->to(at::kLong).contiguous();
      TORCH_CHECK(local_ids.device().is_cpu() && local_ids.numel() == n);
      std::copy(
          local_ids.const_data_ptr<int64_t>(),
          local_ids.const_data_ptr<int64_t>() + n,
          keys.begin());
    } else {
      for (int64_t i = 0; i < n; ++i) {
        std::memcpy(&keys[i], src + i * row_bytes, sizeof(int64_t));
      }
    }
    const auto indices = torch::from_blob(keys.data(), {n}, torch::kInt64);
    const auto count = at::tensor({n}, at::ScalarType::Long);
    std::vector<folly::Future<folly::Unit>> futures;
    for (const auto& [shard_id, indexes] : shard_input(indices, count)) {
      futures.emplace_back(
          folly::via(executor_.get())
              .thenValue([this, shard_id, indexes, &keys, src, row_bytes](
                             folly::Unit) {
                set_int8_rows_of_shard(shard_id, indexes, keys, src, row_bytes);
              }));
    }
    folly::collect(std::move(futures)).get();
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
        // Rows only in the exclusive HBM tier count too, so the trigger
        // bounds every row the rank holds (and publishes).
        auto used_id_count = get_num_rows() + hbm_row_count();
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

  /// [fp16 rows kept for packed rows with no serving fixed point, packed
  /// rows read with the fallback marker but no fp16 row (expected 0)]
  std::vector<int64_t> packed_fallback_stats() {
    int64_t rows = 0;
    for (size_t shard_id = 0; shard_id < fallback_rows_.size(); ++shard_id) {
      const auto lock = kv_store_.by(static_cast<int>(shard_id)).rlock();
      rows += static_cast<int64_t>(fallback_rows_[shard_id].size());
    }
    return {rows, packed_fallback_missing_.load()};
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
    laser_paused_for_forward_ = true;
    laser_write_interrupt_.store(true);
  }

  /// Resume laser write operations (called after forward/backward)
  void resume_laser_write() {
    std::unique_lock<std::mutex> lock(laser_write_mutex_);
    laser_paused_for_forward_ = false;
    if (stream_laser_suspensions_ > 0) {
      return;
    }
    laser_write_interrupt_.store(false);
    laser_write_cv_.notify_all();
  }

  /// Wait until laser write is resumed if paused (like eviction pattern)
  /// Returns immediately if not interrupted, otherwise waits for resume
  bool begin_laser_write(uint64_t write_epoch) {
    std::unique_lock<std::mutex> lock(laser_write_mutex_);
    laser_write_cv_.wait(
        lock, [this] { return !laser_write_interrupt_.load(); });
    if (laser_write_epoch_.load() != write_epoch) {
      return false;
    }
    ++active_laser_writers_;
    return true;
  }

  void end_laser_write() {
    std::scoped_lock lock(laser_write_mutex_);
    CHECK_GT(active_laser_writers_, 0);
    --active_laser_writers_;
    laser_write_cv_.notify_all();
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

  /// Get the feature evict object for callback wiring.
  /// Returns nullptr if feature eviction is disabled.
  FeatureEvict<weight_type>* get_feature_evict() {
    return feature_evict_.get();
  }

  /// Access the internal kv_store for flush iteration.
  auto& get_kv_store() {
    return kv_store_;
  }

  int64_t get_num_shards() const {
    return num_shards_;
  }

  size_t get_executor_num_threads() const {
    return executor_->numThreads();
  }

  bool get_segmented_shard_reads_enabled() const {
    return segmented_shard_reads_;
  }

  int64_t get_block_size() const {
    return block_size_;
  }

  /// Packed writes that rewrote a row as its serving fixed point, and writes
  /// of rows with none, which are kept as fp16.
  std::pair<int64_t, int64_t> get_packed_write_counts() const {
    return {packed_requantized_writes_.load(), packed_fallback_writes_.load()};
  }

  /// INT8 row writes [needing more than one re-quantization, with non-finite
  /// values, not converged].
  std::vector<int64_t> get_int8_write_counts() const {
    return {
        int8_multi_pass_writes_.load(),
        int8_non_finite_writes_.load(),
        int8_not_converged_writes_.load()};
  }

  /// Bytes of a raw INT8 checkpoint row (MetaHeader, codes, fp32 scale and
  /// bias, fp16 tail columns), or 0 without int8_storage.
  int64_t get_int8_row_bytes() override {
    return int8_ ? static_cast<int64_t>(payload_end_bytes()) : 0;
  }

  /// Raw payloads for HBM promotion. Missing rows are zero-filled. All tensors
  /// must be CPU views, including host-mapped workspaces used by CUDA callers.
  void get_int8_payloads(
      const at::Tensor& indices,
      const at::Tensor& rows,
      const at::Tensor& count) {
    TORCH_CHECK(int8_, "INT8 DRAM storage is disabled");
    check_int8_payload_args(indices, rows, count);
    // A promotion capped at promote_max_rows passes only that many keys with
    // the uncapped miss count.
    const int64_t n = std::min<int64_t>(count.item<int64_t>(), indices.numel());
    TORCH_CHECK(n >= 0);
    if (n == 0) {
      return;
    }
    const auto ids = indices.to(at::kLong);
    const auto* keys = ids.const_data_ptr<int64_t>();
    auto* dst = rows.data_ptr<uint8_t>();
    const int64_t row_bytes = int8_->row_bytes();
    std::memset(dst, 0, n * row_bytes);
    auto by_shard = shard_input(ids, at::tensor({n}, at::kLong));
    std::vector<folly::Future<folly::Unit>> futures;
    for (auto& [shard_id, positions] : by_shard) {
      futures.push_back(
          folly::via(
              executor_.get(), [&, shard_id, positions = std::move(positions)] {
                const auto lock = kv_store_.by(shard_id).rlock();
                for (const auto pos : positions) {
                  const auto it = lock->find(keys[pos]);
                  if (it != lock->end()) {
                    std::memcpy(
                        dst + pos * row_bytes,
                        FixedBlockPool::data_ptr<weight_type>(it->second),
                        row_bytes);
                  }
                }
              }));
    }
    folly::collect(std::move(futures)).get();
  }

  void get_int8_payloads_cuda(
      at::Tensor indices_cpu,
      at::Tensor rows_cpu,
      at::Tensor count_cpu) {
    TORCH_CHECK(int8_, "INT8 DRAM storage is disabled");
    hbm_rethrow_async_error();
    // Checked here: the host callback cannot raise.
    check_int8_payload_args(indices_cpu, rows_cpu, count_cpu);
    auto functor = std::make_unique<std::function<void()>>(
        [self =
             std::static_pointer_cast<DramKVEmbeddingCache>(shared_from_this()),
         indices = std::move(indices_cpu),
         rows = std::move(rows_cpu),
         count = std::move(count_cpu)]() mutable {
          self->hbm_run_async([&] {
            try {
              self->get_int8_payloads(indices, rows, count);
            } catch (...) {
              // Zero rows are skipped by the exclusive insert.
              rows.zero_();
              throw;
            }
          });
        });
    AT_CUDA_CHECK(cudaLaunchHostFunc(
        at::cuda::getCurrentCUDAStream(),
        kv_db_utils::cuda_host_func,
        functor.get()));
    // The callback owns it once launched.
    functor.release();
  }

  void check_int8_payload_args(
      const at::Tensor& indices,
      const at::Tensor& rows,
      const at::Tensor& count) const {
    TORCH_CHECK(
        indices.device().is_cpu() && indices.is_contiguous() &&
            indices.dim() == 1,
        "INT8 payload indices must be a contiguous 1D CPU tensor");
    TORCH_CHECK(
        rows.device().is_cpu() && rows.is_contiguous() &&
            rows.scalar_type() == at::kByte && rows.dim() == 2 &&
            rows.size(0) == indices.numel() &&
            rows.size(1) == int8_->row_bytes(),
        "INT8 payload rows must be a contiguous CPU uint8 [n, ",
        int8_->row_bytes(),
        "] tensor matching the indices");
    TORCH_CHECK(
        count.device().is_cpu() && count.numel() == 1,
        "INT8 payload count must be a one-element CPU tensor");
  }

  /// Exclusive HBM tier: the owner-local cache's INT8 copy is the only copy of
  /// a promoted row. The backend keeps the row's MetaHeader word (timestamp,
  /// count) as a tombstone in hbm_rows_ instead of a block, so lookups through
  /// the GPU cache, enrichment, eviction counts and checkpoints still see it.
  /// `cache_state` ([C, 1] int64 tags), `cache_weights` ([C, dim + 8] uint8
  /// codes then fp32 scale and bias; columns past the enrichment dim, unused
  /// optimizer state in an embedding cache, may be left out and then read back
  /// as zero) and `last_access` ([C] int32 seconds) are
  /// the GPU cache's tensors (CPU tensors in tests).
  ///
  /// Invariant: a tombstone exists only while the cache slot of its key holds
  /// that key. GPU inserts only claim empty slots, and every other tag change
  /// and every tombstone change runs on hbm_executor_ (one thread), each
  /// checked against the other side, so a row is never in neither tier.
  void exclusive_hbm_register(
      at::Tensor cache_state,
      at::Tensor cache_weights,
      at::Tensor last_access) {
    TORCH_CHECK(!hbm_, "exclusive HBM tier is already registered");
    TORCH_CHECK(
        (std::is_same_v<weight_type, at::Half>),
        "exclusive HBM tier needs FP16 weights");
    TORCH_CHECK(
        disable_random_init_, "exclusive HBM tier needs disable_random_init");
    TORCH_CHECK(
        !enable_ssd_backend_,
        "exclusive HBM tier does not support the SSD backend");
    TORCH_CHECK(
        !feature_evict_config_.has_value() ||
            (feature_evict_config_.value()->trigger_strategy_ !=
                 EvictTriggerStrategy::BY_L2WEIGHT &&
             feature_evict_config_.value()->trigger_strategy_ !=
                 EvictTriggerStrategy::BY_FEATURE_SCORE),
        "exclusive HBM tier does not support L2 weight or feature score "
        "eviction");
    TORCH_CHECK(
        !enrichment_config_.has_value() ||
            enrichment_config_.value()->enrichment_type_ ==
                EnrichmentType::IGR_LASER_EMBEDDING ||
            enrichment_config_.value()->enrichment_type_ ==
                EnrichmentType::IN_MEMORY_TEST_ONLY,
        "exclusive HBM tier needs an embedding payload");
    TORCH_CHECK(
        cache_state.dim() == 2 && cache_state.size(1) == 1 &&
        cache_state.scalar_type() == at::kLong);
    // hbm_slot() hashes into a uint32_t slot count, as cache_slot() does.
    TORCH_CHECK(
        cache_state.size(0) <= std::numeric_limits<uint32_t>::max(),
        "exclusive HBM tier supports at most 2^32 - 1 slots");
    TORCH_CHECK(
        cache_weights.dim() == 2 && cache_weights.scalar_type() == at::kByte &&
        cache_weights.size(0) == cache_state.size(0));
    const int64_t hbm_dim = cache_weights.size(1) - kHbmQparamsBytes;
    TORCH_CHECK(
        !int8_ || hbm_dim == int8_->dim,
        "INT8 DRAM and exclusive HBM must hold the same embedding dimension");
    TORCH_CHECK(
        hbm_dim > 0 && hbm_dim <= max_D_ &&
            (!enrichment_config_.has_value() ||
             enrichment_config_.value()->enrichment_dim_ <= hbm_dim),
        "exclusive HBM rows hold ",
        hbm_dim,
        " columns, which must cover the enrichment dim and fit max_D ",
        max_D_);
    TORCH_CHECK(
        last_access.dim() == 1 && last_access.scalar_type() == at::kInt &&
        last_access.size(0) == cache_state.size(0));
    // Sized before hbm_ is set: writers check hbm_ and then index these.
    hbm_rows_.resize(num_shards_);
    hbm_pending_invalidations_.resize(num_shards_);
    hbm_ = HbmTier{
        std::move(cache_state),
        std::move(cache_weights),
        std::move(last_access),
        std::nullopt,
        hbm_dim};
    if (hbm_->cache_state.is_cuda()) {
      hbm_->stream = at::cuda::getStreamFromPool(
          /*isHighPriority=*/false, hbm_->cache_state.device().index());
    }
    hbm_chunk_rows_ = std::max<int64_t>(
        1, env_int64("KVZCH_EXCLUSIVE_HBM_CHUNK_ROWS", 65536));
    hbm_ttl_s_ =
        60 * env_int64("KVZCH_EXCLUSIVE_HBM_TTL_MINS", table_ttl_mins());
    hbm_ttl_demote_ = env_int64("KVZCH_EXCLUSIVE_HBM_TTL_DEMOTE", 1) != 0;
    const auto sweep_s = env_int64("KVZCH_EXCLUSIVE_HBM_SWEEP_S", 60);
    hbm_executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(1);
    if (feature_evict_) {
      feature_evict_->set_untracked_rows_counter(
          [this] { return static_cast<int64_t>(hbm_row_count()); });
    }
    if (hbm_ttl_s_ > 0 && sweep_s > 0) {
      hbm_scheduler_.addFunction(
          [this] {
            hbm_executor_->add([this] {
              hbm_run_async([this] { exclusive_hbm_sweep_now(); });
            });
          },
          std::chrono::seconds(sweep_s),
          "exclusive_hbm_sweep",
          std::chrono::seconds(sweep_s));
      hbm_scheduler_.start();
    }
    LOG(INFO) << "KVZCH exclusive HBM tier: slots=" << hbm_->cache_state.size(0)
              << " row_bytes=" << hbm_->cache_weights.size(1)
              << " chunk_rows=" << hbm_chunk_rows_ << " ttl_s=" << hbm_ttl_s_
              << " ttl_demote=" << hbm_ttl_demote_ << " sweep_s=" << sweep_s;
  }

  bool exclusive_hbm_enabled() const {
    return hbm_.has_value();
  }

  /// Rows whose only copy is in the exclusive HBM tier (0 without it).
  size_t hbm_row_count() {
    if (!hbm_) {
      return 0;
    }
    size_t rows = 0;
    for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
      const auto lock = kv_store_.by(shard_id).rlock();
      rows += hbm_rows_[shard_id].size();
    }
    return rows;
  }

  /// Queues the promotion of the keys a GPU insert claimed: once earlier GPU
  /// work on the current stream is done, a host callback copies the first
  /// `promoted_count_cpu[0]` keys of `promoted_keys_cpu` (host views of the
  /// insert's host-mapped outputs) and hands them to hbm_executor_.
  void exclusive_hbm_promote_cuda(
      at::Tensor promoted_keys_cpu,
      at::Tensor promoted_count_cpu) {
    TORCH_CHECK(hbm_, "exclusive HBM tier is not registered");
    hbm_rethrow_async_error();
    // Checked here: the host callback cannot raise.
    TORCH_CHECK(
        promoted_keys_cpu.is_cpu() &&
            promoted_keys_cpu.scalar_type() == at::kLong &&
            promoted_keys_cpu.is_contiguous(),
        "promoted keys must be a contiguous CPU int64 tensor");
    TORCH_CHECK(
        promoted_count_cpu.is_cpu() &&
            promoted_count_cpu.scalar_type() == at::kInt &&
            promoted_count_cpu.numel() >= 1,
        "promoted count must be a CPU int32 tensor");
    // Holds the cache alive until the callback runs.
    auto functor = std::make_unique<std::function<void()>>(
        [self =
             std::static_pointer_cast<DramKVEmbeddingCache>(shared_from_this()),
         keys = std::move(promoted_keys_cpu),
         count = std::move(promoted_count_cpu)]() {
          self->hbm_run_async([&] {
            const auto n = std::min<int64_t>(
                *count.const_data_ptr<int32_t>(), keys.numel());
            const auto* data = keys.const_data_ptr<int64_t>();
            self->exclusive_hbm_promote_async(
                std::vector<int64_t>(data, data + n));
          });
        });
    AT_CUDA_CHECK(cudaLaunchHostFunc(
        at::cuda::getCurrentCUDAStream(),
        kv_db_utils::cuda_host_func,
        functor.get()));
    // The callback owns it once launched.
    functor.release();
  }

  void exclusive_hbm_promote_async(std::vector<int64_t> keys) {
    if (keys.empty()) {
      return;
    }
    hbm_executor_->add([this, keys = std::move(keys)]() mutable {
      hbm_run_async([&] {
        try {
          hbm_promote(keys);
        } catch (...) {
          // A claimed slot without a tombstone would serve a stale row forever.
          hbm_clear_untombstoned_slots(keys);
          throw;
        }
      });
    });
  }

  /// Moves every HBM row back to the backend, emptying its slots. Callers
  /// must first let earlier GPU inserts and their promotion callbacks finish.
  int64_t exclusive_hbm_demote_all() {
    TORCH_CHECK(hbm_, "exclusive HBM tier is not registered");
    hbm_rethrow_async_error();
    return via_hbm_executor([this] {
      int64_t moved = 0;
      for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
        moved += hbm_demote(hbm_keys_of_shard(shard_id), /*drop=*/false);
      }
      return moved;
    });
  }

  /// Runs one TTL pass now: HBM rows whose last access (the later of the
  /// tombstone timestamp and the slot's GPU access time) is older than the TTL
  /// are moved back to the backend, or dropped from both tiers when
  /// KVZCH_EXCLUSIVE_HBM_TTL_DEMOTE=0. Returns the number of rows that left.
  int64_t exclusive_hbm_sweep(int64_t now = -1) {
    TORCH_CHECK(hbm_, "exclusive HBM tier is not registered");
    return via_hbm_executor(
        [this, now] { return exclusive_hbm_sweep_now(now); });
  }

  /// Waits for every queued promotion, invalidation and sweep.
  void exclusive_hbm_wait() {
    if (hbm_) {
      via_hbm_executor([] { return int64_t{0}; });
      hbm_rethrow_async_error();
    }
  }

  /// Tests: makes the next promotion fail after its GPU insert claimed the
  /// slots, as a CUDA error on the HBM stream would.
  void fail_next_hbm_promotion_for_test() {
    hbm_fail_next_promotion_.store(true);
  }

  /// Found-mask value of a row whose only copy is in HBM.
  static constexpr uint8_t kFoundOnlyInHbm = 2;

  /// [HBM rows, promoted, promotions skipped (slot no longer held the key, or
  /// the backend row changed or was evicted after the GPU read), demoted,
  /// dropped, invalidated, rows rewritten to a serving fixed point on demotion,
  /// lookups that found only a tombstone (a lost row; expected 0), found-mask
  /// reads that found only a tombstone (reported as kFoundOnlyInHbm)]
  std::vector<int64_t> exclusive_hbm_stats() {
    hbm_rethrow_async_error();
    int64_t rows = 0;
    if (hbm_) {
      for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
        const auto lock = kv_store_.by(shard_id).rlock();
        rows += static_cast<int64_t>(hbm_rows_[shard_id].size());
      }
    }
    return {
        rows,
        hbm_promoted_.load(),
        hbm_promote_skipped_.load(),
        hbm_demoted_.load(),
        hbm_dropped_.load(),
        hbm_invalidated_.load(),
        hbm_requantized_.load(),
        hbm_tombstone_reads_.load(),
        hbm_tombstone_mask_reads_.load()};
  }

  int64_t serving_qdq_existing_rows() override {
    if constexpr (!std::is_same_v<weight_type, at::Half>) {
      return -1;
    } else {
      const auto format = serving_qdq_format();
      if (format.bit_rate == 0) {
        return 0;
      }
      const auto start_ts = facebook::WallClockUtil::NowInUsecFast();
      const int64_t dim = enrichment_config_.value()->enrichment_dim_;
      serving_qdq::check_round_trip_args(dim, format);
      TORCH_CHECK(dim <= max_D_, "serving_qdq dim ", dim, " > max_D ", max_D_);
      std::vector<folly::Future<serving_qdq::Stats>> futures;
      futures.reserve(num_shards_);
      for (int64_t shard_id = 0; shard_id < num_shards_; ++shard_id) {
        futures.emplace_back(
            folly::via(executor_.get())
                .thenValue([this, shard_id, dim, format](folly::Unit) {
                  return serving_qdq_shard(shard_id, dim, format);
                }));
      }
      serving_qdq::Stats total;
      for (const auto& stats : folly::collect(std::move(futures)).get()) {
        total += stats;
      }
      XLOG(INFO) << "[EmbeddingCacheEnrich] serving_qdq_existing_rows " << total
                 << " packed=" << packed_.has_value()
                 << " packed_requantized_writes="
                 << packed_requantized_writes_.load()
                 << " packed_fallback_writes=" << packed_fallback_writes_.load()
                 << " latency_ms="
                 << (facebook::WallClockUtil::NowInUsecFast() - start_ts) /
              1000;
      return total.changed;
    }
  }

 private:
  using FlatPositions = folly::small_vector<int64_t, 0>;

  /// Publish encoding to round-trip rows through; bit_rate 0 means off. Only
  /// embedding payloads qualify; SID payloads are bit-cast integers.
  serving_qdq::Format serving_qdq_format() const {
    if (!enrichment_config_.has_value()) {
      return {};
    }
    const auto& config = *enrichment_config_.value();
    const bool embedding_payload = config.enrichment_type_ ==
            kv_mem::EnrichmentType::IGR_LASER_EMBEDDING ||
        config.enrichment_type_ == kv_mem::EnrichmentType::IN_MEMORY_TEST_ONLY;
    if (!embedding_payload) {
      return {};
    }
    return {
        static_cast<int>(config.serving_qdq_bit_rate_),
        config.serving_qdq_group_size_};
  }

  /// Round-trips the first `num_rows` rows of a CPU write tensor in place.
  void apply_serving_qdq(at::Tensor& weights, int64_t num_rows) const {
    if constexpr (std::is_same_v<weight_type, at::Half>) {
      const auto format = serving_qdq_format();
      if (format.bit_rate == 0 || num_rows == 0) {
        return;
      }
      CHECK(weights.is_contiguous());
      CHECK(weights.device().is_cpu());
      CHECK_EQ(weights.scalar_type(), at::kHalf);
      const auto stats = serving_qdq::round_trip_rows(
          weights.data_ptr<at::Half>(),
          num_rows,
          weights.size(1),
          enrichment_config_.value()->enrichment_dim_,
          format);
      if (stats.non_finite > 0 || stats.not_converged > 0) {
        XLOG(WARN) << "[EmbeddingCacheEnrich] serving_qdq " << stats;
      } else {
        XLOG_EVERY_MS(INFO, 60000)
            << "[EmbeddingCacheEnrich] serving_qdq " << stats;
      }
    }
  }

  serving_qdq::Stats serving_qdq_shard(
      int64_t shard_id,
      int64_t dim,
      const serving_qdq::Format& format) {
    serving_qdq::Scratch scratch;
    serving_qdq::Stats stats;
    auto wlmap = kv_store_.by(shard_id).wlock();
    auto* pool = kv_store_.pool_by(shard_id);
    for (auto& [key, block] : *wlmap) {
      if (FixedBlockPool::is_weights_all_zero(block, payload_end_bytes())) {
        continue;
      }
      if (packed_) {
        stats.record(serving_qdq_packed_row(
            shard_id, FixedBlockPool::data_ptr<weight_type>(block)));
        continue;
      }
      const auto result = serving_qdq::round_trip_row(
          FixedBlockPool::data_ptr<weight_type>(block), dim, format, scratch);
      stats.record(result);
      if (enable_ssd_backend_ &&
          result != serving_qdq::RowResult::kAlreadyFixed &&
          result != serving_qdq::RowResult::kNonFinite) {
        pool->set_dirty(block, true);
      }
    }
    return stats;
  }

  /// Bytes from the block start through the end of the row payload; the
  /// rest of the block, if any, is alignment padding.
  size_t payload_end_bytes() const {
    if (int8_) {
      return sizeof(FixedBlockPool::MetaHeader) +
          static_cast<size_t>(int8_->payload_bytes());
    }
    return packed_ ? sizeof(FixedBlockPool::MetaHeader) +
            static_cast<size_t>(packed_->payload_bytes())
                   : block_size_;
  }

  static std::optional<serving_qdq::Int8Layout> int8_layout(
      int64_t max_D,
      const std::optional<c10::intrusive_ptr<kv_mem::EnrichmentConfig>>&
          enrichment_config) {
    if (!enrichment_config.has_value() ||
        !enrichment_config.value()->int8_storage_) {
      return std::nullopt;
    }
    const auto& config = *enrichment_config.value();
    TORCH_CHECK(
        !config.packed_storage_ && config.serving_qdq_bit_rate_ == 0,
        "int8_storage excludes packed_storage and serving_qdq_bit_rate, got ",
        config.packed_storage_,
        " and ",
        config.serving_qdq_bit_rate_);
    TORCH_CHECK(
        config.enrichment_dim_ > 0 && config.enrichment_dim_ % 4 == 0 &&
            config.enrichment_dim_ <= max_D,
        "int8_storage needs an enrichment_dim that is a positive multiple of "
        "4 and at most max_D ",
        max_D,
        ", got ",
        config.enrichment_dim_);
    return serving_qdq::Int8Layout{
        config.enrichment_dim_, max_D - config.enrichment_dim_};
  }

  static std::optional<serving_qdq::PackedLayout> packed_layout(
      int64_t max_D,
      const std::optional<c10::intrusive_ptr<kv_mem::EnrichmentConfig>>&
          enrichment_config) {
    if (!enrichment_config.has_value() ||
        !enrichment_config.value()->packed_storage_) {
      return std::nullopt;
    }
    const auto& config = *enrichment_config.value();
    TORCH_CHECK(
        config.serving_qdq_bit_rate_ == 4 && config.serving_qdq_group_size_ > 0,
        "packed_storage needs serving_qdq_bit_rate 4 and a "
        "serving_qdq_group_size, got ",
        config.serving_qdq_bit_rate_,
        " and ",
        config.serving_qdq_group_size_);
    fbgemm_gpu::grouped_int4_e4m3::check_dims(
        config.enrichment_dim_, config.serving_qdq_group_size_);
    TORCH_CHECK(
        config.enrichment_dim_ <= max_D,
        "packed_storage enrichment_dim ",
        config.enrichment_dim_,
        " > max_D ",
        max_D);
    return serving_qdq::PackedLayout{
        config.enrichment_dim_,
        config.serving_qdq_group_size_,
        max_D - config.enrichment_dim_};
  }

  static size_t row_block_size(
      int64_t max_D,
      const std::optional<serving_qdq::PackedLayout>& packed,
      const std::optional<serving_qdq::Int8Layout>& int8) {
    if (!packed && !int8) {
      return FixedBlockPool::calculate_block_size<weight_type>(max_D);
    }
    const size_t alignment =
        FixedBlockPool::calculate_block_alignment<weight_type>();
    const size_t bytes = sizeof(FixedBlockPool::MetaHeader) +
        static_cast<size_t>(int8 ? int8->payload_bytes()
                                 : packed->payload_bytes());
    return (bytes + alignment - 1) / alignment * alignment;
  }

  /// INT8 rows are read and written only through read_payload /
  /// write_payload and the raw INT8 row paths, so every path that touches
  /// payload bytes some other way is refused here.
  void check_int8_storage_supported() const {
    TORCH_CHECK(
        (std::is_same_v<weight_type, at::Half>),
        "int8_storage needs FP16 weights");
    const auto type = enrichment_config_.value()->enrichment_type_;
    TORCH_CHECK(
        type == EnrichmentType::IGR_LASER_EMBEDDING ||
            type == EnrichmentType::IN_MEMORY_TEST_ONLY,
        "int8_storage needs an embedding enrichment payload");
    TORCH_CHECK(disable_random_init_, "int8_storage needs disable_random_init");
    TORCH_CHECK(
        !enable_ssd_backend_,
        "int8_storage does not support the SSD backend (it writes back raw "
        "blocks)");
    TORCH_CHECK(
        !feature_evict_config_.has_value() ||
            feature_evict_config_.value()->trigger_strategy_ !=
                EvictTriggerStrategy::BY_L2WEIGHT,
        "int8_storage does not support L2 weight eviction (it reads raw "
        "blocks)");
  }

  /// Packed rows are read and written only through read_payload /
  /// write_payload, so every path that touches payload bytes some other way
  /// is refused here.
  void check_packed_storage_supported() const {
    TORCH_CHECK(
        (std::is_same_v<weight_type, at::Half>),
        "packed_storage needs FP16 weights");
    TORCH_CHECK(
        serving_qdq_format().group_size > 0,
        "packed_storage needs an embedding enrichment payload");
    TORCH_CHECK(
        disable_random_init_, "packed_storage needs disable_random_init");
    TORCH_CHECK(
        !enable_ssd_backend_,
        "packed_storage does not support the SSD backend (it writes back raw "
        "blocks)");
    TORCH_CHECK(
        !feature_evict_config_.has_value() ||
            feature_evict_config_.value()->trigger_strategy_ !=
                EvictTriggerStrategy::BY_L2WEIGHT,
        "packed_storage does not support L2 weight eviction (it reads raw "
        "blocks)");
  }

  /// Copies columns [offset, offset + width) of the row stored at `data` (a
  /// block's payload) to `dst`. Called under the shard lock.
  void read_payload(
      int shard_id,
      const weight_type* data,
      int64_t offset,
      int64_t width,
      weight_type* dst) const {
    if (int8_) {
      read_int8_payload(data, offset, width, dst);
      return;
    }
    if (!packed_) {
      std::copy(data + offset, data + offset + width, dst);
      return;
    }
    if constexpr (std::is_same_v<weight_type, at::Half>) {
      const auto& layout = *packed_;
      CHECK_LE(offset + width, max_D_);
      const auto* payload = reinterpret_cast<const uint8_t*>(data);
      const auto* tail = payload + layout.packed_bytes();
      if (offset == 0 && width >= layout.dim) {
        read_packed_dims(shard_id, payload, dst);
        std::memcpy(
            dst + layout.dim, tail, (width - layout.dim) * sizeof(weight_type));
        return;
      }
      thread_local std::vector<weight_type> row;
      row.resize(max_D_);
      read_packed_dims(shard_id, payload, row.data());
      std::memcpy(
          row.data() + layout.dim, tail, layout.tail * sizeof(weight_type));
      std::copy(row.data() + offset, row.data() + offset + width, dst);
    }
  }

  /// Stores columns [0, width) of `src` as the row at `data` (a block's
  /// payload); columns past `width` keep their values. Called under the shard
  /// write lock.
  void write_payload(
      int shard_id,
      weight_type* data,
      const weight_type* src,
      int64_t width) {
    if (int8_) {
      write_int8_payload(data, src, width);
      return;
    }
    if (!packed_) {
      std::copy(src, src + width, data);
      return;
    }
    if constexpr (std::is_same_v<weight_type, at::Half>) {
      const auto& layout = *packed_;
      CHECK_LE(width, max_D_);
      auto* payload = reinterpret_cast<uint8_t*>(data);
      if (width < layout.dim) {
        thread_local std::vector<weight_type> row;
        row.resize(layout.dim);
        read_packed_dims(shard_id, payload, row.data());
        std::copy(src, src + width, row.data());
        write_packed_dims(shard_id, payload, row.data());
        return;
      }
      write_packed_dims(shard_id, payload, src);
      std::memcpy(
          payload + layout.packed_bytes(),
          src + layout.dim,
          (width - layout.dim) * sizeof(weight_type));
    }
  }

  /// Like read_payload, with the columns counted from the block start, so the
  /// MetaHeader comes first.
  void read_row_with_metaheader(
      int shard_id,
      const weight_type* block,
      int64_t offset,
      int64_t width,
      weight_type* dst) const {
    if (!packed_ && !int8_) {
      const auto* from =
          FixedBlockPool::ptr_offset_from_front<weight_type>(block, offset);
      std::copy(from, from + width, dst);
      return;
    }
    const int64_t meta_dim = FixedBlockPool::get_metaheader_dim<weight_type>();
    const int64_t meta_cols = std::clamp<int64_t>(meta_dim - offset, 0, width);
    std::copy(block + offset, block + offset + meta_cols, dst);
    if (width > meta_cols) {
      read_payload(
          shard_id,
          FixedBlockPool::data_ptr<weight_type>(block),
          offset + meta_cols - meta_dim,
          width - meta_cols,
          dst + meta_cols);
    }
  }

  /// Like write_payload for a whole row that starts with its MetaHeader.
  void write_row_with_metaheader(
      int shard_id,
      weight_type* block,
      const weight_type* src,
      int64_t width) {
    if (!packed_ && !int8_) {
      std::copy(src, src + width, block);
      return;
    }
    const int64_t meta_dim = FixedBlockPool::get_metaheader_dim<weight_type>();
    CHECK_GE(width, meta_dim);
    std::copy(src, src + meta_dim, block);
    write_payload(
        shard_id,
        FixedBlockPool::data_ptr<weight_type>(block),
        src + meta_dim,
        width - meta_dim);
  }

  void read_packed_dims(int shard_id, const uint8_t* payload, weight_type* dst)
      const {
    if constexpr (std::is_same_v<weight_type, at::Half>) {
      const auto& layout = *packed_;
      if (payload[layout.code_bytes()] != serving_qdq::kFallbackScale) {
        serving_qdq::unpack_row(payload, layout, dst);
        return;
      }
      const auto& fallback = fallback_rows_[shard_id];
      const auto it = fallback.find(payload);
      if (it != fallback.end()) {
        std::copy(it->second.begin(), it->second.end(), dst);
      } else {
        // Every row write stores the marker and its fp16 row together, so
        // this is a block allocated without a row write (the feature score
        // metadata path), read back as a miss.
        packed_fallback_missing_++;
        XLOG_EVERY_MS(ERR, 60000)
            << "[EmbeddingCacheEnrich] packed rows with no fp16 fallback row: "
            << packed_fallback_missing_.load();
        std::fill(dst, dst + layout.dim, weight_type(0));
      }
    }
  }

  /// Packs row[0, dim). A row that is not already a serving fixed point is
  /// first rewritten as one, as serving_qdq_existing_rows would rewrite it.
  /// Rows with no fixed point (non-finite, or not converged) are kept as fp16
  /// outside the block, so no row is ever stored as different values.
  serving_qdq::RowResult
  write_packed_dims(int shard_id, uint8_t* payload, const weight_type* row) {
    if constexpr (std::is_same_v<weight_type, at::Half>) {
      const auto& layout = *packed_;
      auto& fallback = fallback_rows_[shard_id];
      if (serving_qdq::pack_row(row, layout, payload)) {
        if (!fallback.empty()) {
          fallback.erase(payload);
        }
        return serving_qdq::RowResult::kAlreadyFixed;
      }
      thread_local serving_qdq::Scratch scratch;
      thread_local std::vector<weight_type> fixed;
      fixed.assign(row, row + layout.dim);
      const auto result = serving_qdq::round_trip_row(
          fixed.data(), layout.dim, serving_qdq_format(), scratch);
      const bool packed_exactly =
          result != serving_qdq::RowResult::kNonFinite &&
          result != serving_qdq::RowResult::kNotConverged &&
          serving_qdq::pack_row(fixed.data(), layout, payload);
      if (!packed_exactly) {
        if (result != serving_qdq::RowResult::kNonFinite &&
            result != serving_qdq::RowResult::kNotConverged) {
          // A serving fixed point should pack exactly; keep its fp16 values
          // rather than abort the trainer.
          packed_inexact_fixed_points_++;
          XLOG_EVERY_MS(ERR, 60000)
              << "[EmbeddingCacheEnrich] serving fixed points that did not "
                 "pack exactly, kept as fp16: "
              << packed_inexact_fixed_points_.load();
        }
        std::memset(payload, 0, layout.packed_bytes());
        payload[layout.code_bytes()] = serving_qdq::kFallbackScale;
        fallback[payload] = fixed;
        packed_fallback_writes_++;
        XLOG_EVERY_MS(WARN, 60000)
            << "[EmbeddingCacheEnrich] packed rows kept as fp16: "
            << packed_fallback_writes_.load();
        return result;
      }
      if (!fallback.empty()) {
        fallback.erase(payload);
      }
      packed_requantized_writes_++;
      return result;
    }
    return serving_qdq::RowResult::kAlreadyFixed;
  }

  /// serving_qdq_shard for a packed row: only a row kept as fp16 can change.
  serving_qdq::RowResult serving_qdq_packed_row(
      int shard_id,
      weight_type* data) {
    if constexpr (std::is_same_v<weight_type, at::Half>) {
      auto* payload = reinterpret_cast<uint8_t*>(data);
      auto& fallback = fallback_rows_[shard_id];
      const auto it =
          payload[packed_->code_bytes()] == serving_qdq::kFallbackScale
          ? fallback.find(payload)
          : fallback.end();
      if (it == fallback.end()) {
        return serving_qdq::RowResult::kAlreadyFixed;
      }
      const std::vector<weight_type> row = it->second;
      return write_packed_dims(shard_id, payload, row.data());
    }
    return serving_qdq::RowResult::kAlreadyFixed;
  }

  /// read_payload for an INT8 row: the codes decode to columns [0, dim), the
  /// tail columns are stored as fp16.
  void read_int8_payload(
      const weight_type* data,
      int64_t offset,
      int64_t width,
      weight_type* dst) const {
    if constexpr (std::is_same_v<weight_type, at::Half>) {
      const auto& layout = *int8_;
      CHECK_LE(offset + width, max_D_);
      const auto* payload = reinterpret_cast<const uint8_t*>(data);
      const auto* tail = payload + layout.row_bytes();
      if (offset == 0 && width >= layout.dim) {
        serving_qdq::decode_int8_row(payload, layout.dim, dst);
        std::memcpy(
            dst + layout.dim, tail, (width - layout.dim) * sizeof(weight_type));
        return;
      }
      // Optimizer-state reads only touch the fp16 tail.
      if (offset >= layout.dim) {
        std::memcpy(
            dst,
            tail + (offset - layout.dim) * sizeof(weight_type),
            width * sizeof(weight_type));
        return;
      }
      thread_local std::vector<weight_type> row;
      row.resize(max_D_);
      serving_qdq::decode_int8_row(payload, layout.dim, row.data());
      std::memcpy(
          row.data() + layout.dim, tail, layout.tail * sizeof(weight_type));
      std::copy(row.data() + offset, row.data() + offset + width, dst);
    }
  }

  /// write_payload for an INT8 row: columns [0, dim) are quantized, tail
  /// columns copied; a write narrower than dim re-encodes the merged row.
  void
  write_int8_payload(weight_type* data, const weight_type* src, int64_t width) {
    if constexpr (std::is_same_v<weight_type, at::Half>) {
      const auto& layout = *int8_;
      CHECK_LE(width, max_D_);
      auto* payload = reinterpret_cast<uint8_t*>(data);
      if (width < layout.dim) {
        thread_local std::vector<weight_type> row;
        row.resize(layout.dim);
        serving_qdq::decode_int8_row(payload, layout.dim, row.data());
        std::copy(src, src + width, row.data());
        record_int8_write(
            serving_qdq::encode_int8_row(row.data(), layout.dim, payload));
        return;
      }
      record_int8_write(serving_qdq::encode_int8_row(src, layout.dim, payload));
      std::memcpy(
          payload + layout.row_bytes(),
          src + layout.dim,
          (width - layout.dim) * sizeof(weight_type));
    }
  }

  void record_int8_write(serving_qdq::Int8Result result) {
    switch (result) {
      case serving_qdq::Int8Result::kFixed:
        return;
      case serving_qdq::Int8Result::kMultiPass:
        int8_multi_pass_writes_++;
        return;
      case serving_qdq::Int8Result::kNonFinite:
        int8_non_finite_writes_++;
        break;
      case serving_qdq::Int8Result::kNotConverged:
        int8_not_converged_writes_++;
        break;
    }
    XLOG_EVERY_MS(WARN, 60000)
        << "[EmbeddingCacheEnrich] INT8 rows: non_finite="
        << int8_non_finite_writes_.load()
        << " not_converged=" << int8_not_converged_writes_.load();
  }

  // fp32 scale and bias after the codes of each INT8 HBM row.
  static constexpr int64_t kHbmQparamsBytes = 8;

  struct HbmTier {
    at::Tensor cache_state;
    at::Tensor cache_weights;
    at::Tensor last_access;
    std::optional<at::cuda::CUDAStream> stream;
    // Columns an HBM row holds. Columns past it, up to max_D_ (optimizer state,
    // which embedding-cache rows never use), read back as zero.
    int64_t dim;
  };

  /// What the HBM tier holds at a batch of slots, read in one pass.
  struct HbmSlots {
    std::vector<int64_t> tags;
    std::vector<int32_t> last_access;
    at::Tensor rows; // [n, dim + 8] uint8, CPU; undefined unless requested
  };

  static int64_t env_int64(const char* name, int64_t fallback) {
    const char* const value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
      return fallback;
    }
    int64_t parsed = 0;
    const auto [ptr, ec] =
        std::from_chars(value, value + std::strlen(value), parsed);
    if (ec != std::errc{} || *ptr != '\0' || parsed < 0) {
      LOG(WARNING) << "Ignoring invalid " << name << "='" << value << "'";
      return fallback;
    }
    return parsed;
  }

  /// The table's training TTL when its eviction is timestamp based, else 0.
  int64_t table_ttl_mins() const {
    if (!feature_evict_config_.has_value()) {
      return 0;
    }
    const auto& config = *feature_evict_config_.value();
    const bool by_time =
        config.trigger_strategy_ == EvictTriggerStrategy::BY_TIMESTAMP ||
        config.trigger_strategy_ ==
            EvictTriggerStrategy::BY_TIMESTAMP_AND_COUNTER ||
        config.trigger_strategy_ ==
            EvictTriggerStrategy::BY_TIMESTAMP_THRESHOLD;
    if (!by_time || !config.ttls_in_mins_.has_value() ||
        config.ttls_in_mins_->empty()) {
      return 0;
    }
    return *std::max_element(
        config.ttls_in_mins_->begin(), config.ttls_in_mins_->end());
  }

  /// Slot of `key`; must match cache_slot() in split_embeddings_cache_cuda.cuh.
  static int64_t hbm_slot(int64_t key, int64_t num_slots) {
    auto h = static_cast<uint64_t>(key);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<int64_t>(h % static_cast<uint32_t>(num_slots));
  }

  int64_t hbm_slot(int64_t key) const {
    return hbm_slot(key, hbm_->cache_state.size(0));
  }

  static uint32_t meta_timestamp(uint64_t meta) {
    return static_cast<uint32_t>(meta & 0xFFFFFFFFULL);
  }

  static uint64_t with_timestamp(uint64_t meta, uint32_t timestamp) {
    return (meta & ~0xFFFFFFFFULL) | timestamp;
  }

  /// Runs a host-callback or hbm_executor_ task body: an exception there
  /// would terminate the process or be dropped by the executor, so it is kept
  /// and rethrown by the next exclusive_hbm_* call.
  template <typename Fn>
  void hbm_run_async(Fn&& fn) noexcept {
    try {
      std::forward<Fn>(fn)();
    } catch (...) {
      XLOG(ERR) << "[ExclusiveHBM] async task failed: "
                << folly::exceptionStr(std::current_exception());
      auto error = hbm_async_error_.wlock();
      if (!*error) {
        *error = std::current_exception();
      }
    }
  }

  void hbm_rethrow_async_error() {
    auto error = hbm_async_error_.exchange(nullptr);
    if (error) {
      std::rethrow_exception(error);
    }
  }

  /// Empties the slots of `keys` that hold no tombstone (after a failed
  /// promotion), so no slot keeps a row the backend does not account for.
  void hbm_clear_untombstoned_slots(const std::vector<int64_t>& keys) noexcept {
    try {
      std::vector<int64_t> slots;
      for (const auto key : keys) {
        const auto shard_id = kv_db_utils::hash_shard(key, num_shards_);
        const auto lock = kv_store_.by(shard_id).rlock();
        if (!hbm_rows_[shard_id].contains(key)) {
          slots.push_back(hbm_slot(key));
        }
      }
      hbm_clear_slots(slots);
    } catch (...) {
      XLOG(ERR) << "[ExclusiveHBM] clearing slots after a failed promotion: "
                << folly::exceptionStr(std::current_exception());
    }
  }

  template <typename Fn>
  auto via_hbm_executor(Fn fn) {
    // NOLINTNEXTLINE(readability-redundant-smartptr-get)
    return folly::via(hbm_executor_.get(), std::move(fn)).get();
  }

  /// Runs `fn` with the HBM tier's side stream current and waits for it.
  template <typename Fn>
  void on_hbm_stream(Fn fn) {
    if (!hbm_->stream) {
      fn();
      return;
    }
    at::cuda::CUDAStreamGuard guard(*hbm_->stream);
    fn();
    hbm_->stream->synchronize();
  }

  HbmSlots hbm_read_slots(const std::vector<int64_t>& slots, bool rows) {
    HbmSlots out;
    const auto n = static_cast<int64_t>(slots.size());
    if (n == 0) {
      return out;
    }
    const auto index = at::tensor(slots, at::kLong);
    at::Tensor tags;
    at::Tensor access;
    on_hbm_stream([&] {
      const auto device_index = index.to(hbm_->cache_state.device());
      tags = hbm_->cache_state.index_select(0, device_index).view(-1).cpu();
      access = hbm_->last_access.index_select(0, device_index).cpu();
      if (rows) {
        out.rows = hbm_->cache_weights.index_select(0, device_index).cpu();
      }
    });
    out.tags.assign(
        tags.const_data_ptr<int64_t>(), tags.const_data_ptr<int64_t>() + n);
    out.last_access.assign(
        access.const_data_ptr<int32_t>(), access.const_data_ptr<int32_t>() + n);
    return out;
  }

  void hbm_clear_slots(const std::vector<int64_t>& slots) {
    if (slots.empty()) {
      return;
    }
    const auto index = at::tensor(slots, at::kLong);
    on_hbm_stream([&] {
      hbm_->cache_state.index_fill_(
          0, index.to(hbm_->cache_state.device()), int64_t{-1});
    });
  }

  std::vector<int64_t> hbm_keys_of_shard(int shard_id) {
    const auto lock = kv_store_.by(shard_id).rlock();
    std::vector<int64_t> keys;
    keys.reserve(hbm_rows_[shard_id].size());
    for (const auto& [key, meta] : hbm_rows_[shard_id]) {
      keys.push_back(key);
    }
    return keys;
  }

  /// Decodes an HBM row as the GPU gather does (code * scale + bias, rounded
  /// to fp16), then, for a table stored at its serving precision, as its
  /// serving fixed point, so a row leaving HBM is a valid backend row.
  void hbm_decode_row(const uint8_t* src, weight_type* dst) {
    if constexpr (std::is_same_v<weight_type, at::Half>) {
      const int64_t hbm_dim = hbm_->dim;
      const auto* qparams = reinterpret_cast<const float*>(src + hbm_dim);
      const float scale = qparams[0];
      const float bias = qparams[1];
      for (int64_t d = 0; d < hbm_dim; ++d) {
        dst[d] = c10::Half(static_cast<float>(src[d]) * scale + bias);
      }
      std::fill(dst + hbm_dim, dst + max_D_, weight_type(0));
      const auto format = serving_qdq_format();
      if (format.bit_rate == 0) {
        return;
      }
      const int64_t dim = enrichment_config_.value()->enrichment_dim_;
      thread_local serving_qdq::Scratch scratch;
      const auto result =
          serving_qdq::round_trip_row(dst, dim, format, scratch);
      if (result != serving_qdq::RowResult::kAlreadyFixed &&
          result != serving_qdq::RowResult::kNonFinite) {
        hbm_requantized_++;
      }
    }
  }

  /// Encodes row[0, hbm dim) to the bytes kvzch_exclusive_cache_insert_int8
  /// stores for it (the same fp32 operations, so the bytes match exactly).
  void hbm_encode_row(const weight_type* row, uint8_t* dst) const {
    const int64_t dim = hbm_->dim;
    float row_min = std::numeric_limits<float>::max();
    float row_max = std::numeric_limits<float>::lowest();
    for (int64_t d = 0; d < dim; ++d) {
      row_min = std::fmin(row_min, static_cast<float>(row[d]));
      row_max = std::fmax(row_max, static_cast<float>(row[d]));
    }
    const float bias = row_min;
    const float range = row_max - row_min;
    const float scale = range == 0.0f ? 1.0f : range / 255.0f;
    for (int64_t d = 0; d < dim; ++d) {
      dst[d] = static_cast<uint8_t>(std::fmin(
          std::fmax(
              std::rint((static_cast<float>(row[d]) - bias) / scale), 0.0f),
          255.0f));
    }
    std::memcpy(dst + dim, &scale, sizeof(float));
    std::memcpy(dst + dim + sizeof(float), &bias, sizeof(float));
  }

  /// Whether a slot claimed for `block`'s key holds the INT8 encoding of the
  /// block's current row. A write between the GPU's backend read and this
  /// promotion (e.g. a Laser response updating the block in place) changes
  /// the row, and the slot is then stale.
  bool hbm_slot_matches_block(
      int shard_id,
      const weight_type* block,
      const uint8_t* slot_row) const {
    thread_local std::vector<weight_type> row;
    thread_local std::vector<uint8_t> encoded;
    row.resize(hbm_->dim);
    encoded.resize(hbm_->dim + kHbmQparamsBytes);
    read_payload(
        shard_id,
        FixedBlockPool::data_ptr<weight_type>(block),
        0,
        hbm_->dim,
        row.data());
    hbm_encode_row(row.data(), encoded.data());
    return std::memcmp(encoded.data(), slot_row, encoded.size()) == 0;
  }

  /// Whether an HBM row is byte for byte the INT8 part of a backend row, so
  /// promotion and demotion move bytes without re-quantizing.
  bool hbm_rows_are_int8_rows() const {
    return int8_.has_value() && hbm_->dim == int8_->dim;
  }

  /// Stores an HBM row's codes, scale and bias as a block's payload; the tail
  /// columns, which HBM rows do not hold, read back as zero.
  void copy_hbm_row_to_int8_block(const uint8_t* hbm_row, weight_type* block) {
    auto* payload = reinterpret_cast<uint8_t*>(
        FixedBlockPool::data_ptr<weight_type>(block));
    std::memcpy(payload, hbm_row, int8_->row_bytes());
    std::memset(
        payload + int8_->row_bytes(),
        0,
        int8_->payload_bytes() - int8_->row_bytes());
  }

  void hbm_promote(std::vector<int64_t>& keys) {
    if (hbm_fail_next_promotion_.exchange(false)) {
      throw std::runtime_error("injected HBM promotion failure");
    }
    std::ranges::sort(keys);
    keys.erase(std::ranges::unique(keys).begin(), keys.end());
    for (size_t begin = 0; begin < keys.size();
         begin += static_cast<size_t>(hbm_chunk_rows_)) {
      const auto end =
          std::min(keys.size(), begin + static_cast<size_t>(hbm_chunk_rows_));
      std::vector<int64_t> slots;
      slots.reserve(end - begin);
      for (auto i = begin; i < end; ++i) {
        slots.push_back(hbm_slot(keys[i]));
      }
      const bool raw_int8 = hbm_rows_are_int8_rows();
      const auto held = hbm_read_slots(slots, /*rows=*/true);
      const int64_t row_bytes = hbm_->dim + kHbmQparamsBytes;
      std::vector<std::vector<size_t>> by_shard(num_shards_);
      for (auto i = begin; i < end; ++i) {
        if (held.tags[i - begin] == keys[i]) {
          by_shard[kv_db_utils::hash_shard(keys[i], num_shards_)].push_back(
              i - begin);
        } else {
          hbm_promote_skipped_++;
        }
      }
      std::vector<int64_t> stale_slots;
      for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
        if (by_shard[shard_id].empty()) {
          continue;
        }
        auto wlmap = kv_store_.by(shard_id).wlock();
        auto* pool = kv_store_.pool_by(shard_id);
        auto& tombstones = hbm_rows_[shard_id];
        for (const auto j : by_shard[shard_id]) {
          const auto key = keys[begin + j];
          if (tombstones.contains(key)) {
            hbm_promoted_++;
            continue;
          }
          const auto it = wlmap->find(key);
          const auto* slot_row =
              held.rows.template const_data_ptr<uint8_t>() + j * row_bytes;
          // No backend row: the evictor dropped it after the GPU read. A
          // backend row that changed after that read supersedes the slot.
          if (it == wlmap->end() ||
              (raw_int8
                   ? std::memcmp(
                         FixedBlockPool::data_ptr<weight_type>(it->second),
                         slot_row,
                         row_bytes) != 0
                   : !hbm_slot_matches_block(shard_id, it->second, slot_row))) {
            stale_slots.push_back(slots[j]);
            hbm_promote_skipped_++;
            continue;
          }
          weight_type* block = it->second;
          tombstones[key] = FixedBlockPool::get_metaheader_raw(block);
          wlmap->erase(it);
          if (packed_ && !fallback_rows_[shard_id].empty()) {
            fallback_rows_[shard_id].erase(FixedBlockPool::data_ptr(block));
          }
          pool->template deallocate_t<weight_type>(block);
          hbm_promoted_++;
        }
      }
      hbm_clear_slots(stale_slots);
    }
    XLOG_EVERY_MS(INFO, 60000)
        << "[ExclusiveHBM] promoted=" << hbm_promoted_.load()
        << " skipped=" << hbm_promote_skipped_.load()
        << " demoted=" << hbm_demoted_.load()
        << " dropped=" << hbm_dropped_.load()
        << " invalidated=" << hbm_invalidated_.load()
        << " requantized=" << hbm_requantized_.load()
        << " tombstone_reads=" << hbm_tombstone_reads_.load()
        << " tombstone_mask_reads=" << hbm_tombstone_mask_reads_.load();
  }

  /// Moves (or, with `drop`, deletes) the HBM rows of `keys` that are still
  /// HBM rows, in chunks of hbm_chunk_rows_ rows, then empties their slots.
  int64_t hbm_demote(const std::vector<int64_t>& keys, bool drop) {
    int64_t moved = 0;
    std::vector<weight_type> row(max_D_);
    for (size_t begin = 0; begin < keys.size();
         begin += static_cast<size_t>(hbm_chunk_rows_)) {
      const auto end =
          std::min(keys.size(), begin + static_cast<size_t>(hbm_chunk_rows_));
      std::vector<int64_t> slots;
      slots.reserve(end - begin);
      for (auto i = begin; i < end; ++i) {
        slots.push_back(hbm_slot(keys[i]));
      }
      const auto held = hbm_read_slots(slots, /*rows=*/!drop);
      const int64_t row_bytes = hbm_->dim + kHbmQparamsBytes;
      std::vector<std::vector<size_t>> by_shard(num_shards_);
      for (auto i = begin; i < end; ++i) {
        if (held.tags[i - begin] == keys[i]) {
          by_shard[kv_db_utils::hash_shard(keys[i], num_shards_)].push_back(
              i - begin);
        }
      }
      std::vector<int64_t> emptied;
      for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
        if (by_shard[shard_id].empty()) {
          continue;
        }
        auto wlmap = kv_store_.by(shard_id).wlock();
        auto* pool = kv_store_.pool_by(shard_id);
        auto& tombstones = hbm_rows_[shard_id];
        for (const auto j : by_shard[shard_id]) {
          const auto key = keys[begin + j];
          const auto it = tombstones.find(key);
          if (it == tombstones.end()) {
            continue;
          }
          const auto timestamp = std::max<uint32_t>(
              meta_timestamp(it->second),
              static_cast<uint32_t>(held.last_access[j]));
          if (!drop && !wlmap->contains(key)) {
            const auto* hbm_row =
                held.rows.template const_data_ptr<uint8_t>() + j * row_bytes;
            auto* block = pool->template allocate_t<weight_type>();
            std::memcpy(
                reinterpret_cast<char*>(block) + sizeof(int64_t),
                &it->second,
                sizeof(uint64_t));
            FixedBlockPool::set_key(block, key);
            FixedBlockPool::set_used(block, true);
            FixedBlockPool::set_timestamp(block, timestamp);
            if (hbm_rows_are_int8_rows()) {
              copy_hbm_row_to_int8_block(hbm_row, block);
            } else {
              hbm_decode_row(hbm_row, row.data());
              write_payload(
                  shard_id,
                  FixedBlockPool::data_ptr<weight_type>(block),
                  row.data(),
                  max_D_);
            }
            wlmap->insert({key, block});
          }
          tombstones.erase(it);
          emptied.push_back(slots[j]);
          (drop ? hbm_dropped_ : hbm_demoted_)++;
          moved++;
        }
      }
      hbm_clear_slots(emptied);
    }
    return moved;
  }

  int64_t exclusive_hbm_sweep_now(int64_t now = -1) {
    if (hbm_ttl_s_ <= 0) {
      return 0;
    }
    const auto current = now >= 0 ? static_cast<uint32_t>(now)
                                  : FixedBlockPool::current_timestamp();
    int64_t moved = 0;
    for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
      std::vector<std::pair<int64_t, uint32_t>> rows;
      {
        const auto lock = kv_store_.by(shard_id).rlock();
        rows.reserve(hbm_rows_[shard_id].size());
        for (const auto& [key, meta] : hbm_rows_[shard_id]) {
          rows.emplace_back(key, meta_timestamp(meta));
        }
      }
      std::vector<int64_t> expired;
      for (size_t begin = 0; begin < rows.size();
           begin += static_cast<size_t>(hbm_chunk_rows_)) {
        const auto end =
            std::min(rows.size(), begin + static_cast<size_t>(hbm_chunk_rows_));
        std::vector<int64_t> slots;
        slots.reserve(end - begin);
        for (auto i = begin; i < end; ++i) {
          slots.push_back(hbm_slot(rows[i].first));
        }
        const auto held = hbm_read_slots(slots, /*rows=*/false);
        for (auto i = begin; i < end; ++i) {
          const auto last = std::max<uint32_t>(
              rows[i].second,
              static_cast<uint32_t>(held.last_access[i - begin]));
          // A GPU access ahead of the host clock (idle <= 0) is not expired.
          const int64_t idle =
              static_cast<int64_t>(current) - static_cast<int64_t>(last);
          if (idle > 0 && idle > hbm_ttl_s_) {
            expired.push_back(rows[i].first);
          }
        }
      }
      moved += hbm_demote(expired, /*drop=*/!hbm_ttl_demote_);
    }
    return moved;
  }

  /// Called under the shard write lock by every write that gives `key` a
  /// backend row: the written row supersedes the HBM copy, whose slot is
  /// emptied by the next hbm_post_invalidations_locked.
  void hbm_take_tombstone_locked(int shard_id, int64_t key) {
    if (hbm_ && hbm_rows_[shard_id].erase(key) > 0) {
      hbm_pending_invalidations_[shard_id].push_back(key);
    }
  }

  void hbm_post_invalidations_locked(int shard_id) {
    if (!hbm_ || hbm_pending_invalidations_[shard_id].empty()) {
      return;
    }
    std::vector<int64_t> keys;
    keys.swap(hbm_pending_invalidations_[shard_id]);
    hbm_executor_->add([this, keys = std::move(keys)] {
      hbm_run_async([&] {
        std::vector<int64_t> slots;
        slots.reserve(keys.size());
        for (const auto key : keys) {
          slots.push_back(hbm_slot(key));
        }
        const auto held = hbm_read_slots(slots, /*rows=*/false);
        std::vector<int64_t> emptied;
        for (size_t i = 0; i < keys.size(); ++i) {
          if (held.tags[i] != keys[i]) {
            continue;
          }
          const auto shard_id = kv_db_utils::hash_shard(keys[i], num_shards_);
          const auto lock = kv_store_.by(shard_id).rlock();
          if (!hbm_rows_[shard_id].contains(keys[i])) {
            emptied.push_back(slots[i]);
            hbm_invalidated_++;
          }
        }
        hbm_clear_slots(emptied);
      });
    });
  }

  /// Positions of the `ids` whose only copy is in HBM now.
  std::vector<int64_t> hbm_tombstoned_positions(const at::Tensor& ids) {
    std::vector<int64_t> positions;
    if (!hbm_) {
      return positions;
    }
    const auto id_values = ids.to(at::kLong).contiguous();
    const auto* id_ptr = id_values.const_data_ptr<int64_t>();
    for (int64_t i = 0; i < id_values.numel(); ++i) {
      const auto shard_id = kv_db_utils::hash_shard(id_ptr[i], num_shards_);
      const auto lock = kv_store_.by(shard_id).rlock();
      if (hbm_rows_[shard_id].contains(id_ptr[i])) {
        positions.push_back(i);
      }
    }
    return positions;
  }

  /// Fills the rows of `ids` that live only in HBM into `weights` after a
  /// backend read zero-filled them: columns [width_offset, width_offset +
  /// width) of each row, counted from the MetaHeader when `whole_row`.
  ///
  /// A row can move between the tiers during the backend read (the TTL sweep,
  /// a promotion, a write that takes the tombstone), so this resolves every
  /// row that was an HBM row before the read (`tombstoned_before`, from
  /// hbm_tombstoned_positions), is one now, or read back as zero. It resolves
  /// them on hbm_executor_, where promotions and demotions run, so none can
  /// move: a row still tombstoned is read from HBM, any other is re-read from
  /// the backend. HBM is read in chunks of hbm_chunk_rows_ rows.
  void hbm_fill_rows(
      const at::Tensor& ids,
      const at::Tensor& weights,
      int64_t width_offset,
      bool whole_row,
      std::vector<int64_t> tombstoned_before) {
    if (!hbm_ || ids.numel() == 0) {
      return;
    }
    const int64_t width = weights.size(1);
    const int64_t meta_dim = FixedBlockPool::get_metaheader_dim<weight_type>();
    TORCH_CHECK(
        weights.is_contiguous() && width_offset >= 0 &&
            width_offset + width <= (whole_row ? meta_dim : 0) + max_D_,
        "HBM fill columns exceed the row width");
    const int64_t first = whole_row ? width_offset : width_offset + meta_dim;
    const auto id_values = ids.to(at::kLong).contiguous();
    const auto* id_ptr = id_values.const_data_ptr<int64_t>();
    auto positions = hbm_tombstoned_positions(id_values);
    positions.insert(
        positions.end(), tombstoned_before.begin(), tombstoned_before.end());
    // Columns past the HBM row read back as zero for an HBM row anyway.
    if (first < meta_dim + hbm_->dim) {
      const auto zero_rows =
          at::nonzero(weights.view({ids.numel(), width}).abs().amax({1}) == 0)
              .view(-1);
      const auto* zero_ptr = zero_rows.const_data_ptr<int64_t>();
      positions.insert(positions.end(), zero_ptr, zero_ptr + zero_rows.numel());
    }
    std::ranges::sort(positions);
    positions.erase(std::ranges::unique(positions).begin(), positions.end());
    if (positions.empty()) {
      return;
    }
    const int64_t row_bytes = hbm_->dim + kHbmQparamsBytes;
    auto* out = weights.data_ptr<weight_type>();
    via_hbm_executor([&] {
      std::vector<int64_t> reread;
      std::vector<weight_type> row(meta_dim + max_D_);
      for (size_t begin = 0; begin < positions.size();
           begin += static_cast<size_t>(hbm_chunk_rows_)) {
        const auto end = std::min(
            positions.size(), begin + static_cast<size_t>(hbm_chunk_rows_));
        std::vector<int64_t> slots;
        slots.reserve(end - begin);
        for (auto i = begin; i < end; ++i) {
          slots.push_back(hbm_slot(id_ptr[positions[i]]));
        }
        const auto held = hbm_read_slots(slots, /*rows=*/true);
        for (auto i = begin; i < end; ++i) {
          const auto key = id_ptr[positions[i]];
          // Slots change only on this thread; tombstones also under the shard
          // lock, so a row still tombstoned here is the key's current row.
          std::optional<uint64_t> meta;
          if (held.tags[i - begin] == key) {
            const auto shard_id = kv_db_utils::hash_shard(key, num_shards_);
            const auto lock = kv_store_.by(shard_id).rlock();
            const auto it = hbm_rows_[shard_id].find(key);
            if (it != hbm_rows_[shard_id].end()) {
              meta = it->second;
            }
          }
          if (!meta) {
            reread.push_back(positions[i]);
            continue;
          }
          FixedBlockPool::MetaHeader header{};
          std::memcpy(
              reinterpret_cast<char*>(&header) + sizeof(int64_t),
              &*meta,
              sizeof(uint64_t));
          header.key = key;
          header.used = true;
          header.timestamp = std::max<uint32_t>(
              header.timestamp,
              static_cast<uint32_t>(held.last_access[i - begin]));
          std::memcpy(row.data(), &header, sizeof(header));
          hbm_decode_row(
              held.rows.template const_data_ptr<uint8_t>() +
                  (i - begin) * row_bytes,
              row.data() + meta_dim);
          std::copy(
              row.data() + first,
              row.data() + first + width,
              out + positions[i] * width);
        }
      }
      if (reread.empty()) {
        return int64_t{0};
      }
      const auto index = at::tensor(reread, at::kLong);
      auto again = at::zeros({index.numel(), width}, weights.options());
      const auto again_ids = id_values.index_select(0, index);
      const auto count = at::tensor({index.numel()}, at::kLong);
      if (whole_row) {
        get_kv_db_with_metaheader_async_impl(
            again_ids, again, count, width_offset, width)
            .wait();
      } else {
        get_kv_db_async_impl(again_ids, again, count, width_offset, width)
            .wait();
      }
      weights.index_copy_(0, index, again);
      return int64_t{0};
    });
  }

  void check_int8_rows(const at::Tensor& rows) const {
    TORCH_CHECK(int8_, "uint8 rows need int8_storage");
    TORCH_CHECK(
        rows.dim() == 2 && rows.scalar_type() == at::kByte &&
            rows.is_contiguous() && rows.device().is_cpu(),
        "INT8 rows must be a contiguous 2D uint8 CPU tensor");
    TORCH_CHECK(
        rows.size(1) == static_cast<int64_t>(payload_end_bytes()),
        "INT8 rows hold ",
        payload_end_bytes(),
        " bytes, got ",
        rows.size(1));
  }

  void read_int8_blocks(const at::Tensor& id_values, const at::Tensor& rows) {
    const int64_t row_bytes = rows.size(1);
    const auto* id_ptr = id_values.const_data_ptr<int64_t>();
    auto* out = rows.data_ptr<uint8_t>();
    const auto count = at::tensor({id_values.numel()}, at::ScalarType::Long);
    std::vector<folly::Future<folly::Unit>> futures;
    for (const auto& [shard_id, indexes] : shard_input(id_values, count)) {
      futures.emplace_back(
          folly::via(executor_.get())
              .thenValue([this, shard_id, indexes, id_ptr, out, row_bytes](
                             folly::Unit) {
                const auto rlmap = kv_store_.by(shard_id).rlock();
                for (const auto i : indexes) {
                  auto* dst = out + i * row_bytes;
                  const auto it = rlmap->find(id_ptr[i]);
                  if (it == rlmap->end()) {
                    std::memset(dst, 0, row_bytes);
                  } else {
                    std::memcpy(dst, it->second, row_bytes);
                  }
                }
              }));
    }
    folly::collect(futures).wait();
  }

  void set_int8_rows_of_shard(
      int shard_id,
      const std::vector<int64_t>& indexes,
      const std::vector<int64_t>& keys,
      const uint8_t* src,
      int64_t row_bytes) {
    auto wlmap = kv_store_.by(shard_id).wlock();
    auto* pool = kv_store_.pool_by(shard_id);
    const bool restamp = restamp_on_restore_ && is_training_;
    const uint32_t restore_ts =
        restamp ? FixedBlockPool::current_timestamp() : 0;
    for (const auto i : indexes) {
      const auto key = keys[i];
      weight_type* block = nullptr;
      const auto it = wlmap->find(key);
      const bool new_block = it == wlmap->end();
      if (new_block) {
        block = pool->template allocate_t<weight_type>();
        wlmap->insert({key, block});
        hbm_take_tombstone_locked(shard_id, key);
      } else {
        block = it->second;
      }
      const int64_t copied_bytes =
          std::min<int64_t>(row_bytes, payload_end_bytes());
      std::memcpy(block, src + i * row_bytes, copied_bytes);
      std::memset(
          reinterpret_cast<uint8_t*>(block) + copied_bytes,
          0,
          payload_end_bytes() - copied_bytes);
      FixedBlockPool::set_key(block, key);
      FixedBlockPool::set_used(block, true);
      if (restamp) {
        FixedBlockPool::set_timestamp(block, restore_ts);
      }
      if (new_block && is_eviction_enabled()) {
        auto* feature_score_evict =
            dynamic_cast<FeatureScoreBasedEvict<weight_type>*>(
                feature_evict_.get());
        if (feature_score_evict) {
          feature_score_evict->update_feature_score_statistics(
              block, 0, shard_id, true);
        }
      }
    }
    hbm_post_invalidations_locked(shard_id);
  }

  /// get_int8_rows_with_metaheader for rows that live only in HBM: the
  /// MetaHeader from the tombstone (with the later access time) and the HBM
  /// row's bytes. Resolves the same rows as hbm_fill_rows (missing rows read
  /// back all zero), on hbm_executor_.
  void hbm_fill_int8_rows(
      const at::Tensor& id_values,
      const at::Tensor& rows,
      std::vector<int64_t> tombstoned_before) {
    if (!hbm_ || id_values.numel() == 0) {
      return;
    }
    const auto* id_ptr = id_values.const_data_ptr<int64_t>();
    auto positions = hbm_tombstoned_positions(id_values);
    positions.insert(
        positions.end(), tombstoned_before.begin(), tombstoned_before.end());
    const auto zero_rows = at::nonzero(rows.ne(0).sum(1) == 0).view(-1);
    const auto* zero_ptr = zero_rows.const_data_ptr<int64_t>();
    positions.insert(positions.end(), zero_ptr, zero_ptr + zero_rows.numel());
    std::ranges::sort(positions);
    positions.erase(std::ranges::unique(positions).begin(), positions.end());
    if (positions.empty()) {
      return;
    }
    const int64_t row_bytes = rows.size(1);
    const int64_t hbm_row_bytes = hbm_->dim + kHbmQparamsBytes;
    auto* out = rows.data_ptr<uint8_t>();
    via_hbm_executor([&] {
      std::vector<int64_t> reread;
      std::vector<weight_type> decoded(max_D_);
      for (size_t begin = 0; begin < positions.size();
           begin += static_cast<size_t>(hbm_chunk_rows_)) {
        const auto end = std::min(
            positions.size(), begin + static_cast<size_t>(hbm_chunk_rows_));
        std::vector<int64_t> slots;
        slots.reserve(end - begin);
        for (auto i = begin; i < end; ++i) {
          slots.push_back(hbm_slot(id_ptr[positions[i]]));
        }
        const auto held = hbm_read_slots(slots, /*rows=*/true);
        for (auto i = begin; i < end; ++i) {
          const auto key = id_ptr[positions[i]];
          std::optional<uint64_t> meta;
          if (held.tags[i - begin] == key) {
            const auto shard_id = kv_db_utils::hash_shard(key, num_shards_);
            const auto lock = kv_store_.by(shard_id).rlock();
            const auto it = hbm_rows_[shard_id].find(key);
            if (it != hbm_rows_[shard_id].end()) {
              meta = it->second;
            }
          }
          if (!meta) {
            reread.push_back(positions[i]);
            continue;
          }
          FixedBlockPool::MetaHeader header{};
          std::memcpy(
              reinterpret_cast<char*>(&header) + sizeof(int64_t),
              &*meta,
              sizeof(uint64_t));
          header.key = key;
          header.used = true;
          header.timestamp = std::max<uint32_t>(
              header.timestamp,
              static_cast<uint32_t>(held.last_access[i - begin]));
          auto* dst = out + positions[i] * row_bytes;
          std::memset(dst, 0, row_bytes);
          std::memcpy(dst, &header, sizeof(header));
          auto* block = reinterpret_cast<weight_type*>(dst);
          const auto* hbm_row = held.rows.template const_data_ptr<uint8_t>() +
              (i - begin) * hbm_row_bytes;
          if (hbm_rows_are_int8_rows()) {
            copy_hbm_row_to_int8_block(hbm_row, block);
          } else {
            hbm_decode_row(hbm_row, decoded.data());
            write_int8_payload(
                FixedBlockPool::data_ptr<weight_type>(block),
                decoded.data(),
                max_D_);
          }
        }
      }
      if (!reread.empty()) {
        const auto index = at::tensor(reread, at::kLong);
        auto again = at::zeros({index.numel(), row_bytes}, rows.options());
        read_int8_blocks(id_values.index_select(0, index), again);
        rows.index_copy_(0, index, again);
      }
      return int64_t{0};
    });
  }

  /// get_kv_zch_eviction_metadata_impl for the ids the backend pass did not
  /// find (left 0): rows that live only in HBM, or that were demoted back to
  /// the backend after that pass. An id in neither tier keeps 0.
  void hbm_fill_metadata(const at::Tensor& indices, at::Tensor& metadata) {
    const auto id_values = indices.to(at::kLong).contiguous();
    const auto* id_ptr = id_values.const_data_ptr<int64_t>();
    auto* out = metadata.data_ptr<int64_t>();
    std::vector<int64_t> positions;
    std::vector<uint64_t> metas;
    for (int64_t i = 0; i < id_values.numel(); ++i) {
      if (out[i] != 0) {
        continue;
      }
      const auto shard_id = kv_db_utils::hash_shard(id_ptr[i], num_shards_);
      const auto lock = kv_store_.by(shard_id).rlock();
      const auto it = hbm_rows_[shard_id].find(id_ptr[i]);
      if (it != hbm_rows_[shard_id].end()) {
        positions.push_back(i);
        metas.push_back(it->second);
      } else if (const auto block = lock->find(id_ptr[i]);
                 block != lock->end()) {
        out[i] = static_cast<int64_t>(
            FixedBlockPool::get_metaheader_raw(block->second));
      }
    }
    via_hbm_executor([&] {
      for (size_t begin = 0; begin < positions.size();
           begin += static_cast<size_t>(hbm_chunk_rows_)) {
        const auto end = std::min(
            positions.size(), begin + static_cast<size_t>(hbm_chunk_rows_));
        std::vector<int64_t> slots;
        for (auto i = begin; i < end; ++i) {
          slots.push_back(hbm_slot(id_ptr[positions[i]]));
        }
        const auto held = hbm_read_slots(slots, /*rows=*/false);
        for (auto i = begin; i < end; ++i) {
          const auto last = std::max<uint32_t>(
              meta_timestamp(metas[i]),
              static_cast<uint32_t>(held.last_access[i - begin]));
          out[positions[i]] =
              static_cast<int64_t>(with_timestamp(metas[i], last));
        }
      }
      return int64_t{0};
    });
  }

  struct FlatShardInput {
    std::shared_ptr<const FlatPositions> positions;
    std::vector<size_t> shard_offsets;
  };

  size_t get_executor_thread_count(const int64_t configured_threads) const {
    const auto default_threads = std::max<size_t>(
        configured_threads, facebook::Proc::getCpuInfo().numCpuCores);
    const char* const value = std::getenv("KVZCH_DRAM_EXECUTOR_THREAD_CAP");
    if (value == nullptr || *value == '\0') {
      return default_threads;
    }

    int cap = 0;
    const auto [ptr, ec] =
        std::from_chars(value, value + std::strlen(value), cap);
    if (ec != std::errc{} || *ptr != '\0' || cap < 1) {
      LOG(WARNING) << "Ignoring invalid KVZCH_DRAM_EXECUTOR_THREAD_CAP='"
                   << value << "'";
      return default_threads;
    }
    if (!disable_random_init_ || collect_perf_stats_ ||
        !parallel_shard_input_) {
      LOG(WARNING) << "KVZCH_DRAM_EXECUTOR_THREAD_CAP requested but refused: "
                      "disable_random_init, KVZCH_SKIP_DRAM_PERF_TIMING, and "
                      "KVZCH_PARALLEL_DRAM_SHARD_INPUT must be enabled";
      return default_threads;
    }

    const auto executor_threads =
        std::min(default_threads, static_cast<size_t>(cap));
    LOG(INFO) << "KVZCH DRAM executor threads capped from " << default_threads
              << " to " << executor_threads;
    return executor_threads;
  }

  static bool is_env_flag_set(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr && std::string_view(value) == "1";
  }

  static int get_requested_lookup_prefetch_distance() {
    constexpr int kMaxLookupPrefetchDistance = 64;
    const char* const value =
        std::getenv("KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE");
    if (value == nullptr || *value == '\0') {
      return 0;
    }

    int distance = 0;
    const auto [ptr, ec] =
        std::from_chars(value, value + std::strlen(value), distance);
    if (ec != std::errc{} || *ptr != '\0' || distance < 0) {
      LOG(WARNING) << "Ignoring invalid KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE='"
                   << value << "'";
      return 0;
    }
    if (distance > kMaxLookupPrefetchDistance) {
      LOG(WARNING) << "Clamping KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE from "
                   << distance << " to " << kMaxLookupPrefetchDistance;
      return kMaxLookupPrefetchDistance;
    }
    return distance;
  }

  static int get_requested_read_splits_per_shard() {
    constexpr int kMaxReadSplitsPerShard = 8;
    const char* const value = std::getenv("KVZCH_DRAM_READ_SPLITS_PER_SHARD");
    if (value == nullptr || *value == '\0') {
      return 1;
    }

    int splits = 0;
    const auto [ptr, ec] =
        std::from_chars(value, value + std::strlen(value), splits);
    if (ec != std::errc{} || *ptr != '\0' || splits < 1) {
      LOG(WARNING) << "Ignoring invalid KVZCH_DRAM_READ_SPLITS_PER_SHARD='"
                   << value << "'";
      return 1;
    }
    if (splits > kMaxReadSplitsPerShard) {
      LOG(WARNING) << "Clamping KVZCH_DRAM_READ_SPLITS_PER_SHARD from "
                   << splits << " to " << kMaxReadSplitsPerShard;
      return kMaxReadSplitsPerShard;
    }
    return splits;
  }

  // True when feature eviction is configured and enabled.
  bool is_eviction_enabled() const {
    return feature_evict_config_.has_value() &&
        feature_evict_config_.value()->trigger_mode_ !=
        EvictTriggerMode::DISABLED &&
        feature_evict_;
  }

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

    TORCH_CHECK(
        row_storage_data_ptr != nullptr,
        "DRAM initializer storage is required when random initialization is enabled");

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
          constexpr int64_t kMinIndicesPerParallelChunk = 4096;
          const int64_t num_chunks = parallel_shard_input_ && num_shards_ > 1
              ? std::min<int64_t>(
                    num_shards_,
                    (conv_count + kMinIndicesPerParallelChunk - 1) /
                        kMinIndicesPerParallelChunk)
              : 1;

          if (num_chunks > 1) {
            using ShardBuckets = std::vector<std::vector<int64_t>>;
            std::vector<ShardBuckets> chunk_buckets(
                num_chunks, ShardBuckets(num_shards_));
            std::vector<folly::Future<folly::Unit>> futures;
            futures.reserve(num_chunks);
            const int64_t indices_per_chunk =
                (conv_count + num_chunks - 1) / num_chunks;

            for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
              const int64_t begin = chunk * indices_per_chunk;
              const int64_t end =
                  std::min(conv_count, begin + indices_per_chunk);
              futures.emplace_back(
                  folly::via(executor_.get())
                      .thenValue([this,
                                  chunk,
                                  begin,
                                  end,
                                  indices_data_ptr,
                                  &chunk_buckets](folly::Unit) {
                        auto& buckets = chunk_buckets[chunk];
                        const size_t estimated_bucket_size =
                            static_cast<size_t>(end - begin) / num_shards_ + 1;
                        for (auto& bucket : buckets) {
                          bucket.reserve(estimated_bucket_size);
                        }
                        for (int64_t i = begin; i < end; ++i) {
                          const int64_t index = indices_data_ptr[i];
                          if (index < 0) {
                            continue;
                          }
                          const auto shard_id =
                              kv_db_utils::hash_shard(index, num_shards_);
                          buckets[shard_id].push_back(i);
                        }
                      }));
            }

            auto results = folly::collectAll(futures).get();
            for (auto& result : results) {
              (void)result.value();
            }

            shardid_to_indexes.reserve(num_shards_);
            for (int shard_id = 0; shard_id < num_shards_; ++shard_id) {
              size_t shard_size = 0;
              for (const auto& buckets : chunk_buckets) {
                shard_size += buckets[shard_id].size();
              }
              if (shard_size == 0) {
                continue;
              }
              auto& shard_indexes = shardid_to_indexes[shard_id];
              shard_indexes.reserve(shard_size);
              // Chunk order is input order, preserving last-write-wins for
              // duplicate IDs within a shard.
              for (auto& buckets : chunk_buckets) {
                auto& chunk_indexes = buckets[shard_id];
                shard_indexes.insert(
                    shard_indexes.end(),
                    std::make_move_iterator(chunk_indexes.begin()),
                    std::make_move_iterator(chunk_indexes.end()));
              }
            }
            return;
          }

          // Size the map once, and each shard's vector on first touch. Without
          // this every shard grows from empty, so a call spreading ~264k ids
          // over 32 shards does ~13 reallocate-and-copy rounds per shard.
          // Reserving on first touch rather than up front matters: creating an
          // entry for every shard would make empty shards visible to the
          // callers, all of which iterate this map and spawn one executor task
          // per entry.
          shardid_to_indexes.reserve(num_shards_);
          const size_t est_per_shard = conv_count > 0
              ? static_cast<size_t>(conv_count / num_shards_) + 16
              : 0;
          // There could be negative indices, which we should skipp
          for (int i = 0; i < conv_count; i++) {
            auto index = int64_t(indices_data_ptr[i]);
            if (index < 0) {
              continue;
            }

            const auto shard_id = kv_db_utils::hash_shard(index, num_shards_);
            auto& shard_indexes = shardid_to_indexes[shard_id];
            if (shard_indexes.empty()) {
              shard_indexes.reserve(est_per_shard);
            }
            shard_indexes.push_back(i);
          }
        });

    // chunk request based on bucket sharding
    return shardid_to_indexes;
  }

  std::shared_ptr<FlatShardInput> shard_input_for_read(
      const at::Tensor& indices,
      const at::Tensor& count) {
    struct ChunkShardInput {
      int64_t begin{0};
      std::vector<uint16_t> shard_ids;
      std::vector<size_t> shard_counts;
    };

    auto result = std::make_shared<FlatShardInput>();

    FBGEMM_DISPATCH_INTEGRAL_TYPES(
        indices.scalar_type(),
        "dram_shard_input_for_read",
        [this, &indices, &count, &result] {
          using index_t = scalar_t;
          const auto conv_count = count.scalar_type() == at::ScalarType::Long
              ? *(count.const_data_ptr<int64_t>())
              : *(count.const_data_ptr<int32_t>());
          constexpr int64_t kMinIndicesPerParallelChunk = 4096;
          const int64_t num_chunks = std::min<int64_t>(
              num_shards_,
              (conv_count + kMinIndicesPerParallelChunk - 1) /
                  kMinIndicesPerParallelChunk);
          CHECK_GT(num_chunks, 1);
          CHECK(indices.is_contiguous());
          const auto* const indices_data_ptr =
              indices.const_data_ptr<index_t>();

          std::vector<ChunkShardInput> chunks(num_chunks);
          std::vector<folly::Future<folly::Unit>> futures;
          futures.reserve(num_chunks);
          const int64_t indices_per_chunk =
              (conv_count + num_chunks - 1) / num_chunks;
          for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
            const int64_t begin = chunk * indices_per_chunk;
            const int64_t end = std::min(conv_count, begin + indices_per_chunk);
            chunks[chunk].begin = begin;
            futures.emplace_back(
                folly::via(executor_.get())
                    .thenValue(
                        [this, &chunks, chunk, begin, end, indices_data_ptr](
                            folly::Unit) {
                          constexpr uint16_t kInvalidShard =
                              std::numeric_limits<uint16_t>::max();
                          auto& output = chunks[chunk];
                          output.shard_ids.reserve(end - begin);
                          output.shard_counts.resize(num_shards_);
                          for (int64_t i = begin; i < end; ++i) {
                            const int64_t index = indices_data_ptr[i];
                            if (index < 0) {
                              output.shard_ids.push_back(kInvalidShard);
                              continue;
                            }
                            const auto shard_id =
                                kv_db_utils::hash_shard(index, num_shards_);
                            output.shard_ids.push_back(
                                static_cast<uint16_t>(shard_id));
                            ++output.shard_counts[shard_id];
                          }
                        }));
          }

          auto completed = folly::collectAll(futures).get();
          for (auto& task : completed) {
            (void)task.value();
          }

          result->shard_offsets.resize(num_shards_ + 1);
          for (int64_t shard = 0; shard < num_shards_; ++shard) {
            size_t shard_size = 0;
            for (const auto& chunk : chunks) {
              shard_size += chunk.shard_counts[shard];
            }
            result->shard_offsets[shard + 1] =
                result->shard_offsets[shard] + shard_size;
          }

          auto positions = std::make_shared<FlatPositions>();
          positions->resize_without_initialization(
              result->shard_offsets.back());
          std::vector<size_t> chunk_shard_write_offsets(
              num_chunks * num_shards_);
          for (int64_t shard = 0; shard < num_shards_; ++shard) {
            auto write_offset = result->shard_offsets[shard];
            for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
              chunk_shard_write_offsets[chunk * num_shards_ + shard] =
                  write_offset;
              write_offset += chunks[chunk].shard_counts[shard];
            }
            CHECK_EQ(write_offset, result->shard_offsets[shard + 1]);
          }

          futures.clear();
          for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
            futures.emplace_back(
                folly::via(executor_.get())
                    .thenValue([this,
                                &chunks,
                                &chunk_shard_write_offsets,
                                positions,
                                chunk](folly::Unit) {
                      constexpr uint16_t kInvalidShard =
                          std::numeric_limits<uint16_t>::max();
                      const auto& input = chunks[chunk];
                      auto* const shard_write_offsets =
                          chunk_shard_write_offsets.data() +
                          chunk * num_shards_;
                      for (size_t offset = 0; offset < input.shard_ids.size();
                           ++offset) {
                        const auto shard_id = input.shard_ids[offset];
                        if (shard_id == kInvalidShard) {
                          continue;
                        }
                        (*positions)[shard_write_offsets[shard_id]++] =
                            input.begin + static_cast<int64_t>(offset);
                      }
                    }));
          }

          completed = folly::collectAll(futures).get();
          for (auto& task : completed) {
            (void)task.value();
          }
          result->positions = std::move(positions);
        });
    return result;
  }

  void flush_or_compact(const int64_t /*timestep*/) override {}

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
    CHECK_LE(
        row_width,
        (packed_ || int8_)
            ? FixedBlockPool::get_metaheader_dim<weight_type>() + max_D_
            : block_size_);
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
                                row_width * sizeof(weight_type));
                            continue;
                          }

                          // For weight KVT, offset=0 and it will read the whole
                          // row. For optimizer, offset=dim(metaheader) +
                          // emb_dim so it will only read the optimizer part
                          read_row_with_metaheader(
                              shard_id,
                              cached_iter->second,
                              width_offset,
                              row_width,
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
  template <bool insert_missing = true>
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
                        const bool restamp =
                            restamp_on_restore_ && is_training_;
                        const uint32_t restore_ts =
                            restamp ? FixedBlockPool::current_timestamp() : 0;
                        for (const auto& id_index : indexes) {
                          auto id = int64_t(indices_data_ptr[id_index]);
                          auto it = wlmap->find(id);
                          if (it == wlmap->end() && !insert_missing) {
                            continue;
                          }

                          FixedBlockPool::set_used(
                              weights_data_ptr + id_index * stride, true);

                          weight_type* block = nullptr;
                          bool new_block = false;
                          if (it != wlmap->end()) {
                            block = it->second;
                          } else {
                            // Key doesn't exist, allocate new block and
                            // insert.
                            block = pool->template allocate_t<weight_type>();
                            wlmap->insert({id, block});
                            new_block = true;
                            hbm_take_tombstone_locked(shard_id, id);
                          }
                          write_row_with_metaheader(
                              shard_id,
                              block,
                              weights_data_ptr + id_index * stride,
                              stride);
                          // The saved timestamp predates the outage, so
                          // after a long resume gap every restored row
                          // would be past TTL at once.
                          if (restamp) {
                            FixedBlockPool::set_timestamp(block, restore_ts);
                          }
                          if (enable_ssd_backend_) {
                            pool->set_dirty(block, true);
                          }

                          if (new_block) {
                            if (is_eviction_enabled()) {
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
                        hbm_post_invalidations_locked(shard_id);
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
  // Set when rows are stored in their group-wise INT4 publish encoding
  std::optional<serving_qdq::PackedLayout> packed_;
  // Set when rows are stored as row-wise INT8 (int8_storage)
  std::optional<serving_qdq::Int8Layout> int8_;
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
  std::atomic<bool> row_write_in_progress_{false};

  // Enrichment write pause/resume mechanism (similar to eviction)
  std::atomic<bool> laser_write_interrupt_{false};
  std::mutex laser_write_mutex_;
  std::condition_variable laser_write_cv_;
  int64_t stream_laser_suspensions_{0};
  bool laser_paused_for_forward_{false};
  int64_t active_laser_writers_{0};
  std::atomic<uint64_t> laser_write_epoch_{0};

  // Pre-initialized LaserClient for IGR enrichment (reused across fetches)
  std::shared_ptr<facebook::laser::LaserClient> laser_client_;

  // OpenTab/Maple reader for ONEFLOW_OPENTAB_SID enrichment (type-erased)
  oneflow_enrichment::ReaderPtr open_tab_reader_;

  // Optional SSD backend for existence checks during enrichment.
  // When set, enrichment will skip IDs that already exist in SSD,
  // avoiding unnecessary calls to external data sources.
  std::atomic<bool> enable_ssd_backend_{false};
  bool collect_perf_stats_{true};
  int dram_lookup_prefetch_distance_{0};
  int dram_read_splits_per_shard_{1};
  bool dram_read_split_zero_copy_{false};
  bool parallel_shard_input_{false};
  bool segmented_shard_reads_{false};
  bool skip_disabled_initializers_{false};
  bool refresh_timestamp_on_enrich_hit_{false};
  bool restamp_on_restore_{false};
  // Drops a refresh-only pass while the previous one is still running.
  std::atomic<bool> timestamp_refresh_in_progress_{false};
  std::atomic<int64_t> timestamp_refresh_ids_{0};
  std::atomic<int64_t> timestamp_refresh_wlock_us_{0};
  std::atomic<int64_t> timestamp_refresh_interrupts_{0};
  size_t executor_thread_count_{0};

  // Per shard, fp16 values of packed rows with no serving fixed point, keyed
  // by block payload address and guarded by the shard lock. An entry is
  // dropped when its block is rewritten; one left by an evicted block is
  // never read and goes when the block is reused.
  std::vector<folly::F14FastMap<const void*, std::vector<weight_type>>>
      fallback_rows_;
  std::atomic<int64_t> packed_fallback_writes_{0};
  mutable std::atomic<int64_t> packed_fallback_missing_{0};
  std::atomic<int64_t> packed_inexact_fixed_points_{0};
  std::atomic<int64_t> packed_requantized_writes_{0};
  // INT8 row writes whose encoding needed more than one re-quantization, held
  // a non-finite value (clamped), or did not converge.
  std::atomic<int64_t> int8_multi_pass_writes_{0};
  std::atomic<int64_t> int8_non_finite_writes_{0};
  std::atomic<int64_t> int8_not_converged_writes_{0};

  // Exclusive HBM tier (exclusive_hbm_register); unset when off.
  std::optional<HbmTier> hbm_;
  // Per shard, MetaHeader word (get_metaheader_raw) of each row whose only
  // copy is in HBM, guarded by the shard lock.
  std::vector<folly::F14FastMap<int64_t, uint64_t>> hbm_rows_;
  // Per shard, keys whose tombstone a write removed, still holding a slot;
  // guarded by the shard lock.
  std::vector<std::vector<int64_t>> hbm_pending_invalidations_;
  int64_t hbm_chunk_rows_{65536};
  int64_t hbm_ttl_s_{0};
  bool hbm_ttl_demote_{true};
  std::atomic<int64_t> hbm_promoted_{0};
  folly::Synchronized<std::exception_ptr> hbm_async_error_;
  std::atomic<bool> hbm_fail_next_promotion_{false};
  std::atomic<int64_t> hbm_promote_skipped_{0};
  std::atomic<int64_t> hbm_demoted_{0};
  std::atomic<int64_t> hbm_dropped_{0};
  std::atomic<int64_t> hbm_invalidated_{0};
  std::atomic<int64_t> hbm_requantized_{0};
  std::atomic<int64_t> hbm_tombstone_reads_{0};
  std::atomic<int64_t> hbm_tombstone_mask_reads_{0};
  // One thread: tag and tombstone changes run in order.
  std::unique_ptr<folly::CPUThreadPoolExecutor> hbm_executor_;
  // Destroyed before hbm_executor_, which its sweeps run on.
  folly::FunctionScheduler hbm_scheduler_;

  // Destroyed first so queued enrichment/refresh tasks finish while every
  // cache member they use, including the refresh atomics, is still alive.
  std::unique_ptr<folly::CPUThreadPoolExecutor> enrichment_executor_;

  // Reporting-only clock. Returns 0 when perf stats are off, which leaves every
  // *_duration accumulator at 0 - they are logged, never branched on.
  inline int64_t perf_now_usec() const {
    return collect_perf_stats_ ? facebook::WallClockUtil::NowInUsecFast() : 0;
  }
}; // class DramKVEmbeddingCache

} // namespace kv_mem
