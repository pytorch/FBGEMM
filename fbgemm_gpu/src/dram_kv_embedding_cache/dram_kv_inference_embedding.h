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
#include <folly/logging/xlog.h>
#include <servicerouter/client/cpp2/ServiceRouter.h>
#include <thrift/lib/cpp2/protocol/CompactProtocol.h>
#include <thrift/lib/cpp2/protocol/Serializer.h>
#include <torch/script.h>
#include "common/time/Time.h"

#include "../ssd_split_embeddings_cache/initializer.h"
#include "SynchronizedShardedMap.h"
#include "fbgemm_gpu/split_embeddings_cache/kv_db_cpp_utils.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "feature_evict.h"
#include "fixed_block_pool.h"
#include "kv_inference_embedding_interface.h"

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
class DramKVInferenceEmbedding
    : public KVInferenceEmbeddingInterface<weight_type> {
 public:
  /// DramKVInferenceEmbedding constructor
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
  explicit DramKVInferenceEmbedding(
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      std::optional<c10::intrusive_ptr<kv_mem::FeatureEvictConfig>>
          feature_evict_config,
      int64_t num_shards = 8,
      int64_t num_threads = 32,
      int64_t row_storage_bitwidth = 32,
      bool enable_async_update = false,
      std::optional<at::Tensor> table_dims = std::nullopt,
      std::optional<at::Tensor> hash_size_cumsum = std::nullopt,
      bool disable_random_init = false)
      : max_D_(max_D),
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
        feature_evict_config_(std::move(feature_evict_config)),
        disable_random_init_(disable_random_init) {
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(std::max<size_t>(
        num_threads, facebook::Proc::getCpuInfo().numCpuCores));
    initialize_initializers(
        num_shards,
        max_D,
        uniform_init_lower,
        uniform_init_upper,
        row_storage_bitwidth,
        false /* disable_random_init */);
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
          false /* is_train */);
    }
    LOG(INFO) << "DramKVInferenceEmbedding initialized: disable_random_init "
              << disable_random_init_;
  }

  void initialize_initializers(
      int64_t num_shards,
      int64_t max_D,
      double uniform_init_lower,
      double uniform_init_upper,
      int64_t row_storage_bitwidth,
      bool disable_random_init) override {
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

  void set_kv_db_sync(
      const at::Tensor& /*indices*/,
      const at::Tensor& /*weights*/,
      const at::Tensor& /*count*/,
      std::optional<uint32_t> /*inplace_update_ts*/) override {
    throw std::runtime_error("set_kv_db_sync is not implemented for DRAM");
  }

  void get_kv_db_sync(
      const at::Tensor& /*indices*/,
      const at::Tensor& /*weights*/,
      const at::Tensor& /*count*/) override {
    throw std::runtime_error("get_kv_db_sync is not implemented for DRAM");
  }

  folly::SemiFuture<std::vector<folly::Unit>> inference_set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      std::optional<uint32_t> inplace_update_ts) override {
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
                        {
                          // 1st step, collect hit/miss per inplace update chunk
                          auto rlmap = kv_store_.by(shard_id).rlock();
                          for (const auto idx : indexes) {
                            auto id = int64_t(indices_data_ptr[idx]);
                            if (auto it = rlmap->find(id); it != rlmap->end()) {
                              hit_info.emplace_back(idx, it->second);
                            } else {
                              miss_info.emplace_back(id, idx);
                            }
                          }
                          hit_cnt = hit_info.size();
                          miss_cnt = miss_info.size();
                          // 2nd step, update hits while holding rlock to prevent
                          // eviction from invalidating block pointers
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
                        }  // rlmap released here

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
        .thenValue([this](
                       const std::vector<std::tuple<int64_t, int64_t>>&
                           tuples) {
          auto hit_cnt = 0;
          auto miss_cnt = 0;
          for (const auto& [h_cnt, m_cnt] : tuples) {
            hit_cnt += h_cnt;
            miss_cnt += m_cnt;
          }
          inplace_update_hit_cnt_ += hit_cnt;
          inplace_update_miss_cnt_ += miss_cnt;
          auto total_count = hit_cnt + miss_cnt;
          LOG_EVERY_MS(INFO, 5000) << fmt::format(
              "inference_set_kv_db_async: hit count {}, miss count {}, inplace update hit rate {}",
              hit_cnt,
              miss_cnt,
              total_count ? static_cast<double>(hit_cnt) / total_count : 0.0);

          return std::vector<folly::Unit>(tuples.size());
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
    // assuming get is called once each iteration and only by train
    // iteration(excluding state_dict)
    auto start_ts = facebook::WallClockUtil::NowInUsecFast();
    pause_ongoing_eviction(); // noop calls, no impact if called multiple times
    std::vector<
        folly::Future<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>>>
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
                FBGEMM_DISPATCH_INTEGRAL_TYPES(
                    indices.scalar_type(),
                    "get_kv_db_async_impl",
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
                     &local_read_missing_load] {
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

                      if (!wlmap->empty() && !disable_random_init_) {
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
                        }
                      }
                    });
                return std::tuple{
                    local_read_lookup_cache_total_duration,
                    local_read_fill_row_storage_total_duration,
                    local_read_cache_hit_copy_total_duration,
                    local_read_aquire_lock_duration,
                    local_read_missing_load};
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
          int64_t read_lookup_cache_total_duration = 0;
          int64_t read_fill_row_storage_total_duration = 0;
          int64_t read_cache_hit_copy_total_duration = 0;
          int64_t read_acquire_lock_total_duration = 0;
          int64_t read_missing_load = 0;
          for (
              const auto& [lookup_cache_dur, fill_row_storage_dur, cache_hit_copy_dur, acquire_lock_dur, missing_load] :
              results) {
            read_lookup_cache_total_duration += lookup_cache_dur;
            read_fill_row_storage_total_duration += fill_row_storage_dur;
            read_cache_hit_copy_total_duration += cache_hit_copy_dur;
            read_acquire_lock_total_duration += acquire_lock_dur;
            read_missing_load += missing_load;
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

  void compact() override {}

  void trigger_feature_evict(
      std::optional<uint32_t> inplace_update_ts = std::nullopt) override {
    if (feature_evict_) {
      if (inplace_update_ts.has_value() &&
          feature_evict_config_.value()->trigger_strategy_ ==
              EvictTriggerStrategy::BY_TIMESTAMP_THRESHOLD) {
        auto* tt_evict = dynamic_cast<TimeThresholdBasedEvict<weight_type>*>(
            feature_evict_.get());
        CHECK(tt_evict != nullptr);
        tt_evict->set_eviction_timestamp_threshold(inplace_update_ts.value());
      }
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

  size_t get_map_actual_used_chunk_in_bytes() const override {
    return kv_store_.getActualUsedChunkInBytes();
  }

  size_t get_num_rows() const override {
    return kv_store_.getNumRows();
  }

  void resume_ongoing_eviction(bool force_resume = false) override {
    if (!force_resume) {
      return;
    }
    if (feature_evict_) {
      feature_evict_->resume();
    }
  }

  void pause_ongoing_eviction(bool force_pause = false) override {
    if (!force_pause) {
      return;
    }
    if (!feature_evict_config_.has_value()) {
      return;
    }
    if (feature_evict_config_.value()->trigger_mode_ !=
        EvictTriggerMode::DISABLED) {
      if (feature_evict_) {
        feature_evict_->pause();
      }
    }
  }

  // for inference only, this logs the total hit/miss count
  // this should be called at the end of full/delta snapshot chunk by chunk
  // update
  void log_inplace_update_stats() override {
    int reset_val = 0;

    auto inplace_update_hit_cnt = inplace_update_hit_cnt_.exchange(reset_val);
    auto inplace_update_miss_cnt = inplace_update_miss_cnt_.exchange(reset_val);
    auto total_cnt = inplace_update_hit_cnt + inplace_update_miss_cnt;
    LOG_EVERY_MS(INFO, 5000)
        << "inplace update stats: hit count: " << inplace_update_hit_cnt
        << ", miss count: " << inplace_update_miss_cnt
        << ", total count: " << total_cnt << ", hit ratio: "
        << (total_cnt > 0 ? (double)inplace_update_hit_cnt / total_cnt : 0.0);
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

  std::vector<double> get_dram_kv_perf(
      const int64_t step,
      const int64_t interval) override {
    std::vector<double> ret(23, 0); // num metrics
    if (step > 0 && step % interval == 0) {
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

      ret[0] = dram_read_total_duration / interval;
      ret[1] = dram_read_sharding_total_duration / interval;
      ret[2] = dram_read_cache_hit_copy_duration / interval;
      ret[3] = dram_read_fill_row_storage_duration / interval;
      ret[4] = dram_read_lookup_cache_duration / interval;
      ret[5] = dram_read_acquire_lock_duration / interval;
      ret[6] = dram_read_missing_load / interval;
      ret[7] = dram_write_sharding_total_duration / interval;

      ret[8] = dram_fwd_l1_eviction_write_total_duration / interval;
      ret[9] = dram_fwd_l1_eviction_write_allocate_duration / interval;
      ret[10] = dram_fwd_l1_eviction_write_cache_copy_duration / interval;
      ret[11] = dram_fwd_l1_eviction_write_lookup_cache_duration / interval;
      ret[12] = dram_fwd_l1_eviction_write_acquire_lock_duration_ / interval;
      ret[13] = dram_fwd_l1_eviction_write_missing_load_ / interval;

      ret[14] = dram_bwd_l1_cnflct_miss_write_total_duration / interval;
      ret[15] = dram_bwd_l1_cnflct_miss_write_allocate_duration / interval;
      ret[16] = dram_bwd_l1_cnflct_miss_write_cache_copy_duration / interval;
      ret[17] = dram_bwd_l1_cnflct_miss_write_lookup_cache_duration / interval;
      ret[18] = dram_bwd_l1_cnflct_miss_write_acquire_lock_duration_ / interval;
      ret[19] = dram_bwd_l1_cnflct_miss_write_missing_load_ / interval;

      ret[20] = get_map_used_memsize_in_bytes();
      ret[21] = get_map_actual_used_chunk_in_bytes();

      ret[22] = get_num_rows();
    }
    return ret;
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
  std::vector<int64_t> sub_table_dims_;
  std::vector<int64_t> sub_table_hash_cumsum_;
  std::optional<c10::intrusive_ptr<FeatureEvictConfig>> feature_evict_config_;
  std::unique_ptr<FeatureEvict<weight_type>> feature_evict_;
  std::atomic<int> current_iter_{0};
  bool disable_random_init_ = false;

  // perf stats
  std::atomic<int64_t> read_total_duration_{0};
  std::atomic<int64_t> read_sharding_total_duration_{0};
  std::atomic<int64_t> read_cache_hit_copy_avg_duration_{0};
  std::atomic<int64_t> read_fill_row_storage_avg_duration_{0};
  std::atomic<int64_t> read_lookup_cache_total_avg_duration_{0};
  std::atomic<int64_t> read_acquire_lock_avg_duration_{0};
  std::atomic<int64_t> read_missing_load_avg_{0};
  std::atomic<int64_t> write_sharding_total_duration_{0};

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
}; // class DramKVInferenceEmbedding

} // namespace kv_mem
