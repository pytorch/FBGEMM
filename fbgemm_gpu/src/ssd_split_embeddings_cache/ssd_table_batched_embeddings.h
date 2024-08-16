/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/experimental/coro/Collect.h>
#include <folly/experimental/coro/Task.h>
#include <torch/nn/init.h>
#include <iostream>
#ifdef FBGEMM_FBCODE
#include "common/strings/UUID.h"
#include "fb_rocksdb/DBMonitor/DBMonitor.h"
#include "fb_rocksdb/FbRocksDb.h"
#include "rocks/utils/FB303Stats.h"
#endif
#include "kv_db_table_batched_embeddings.h"
#include "torch/csrc/autograd/record_function_ops.h"

namespace ssd {

using namespace at;

// TODO: does this need to be different from the cache slot hashing function?
// Probably not right?
inline size_t db_shard(int64_t id, size_t num_shards) {
  auto hash = folly::hash::fnv64_buf(
      reinterpret_cast<const char*>(&id), sizeof(int64_t));
  __uint128_t wide = __uint128_t{num_shards} * hash;
  return static_cast<size_t>(wide >> 64);
}

// We can be a bit sloppy with host memory here.
constexpr size_t kRowInitBufferSize = 32 * 1024;

#ifdef FBGEMM_FBCODE
constexpr size_t num_ssd_drives = 8;
const std::string ssd_mount_point = "/data00_nvidia";
const size_t base_port = 136000;
#endif

class Initializer {
 public:
  Initializer(
      uint64_t random_seed,
      int64_t max_D,
      float uniform_init_lower,
      float uniform_init_upper,
      int64_t row_storage_bitwidth = 32)
      : producer_queue_(), consumer_queue_() {
    CHECK(
        row_storage_bitwidth == 32 || row_storage_bitwidth == 16 ||
        row_storage_bitwidth == 8);
    if (row_storage_bitwidth == 32) {
      row_storage_ = at::empty(
          {kRowInitBufferSize, max_D}, at::TensorOptions().dtype(at::kFloat));
    } else if (row_storage_bitwidth == 16) {
      row_storage_ = at::empty(
          {kRowInitBufferSize, max_D}, at::TensorOptions().dtype(at::kHalf));
    } else {
      row_storage_ = at::empty(
          {kRowInitBufferSize, max_D}, at::TensorOptions().dtype(at::kByte));
    }
    // Sanity check
    CHECK_EQ(row_storage_.element_size(), row_storage_bitwidth / 8);
    producer_ = std::make_unique<std::thread>([=] {
      const auto init = row_storage_.scalar_type() == at::ScalarType::Float ||
          row_storage_.scalar_type() == at::ScalarType::Half;
      if (init) {
        torch::nn::init::uniform_(
            row_storage_, uniform_init_lower, uniform_init_upper);
      }
      for (auto i = 0; i < kRowInitBufferSize; ++i) {
        producer_queue_.enqueue(i);
      }

      while (!stop_) {
        int64_t i;
        while (!stop_ &&
               !consumer_queue_.try_dequeue_until(
                   i,
                   std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(100))) {
          // loop.
        }
        if (stop_) {
          return;
        }

        if (init) {
          // dequeued a row. Reinitialize and enqueue it.
          torch::nn::init::uniform_(
              row_storage_[i], uniform_init_lower, uniform_init_upper);
        }
        producer_queue_.enqueue(i);
      }
    });
  }

  ~Initializer() {
    stop_ = true;
    producer_->join();
  }

  folly::USPSCQueue<int64_t, true> producer_queue_;
  folly::USPSCQueue<int64_t, true> consumer_queue_;

  Tensor row_storage_;
  std::atomic<bool> stop_{false};
  std::unique_ptr<std::thread> producer_;
};

class EmbeddingRocksDB : public kv_db::EmbeddingKVDB {
 public:
  explicit EmbeddingRocksDB(
      std::string path,
      int64_t num_shards,
      int64_t num_threads,
      int64_t memtable_flush_period,
      int64_t memtable_flush_offset,
      int64_t l0_files_per_compact,
      int64_t max_D,
      int64_t rate_limit_mbps,
      int64_t size_ratio,
      int64_t compaction_trigger,
      int64_t write_buffer_size,
      int64_t max_write_buffer_num,
      float uniform_init_lower,
      float uniform_init_upper,
      int64_t row_storage_bitwidth = 32,
      int64_t cache_size = 0,
      bool use_passed_in_path = false,
      int64_t tbe_unqiue_id = 0,
      int64_t l2_cache_size_gb = 0)
      : kv_db::EmbeddingKVDB(l2_cache_size_gb) {
    // TODO: lots of tunables. NNI or something for this?
    rocksdb::Options options;
    options.create_if_missing = true;

    // TODO: probably not very compressible.
    options.compression = rocksdb::kNoCompression;

    // Lots of free memory on the TC, use large write buffers.
    // max_write_buffer_num is per rocksdb shard level, write_buffer_size is tbe
    // level to calc individual buffer size we need to have total buffer size
    // per tbe / # db shards / # buffer per shards
    int64_t write_buffer_size_per_buffer =
        int64_t(write_buffer_size / num_shards / max_write_buffer_num);
    options.write_buffer_size = write_buffer_size_per_buffer;
    options.max_write_buffer_number = max_write_buffer_num;
    options.min_write_buffer_number_to_merge = 2;
    options.target_file_size_base = int64_t(2) * 1024 * 1024 * 1024;

    options.compaction_style = rocksdb::kCompactionStyleUniversal;
    options.compaction_options_universal.size_ratio = size_ratio;
    options.compaction_options_universal.min_merge_width = 2;
    // size amplification ratio = (size(R1) + size(R2) + ... size(Rn-1)) /
    // size(Rn)
    options.compaction_options_universal.max_size_amplification_percent = 400;
    options.level0_file_num_compaction_trigger = compaction_trigger;
    options.level0_slowdown_writes_trigger = 32;
    options.level0_stop_writes_trigger = 64;
    options.prefix_extractor.reset(
        rocksdb::NewFixedPrefixTransform(sizeof(int64_t)));
    // Partial Pipeline Options
    // options.allow_concurrent_memtable_write = false;
    // options.inplace_update_support = true;
    // Full Pipeline Options
    options.allow_concurrent_memtable_write = false;
    options.enable_write_thread_adaptive_yield = true;
    // inplace_update_support = false means we will apend kv pair in write
    // buffer even we saw duplications, this quickly fills up the buffer and
    // causing flush set this to true to make update on the existing key
    // allow_concurrent_memtable_write is toggled in pair with
    // inplace_update_support
    options.inplace_update_support = true;
    options.avoid_unnecessary_blocking_io = true;

    options.use_direct_reads = true;
    options.use_direct_io_for_flush_and_compaction = true;

    if (rate_limit_mbps > 0) {
      rate_limiter_.reset(
          rocksdb::NewGenericRateLimiter(rate_limit_mbps * 1024 * 1024));
    }
    options.rate_limiter = rate_limiter_;

    // TODO: use fb303?
#ifdef FBGEMM_FBCODE
    options.statistics =
        std::make_shared<facebook::rocks::FB303Stats>("tbe_metrics");
#else
    options.statistics = rocksdb::CreateDBStatistics();
#endif
    options.stats_dump_period_sec = 600;

    // no bloom filter on the last level, checkout https://fburl.com/ne99girf
    options.optimize_filters_for_hits = true;

    rocksdb::BlockBasedTableOptions table_options;

    if (cache_size > 0) {
      table_options.block_cache = rocksdb::NewLRUCache(cache_size);
      table_options.cache_index_and_filter_blocks = true;
    } else {
      table_options.no_block_cache = true;
    }

    table_options.index_type = rocksdb::BlockBasedTableOptions::kHashSearch;
    table_options.data_block_index_type =
        rocksdb::BlockBasedTableOptions::kDataBlockBinaryAndHash;
    table_options.data_block_hash_table_util_ratio = 0.75;
    table_options.checksum = rocksdb::ChecksumType::kNoChecksum;
    table_options.format_version = 5;
    table_options.read_amp_bytes_per_bit = 1;

    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(16));
    options.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));
    options.memtable_prefix_bloom_size_ratio = 0.05;
    options.memtable_whole_key_filtering = true;
    options.max_background_jobs = num_threads;
    options.env->SetBackgroundThreads(4, rocksdb::Env::HIGH);
    options.env->SetBackgroundThreads(1, rocksdb::Env::LOW);

    options.max_open_files = -1;

#ifdef FBGEMM_FBCODE
    auto serviceInfo = std::make_shared<facebook::fb_rocksdb::ServiceInfo>();
    serviceInfo->oncall = "pyper_training";
    serviceInfo->service_name = "ssd_offloading_rocksb";
    auto db_monitor_options = facebook::fb_rocksdb::DBMonitorOptions();
    db_monitor_options.fb303Prefix = "tbe_metrics";

    std::string tbe_uuid = "";
    if (!use_passed_in_path) {
      path = ssd_mount_point;
      tbe_uuid = facebook::strings::generateUUID();
    }
    std::string used_path = "";
#endif
    for (auto i = 0; i < num_shards; ++i) {
#ifdef FBGEMM_FBCODE
      int ssd_drive_idx = i % num_ssd_drives;
      std::string ssd_idx_tbe_id_str = "";
      if (!use_passed_in_path) {
        ssd_idx_tbe_id_str =
            std::to_string(ssd_drive_idx) + std::string("/") + tbe_uuid;
      }
      auto shard_path =
          path + ssd_idx_tbe_id_str + std::string("_shard") + std::to_string(i);
      used_path += shard_path + ", ";
#else
      auto shard_path = path + std::string("/shard_") + std::to_string(i);
#endif
      rocksdb::DB* db;

#ifdef FBGEMM_FBCODE
      auto s = facebook::fb_rocksdb::openRocksDB(
          options,
          shard_path,
          &db,
          serviceInfo,
          facebook::fb_rocksdb::getDefaultProfileOptions(),
          db_monitor_options);
#else
      auto s = rocksdb::DB::Open(options, shard_path, &db);
#endif
      if (!s.ok() && s.code() == rocksdb::Status::kInvalidArgument &&
          (options.use_direct_reads ||
           options.use_direct_io_for_flush_and_compaction)) {
        LOG(WARNING)
            << "Warning, Requested DirectIO, but not supported on destination: "
            << shard_path;
        options.use_direct_reads = false;
        options.use_direct_io_for_flush_and_compaction = false;
        LOG(WARNING)
            << "Trying again, any subsequent failures will be fatal...";
        s = rocksdb::DB::Open(options, shard_path, &db);
      }
      CHECK(s.ok()) << s.ToString();
      dbs_.emplace_back(db);
      auto* gen = at::check_generator<at::CPUGeneratorImpl>(
          at::detail::getDefaultCPUGenerator());
      {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        initializers_.push_back(std::make_unique<Initializer>(
            gen->random64(),
            max_D,
            uniform_init_lower,
            uniform_init_upper,
            row_storage_bitwidth));
      }
    }
#ifdef FBGEMM_FBCODE
    LOG(INFO) << "TBE actual used_path: " << used_path;
#endif
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(num_shards);
    ro_.verify_checksums = false;
    ro_.async_io = true;
    wo_.disableWAL = true;
    wo_.sync = false;

    // Setup staggered manual compaction data members
    memtable_flush_period_ = memtable_flush_period;
    if (memtable_flush_period_ > 0) {
      done_staggered_flushes_ = false;
      memtable_flush_offset_ = memtable_flush_offset;
      l0_files_per_compact_ = l0_files_per_compact;
      compaction_period_ = memtable_flush_period_ * l0_files_per_compact *
          options.min_write_buffer_number_to_merge;
      int64_t period_per_shard = memtable_flush_period_ / num_shards;
      CHECK_GT(period_per_shard, 0);
      // We want to stagger memory flushes (and then later
      // stagger all compactions)

      for (int64_t i = 0; i < num_shards; i++) {
        shard_flush_compaction_deadlines_.push_back(
            memtable_flush_offset_ + (i * period_per_shard));
      }
    }
  }

  folly::coro::Task<void> set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    RECORD_USER_SCOPE("EmbeddingRocksDB::set");
    std::vector<folly::coro::TaskWithExecutor<void>> tasks;
    auto count_ = count.item().toLong();

    for (auto shard = 0; shard < dbs_.size(); ++shard) {
      tasks.emplace_back(
          folly::coro::co_invoke(
              [this, &indices, &weights, count_, shard]() mutable
              -> folly::coro::Task<void> {
                FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
                    weights.scalar_type(), "ssd_set", [&] {
                      CHECK(indices.is_contiguous());
                      CHECK(weights.is_contiguous());
                      auto indices_acc = indices.accessor<int64_t, 1>();
                      auto D = weights.size(1);
                      CHECK_EQ(indices.size(0), weights.size(0));
                      {
                        rocksdb::WriteBatch batch(
                            (2 * (count_ + dbs_.size() - 1) / dbs_.size()) *
                            (sizeof(int64_t) + sizeof(scalar_t) * D));
                        for (auto i = 0; i < count_; ++i) {
                          // TODO: Check whether this is OK
                          if (indices_acc[i] == -1) {
                            continue;
                          }
                          if (db_shard(indices_acc[i], dbs_.size()) != shard) {
                            continue;
                          }
                          batch.Put(
                              rocksdb::Slice(
                                  reinterpret_cast<const char*>(
                                      &(indices.data_ptr<int64_t>()[i])),
                                  sizeof(int64_t)),
                              rocksdb::Slice(
                                  reinterpret_cast<const char*>(
                                      &(weights.data_ptr<scalar_t>()[i * D])),
                                  D * sizeof(scalar_t)));
                        }
                        auto s = dbs_[shard]->Write(wo_, &batch);
                        CHECK(s.ok());
                      }
                    });
                co_return;
              })
              .scheduleOn(executor_.get()));
    }
    co_await folly::coro::collectAllRange(std::move(tasks));
  }

  folly::coro::Task<void> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    RECORD_USER_SCOPE("EmbeddingRocksDB::get");
    std::vector<folly::coro::TaskWithExecutor<void>> tasks;
    auto count_ = count.item().toLong();

    for (auto shard = 0; shard < dbs_.size(); ++shard) {
      tasks.emplace_back(
          folly::coro::co_invoke(
              [this, &indices, &weights, count_, shard]() mutable
              -> folly::coro::Task<void> {
                FBGEMM_DISPATCH_FLOAT_HALF_AND_BYTE(
                    weights.scalar_type(), "ssd_get", [&] {
                      CHECK(indices.is_contiguous());
                      CHECK(weights.is_contiguous());
                      auto indices_data_ptr = indices.data_ptr<int64_t>();
                      auto D = weights.size(1);
                      CHECK_EQ(indices.size(0), weights.size(0));
                      auto weights_data_ptr = weights.data_ptr<scalar_t>();
                      FOLLY_DECLARE_REUSED(keys, std::vector<rocksdb::Slice>);
                      FOLLY_DECLARE_REUSED(shard_ids, std::vector<int32_t>);
                      FOLLY_DECLARE_REUSED(
                          cfs, std::vector<rocksdb::ColumnFamilyHandle*>);
                      FOLLY_DECLARE_REUSED(
                          values, std::vector<rocksdb::PinnableSlice>);
                      FOLLY_DECLARE_REUSED(
                          statuses, std::vector<rocksdb::Status>);
                      auto* dcf = dbs_[shard]->DefaultColumnFamily();
                      for (auto i = 0; i < count_; ++i) {
                        // "no-op"/empty evicted tensor
                        if (indices_data_ptr[i] == -1) {
                          continue;
                        }
                        if (db_shard(indices_data_ptr[i], dbs_.size()) !=
                            shard) {
                          continue;
                        }
                        shard_ids.push_back(i);
                      }
                      std::sort(
                          shard_ids.begin(),
                          shard_ids.end(),
                          [&](int32_t lhs, int32_t rhs) {
                            const auto lhs_key = rocksdb::Slice(
                                reinterpret_cast<const char*>(
                                    &(indices_data_ptr[lhs])),
                                sizeof(int64_t));
                            const auto rhs_key = rocksdb::Slice(
                                reinterpret_cast<const char*>(
                                    &(indices_data_ptr[rhs])),
                                sizeof(int64_t));
                            return lhs_key.compare(rhs_key) < 0;
                          });
                      for (const auto& i : shard_ids) {
                        const auto key = rocksdb::Slice(
                            reinterpret_cast<const char*>(
                                &(indices_data_ptr[i])),
                            sizeof(int64_t));
                        keys.push_back(key);
                        cfs.push_back(dcf);
                      }
                      CHECK_EQ(shard_ids.size(), keys.size());
                      CHECK_EQ(shard_ids.size(), cfs.size());

                      values.resize(keys.size());
                      statuses.resize(keys.size());
                      dbs_[shard]->MultiGet(
                          ro_,
                          keys.size(),
                          cfs.data(),
                          keys.data(),
                          values.data(),
                          statuses.data(),
                          /*sorted_input=*/true);
                      const auto& init_storage =
                          initializers_[shard]->row_storage_;
                      // Sanity check
                      TORCH_CHECK(
                          init_storage.scalar_type() == weights.scalar_type(),
                          "init_storage (",
                          toString(init_storage.scalar_type()),
                          ") and weights scalar (",
                          toString(weights.scalar_type()),
                          ") types mismatch");
                      auto row_storage_data_ptr =
                          init_storage.data_ptr<scalar_t>();
                      for (auto j = 0; j < keys.size(); ++j) {
                        const auto& s = statuses[j];
                        int64_t i = shard_ids[j];
                        const auto& value = values[j];
                        if (s.ok()) {
                          if (!std::is_same<scalar_t, uint8_t>::value) {
                            CHECK_EQ(value.size(), D * sizeof(scalar_t));
                          }
                          std::copy(
                              reinterpret_cast<const scalar_t*>(value.data()),
                              reinterpret_cast<const scalar_t*>(
                                  value.data() + value.size()),
                              &(weights_data_ptr[i * D]));
                        } else {
                          CHECK(s.IsNotFound());
                          int64_t row_index;
                          initializers_[shard]->producer_queue_.dequeue(
                              row_index);
                          std::copy(
                              &(row_storage_data_ptr[row_index * D]),
                              &(row_storage_data_ptr[row_index * D + D]),
                              &(weights_data_ptr[i * D]));
                          initializers_[shard]->consumer_queue_.enqueue(
                              row_index);
                        }
                      }
                    });
                co_return;
              })
              .scheduleOn(executor_.get()));
    }
    co_await folly::coro::collectAllRange(std::move(tasks));
  }

  void compact() override {
    for (auto& db : dbs_) {
      db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
    }
  }

  void flush() override {
    for (auto& db : dbs_) {
      db->Flush(rocksdb::FlushOptions());
    }
  }

 private:
  void flush_or_compact(const int64_t timestep) override {
    // Only do manual Flush/Compactions if enabled
    if (memtable_flush_period_ > 0) {
      {
        RECORD_USER_SCOPE("FlushCompactIfNecessary");
        if (!done_staggered_flushes_) {
          flush_if_necessary(timestep);
        } else {
          compact_if_necessary(timestep);
        }
      }
    }
  }

  void flush_if_necessary(const int64_t timestep) {
    for (int64_t i = 0; i < dbs_.size(); i++) {
      if (shard_flush_compaction_deadlines_[i] == timestep) {
        rocksdb::FlushOptions fo;
        fo.wait = false;
        fo.allow_write_stall = false;
        dbs_[i]->Flush(fo);
        if (i == dbs_.size() - 1) {
          done_staggered_flushes_ = true;
          int64_t period_per_shard = compaction_period_ / dbs_.size();
          int64_t offset = memtable_flush_offset_ + compaction_period_;
          for (int64_t j = 0; j < dbs_.size(); j++) {
            shard_flush_compaction_deadlines_[j] =
                offset + (j * period_per_shard);
          }
        }
      }
    }
  }

  void compact_if_necessary(const int64_t timestep) {
    for (int64_t i = 0; i < dbs_.size(); i++) {
      if (shard_flush_compaction_deadlines_[i] == timestep) {
        rocksdb::ColumnFamilyMetaData meta;
        dbs_[i]->GetColumnFamilyMetaData(&meta);
        int32_t num_level0 = meta.levels[0].files.size();
        if (num_level0 >= l0_files_per_compact_) {
          dbs_[i]->CompactRange(
              rocksdb::CompactRangeOptions(), nullptr, nullptr);
        }
        shard_flush_compaction_deadlines_[i] += compaction_period_;
      }
    }
  }

  std::vector<std::unique_ptr<rocksdb::DB>> dbs_;
  std::vector<std::unique_ptr<Initializer>> initializers_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;
  rocksdb::ReadOptions ro_{};
  rocksdb::WriteOptions wo_{};
  std::shared_ptr<rocksdb::RateLimiter> rate_limiter_;
  std::vector<int64_t> shard_flush_compaction_deadlines_;
  bool done_staggered_flushes_;
  int64_t memtable_flush_offset_;
  int64_t memtable_flush_period_;
  int64_t compaction_period_;
  int64_t l0_files_per_compact_;
}; // class EmbeddingKVDB

} // namespace ssd
