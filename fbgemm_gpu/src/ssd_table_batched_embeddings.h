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

namespace ssd {

using namespace at;

void hostAsynchronousThreadPoolExecutor(void (*f)(void*), void* userData) {
  static folly::CPUThreadPoolExecutor g(1);
  g.add([f, userData]() { f(userData); });
}

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
    producer_ = std::make_unique<std::thread>([=] {
#if defined(__x86_64__) || defined(__i386__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
      VSLStreamStatePtr stream;
      CHECK_EQ(
          VSL_ERROR_OK, vslNewStream(&stream, VSL_BRNG_SFMT19937, random_seed));
      auto rng_uniform = [&](size_t n, float* ptr) {
        CHECK_EQ(
            VSL_ERROR_OK,
            vsRngUniform(
                VSL_RNG_METHOD_UNIFORM_STD,
                stream,
                n,
                ptr,
                uniform_init_lower,
                uniform_init_upper));
      };
      SCOPE_EXIT {
        vslDeleteStream(&stream);
      };

#else
      folly::Random::DefaultGenerator gen(random_seed);
      auto rng_uniform = [&](size_t n, float* ptr) {
        std::uniform_real_distribution<float> dis(
            uniform_init_lower, uniform_init_upper);
        for (auto i = 0; i < n; i++) {
          ptr[i] = dis(gen);
        }
      };
#endif
      if (row_storage_bitwidth == 32) {
        rng_uniform(kRowInitBufferSize * max_D, row_storage_.data_ptr<float>());
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
        // dequeued a row. Reinitialize and enqueue it.
        if (row_storage_bitwidth == 32) {
          rng_uniform(max_D, row_storage_.data_ptr<float>() + i * max_D);
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

class EmbeddingRocksDB : public std::enable_shared_from_this<EmbeddingRocksDB> {
 public:
  EmbeddingRocksDB(
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
      int64_t row_storage_bitwidth = 32) {
    // TODO: lots of tunables. NNI or something for this?
    rocksdb::Options options;
    options.create_if_missing = true;

    // TODO: probably not very compressible.
    options.compression = rocksdb::kNoCompression;

    // Lots of free memory on the TC, use large write buffers.
    options.write_buffer_size = write_buffer_size;
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
    options.allow_concurrent_memtable_write = true;
    options.enable_write_thread_adaptive_yield = true;
    options.inplace_update_support = false;
    options.avoid_unnecessary_blocking_io = true;

    options.use_direct_reads = true;
    options.use_direct_io_for_flush_and_compaction = true;

    if (rate_limit_mbps > 0) {
      rate_limiter_.reset(
          rocksdb::NewGenericRateLimiter(rate_limit_mbps * 1024 * 1024));
    }
    options.rate_limiter = rate_limiter_;

    // TODO: use fb303?
    options.statistics = rocksdb::CreateDBStatistics();
    options.stats_dump_period_sec = 600;

    rocksdb::BlockBasedTableOptions table_options;
    // Don't use block cache since we have a "user-mode" UVM/HBM row cache.
    table_options.no_block_cache = true;
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
    for (auto i = 0; i < num_shards; ++i) {
      auto shard_path = path + std::string("/shard_") + std::to_string(i);
      rocksdb::DB* db;
      auto s = rocksdb::DB::Open(options, shard_path, &db);
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
    executor_ = std::make_unique<folly::CPUThreadPoolExecutor>(num_shards);
    ro_.verify_checksums = false;
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

  void set(Tensor indices, Tensor weights, Tensor count) {
    RECORD_USER_SCOPE("EmbeddingRocksDB::set");
    std::vector<folly::Future<folly::Unit>> futures;
    auto count_ = count.item().toLong();
    for (auto shard = 0; shard < dbs_.size(); ++shard) {
      auto f =
          folly::via(executor_.get())
              .thenValue([=, &indices, &weights](folly::Unit) {
                AT_DISPATCH_FLOATING_TYPES_AND2(
                    at::ScalarType::Half,
                    at::ScalarType::Byte,
                    weights.scalar_type(),
                    "ssd_set",
                    [&] {
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
              });
      futures.push_back(std::move(f));
    }
    folly::collect(futures).wait();
  }

  void compact() {
    for (auto& db : dbs_) {
      db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
    }
  }
  void flush() {
    for (auto& db : dbs_) {
      db->Flush(rocksdb::FlushOptions());
    }
  }

  void get(Tensor indices, Tensor weights, Tensor count) {
    RECORD_USER_SCOPE("EmbeddingRocksDB::get");
    std::vector<folly::Future<folly::Unit>> futures;
    auto count_ = count.item().toLong();

    for (auto shard = 0; shard < dbs_.size(); ++shard) {
      auto f =
          folly::via(executor_.get())
              .thenValue([=, &indices, &weights](folly::Unit) {
                AT_DISPATCH_FLOATING_TYPES_AND2(
                    at::ScalarType::Half,
                    at::ScalarType::Byte,
                    weights.scalar_type(),
                    "ssd_get",
                    [&] {
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
                      auto row_storage_data_ptr =
                          initializers_[shard]
                              ->row_storage_.data_ptr<scalar_t>();
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
              });
      futures.push_back(std::move(f));
    }
    folly::collect(futures).wait();
  }
  void get_cuda(Tensor indices, Tensor weights, Tensor count) {
    // take reference to self to avoid lifetime issues.
    auto self = shared_from_this();
    std::function<void()>* functor = new std::function<void()>(
        [=]() { self->get(indices, weights, count); });
    auto callFunctor =
        [](cudaStream_t stream, cudaError_t status, void* userData) -> void {
      AT_CUDA_CHECK(status);
      auto* f = reinterpret_cast<std::function<void()>*>(userData);
      AT_CUDA_CHECK(cudaGetLastError());
      (*f)();
      // delete f; // unfortunately, this invoke destructors that call CUDA
      // API functions (e.g. caching host allocators issue cudaGetDevice(..),
      // etc)
      hostAsynchronousThreadPoolExecutor(
          [](void* userData) {
            auto* f = reinterpret_cast<std::function<void()>*>(userData);
            delete f;
          },
          userData);
    };
    AT_CUDA_CHECK(cudaStreamAddCallback(
        at::cuda::getCurrentCUDAStream(), callFunctor, functor, 0));
  }

  void
  set_cuda(Tensor indices, Tensor weights, Tensor count, int64_t timestep) {
    // take reference to self to avoid lifetime issues.
    auto self = shared_from_this();
    std::function<void()>* functor = new std::function<void()>([=]() {
      self->set(indices, weights, count);
      // Only do manual Flush/Compactions if enabled
      if (memtable_flush_period_ > 0) {
        {
          RECORD_USER_SCOPE("FlushCompactIfNecessary");
          if (!done_staggered_flushes_) {
            self->flush_if_necessary(timestep);
          } else {
            self->compact_if_necessary(timestep);
          }
        }
      }
    });
    auto callFunctor =
        [](cudaStream_t stream, cudaError_t status, void* userData) -> void {
      AT_CUDA_CHECK(status);
      auto* f = reinterpret_cast<std::function<void()>*>(userData);
      AT_CUDA_CHECK(cudaGetLastError());
      (*f)();
      // delete f; // unfortunately, this invoke destructors that call CUDA
      // API functions (e.g. caching host allocators issue cudaGetDevice(..),
      // etc)
      hostAsynchronousThreadPoolExecutor(
          [](void* userData) {
            auto* f = reinterpret_cast<std::function<void()>*>(userData);
            delete f;
          },
          userData);
    };
    AT_CUDA_CHECK(cudaStreamAddCallback(
        at::cuda::getCurrentCUDAStream(), callFunctor, functor, 0));
  }

  void flush_if_necessary(int64_t timestep) {
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

  void compact_if_necessary(int64_t timestep) {
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

 private:
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
};

} // namespace ssd
