/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <fb303/ServiceData.h>
#include <gtest/gtest.h>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_inference_wrapper.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_inference_embedding.h"

namespace fbgemm_gpu {
namespace {

class Gate {
 public:
  void pause() {
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    changed_.notify_all();
    TORCH_CHECK(
        changed_.wait_for(
            lock, std::chrono::seconds(60), [&] { return released_; }),
        "Timed out at controlled backend gate");
  }

  bool wait_entered() {
    std::unique_lock<std::mutex> lock(mutex_);
    return changed_.wait_for(
        lock, std::chrono::seconds(60), [&] { return entered_; });
  }

  void release() {
    std::scoped_lock lock(mutex_);
    released_ = true;
    changed_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool entered_ = false;
  bool released_ = false;
};

struct ReleaseGate {
  std::shared_ptr<Gate> gate;
  ~ReleaseGate() {
    gate->release();
  }
};

class ControlledBackend : public kv_mem::DramKVInferenceEmbedding<uint8_t> {
 public:
  ControlledBackend()
      : DramKVInferenceEmbedding(
            72,
            0,
            0,
            c10::make_intrusive<kv_mem::FeatureEvictConfig>(
                3,
                4,
                0,
                0,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                0,
                0,
                0),
            4,
            4,
            8,
            false,
            std::nullopt,
            at::tensor({0, 10000}, at::kLong),
            /*disable_random_init=*/true) {}

  void pause_next_read(const std::shared_ptr<Gate>& gate) {
    std::scoped_lock lock(control_mutex_);
    read_gate_ = gate;
  }

  void pause_next_write(const std::shared_ptr<Gate>& gate) {
    std::scoped_lock lock(control_mutex_);
    write_gates_.push_back(gate);
  }

  void fail_next_write() {
    std::scoped_lock lock(control_mutex_);
    fail_write_ = true;
  }

  folly::SemiFuture<std::vector<folly::Unit>> get_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count) override {
    std::shared_ptr<Gate> gate;
    {
      std::scoped_lock lock(control_mutex_);
      gate = std::exchange(read_gate_, nullptr);
    }
    return DramKVInferenceEmbedding::get_kv_db_async(indices, weights, count)
        .deferValue([gate](std::vector<folly::Unit> result) {
          if (gate) {
            gate->pause();
          }
          return result;
        });
  }

  folly::SemiFuture<std::vector<folly::Unit>> inference_set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      std::optional<uint32_t> timestamp) override {
    std::shared_ptr<Gate> gate;
    bool fail;
    {
      std::scoped_lock lock(control_mutex_);
      if (!write_gates_.empty()) {
        gate = write_gates_.front();
        write_gates_.pop_front();
      }
      fail = std::exchange(fail_write_, false);
    }
    return DramKVInferenceEmbedding::inference_set_kv_db_async(
               indices, weights, count, timestamp)
        .deferValue([gate, fail](std::vector<folly::Unit> result) {
          if (gate) {
            gate->pause();
          }
          TORCH_CHECK(!fail, "Controlled failure after backend write");
          return result;
        });
  }

  void set_kv_db_sync(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      std::optional<uint32_t> timestamp) override {
    folly::coro::blockingWait(
        inference_set_kv_db_async(indices, weights, count, timestamp));
  }

  void trigger_feature_evict(std::optional<uint32_t> /*timestamp*/) override {
    TORCH_CHECK(!fail_evict_start, "Controlled eviction start failure");
  }

  void resume_ongoing_eviction(bool /*force_resume*/) override {}

  void set_disable_random_init(bool disable_random_init) override {
    ++disable_random_init_calls;
    DramKVInferenceEmbedding::set_disable_random_init(disable_random_init);
  }

  void wait_until_eviction_done() override {
    if (evict_gate) {
      evict_gate->pause();
    }
    const auto keys = at::tensor({evicted_key}, at::kLong);
    const auto rows = at::zeros({1, 72}, at::kByte);
    const auto count = at::tensor({1}, at::kLong);
    folly::coro::blockingWait(
        DramKVInferenceEmbedding::inference_set_kv_db_async(
            keys, rows, count, 1000));
    TORCH_CHECK(
        !fail_evict_completion, "Controlled eviction completion failure");
  }

  std::shared_ptr<Gate> evict_gate;
  std::atomic<int> disable_random_init_calls{0};
  int64_t evicted_key = 1;
  bool fail_evict_start = false;
  bool fail_evict_completion = false;

 private:
  std::mutex control_mutex_;
  std::shared_ptr<Gate> read_gate_;
  std::deque<std::shared_ptr<Gate>> write_gates_;
  bool fail_write_ = false;
};

// {keys, zero rows, backend keys, HBM unique keys, HBM unique hits} exported
// to fb303 so far, summed over devices.
std::vector<int64_t> exported_coverage() {
  const std::vector<std::string> names{
      "keys",
      "zero_rows",
      "backend_keys",
      "hbm_unique_keys",
      "hbm_unique_hits"};
  std::map<std::string, int64_t> counters;
  facebook::fb303::fbData->getCounters(counters);
  std::vector<int64_t> totals(names.size());
  for (const auto& [key, value] : counters) {
    if (!key.starts_with("fbgemm.kv_lookup.")) {
      continue;
    }
    for (size_t i = 0; i < names.size(); ++i) {
      if (key.ends_with("." + names[i] + ".sum")) {
        totals[i] += value;
      }
    }
  }
  return totals;
}

std::vector<int64_t> minus(
    const std::vector<int64_t>& after,
    const std::vector<int64_t>& before) {
  std::vector<int64_t> delta(after.size());
  for (size_t i = 0; i < after.size(); ++i) {
    delta[i] = after[i] - before[i];
  }
  return delta;
}

int64_t disabled_backends_gauge() {
  return facebook::fb303::fbData
      ->getCounterIfExists("fbgemm.kv_hbm_cache.disabled_backends")
      .value_or(0);
}

class HbmGenerationTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (at::getNumGPUs() < 2) {
      GTEST_SKIP() << "Needs 2 GPUs";
    }
    // Disabled caches stay disabled unless a test opts into recovery.
    set_kv_hbm_cache_recover_interval_ms_for_testing(0);
    backend_ = std::make_shared<ControlledBackend>();
    full_ = make_wrapper(backend_);
    early_ = make_wrapper(backend_);
    write(full_, 1, 10);
    write(full_, 2, 11);
  }

  void TearDown() override {
    set_kv_hbm_cache_recover_interval_ms_for_testing(60000);
  }

  static std::shared_ptr<DramKVEmbeddingInferenceWrapper>
  make_unplanned_wrapper(const std::shared_ptr<ControlledBackend>& backend) {
    auto wrapper = std::make_shared<DramKVEmbeddingInferenceWrapper>(
        4, 0.0, 0.0, /*disable_random_init=*/true);
    wrapper->set_kv_backend(backend);
    wrapper->init_with_row_alignments({{10000, 128, 3}}, 1, 1, 8, std::nullopt);
    return wrapper;
  }

  std::shared_ptr<DramKVEmbeddingInferenceWrapper> make_wrapper(
      const std::shared_ptr<ControlledBackend>& backend) {
    auto wrapper = make_unplanned_wrapper(backend);
    wrapper->set_hbm_cache_rows(1024);
    wrapper->set_hbm_cache_fused_lookup(true);
    wrapper->set_hbm_cache_fused_admission(true);
    wrapper->set_hbm_cache_direct_output(GetParam());
    return wrapper;
  }

  static void write(
      const std::shared_ptr<DramKVEmbeddingInferenceWrapper>& wrapper,
      int64_t key,
      uint8_t value) {
    wrapper->set_embeddings(
        at::tensor({key}, at::kLong),
        at::full({1, 72}, value, at::kByte),
        1000);
  }

  static at::Tensor read(
      const std::shared_ptr<DramKVEmbeddingInferenceWrapper>& wrapper,
      int64_t key,
      int device) {
    return read_keys(wrapper, at::tensor({key, key}, at::kLong), device);
  }

  void write_backend(int64_t key, uint8_t value) {
    backend_->set_kv_db_sync(
        at::tensor({key}, at::kLong),
        at::full({1, 72}, value, at::kByte),
        at::tensor({1}, at::kLong),
        1000);
  }

  static at::Tensor read_keys(
      const std::shared_ptr<DramKVEmbeddingInferenceWrapper>& wrapper,
      const at::Tensor& keys,
      int device) {
    const c10::Device cuda(at::kCUDA, device);
    const c10::DeviceGuard guard(cuda);
    return wrapper->get_embeddings(keys.to(cuda)).cpu();
  }

  static void expect_rows(const at::Tensor& rows, uint8_t value) {
    EXPECT_TRUE(rows.narrow(1, 0, 72).eq(value).all().item<bool>());
    EXPECT_TRUE(rows.narrow(1, 72, 8).eq(0).all().item<bool>());
  }

  std::shared_ptr<ControlledBackend> backend_;
  std::shared_ptr<DramKVEmbeddingInferenceWrapper> full_;
  std::shared_ptr<DramKVEmbeddingInferenceWrapper> early_;
};

TEST_P(HbmGenerationTest, RejectsAdmissionAfterCompletedUpdate) {
  const auto keys = at::arange(40, at::kLong);
  full_->set_embeddings(keys, at::full({40, 72}, 10, at::kByte), 1000);
  auto gate = std::make_shared<Gate>();
  backend_->pause_next_read(gate);
  auto pending =
      std::async(std::launch::async, [&] { return read_keys(full_, keys, 0); });
  const ReleaseGate release{gate};
  ASSERT_TRUE(gate->wait_entered());
  early_->set_embeddings(keys, at::full({40, 72}, 20, at::kByte), 1000);
  gate->release();
  expect_rows(pending.get(), 10);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[3], 40);
  EXPECT_EQ(full_->get_hbm_cache_fused_admission_stats()[0], 0);
  expect_rows(read_keys(full_, keys, 0), 20);
  EXPECT_EQ(full_->get_hbm_cache_fused_admission_stats()[1], 1);
  const auto admitted = full_->get_hbm_cache_stats()[2];
  expect_rows(read_keys(full_, keys, 0), 20);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], admitted);
}

TEST_P(HbmGenerationTest, LoaderShapedWriteInvalidatesLiveCache) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(early_, 1, 1), 10);
  full_->set_embeddings(
      at::tensor({1}, at::kLong).reshape({1, 1}),
      at::full({1, 72}, 20, at::kByte),
      1000);
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 1, 1), 20);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
}

// Invalidation probes a large write in chunks; keys in the first and last
// chunk are both invalidated.
TEST_P(HbmGenerationTest, LargeWriteInvalidatesKeysOfEveryChunk) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(full_, 2, 0), 11);
  auto keys = at::full({200000}, 3, at::kLong);
  keys[0] = 1;
  keys[-1] = 2;
  full_->set_embeddings(keys, at::full({200000, 72}, 20, at::kByte), 1000);
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(full_, 2, 0), 20);
}

TEST_P(HbmGenerationTest, LoadAllocatedCacheIsTheOnlyCacheOfItsTable) {
  full_->allocate_hbm_cache(c10::Device(at::kCUDA, 0));
  EXPECT_NE(full_->get_hbm_cache_data_ptr(), 0);

  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(early_, 1, 0), 10);
  EXPECT_EQ(full_->get_hbm_cache_data_ptr(), early_->get_hbm_cache_data_ptr());
  EXPECT_THROW(read(early_, 1, 1), c10::Error);
  EXPECT_THROW(
      early_->allocate_hbm_cache(c10::Device(at::kCUDA, 1)), c10::Error);
}

TEST_P(HbmGenerationTest, AllocationAdoptingALazyCacheStaysExclusive) {
  expect_rows(read(full_, 1, 0), 10);
  full_->allocate_hbm_cache(c10::Device(at::kCUDA, 0));
  EXPECT_THROW(read(early_, 1, 1), c10::Error);

  auto backend = std::make_shared<ControlledBackend>();
  auto first = make_wrapper(backend);
  auto second = make_wrapper(backend);
  read(first, 1, 0);
  read(second, 1, 1);
  EXPECT_THROW(
      first->allocate_hbm_cache(c10::Device(at::kCUDA, 0)), c10::Error);
}

TEST_P(HbmGenerationTest, SharedCacheNeedsOneLookupRowWidth) {
  expect_rows(read(full_, 1, 0), 10);
  // 120 INT4 values plus scale and bias are 68 bytes; they pad to the same
  // 80-byte CUDA row as the fixture's 72-byte rows.
  auto narrower = std::make_shared<DramKVEmbeddingInferenceWrapper>(
      4, 0.0, 0.0, /*disable_random_init=*/true);
  narrower->set_kv_backend(backend_);
  narrower->init_with_row_alignments({{10000, 120, 3}}, 1, 1, 8, std::nullopt);
  narrower->set_hbm_cache_rows(1024);
  EXPECT_THROW(read(narrower, 1, 0), c10::Error);
}

TEST_P(HbmGenerationTest, LookupWidthMismatchIsAReuseMismatch) {
  expect_rows(read(full_, 1, 0), 10);
  auto narrower = std::make_shared<DramKVEmbeddingInferenceWrapper>(
      4, 0.0, 0.0, /*disable_random_init=*/true);
  narrower->set_kv_backend(backend_);
  narrower->init_with_row_alignments({{10000, 120, 3}}, 1, 1, 8, std::nullopt);
  narrower->set_hbm_cache_rows(1024);

  const auto mismatch =
      narrower->hbm_cache_reuse_mismatch(c10::Device(at::kCUDA, 0));
  ASSERT_TRUE(mismatch.has_value());
  EXPECT_NE(mismatch->find("68-byte rows"), std::string::npos);
  EXPECT_FALSE(
      early_->hbm_cache_reuse_mismatch(c10::Device(at::kCUDA, 0)).has_value());
}

TEST_P(HbmGenerationTest, RebindingTheSameBackendReappliesDisableRandomInit) {
  const auto calls = backend_->disable_random_init_calls.load();
  full_->set_kv_backend(backend_);
  EXPECT_EQ(backend_->disable_random_init_calls.load(), calls + 1);
}

TEST_P(HbmGenerationTest, PaddedRowsMustBeZeroPastTheStorageWidth) {
  auto padded = at::zeros({1, 80}, at::kByte);
  padded.narrow(1, 0, 72).fill_(30);
  full_->set_embeddings(at::tensor({3}, at::kLong), padded, 1000);
  expect_rows(read(full_, 3, 0), 30);

  padded.narrow(1, 72, 8).fill_(1);
  EXPECT_THROW(
      full_->set_embeddings(at::tensor({3}, at::kLong), padded, 1000),
      c10::Error);
}

// A table without planned HBM cache rows stores CUDA-padded rows only when the
// padding is all zero; other wider rows are rejected as on trunk.
TEST_P(HbmGenerationTest, OnlyAServingShardPlannedWithoutRowsTakesZeroPadding) {
  auto padded = at::zeros({1, 80}, at::kByte);
  padded.narrow(1, 0, 72).fill_(30);

  auto unplanned =
      make_unplanned_wrapper(std::make_shared<ControlledBackend>());
  EXPECT_THROW(
      unplanned->set_embeddings(at::tensor({3}, at::kLong), padded, 1000),
      c10::Error);

  auto no_rows = make_unplanned_wrapper(std::make_shared<ControlledBackend>());
  no_rows->set_hbm_cache_rows(0);
  no_rows->set_embeddings(at::tensor({3}, at::kLong), padded, 1000);
  expect_rows(read(no_rows, 3, 0), 30);

  padded.narrow(1, 72, 8).fill_(1);
  EXPECT_THROW(
      no_rows->set_embeddings(at::tensor({3}, at::kLong), padded, 1000),
      c10::Error);
  EXPECT_THROW(
      no_rows->set_embeddings(
          at::tensor({3}, at::kLong), at::zeros({1, 96}, at::kByte), 1000),
      c10::Error);
}

TEST_P(HbmGenerationTest, FailedSnapshotOfBackendWithoutCachesIsNotCounted) {
  auto unplanned =
      make_unplanned_wrapper(std::make_shared<ControlledBackend>());
  const auto before = disabled_backends_gauge();
  const auto failed_load = [&] {
    auto snapshot = unplanned->begin_snapshot_update();
    throw std::runtime_error("Controlled snapshot load failure");
  };
  EXPECT_THROW(failed_load(), std::runtime_error);
  EXPECT_EQ(disabled_backends_gauge(), before);
}

TEST_P(HbmGenerationTest, UncachedBackendEvictsWithoutTheEvictionLock) {
  auto backend = std::make_shared<ControlledBackend>();
  auto first = make_unplanned_wrapper(backend);
  auto second = make_unplanned_wrapper(backend);
  backend->evict_gate = std::make_shared<Gate>();
  const ReleaseGate release{backend->evict_gate};
  auto waiting =
      std::async(std::launch::async, [&] { first->wait_evict_completion(); });
  ASSERT_TRUE(backend->evict_gate->wait_entered());

  auto evicting =
      std::async(std::launch::async, [&] { second->trigger_evict(1000); });
  EXPECT_EQ(
      evicting.wait_for(std::chrono::seconds(10)), std::future_status::ready);
  backend->evict_gate->release();
  waiting.get();
  evicting.get();
}

TEST_P(HbmGenerationTest, DisabledBackendsAreExportedToFb303) {
  expect_rows(read(full_, 1, 0), 10);
  const auto before = disabled_backends_gauge();
  backend_->fail_evict_start = true;
  EXPECT_THROW(full_->trigger_evict(1000), c10::Error);
  backend_->fail_evict_start = false;
  EXPECT_EQ(disabled_backends_gauge(), before + 1);
  EXPECT_TRUE(full_->recover_hbm_caches());
}

TEST_P(HbmGenerationTest, RowsTheBackendSkipsAreZero) {
  // A host fetch leaves non-zero bytes in the pinned pool.
  expect_rows(read(full_, 2, 0), 11);
  for (const int64_t key : {-1, -2}) {
    expect_rows(read(full_, key, 0), 0);
    expect_rows(read(full_, key, 0), 0);
  }
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], 1);
}

// The registry does not keep a cache alive: it is freed with the last wrapper
// that uses it, even while its backend lives on, e.g. bound to a table that
// has no planned rows.
TEST_P(HbmGenerationTest, CacheIsFreedWithTheLastWrapperThatUsesIt) {
  auto backend = std::make_shared<ControlledBackend>();
  const auto unplanned = make_unplanned_wrapper(backend);
  const c10::Device cuda(at::kCUDA, 0);
  auto planned = make_wrapper(backend);
  auto sharing = make_wrapper(backend);
  planned->allocate_hbm_cache(cuda);
  expect_rows(read(sharing, 1, 0), 0);
  EXPECT_EQ(
      planned->get_hbm_cache_data_ptr(), sharing->get_hbm_cache_data_ptr());

  planned.reset();
  EXPECT_TRUE(unplanned->hbm_cache_reuse_mismatch(cuda).has_value());
  sharing.reset();
  EXPECT_EQ(unplanned->hbm_cache_reuse_mismatch(cuda), std::nullopt);
}

TEST_P(HbmGenerationTest, FootprintCoversRowsTagsAndLru) {
  EXPECT_EQ(
      full_->hbm_cache_footprint_bytes(),
      1024 * (full_->get_cuda_row_bytes() + 2 * sizeof(int64_t)));
}

TEST_P(HbmGenerationTest, InvalidatesBothMethodsAndCards) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(early_, 1, 1), 10);
  const auto address0 = full_->get_hbm_cache_data_ptr();
  const auto address1 = early_->get_hbm_cache_data_ptr();
  EXPECT_NE(address0, address1);
  write(early_, 1, 20);
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 1, 1), 20);
  EXPECT_EQ(
      full_->get_hbm_cache_generation_stats()[0],
      early_->get_hbm_cache_generation_stats()[0]);
  EXPECT_GT(full_->get_hbm_cache_stats()[4], 0);
  EXPECT_GT(early_->get_hbm_cache_stats()[4], 0);
}

TEST_P(HbmGenerationTest, OverlappingWritersKeepCacheBypassed) {
  expect_rows(read(full_, 1, 0), 10);
  auto first_gate = std::make_shared<Gate>();
  auto second_gate = std::make_shared<Gate>();
  backend_->pause_next_write(first_gate);
  backend_->pause_next_write(second_gate);
  auto first = std::async(std::launch::async, [&] { write(full_, 1, 20); });
  const ReleaseGate release_first{first_gate};
  ASSERT_TRUE(first_gate->wait_entered());
  auto second = std::async(std::launch::async, [&] { write(early_, 2, 21); });
  const ReleaseGate release_second{second_gate};
  ASSERT_TRUE(second_gate->wait_entered());
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 2);
  const auto admitted = full_->get_hbm_cache_stats()[2];
  expect_rows(read(full_, 1, 0), 20);
  first_gate->release();
  first.get();
  EXPECT_EQ(early_->get_hbm_cache_generation_stats()[1], 1);
  expect_rows(read(full_, 2, 0), 21);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], admitted);
  second_gate->release();
  second.get();
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 2, 1), 21);
  EXPECT_GT(full_->get_hbm_cache_generation_stats()[2], 0);
}

TEST_P(HbmGenerationTest, FailedWriteClearsRowsAndReleasesWriter) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(early_, 1, 1), 10);
  backend_->fail_next_write();
  EXPECT_THROW(write(early_, 1, 20), c10::Error);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 1, 1), 20);
}

TEST_P(HbmGenerationTest, VersionedWriteInvalidatesCache) {
  expect_rows(read(full_, 1, 0), 10);
  early_->set_embeddings_with_value_versions(
      at::tensor({1}, at::kLong),
      at::full({1, 72}, 20, at::kByte),
      1000,
      at::tensor({1}, at::kLong),
      1,
      1);
  expect_rows(read(full_, 1, 0), 20);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
}

TEST_P(HbmGenerationTest, EvictionGuardLastsThroughCompletion) {
  expect_rows(read(full_, 1, 0), 10);
  backend_->evict_gate = std::make_shared<Gate>();
  full_->trigger_evict(1000);
  auto pending =
      std::async(std::launch::async, [&] { early_->wait_evict_completion(); });
  const ReleaseGate release{backend_->evict_gate};
  ASSERT_TRUE(backend_->evict_gate->wait_entered());
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 1);
  const auto admitted = full_->get_hbm_cache_stats()[2];
  expect_rows(read(full_, 1, 0), 10);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], admitted);
  backend_->evict_gate->release();
  pending.get();
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
  expect_rows(read(full_, 1, 0), 0);
}

TEST_P(HbmGenerationTest, FailedEvictionStartReleasesWriterAndBypassesCache) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(early_, 1, 1), 10);
  backend_->fail_evict_start = true;
  EXPECT_THROW(full_->trigger_evict(1000), c10::Error);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
  write(early_, 1, 20);
  const auto admitted = full_->get_hbm_cache_stats()[2];
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 1, 1), 20);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], admitted);
}

TEST_P(HbmGenerationTest, FailedEvictionCompletionNeverUsesStaleRows) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(early_, 1, 1), 10);
  full_->trigger_evict(1000);
  backend_->fail_evict_completion = true;
  EXPECT_THROW(early_->wait_evict_completion(), c10::Error);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
  const auto admitted = full_->get_hbm_cache_stats()[2];
  expect_rows(read(full_, 1, 0), 0);
  expect_rows(read(early_, 1, 1), 0);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], admitted);
}

TEST_P(HbmGenerationTest, DisabledCachesAreClearedAndReenabled) {
  expect_rows(read(full_, 1, 0), 10);
  const auto gauge = disabled_backends_gauge();
  backend_->fail_evict_start = true;
  EXPECT_THROW(full_->trigger_evict(1000), c10::Error);
  backend_->fail_evict_start = false;
  ASSERT_TRUE(early_->hbm_cache_disabled());
  EXPECT_EQ(disabled_backends_gauge(), gauge + 1);
  // A write the caches never saw: recovery must not serve the cached row.
  write_backend(1, 20);
  const auto admitted = full_->get_hbm_cache_stats()[2];
  expect_rows(read(full_, 1, 0), 20);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], admitted);

  EXPECT_TRUE(early_->recover_hbm_caches());

  EXPECT_FALSE(full_->hbm_cache_disabled());
  EXPECT_EQ(disabled_backends_gauge(), gauge);
  expect_rows(read(full_, 1, 0), 20);
  EXPECT_GT(full_->get_hbm_cache_stats()[2], admitted);
  const auto hits = full_->get_hbm_cache_stats()[0];
  expect_rows(read(full_, 1, 0), 20);
  EXPECT_GT(full_->get_hbm_cache_stats()[0], hits);
}

TEST_P(HbmGenerationTest, RecoveryWaitsForWritersAndRunsFromLookups) {
  expect_rows(read(full_, 1, 0), 10);
  backend_->fail_evict_start = true;
  EXPECT_THROW(full_->trigger_evict(1000), c10::Error);
  backend_->fail_evict_start = false;
  ASSERT_TRUE(full_->hbm_cache_disabled());
  backend_->evict_gate = std::make_shared<Gate>();
  full_->trigger_evict(1000);
  auto pending =
      std::async(std::launch::async, [&] { early_->wait_evict_completion(); });
  {
    const ReleaseGate release{backend_->evict_gate};
    ASSERT_TRUE(backend_->evict_gate->wait_entered());
    EXPECT_FALSE(full_->recover_hbm_caches());
    EXPECT_TRUE(full_->hbm_cache_disabled());
    backend_->evict_gate->release();
    pending.get();
  }
  // Off in this fixture: lookups leave the caches disabled.
  read(full_, 2, 0);
  EXPECT_TRUE(full_->hbm_cache_disabled());
  set_kv_hbm_cache_recover_interval_ms_for_testing(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  expect_rows(read(full_, 2, 0), 11);
  EXPECT_FALSE(full_->hbm_cache_disabled());
}

TEST_P(HbmGenerationTest, ReplacementKeepsInflightReadOnOriginalBackend) {
  auto replacement_backend = std::make_shared<ControlledBackend>();
  auto replacement = make_wrapper(replacement_backend);
  write(replacement, 1, 20);
  auto gate = std::make_shared<Gate>();
  backend_->pause_next_read(gate);
  auto pending =
      std::async(std::launch::async, [&] { return read(full_, 1, 0); });
  const ReleaseGate release{gate};
  ASSERT_TRUE(gate->wait_entered());
  full_->set_kv_backend(replacement_backend);
  gate->release();
  expect_rows(pending.get(), 10);
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 1, 1), 10);
}

TEST_P(HbmGenerationTest, SnapshotGuardLastsBetweenChunksOnBothCards) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(early_, 1, 1), 10);
  auto snapshot = full_->begin_snapshot_update();
  const auto admitted = full_->get_hbm_cache_stats()[2];
  write_backend(1, 20);
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 1, 1), 20);
  write(early_, 2, 21);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 1);
  expect_rows(read(full_, 2, 0), 21);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], admitted);
  snapshot->complete();
  snapshot.reset();
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 1, 1), 20);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
  EXPECT_GT(full_->get_hbm_cache_stats()[2], admitted);
}

TEST_P(HbmGenerationTest, SnapshotFailureBypassesUntilSuccessfulSnapshot) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(early_, 1, 1), 10);
  const auto failed_load = [&] {
    auto snapshot = full_->begin_snapshot_update();
    write_backend(1, 20);
    throw std::runtime_error("Controlled snapshot load failure");
  };
  EXPECT_THROW(failed_load(), std::runtime_error);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
  const auto admitted = full_->get_hbm_cache_stats()[2];
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 1, 1), 20);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], admitted);
  auto replacement = early_->begin_snapshot_update();
  write_backend(1, 30);
  replacement->complete();
  replacement.reset();
  expect_rows(read(full_, 1, 0), 30);
  expect_rows(read(early_, 1, 1), 30);
  EXPECT_GT(full_->get_hbm_cache_stats()[2], admitted);
}

TEST_P(HbmGenerationTest, OverlappingSuccessCannotClearSnapshotFailure) {
  expect_rows(read(full_, 1, 0), 10);
  auto first = full_->begin_snapshot_update();
  auto second = early_->begin_snapshot_update();
  write_backend(1, 20);
  first.reset();
  EXPECT_EQ(early_->get_hbm_cache_generation_stats()[1], 1);
  second->complete();
  second.reset();
  const auto admitted = full_->get_hbm_cache_stats()[2];
  expect_rows(read(full_, 1, 0), 20);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], admitted);
  auto recovery = full_->begin_snapshot_update();
  write_backend(1, 30);
  recovery->complete();
  recovery.reset();
  expect_rows(read(full_, 1, 0), 30);
  EXPECT_GT(full_->get_hbm_cache_stats()[2], admitted);
}

TEST_P(HbmGenerationTest, SnapshotFailureIsReportedAsDisabled) {
  expect_rows(read(full_, 1, 0), 10);
  const auto before = disabled_backends_gauge();
  const auto failed_load = [&] {
    auto snapshot = full_->begin_snapshot_update();
    write_backend(1, 20);
    throw std::runtime_error("Controlled snapshot load failure");
  };
  EXPECT_THROW(failed_load(), std::runtime_error);
  const auto gauge = disabled_backends_gauge();
  EXPECT_EQ(gauge, before + 1);
  EXPECT_FALSE(early_->recover_hbm_caches());

  auto replacement = early_->begin_snapshot_update();
  replacement->complete();
  replacement.reset();

  EXPECT_EQ(disabled_backends_gauge(), gauge - 1);
  EXPECT_TRUE(early_->recover_hbm_caches());
}

TEST_P(HbmGenerationTest, SnapshotGuardSurvivesBackendReplacement) {
  auto replacement_backend = std::make_shared<ControlledBackend>();
  auto replacement = make_wrapper(replacement_backend);
  write(replacement, 1, 20);
  auto snapshot = full_->begin_snapshot_update();
  full_->set_kv_backend(replacement_backend);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
  EXPECT_EQ(early_->get_hbm_cache_generation_stats()[1], 1);
  snapshot->complete();
  snapshot.reset();
  EXPECT_EQ(early_->get_hbm_cache_generation_stats()[1], 0);
  expect_rows(read(full_, 1, 0), 20);
  expect_rows(read(early_, 1, 1), 10);
}

TEST_P(HbmGenerationTest, SnapshotBeginRejectsInflightMissAdmission) {
  auto gate = std::make_shared<Gate>();
  backend_->pause_next_read(gate);
  auto pending =
      std::async(std::launch::async, [&] { return read(full_, 1, 0); });
  const ReleaseGate release{gate};
  ASSERT_TRUE(gate->wait_entered());
  auto snapshot = early_->begin_snapshot_update();
  write_backend(1, 20);
  snapshot->complete();
  snapshot.reset();
  gate->release();
  expect_rows(pending.get(), 10);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], 0);
  expect_rows(read(full_, 1, 0), 20);
}

TEST_P(HbmGenerationTest, SnapshotGuardSharesOwnerBetweenMethods) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(early_, 1, 0), 10);
  EXPECT_EQ(full_->get_hbm_cache_data_ptr(), early_->get_hbm_cache_data_ptr());
  auto snapshot = early_->begin_snapshot_update();
  const auto admitted = full_->get_hbm_cache_stats()[2];
  write_backend(1, 20);
  expect_rows(read(full_, 1, 0), 20);
  EXPECT_EQ(early_->get_hbm_cache_stats()[2], admitted);
  snapshot->complete();
  snapshot.reset();
  expect_rows(read(early_, 1, 0), 20);
  EXPECT_EQ(full_->get_hbm_cache_data_ptr(), early_->get_hbm_cache_data_ptr());
  EXPECT_GT(full_->get_hbm_cache_stats()[2], admitted);
}

TEST_P(HbmGenerationTest, AbsentIdsAreNotAdmitted) {
  const auto keys = at::tensor({1, 5}, at::kLong);
  for (int i = 0; i < 2; ++i) {
    const auto rows = read_keys(full_, keys, 0);
    expect_rows(rows.narrow(0, 0, 1), 10);
    expect_rows(rows.narrow(0, 1, 1), 0);
  }

  const std::vector<int64_t> expected_policy{0, 2};
  EXPECT_EQ(full_->get_hbm_cache_admission_policy_stats(), expected_policy);
  EXPECT_EQ(full_->get_hbm_cache_stats()[2], 1);
}

TEST_P(HbmGenerationTest, AbsentIdIsServedAndAdmittedAfterDelta) {
  const auto keys = at::tensor({5, 5}, at::kLong);
  expect_rows(read_keys(full_, keys, 0), 0);

  write(early_, 5, 30);
  expect_rows(read_keys(full_, keys, 0), 30);
  const auto admitted = full_->get_hbm_cache_stats()[2];
  const auto hits = full_->get_hbm_cache_stats()[0];
  expect_rows(read_keys(full_, keys, 0), 30);

  EXPECT_EQ(admitted, 1);
  EXPECT_GT(full_->get_hbm_cache_stats()[0], hits);
  EXPECT_EQ(full_->get_hbm_cache_admission_policy_stats()[1], 1);
}

TEST_P(HbmGenerationTest, KeyMinusOneNeverMatchesInvalidatedSlots) {
  // Fill every set, then invalidate the rows so each empty way keeps the bytes
  // of the key it last held.
  const auto keys = at::arange(1, 4097, at::kLong);
  for (int64_t key = 1; key <= 4096; ++key) {
    write_backend(key, 7);
  }
  read_keys(full_, keys, 0);
  early_->set_embeddings(keys, at::full({4096, 72}, 9, at::kByte), 1000);
  expect_rows(read(full_, -1, 0), 0);
  expect_rows(read(full_, -1, 0), 0);
}

TEST_P(HbmGenerationTest, InvalidatedSlotIsReusedBeforeLiveRows) {
  auto backend = std::make_shared<ControlledBackend>();
  auto wrapper = make_unplanned_wrapper(backend);
  // One set of 32 ways: every key competes for the same slots.
  wrapper->set_hbm_cache_rows(32);
  wrapper->set_hbm_cache_direct_output(GetParam());
  wrapper->set_embeddings(
      at::arange(33, at::kLong), at::full({33, 72}, 20, at::kByte), 1000);
  read_keys(wrapper, at::arange(32, at::kLong), 0);
  read_keys(wrapper, at::tensor({5}, at::kLong), 0);
  write(wrapper, 5, 30);
  read_keys(wrapper, at::tensor({32}, at::kLong), 0);
  const auto before = wrapper->get_hbm_cache_stats();

  read_keys(
      wrapper,
      at::cat({at::arange(5, at::kLong), at::arange(6, 32, at::kLong)}),
      0);

  const auto after = wrapper->get_hbm_cache_stats();
  EXPECT_EQ(after[0] - before[0], 31);
  EXPECT_EQ(after[1], before[1]);
}

TEST_P(HbmGenerationTest, ReportsLookupCoverage) {
  if (GetParam()) {
    GTEST_SKIP() << "Direct output reports unique misses only";
  }
  flush_kv_lookup_coverage();
  const auto before = exported_coverage();
  const auto keys = at::tensor({1, 1, 3}, at::kLong);
  full_->get_embeddings(keys);
  read_keys(full_, keys, 0);
  read_keys(full_, at::tensor({1, 1}, at::kLong), 0);
  flush_kv_lookup_coverage();

  // {keys, zero rows, backend keys, HBM unique keys, HBM unique hits} summed
  // over a CPU lookup, a cold CUDA lookup and a warm CUDA lookup.
  const std::vector<int64_t> expected{3 + 3 + 2, 1 + 1, 3 + 3, 2 + 1, 1};
  EXPECT_EQ(minus(exported_coverage(), before), expected);
}

// A KV table without a planned HBM cache (any table not declared
// KV_ZCH_GPU_BUCKET_WISE) keeps trunk behavior even when other tables in the
// process have caches and export coverage.
TEST_P(HbmGenerationTest, TableWithoutPlannedRowsHasNoCacheAndNoCoverage) {
  auto backend = std::make_shared<ControlledBackend>();
  auto unplanned = make_unplanned_wrapper(backend);
  write(unplanned, 1, 10);
  flush_kv_lookup_coverage();
  const auto before = exported_coverage();
  for (int i = 0; i < 2; ++i) {
    expect_rows(read(unplanned, 1, 0), 10);
  }
  unplanned->get_embeddings(at::tensor({1}, at::kLong));
  flush_kv_lookup_coverage();

  EXPECT_EQ(unplanned->get_hbm_cache_device(), "");
  EXPECT_EQ(unplanned->get_hbm_cache_stats(), std::vector<int64_t>(5, 0));
  EXPECT_EQ(unplanned->hbm_cache_footprint_bytes(), 0);
  EXPECT_EQ(minus(exported_coverage(), before), std::vector<int64_t>(5, 0));
}

// Writes of a table without a planned HBM cache take the trunk path: the
// backend's write guards are untouched while it has no HBM cache.
TEST_P(HbmGenerationTest, TableWithoutPlannedRowsWritesWithoutCacheGuards) {
  auto backend = std::make_shared<ControlledBackend>();
  auto unplanned = make_unplanned_wrapper(backend);
  const auto before = unplanned->get_hbm_cache_generation_stats();

  write(unplanned, 1, 10);
  unplanned->set_embeddings(
      at::tensor({2}, at::kLong), at::full({1, 72}, 11, at::kByte));

  EXPECT_EQ(unplanned->get_hbm_cache_generation_stats(), before);
  expect_rows(read(unplanned, 2, 0), 11);
  try {
    unplanned->set_embeddings(
        at::tensor({3}, at::kLong), at::full({1, 96}, 12, at::kByte));
    ADD_FAILURE() << "An over-wide row must be rejected";
  } catch (const c10::Error& e) {
    EXPECT_NE(
        std::string(e.what()).find(
            "Embedding update row is wider than KV storage"),
        std::string::npos);
  }
}

// A table without a planned cache that shares a backend with one that has a
// cache still guards that backend's caches on writes.
TEST_P(HbmGenerationTest, TableWithoutPlannedRowsGuardsASharedCachedBackend) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(full_, 1, 0), 10);
  auto unplanned = make_unplanned_wrapper(backend_);
  const auto generation = full_->get_hbm_cache_generation_stats()[0];

  write(unplanned, 1, 20);

  EXPECT_GT(full_->get_hbm_cache_generation_stats()[0], generation);
  expect_rows(read(full_, 1, 0), 20);
}

// Models a compact SSD backend: snapshot rows written while staging stay
// invisible until trigger_feature_evict publishes them as the only live rows.
class StagedBackend : public ControlledBackend {
 public:
  // Pauses the next write before it reaches the store.
  std::shared_ptr<Gate> write_gate;
  // Pauses the next key check, e.g. a snapshot guard's sweep.
  std::shared_ptr<Gate> contains_gate;

  void begin_staging() {
    std::scoped_lock lock(mutex_);
    staging_ = true;
    staged_.clear();
  }

  void abort_staging() {
    std::scoped_lock lock(mutex_);
    staging_ = false;
    staged_.clear();
  }

  folly::SemiFuture<std::vector<folly::Unit>> inference_set_kv_db_async(
      const at::Tensor& indices,
      const at::Tensor& weights,
      const at::Tensor& count,
      std::optional<uint32_t> timestamp) override {
    if (auto gate = std::exchange(write_gate, nullptr)) {
      gate->pause();
    }
    const auto keys = indices.reshape({-1});
    {
      std::scoped_lock lock(mutex_);
      for (int64_t i = 0; i < keys.numel(); ++i) {
        const auto key = keys[i].item<int64_t>();
        if (staging_ && timestamp) {
          staged_[key] = weights[i].clone();
        } else {
          live_.insert(key);
        }
      }
      if (staging_ && timestamp) {
        return folly::makeSemiFuture(std::vector<folly::Unit>{});
      }
    }
    return ControlledBackend::inference_set_kv_db_async(
        indices, weights, count, timestamp);
  }

  void trigger_feature_evict(std::optional<uint32_t> /*timestamp*/) override {
    std::scoped_lock lock(mutex_);
    staging_ = false;
    for (const auto key : live_) {
      if (!staged_.contains(key)) {
        write_live(key, at::zeros({72}, at::kByte));
      }
    }
    live_.clear();
    for (const auto& [key, row] : staged_) {
      write_live(key, row);
      live_.insert(key);
    }
    staged_.clear();
  }

  void wait_until_eviction_done() override {}

  std::optional<at::Tensor> contains_keys(const at::Tensor& indices) override {
    if (auto gate = std::exchange(contains_gate, nullptr)) {
      gate->pause();
    }
    std::scoped_lock lock(mutex_);
    auto present = at::zeros({indices.numel()}, at::kBool);
    for (int64_t i = 0; i < indices.numel(); ++i) {
      present[i] = live_.contains(indices[i].item<int64_t>());
    }
    return present;
  }

 private:
  void write_live(int64_t key, const at::Tensor& row) {
    // Skips ControlledBackend's override so staged rows reach live storage.
    folly::coro::blockingWait(
        // NOLINTNEXTLINE(bugprone-parent-virtual-call)
        DramKVInferenceEmbedding::inference_set_kv_db_async(
            at::tensor({key}, at::kLong),
            row.reshape({1, -1}),
            at::tensor({1}, at::kLong),
            1000));
  }

  std::mutex mutex_;
  bool staging_ = false;
  std::set<int64_t> live_;
  std::map<int64_t, at::Tensor> staged_;
};

class ServedSnapshotTest : public HbmGenerationTest {
 protected:
  void SetUp() override {
    HbmGenerationTest::SetUp();
    staged_ = std::make_shared<StagedBackend>();
    full_ = make_wrapper(staged_);
    early_ = make_wrapper(staged_);
    full_->set_values_immutable(true);
    write(full_, 1, 10);
    write(full_, 2, 11);
  }

  std::unique_ptr<KVBackendSnapshotGuard> begin_staged_snapshot() {
    auto snapshot = full_->begin_snapshot_update();
    staged_->begin_staging();
    return snapshot;
  }

  void publish(std::unique_ptr<KVBackendSnapshotGuard> snapshot) {
    publish_without_ending(*snapshot);
  }

  void publish_without_ending(KVBackendSnapshotGuard& snapshot) {
    full_->trigger_evict(1000);
    full_->wait_evict_completion();
    snapshot.complete();
  }

  int64_t hits() {
    return full_->get_hbm_cache_stats()[0];
  }

  std::shared_ptr<StagedBackend> staged_;
};

TEST_P(ServedSnapshotTest, CachedRowsServeWhileSnapshotIsStaged) {
  expect_rows(read(full_, 1, 0), 10);
  auto snapshot = begin_staged_snapshot();
  write(early_, 1, 10);
  write(early_, 3, 12);
  const auto hits_before = hits();

  expect_rows(read(full_, 1, 0), 10);

  EXPECT_EQ(hits(), hits_before + 1);
  expect_rows(read(full_, 3, 0), 0);
  publish(std::move(snapshot));
}

TEST_P(ServedSnapshotTest, TableWithoutImmutableValuesBypassesWhileStaged) {
  full_->set_values_immutable(false);
  expect_rows(read(full_, 1, 0), 10);
  auto snapshot = begin_staged_snapshot();
  const auto hits_before = hits();

  expect_rows(read(full_, 1, 0), 10);

  EXPECT_EQ(hits(), hits_before);
  publish(std::move(snapshot));
}

TEST_P(ServedSnapshotTest, PublicationInvalidatesKeysTheSnapshotDropped) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(full_, 2, 0), 11);
  auto snapshot = begin_staged_snapshot();
  write(early_, 1, 10);

  publish(std::move(snapshot));

  const auto hits_before = hits();
  expect_rows(read(full_, 1, 0), 10);
  EXPECT_EQ(hits(), hits_before + 1);
  expect_rows(read(full_, 2, 0), 0);
  const std::vector<int64_t> expected_sweep{1, 2, 1};
  const auto sweep = full_->get_hbm_cache_snapshot_sweep_stats();
  EXPECT_EQ(
      std::vector<int64_t>(sweep.begin(), sweep.begin() + 3), expected_sweep);
}

TEST_P(ServedSnapshotTest, RollbackBypassesCachesUntilGuardRefreshes) {
  expect_rows(read(full_, 1, 0), 10);
  auto snapshot = begin_staged_snapshot();
  early_->set_embeddings(
      at::tensor({1}, at::kLong), at::full({1, 72}, 15, at::kByte));
  snapshot->begin_rollback();
  // The backend may expose the staged row from here on.
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 1);
  const auto hits_before = hits();
  read(full_, 1, 0);
  EXPECT_EQ(hits(), hits_before);

  staged_->abort_staging();
  snapshot->rolled_back();
  snapshot.reset();
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
}

TEST_P(ServedSnapshotTest, BypassOutlastsAnOverlappingRefresh) {
  expect_rows(read(full_, 1, 0), 10);
  auto first = begin_staged_snapshot();
  publish_without_ending(*first);
  staged_->contains_gate = std::make_shared<Gate>();
  auto gate = staged_->contains_gate;
  const ReleaseGate release{gate};
  // The first guard ends its snapshot and pauses in its refresh.
  auto refreshing = std::async(std::launch::async, [&] { first.reset(); });
  ASSERT_TRUE(gate->wait_entered());

  auto second = begin_staged_snapshot();
  publish(std::move(second));

  // The first refresh is still running, so lookups keep bypassing.
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 1);
  gate->release();
  refreshing.get();
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
}

TEST_P(ServedSnapshotTest, PublicationBypassesCachesUntilGuardRefreshes) {
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(full_, 2, 0), 11);
  auto snapshot = begin_staged_snapshot();
  write(early_, 1, 10);
  early_->set_embeddings(
      at::tensor({1}, at::kLong), at::full({1, 72}, 14, at::kByte));
  expect_rows(read(full_, 1, 0), 14);

  full_->trigger_evict(1000);

  const auto hits_before = hits();
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(full_, 2, 0), 0);
  EXPECT_EQ(hits(), hits_before);
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 1);
  full_->wait_evict_completion();
  snapshot->complete();
  snapshot.reset();
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[1], 0);
  expect_rows(read(full_, 1, 0), 10);
  expect_rows(read(full_, 2, 0), 0);
}

TEST_P(ServedSnapshotTest, RolledBackSnapshotKeepsCachedRows) {
  expect_rows(read(full_, 1, 0), 10);
  auto snapshot = begin_staged_snapshot();
  write(early_, 3, 12);
  staged_->abort_staging();

  snapshot->rolled_back();
  snapshot.reset();

  const auto hits_before = hits();
  expect_rows(read(full_, 1, 0), 10);
  EXPECT_EQ(hits(), hits_before + 1);
  expect_rows(read(full_, 3, 0), 0);
}

// A failed served snapshot clears the caches, so it drops the keys of the
// deltas written while it staged instead of keeping them for a later one.
TEST_P(ServedSnapshotTest, FailedSnapshotDropsTheKeysOfDeltasItStaged) {
  expect_rows(read(full_, 1, 0), 10);
  auto snapshot = begin_staged_snapshot();
  early_->set_embeddings(
      at::tensor({1, 5}, at::kLong), at::full({2, 72}, 10, at::kByte));
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[4], 2);
  staged_->abort_staging();

  snapshot.reset();

  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[4], 0);
  const auto hits_before = hits();
  expect_rows(read(full_, 1, 0), 10);
  EXPECT_EQ(hits(), hits_before);
}

TEST_P(ServedSnapshotTest, StagedKeysOutliveAnOverlappingServedSnapshot) {
  expect_rows(read(full_, 1, 0), 10);
  auto first = begin_staged_snapshot();
  auto second = full_->begin_snapshot_update();
  early_->set_embeddings(
      at::tensor({5}, at::kLong), at::full({1, 72}, 13, at::kByte));
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[4], 1);

  first->rolled_back();
  first.reset();
  // The second snapshot still hides the delta's row, so its end must
  // invalidate the key again.
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[4], 1);

  staged_->abort_staging();
  second->rolled_back();
  second.reset();
  EXPECT_EQ(full_->get_hbm_cache_generation_stats()[4], 0);
}

TEST_P(ServedSnapshotTest, NewIdDeltaKeepsCachesServing) {
  expect_rows(read(full_, 1, 0), 10);
  staged_->write_gate = std::make_shared<Gate>();
  const ReleaseGate release{staged_->write_gate};
  auto gate = staged_->write_gate;
  auto pending = std::async(std::launch::async, [&] {
    early_->set_embeddings(
        at::tensor({5}, at::kLong), at::full({1, 72}, 13, at::kByte));
  });
  ASSERT_TRUE(gate->wait_entered());
  // While the new id is written, cached rows still serve.
  auto hits_before = hits();
  expect_rows(read(full_, 1, 0), 10);
  EXPECT_EQ(hits(), hits_before + 1);
  gate->release();
  pending.get();

  expect_rows(read(full_, 5, 0), 13);
  hits_before = hits();
  expect_rows(read(full_, 1, 0), 10);
  EXPECT_EQ(hits(), hits_before + 1);
}

TEST_P(ServedSnapshotTest, DeltaOfAnExistingIdStillBypassesCaches) {
  expect_rows(read(full_, 1, 0), 10);
  early_->set_embeddings(
      at::tensor({1}, at::kLong), at::full({1, 72}, 14, at::kByte));
  expect_rows(read(full_, 1, 0), 14);
}

// The new-id delta path belongs to tables with a planned HBM cache: a new-id
// delta through a wrapper without one, into a backend with caches, takes the
// regular write guard, which advances the backend generation at its begin and
// end.
TEST_P(ServedSnapshotTest, NewIdDeltaWithoutPlannedRowsTakesTheWriteGuard) {
  const auto delta =
      [](const std::shared_ptr<DramKVEmbeddingInferenceWrapper>& wrapper,
         int64_t key) {
        const auto before = wrapper->get_hbm_cache_generation_stats()[0];
        wrapper->set_embeddings(
            at::tensor({key}, at::kLong), at::full({1, 72}, 13, at::kByte));
        return wrapper->get_hbm_cache_generation_stats()[0] - before;
      };
  expect_rows(read(full_, 1, 0), 10);
  EXPECT_EQ(delta(early_, 5), 1);
  EXPECT_EQ(delta(make_unplanned_wrapper(staged_), 6), 2);
}

TEST_P(ServedSnapshotTest, PublicationMakesNewIdVisible) {
  auto snapshot = begin_staged_snapshot();
  write(early_, 1, 10);
  write(early_, 5, 30);

  publish(std::move(snapshot));

  expect_rows(read(full_, 5, 0), 30);
}

INSTANTIATE_TEST_SUITE_P(F1F2, HbmGenerationTest, ::testing::Bool());
INSTANTIATE_TEST_SUITE_P(F1F2, ServedSnapshotTest, ::testing::Bool());

} // namespace
} // namespace fbgemm_gpu
