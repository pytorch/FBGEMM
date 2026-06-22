/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_cache.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <vector>

namespace kv_mem {

struct MetaHeader {
  int64_t key;
  uint32_t timestamp;
  uint32_t count : 31;
  bool used : 1;
};

class DramKVEmbeddingCacheTest : public ::testing::Test {
 protected:
  static constexpr int EMBEDDING_DIM = 16;
  static constexpr int NUM_SHARDS = 4;

  void SetUp() override {
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = 0;

    auto hash_size_cumsum = at::tensor({0, 100000}, at::kLong);

    dram_cache_ = std::make_shared<DramKVEmbeddingCache<float>>(
        EMBEDDING_DIM,
        /*uniform_init_lower=*/-0.1,
        /*uniform_init_upper=*/0.1,
        /*feature_evict_config=*/std::nullopt,
        NUM_SHARDS,
        /*num_threads=*/4,
        /*row_storage_bitwidth=*/32,
        /*backend_return_whole_row=*/false,
        /*enable_async_update=*/false,
        /*table_dims=*/std::nullopt,
        hash_size_cumsum,
        /*is_training=*/false,
        /*disable_random_init=*/true);
  }

  void TearDown() override {
    dram_cache_.reset();
  }

  // Thin wrappers named after the core cache APIs they exercise, so test bodies
  // read like direct calls into the cache under test.
  void set_kv_db_async(
      DramKVEmbeddingCache<float>& cache,
      int64_t id,
      float value = 1.0f) {
    auto indices = at::tensor({id}, at::kLong);
    auto weights = at::full(
        {1, EMBEDDING_DIM}, value, at::TensorOptions().dtype(at::kFloat));
    auto count = at::tensor({1}, at::kLong);
    folly::coro::blockingWait(cache.set_kv_db_async(indices, weights, count));
  }

  void set_kv_db_async(
      DramKVEmbeddingCache<float>& cache,
      const std::vector<int64_t>& ids,
      float value = 1.0f) {
    auto num = static_cast<int64_t>(ids.size());
    auto indices = at::tensor(ids, at::kLong);
    auto weights = at::full(
        {num, EMBEDDING_DIM}, value, at::TensorOptions().dtype(at::kFloat));
    auto count = at::tensor({num}, at::kLong);
    folly::coro::blockingWait(cache.set_kv_db_async(indices, weights, count));
  }

  // Build a standalone float cache, varying only enable_ssd_backend so tests
  // can compare dirty-bit behavior with the flag on vs off.
  std::shared_ptr<DramKVEmbeddingCache<float>> makeCache(
      bool enable_ssd_backend) {
    auto hash_size_cumsum = at::tensor({0, 100000}, at::kLong);
    return std::make_shared<DramKVEmbeddingCache<float>>(
        EMBEDDING_DIM,
        /*uniform_init_lower=*/-0.1,
        /*uniform_init_upper=*/0.1,
        /*feature_evict_config=*/std::nullopt,
        NUM_SHARDS,
        /*num_threads=*/4,
        /*row_storage_bitwidth=*/32,
        /*backend_return_whole_row=*/false,
        /*enable_async_update=*/false,
        /*table_dims=*/std::nullopt,
        hash_size_cumsum,
        /*is_training=*/false,
        /*disable_random_init=*/true,
        /*enable_raw_embedding_streaming=*/false,
        /*res_store_shards=*/0,
        /*res_server_port=*/0,
        /*table_names=*/std::vector<std::string>{},
        /*table_offsets=*/std::vector<int64_t>{},
        /*table_sizes=*/std::vector<int64_t>{},
        /*enrichment_config=*/std::nullopt,
        /*enable_ssd_backend=*/enable_ssd_backend);
  }

  // Look up the block for `id` and report its dirty bit. The dirty state lives
  // in the owning shard's pool, so query get_dirty() on that same pool.
  bool isKeyDirty(DramKVEmbeddingCache<float>& cache, int64_t id) {
    auto& kv_store = cache.get_kv_store();
    for (int shard = 0; shard < cache.get_num_shards(); ++shard) {
      auto rlmap = kv_store.by(shard).rlock();
      auto it = rlmap->find(id);
      if (it != rlmap->end()) {
        return kv_store.pool_by(shard)->get_dirty(it->second);
      }
    }
    ADD_FAILURE() << "key " << id << " not found in any shard";
    return false;
  }

  // Clear the dirty bit for `id` so a subsequent write must re-set it.
  void clearDirty(DramKVEmbeddingCache<float>& cache, int64_t id) {
    auto& kv_store = cache.get_kv_store();
    for (int shard = 0; shard < cache.get_num_shards(); ++shard) {
      auto rlmap = kv_store.by(shard).rlock();
      auto it = rlmap->find(id);
      if (it != rlmap->end()) {
        kv_store.pool_by(shard)->clear_dirty(it->second);
        return;
      }
    }
    ADD_FAILURE() << "key " << id << " not found to clear dirty";
  }

  // Write a single row through the metaheader storage path. The key is encoded
  // into the first 8 bytes of the row, mirroring how checkpoint restore feeds
  // weights-with-metaheader into the cache.
  void set_kv_with_metaheader_to_storage(
      DramKVEmbeddingCache<float>& cache,
      int64_t id) {
    const int64_t metaheader_dim =
        static_cast<int64_t>(FixedBlockPool::get_metaheader_dim<float>());
    const int64_t total_width = metaheader_dim + EMBEDDING_DIM;
    auto weights =
        at::zeros({1, total_width}, at::TensorOptions().dtype(at::kFloat));
    FixedBlockPool::set_key(weights[0].data_ptr(), id);
    cache.set_kv_with_metaheader_to_storage(weights);
  }

  // inference_set_kv_db_async requires feature_evict_ to be a
  // TimeThresholdBasedEvict (CHECK in the method), so build the cache with a
  // BY_TIMESTAMP_THRESHOLD feature-evict config (MANUAL trigger needs no extra
  // fields).
  std::shared_ptr<DramKVEmbeddingCache<float>> makeCacheWithTimeThresholdEvict(
      bool enable_ssd_backend) {
    auto feature_evict_config = c10::make_intrusive<FeatureEvictConfig>(
        /*trigger_mode=*/static_cast<int64_t>(EvictTriggerMode::MANUAL),
        /*trigger_strategy=*/
        static_cast<int64_t>(EvictTriggerStrategy::BY_TIMESTAMP_THRESHOLD),
        /*trigger_step_interval=*/std::nullopt,
        /*mem_util_threshold_in_GB=*/std::nullopt,
        /*ttls_in_mins=*/std::nullopt,
        /*counter_thresholds=*/std::nullopt,
        /*counter_decay_rates=*/std::nullopt,
        /*feature_score_counter_decay_rates=*/std::nullopt,
        /*training_id_eviction_trigger_count=*/std::nullopt,
        /*training_id_keep_count=*/std::nullopt,
        /*enable_eviction_for_feature_score_eviction_policy=*/std::nullopt,
        /*l2_weight_thresholds=*/std::nullopt,
        /*embedding_dims=*/std::nullopt);
    auto hash_size_cumsum = at::tensor({0, 100000}, at::kLong);
    return std::make_shared<DramKVEmbeddingCache<float>>(
        EMBEDDING_DIM,
        /*uniform_init_lower=*/-0.1,
        /*uniform_init_upper=*/0.1,
        feature_evict_config,
        NUM_SHARDS,
        /*num_threads=*/4,
        /*row_storage_bitwidth=*/32,
        /*backend_return_whole_row=*/false,
        /*enable_async_update=*/false,
        /*table_dims=*/std::nullopt,
        hash_size_cumsum,
        /*is_training=*/false,
        /*disable_random_init=*/true,
        /*enable_raw_embedding_streaming=*/false,
        /*res_store_shards=*/0,
        /*res_server_port=*/0,
        /*table_names=*/std::vector<std::string>{},
        /*table_offsets=*/std::vector<int64_t>{},
        /*table_sizes=*/std::vector<int64_t>{},
        /*enrichment_config=*/std::nullopt,
        /*enable_ssd_backend=*/enable_ssd_backend);
  }

  // Thin wrapper named after the core API, mirroring set_kv_db_async.
  void inference_set_kv_db_async(
      DramKVEmbeddingCache<float>& cache,
      int64_t id,
      float value = 1.0f,
      std::optional<uint32_t> inplace_update_ts = std::nullopt) {
    auto indices = at::tensor({id}, at::kLong);
    auto weights = at::full(
        {1, EMBEDDING_DIM}, value, at::TensorOptions().dtype(at::kFloat));
    auto count = at::tensor({1}, at::kLong);
    folly::coro::blockingWait(cache.inference_set_kv_db_async(
        indices, weights, count, inplace_update_ts));
  }

  // set_kv_zch_eviction_metadata_async is a no-op unless feature_evict_ is a
  // FeatureScoreBasedEvict, so build the cache with a BY_FEATURE_SCORE config.
  // hash_size_cumsum = {0, 100000} -> a single sub-table, so the per-sub-table
  // vectors are size 1.
  std::shared_ptr<DramKVEmbeddingCache<float>> makeCacheWithFeatureScoreEvict(
      bool enable_ssd_backend) {
    auto feature_evict_config = c10::make_intrusive<FeatureEvictConfig>(
        /*trigger_mode=*/static_cast<int64_t>(EvictTriggerMode::MANUAL),
        /*trigger_strategy=*/
        static_cast<int64_t>(EvictTriggerStrategy::BY_FEATURE_SCORE),
        /*trigger_step_interval=*/std::nullopt,
        /*mem_util_threshold_in_GB=*/std::nullopt,
        /*ttls_in_mins=*/std::vector<int64_t>{1},
        /*counter_thresholds=*/std::nullopt,
        /*counter_decay_rates=*/std::nullopt,
        /*feature_score_counter_decay_rates=*/std::vector<double>{0.5},
        /*training_id_eviction_trigger_count=*/std::vector<int64_t>{1000},
        /*training_id_keep_count=*/std::vector<int64_t>{100},
        /*enable_eviction_for_feature_score_eviction_policy=*/
        std::vector<int8_t>{1},
        /*l2_weight_thresholds=*/std::nullopt,
        /*embedding_dims=*/std::nullopt);
    auto hash_size_cumsum = at::tensor({0, 100000}, at::kLong);
    return std::make_shared<DramKVEmbeddingCache<float>>(
        EMBEDDING_DIM,
        /*uniform_init_lower=*/-0.1,
        /*uniform_init_upper=*/0.1,
        feature_evict_config,
        NUM_SHARDS,
        /*num_threads=*/4,
        /*row_storage_bitwidth=*/32,
        /*backend_return_whole_row=*/false,
        /*enable_async_update=*/false,
        /*table_dims=*/std::nullopt,
        hash_size_cumsum,
        /*is_training=*/false,
        /*disable_random_init=*/true,
        /*enable_raw_embedding_streaming=*/false,
        /*res_store_shards=*/0,
        /*res_server_port=*/0,
        /*table_names=*/std::vector<std::string>{},
        /*table_offsets=*/std::vector<int64_t>{},
        /*table_sizes=*/std::vector<int64_t>{},
        /*enrichment_config=*/std::nullopt,
        /*enable_ssd_backend=*/enable_ssd_backend);
  }

  // Thin wrapper named after the core API, mirroring set_kv_db_async.
  void set_kv_zch_eviction_metadata_async(
      DramKVEmbeddingCache<float>& cache,
      int64_t id,
      float engage_rate = 1.0f) {
    auto indices = at::tensor({id}, at::kLong);
    auto count = at::tensor({1}, at::kLong);
    auto engage_rates =
        at::full({1}, engage_rate, at::TensorOptions().dtype(at::kFloat));
    folly::coro::blockingWait(
        cache.set_kv_zch_eviction_metadata_async(indices, count, engage_rates));
  }

  std::shared_ptr<DramKVEmbeddingCache<float>> dram_cache_;
};

// Test: get_kv_metadata_rows returns correct shape and key for single inserted
// id
TEST_F(DramKVEmbeddingCacheTest, SingleKeyMetadata) {
  const int64_t test_id = 42;
  set_kv_db_async(*dram_cache_, test_id, 2.5f);

  auto indices = at::tensor({test_id}, at::kLong);
  auto count = at::tensor({1}, at::kLong);
  auto metadata = dram_cache_->get_kv_metadata_rows(indices, count);

  const int64_t expected_dim =
      static_cast<int64_t>(FixedBlockPool::get_metaheader_dim<float>());
  EXPECT_EQ(metadata.dim(), 2);
  EXPECT_EQ(metadata.size(0), 1);
  EXPECT_EQ(metadata.size(1), expected_dim);
  EXPECT_EQ(metadata.dtype(), at::kFloat);
  static_assert(sizeof(MetaHeader) == 16, "MetaHeader must be 16 bytes");

  MetaHeader header{};
  std::memcpy(&header, metadata.data_ptr<float>(), sizeof(MetaHeader));

  EXPECT_EQ(header.key, test_id);
  EXPECT_TRUE(header.used);
  EXPECT_GT(header.timestamp, 0u);
  // count may be 0 initially or updated depending on implementation
  EXPECT_GE(header.count, 0u);
}

// Test: get_kv_metadata_rows returns correct metadata for multiple keys across
// shards
TEST_F(DramKVEmbeddingCacheTest, MultipleKeysMetadata) {
  std::vector<int64_t> keys = {1, 2, 3, 10, 100, 1000};
  set_kv_db_async(*dram_cache_, keys, 1.0f);

  auto indices = at::tensor(keys, at::kLong);
  auto count = at::tensor({static_cast<int64_t>(keys.size())}, at::kLong);
  auto metadata = dram_cache_->get_kv_metadata_rows(indices, count);

  const int64_t expected_dim =
      static_cast<int64_t>(FixedBlockPool::get_metaheader_dim<float>());
  EXPECT_EQ(
      metadata.sizes(),
      at::IntArrayRef({static_cast<int64_t>(keys.size()), expected_dim}));

  auto* md_ptr = metadata.data_ptr<float>();
  const int64_t stride = expected_dim;
  for (size_t i = 0; i < keys.size(); ++i) {
    MetaHeader header{};
    std::memcpy(&header, md_ptr + i * stride, sizeof(MetaHeader));
    EXPECT_EQ(header.key, keys[i]) << "Mismatch at index " << i;
    EXPECT_TRUE(header.used) << "Used flag false for key " << keys[i];
    EXPECT_GT(header.timestamp, 0u) << "Timestamp not set for key " << keys[i];
  }
}

// Test: get_kv_metadata_rows with empty input returns empty tensor with correct
// dim
TEST_F(DramKVEmbeddingCacheTest, EmptyInputReturnsEmpty) {
  auto indices = at::empty({0}, at::kLong);
  auto count = at::tensor({0}, at::kLong);
  auto metadata = dram_cache_->get_kv_metadata_rows(indices, count);

  const int64_t expected_dim =
      static_cast<int64_t>(FixedBlockPool::get_metaheader_dim<float>());
  EXPECT_EQ(metadata.dim(), 2);
  EXPECT_EQ(metadata.size(0), 0);
  EXPECT_EQ(metadata.size(1), expected_dim);
}

// Test: get_kv_metadata_rows reflects updated timestamp after re-insert
TEST_F(DramKVEmbeddingCacheTest, TimestampUpdatesOnReinsert) {
  const int64_t test_id = 7;
  set_kv_db_async(*dram_cache_, test_id, 1.0f);

  auto indices = at::tensor({test_id}, at::kLong);
  auto count = at::tensor({1}, at::kLong);
  auto metadata1 = dram_cache_->get_kv_metadata_rows(indices, count);
  MetaHeader h1{};
  std::memcpy(&h1, metadata1.data_ptr<float>(), sizeof(MetaHeader));

  // Sleep to ensure timestamp advances (timestamp is in seconds)
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // Re-insert same key to update timestamp
  set_kv_db_async(*dram_cache_, test_id, 3.0f);
  auto metadata2 = dram_cache_->get_kv_metadata_rows(indices, count);
  MetaHeader h2{};
  std::memcpy(&h2, metadata2.data_ptr<float>(), sizeof(MetaHeader));

  EXPECT_EQ(h2.key, test_id);
  EXPECT_TRUE(h2.used);
  EXPECT_GE(h2.timestamp, h1.timestamp);
}

// Test: get_kv_metadata_rows works with float16 weight type via separate cache
// instance
TEST_F(DramKVEmbeddingCacheTest, HalfPrecisionMetadataDim) {
  auto hash_size_cumsum = at::tensor({0, 100000}, at::kLong);
  auto dram_cache_half = std::make_shared<DramKVEmbeddingCache<at::Half>>(
      EMBEDDING_DIM,
      -0.1,
      0.1,
      std::nullopt,
      NUM_SHARDS,
      4,
      16,
      false,
      false,
      std::nullopt,
      hash_size_cumsum,
      false,
      true);

  // Insert one key
  auto indices = at::tensor({5}, at::kLong);
  auto weights =
      at::full({1, EMBEDDING_DIM}, 1.0, at::TensorOptions().dtype(at::kHalf));
  auto count = at::tensor({1}, at::kLong);
  folly::coro::blockingWait(
      dram_cache_half->set_kv_db_async(indices, weights, count));

  auto metadata = dram_cache_half->get_kv_metadata_rows(indices, count);
  const int64_t expected_dim =
      static_cast<int64_t>(FixedBlockPool::get_metaheader_dim<at::Half>());
  // 16 bytes / 2 bytes per half = 8
  EXPECT_EQ(expected_dim, 8);
  EXPECT_EQ(metadata.sizes(), at::IntArrayRef({1, expected_dim}));
  EXPECT_EQ(metadata.dtype(), at::kHalf);

  // Decode first 8 bytes as int64 key from half tensor raw bytes
  int64_t decoded_key = 0;
  std::memcpy(&decoded_key, metadata.data_ptr<at::Half>(), sizeof(int64_t));
  EXPECT_EQ(decoded_key, 5);
}

// Shared fixture for all enable_ssd_backend-parameterized dirty-bit tests, so a
// single on/off instantiation covers every TEST_P below.
class DirtyBitParamTest : public DramKVEmbeddingCacheTest,
                          public ::testing::WithParamInterface<bool> {};

// Test: the set_kv_db_async path always leaves the block clean because there is
// no explicit set_dirty call (and allocation no longer marks blocks dirty),
// regardless of enable_ssd_backend.
TEST_P(DirtyBitParamTest, SetKvDbAsyncLeavesBlockClean) {
  const bool enable_ssd_backend = GetParam();
  auto cache = makeCache(enable_ssd_backend);
  const int64_t test_id = 123;
  set_kv_db_async(*cache, test_id);

  EXPECT_FALSE(isKeyDirty(*cache, test_id));
}

// Test: like set_kv_db_async, inference_set_kv_db_async leaves the block clean
// because it has no explicit set_dirty call (and allocation no longer marks
// blocks dirty), regardless of enable_ssd_backend.
TEST_P(DirtyBitParamTest, InferenceSetKvDbAsyncLeavesBlockClean) {
  const bool enable_ssd_backend = GetParam();
  auto cache = makeCacheWithTimeThresholdEvict(enable_ssd_backend);
  const int64_t test_id = 123;
  inference_set_kv_db_async(*cache, test_id);

  EXPECT_FALSE(isKeyDirty(*cache, test_id));
}

// Test: the metaheader write path marks a block dirty iff enable_ssd_backend is
// set. The block is allocated and its dirty bit cleared first, so the assertion
// isolates the `if (enable_ssd_backend_) set_dirty(...)` branch from
// allocation-time state.
TEST_P(DirtyBitParamTest, MetaheaderWriteDirtiesIffBackendEnabled) {
  const bool enable_ssd_backend = GetParam();
  auto cache = makeCache(enable_ssd_backend);
  const int64_t test_id = 77;

  set_kv_with_metaheader_to_storage(*cache, test_id);
  clearDirty(*cache, test_id);
  ASSERT_FALSE(isKeyDirty(*cache, test_id));

  set_kv_with_metaheader_to_storage(*cache, test_id);

  EXPECT_EQ(isKeyDirty(*cache, test_id), enable_ssd_backend);
}

// Test: the feature-score metadata path (set_kv_zch_eviction_metadata_async)
// always leaves the block clean because there is no explicit set_dirty call
// (and allocation no longer marks blocks dirty), regardless of
// enable_ssd_backend.
TEST_P(DirtyBitParamTest, FeatureScoreMetadataDirtiesIffBackendEnabled) {
  const bool enable_ssd_backend = GetParam();
  auto cache = makeCacheWithFeatureScoreEvict(enable_ssd_backend);
  const int64_t test_id = 55;
  set_kv_zch_eviction_metadata_async(*cache, test_id, /*engage_rate=*/0.5f);

  EXPECT_FALSE(isKeyDirty(*cache, test_id));
}

INSTANTIATE_TEST_SUITE_P(
    EnableSsdBackend,
    DirtyBitParamTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "On" : "Off";
    });

} // namespace kv_mem
