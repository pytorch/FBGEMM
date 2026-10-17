/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/fb/src/dram_kv_embedding_cache/oneflow_enrichment.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_embedding_cache.h"

#include <fmt/format.h>
#include <folly/system/EnvUtil.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <utility>
#include <vector>

namespace kv_mem {

namespace {

int64_t decodeInt64(const float* encoded) {
  uint64_t value = 0;
  for (int64_t index = 0; index < oneflow_enrichment::kDibitsPerInt64;
       ++index) {
    const auto dibit =
        static_cast<uint64_t>(std::lround(encoded[index] * 2.0f - 1.0f)) & 0x3;
    value |= dibit << (index * 2);
  }
  return static_cast<int64_t>(value);
}

} // namespace

TEST(OneFlowEnrichmentTest, EncodesVariableLengthIdListsWithPadding) {
  const std::vector<int64_t> hashedIds{101, 202};
  const std::vector<int64_t> unhashedIds{1001, 2002};
  const oneflow_enrichment::OpenTabBackend::IdListPayloadMap payloads{
      {1001, {11, 22}},
      {2002, {31, 32, 33, 34, 35, 36, 37, 38, 39}},
  };
  constexpr int64_t kMaxTags = 8;
  constexpr int64_t kCacheDim = kMaxTags * oneflow_enrichment::kDibitsPerInt64;

  const auto result = oneflow_enrichment::prepareInt64ListPayloadTensors<float>(
      hashedIds, unhashedIds, payloads, kCacheDim);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->indices.sizes(), at::IntArrayRef({2}));
  EXPECT_EQ(result->weights.sizes(), at::IntArrayRef({2, kCacheDim}));
  EXPECT_EQ(result->count.item<int64_t>(), 2);
  const auto weights = result->weights.contiguous();
  const float* values = weights.const_data_ptr<float>();
  EXPECT_EQ(decodeInt64(values), 11);
  EXPECT_EQ(decodeInt64(values + 32), 22);
  EXPECT_EQ(decodeInt64(values + 64), 0);
  EXPECT_EQ(decodeInt64(values + kCacheDim), 31);
  EXPECT_EQ(decodeInt64(values + kCacheDim + 7 * 32), 38);
}

TEST(OneFlowEnrichmentTest, EncodesIdListsIntoRowWithOptimizerStatePadding) {
  // Physical cache row = 256-dim embedding + pad4(rowwise optimizer state).
  const std::vector<int64_t> hashedIds{101};
  const std::vector<int64_t> unhashedIds{1001};
  const oneflow_enrichment::OpenTabBackend::IdListPayloadMap payloads{
      {1001, {11, 22, 33, 44, 55, 66, 77, 88, 99}},
  };
  constexpr int64_t kIdWidth = 8 * oneflow_enrichment::kDibitsPerInt64;
  constexpr int64_t kPhysicalRowDim = kIdWidth + 4;

  const auto result = oneflow_enrichment::prepareInt64ListPayloadTensors<float>(
      hashedIds, unhashedIds, payloads, kPhysicalRowDim);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->weights.sizes(), at::IntArrayRef({1, kIdWidth}));
  const auto weights = result->weights.contiguous();
  const float* values = weights.const_data_ptr<float>();
  EXPECT_EQ(decodeInt64(values), 11);
  EXPECT_EQ(decodeInt64(values + 7 * 32), 88);
}

TEST(OneFlowEnrichmentTest, RejectsRowNarrowerThanOneId) {
  const std::vector<int64_t> hashedIds{101};
  const std::vector<int64_t> unhashedIds{1001};
  const oneflow_enrichment::OpenTabBackend::IdListPayloadMap payloads{
      {1001, {11}},
  };
  EXPECT_FALSE(
      oneflow_enrichment::prepareInt64ListPayloadTensors<float>(
          hashedIds, unhashedIds, payloads, 16)
          .has_value());
}

TEST(OneFlowEnrichmentTest, WritesNegativeRowForEmptyListAndSkipsAbsentId) {
  // 1001 answered with no IDs (negative cache entry), 2002 never answered.
  const std::vector<int64_t> hashedIds{101, 202};
  const std::vector<int64_t> unhashedIds{1001, 2002};
  const oneflow_enrichment::OpenTabBackend::IdListPayloadMap payloads{
      {1001, {}},
  };
  constexpr int64_t kCacheDim = 8 * oneflow_enrichment::kDibitsPerInt64;

  const auto result = oneflow_enrichment::prepareInt64ListPayloadTensors<float>(
      hashedIds, unhashedIds, payloads, kCacheDim);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->indices.item<int64_t>(), 101);
  const auto weights = result->weights.contiguous();
  const float* values = weights.const_data_ptr<float>();
  // Every slot holds the no-payload marker, which encodes to non-zero weights
  // so the cache does not re-fetch the row (is_weights_all_zero is false).
  for (int64_t slot = 0; slot < 8; ++slot) {
    EXPECT_EQ(
        decodeInt64(values + slot * 32),
        oneflow_enrichment::kIdListNoPayloadId);
  }
  EXPECT_TRUE(std::all_of(values, values + kCacheDim, [](float value) {
    return value == 2.0f;
  }));
}

TEST(OneFlowEnrichmentTest, DropsReservedIdsFromPayloadAndPadsWithZero) {
  const std::vector<int64_t> hashedIds{101};
  const std::vector<int64_t> unhashedIds{1001};
  const oneflow_enrichment::OpenTabBackend::IdListPayloadMap payloads{
      {1001,
       {oneflow_enrichment::kIdListNoPayloadId,
        11,
        oneflow_enrichment::kIdListPaddingId,
        22}},
  };
  constexpr int64_t kCacheDim = 3 * oneflow_enrichment::kDibitsPerInt64;

  const auto result = oneflow_enrichment::prepareInt64ListPayloadTensors<float>(
      hashedIds, unhashedIds, payloads, kCacheDim);

  ASSERT_TRUE(result.has_value());
  const auto weights = result->weights.contiguous();
  const float* values = weights.const_data_ptr<float>();
  const std::vector<int64_t> expected{
      11, 22, oneflow_enrichment::kIdListPaddingId};
  std::vector<int64_t> decoded;
  decoded.reserve(expected.size());
  for (int64_t slot = 0; slot < 3; ++slot) {
    decoded.push_back(decodeInt64(values + slot * 32));
  }
  EXPECT_EQ(decoded, expected);
}

TEST(OneFlowEnrichmentTest, WritesNegativeRowForAllReservedPayload) {
  // A payload holding only reserved values carries no real IDs: it must become
  // a negative-cache row, not an all-zero row that is re-fetched every step.
  const std::vector<int64_t> hashedIds{101};
  const std::vector<int64_t> unhashedIds{1001};
  const oneflow_enrichment::OpenTabBackend::IdListPayloadMap payloads{
      {1001,
       {oneflow_enrichment::kIdListPaddingId,
        oneflow_enrichment::kIdListNoPayloadId}},
  };
  constexpr int64_t kCacheDim = 3 * oneflow_enrichment::kDibitsPerInt64;

  const auto result = oneflow_enrichment::prepareInt64ListPayloadTensors<float>(
      hashedIds, unhashedIds, payloads, kCacheDim);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->indices.item<int64_t>(), 101);
  const auto weights = result->weights.contiguous();
  const float* values = weights.const_data_ptr<float>();
  const std::vector<int64_t> expected(
      3, oneflow_enrichment::kIdListNoPayloadId);
  std::vector<int64_t> decoded;
  decoded.reserve(expected.size());
  for (int64_t slot = 0; slot < 3; ++slot) {
    decoded.push_back(decodeInt64(values + slot * 32));
  }
  EXPECT_EQ(decoded, expected);
}

TEST(OneFlowEnrichmentTest, CountsOnlyNonEmptyIdListPayloads) {
  const oneflow_enrichment::OpenTabBackend::IdListPayloadMap payloads{
      {1001, {11, 22}},
      {2002, {}},
      {3003, {33}},
  };
  EXPECT_EQ(oneflow_enrichment::countNonEmptyPayloads(payloads), 2);
}

struct MetaHeader {
  int64_t key;
  uint32_t timestamp;
  uint32_t count : 31;
  bool used : 1;
};

template <typename WeightType>
void expectMissingWholeRowZeroed(at::ScalarType scalarType) {
  constexpr int64_t kEmbeddingDim = 16;
  const auto hashSizeCumsum = at::tensor({0, 100000}, at::kLong);
  DramKVEmbeddingCache<WeightType> cache(
      kEmbeddingDim,
      /*uniform_init_lower=*/-0.1,
      /*uniform_init_upper=*/0.1,
      /*feature_evict_config=*/std::nullopt,
      /*num_shards=*/4,
      /*num_threads=*/4,
      /*row_storage_bitwidth=*/sizeof(WeightType) * 8,
      /*backend_return_whole_row=*/true,
      /*enable_async_update=*/false,
      /*table_dims=*/std::nullopt,
      hashSizeCumsum,
      /*is_training=*/false,
      /*disable_random_init=*/true);
  auto output = at::ones({1, kEmbeddingDim}, scalarType);

  cache.get_range_from_snapshot(
      output, /*start=*/0, /*length=*/1, /*snapshot_handle=*/nullptr);

  EXPECT_TRUE(at::equal(output, at::zeros_like(output))) << output;
}

TEST(DramKVEmbeddingCacheMissingRowTest, ZeroesCompleteFloatRow) {
  expectMissingWholeRowZeroed<float>(at::kFloat);
}

TEST(DramKVEmbeddingCacheMissingRowTest, ZeroesCompleteHalfRow) {
  expectMissingWholeRowZeroed<at::Half>(at::kHalf);
}

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

  // Build a standalone float cache with focused overrides for each test.
  std::shared_ptr<DramKVEmbeddingCache<float>> makeCache(
      bool enable_ssd_backend,
      int num_shards = NUM_SHARDS,
      bool is_training = false,
      bool disable_random_init = true,
      double uniform_init_lower = -0.1,
      double uniform_init_upper = 0.1) {
    auto hash_size_cumsum = at::tensor({0, 100000}, at::kLong);
    return std::make_shared<DramKVEmbeddingCache<float>>(
        EMBEDDING_DIM,
        uniform_init_lower,
        uniform_init_upper,
        /*feature_evict_config=*/std::nullopt,
        num_shards,
        /*num_threads=*/4,
        /*row_storage_bitwidth=*/32,
        /*backend_return_whole_row=*/false,
        /*enable_async_update=*/false,
        /*table_dims=*/std::nullopt,
        hash_size_cumsum,
        is_training,
        disable_random_init,
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

  // pause/resume are no-ops unless is_training, and a MANUAL trigger lets the
  // test decide when a round starts. A triggered round starts paused.
  std::shared_ptr<DramKVEmbeddingCache<float>>
  makeTrainingCacheWithManualEvict() {
    auto feature_evict_config = c10::make_intrusive<FeatureEvictConfig>(
        /*trigger_mode=*/static_cast<int64_t>(EvictTriggerMode::MANUAL),
        /*trigger_strategy=*/
        static_cast<int64_t>(EvictTriggerStrategy::BY_TIMESTAMP),
        /*trigger_step_interval=*/std::nullopt,
        /*mem_util_threshold_in_GB=*/std::nullopt,
        /*ttls_in_mins=*/std::vector<int64_t>{60},
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
        /*is_training=*/true,
        /*disable_random_init=*/true);
  }

  // Thin wrapper over EmbeddingKVDB::get, the host callback behind get_cuda.
  void get(
      DramKVEmbeddingCache<float>& cache,
      const std::vector<int64_t>& ids) {
    auto num = static_cast<int64_t>(ids.size());
    auto weights =
        at::zeros({num, EMBEDDING_DIM}, at::TensorOptions().dtype(at::kFloat));
    cache.get(
        at::tensor(ids, at::kLong), weights, at::tensor({num}, at::kLong));
  }

  // Returns false if the eviction round is still ongoing after `timeout`.
  bool waitForEvictionIdle(
      DramKVEmbeddingCache<float>& cache,
      std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (cache.is_evicting()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
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

// A key can be evicted after input sharding but before its shard lock is
// acquired. Missing rows retain the zero-initialized metadata instead of
// aborting the process.
TEST_F(DramKVEmbeddingCacheTest, MissingKeyMetadataReturnsZeroRow) {
  constexpr int64_t present_id = 42;
  constexpr int64_t missing_id = 99;
  set_kv_db_async(*dram_cache_, present_id, 2.5f);

  auto indices = at::tensor({present_id, missing_id}, at::kLong);
  auto count = at::tensor({2}, at::kLong);
  auto metadata = dram_cache_->get_kv_metadata_rows(indices, count);

  const int64_t metadata_dim =
      static_cast<int64_t>(FixedBlockPool::get_metaheader_dim<float>());
  MetaHeader present_header{};
  std::memcpy(
      &present_header, metadata[0].data_ptr<float>(), sizeof(MetaHeader));
  EXPECT_EQ(present_header.key, present_id);
  EXPECT_TRUE(present_header.used);
  EXPECT_TRUE(at::equal(metadata[1], at::zeros({metadata_dim}, at::kFloat)));
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

// Test: by default get() leaves a triggered eviction round paused; only the
// backward write-back resumes it.
TEST_F(DramKVEmbeddingCacheTest, GetLeavesTriggeredEvictionPausedByDefault) {
  auto cache = makeTrainingCacheWithManualEvict();
  const std::vector<int64_t> ids{1, 2, 3, 4, 5, 6, 7, 8};
  set_kv_db_async(*cache, ids);
  cache->trigger_feature_evict();
  ASSERT_TRUE(cache->is_evicting());

  get(*cache, ids);

  EXPECT_FALSE(waitForEvictionIdle(*cache, std::chrono::milliseconds(500)));
  // FeatureEvict destruction blocks on a round that is still paused.
  cache->wait_until_eviction_done();
}

// Test: with resume-after-get enabled, get() resumes a triggered eviction
// round, which then runs to completion.
TEST_F(DramKVEmbeddingCacheTest, GetResumesTriggeredEvictionWhenEnabled) {
  auto cache = makeTrainingCacheWithManualEvict();
  cache->set_resume_eviction_after_get(true);
  const std::vector<int64_t> ids{1, 2, 3, 4, 5, 6, 7, 8};
  set_kv_db_async(*cache, ids);
  cache->trigger_feature_evict();
  ASSERT_TRUE(cache->is_evicting());

  get(*cache, ids);

  EXPECT_TRUE(waitForEvictionIdle(*cache, std::chrono::seconds(10)));
  cache->wait_until_eviction_done();
}

// Test: a get() with zero lookups returns early but still resumes eviction when
// resume-after-get is enabled.
TEST_F(DramKVEmbeddingCacheTest, EmptyGetResumesTriggeredEvictionWhenEnabled) {
  auto cache = makeTrainingCacheWithManualEvict();
  cache->set_resume_eviction_after_get(true);
  set_kv_db_async(*cache, std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8});
  cache->trigger_feature_evict();
  ASSERT_TRUE(cache->is_evicting());

  get(*cache, std::vector<int64_t>{});

  EXPECT_TRUE(waitForEvictionIdle(*cache, std::chrono::seconds(10)));
  cache->wait_until_eviction_done();
}

TEST_F(DramKVEmbeddingCacheTest, PipelinedReadMatchesLegacy) {
  folly::test::EnvVarSaver env_saver;
  unsetenv("KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE");
  unsetenv("KVZCH_SKIP_DRAM_PERF_TIMING");
  auto legacy_cache = makeCache(/*enable_ssd_backend=*/false, /*num_shards=*/1);

  setenv("KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE", "8", /*overwrite=*/1);
  setenv("KVZCH_SKIP_DRAM_PERF_TIMING", "1", /*overwrite=*/1);
  auto pipelined_cache =
      makeCache(/*enable_ssd_backend=*/false, /*num_shards=*/1);

  std::vector<int64_t> stored_ids;
  for (int64_t id = 1; id <= 20; ++id) {
    stored_ids.push_back(id);
  }
  set_kv_db_async(*legacy_cache, stored_ids, /*value=*/2.5f);
  set_kv_db_async(*pipelined_cache, stored_ids, /*value=*/2.5f);

  std::vector<int64_t> requested_ids = stored_ids;
  requested_ids.push_back(3);
  requested_ids.push_back(100001);
  requested_ids.push_back(-1);
  requested_ids.push_back(25);
  auto indices = at::tensor(requested_ids, at::kLong);
  auto count = at::tensor({23}, at::kLong);
  auto legacy_weights = at::full(
      {24, EMBEDDING_DIM}, -7.0f, at::TensorOptions().dtype(at::kFloat));
  auto pipelined_weights = legacy_weights.clone();

  folly::coro::blockingWait(
      legacy_cache->get_kv_db_async(indices, legacy_weights, count));
  folly::coro::blockingWait(
      pipelined_cache->get_kv_db_async(indices, pipelined_weights, count));

  EXPECT_TRUE(at::equal(legacy_weights, pipelined_weights));
  auto expected = at::full(
      {24, EMBEDDING_DIM}, -7.0f, at::TensorOptions().dtype(at::kFloat));
  expected.slice(/*dim=*/0, /*start=*/0, /*end=*/21).fill_(2.5f);
  expected[21].zero_();
  EXPECT_TRUE(at::equal(expected, pipelined_weights));
}

TEST_F(DramKVEmbeddingCacheTest, FoundMaskLeavesAbsentRowsUnwritten) {
  auto cache = makeCache(/*enable_ssd_backend=*/false, /*num_shards=*/1);
  set_kv_db_async(*cache, std::vector<int64_t>{1, 2}, /*value=*/2.5f);

  auto indices = at::tensor({1, 100001, -1, 2, 25}, at::kLong);
  auto count = at::tensor({4}, at::kLong);
  auto weights = at::full(
      {indices.numel(), EMBEDDING_DIM},
      -7.0f,
      at::TensorOptions().dtype(at::kFloat));
  auto found_mask =
      at::full({indices.numel()}, 7, at::TensorOptions().dtype(at::kByte));

  folly::coro::blockingWait(cache->get_kv_db_async_with_found_mask(
      indices, weights, count, found_mask));

  auto expected_weights = at::full_like(weights, -7.0f);
  expected_weights[0].fill_(2.5f);
  expected_weights[3].fill_(2.5f);
  EXPECT_TRUE(at::equal(expected_weights, weights));
  EXPECT_TRUE(
      at::equal(
          at::tensor({1, 0, 7, 1, 7}, at::TensorOptions().dtype(at::kByte)),
          found_mask));
}

TEST_F(
    DramKVEmbeddingCacheTest,
    FoundMaskWorksWithParallelShardInputAndReadSplits) {
  folly::test::EnvVarSaver env_saver;
  setenv("KVZCH_PARALLEL_DRAM_SHARD_INPUT", "1", /*overwrite=*/1);
  setenv("KVZCH_DRAM_READ_SPLITS_PER_SHARD", "2", /*overwrite=*/1);
  setenv("KVZCH_DRAM_READ_SPLIT_ZERO_COPY", "1", /*overwrite=*/1);
  setenv("KVZCH_SKIP_DRAM_PERF_TIMING", "1", /*overwrite=*/1);
  auto cache = makeCache(/*enable_ssd_backend=*/false, NUM_SHARDS);

  constexpr int64_t kNumRows = 16384;
  std::vector<int64_t> stored_ids;
  stored_ids.reserve(kNumRows / 2);
  for (int64_t id = 0; id < kNumRows; id += 2) {
    stored_ids.push_back(id);
  }
  set_kv_db_async(*cache, stored_ids, /*value=*/2.5f);

  auto indices = at::arange(kNumRows, at::TensorOptions().dtype(at::kLong));
  auto count = at::tensor({kNumRows}, at::kLong);
  auto weights = at::full(
      {kNumRows, EMBEDDING_DIM}, -7.0f, at::TensorOptions().dtype(at::kFloat));
  auto found_mask =
      at::full({kNumRows}, 7, at::TensorOptions().dtype(at::kByte));

  folly::coro::blockingWait(cache->get_kv_db_async_with_found_mask(
      indices, weights, count, found_mask));

  const auto hit_indices =
      at::arange(0, kNumRows, 2, at::TensorOptions().dtype(at::kLong));
  auto expected_weights = at::full_like(weights, -7.0f);
  expected_weights.index_fill_(0, hit_indices, 2.5f);
  auto expected_found_mask = at::zeros_like(found_mask);
  expected_found_mask.index_fill_(0, hit_indices, 1);
  EXPECT_TRUE(at::equal(expected_weights, weights));
  EXPECT_TRUE(at::equal(expected_found_mask, found_mask));
}

TEST_F(DramKVEmbeddingCacheTest, ParallelReadSplitsMatchLegacy) {
  folly::test::EnvVarSaver env_saver;
  unsetenv("KVZCH_DRAM_READ_SPLITS_PER_SHARD");
  unsetenv("KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE");
  unsetenv("KVZCH_SKIP_DRAM_PERF_TIMING");
  unsetenv("KVZCH_SKIP_DISABLED_INITIALIZERS");
  unsetenv("KVZCH_DRAM_READ_SPLIT_ZERO_COPY");
  auto legacy_cache = makeCache(/*enable_ssd_backend=*/false, /*num_shards=*/1);

  setenv("KVZCH_DRAM_READ_SPLITS_PER_SHARD", "4", /*overwrite=*/1);
  setenv("KVZCH_SKIP_DRAM_PERF_TIMING", "1", /*overwrite=*/1);
  setenv("KVZCH_SKIP_DISABLED_INITIALIZERS", "1", /*overwrite=*/1);
  auto copied_split_cache =
      makeCache(/*enable_ssd_backend=*/false, /*num_shards=*/1);
  setenv("KVZCH_DRAM_READ_SPLIT_ZERO_COPY", "1", /*overwrite=*/1);
  setenv("KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE", "8", /*overwrite=*/1);
  auto zero_copy_split_cache =
      makeCache(/*enable_ssd_backend=*/false, /*num_shards=*/1);

  constexpr int64_t stored_count = 8192;
  std::vector<int64_t> stored_ids;
  stored_ids.reserve(stored_count);
  for (int64_t id = 1; id <= stored_count; ++id) {
    stored_ids.push_back(id);
  }
  set_kv_db_async(*legacy_cache, stored_ids, /*value=*/2.5f);
  set_kv_db_async(*copied_split_cache, stored_ids, /*value=*/2.5f);
  set_kv_db_async(*zero_copy_split_cache, stored_ids, /*value=*/2.5f);

  std::vector<int64_t> requested_ids = stored_ids;
  requested_ids.push_back(7);
  requested_ids.push_back(100001);
  requested_ids.push_back(-1);
  requested_ids.push_back(9);
  const auto active_count = static_cast<int64_t>(requested_ids.size()) - 1;

  for (const auto index_type : {at::kInt, at::kLong}) {
    auto indices =
        at::tensor(requested_ids, at::TensorOptions().dtype(index_type));
    auto count =
        at::tensor({active_count}, at::TensorOptions().dtype(index_type));
    auto legacy_weights = at::full(
        {static_cast<int64_t>(requested_ids.size()), EMBEDDING_DIM},
        -7.0f,
        at::TensorOptions().dtype(at::kFloat));
    auto copied_split_weights = legacy_weights.clone();
    auto zero_copy_split_weights = legacy_weights.clone();

    folly::coro::blockingWait(
        legacy_cache->get_kv_db_async(indices, legacy_weights, count));
    folly::coro::blockingWait(copied_split_cache->get_kv_db_async(
        indices, copied_split_weights, count));
    folly::coro::blockingWait(zero_copy_split_cache->get_kv_db_async(
        indices, zero_copy_split_weights, count));

    EXPECT_TRUE(at::equal(legacy_weights, copied_split_weights));
    EXPECT_TRUE(at::equal(legacy_weights, zero_copy_split_weights));
    auto expected = at::full_like(zero_copy_split_weights, -7.0f);
    expected.slice(/*dim=*/0, /*start=*/0, /*end=*/stored_count + 1)
        .fill_(2.5f);
    expected[stored_count + 1].zero_();
    EXPECT_TRUE(at::equal(expected, zero_copy_split_weights));
  }
}

TEST_F(
    DramKVEmbeddingCacheTest,
    SkipDisabledInitializersKeepsEnabledRandomInitialization) {
  folly::test::EnvVarSaver env_saver;
  setenv("KVZCH_SKIP_DISABLED_INITIALIZERS", "1", /*overwrite=*/1);
  auto cache = makeCache(
      /*enable_ssd_backend=*/false,
      /*num_shards=*/1,
      /*is_training=*/true,
      /*disable_random_init=*/false,
      /*uniform_init_lower=*/2.0,
      /*uniform_init_upper=*/3.0);

  auto indices = at::tensor({42}, at::kLong);
  auto weights = at::full(
      {1, EMBEDDING_DIM}, -7.0f, at::TensorOptions().dtype(at::kFloat));
  auto count = at::tensor({1}, at::kLong);
  folly::coro::blockingWait(cache->get_kv_db_async(indices, weights, count));

  EXPECT_TRUE(at::all(weights >= 2.0).item<bool>());
  EXPECT_TRUE(at::all(weights < 3.0).item<bool>());
}

TEST_F(DramKVEmbeddingCacheTest, ParallelShardInputMatchesLegacy) {
  folly::test::EnvVarSaver env_saver;
  constexpr int64_t active_count = 16384;
  constexpr int64_t tail_size = 8;
  constexpr int64_t num_keys = 1021;

  std::vector<int64_t> ids(active_count + tail_size);
  std::vector<float> last_value(num_keys + 1, -1.0f);
  for (int64_t i = 0; i < active_count; ++i) {
    ids[i] = i % 113 == 0 ? -1 : (i * 37) % num_keys + 1;
    if (ids[i] >= 0) {
      last_value[ids[i]] = static_cast<float>(i);
    }
  }
  for (int64_t i = active_count; std::cmp_less(i, ids.size()); ++i) {
    ids[i] = 50000 + i;
  }

  std::vector<int64_t> query_ids;
  query_ids.reserve(num_keys + tail_size + 3);
  for (int64_t id = 1; id <= num_keys; ++id) {
    query_ids.push_back(id);
  }
  for (int64_t i = 0; i < tail_size; ++i) {
    query_ids.push_back(50000 + active_count + i);
  }
  query_ids.push_back(-1);
  query_ids.push_back(99999);
  query_ids.push_back(7);

  for (const auto index_type : {at::kInt, at::kLong}) {
    unsetenv("KVZCH_PARALLEL_DRAM_SHARD_INPUT");
    auto legacy_cache = makeCache(/*enable_ssd_backend=*/false);
    setenv("KVZCH_PARALLEL_DRAM_SHARD_INPUT", "1", /*overwrite=*/1);
    auto parallel_cache = makeCache(/*enable_ssd_backend=*/false);

    auto indices = at::tensor(ids, at::TensorOptions().dtype(index_type));
    auto weights =
        at::arange(
            static_cast<double>(ids.size()),
            at::TensorOptions().dtype(at::kFloat))
            .view({static_cast<int64_t>(ids.size()), 1})
            .expand({static_cast<int64_t>(ids.size()), EMBEDDING_DIM})
            .contiguous();
    auto count =
        at::tensor({active_count}, at::TensorOptions().dtype(index_type));
    folly::coro::blockingWait(
        legacy_cache->set_kv_db_async(indices, weights, count));
    folly::coro::blockingWait(
        parallel_cache->set_kv_db_async(indices, weights, count));

    auto query = at::tensor(query_ids, at::TensorOptions().dtype(index_type));
    auto query_count = at::tensor(
        {static_cast<int64_t>(query_ids.size() - 1)},
        at::TensorOptions().dtype(index_type));
    auto legacy_weights = at::full(
        {static_cast<int64_t>(query_ids.size()), EMBEDDING_DIM},
        -7.0f,
        at::TensorOptions().dtype(at::kFloat));
    auto parallel_weights = legacy_weights.clone();
    folly::coro::blockingWait(
        legacy_cache->get_kv_db_async(query, legacy_weights, query_count));
    folly::coro::blockingWait(
        parallel_cache->get_kv_db_async(query, parallel_weights, query_count));

    EXPECT_TRUE(at::equal(legacy_weights, parallel_weights));
    auto expected = at::full_like(legacy_weights, -7.0f);
    for (int64_t row = 0; std::cmp_less(row, query_ids.size() - 1); ++row) {
      const auto id = query_ids[row];
      if (id < 0) {
        continue;
      }
      if (id <= num_keys) {
        expected[row].fill_(last_value[id]);
      } else {
        expected[row].zero_();
      }
    }
    EXPECT_TRUE(at::equal(expected, parallel_weights));
  }
}

TEST_F(DramKVEmbeddingCacheTest, ExecutorThreadCapRequiresTargetFastPath) {
  folly::test::EnvVarSaver env_saver;
  setenv("KVZCH_DRAM_EXECUTOR_THREAD_CAP", "1", /*overwrite=*/1);
  setenv("KVZCH_PARALLEL_DRAM_SHARD_INPUT", "1", /*overwrite=*/1);
  setenv("KVZCH_SKIP_DRAM_PERF_TIMING", "1", /*overwrite=*/1);
  auto single_thread_cache = makeCache(/*enable_ssd_backend=*/false);
  EXPECT_EQ(single_thread_cache->get_executor_num_threads(), 1);

  constexpr int64_t kNumRows = 8192;
  auto indices = at::arange(kNumRows, at::TensorOptions().dtype(at::kLong));
  auto weights = at::full(
      {kNumRows, EMBEDDING_DIM}, 2.5f, at::TensorOptions().dtype(at::kFloat));
  auto count = at::tensor({kNumRows}, at::kLong);
  folly::coro::blockingWait(
      single_thread_cache->set_kv_db_async(indices, weights, count));
  auto output = at::zeros_like(weights);
  folly::coro::blockingWait(
      single_thread_cache->get_kv_db_async(indices, output, count));
  EXPECT_TRUE(at::equal(weights, output));

  setenv("KVZCH_DRAM_EXECUTOR_THREAD_CAP", "2", /*overwrite=*/1);
  auto capped_cache = makeCache(/*enable_ssd_backend=*/false);
  EXPECT_EQ(capped_cache->get_executor_num_threads(), 2);

  folly::coro::blockingWait(
      capped_cache->set_kv_db_async(indices, weights, count));
  output.zero_();
  folly::coro::blockingWait(
      capped_cache->get_kv_db_async(indices, output, count));
  EXPECT_TRUE(at::equal(weights, output));

  unsetenv("KVZCH_SKIP_DRAM_PERF_TIMING");
  auto refused_cache = makeCache(/*enable_ssd_backend=*/false);
  EXPECT_GE(refused_cache->get_executor_num_threads(), 4);

  setenv("KVZCH_SKIP_DRAM_PERF_TIMING", "1", /*overwrite=*/1);
  unsetenv("KVZCH_PARALLEL_DRAM_SHARD_INPUT");
  refused_cache = makeCache(/*enable_ssd_backend=*/false);
  EXPECT_GE(refused_cache->get_executor_num_threads(), 4);

  setenv("KVZCH_PARALLEL_DRAM_SHARD_INPUT", "1", /*overwrite=*/1);
  refused_cache = makeCache(
      /*enable_ssd_backend=*/false,
      NUM_SHARDS,
      /*is_training=*/false,
      /*disable_random_init=*/false);
  EXPECT_GE(refused_cache->get_executor_num_threads(), 4);
}

TEST_F(DramKVEmbeddingCacheTest, SegmentedShardReadsMatchMergedInput) {
  folly::test::EnvVarSaver env_saver;
  setenv("KVZCH_PARALLEL_DRAM_SHARD_INPUT", "1", /*overwrite=*/1);
  setenv("KVZCH_SKIP_DRAM_PERF_TIMING", "1", /*overwrite=*/1);
  setenv("KVZCH_DRAM_LOOKUP_PREFETCH_DISTANCE", "8", /*overwrite=*/1);
  unsetenv("KVZCH_DRAM_READ_SPLITS_PER_SHARD");
  unsetenv("KVZCH_DRAM_EXECUTOR_THREAD_CAP");

  constexpr int64_t kStoredRows = 8192;
  constexpr int64_t kActiveRows = 16384;
  constexpr int64_t kTailRows = 8;
  std::vector<int64_t> stored_ids;
  stored_ids.reserve(kStoredRows);
  for (int64_t id = 1; id <= kStoredRows; ++id) {
    stored_ids.push_back(id);
  }

  for (const auto index_type : {at::kInt, at::kLong}) {
    unsetenv("KVZCH_DRAM_SEGMENTED_SHARD_READS");
    auto merged_cache = makeCache(/*enable_ssd_backend=*/false);
    setenv("KVZCH_DRAM_SEGMENTED_SHARD_READS", "1", /*overwrite=*/1);
    auto segmented_cache = makeCache(/*enable_ssd_backend=*/false);
    EXPECT_TRUE(segmented_cache->get_segmented_shard_reads_enabled());
    set_kv_db_async(*merged_cache, stored_ids, /*value=*/2.5f);
    set_kv_db_async(*segmented_cache, stored_ids, /*value=*/2.5f);

    std::vector<int64_t> query_ids(kActiveRows + kTailRows);
    for (int64_t i = 0; i < kActiveRows; ++i) {
      query_ids[i] = i % 113 == 0 ? -1 : i % 12000 + 1;
    }
    for (size_t i = kActiveRows; i < query_ids.size(); ++i) {
      query_ids[i] = 50000 + static_cast<int64_t>(i);
    }
    auto query = at::tensor(query_ids, at::TensorOptions().dtype(index_type));
    auto count =
        at::tensor({kActiveRows}, at::TensorOptions().dtype(index_type));
    auto merged_weights = at::full(
        {static_cast<int64_t>(query_ids.size()), EMBEDDING_DIM},
        -7.0f,
        at::TensorOptions().dtype(at::kFloat));
    auto segmented_weights = merged_weights.clone();
    auto merged_found_mask =
        at::full({query.numel()}, 7, at::TensorOptions().dtype(at::kByte));
    auto segmented_found_mask = merged_found_mask.clone();

    folly::coro::blockingWait(merged_cache->get_kv_db_async_with_found_mask(
        query, merged_weights, count, merged_found_mask));
    folly::coro::blockingWait(segmented_cache->get_kv_db_async_with_found_mask(
        query, segmented_weights, count, segmented_found_mask));

    auto expected = at::full_like(merged_weights, -7.0f);
    auto expected_found_mask = at::full_like(merged_found_mask, 7);
    for (int64_t row = 0; row < kActiveRows; ++row) {
      const auto id = query_ids[row];
      if (id < 0) {
        continue;
      }
      if (id <= kStoredRows) {
        expected[row].fill_(2.5f);
        expected_found_mask[row] = 1;
      } else {
        expected_found_mask[row] = 0;
      }
    }
    EXPECT_TRUE(at::equal(merged_weights, segmented_weights));
    EXPECT_TRUE(at::equal(expected, segmented_weights));
    EXPECT_TRUE(at::equal(merged_found_mask, segmented_found_mask));
    EXPECT_TRUE(at::equal(expected_found_mask, segmented_found_mask));

    auto negative_query =
        at::full({kActiveRows}, -1, at::TensorOptions().dtype(index_type));
    auto negative_count =
        at::tensor({kActiveRows}, at::TensorOptions().dtype(index_type));
    auto negative_weights = at::full(
        {kActiveRows, EMBEDDING_DIM},
        -7.0f,
        at::TensorOptions().dtype(at::kFloat));
    auto negative_found_mask =
        at::full({kActiveRows}, 7, at::TensorOptions().dtype(at::kByte));
    folly::coro::blockingWait(segmented_cache->get_kv_db_async_with_found_mask(
        negative_query, negative_weights, negative_count, negative_found_mask));
    EXPECT_TRUE(at::all(negative_weights == -7.0f).item<bool>());
    EXPECT_TRUE(at::all(negative_found_mask == 7).item<bool>());
  }
}

TEST_F(DramKVEmbeddingCacheTest, SegmentedShardReadsFallBackForSingleShard) {
  folly::test::EnvVarSaver env_saver;
  setenv("KVZCH_PARALLEL_DRAM_SHARD_INPUT", "1", /*overwrite=*/1);
  setenv("KVZCH_SKIP_DRAM_PERF_TIMING", "1", /*overwrite=*/1);
  unsetenv("KVZCH_DRAM_READ_SPLITS_PER_SHARD");

  constexpr int64_t kNumRows = 8192;
  auto indices = at::arange(kNumRows, at::TensorOptions().dtype(at::kLong));
  auto count = at::tensor({kNumRows}, at::kLong);

  unsetenv("KVZCH_DRAM_SEGMENTED_SHARD_READS");
  auto legacy_cache = makeCache(/*enable_ssd_backend=*/false, /*num_shards=*/1);
  setenv("KVZCH_DRAM_SEGMENTED_SHARD_READS", "1", /*overwrite=*/1);
  auto fallback_cache =
      makeCache(/*enable_ssd_backend=*/false, /*num_shards=*/1);
  EXPECT_FALSE(fallback_cache->get_segmented_shard_reads_enabled());
  set_kv_db_async(*legacy_cache, std::vector<int64_t>{1, 2}, /*value=*/2.5f);
  set_kv_db_async(*fallback_cache, std::vector<int64_t>{1, 2}, /*value=*/2.5f);

  auto legacy_weights = at::full(
      {kNumRows, EMBEDDING_DIM}, -7.0f, at::TensorOptions().dtype(at::kFloat));
  auto fallback_weights = legacy_weights.clone();
  auto legacy_found_mask =
      at::full({kNumRows}, 7, at::TensorOptions().dtype(at::kByte));
  auto fallback_found_mask = legacy_found_mask.clone();

  folly::coro::blockingWait(legacy_cache->get_kv_db_async_with_found_mask(
      indices, legacy_weights, count, legacy_found_mask));
  folly::coro::blockingWait(fallback_cache->get_kv_db_async_with_found_mask(
      indices, fallback_weights, count, fallback_found_mask));

  EXPECT_TRUE(at::equal(legacy_weights, fallback_weights));
  EXPECT_TRUE(at::equal(legacy_found_mask, fallback_found_mask));
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
