/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_inference_embedding.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <vector>

namespace kv_mem {

class KVEmbeddingInferenceTest : public ::testing::Test {
 protected:
  static constexpr int EMBEDDING_DIM = 128;
  static constexpr int NUM_SHARDS = 8;

  void SetUp() override {
    FLAGS_logtostderr = true;
    FLAGS_minloglevel = 0;
    FLAGS_v = 1;

    auto feature_evict_config = c10::make_intrusive<FeatureEvictConfig>(
        3,
        4,
        std::nullopt,
        std::nullopt,
        std::vector<int64_t>{1},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::vector<int64_t>{EMBEDDING_DIM},
        std::nullopt,
        std::nullopt,
        0,
        0,
        0);

    auto hash_size_cumsum = at::tensor({0, 100000}, at::kLong);

    backend_ = std::make_unique<DramKVInferenceEmbedding<float>>(
        EMBEDDING_DIM,
        -0.1,
        0.1,
        feature_evict_config,
        NUM_SHARDS,
        32,
        32,
        false,
        std::nullopt,
        hash_size_cumsum,
        false);
  }

  void TearDown() override {
    backend_.reset();
  }

  static std::vector<float> generateEmbedding(int64_t embedding_id) {
    std::vector<float> embedding(EMBEDDING_DIM);

    // Use both embedding_id and current time as seed for randomness
    auto now = std::chrono::system_clock::now();
    auto time_seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         now.time_since_epoch())
                         .count();
    uint32_t combined_seed = static_cast<uint32_t>(embedding_id ^ time_seed);

    std::mt19937 rng(combined_seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
      embedding[i] = dist(rng);
    }
    return embedding;
  }

  std::unique_ptr<DramKVInferenceEmbedding<float>> backend_;
};

TEST_F(KVEmbeddingInferenceTest, InferenceLifecycleWithMetadata) {
  const int64_t embedding_id = 12345;

  auto now = std::chrono::system_clock::now();
  auto now_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
          .count();
  const uint32_t snapshot_timestamp = static_cast<uint32_t>(now_seconds - 120);

  auto embedding_data = generateEmbedding(embedding_id);

  LOG(INFO) << "STEP 1: Define test embedding";
  LOG(INFO) << "Embedding ID: " << embedding_id;
  LOG(INFO) << "Timestamp: " << snapshot_timestamp
            << " (current time - 2 minutes)";
  LOG(INFO) << "Dimension: " << EMBEDDING_DIM;
  LOG(INFO) << "First 5 elements: [" << embedding_data[0] << ", "
            << embedding_data[1] << ", " << embedding_data[2] << ", "
            << embedding_data[3] << ", " << embedding_data[4] << "]";

  auto indices_tensor = at::tensor({embedding_id}, at::kLong);
  auto weights_tensor = at::from_blob(
      embedding_data.data(),
      {1, EMBEDDING_DIM},
      at::TensorOptions().dtype(at::kFloat));
  auto count_tensor = at::tensor({1}, at::kInt);

  LOG(INFO) << "STEP 2: Insert embedding into cache";
  folly::coro::blockingWait(backend_->inference_set_kv_db_async(
      indices_tensor, weights_tensor, count_tensor, snapshot_timestamp));
  LOG(INFO) << "Insertion completed";

  auto retrieved_embedding = at::zeros({1, EMBEDDING_DIM}, at::kFloat);

  LOG(INFO) << "STEP 3: Retrieve embedding from cache";
  folly::coro::blockingWait(backend_->get_kv_db_async(
      indices_tensor, retrieved_embedding, count_tensor));
  LOG(INFO) << "Retrieval completed";

  auto retrieved_ptr = retrieved_embedding.data_ptr<float>();
  bool all_match = true;
  int mismatch_count = 0;

  LOG(INFO) << "STEP 4: Verify embedding consistency";
  for (int i = 0; i < EMBEDDING_DIM; ++i) {
    if (std::abs(retrieved_ptr[i] - embedding_data[i]) > 1e-5f) {
      all_match = false;
      mismatch_count++;
    }
  }

  if (all_match) {
    LOG(INFO) << "All " << EMBEDDING_DIM << " dimensions match";
  } else {
    LOG(ERROR) << "Found " << mismatch_count << " mismatches out of "
               << EMBEDDING_DIM << " dimensions";
  }

  ASSERT_TRUE(all_match) << "Retrieved embedding must match inserted embedding";

  LOG(INFO) << "STEP 5: Test repeated reads";
  for (int iteration = 1; iteration <= 3; ++iteration) {
    auto read_again = at::zeros({1, EMBEDDING_DIM}, at::kFloat);
    folly::coro::blockingWait(
        backend_->get_kv_db_async(indices_tensor, read_again, count_tensor));

    auto read_ptr = read_again.data_ptr<float>();
    bool matches = true;
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
      if (std::abs(read_ptr[i] - embedding_data[i]) > 1e-5f) {
        matches = false;
        break;
      }
    }
    LOG(INFO) << "Read #" << iteration << ": "
              << (matches ? "Match" : "Mismatch");
  }

  LOG(INFO) << "STEP 6: Trigger eviction";
  auto eviction_time = std::chrono::system_clock::now();
  auto eviction_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                              eviction_time.time_since_epoch())
                              .count();
  uint32_t eviction_threshold = static_cast<uint32_t>(eviction_seconds - 60);

  LOG(INFO) << "Eviction threshold: " << eviction_threshold;
  backend_->trigger_feature_evict(eviction_threshold);
  backend_->wait_until_eviction_done();
  LOG(INFO) << "Eviction completed";

  auto post_eviction_embedding = at::zeros({1, EMBEDDING_DIM}, at::kFloat);

  LOG(INFO) << "STEP 7: Read embedding after eviction";
  folly::coro::blockingWait(backend_->get_kv_db_async(
      indices_tensor, post_eviction_embedding, count_tensor));

  auto post_eviction_ptr = post_eviction_embedding.data_ptr<float>();
  bool values_changed = false;
  int differences = 0;

  for (int i = 0; i < EMBEDDING_DIM; ++i) {
    if (std::abs(post_eviction_ptr[i] - embedding_data[i]) > 1e-5f) {
      values_changed = true;
      differences++;
    }
  }

  LOG(INFO) << "Differences found: " << differences << "/" << EMBEDDING_DIM;

  if (values_changed) {
    LOG(INFO) << "Eviction successful - values changed";
  } else {
    LOG(ERROR) << "Eviction may have failed - values unchanged";
  }

  LOG(INFO) << "Original (cached): [" << embedding_data[0] << ", "
            << embedding_data[1] << ", " << embedding_data[2] << ", "
            << embedding_data[3] << ", " << embedding_data[4] << "]";
  LOG(INFO) << "After eviction: [" << post_eviction_ptr[0] << ", "
            << post_eviction_ptr[1] << ", " << post_eviction_ptr[2] << ", "
            << post_eviction_ptr[3] << ", " << post_eviction_ptr[4] << "]";

  ASSERT_TRUE(values_changed) << "Embedding should be different after eviction";

  LOG(INFO) << "Test completed successfully";
}

} // namespace kv_mem
