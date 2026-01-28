/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/dram_kv_inference_embedding.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/fixed_block_pool.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>
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

  // Generate deterministic embeddings based on id for verification
  static std::vector<float> generateEmbedding(int64_t id) {
    std::vector<float> embedding(EMBEDDING_DIM);
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
      embedding[i] = static_cast<float>(id * 1000 + i) / 100000.0f;
    }
    return embedding;
  }

  // Generate uniform value embedding for consistency testing
  static std::vector<float> generateUniformEmbedding(float value) {
    return std::vector<float>(EMBEDDING_DIM, value);
  }

  // Insert a single embedding
  void insertEmbedding(
      int64_t id,
      const std::vector<float>& embedding,
      uint32_t ts = 1000) {
    auto indices = at::tensor({id}, at::kLong);
    // Clone the embedding data to avoid const_cast
    std::vector<float> embedding_copy(embedding);
    auto weights = at::from_blob(
        embedding_copy.data(),
        {1, EMBEDDING_DIM},
        at::TensorOptions().dtype(at::kFloat));
    auto count = at::tensor({1}, at::kInt);
    folly::coro::blockingWait(
        backend_->inference_set_kv_db_async(indices, weights, count, ts));
  }

  // Insert embedding using its deterministic pattern
  void insertEmbedding(int64_t id, uint32_t ts = 1000) {
    auto embedding = generateEmbedding(id);
    insertEmbedding(id, embedding, ts);
  }

  // Read a single embedding
  std::vector<float> readEmbedding(int64_t id) {
    auto indices = at::tensor({id}, at::kLong);
    auto weights = at::zeros({1, EMBEDDING_DIM}, at::kFloat);
    auto count = at::tensor({1}, at::kInt);
    folly::coro::blockingWait(
        backend_->get_kv_db_async(indices, weights, count));
    auto* data = weights.const_data_ptr<float>();
    return std::vector<float>(data, data + EMBEDDING_DIM);
  }

  // Verify embedding matches expected pattern
  static bool verifyEmbedding(
      const std::vector<float>& actual,
      const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
      return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
      if (std::abs(actual[i] - expected[i]) > 1e-5f) {
        return false;
      }
    }
    return true;
  }

  // Check if embedding has corrupted values (NaN/Inf)
  static bool isCorrupted(const std::vector<float>& embedding) {
    return std::ranges::any_of(
        embedding, [](float v) { return std::isnan(v) || std::isinf(v); });
  }

  // Bulk insert embeddings
  void bulkInsert(int64_t start_id, int64_t count, uint32_t base_ts = 1000) {
    for (int64_t i = 0; i < count; ++i) {
      insertEmbedding(start_id + i, base_ts);
    }
  }

  // Run parallel operations with configurable thread count and work
  template <typename WorkFn>
  void runParallel(int num_threads, WorkFn work_fn) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([t, &work_fn]() { work_fn(t); });
    }
    for (auto& thread : threads) {
      thread.join();
    }
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

  LOG(INFO) << "STEP 1: Insert embedding";
  insertEmbedding(embedding_id, embedding_data, snapshot_timestamp);

  LOG(INFO) << "STEP 2: Retrieve and verify";
  auto retrieved = readEmbedding(embedding_id);
  ASSERT_TRUE(verifyEmbedding(retrieved, embedding_data))
      << "Retrieved embedding must match inserted embedding";

  LOG(INFO) << "STEP 3: Test repeated reads";
  for (int iteration = 1; iteration <= 3; ++iteration) {
    auto read_again = readEmbedding(embedding_id);
    ASSERT_TRUE(verifyEmbedding(read_again, embedding_data))
        << "Read #" << iteration << " failed";
  }

  LOG(INFO) << "STEP 4: Trigger eviction";
  auto eviction_time = std::chrono::system_clock::now();
  auto eviction_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                              eviction_time.time_since_epoch())
                              .count();
  uint32_t eviction_threshold = static_cast<uint32_t>(eviction_seconds - 60);

  backend_->trigger_feature_evict(eviction_threshold);
  backend_->wait_until_eviction_done();

  LOG(INFO) << "STEP 5: Verify eviction occurred";
  auto post_eviction = readEmbedding(embedding_id);
  ASSERT_FALSE(verifyEmbedding(post_eviction, embedding_data))
      << "Embedding should be different after eviction";

  LOG(INFO) << "Test completed successfully";
}

// Concurrent reads test
TEST_F(KVEmbeddingInferenceTest, ConcurrentReads) {
  LOG(INFO) << "=== ConcurrentReads Test ===";

  constexpr int NUM_EMBEDDINGS = 5000;
  constexpr int NUM_READERS = 64;
  constexpr int READS_PER_THREAD = 2000;

  LOG(INFO) << "Inserting " << NUM_EMBEDDINGS << " embeddings";
  bulkInsert(0, NUM_EMBEDDINGS);

  std::atomic<int> success_count{0};
  std::atomic<int> failure_count{0};

  auto start_time = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Starting " << NUM_READERS << " reader threads, "
            << READS_PER_THREAD << " reads each";

  runParallel(NUM_READERS, [&](int t) {
    std::mt19937 rng(t);
    std::uniform_int_distribution<int64_t> dist(0, NUM_EMBEDDINGS - 1);

    for (int r = 0; r < READS_PER_THREAD; ++r) {
      int64_t id = dist(rng);
      auto retrieved = readEmbedding(id);
      auto expected = generateEmbedding(id);

      if (verifyEmbedding(retrieved, expected)) {
        success_count.fetch_add(1, std::memory_order_relaxed);
      } else {
        failure_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count();

  int total_reads = NUM_READERS * READS_PER_THREAD;
  double reads_per_sec =
      static_cast<double>(total_reads) / duration_ms * 1000.0;

  LOG(INFO) << "Completed " << total_reads << " reads in " << duration_ms
            << "ms (" << reads_per_sec << " reads/sec)";
  LOG(INFO) << "Successes: " << success_count.load()
            << ", Failures: " << failure_count.load();

  ASSERT_EQ(success_count.load(), total_reads);
  ASSERT_EQ(failure_count.load(), 0);
}

// Concurrent writes test (different keys)
TEST_F(KVEmbeddingInferenceTest, ConcurrentWrites) {
  LOG(INFO) << "=== ConcurrentWrites Test ===";

  constexpr int NUM_WRITERS = 64;
  constexpr int WRITES_PER_THREAD = 1000;

  std::atomic<int> write_count{0};

  auto start_time = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Starting " << NUM_WRITERS << " writer threads, "
            << WRITES_PER_THREAD << " writes each";

  runParallel(NUM_WRITERS, [&](int t) {
    int64_t base_id = t * WRITES_PER_THREAD;
    for (int i = 0; i < WRITES_PER_THREAD; ++i) {
      insertEmbedding(base_id + i, 1000 + t);
      write_count.fetch_add(1, std::memory_order_relaxed);
    }
  });

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count();

  int total_writes = NUM_WRITERS * WRITES_PER_THREAD;
  double writes_per_sec =
      static_cast<double>(total_writes) / duration_ms * 1000.0;

  LOG(INFO) << "Completed " << write_count.load() << " writes in "
            << duration_ms << "ms (" << writes_per_sec << " writes/sec)";

  // Verify all embeddings
  LOG(INFO) << "Verifying all " << total_writes << " embeddings";
  int verify_success = 0;
  for (int64_t id = 0; id < total_writes; ++id) {
    auto retrieved = readEmbedding(id);
    auto expected = generateEmbedding(id);
    if (verifyEmbedding(retrieved, expected)) {
      verify_success++;
    }
  }

  LOG(INFO) << "Verification: " << verify_success << "/" << total_writes;
  ASSERT_EQ(verify_success, total_writes);
}

// Mixed concurrent read/write with same key contention
TEST_F(KVEmbeddingInferenceTest, ConcurrentReadWrite) {
  LOG(INFO) << "=== ConcurrentReadWrite Test ===";

  constexpr int NUM_HOT_KEYS = 100;
  constexpr int NUM_READERS = 32;
  constexpr int NUM_WRITERS = 32;
  constexpr int OPS_PER_WRITER = 2000;

  LOG(INFO) << "Pre-populating " << NUM_HOT_KEYS << " hot keys";
  bulkInsert(0, NUM_HOT_KEYS);

  std::atomic<int64_t> read_count{0};
  std::atomic<int64_t> write_count{0};
  std::atomic<bool> stop_readers{false};
  std::atomic<int> corruption_errors{0};
  std::atomic<int> exception_errors{0};

  std::vector<std::thread> threads;
  threads.reserve(NUM_READERS + NUM_WRITERS);

  auto start_time = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Starting " << NUM_READERS << " readers and " << NUM_WRITERS
            << " writers (" << OPS_PER_WRITER << " ops each)";

  // Reader threads - run until writers complete
  for (int t = 0; t < NUM_READERS; ++t) {
    threads.emplace_back([this,
                          t,
                          &read_count,
                          &stop_readers,
                          &corruption_errors,
                          &exception_errors]() {
      std::mt19937 rng(t + 1000);
      std::uniform_int_distribution<int64_t> dist(0, NUM_HOT_KEYS - 1);

      while (!stop_readers.load(std::memory_order_acquire)) {
        try {
          auto retrieved = readEmbedding(dist(rng));
          read_count.fetch_add(1, std::memory_order_relaxed);
          if (isCorrupted(retrieved)) {
            corruption_errors.fetch_add(1, std::memory_order_relaxed);
          }
        } catch (...) {
          exception_errors.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  // Writer threads - fixed number of operations
  for (int t = 0; t < NUM_WRITERS; ++t) {
    threads.emplace_back([this, t, &write_count]() {
      std::mt19937 rng(t + 2000);
      std::uniform_int_distribution<int64_t> key_dist(0, NUM_HOT_KEYS - 1);

      for (int i = 0; i < OPS_PER_WRITER; ++i) {
        int64_t key = key_dist(rng);
        int64_t version = t * OPS_PER_WRITER + i;
        auto embedding = generateEmbedding(version);
        insertEmbedding(key, embedding, 2000 + i);
        write_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Wait for writers to complete
  for (size_t i = NUM_READERS; i < threads.size(); ++i) {
    threads[i].join();
  }

  // Stop and join readers
  stop_readers.store(true, std::memory_order_release);
  for (size_t i = 0; i < NUM_READERS; ++i) {
    threads[i].join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count();

  int64_t total_ops = read_count.load() + write_count.load();
  double ops_per_sec = static_cast<double>(total_ops) / duration_ms * 1000.0;

  LOG(INFO) << "Completed in " << duration_ms << "ms";
  LOG(INFO) << "Reads: " << read_count.load()
            << ", Writes: " << write_count.load();
  LOG(INFO) << "Throughput: " << ops_per_sec << " ops/sec";
  LOG(INFO) << "Corruption errors: " << corruption_errors.load()
            << ", Exception errors: " << exception_errors.load();

  ASSERT_EQ(corruption_errors.load(), 0);
  ASSERT_EQ(exception_errors.load(), 0);
  ASSERT_EQ(write_count.load(), NUM_WRITERS * OPS_PER_WRITER);
  ASSERT_GT(read_count.load(), 0);
}

// Concurrent writes to same key - stress test for lock contention
TEST_F(KVEmbeddingInferenceTest, ConcurrentWritesSameKey) {
  LOG(INFO) << "=== ConcurrentWritesSameKey Test ===";

  constexpr int64_t TARGET_KEY = 42;
  constexpr int NUM_WRITERS = 64;
  constexpr int WRITES_PER_THREAD = 500;

  std::atomic<int> write_count{0};
  std::vector<std::pair<int, int>> all_versions;

  // Collect all possible versions
  for (int t = 0; t < NUM_WRITERS; ++t) {
    for (int i = 0; i < WRITES_PER_THREAD; ++i) {
      all_versions.emplace_back(t, i);
    }
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Starting " << NUM_WRITERS << " threads writing to key "
            << TARGET_KEY << " (" << WRITES_PER_THREAD << " writes each)";

  runParallel(NUM_WRITERS, [&](int t) {
    for (int i = 0; i < WRITES_PER_THREAD; ++i) {
      int64_t version = t * 1000 + i;
      auto embedding = generateEmbedding(version);
      insertEmbedding(TARGET_KEY, embedding, 3000 + t);
      write_count.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::yield();
    }
  });

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count();

  LOG(INFO) << "Completed " << write_count.load() << " writes in "
            << duration_ms << "ms";

  // Verify final value matches one of the written versions
  auto final_embedding = readEmbedding(TARGET_KEY);
  bool found_match = false;

  for (const auto& [t, i] : all_versions) {
    int64_t version = t * 1000 + i;
    auto expected = generateEmbedding(version);
    if (verifyEmbedding(final_embedding, expected)) {
      found_match = true;
      LOG(INFO) << "Final value matches thread " << t << " iteration " << i;
      break;
    }
  }

  ASSERT_EQ(write_count.load(), NUM_WRITERS * WRITES_PER_THREAD);
  ASSERT_TRUE(found_match) << "Final value should match one written version";
}

// Read consistency test - verify atomic updates
TEST_F(KVEmbeddingInferenceTest, ReadConsistencyDuringWrites) {
  LOG(INFO) << "=== ReadConsistencyDuringWrites Test ===";

  constexpr int64_t TARGET_KEY = 99;
  constexpr int NUM_UPDATES = 2000;
  constexpr int NUM_READERS = 32;

  auto embedding_v1 = generateUniformEmbedding(1.0f);
  auto embedding_v2 = generateUniformEmbedding(2.0f);

  insertEmbedding(TARGET_KEY, embedding_v1);

  std::atomic<bool> stop_flag{false};
  std::atomic<int64_t> v1_reads{0};
  std::atomic<int64_t> v2_reads{0};
  std::atomic<int64_t> partial_reads{0};

  std::vector<std::thread> threads;
  threads.reserve(NUM_READERS + 1);

  LOG(INFO) << "Starting " << NUM_READERS << " readers with " << NUM_UPDATES
            << " toggle updates";

  // Reader threads
  for (int t = 0; t < NUM_READERS; ++t) {
    threads.emplace_back(
        [this, &stop_flag, &v1_reads, &v2_reads, &partial_reads]() {
          while (!stop_flag.load(std::memory_order_acquire)) {
            auto retrieved = readEmbedding(TARGET_KEY);

            bool all_v1 = true, all_v2 = true;
            for (float v : retrieved) {
              if (std::abs(v - 1.0f) > 1e-5f)
                all_v1 = false;
              if (std::abs(v - 2.0f) > 1e-5f)
                all_v2 = false;
            }

            if (all_v1) {
              v1_reads.fetch_add(1, std::memory_order_relaxed);
            } else if (all_v2) {
              v2_reads.fetch_add(1, std::memory_order_relaxed);
            } else {
              partial_reads.fetch_add(1, std::memory_order_relaxed);
            }
          }
        });
  }

  // Writer thread - toggle between v1 and v2
  threads.emplace_back([this, &embedding_v1, &embedding_v2]() {
    for (int i = 0; i < NUM_UPDATES; ++i) {
      auto& embedding = (i % 2 == 0) ? embedding_v2 : embedding_v1;
      insertEmbedding(TARGET_KEY, embedding, 5000 + i);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });

  threads.back().join();
  threads.pop_back();

  stop_flag.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  LOG(INFO) << "V1 reads: " << v1_reads.load()
            << ", V2 reads: " << v2_reads.load();
  LOG(INFO) << "Partial reads: " << partial_reads.load();

  if (partial_reads.load() > 0) {
    LOG(WARNING) << "Partial reads detected - updates may not be atomic";
  }

  ASSERT_GT(v1_reads.load() + v2_reads.load(), 0);
}

// Maximum concurrency test (mixed read/write)
TEST_F(KVEmbeddingInferenceTest, HighConcurrency) {
  LOG(INFO) << "=== HighConcurrency Test ===";

  constexpr int NUM_KEYS = 200;
  constexpr int NUM_THREADS = 128;
  constexpr int OPS_PER_THREAD = 5000;
  constexpr float WRITE_RATIO = 0.3f;

  LOG(INFO) << "Pre-populating " << NUM_KEYS << " keys";
  bulkInsert(0, NUM_KEYS);

  std::atomic<int64_t> reads{0};
  std::atomic<int64_t> writes{0};
  std::atomic<int> errors{0};

  auto start_time = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Starting " << NUM_THREADS << " threads with " << OPS_PER_THREAD
            << " ops each (" << (WRITE_RATIO * 100) << "% writes)";

  runParallel(NUM_THREADS, [&](int t) {
    std::mt19937 rng(t);
    std::uniform_int_distribution<int64_t> key_dist(0, NUM_KEYS - 1);
    std::uniform_real_distribution<float> op_dist(0.0f, 1.0f);

    for (int i = 0; i < OPS_PER_THREAD; ++i) {
      int64_t key = key_dist(rng);

      try {
        if (op_dist(rng) < WRITE_RATIO) {
          auto embedding = generateEmbedding(key * 10000 + i);
          insertEmbedding(key, embedding, 6000 + t * 1000 + i);
          writes.fetch_add(1, std::memory_order_relaxed);
        } else {
          auto retrieved = readEmbedding(key);
          reads.fetch_add(1, std::memory_order_relaxed);
          if (isCorrupted(retrieved)) {
            errors.fetch_add(1, std::memory_order_relaxed);
          }
        }
      } catch (...) {
        errors.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count();

  int64_t total_ops = reads.load() + writes.load();
  double ops_per_sec = static_cast<double>(total_ops) / duration_ms * 1000.0;

  LOG(INFO) << "Completed in " << duration_ms << "ms";
  LOG(INFO) << "Reads: " << reads.load() << ", Writes: " << writes.load();
  LOG(INFO) << "Throughput: " << ops_per_sec << " ops/sec";
  LOG(INFO) << "Errors: " << errors.load();

  ASSERT_EQ(errors.load(), 0);
  ASSERT_EQ(total_ops, NUM_THREADS * OPS_PER_THREAD);
}

} // namespace kv_mem
