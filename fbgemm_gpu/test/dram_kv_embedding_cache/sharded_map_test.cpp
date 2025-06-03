/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <iostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common/time/Time.h>
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/SynchronizedShardedMap.h" // @manual
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/store_value_utils.h" // @manual

using namespace ::testing;
namespace kv_mem {
std::vector<float> generateFixedEmbedding(int dimension) {
  return std::vector<float>(dimension, 1.0);
}

std::vector<double>
memPoolEmbedding(int dimension, size_t numInserts, size_t numLookups) {
  const size_t numShards = 1;

  SynchronizedShardedMap<unsigned long, float*> embeddingMap(
      numShards,
      dimension * sizeof(float), // block_size
      alignof(float), // block_alignment
      8192); // blocks_per_chunk
  double insertTime, lookupTime;
  {
    std::vector<float> fixedEmbedding = generateFixedEmbedding(dimension);

    auto wlock = embeddingMap.by(0).wlock();
    auto* pool = embeddingMap.pool_by(0);
    std::pmr::polymorphic_allocator<float> alloc(pool);

    auto startInsert = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numInserts; i++) {
      float* arr = alloc.allocate(dimension);
      std::copy(fixedEmbedding.begin(), fixedEmbedding.end(), arr);
      wlock->insert_or_assign(i, arr);
    }
    auto endInsert = std::chrono::high_resolution_clock::now();
    insertTime =
        std::chrono::duration<double, std::milli>(endInsert - startInsert)
            .count();
  }

  std::vector<float> lookEmbedding(dimension);
  size_t hitCount = 0;
  {
    auto rlock = embeddingMap.by(0).rlock();
    auto startLookup = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numLookups; i++) {
      auto it = rlock->find(i % numInserts);
      if (it != rlock->end()) {
        hitCount++;
        std::copy(it->second, it->second + dimension, lookEmbedding.data());
      }
    }
    auto endLookup = std::chrono::high_resolution_clock::now();
    lookupTime =
        std::chrono::duration<double, std::milli>(endLookup - startLookup)
            .count();
  }

  std::cout << std::left << std::setw(20) << dimension;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::setw(20) << insertTime;
  std::cout << std::setw(20) << lookupTime;
  std::cout << std::setw(20) << (100.0 * (double)hitCount / (double)numLookups);
  std::cout << std::endl;
  return std::vector<double>(
      {insertTime, lookupTime, (double)hitCount / (double)numLookups});
}

std::vector<double>
memPoolEmbeddingWithTime(int dimension, size_t numInserts, size_t numLookups) {
  const size_t numShards = 1;
  size_t block_size = StoreValueUtils::calculate_block_size<float>(dimension);
  size_t block_alignment = StoreValueUtils::calculate_block_alignment<float>();

  SynchronizedShardedMap<unsigned long, float*> embeddingMap(
      numShards,
      block_size, // block_size
      block_alignment, // block_alignment
      8192); // blocks_per_chunk
  double insertTime, lookupTime;
  {
    std::vector<float> fixedEmbedding = generateFixedEmbedding(dimension);

    auto wlock = embeddingMap.by(0).wlock();
    auto* pool = embeddingMap.pool_by(0);

    auto startInsert = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numInserts; i++) {
      auto* block =
          StoreValueUtils::allocate<float>(block_size, block_alignment, pool);
      auto* data_ptr = StoreValueUtils::data_ptr<float>(block);
      std::copy(fixedEmbedding.begin(), fixedEmbedding.end(), data_ptr);
      wlock->insert_or_assign(i, block);
    }
    auto endInsert = std::chrono::high_resolution_clock::now();
    insertTime =
        std::chrono::duration<double, std::milli>(endInsert - startInsert)
            .count();
  }

  std::vector<float> lookEmbedding(dimension);
  size_t hitCount = 0;
  {
    auto rlock = embeddingMap.by(0).rlock();
    auto startLookup = std::chrono::high_resolution_clock::now();
    auto now = facebook::WallClockUtil::NowInUsecFast();
    for (size_t i = 0; i < numLookups; i++) {
      auto it = rlock->find(i % numInserts);
      if (it != rlock->end()) {
        hitCount++;
        const float* data_ptr = StoreValueUtils::data_ptr<float>(it->second);
        // update timestamp
        StoreValueUtils::update_timestamp<float>(it->second, now);
        std::copy(data_ptr, data_ptr + dimension, lookEmbedding.data());
      }
    }
    auto endLookup = std::chrono::high_resolution_clock::now();
    lookupTime =
        std::chrono::duration<double, std::milli>(endLookup - startLookup)
            .count();
  }

  std::cout << std::left << std::setw(20) << dimension;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::setw(20) << insertTime;
  std::cout << std::setw(20) << lookupTime;
  std::cout << std::setw(20) << (100.0 * (double)hitCount / (double)numLookups);
  std::cout << std::endl;
  return std::vector<double>(
      {insertTime, lookupTime, (double)hitCount / (double)numLookups});
}

int benchmark() {
  std::vector<int> dimensions = {64, 128, 256, 512, 1024};
  const size_t numInserts = 1'000'000; // 1 million insert
  const size_t numLookups = 1'000'000; // 1 million find

  std::cout
      << "======================= mempool ===================================="
      << std::endl;
  std::cout << std::left << std::setw(20) << "dim" << std::setw(20)
            << "insert time (ms)" << std::setw(20) << "find time (ms)"
            << std::setw(20) << "hit rate (%)" << std::endl;
  std::vector<std::vector<double>> results_by_dim_wo_ts;
  std::vector<std::vector<double>> results_by_dim_with_ts;
  for (int dim : dimensions) {
    results_by_dim_wo_ts.push_back(
        memPoolEmbedding(dim, numInserts, numLookups));
  }
  std::cout << std::endl << std ::endl;

  std::cout << "======================= mempool with time "
               "===================================="
            << std::endl;
  std::cout << std::left << std::setw(20) << "dim" << std::setw(20)
            << "insert time (ms)" << std::setw(20) << "find time (ms)"
            << std::setw(20) << "hit rate (%)" << std::endl;
  for (int dim : dimensions) {
    results_by_dim_with_ts.push_back(
        memPoolEmbeddingWithTime(dim, numInserts, numLookups));
  }
  std::cout << std::endl << std ::endl;
  EXPECT_EQ(results_by_dim_with_ts.size(), results_by_dim_wo_ts.size());
  for (int i = 0; i < results_by_dim_with_ts.size(); i++) {
    double perf_insert_ratio =
        results_by_dim_with_ts[i][0] / results_by_dim_wo_ts[i][0];
    double perf_lookup_ratio =
        results_by_dim_with_ts[i][1] / results_by_dim_wo_ts[i][1];
    EXPECT_THAT(perf_insert_ratio, AllOf(Ge(0.5), Le(1.5)));
    EXPECT_THAT(perf_lookup_ratio, AllOf(Ge(0.5), Le(1.5)));
    EXPECT_EQ(results_by_dim_with_ts[i][2], results_by_dim_wo_ts[i][2]);
    EXPECT_EQ(results_by_dim_with_ts[i][2], 1);
  }
  return 0;
}
TEST(SynchronizedShardedMap, benchmark) {
  benchmark();
}

} // namespace kv_mem
