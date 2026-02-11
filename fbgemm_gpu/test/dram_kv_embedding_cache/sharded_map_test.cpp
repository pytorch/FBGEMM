/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <iostream>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gtest/gtest.h>
#include <array>

#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/SynchronizedShardedMap.h"
#include "deeplearning/fbgemm/fbgemm_gpu/src/dram_kv_embedding_cache/fixed_block_pool.h"

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

  fmt::print(
      "{:<20}{:<20.2f}{:<20.2f}{:<20.2f}\n",
      dimension,
      insertTime,
      lookupTime,
      100.0 * static_cast<double>(hitCount) / static_cast<double>(numLookups));
  return std::vector<double>(
      {insertTime, lookupTime, (double)hitCount / (double)numLookups});
}

std::vector<double>
memPoolEmbeddingWithTime(int dimension, size_t numInserts, size_t numLookups) {
  const size_t numShards = 1;
  size_t block_size = FixedBlockPool::calculate_block_size<float>(dimension);
  size_t block_alignment = FixedBlockPool::calculate_block_alignment<float>();

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
      auto* block = pool->allocate_t<float>();
      auto* data_ptr = FixedBlockPool::data_ptr<float>(block);
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
    for (size_t i = 0; i < numLookups; i++) {
      auto it = rlock->find(i % numInserts);
      if (it != rlock->end()) {
        hitCount++;
        const float* data_ptr = FixedBlockPool::data_ptr<float>(it->second);
        // update timestamp
        FixedBlockPool::update_timestamp(it->second);
        std::copy(data_ptr, data_ptr + dimension, lookEmbedding.data());
      }
    }
    auto endLookup = std::chrono::high_resolution_clock::now();
    lookupTime =
        std::chrono::duration<double, std::milli>(endLookup - startLookup)
            .count();
  }

  fmt::print(
      "{:<20}{:<20.2f}{:<20.2f}{:<20.2f}\n",
      dimension,
      insertTime,
      lookupTime,
      100.0 * static_cast<double>(hitCount) / static_cast<double>(numLookups));
  return std::vector<double>(
      {insertTime, lookupTime, (double)hitCount / (double)numLookups});
}

void memPoolEmbeddingMemSize(int dimension, size_t numInserts) {
  const size_t numShards = 4;
  size_t block_size = FixedBlockPool::calculate_block_size<float>(dimension);
  size_t block_alignment = FixedBlockPool::calculate_block_alignment<float>();

  SynchronizedShardedMap<unsigned long, float*> embeddingMap(
      numShards,
      block_size, // block_size
      block_alignment, // block_alignment
      8192); // blocks_per_chunk
  {
    std::vector<float> fixedEmbedding = generateFixedEmbedding(dimension);

    auto wlock = embeddingMap.by(0).wlock();
    auto* pool = embeddingMap.pool_by(0);

    for (size_t i = 0; i < numInserts; i++) {
      auto* block = pool->allocate_t<float>();
      auto* data_ptr = FixedBlockPool::data_ptr<float>(block);
      std::copy(fixedEmbedding.begin(), fixedEmbedding.end(), data_ptr);
      wlock->insert_or_assign(i, block);
    }
  }
  size_t totalMemory = embeddingMap.getUsedMemSizeInBytes();
  size_t actualUsedChunkInBytes = embeddingMap.getActualUsedChunkInBytes();
  size_t numRows = embeddingMap.getNumRows();
  fmt::print(
      "{:<20}{:<20}{:<20.2f}{:<20.2f}{:<20}\n",
      dimension,
      numInserts,
      static_cast<double>(totalMemory) / (1024 * 1024),
      static_cast<double>(actualUsedChunkInBytes) / (1024 * 1024), // MB
      numRows);
}

int benchmark() {
  std::vector<int> dimensions = {64, 128, 256, 512, 1024};
  const size_t numInserts = 1'000'000; // 1 million insert
  const size_t numLookups = 1'000'000; // 1 million find

  fmt::print(
      "======================= mempool ====================================\n");
  fmt::print(
      "{:<20}{:<20}{:<20}{:<20}\n",
      "dim",
      "insert time (ms)",
      "find time (ms)",
      "hit rate (%)");
  std::vector<std::vector<double>> results_by_dim_wo_ts;
  std::vector<std::vector<double>> results_by_dim_with_ts;
  for (int dim : dimensions) {
    results_by_dim_wo_ts.push_back(
        memPoolEmbedding(dim, numInserts, numLookups));
  }
  fmt::print("\n\n");
  std::fflush(stdout);

  fmt::print(
      "======================= mempool with time "
      "====================================\n");
  fmt::print(
      "{:<20}{:<20}{:<20}{:<20}\n",
      "dim",
      "insert time (ms)",
      "find time (ms)",
      "hit rate (%)");
  for (int dim : dimensions) {
    results_by_dim_with_ts.push_back(
        memPoolEmbeddingWithTime(dim, numInserts, numLookups));
  }
  fmt::print("\n\n");

  fmt::print(
      "======================= memory usage statistics "
      "====================================\n");
  fmt::print(
      "{:<20}{:<20}{:<20}{:<20}{:<20}\n",
      "dim",
      "numInserts",
      "total memory (MB)",
      "actual used chunk (MB))",
      "numRows");
  for (int dim : dimensions) {
    memPoolEmbeddingMemSize(dim, numInserts);
  }
  return 0;
}

// comment out on purpose to skip CI and diff landing test
// uncomment it for local benchmarking
// TEST(SynchronizedShardedMap, benchmark) {
//   benchmark();
// }

} // namespace kv_mem
