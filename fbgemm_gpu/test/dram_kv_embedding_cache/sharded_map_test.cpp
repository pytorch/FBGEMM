#include <cstdio>
#include <iostream>

#include <array>
#include <gtest/gtest.h>

#include "fbgemm_gpu/src/dram_kv_embedding_cache/SynchronizedShardedMap.h"
#include "fbgemm_gpu/src/dram_kv_embedding_cache/fixed_block_pool.h"

namespace kv_mem {
std::vector<float> generateFixedEmbedding(int dimension) {
  return std::vector<float>(dimension, 1.0);
}

void memPoolEmbedding(int dimension, size_t numInserts, size_t numLookups) {
  const size_t numShards = 1;

  SynchronizedShardedMap<unsigned long, float*> embeddingMap(
      numShards,
      dimension * sizeof(float),  // block_size
      alignof(float),             // block_alignment
      8192);                      // blocks_per_chunk
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
}

void memPoolEmbeddingWithTime(int dimension,
                              size_t numInserts,
                              size_t numLookups) {
  const size_t numShards = 1;
  size_t block_size = MemPoolUtils::calculate_block_size<float>(dimension);
  size_t block_alignment = MemPoolUtils::calculate_block_alignment<float>();

  SynchronizedShardedMap<unsigned long, float*> embeddingMap(
      numShards,
      block_size,       // block_size
      block_alignment,  // block_alignment
      8192);            // blocks_per_chunk
  double insertTime, lookupTime;
  {
    std::vector<float> fixedEmbedding = generateFixedEmbedding(dimension);

    auto wlock = embeddingMap.by(0).wlock();
    auto* pool = embeddingMap.pool_by(0);

    auto startInsert = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numInserts; i++) {
      auto* block =
          MemPoolUtils::allocate<float>(block_size, block_alignment, pool);
      auto* data_ptr = MemPoolUtils::data_ptr<float>(block);
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
        const float* data_ptr = MemPoolUtils::data_ptr<float>(it->second);
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

  std::cout << std::left << std::setw(20) << dimension;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::setw(20) << insertTime;
  std::cout << std::setw(20) << lookupTime;
  std::cout << std::setw(20) << (100.0 * (double)hitCount / (double)numLookups);
  std::cout << std::endl;
}

int benchmark() {
  std::vector<int> dimensions = {4, 8, 16, 32, 64};
  const size_t numInserts = 1'000'000;  // 1 million insert
  const size_t numLookups = 1'000'000;  // 1 million find

  std::cout
      << "======================= mempool ===================================="
      << std::endl;
  std::cout << std::left << std::setw(20) << "dim" << std::setw(20)
            << "insert time (ms)" << std::setw(20) << "find time (ms)"
            << std::setw(20) << "hit rate (%)" << std::endl;
  for (int dim : dimensions) {
    memPoolEmbedding(dim, numInserts, numLookups);
  }
  std::cout << std::endl << std ::endl;

  std::cout << "======================= mempool with time "
               "===================================="
            << std::endl;
  std::cout << std::left << std::setw(20) << "dim" << std::setw(20)
            << "insert time (ms)" << std::setw(20) << "find time (ms)"
            << std::setw(20) << "hit rate (%)" << std::endl;
  for (int dim : dimensions) {
    memPoolEmbeddingWithTime(dim, numInserts, numLookups);
  }
  std::cout << std::endl << std ::endl;
  return 0;
}
TEST(SynchronizedShardedMap, benchmark) { benchmark(); }

}  // namespace kv_mem