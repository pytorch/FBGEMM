//
// Created by arron on 2025/5/22.
//
#include <cstdio>
#include <iostream>
#include <random>

#include <array>
#include <cmath>
#include <gtest/gtest.h>

#include "fbgemm_gpu/src/dram_kv_embedding_cache/SynchronizedShardedMap.h"

namespace kv_mem {

// Zipf分布生成器实现
// alpha=1.3 → 约90%重复率
// alpha=1.5 → 约95%重复率
// alpha=2.0 → 约99%重复率
class ZipfGenerator {
 public:
  ZipfGenerator(double alpha, unsigned long n) : alpha_(alpha), n_(n), dist_(0.0, 1.0) {
    // 预计算调和数
    c_ = 0.0;
    for (unsigned long i = 1; i <= n_; ++i) c_ += 1.0 / std::pow(i, alpha_);
    c_ = 1.0 / c_;
  }

  template <typename Generator>
  unsigned long operator()(Generator& gen) {
    while (true) {
      double u = dist_(gen);
      double v = dist_(gen);
      unsigned long k = static_cast<unsigned long>(std::floor(std::pow(u, -1.0 / (alpha_ - 1.0))));
      if (k > n_) continue;
      double T = std::pow((k + 1.0) / k, alpha_ - 1.0);
      double accept_prob = (std::pow(k, -alpha_)) / (c_ * v * (T - 1.0) * k / n_);
      if (accept_prob >= 1.0 || dist_(gen) < accept_prob) {
        return k;
      }
    }
  }

 private:
  double alpha_;     // 分布参数（>1.0）
  unsigned long n_;  // 元素总数
  double c_;         // 归一化常数
  std::uniform_real_distribution<double> dist_;
};

std::vector<float> generateFixedEmbedding(int dimension) { return std::vector<float>(dimension, 1.0); }

void memPoolEmbeddingWithTime(int dimension, size_t numInserts, size_t numLookups) {
  const size_t numShards = 1;
  size_t block_size = FixedBlockPool::calculate_block_size<float>(dimension);
  size_t block_alignment = FixedBlockPool::calculate_block_alignment<float>();

  const size_t TOTAL_KEYS = 1'000'000;  // 1百万个可能的键
  const double ZIPF_ALPHA = 1.5;        // 调整这个参数控制热点程度

  ZipfGenerator zipf(ZIPF_ALPHA, TOTAL_KEYS);
  std::random_device rd;
  std::mt19937 gen(rd());

  SynchronizedShardedMap<unsigned long, float*> embeddingMap(numShards,
                                                             block_size,       // block_size
                                                             block_alignment,  // block_alignment
                                                             8192);            // blocks_per_chunk
  double insertTime, lookupTime;
  {
    std::vector<float> fixedEmbedding = generateFixedEmbedding(dimension);

    auto wlmap = embeddingMap.by(0).wlock();
    auto* pool = embeddingMap.pool_by(0);

    auto startInsert = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numInserts; i++) {
      auto id = zipf(gen);
      // use mempool
      float* block = nullptr;
      // First check if the key already exists
      auto it = wlmap->find(id);
      if (it != wlmap->end()) {
        block = it->second;
      } else {
        // Key doesn't exist, allocate new block and insert.
        block = FixedBlockPool::allocate_t<float>(block_size, block_alignment, pool);
        FixedBlockPool::set_key(block, id);
        FixedBlockPool::set_score(block, 0);
        FixedBlockPool::set_used(block, true);

        wlmap->insert({id, block});
      }
      FixedBlockPool::update_score(block);
      auto* data_ptr = FixedBlockPool::data_ptr<float>(block);
      std::copy(fixedEmbedding.begin(), fixedEmbedding.end(), data_ptr);
    }
    auto endInsert = std::chrono::high_resolution_clock::now();
    insertTime = std::chrono::duration<double, std::milli>(endInsert - startInsert).count();
  }

  std::vector<float> lookEmbedding(dimension);
  size_t hitCount = 0;
  {
    auto rlock = embeddingMap.by(0).rlock();
    auto startLookup = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numLookups; i++) {
      auto id = zipf(gen);
      auto it = rlock->find(id);
      if (it != rlock->end()) {
        hitCount++;
        const float* data_ptr = FixedBlockPool::data_ptr<float>(it->second);
        std::copy(data_ptr, data_ptr + dimension, lookEmbedding.data());
      }
    }
    auto endLookup = std::chrono::high_resolution_clock::now();
    lookupTime = std::chrono::duration<double, std::milli>(endLookup - startLookup).count();
  }

  {
    size_t score_sum = 0;
    auto rlock = embeddingMap.by(0).rlock();
    for (const auto& [key, block] : *rlock) {
      score_sum += FixedBlockPool::get_score(block);
    }
    ASSERT_EQ(score_sum, numInserts);
  }

  // 遍历 chunk 找到要淘汰的 key
  // 对 map 进行加锁，释放资源
  std::vector<uint64_t> low_keys;
  {
    auto rlock = embeddingMap.by(0).rlock();
    std::cout << "map num:" << rlock->size() << std::endl;
    auto* pool = embeddingMap.pool_by(0);
    FixedBlockPool::get_keys_with_low_score<float>(pool, 1, 0.99, low_keys);
    std::cout << "low key num:" << low_keys.size() << std::endl;
  }

  // 获取写锁，进行map 删除， pool 内存释放
  {
    // 获取写锁，进行map删除和pool内存释放
    auto wlock = embeddingMap.by(0).wlock();
    auto* pool = embeddingMap.pool_by(0);

    for (auto& key : low_keys) {
      // 1. 从map中查找并获取对应的block指针
      auto it = wlock->find(key);
      if (it != wlock->end()) {
        float* block = it->second;
        FixedBlockPool::deallocate_t<float>(block, block_size, block_alignment, pool);
        // 3. 从map中移除该键值对
        wlock->erase(it);
      }
    }
    std::cout << "after delete, map size:" << wlock->size() << std::endl;
  }

  // 删除阶段：分批次处理，每次处理1000个key
  const size_t batch_size = 1000;
  for (size_t i = 0; i < low_keys.size(); i += batch_size) {
    auto start = low_keys.begin() + i;
    auto end = (i + batch_size < low_keys.size()) ? low_keys.begin() + i + batch_size : low_keys.end();
    std::vector<uint64_t> batch(start, end);

    // 获取写锁处理当前批次
    auto wlock = embeddingMap.by(0).wlock();
    auto* pool = embeddingMap.pool_by(0);

    for (auto key : batch) {
      auto it = wlock->find(key);
      if (it != wlock->end()) {
        float* block = it->second;
        FixedBlockPool::deallocate_t<float>(block, block_size, block_alignment, pool);
        wlock->erase(it);
      }
    }
    std::cout << "after delete, map size:" << wlock->size() << std::endl;
  }

  std::cout << std::left << std::setw(20) << dimension;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::setw(20) << insertTime;
  std::cout << std::setw(20) << lookupTime;
  std::cout << std::setw(20) << (100.0 * (double)hitCount / (double)numLookups);
  std::cout << std::endl;
}

int benchmark() {
  std::vector<int> dimensions = {4};
  const size_t numInserts = 1'000'000;  // 1 million insert
  const size_t numLookups = 1'000'000;  // 1 million find

  std::cout << "======================= mempool ====================================" << std::endl;
  std::cout << std::left << std::setw(20) << "dim" << std::setw(20) << "insert time (ms)" << std::setw(20) << "find time (ms)" << std::setw(20) << "hit rate (%)" << std::endl;
  for (int dim : dimensions) {
    memPoolEmbeddingWithTime(dim, numInserts, numLookups);
  }
  return 0;
}
TEST(Evict, benchmark) { benchmark(); }
}  // namespace kv_mem
