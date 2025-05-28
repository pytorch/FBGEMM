/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>

#include "fixed_block_pool.h"

namespace kv_mem {

/// @ingroup embedding-dram-kvstore
///
/// @brief generic sharded synchronized hashmap
// Sharded hash map. Each shard is synchronized.
// User needs to managed logic of sharding entries into different shards_.
//
// Example:
//    ShardedSynchronizedMap<std::string, int> map;
//    wlmap = map.by(shard_id).wlock();
//    wlmap[topic] = 19;
//
template <typename K, typename V, typename M = folly::SharedMutexWritePriority>
class SynchronizedShardedMap {
 public:
  using iterator = typename folly::F14FastMap<K, V>::const_iterator;

  explicit SynchronizedShardedMap(std::size_t numShards,
                                  std::size_t block_size,
                                  std::size_t block_alignment,
                                  std::size_t blocks_per_chunk = 8192)
      : shards_(numShards), mempools_(numShards) {
    // Init mempools_
    for (auto& pool : mempools_) {
      pool = std::make_unique<kv_mem::FixedBlockPool>(
          block_size, block_alignment, blocks_per_chunk);
    }
  }

  // Get shard map by index
  auto& by(int index) { return shards_.at(index % shards_.size()); }

  // Get shard pool by index
  auto* pool_by(int index) {
    return mempools_.at(index % shards_.size()).get();
  }

  auto getNumShards() { return shards_.size(); }

  auto getUsedMemSize() {
    size_t used_mem_size = 0;
    size_t block_size = mempools_[0]->get_aligned_block_size();
    for (size_t i = 0; i < shards_.size(); ++i) {
      auto rlmap = shards_[i].rlock();
      // only calculate the sizes of K, V and block that are used
      used_mem_size += rlmap->size() * (sizeof(K) + sizeof(V) + block_size);
    }
    return used_mem_size;
  }

 private:
  std::vector<folly::Synchronized<folly::F14FastMap<K, V>, M>> shards_;
  std::vector<std::unique_ptr<FixedBlockPool>> mempools_;
};
}  // namespace kv_mem
