/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/container/F14Map.h>
#include "folly/Synchronized.h"

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

  explicit SynchronizedShardedMap(std::size_t numShards) : shards_(numShards) {}

  // Get shard map by index
  auto& by(int index) {
    return shards_.at(index % shards_.size());
  }

  auto getNumShards() {
    return shards_.size();
  }

 private:
  std::vector<folly::Synchronized<folly::F14FastMap<K, V>, M>> shards_;
};
} // namespace kv_mem
