/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "SynchronizedShardedMap.h"
#include "inference_fixed_block_pool.h"

namespace kv_mem {

/// @ingroup embedding-dram-kvstore
///
/// @brief Sharded synchronized hashmap for inference workloads.
///
/// Type alias for SynchronizedShardedMap using InferenceFixedBlockPool with
/// 12-byte MetaHeader for memory efficiency. This saves 4 bytes per embedding
/// compared to the standard SynchronizedShardedMap.
///
/// Example:
///    InferenceSynchronizedShardedMap<std::string, int> map;
///    wlmap = map.by(shard_id).wlock();
///    wlmap[topic] = 19;
///
template <typename K, typename V, typename M = folly::SharedMutexWritePriority>
using InferenceSynchronizedShardedMap =
    SynchronizedShardedMap<K, V, M, InferenceFixedBlockPool>;

} // namespace kv_mem
