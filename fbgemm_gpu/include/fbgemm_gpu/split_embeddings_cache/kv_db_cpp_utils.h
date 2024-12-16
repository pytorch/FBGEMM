/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <folly/hash/Hash.h>
#include <stddef.h>
#include <stdint.h>

/// @defgroup embedding-ssd Embedding SSD Operators
///

namespace kv_db_utils {

/// @ingroup embedding-ssd
///
/// @brief hash function used for SSD L2 cache and rocksdb sharding algorithm
///
/// @param id sharding key
/// @param num_shards sharding range
///
/// @return shard id ranges from [0, num_shards)
inline size_t hash_shard(int64_t id, size_t num_shards) {
  auto hash = folly::hash::fnv64_buf(
      reinterpret_cast<const char*>(&id), sizeof(int64_t));
  __uint128_t wide = __uint128_t{num_shards} * hash;
  return static_cast<size_t>(wide >> 64);
}

}; // namespace kv_db_utils
