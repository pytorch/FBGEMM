/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <folly/hash/Hash.h>
#include <stddef.h>
#include <stdint.h>
#include <optional>

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

/// @ingroup embedding-ssd
///
/// @brief given a tensor containing ids in random order, returns 2 tensors
/// tensor1 contains ids sorted in bucket ascending order, for example given
/// [1,2,3,4] with 2 buckets [1, 4) and [4, 7), the output will be [1,2,3,4] or
/// [2, 1, 3, 4], id 1, 2, 3 must be prior to 4, but 1 2 3 can be in any order
/// tensor2 contains number of embeddings in each bucket id(tensor offset), in
/// the above example tensor2 will be [3, 1] where first item corresponds to the
/// first bucket id, value 3 means there are 3 ids in the first bucket id
///
/// @param unordered_indices unordered ids, the id here might be
/// original(unlinearized) id
/// @param hash_mode 0 for chunk-based hashing, 1 for interleaved-based hashing
/// @param bucket_start global bucket id, the start of the bucket range
/// @param bucket_end global bucket id, the end of the bucket range
/// @param bucket_size an optional, virtual size(input space, e.g. 2^50) of a
/// bucket
/// @param total_num_buckets an optional with the total number of buckets per
/// training model
///
/// @return list of 2 tensors, first tensor is bucket sorted ids, second tensor
/// is bucket size

std::tuple<at::Tensor, at::Tensor> get_bucket_sorted_indices_and_bucket_tensor(
    const at::Tensor& unordered_indices,
    int64_t hash_mode,
    int64_t bucket_start,
    int64_t bucket_end,
    std::optional<int64_t> bucket_size,
    std::optional<int64_t> total_num_buckets);

}; // namespace kv_db_utils
