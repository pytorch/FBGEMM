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
#include <glog/logging.h>
#include <stddef.h>
#include <stdint.h>
#include <filesystem>
#include <optional>
/// @defgroup embedding-ssd Embedding SSD Operators
///

namespace kv_db_utils {

#ifdef FBGEMM_FBCODE
constexpr size_t num_ssd_drives = 8;
#endif

/// @ingroup embedding-ssd
///
/// @brief hash function used for SSD L2 cache and rocksdb sharding algorithm
///
/// @param id sharding key
/// @param num_shards sharding range
///
/// @return shard id ranges from [0, num_shards)
inline size_t hash_shard(int64_t id, size_t num_shards) {
  auto hash = folly::hash::fnv64_buf_BROKEN(
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

/// @ingroup embedding-ssd
///
/// @brief default way to generate rocksdb path based on a user provided
/// base_path the file hierarchy will be
///   <base_path><ssd_idx>/<tbe_uuid> for default SSD mount
///   <base_path>/<tbe_uuid> for user provided base path
///
/// @param base_path the base path for all the rocksdb shards tied to one
/// TBE/EmbeddingRocksDB
/// @param db_shard_id the rocksdb shard index, this is used to determine which
/// SSD to use
/// @param tbe_uuid unique identifier per TBE at the lifetime of a training job
/// @param default_path whether the base_path is default SSD mount or
/// user-provided
///
/// @return the base path to that rocksdb shard
inline std::string get_rocksdb_path(
    const std::string& base_path,
    int db_shard_id,
    const std::string& tbe_uuid,
    bool default_path) {
  std::string rocksdb_path;
  if (default_path) {
    int ssd_drive_idx = db_shard_id % num_ssd_drives;
    std::string ssd_idx_tbe_id_str =
        std::to_string(ssd_drive_idx) + std::string("/") + tbe_uuid;
    rocksdb_path = base_path + ssd_idx_tbe_id_str;
  } else {
    rocksdb_path = base_path + std::string("/") + tbe_uuid;
  }
  LOG(INFO) << "[SSD Offloading] rocksdb path: " << rocksdb_path;

  // check if the SSD is mounted
  try {
    std::filesystem::create_directories(rocksdb_path);
  } catch (const std::filesystem::filesystem_error& e) {
    if (e.code() == std::errc::permission_denied) {
      // if no SSDs are mounted, override the path with prefix `/var/tmp`
      rocksdb_path = std::string("/var/tmp") + rocksdb_path;
      LOG(INFO) << "[SSD Offloading] No SSDs mounted, overriding using "
                << rocksdb_path;
    } else {
      LOG(INFO) << "Failed to create directory " << rocksdb_path
                << "with error " << e.what();
    }
  }

  LOG(INFO) << "[SSD Offloading] Returning rocksdb path: " << rocksdb_path;
  return rocksdb_path;
}

/// @ingroup embedding-ssd
///
/// @brief generate rocksdb shard path, based on rocksdb_path
/// the file hierarchy will be
///   <rocksdb_shard_path>/shard_<db_shard>
///
/// @param db_shard_id the rocksdb shard index
/// @param rocksdb_path the base path for rocksdb shard
///
/// @return the rocksdb shard path
inline std::string get_rocksdb_shard_path(
    int db_shard_id,
    const std::string& rocksdb_path) {
  return rocksdb_path + std::string("/shard_") + std::to_string(db_shard_id);
}

/// @ingroup embedding-ssd
///
/// @brief generate a directory to hold rocksdb checkpoint for a particular
/// rocksdb shard path the file hierarchy will be
///   <rocksdb_shard_path>/checkpoint_shard_<db_shard>
///
/// @param db_shard_id the rocksdb shard index
/// @param rocksdb_path the base path for rocksdb shard
///
/// @return the directory that holds rocksdb checkpoints for one rocksdb shard
inline std::string get_rocksdb_checkpoint_dir(
    int db_shard_id,
    const std::string& rocksdb_path) {
  return rocksdb_path + std::string("/checkpoint_shard_") +
      std::to_string(db_shard_id);
}

inline void create_dir(const std::string& dir_path) {
  try {
    std::filesystem::path fs_path(dir_path);
    bool res = std::filesystem::create_directories(fs_path);
    if (!res) {
      LOG(ERROR) << "dir: " << dir_path << " already exists";
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error creating directory: " << e.what();
  }
}

inline void remove_dir(const std::string& path) {
  if (std::filesystem::exists(path)) {
    try {
      if (std::filesystem::is_directory(path)) {
        std::filesystem::remove_all(path);
      } else {
        std::filesystem::remove(path);
      }
    } catch (const std::filesystem::filesystem_error& e) {
      LOG(ERROR) << "Error removing path: " << path
                 << ", exception:" << e.what();
    }
  }
}
}; // namespace kv_db_utils
