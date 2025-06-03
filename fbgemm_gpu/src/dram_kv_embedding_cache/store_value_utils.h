/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <chrono>

#include "fixed_block_pool.h"

namespace kv_mem {

class StoreValueUtils {
 public:
  // Metadata structure (publicly accessible)
  // alignas(8) MetaHeader >= sizeof(void*), avoid mempool block too small.
  struct alignas(8) MetaHeader {
    int64_t timestamp; // 8 bytes
    // Can be extended with other fields: uint32_t counter, uint64_t key, etc.
  };

  // Create memory block with metadata
  template <typename scalar_t>
  static scalar_t*
  allocate(size_t& block_size, size_t& alignment, FixedBlockPool* pool) {
    return reinterpret_cast<scalar_t*>(pool->allocate(block_size, alignment));
  }

  // Destroy memory block
  template <typename scalar_t>
  static void deallocate(
      scalar_t* block,
      size_t& block_size,
      size_t& alignment,
      FixedBlockPool* pool) {
    pool->deallocate(block, block_size, alignment);
  }

  // Calculate storage size
  template <typename scalar_t>
  static size_t calculate_block_size(size_t dimension) {
    return sizeof(MetaHeader) + dimension * sizeof(scalar_t);
  }

  // Calculate alignment requirements
  template <typename scalar_t>
  static size_t calculate_block_alignment() {
    return std::max(alignof(MetaHeader), alignof(scalar_t));
  }

  // Metadata operations
  template <typename scalar_t>
  static int64_t get_timestamp(const scalar_t* block) {
    return reinterpret_cast<const MetaHeader*>(block)->timestamp;
  }

  template <typename scalar_t>
  static void set_timestamp(scalar_t* block, int64_t ts) {
    reinterpret_cast<MetaHeader*>(block)->timestamp = ts;
  }

  template <typename scalar_t>
  static void update_timestamp(scalar_t* block, const int64_t& ts) {
    reinterpret_cast<MetaHeader*>(block)->timestamp = ts;
  }

  // Data pointer retrieval
  template <typename scalar_t>
  static scalar_t* data_ptr(scalar_t* block) {
    return reinterpret_cast<scalar_t*>(
        reinterpret_cast<char*>(block) + sizeof(MetaHeader));
  }

  template <typename scalar_t>
  static const scalar_t* data_ptr(const scalar_t* block) {
    return reinterpret_cast<const scalar_t*>(
        reinterpret_cast<const char*>(block) + sizeof(MetaHeader));
  }

  static int64_t current_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
    // facebook::WallClockUtil::NowInUsecFast();
  }
};
} // namespace kv_mem
