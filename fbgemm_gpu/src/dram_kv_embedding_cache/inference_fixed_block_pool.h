/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "fixed_block_pool.h"

namespace kv_mem {

/// @brief A memory pool optimized for inference workloads.
///
/// Uses a compact 12-byte MetaHeader (vs 16-byte in base class) to save
/// 4 bytes per embedding. Only supports timestamp-based TTL eviction
/// (no count field for LFU eviction).
///
/// Memory layout per block:
/// ┌──────────────────────────────────────────────────────────────────┐
/// │ MetaHeader (12 bytes)          │ Embedding weights (D × scalar)  │
/// ├────────────┬───────────────────┼──────────────────────────────────┤
/// │ key (8B)   │ timestamp:31 + used:1 (4B) │ weight_type[max_D]      │
/// └────────────┴───────────────────┴──────────────────────────────────┘
///
class InferenceFixedBlockPool : public FixedBlockPool {
 public:
  // 12-byte header (uses #pragma pack to avoid padding to 16 bytes)
#pragma pack(push, 4)
  struct MetaHeader {
    int64_t key; // Feature key (8 bytes)
    uint32_t timestamp : 31; // Last update time in seconds (~68 years range)
    bool used : 1; // Block in-use flag
  };
#pragma pack(pop)
  static_assert(
      sizeof(MetaHeader) == 12,
      "InferenceFixedBlockPool::MetaHeader must be exactly 12 bytes");

  // Key operations - reuse from base class (key is at offset 0 in both)
  using FixedBlockPool::get_key;
  using FixedBlockPool::set_key;

  // Used flag operations
  static bool get_used(const void* block) {
    return reinterpret_cast<const MetaHeader*>(block)->used;
  }
  static void set_used(void* block, bool used) {
    reinterpret_cast<MetaHeader*>(block)->used = used;
  }

  // Timestamp operations
  static uint32_t get_timestamp(const void* block) {
    return reinterpret_cast<const MetaHeader*>(block)->timestamp;
  }
  static void set_timestamp(void* block, uint32_t time) {
    reinterpret_cast<MetaHeader*>(block)->timestamp = time;
  }
  static void update_timestamp(void* block) {
    reinterpret_cast<MetaHeader*>(block)->timestamp = current_timestamp();
  }

  // Block size: 12-byte header + embedding data
  template <typename scalar_t>
  static size_t calculate_block_size(size_t dimension) {
    return sizeof(MetaHeader) + dimension * sizeof(scalar_t);
  }

  template <typename scalar_t>
  static size_t calculate_block_alignment() {
    return std::max(alignof(MetaHeader), alignof(scalar_t));
  }

  // Get pointer to embedding data (skips 12-byte MetaHeader)
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

  explicit InferenceFixedBlockPool(
      std::size_t block_size,
      std::size_t block_alignment,
      std::size_t blocks_per_chunk = 8192,
      std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
      : FixedBlockPool(
            block_size,
            block_alignment,
            blocks_per_chunk,
            upstream) {}

  // Get block by index (used for eviction traversal)
  // Uses InferenceFixedBlockPool::get_used for 12-byte header layout
  template <typename scalar_t>
  scalar_t* get_block(size_t index) {
    std::lock_guard<std::mutex> guard(chunks_mutex_);
    char* current_chunk =
        static_cast<char*>(chunks_[index / blocks_per_chunk_].ptr);
    char* block = current_chunk + block_size_ * (index % blocks_per_chunk_);
    if (get_used(block)) {
      return reinterpret_cast<scalar_t*>(block);
    } else {
      return nullptr;
    }
  }

 protected:
  // Override to initialize block with 12-byte MetaHeader layout only
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (bytes != block_size_ || alignment != block_alignment_) {
      throw std::bad_alloc();
    }

    if (!free_list_) {
      allocate_chunk();
    }

    void* result = free_list_;
    free_list_ = *static_cast<void**>(free_list_);
    set_used(result, true);
    update_timestamp(result);
    return result;
  }

  void do_deallocate(
      void* p,
      [[maybe_unused]] std::size_t bytes,
      [[maybe_unused]] std::size_t alignment) override {
    *static_cast<void**>(p) = free_list_;
    free_list_ = p;
    set_used(p, false);
  }

  // Override to initialize 12-byte header's used flag to false
  void allocate_chunk() override {
    const std::size_t chunk_size = block_size_ * blocks_per_chunk_;

    // Allocate aligned memory through upstream resource
    void* chunk_ptr = upstream_->allocate(chunk_size, block_alignment_);

    // Record chunk information for later release
    {
      std::lock_guard<std::mutex> guard(chunks_mutex_);
      chunks_.push_back({chunk_ptr, chunk_size, block_alignment_});
    }

    // Initialize free list: link blocks in reverse order from chunk end to
    // beginning (improves locality)
    char* current = static_cast<char*>(chunk_ptr) + chunk_size;
    for (std::size_t i = 0; i < blocks_per_chunk_; ++i) {
      current -= block_size_;
      *reinterpret_cast<void**>(current) = free_list_;
      set_used(current, false);
      free_list_ = current;
    }
  }
};

} // namespace kv_mem
