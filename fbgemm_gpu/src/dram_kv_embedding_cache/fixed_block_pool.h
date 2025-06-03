/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <memory_resource>
#include <stdexcept>
#include <vector>

#include <cassert>

namespace kv_mem {
class FixedBlockPool : public std::pmr::memory_resource {
 public:
  explicit FixedBlockPool(
      std::size_t block_size, // Size of each memory block
      std::size_t block_alignment, // Memory block alignment requirement
      std::size_t blocks_per_chunk = 8192, // Number of blocks per chunk
      std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
      // Minimum block size is 8 bytes
      : block_size_(std::max(block_size, sizeof(void*))),
        block_alignment_(block_alignment),
        blocks_per_chunk_(blocks_per_chunk),
        upstream_(upstream),
        chunks_(upstream) {
    // Validate minimum data size, whether it's less than 8 bytes
    // half type, 2 bytes, minimum embedding length 4
    // float type, 4 bytes, minimum embedding length 2
    // Large objects use memory pool, small objects are placed directly in the
    // hashtable
    if (block_size < sizeof(void*)) {
      // Block size must be at least able to store a pointer (for free list)
      throw std::invalid_argument("Block size must be at least sizeof(void*)");
    }

    // Validate that alignment requirement is a power of 2
    if ((block_alignment_ & (block_alignment_ - 1)) != 0) {
      throw std::invalid_argument("Alignment must be power of two");
    }

    // Validate that block size is a multiple of alignment
    if (block_size_ % block_alignment_ != 0) {
      throw std::invalid_argument("Block size must align with alignment");
    }

    // Ensure block size is at least 1
    if (block_size_ < 1) {
      throw std::invalid_argument("Block size must be at least 1");
    }
  }

  // Release all allocated memory during destruction
  ~FixedBlockPool() override {
    for (auto&& chunk : chunks_) {
      upstream_->deallocate(chunk.ptr, chunk.size, chunk.alignment);
    }
  }

 protected:
  // Core allocation function
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    // Only handle matching block size and alignment requirements
    if (bytes != block_size_ || alignment != block_alignment_) {
      throw std::bad_alloc();
    }

    // Allocate a new chunk when no blocks are available
    if (!free_list_) {
      allocate_chunk();
    }

    // Take a block from the head of the free list
    void* result = free_list_;
    free_list_ = *static_cast<void**>(free_list_);
    return result;
  }

  // Core deallocation function
  void do_deallocate(
      void* p,
      [[maybe_unused]] std::size_t bytes,
      [[maybe_unused]] std::size_t alignment) override {
    // Insert memory block back to the head of free list
    *static_cast<void**>(p) = free_list_;
    free_list_ = p;
  }

  // Resource equality comparison (only the same object is equal)
  [[nodiscard]] bool do_is_equal(
      const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

 private:
  // Chunk metadata
  struct chunk_info {
    void* ptr; // Memory block pointer
    std::size_t size; // Total size
    std::size_t alignment;
  };

  // Allocate a new memory chunk
  void allocate_chunk() {
    const std::size_t chunk_size = block_size_ * blocks_per_chunk_;

    // Allocate aligned memory through upstream resource
    void* chunk_ptr = upstream_->allocate(chunk_size, block_alignment_);

    // Record chunk information for later release
    chunks_.push_back({chunk_ptr, chunk_size, block_alignment_});

    // Initialize free list: link blocks in reverse order from chunk end to
    // beginning (improves locality)
    char* current = static_cast<char*>(chunk_ptr) + chunk_size;
    for (std::size_t i = 0; i < blocks_per_chunk_; ++i) {
      current -= block_size_;
      *reinterpret_cast<void**>(current) = free_list_;
      free_list_ = current;
    }
  }

  // Member variables
  const std::size_t block_size_; // Block size (not less than pointer size)
  const std::size_t block_alignment_; // Block alignment requirement
  const std::size_t blocks_per_chunk_; // Number of blocks per chunk
  std::pmr::memory_resource* upstream_; // Upstream memory resource
  std::pmr::vector<chunk_info> chunks_{1024}; // Records of all allocated chunks
  void* free_list_ = nullptr; // Free block list head pointer
};
} // namespace kv_mem
