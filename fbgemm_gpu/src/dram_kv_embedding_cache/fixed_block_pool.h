#pragma once

#include <chrono>
#include <cstddef>
#include <memory_resource>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <numeric>

#include <cassert>

namespace kv_mem {
static constexpr uint32_t kMaxInt31Counter = 2147483647;

class FixedBlockPool : public std::pmr::memory_resource {
 public:
  // Chunk metadata
  struct ChunkInfo {
    void* ptr;         // Memory block pointer
    std::size_t size;  // Total size
    std::size_t alignment;
  };

  // Metadata structure (publicly accessible)
  // alignas(8) MetaHeader >= sizeof(void*), avoid mempool block too small.
  struct alignas(8) MetaHeader {  // 16bytes
    int64_t key;                  // feature key 8bytes
    uint32_t timestamp;           // 4 bytes，the unit is second, uint32 indicates a range of over 120 years
    uint32_t count : 31;          // only 31 bit is used, max value is 2147483647
    bool used : 1;                // Mark whether this block is in use for the judgment of memory pool traversal
    // Can be extended with other fields: uint32_t click, etc.
  };

  // Metadata operations

  // Key operations
  static uint64_t get_key(const void* block) { return reinterpret_cast<const MetaHeader*>(block)->key; }
  static void set_key(void* block, uint64_t key) { reinterpret_cast<MetaHeader*>(block)->key = key; }

  // used operations
  static bool get_used(const void* block) { return reinterpret_cast<const MetaHeader*>(block)->used; }
  static void set_used(void* block, bool used) { reinterpret_cast<MetaHeader*>(block)->used = used; }

  // Score operations
  static uint32_t get_count(const void* block) { return reinterpret_cast<const MetaHeader*>(block)->count; }
  static void set_count(void* block, uint32_t count) { reinterpret_cast<MetaHeader*>(block)->count = count; }
  static void update_count(void* block) {
    // Avoid addition removal
    if (reinterpret_cast<MetaHeader*>(block)->count < kMaxInt31Counter) {
      reinterpret_cast<MetaHeader*>(block)->count++;
    }
  }
  // timestamp operations
  static uint32_t get_timestamp(const void* block) { return reinterpret_cast<const MetaHeader*>(block)->timestamp; }
  static void update_timestamp(void* block) { reinterpret_cast<MetaHeader*>(block)->timestamp = current_timestamp(); }
  static uint32_t current_timestamp() {
    return std::time(nullptr);
  }

  // Calculate storage size
  template <typename scalar_t>
  static size_t calculate_block_size(size_t dimension) {
    return sizeof(FixedBlockPool::MetaHeader) + dimension * sizeof(scalar_t);
  }

  // Calculate alignment requirements
  template <typename scalar_t>
  static size_t calculate_block_alignment() {
    return std::max(alignof(FixedBlockPool::MetaHeader), alignof(scalar_t));
  }

  // Data pointer retrieval
  template <typename scalar_t>
  static scalar_t* data_ptr(scalar_t* block) {
    return reinterpret_cast<scalar_t*>(reinterpret_cast<char*>(block) + sizeof(FixedBlockPool::MetaHeader));
  }

  template <typename scalar_t>
  static const scalar_t* data_ptr(const scalar_t* block) {
    return reinterpret_cast<const scalar_t*>(reinterpret_cast<const char*>(block) + sizeof(FixedBlockPool::MetaHeader));
  }

  template <typename scalar_t>
  static scalar_t get_l2weight(scalar_t* block, size_t dimension) {
    scalar_t* data = FixedBlockPool::data_ptr(block);
    return std::sqrt(
        std::accumulate(data, data + dimension, scalar_t(0),
                        [](scalar_t sum, scalar_t val) { return sum + val * val; }));
  }

  explicit FixedBlockPool(std::size_t block_size,               // Size of each memory block
                          std::size_t block_alignment,          // Memory block alignment requirement
                          std::size_t blocks_per_chunk = 8192,  // Number of blocks per chunk
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

  // Create memory block with metadata
  template <typename scalar_t>
  scalar_t* allocate_t() {
    return reinterpret_cast<scalar_t*>(this->allocate(block_size_, block_alignment_));
  }

  // Destroy memory block
  template <typename scalar_t>
  void deallocate_t(scalar_t* block) {
    this->deallocate(block, block_size_, block_alignment_);
  }

  template <typename scalar_t>
  scalar_t* get_block(size_t index) {
    char* current_chunk = static_cast<char*>(chunks_[index / blocks_per_chunk_].ptr);
    char* block = current_chunk + block_size_ * (index % blocks_per_chunk_);
    if (FixedBlockPool::get_used(block)) {
      return reinterpret_cast<scalar_t*>(block);
    } else {
      return nullptr;
    }
  };

  [[nodiscard]] const auto& get_chunks() const noexcept { return chunks_; }
  [[nodiscard]] std::size_t get_block_size() const noexcept { return block_size_; }
  [[nodiscard]] std::size_t get_block_alignment() const noexcept { return block_alignment_; }
  [[nodiscard]] std::size_t get_blocks_per_chunk() const noexcept { return blocks_per_chunk_; }

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
    FixedBlockPool::set_used(result, true);
    return result;
  }

  // Core deallocation function
  void do_deallocate(void* p, [[maybe_unused]] std::size_t bytes, [[maybe_unused]] std::size_t alignment) override {
    // Insert memory block back to the head of free list
    *static_cast<void**>(p) = free_list_;
    free_list_ = p;
    FixedBlockPool::set_used(free_list_, false);
  }

  // Resource equality comparison (only the same object is equal)
  [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }

 private:
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
      FixedBlockPool::set_used(current, false);
      free_list_ = current;
    }
  }

  // Member variables
  const std::size_t block_size_;              // Block size (not less than pointer size)
  const std::size_t block_alignment_;         // Block alignment requirement
  const std::size_t blocks_per_chunk_;        // Number of blocks per chunk
  std::pmr::memory_resource* upstream_;       // Upstream memory resource
  std::pmr::vector<ChunkInfo> chunks_{1024};  // Records of all allocated chunks
  void* free_list_ = nullptr;                 // Free block list head pointer
};
}  // namespace kv_mem
