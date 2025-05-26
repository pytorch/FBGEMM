#pragma once

#include <chrono>
#include <cstddef>
#include <memory_resource>
#include <stdexcept>
#include <vector>

#include <cassert>

namespace kv_mem {
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
  struct alignas(8) MetaHeader {
    uint64_t key;   // 8 bytes
    int32_t score;  // 4 bytes
    bool used;      // 1 byte
  };

  // Metadata operations

  // Key operations
  static uint64_t get_key(const void* block) {
    return reinterpret_cast<const MetaHeader*>(block)->key;
  }
  static void set_key(void* block, uint64_t key) {
    reinterpret_cast<MetaHeader*>(block)->key = key;
  }

  // used operations
  static bool get_used(const void* block) {
    return reinterpret_cast<const MetaHeader*>(block)->used;
  }
  static void set_used(void* block, bool used) {
    reinterpret_cast<MetaHeader*>(block)->used = used;
  }

  // Score operations
  static int32_t get_score(const void* block) {
    return reinterpret_cast<const MetaHeader*>(block)->score;
  }
  static void set_score(void* block, int32_t score) {
    reinterpret_cast<MetaHeader*>(block)->score = score;
  }
  static void update_score(void* block) {
    auto& score = reinterpret_cast<MetaHeader*>(block)->score;
    // Avoid addition removal
    if (score < std::numeric_limits<int32_t>::max()) {
      score++;
    }
  }
  // timestamp operations
  static void update_timestamp(void* block) {
    reinterpret_cast<MetaHeader*>(block)->score = current_timestamp();
  }
  static int32_t current_timestamp() {
    auto stamp = std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
    return static_cast<int32_t>(stamp);
    // facebook::WallClockUtil::NowInUsecFast();
  }

  // 与类型有关
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
    return reinterpret_cast<scalar_t*>(reinterpret_cast<char*>(block) +
                                       sizeof(FixedBlockPool::MetaHeader));
  }

  template <typename scalar_t>
  static const scalar_t* data_ptr(const scalar_t* block) {
    return reinterpret_cast<const scalar_t*>(
        reinterpret_cast<const char*>(block) +
        sizeof(FixedBlockPool::MetaHeader));
  }

  // Create memory block with metadata
  template <typename scalar_t>
  static scalar_t* allocate_t(size_t& block_size,
                              size_t& alignment,
                              FixedBlockPool* pool) {
    auto* block =
        reinterpret_cast<scalar_t*>(pool->allocate(block_size, alignment));
    return block;
  }

  // Destroy memory block
  template <typename scalar_t>
  static void deallocate_t(scalar_t* block,
                           size_t& block_size,
                           size_t& alignment,
                           FixedBlockPool* pool) {
    pool->deallocate(block, block_size, alignment);
  }

  // 使用示例
  template <typename scalar_t>
  static void get_keys_with_low_score(FixedBlockPool* pool,
                                      int32_t threshold,
                                      float decay,
                                      std::vector<uint64_t>& result) {
    pool->for_each_block([&decay, &threshold, &result](void* block) {
      if (FixedBlockPool::get_used(block)) {
        auto score = FixedBlockPool::get_score(static_cast<scalar_t*>(block));
        score = score * decay;
        FixedBlockPool::set_score(static_cast<scalar_t*>(block), score);
        if (score < threshold) {
          result.push_back(
              FixedBlockPool::get_key(static_cast<scalar_t*>(block)));
        }
      }
    });
  }

  explicit FixedBlockPool(
      std::size_t block_size,       // Size of each memory block
      std::size_t block_alignment,  // Memory block alignment requirement
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

  // 新增获取chunks信息的接口
  [[nodiscard]] const auto& get_chunks() const noexcept { return chunks_; }

  // 新增遍历所有block的接口
  template <typename Func>
  void for_each_block(Func&& func) const {
    for (const auto& chunk : chunks_) {
      char* current = static_cast<char*>(chunk.ptr);
      for (size_t i = 0; i < blocks_per_chunk_; ++i) {
        func(current);
        current += block_size_;
      }
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
    FixedBlockPool::set_used(result, true);
    return result;
  }

  // Core deallocation function
  void do_deallocate(void* p,
                     [[maybe_unused]] std::size_t bytes,
                     [[maybe_unused]] std::size_t alignment) override {
    // Insert memory block back to the head of free list
    *static_cast<void**>(p) = free_list_;
    free_list_ = p;
    FixedBlockPool::set_used(free_list_, false);
  }

  // Resource equality comparison (only the same object is equal)
  [[nodiscard]] bool do_is_equal(
      const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

 private:
  // Allocate a new memory chunk
  void allocate_chunk() {
    const std::size_t chunk_size = block_size_ * blocks_per_chunk_;

    // Allocate aligned memory through upstream resource
    void* chunk_ptr = upstream_->allocate(chunk_size, block_alignment_);

    // Block used flag set false.
    FixedBlockPool::set_used(chunk_ptr, false);

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
  const std::size_t block_size_;  // Block size (not less than pointer size)
  const std::size_t block_alignment_;         // Block alignment requirement
  const std::size_t blocks_per_chunk_;        // Number of blocks per chunk
  std::pmr::memory_resource* upstream_;       // Upstream memory resource
  std::pmr::vector<ChunkInfo> chunks_{1024};  // Records of all allocated chunks
  void* free_list_ = nullptr;                 // Free block list head pointer
};
}  // namespace kv_mem
