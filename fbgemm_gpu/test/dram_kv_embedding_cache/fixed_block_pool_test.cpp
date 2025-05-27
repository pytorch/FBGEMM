#include "fbgemm_gpu/src/dram_kv_embedding_cache/fixed_block_pool.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace kv_mem {

double test_std_vector(size_t vector_size, size_t repeat_count) {
  float sum = 0.0f;  // Prevent optimization
  std::vector<std::vector<float>>
      all_vectors;  // Store all vectors to prevent release
  all_vectors.reserve(repeat_count);

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < repeat_count; ++i) {
    all_vectors.emplace_back(vector_size);
    auto& vec = all_vectors.back();

    for (size_t j = 0; j < vector_size; ++j) {
      vec[j] = static_cast<float>(j);
    }

    // Simple usage to prevent optimization
    sum += vec[0];
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// Testing memory pool allocation
double test_pool_vector(size_t vector_size, size_t repeat_count) {
  // Create a memory pool large enough
  FixedBlockPool pool(vector_size * sizeof(float), alignof(float), 8092);
  std::pmr::polymorphic_allocator<float> alloc(&pool);

  auto start = std::chrono::high_resolution_clock::now();
  float sum = 0.0f;  // Prevent optimization
  for (size_t i = 0; i < repeat_count; ++i) {
    float* arr = alloc.allocate(vector_size);

    for (size_t j = 0; j < vector_size; ++j) {
      arr[j] = static_cast<float>(j);
    }

    // Simple usage to prevent optimization
    sum += arr[0];

    // Removed deallocate statement, no longer releasing memory to avoid memory
    // reuse
    //  alloc.deallocate(arr, dim);
  }

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void benchmark_memory_allocators() {
  std::cout << "====== Testing performance difference between memory pool and "
               "native vector allocation for 10 million "
               "times ======"
            << std::endl;

  // Vector sizes to test (in number of float elements)
  std::vector<size_t> vector_sizes = {4, 8, 16, 32, 64, 128, 256};

  // Repeat count (10 million times)
  const size_t repeat_count = 10'000'000;

  for (const auto& size : vector_sizes) {
    std::cout << "Vector size: " << size << " floats ("
              << (size * sizeof(float)) << " bytes)" << std::endl;

    // Testing standard vector
    double std_time = test_std_vector(size, repeat_count);
    std::cout << "  Standard vector: " << std::fixed << std::setprecision(2)
              << std_time << " ms" << std::endl;

    // Testing memory pool
    double pool_time = test_pool_vector(size, repeat_count);
    std::cout << "  Memory pool: " << std::fixed << std::setprecision(2)
              << pool_time << " ms" << std::endl;

    // Calculate speed improvement
    double speedup = std_time / pool_time;
    std::cout << "  Speed improvement: " << std::fixed << std::setprecision(2)
              << speedup << "x" << std::endl;

    std::cout << std::endl;
    std::cout << "============================" << std::endl;
  }
}

// Basic functionality test: Integer keys
TEST(FixedBlockPoolTest, benchmark_memory_allocators) {
  benchmark_memory_allocators();
}

// Test constructor normal case
TEST(FixedBlockPoolTest, ConstructorNormal) {
  EXPECT_NO_THROW({ kv_mem::FixedBlockPool pool(16, 8); });
}

// Test constructor exception cases
TEST(FixedBlockPoolTest, ConstructorExceptions) {
  // Block size smaller than pointer size
  EXPECT_THROW({ kv_mem::FixedBlockPool pool(1, 1); }, std::invalid_argument);

  // Alignment not a power of 2
  EXPECT_THROW({ kv_mem::FixedBlockPool pool(16, 3); }, std::invalid_argument);

  // Block size not a multiple of alignment
  EXPECT_THROW({ kv_mem::FixedBlockPool pool(10, 8); }, std::invalid_argument);
}

// Test basic memory allocation and deallocation
TEST(FixedBlockPoolTest, BasicAllocation) {
  const size_t block_size = 16;
  const size_t alignment = 8;
  kv_mem::FixedBlockPool pool(block_size, alignment);

  void* p = pool.allocate(block_size, alignment);
  EXPECT_NE(p, nullptr);

  // Verify allocated memory is usable
  std::memset(p, 0xAB, block_size);

  pool.deallocate(p, block_size, alignment);
}

// Test multiple allocations and deallocations
TEST(FixedBlockPoolTest, MultipleAllocations) {
  const size_t block_size = 32;
  const size_t alignment = 8;
  kv_mem::FixedBlockPool pool(block_size, alignment);

  std::vector<void*> blocks;
  const int NUM_BLOCKS = 100;

  // Allocate multiple blocks
  for (int i = 0; i < NUM_BLOCKS; ++i) {
    void* p = pool.allocate(block_size, alignment);
    EXPECT_NE(p, nullptr);
    // Write some data
    *static_cast<int*>(p) = i;
    blocks.push_back(p);
  }

  // Verify data
  for (int i = 0; i < NUM_BLOCKS; ++i) {
    EXPECT_EQ(*static_cast<int*>(blocks[i]), i);
  }

  // Release all blocks
  for (auto p : blocks) {
    pool.deallocate(p, block_size, alignment);
  }
}

// Test cross-chunk allocation (each chunk has only 10 blocks)
TEST(FixedBlockPoolTest, CrossChunkAllocation) {
  const size_t block_size = 16;
  const size_t alignment = 8;
  const size_t blocks_per_chunk = 10;
  kv_mem::FixedBlockPool pool(block_size, alignment, blocks_per_chunk);

  std::vector<void*> blocks;
  const int NUM_BLOCKS = 25;  // Exceeds 2 chunks

  // Allocate blocks beyond a single chunk capacity
  for (int i = 0; i < NUM_BLOCKS; ++i) {
    void* p = pool.allocate(block_size, alignment);
    EXPECT_NE(p, nullptr);
    blocks.push_back(p);
  }

  // Release all blocks
  for (auto p : blocks) {
    pool.deallocate(p, block_size, alignment);
  }
}

// Test memory alignment
TEST(FixedBlockPoolTest, MemoryAlignment) {
  const size_t block_size = 64;
  const size_t alignment = 32;
  kv_mem::FixedBlockPool pool(block_size, alignment);

  void* p = pool.allocate(block_size, alignment);
  EXPECT_NE(p, nullptr);

  // Verify address is aligned to specified alignment
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  EXPECT_EQ(addr % alignment, 0);

  pool.deallocate(p, block_size, alignment);
}

// Test error handling - allocating blocks with mismatched size or alignment
TEST(FixedBlockPoolTest, ErrorHandling) {
  const size_t block_size = 16;
  const size_t alignment = 8;
  kv_mem::FixedBlockPool pool(block_size, alignment);

  // Try to allocate memory with incorrect size
  EXPECT_THROW(
      { [[maybe_unused]] void* p = pool.allocate(block_size * 2, alignment); },
      std::bad_alloc);

  // Try to allocate memory with incorrect alignment
  EXPECT_THROW(
      { [[maybe_unused]] void* p = pool.allocate(block_size, alignment * 2); },
      std::bad_alloc);
}

// Test memory reuse after deallocation
TEST(FixedBlockPoolTest, ReuseAfterDeallocation) {
  const size_t block_size = 16;
  const size_t alignment = 8;
  kv_mem::FixedBlockPool pool(block_size, alignment);

  void* p1 = pool.allocate(block_size, alignment);
  void* p2 = pool.allocate(block_size, alignment);

  // Release the first block
  pool.deallocate(p1, block_size, alignment);

  // Reallocate, should get the recently freed block (due to LIFO order)
  void* p3 = pool.allocate(block_size, alignment);
  EXPECT_EQ(p3, p1);

  // Cleanup
  pool.deallocate(p2, block_size, alignment);
  pool.deallocate(p3, block_size, alignment);
}

// Test custom upstream memory resource
TEST(FixedBlockPoolTest, CustomUpstreamResource) {
  const size_t block_size = 16;
  const size_t alignment = 8;

  // Use custom memory resource that tracks allocations
  int allocate_count = 0;
  int deallocate_count = 0;

  class CountingResource : public std::pmr::memory_resource {
   public:
    CountingResource(int& alloc_count, int& dealloc_count)
        : alloc_count_(alloc_count), dealloc_count_(dealloc_count) {}

   protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
      ++alloc_count_;
      return std::pmr::new_delete_resource()->allocate(bytes, alignment);
    }

    void do_deallocate(void* p, size_t bytes, size_t alignment) override {
      ++dealloc_count_;
      std::pmr::new_delete_resource()->deallocate(p, bytes, alignment);
    }

    bool do_is_equal(
        const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }

   private:
    int& alloc_count_;
    int& dealloc_count_;
  };

  CountingResource upstream(allocate_count, deallocate_count);
  {
    kv_mem::FixedBlockPool pool(block_size, alignment, 1024, &upstream);

    // Allocate some blocks to trigger chunk allocation
    std::vector<void*> blocks;
    for (int i = 0; i < 10; ++i) {
      blocks.push_back(pool.allocate(block_size, alignment));
    }

    // Verify upstream resource was called
    EXPECT_GT(allocate_count, 0);
    EXPECT_EQ(deallocate_count, 0);

    // Release all blocks
    for (auto p : blocks) {
      pool.deallocate(p, block_size, alignment);
    }
  }
  // Destructor should release all chunks
  EXPECT_GT(deallocate_count, 0);
}

TEST(FixedBlockPool, BasicFunctionality) {
  constexpr int dim = 4;
  size_t block_size = FixedBlockPool ::calculate_block_size<float>(dim);
  size_t alignment = FixedBlockPool::calculate_block_alignment<float>();

  // Initialize memory pool
  FixedBlockPool pool(block_size, alignment, 1024);

  // Test memory allocation
  auto* block = pool.allocate_t<float>();
  FixedBlockPool::update_timestamp(block);
  ASSERT_NE(block, nullptr);

  // Verify metadata header
  int64_t ts1 = FixedBlockPool::get_timestamp(block);
  EXPECT_LE(FixedBlockPool::current_timestamp(), ts1);

  // Test data pointer offset
  float* data = FixedBlockPool::data_ptr<float>(block);
  ASSERT_EQ(reinterpret_cast<char*>(data) - reinterpret_cast<char*>(block),
            sizeof(FixedBlockPool::MetaHeader));

  // Test timestamp update
  FixedBlockPool::update_timestamp(block);
  int64_t ts2 = FixedBlockPool::get_timestamp(block);
  EXPECT_GE(ts2, ts1);  // New timestamp should be greater or equal

  // Test memory deallocation
  EXPECT_NO_THROW(pool.deallocate_t<float>(block));
}

TEST(FixedBlockPool, MultiDimensionTest) {
  // Test memory alignment for different dimensions
  const std::vector<int> test_dims = {1, 4, 16, 64, 256};
  for (int dim : test_dims) {
    size_t block_size = FixedBlockPool::calculate_block_size<float>(dim);
    size_t alignment = FixedBlockPool::calculate_block_alignment<float>();

    // Verify alignment requirements
    EXPECT_EQ(alignment % alignof(FixedBlockPool::MetaHeader), 0);
    EXPECT_EQ(alignment % alignof(float), 0);

    // Verify block size calculation
    const size_t expected_size =
        sizeof(FixedBlockPool::MetaHeader) + dim * sizeof(float);
    EXPECT_EQ(block_size, expected_size);
  }
}

TEST(FixedBlockPool, TimestampPrecision) {
  // Test timestamp precision accuracy
  constexpr int test_iterations = 1000;
  int64_t prev_ts = FixedBlockPool::current_timestamp();

  for (int i = 0; i < test_iterations; ++i) {
    int64_t curr_ts = FixedBlockPool::current_timestamp();
    EXPECT_GE(curr_ts,
              prev_ts);  // Timestamps should be monotonically increasing
    prev_ts = curr_ts;
  }
}

TEST(FixedBlockPool, DataIntegrity) {
  // Test data storage integrity
  constexpr int dim = 8;
  std::vector<float> src_data(dim, 3.14f);

  size_t block_size = FixedBlockPool::calculate_block_size<float>(dim);
  size_t alignment = FixedBlockPool::calculate_block_alignment<float>();
  FixedBlockPool pool(block_size, alignment, 1024);

  // Allocate and write data
  auto* block = pool.allocate_t<float>();
  auto* data_ptr = FixedBlockPool::data_ptr<float>(block);
  std::copy(src_data.begin(), src_data.end(), data_ptr);

  // Verify data consistency
  for (int i = 0; i < dim; ++i) {
    EXPECT_FLOAT_EQ(data_ptr[i], src_data[i]);
  }
  pool.deallocate_t<float>(block);
}

}  // namespace kv_mem