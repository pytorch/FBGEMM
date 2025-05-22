#include "fbgemm_gpu/src/dram_kv_embedding_cache/store_value_utils.h"

#include "gtest/gtest.h"
namespace kv_mem {

TEST(StoreValueUtils, BasicFunctionality) {
  constexpr int dim = 4;
  size_t block_size = StoreValueUtils::calculate_block_size<float>(dim);
  size_t alignment = StoreValueUtils::calculate_block_alignment<float>();

  // Initialize memory pool
  FixedBlockPool pool(block_size, alignment, 1024);

  // Test memory allocation
  float* block = StoreValueUtils::allocate<float>(block_size, alignment, &pool);
  StoreValueUtils::update_timestamp(block);
  ASSERT_NE(block, nullptr);

  // Verify metadata header
  int64_t ts1 = StoreValueUtils::get_timestamp<float>(block);
  EXPECT_LE(StoreValueUtils::current_timestamp(), ts1);

  // Test data pointer offset
  float* data = StoreValueUtils::data_ptr<float>(block);
  ASSERT_EQ(reinterpret_cast<char*>(data) - reinterpret_cast<char*>(block), sizeof(StoreValueUtils::MetaHeader));

  // Test timestamp update
  StoreValueUtils::update_timestamp<float>(block);
  int64_t ts2 = StoreValueUtils::get_timestamp<float>(block);
  EXPECT_GE(ts2, ts1);  // New timestamp should be greater or equal

  // Test memory deallocation
  EXPECT_NO_THROW(StoreValueUtils::deallocate<float>(block, block_size, alignment, &pool));
}

TEST(StoreValueUtils, MultiDimensionTest) {
  // Test memory alignment for different dimensions
  const std::vector<int> test_dims = {1, 4, 16, 64, 256};
  for (int dim : test_dims) {
    size_t block_size = StoreValueUtils::calculate_block_size<float>(dim);
    size_t alignment = StoreValueUtils::calculate_block_alignment<float>();

    // Verify alignment requirements
    EXPECT_EQ(alignment % alignof(StoreValueUtils::MetaHeader), 0);
    EXPECT_EQ(alignment % alignof(float), 0);

    // Verify block size calculation
    const size_t expected_size = sizeof(StoreValueUtils::MetaHeader) + dim * sizeof(float);
    EXPECT_EQ(block_size, expected_size);
  }
}

TEST(StoreValueUtils, TimestampPrecision) {
  // Test timestamp precision accuracy
  constexpr int test_iterations = 1000;
  int64_t prev_ts = StoreValueUtils::current_timestamp();

  for (int i = 0; i < test_iterations; ++i) {
    int64_t curr_ts = StoreValueUtils::current_timestamp();
    EXPECT_GE(curr_ts, prev_ts);  // Timestamps should be monotonically increasing
    prev_ts = curr_ts;
  }
}

TEST(StoreValueUtils, DataIntegrity) {
  // Test data storage integrity
  constexpr int dim = 8;
  std::vector<float> src_data(dim, 3.14f);

  size_t block_size = StoreValueUtils::calculate_block_size<float>(dim);
  size_t alignment = StoreValueUtils::calculate_block_alignment<float>();
  FixedBlockPool pool(block_size, alignment, 1024);

  // Allocate and write data
  float* block = StoreValueUtils::allocate<float>(block_size, alignment, &pool);
  float* data_ptr = StoreValueUtils::data_ptr<float>(block);
  std::copy(src_data.begin(), src_data.end(), data_ptr);

  // Verify data consistency
  for (int i = 0; i < dim; ++i) {
    EXPECT_FLOAT_EQ(data_ptr[i], src_data[i]);
  }

  StoreValueUtils::deallocate<float>(block, block_size, alignment, &pool);
}
}  // namespace kv_mem