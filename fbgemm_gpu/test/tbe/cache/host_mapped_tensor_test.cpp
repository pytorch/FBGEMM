/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>

#include "fbgemm_gpu/cumem_utils.h"

using namespace ::testing;

namespace {

std::atomic<int> g_alloc_count{0};
std::atomic<int> g_dealloc_count{0};

void* trackingAllocate(size_t size) {
  void* ptr = std::malloc(size); // NOLINT(cppcoreguidelines-no-malloc)
  g_alloc_count.fetch_add(1);
  return ptr;
}

void trackingDeallocate(void* ptr) {
  g_dealloc_count.fetch_add(1);
  std::free(ptr); // NOLINT(cppcoreguidelines-no-malloc)
}

void resetCounters() {
  g_alloc_count.store(0);
  g_dealloc_count.store(0);
}

} // namespace

TEST(HostMappedTensorTest, CustomAllocatorIsUsed) {
  resetCounters();

  auto self = at::empty({0}, at::device(at::kCUDA).dtype(at::kByte));
  std::vector<int64_t> sizes = {1024};

  {
    auto tensor = fbgemm_gpu::new_host_mapped_tensor_with_allocator(
        self, sizes, &trackingAllocate, &trackingDeallocate);

    EXPECT_EQ(g_alloc_count.load(), 1)
        << "Custom allocator should have been called exactly once";
    EXPECT_EQ(tensor.numel(), 1024);
  }

  EXPECT_EQ(g_dealloc_count.load(), 1)
      << "Custom deallocator should have been called on tensor destruction";
}

TEST(HostMappedTensorTest, DefaultAllocatorWhenNull) {
  auto self = at::empty({0}, at::device(at::kCUDA).dtype(at::kByte));
  std::vector<int64_t> sizes = {512};

  auto tensor = fbgemm_gpu::new_host_mapped_tensor_with_allocator(
      self, sizes, nullptr, nullptr);

  EXPECT_EQ(tensor.numel(), 512);
}

TEST(HostMappedTensorTest, RejectsMismatchedAllocatorPair) {
  auto self = at::empty({0}, at::device(at::kCUDA).dtype(at::kByte));
  std::vector<int64_t> sizes = {512};

  EXPECT_THROW(
      fbgemm_gpu::new_host_mapped_tensor_with_allocator(
          self, sizes, &trackingAllocate, nullptr),
      c10::Error);
  EXPECT_THROW(
      fbgemm_gpu::new_host_mapped_tensor_with_allocator(
          self, sizes, nullptr, &trackingDeallocate),
      c10::Error);
}
