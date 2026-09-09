/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "fbgemm_gpu/utils/cuda_utilities.cuh"

namespace fbgemm_gpu::utils::cuda {

static_assert(kMaxGridDimX == 2147483647);
static_assert(kMaxGridDimY == 65535);

TEST(CapGridDimensionTest, AppliesDriverLimits) {
  const auto stream = at::cuda::getCurrentCUDAStream();

  EXPECT_EQ(
      cap_grid_dim_x(int64_t{1} << 31, 1, stream, BlockCapPolicy::Never),
      static_cast<uint32_t>(kMaxGridDimX));
  EXPECT_EQ(
      cap_grid_dim_y_with_xz_blocks(
          kMaxGridDimY + 1, 1, 1, stream, BlockCapPolicy::Never),
      static_cast<uint32_t>(kMaxGridDimY));
  EXPECT_EQ(
      cap_grid_dim_x(
          std::numeric_limits<int64_t>::max(),
          1,
          stream,
          BlockCapPolicy::Never),
      static_cast<uint32_t>(kMaxGridDimX));
}

TEST(CapGridDimensionTest, FloorsNonpositiveBlockCounts) {
  const auto stream = at::cuda::getCurrentCUDAStream();

  EXPECT_EQ(cap_grid_dim_x(0, 1, stream, BlockCapPolicy::Never), uint32_t{1});
  EXPECT_EQ(cap_grid_dim_x(-1, 1, stream, BlockCapPolicy::Never), uint32_t{1});
}

TEST(CapGridDimensionTest, PreservesLegacyHandlingForNonpositiveMultipliers) {
  const auto stream = at::cuda::getCurrentCUDAStream();

  EXPECT_EQ(
      cap_grid_dim_x(1234, 0, stream, BlockCapPolicy::OverflowOnly),
      uint32_t{1234});
  EXPECT_EQ(
      cap_grid_dim_x(1234, -1, stream, BlockCapPolicy::OverflowOnly),
      uint32_t{1234});
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          1234, 1, 0, stream, BlockCapPolicy::OverflowOnly),
      uint32_t{1234});
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          1234, 1, -1, stream, BlockCapPolicy::OverflowOnly),
      uint32_t{1234});
}

TEST(CapGridDimensionTest, DerivesOneDimensionalGridX) {
  const auto stream = at::cuda::getCurrentCUDAStream();

  EXPECT_EQ(
      cap_grid_dim_x_from_workload(1025, 1024, stream, BlockCapPolicy::Never),
      uint32_t{2});
}

TEST(CapGridDimensionTest, AccountsForFixedGridPlane) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t kThreadsPerBlock = 1;
  constexpr int64_t kFixedGridPlane = 65537;
  constexpr int64_t kLastValidBlocks = 65535;
  constexpr int64_t kFirstInvalidBlocks = 65536;
  static_assert(
      kLastValidBlocks * kFixedGridPlane * kThreadsPerBlock ==
      kMaxThreadsPerLaunch);
  static_assert(
      kFirstInvalidBlocks * kFixedGridPlane * kThreadsPerBlock >
      kMaxThreadsPerLaunch);

#ifdef USE_ROCM
  const auto expected_overflow_cap = static_cast<uint32_t>(std::min<int64_t>(
      kLastValidBlocks,
      static_cast<int64_t>(MAX_THREAD_BLOCKS_FACTOR) *
          at::cuda::getDeviceProperties(stream.device_index())
              ->multiProcessorCount));
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          kLastValidBlocks, kThreadsPerBlock, kFixedGridPlane, stream),
      kLastValidBlocks);
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          kFirstInvalidBlocks, kThreadsPerBlock, kFixedGridPlane, stream),
      expected_overflow_cap);
  EXPECT_EQ(
      cap_grid_dim_y_with_xz_blocks(
          kFirstInvalidBlocks, kThreadsPerBlock, kFixedGridPlane, stream),
      expected_overflow_cap);
#else
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          kFirstInvalidBlocks, kThreadsPerBlock, kFixedGridPlane, stream),
      kFirstInvalidBlocks);
  EXPECT_EQ(
      cap_grid_dim_y_with_xz_blocks(
          kFirstInvalidBlocks, kThreadsPerBlock, kFixedGridPlane, stream),
      static_cast<uint32_t>(kMaxGridDimY));
#endif
}

TEST(CapGridDimensionTest, RejectsUnrepairableGridPlaneOnRocm) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t kThreadsPerBlock = 1024;
  constexpr int64_t kGridPlaneBlocks = 1LL << 22;

#ifdef USE_ROCM
  EXPECT_THROW(
      cap_grid_dim_x_with_yz_blocks(
          1, kThreadsPerBlock, kGridPlaneBlocks, stream),
      c10::Error);
  EXPECT_THROW(
      cap_grid_dim_x_with_yz_blocks(
          1, kThreadsPerBlock, kGridPlaneBlocks, stream, BlockCapPolicy::Never),
      c10::Error);
#else
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          1, kThreadsPerBlock, kGridPlaneBlocks, stream),
      uint32_t{1});
#endif
}

TEST(CapGridDimensionTest, NeverSkipsRepairableRocmCap) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t kThreadsPerBlock = 1;
  constexpr int64_t kFixedGridPlane = 65537;
  constexpr int64_t kFirstInvalidBlocks = 65536;

  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          kFirstInvalidBlocks,
          kThreadsPerBlock,
          kFixedGridPlane,
          stream,
          BlockCapPolicy::Never),
      kFirstInvalidBlocks);
}

TEST(CapGridDimensionTest, OverflowOnlyPreservesPerformanceCapOnRocm) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto performance_cap = static_cast<uint32_t>(
      MAX_THREAD_BLOCKS_FACTOR *
      at::cuda::getDeviceProperties(stream.device_index())
          ->multiProcessorCount);

#ifdef USE_ROCM
  EXPECT_EQ(cap_grid_dim_x(kMaxGridDimX, 1024, stream), performance_cap);
#else
  EXPECT_EQ(
      cap_grid_dim_x(kMaxGridDimX, 1024, stream),
      static_cast<uint32_t>(kMaxGridDimX));
#endif
}

TEST(CapGridDimensionTest, AlwaysAppliesPerformanceCap) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto performance_cap = static_cast<uint32_t>(
      MAX_THREAD_BLOCKS_FACTOR *
      at::cuda::getDeviceProperties(stream.device_index())
          ->multiProcessorCount);

  EXPECT_EQ(
      cap_grid_dim_x(kMaxGridDimX, 1, stream, BlockCapPolicy::Always),
      performance_cap);
}

TEST(CapGridDimensionTest, AlwaysAppliesRocmSafetyCap) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int64_t kThreadsPerBlock = 1024;
  constexpr int64_t kFixedGridPlane = int64_t{1} << 20;
  constexpr int64_t kRequestedBlocks = 4;

#ifdef USE_ROCM
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          kRequestedBlocks,
          kThreadsPerBlock,
          kFixedGridPlane,
          stream,
          BlockCapPolicy::Always),
      uint32_t{3});
#else
  EXPECT_EQ(
      cap_grid_dim_x_with_yz_blocks(
          kRequestedBlocks,
          kThreadsPerBlock,
          kFixedGridPlane,
          stream,
          BlockCapPolicy::Always),
      uint32_t{4});
#endif
}

} // namespace fbgemm_gpu::utils::cuda
