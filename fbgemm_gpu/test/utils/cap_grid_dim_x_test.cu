/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "fbgemm_gpu/utils/cuda_utilities.cuh"

namespace fbgemm_gpu::utils::cuda {

// The grid x-dimension launch limit is 2^31 - 1 on both CUDA and HIP.
static_assert(kMaxGridDimX == 2147483647, "grid.x max must be 2^31 - 1");

// BlockCapPolicy::Never exercises the final clamp in isolation, without
// touching device properties (#SMs) or the ROCm overflow threshold, so these
// assertions hold identically on CUDA and ROCm.

TEST(CapGridDimXTest, ClampsOutOfRangeCountsToGridXMax) {
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Just past the grid.x limit (2^31) is clamped down instead of passing
  // through as an invalid dimension.
  EXPECT_EQ(
      cap_grid_dim_x(
          int64_t{1} << 31,
          /*threads_per_block=*/1024,
          stream,
          BlockCapPolicy::Never),
      static_cast<uint32_t>(kMaxGridDimX));

  // uint32_t max (2^32 - 1): the exact value the launcher rejected before the
  // fix ("grid.x value 4294967295 is not within the range (0, 2147483647]").
  EXPECT_EQ(
      cap_grid_dim_x(
          std::numeric_limits<uint32_t>::max(),
          1024,
          stream,
          BlockCapPolicy::Never),
      static_cast<uint32_t>(kMaxGridDimX));

  // An arbitrarily large 64-bit count is still clamped into range (guards the
  // int64-overflow path that produced the wrapped dimension).
  EXPECT_EQ(
      cap_grid_dim_x(
          std::numeric_limits<int64_t>::max(),
          1024,
          stream,
          BlockCapPolicy::Never),
      static_cast<uint32_t>(kMaxGridDimX));
}

TEST(CapGridDimXTest, PassesThroughValidCounts) {
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Exactly at the limit is valid and returned unchanged.
  EXPECT_EQ(
      cap_grid_dim_x(kMaxGridDimX, 1024, stream, BlockCapPolicy::Never),
      static_cast<uint32_t>(kMaxGridDimX));

  // A normal small count is returned as-is.
  EXPECT_EQ(
      cap_grid_dim_x(1234, 1024, stream, BlockCapPolicy::Never),
      uint32_t{1234});
}

TEST(CapGridDimXTest, FloorsNonPositiveCountsToOne) {
  const auto stream = at::cuda::getCurrentCUDAStream();

  // grid.x must be >= 1; a zero or negative (e.g. int-overflowed) count is
  // floored to 1 rather than wrapping to a huge value via the uint32_t cast.
  EXPECT_EQ(
      cap_grid_dim_x(0, 1024, stream, BlockCapPolicy::Never), uint32_t{1});
  EXPECT_EQ(
      cap_grid_dim_x(-1, 1024, stream, BlockCapPolicy::Never), uint32_t{1});
}

} // namespace fbgemm_gpu::utils::cuda
