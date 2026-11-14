/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>

#include "fbgemm_gpu/layout_transform_ops.cuh"
#include "fbgemm_gpu/utils/dispatch_macros.h"

namespace {

class RecatCopyKernelTest
    : public testing::TestWithParam<std::tuple<at::ScalarType, int>> {};

TEST_P(RecatCopyKernelTest, MixedDimensionsAndTail) {
  const auto [dtype, columns] = GetParam();
  const auto options = at::TensorOptions().device(at::kCUDA).dtype(dtype);
  const std::vector<std::vector<int64_t>> dimensions = {
      {1}, {2}, {3}, {7}, {65}, {4, 7, 0, 3}};
  for (const auto& dims : dimensions) {
    int64_t width = 0;
    std::vector<int64_t> offsets;
    for (const auto dim : dims) {
      offsets.push_back(width);
      width += dim;
    }
    const auto sizes = at::tensor(dims, at::kLong).cuda();
    const auto starts = at::tensor(offsets, at::kLong).cuda();
    for (const int64_t batches : {1, 3}) {
      for (const int64_t offset : {0, 1}) {
        const auto input = at::arange(batches * width + offset, options)
                               .slice(0, offset)
                               .view({batches, width});
        auto storage = at::full({batches * width + 2}, -1, options);
        auto output = storage.slice(0, offset, offset + batches * width);
        const auto stream = at::cuda::getCurrentCUDAStream();
        FBGEMM_DISPATCH_FLOAT_AND_HALF(dtype, "recat_copy_test", [&] {
          recat_copy_async_kernel<scalar_t><<<1, dim3(columns, 2), 0, stream>>>(
              sizes.data_ptr<int64_t>(),
              starts.data_ptr<int64_t>(),
              input.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              static_cast<int64_t>(dims.size()),
              batches,
              width);
        });
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        std::vector<at::Tensor> expected;
        for (const auto& part : input.cpu().split_with_sizes(dims, 1)) {
          expected.push_back(part.contiguous().view(-1));
        }
        ASSERT_TRUE(at::equal(output.cpu(), at::cat(expected)));
        ASSERT_EQ(storage[-1].item<float>(), -1);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    LaunchShapes,
    RecatCopyKernelTest,
    testing::Combine(
        testing::Values(at::kFloat, at::kHalf),
        testing::Values(1, 2, 4, 32, 64)));

} // namespace
