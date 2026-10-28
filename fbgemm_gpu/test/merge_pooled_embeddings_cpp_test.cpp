/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fbgemm_gpu/merge_pooled_embedding_ops_gpu.h"

using namespace testing;

TEST(MergePooledEmbeddingsTest, copy_stream_on_another_device_fails) {
  if (at::cuda::getNumGPUs() < 2) {
    GTEST_SKIP() << "Requires at least two devices to transfer between";
  }
  const auto source_device = at::Device(at::kCUDA, 0);
  const auto target_device = at::Device(at::kCUDA, 1);
  const auto src = at::ones({4, 8}, at::TensorOptions().device(source_device));
  auto dst = at::empty_like(src, src.options().device(target_device));
  const auto target_stream =
      at::cuda::getCurrentCUDAStream(target_device.index());

  EXPECT_THAT(
      [&] { fbgemm_gpu::copy_to_device_async(src, dst, target_stream); },
      Throws<c10::Error>(Property(
          &c10::Error::what,
          AllOf(
              HasSubstr("stream is on cuda:1"),
              HasSubstr("source tensor is on cuda:0")))));
}
