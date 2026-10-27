/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

namespace fbgemm_gpu {

// Internal to merge_pooled_embedding_ops_gpu.cpp; declared here so that its
// tests can call it directly.
//
// `copy_stream` must be a stream on the source device.
void copy_to_device_async(
    const at::Tensor& src,
    at::Tensor& dst,
    const at::cuda::CUDAStream& copy_stream);

} // namespace fbgemm_gpu
