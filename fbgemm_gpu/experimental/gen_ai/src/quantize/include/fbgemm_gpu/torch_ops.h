/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// For FBGEMM ops being exposed into torch core add them here.
// The reason is that we need to only expose the declarations we will build,
// otherwise we will have undefined symbols during linking.

#pragma once

#include <ATen/core/Tensor.h>

namespace fbgemm_gpu {

#ifdef USE_ROCM

// Generic PyTorch grouped GEMM API is only available on AMD for now.
at::Tensor f8f8bf16_rowwise_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> offsets,
    at::Tensor& output);

#else

// Torch compliant MXFP8 grouped GEMM only on CUDA for now.
at::Tensor mx8mx8bf16_grouped_mm(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor offsets,
    std::optional<at::Tensor> output = std::nullopt);

#endif

} // namespace fbgemm_gpu
