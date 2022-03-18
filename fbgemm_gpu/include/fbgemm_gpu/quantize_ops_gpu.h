/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

namespace fbgemm_gpu {
at::Tensor _float_to_hfp8_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias,
    const double max_pos);

at::Tensor _hfp8_to_float_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t exponent_bias);
} // namespace fbgemm_gpu
