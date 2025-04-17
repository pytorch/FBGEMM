/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace fbgemm_gpu {
std::vector<at::Tensor> coalesce_batches_cpu(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& output,
    const at::Tensor& old_bids,
    const at::Tensor& new_bids);

std::vector<at::Tensor> coalesce_batches_gpu(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& output,
    const at::Tensor& old_bids,
    const at::Tensor& new_bids);

} // namespace fbgemm_gpu
