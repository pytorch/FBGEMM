/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace at {

// Return array of size T_in.numel(), representing incomplete exclusive cumsum
Tensor asynchronous_exclusive_cumsum(const Tensor& t_in);

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights);

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_sparse_data_cpu(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights);

at::Tensor _float_to_fused8bitrowwise_gpu(const at::Tensor& input);
at::Tensor _fused8bitrowwise_to_float_gpu(const at::Tensor& input);

} // namespace at
