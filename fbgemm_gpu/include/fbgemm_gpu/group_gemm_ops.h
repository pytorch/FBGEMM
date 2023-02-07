/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {

/**
 * A helper function of group_linear_forward for checking if the inputs are
 * valid and preparing inputs that will be passed to gemm_grouped_cpu or
 * gemm_grouped_gpu
 *
 * @param input_group a vector of input tensors (2D-tensors)
 * @param weight_group a vector of weight tensors (2D-tensors)
 * @param bias_group a vector of bias tensors (1D-tensors)
 *
 * @returns a vector of transposed weight tensors (2D-tensors)
 */
std::vector<at::Tensor> group_linear_forward_helper(
    const std::vector<at::Tensor>& input_group,
    const std::vector<at::Tensor>& weight_group,
    const c10::optional<std::vector<at::Tensor>>& bias_group);

} // namespace fbgemm_gpu
