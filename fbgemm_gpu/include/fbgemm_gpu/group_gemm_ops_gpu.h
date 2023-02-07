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
 * Compute N GEMMs where each GEMM can have different sizes. The i'th GEMM
 * performs matrix multiplication between a_group[i] and b_group[i] tensors
 * (each of them is a 2D-tensor). If c_group is passed to the function,
 * c_group[i] will be added to the product of a_group[i] @ b_group[i] (@ =
 * matrix multiplication).  The function returns a vector of output tensors.
 * The i'th GEMM product is stored in the i'th tensor in the output vector.
 *
 * Suppose each GEMM is A * B + C:
 *
 * @param a_group a vector of A tensors (2D-tensors)
 * @param b_group a vector of B tensors (2D-tensors)
 * @param c_group an optional vector of C tensors (1D-tensors or 2D-tensors;
 *                1D-tensors will be automatically broadcast)
 */
template <typename scalar_t, typename LayoutB, typename ArchTag>
std::vector<at::Tensor> gemm_grouped_cuda(
    const std::vector<at::Tensor>& a_group,
    const std::vector<at::Tensor>& b_group,
    const c10::optional<std::vector<at::Tensor>>& c_group);

} // namespace fbgemm_gpu
