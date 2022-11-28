/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm_gpu {

using Tensor = at::Tensor;

///@ingroup jagged-tensor-ops-meta
Tensor jagged_to_padded_dense_forward_meta(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const std::vector<int64_t>& max_lengths,
    const double padding_value = 0) {
  const size_t num_jagged_dim = offsets.size();
  TORCH_CHECK(
      max_lengths.size() == num_jagged_dim,
      "max_lengths.size(), ",
      max_lengths.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  at::DimVector padded_values_shape({offsets[0].size(0) - 1});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());
  if (values.dim() > 1) {
    padded_values_shape.push_back(values.size(-1));
  }
  return at::empty(padded_values_shape, values.options());
}

Tensor jagged_to_padded_dense_backward_meta(
    const at::Tensor& grad_output,
    const std::vector<Tensor>& offsets,
    const int64_t total_L) {
  auto grad_padded_values = grad_output;

  int32_t D = grad_padded_values.size(-1);
  // Initialize with zeros so output will be zero for the portion truncated
  // in forward.
  auto grad_values = at::zeros({total_L, D}, grad_padded_values.options());

  TORCH_CHECK(grad_values.is_meta());
  return grad_values;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl(
      "jagged_to_padded_dense_forward",
      TORCH_FN(fbgemm_gpu::jagged_to_padded_dense_forward_meta));
  m.impl(
      "jagged_to_padded_dense_backward",
      TORCH_FN(fbgemm_gpu::jagged_to_padded_dense_backward_meta));
}
