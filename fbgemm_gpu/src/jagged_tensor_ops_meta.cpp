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

at::Tensor jagged_dense_dense_elementwise_add_jagged_output_forward_meta(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y_0,
    const at::Tensor& y_1) {
  TORCH_CHECK(y_0.sizes() == y_0.sizes());
  return at::empty_like(x_values);
}

Tensor jagged_dense_elementwise_add_forward_meta(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  return at::empty_like(x_values);
}

std::tuple<Tensor, Tensor> jagged_dense_elementwise_add_backward_meta(
    const Tensor& grad_output,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& x_values) {
  Tensor x_values_grad = at::empty_like(grad_output);
  Tensor y_grad = at::empty_like(y);

  return {x_values_grad, y_grad};
}

Tensor dense_to_jagged_forward_meta(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    const c10::optional<int64_t>& total_L) {
  auto dense_values = dense;
  int32_t D = dense_values.size(-1);
  TORCH_CHECK(total_L.has_value(), "total_L is required for meta backend");
  int64_t total_L_computed = total_L.value();
  auto values = at::zeros({total_L_computed, D}, dense_values.options());

  TORCH_CHECK(values.is_meta());
  return values;
}

Tensor jagged_dense_elementwise_mul_forward_meta(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  return at::empty_like(x_values);
}

std::tuple<Tensor, Tensor> jagged_dense_elementwise_mul_backward_meta(
    const Tensor& grad_output,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& x_values) {
  Tensor x_values_grad = at::empty_like(grad_output);
  Tensor y_grad = at::empty_like(y);

  return {x_values_grad, y_grad};
}

Tensor jagged_dense_elementwise_add_jagged_output_meta(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y_0,
    const at::Tensor& y_1) {
  return at::empty_like(x_values);
}

Tensor batched_dense_vec_jagged_2d_mul_forward_meta(
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  const int B = a_offsets.numel() - 1;
  TORCH_CHECK(
      B == 0 || v.size(0) % B == 0,
      "B, ",
      B,
      " doesn't divide v.size(0), ",
      v.size(0));
  const int H = B == 0 ? 1 : v.size(0) / B;
  const int D = a_values.size(-1) / H;
  return at::empty({B * H, D}, v.options());
}

std::tuple<Tensor, Tensor> batched_dense_vec_jagged_2d_mul_backward_meta(
    const Tensor& grad_output,
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  Tensor a_values_grad = at::zeros_like(a_values);
  Tensor v_grad = at::empty_like(v);
  return {v_grad, a_values_grad};
}

Tensor jagged_dense_bmm_forward_meta(
    const Tensor& x_values,
    const Tensor& x_offsets,
    const Tensor& y,
    const int64_t max_L) {
  const int N = y.size(-1);
  const int total_L = x_values.size(0);
  return at::zeros({total_L, N}, x_values.options());
}

Tensor jagged_softmax_forward_meta(
    const Tensor& values,
    const Tensor& offsets,
    const int64_t max_L) {
  return at::empty_like(values);
}

Tensor jagged_jagged_bmm_forward_meta(
    const Tensor& x_values,
    const Tensor& y_values,
    const Tensor& offsets,
    const int64_t max_L) {
  const int B = offsets.size(0) - 1;
  const int M = x_values.size(-1);
  const int N = y_values.size(-1);
  auto output = at::zeros({B, M, N}, x_values.options());
  return output;
}

Tensor jagged_softmax_backward_meta(
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& offsets,
    const int64_t max_L) {
  return at::empty_like(grad_output);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl(
      "jagged_to_padded_dense_forward",
      TORCH_FN(fbgemm_gpu::jagged_to_padded_dense_forward_meta));
  m.impl(
      "jagged_to_padded_dense_backward",
      TORCH_FN(fbgemm_gpu::jagged_to_padded_dense_backward_meta));
  m.impl(
      "dense_to_jagged_forward",
      TORCH_FN(fbgemm_gpu::dense_to_jagged_forward_meta));
  m.impl(
      "jagged_dense_dense_elementwise_add_jagged_output_forward",
      TORCH_FN(
          fbgemm_gpu::
              jagged_dense_dense_elementwise_add_jagged_output_forward_meta));
  m.impl(
      "jagged_dense_elementwise_add_forward",
      TORCH_FN(fbgemm_gpu::jagged_dense_elementwise_add_forward_meta));
  m.impl(
      "jagged_dense_elementwise_add_backward",
      TORCH_FN(fbgemm_gpu::jagged_dense_elementwise_add_backward_meta));
  m.impl(
      "jagged_dense_elementwise_mul_forward",
      TORCH_FN(fbgemm_gpu::jagged_dense_elementwise_mul_forward_meta));
  m.impl(
      "jagged_dense_elementwise_mul_backward",
      TORCH_FN(fbgemm_gpu::jagged_dense_elementwise_mul_backward_meta));
  m.impl(
      "batched_dense_vec_jagged_2d_mul_forward",
      TORCH_FN(fbgemm_gpu::batched_dense_vec_jagged_2d_mul_forward_meta));
  m.impl(
      "batched_dense_vec_jagged_2d_mul_backward",
      TORCH_FN(fbgemm_gpu::batched_dense_vec_jagged_2d_mul_backward_meta));
  m.impl(
      "jagged_dense_bmm_forward",
      TORCH_FN(fbgemm_gpu::jagged_dense_bmm_forward_meta));
  m.impl(
      "jagged_softmax_forward",
      TORCH_FN(fbgemm_gpu::jagged_softmax_forward_meta));
  m.impl(
      "jagged_jagged_bmm_forward",
      TORCH_FN(fbgemm_gpu::jagged_jagged_bmm_forward_meta));
  m.impl(
      "jagged_softmax_backward",
      TORCH_FN(fbgemm_gpu::jagged_softmax_backward_meta));
  m.impl(
      "jagged_jagged_bmm",
      TORCH_FN(fbgemm_gpu::jagged_jagged_bmm_forward_meta));
}
