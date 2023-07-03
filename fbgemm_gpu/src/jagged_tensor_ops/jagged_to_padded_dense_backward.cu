/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

at::Tensor jagged_to_padded_dense_backward(
    const Tensor& grad_output,
    const std::vector<Tensor>& offsets,
    const at::SymInt& total_L) {
  auto grad_padded_values = grad_output;
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_padded_values.get_device());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = grad_padded_values.dim() == offsets.size() + 1;
  Tensor grad_padded_values_view =
      D_folded ? grad_padded_values.unsqueeze(-1) : grad_padded_values;
  int32_t D = grad_padded_values_view.size(-1);

  // Initialize with zeros so output will be zero for the portion truncated
  // in forward.
  auto grad_values =
      at::zeros_symint({total_L, D}, grad_padded_values.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_padded_values.scalar_type(),
      "jagged_to_dense_backward_kernel",
      [&] {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            grad_values, // dummy not used in the lambda function
            {offsets},
            grad_padded_values_view,
            grad_values,
            [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
              return y;
            });
      });

  return D_folded ? grad_values.squeeze(-1) : grad_values;
}

Tensor jagged_2d_to_dense_gpu_backward(
    Tensor grad_output,
    at::Tensor offsets,
    int64_t max_lengths) {
  return jagged_to_padded_dense_backward(grad_output, {offsets}, max_lengths);
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_to_padded_dense_backward",
    fbgemm_gpu::jagged_to_padded_dense_backward);
