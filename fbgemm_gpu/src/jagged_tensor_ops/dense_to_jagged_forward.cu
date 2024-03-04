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

Tensor dense_to_jagged_forward(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    c10::optional<at::SymInt> total_L) {
  // D is the embedding dimension
  auto D = dense.size(-1);

  // If total_L is not given then compute it
  at::SymInt total_L_computed;
  if (total_L.has_value()) {
    total_L_computed = total_L.value();
  } else {
    total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
  }
  auto values = at::empty_symint({total_L_computed, D}, dense.options());
  auto output = at::empty_like(values);

  CUDA_DEVICE_GUARD(dense);

  AT_DISPATCH_SWITCH(
      values.scalar_type(),
      "dense_to_jagged_gpu_op_forward",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&] {
            jagged_dense_elementwise_jagged_output_opt_<scalar_t>(
                values,
                offsets,
                dense,
                output,
                [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
                  return y;
                });
          })

          FBGEMM_DISPATCH_ALL_TYPES_BUT_HALF_CASE([&] {
            jagged_dense_elementwise_jagged_output_<scalar_t>(
                values,
                offsets,
                dense,
                output,
                [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
                  return y;
                });
          }));

  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "dense_to_jagged_forward",
    fbgemm_gpu::dense_to_jagged_forward);
