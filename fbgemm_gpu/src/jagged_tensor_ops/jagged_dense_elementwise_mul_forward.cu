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

Tensor jagged_dense_elementwise_mul_forward(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  CUDA_DEVICE_GUARD(x_values);

  Tensor output = at::empty_like(x_values);

  AT_DISPATCH_SWITCH(
      x_values.scalar_type(),
      "jagged_dense_elementwise_mul_jagged_output_forward",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&] {
            jagged_dense_elementwise_jagged_output_opt_<scalar_t>(
                x_values,
                x_offsets,
                y,
                output,
                [] __device__(scalar_t x, scalar_t y) -> scalar_t {
                  return x * y;
                });
          } // lambda
          ) // CASE
      AT_DISPATCH_CASE_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          [&] {
            jagged_dense_elementwise_jagged_output_<scalar_t>(
                x_values,
                x_offsets,
                y,
                output,
                [] __device__(scalar_t x, scalar_t y) -> scalar_t {
                  return x * y;
                });
          } // lambda
          ) // CASE_FLOATING_TYPES_AND

  ); // SWITCH

  return output;
}
} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "jagged_dense_elementwise_mul_forward",
    fbgemm_gpu::jagged_dense_elementwise_mul_forward);
