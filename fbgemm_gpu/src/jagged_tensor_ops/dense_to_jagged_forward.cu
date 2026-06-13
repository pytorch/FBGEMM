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
    std::optional<at::SymInt> total_L) {
  // D is the embedding dimension
  auto D = dense.size(-1);
  TORCH_CHECK(D >= 0, "D must be >= 0, but got ", D);

  // If total_L is not given then compute it
  int64_t total_L_computed;
  if (total_L.has_value()) {
    total_L_computed = total_L.value().expect_int();
    TORCH_CHECK_VALUE(
        total_L_computed >= 0,
        "total_L passed to dense_to_jagged_forward must be >= 0, but got ",
        total_L_computed,
        ". This indicates total_L is corrupted somewhere prior to dense_to_jagged.");
  } else {
    total_L_computed = offsets.back().max().item<int64_t>();
    TORCH_CHECK_VALUE(
        total_L_computed >= 0,
        "total_L must be >= 0, but got ",
        total_L_computed,
        ". This indicates corrupted offsets (offsets.back() contains a garbage/negative value).",
        " offsets.size() = ",
        offsets.size(),
        " offsets.back().size(-1) = ",
        offsets.back().size(-1),
        " offsets.back()[-1] = ",
        offsets.back()[offsets.back().size(-1) - 1].item<int64_t>());
  }
  constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
  TORCH_CHECK_VALUE(
      D == 0 || total_L_computed <= kInt32Max / D,
      "total_L_computed * D overflows int32 max. total_L_computed = ",
      total_L_computed,
      " D = ",
      D,
      ". `values` is defined as PTA32. Contact FBGEMM team for int64 support.");
  TORCH_CHECK_VALUE(
      dense.numel() <= kInt32Max,
      "Expect dense.numel() <= int32 max, but got ",
      dense.numel(),
      ". y_0/y_1/y_reshaped is defined as PTA32. Contact FBGEMM team for int64 support.");
  // offsets are int32-indexed in the binary search (the non-opt kernel handles
  // num_jagged_dim up to kStackArrayMaxDims), so each offsets tensor's numel
  // must fit int32.
  for (const auto& off : offsets) {
    TORCH_CHECK_VALUE(
        off.numel() <= kInt32Max,
        "offsets numel must be <= int32 max, but got ",
        off.numel(),
        ". offsets are int32-indexed. Contact FBGEMM team for int64 support.");
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
