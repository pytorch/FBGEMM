/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

///@ingroup quantize-data-meta
Tensor FP8rowwise_to_float_meta(
    const Tensor& input,
    [[maybe_unused]] bool forward) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int ncols = input_sizes[last_dim];
  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned - 2 * sizeof(float);

  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  return at::empty(output_dims, input.options().dtype(at::kFloat));
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl(
      "FP8RowwiseQuantizedToFloat",
      TORCH_FN(fbgemm_gpu::FP8rowwise_to_float_meta));
}
