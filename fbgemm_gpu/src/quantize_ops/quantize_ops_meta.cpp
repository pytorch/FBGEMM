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

#include "c10/core/ScalarType.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/quantize_ops_utils.h"
#include "fbgemm_gpu/sparse_ops.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

/// @ingroup quantize-data-meta
///
Tensor FP8rowwise_to_float_meta(
    const Tensor& input,
    [[maybe_unused]] bool forward,
    const int64_t output_dtype) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  const at::SymIntArrayRef input_sizes = input.sym_sizes();

  const auto last_dim = input_sizes.size() - 1;
  const at::SymInt ncols = input_sizes[last_dim];
  const at::SymInt ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const at::SymInt output_columns = ncols_aligned - 2 * sizeof(float);

  c10::SymDimVector output_dims(input_sizes.begin(), input_sizes.end());
  output_dims[last_dim] = output_columns;
  SparseType output_sparse_dtype = static_cast<SparseType>(output_dtype);
  switch (output_sparse_dtype) {
    case SparseType::FP32:
      return at::empty_symint(output_dims, input.options().dtype(at::kFloat));
    case SparseType::FP16:
      return at::empty_symint(output_dims, input.options().dtype(at::kHalf));
    case SparseType::BF16:
      return at::empty_symint(
          output_dims, input.options().dtype(at::kBFloat16));
    default:
      TORCH_CHECK(false, "Unsupported output dtype ");
  }
}

/// @ingroup quantize-data-meta
///
Tensor FloatToFP8RowwiseQuantized_meta(const Tensor& input, bool forward) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  const at::SymIntArrayRef input_sizes = input.sym_sizes();

  const auto last_dim = input_sizes.size() - 1;
  const at::SymInt ncols = input_sizes[last_dim];
  const at::SymInt ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const at::SymInt output_columns = ncols_aligned + 2 * sizeof(float);

  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  return at::empty_symint(output_dims, input.options().dtype(at::kByte));
}

/// @ingroup quantize-data-meta
///
Tensor fusednbitrowwise_to_float_or_half_meta(
    const Tensor& input,
    const int64_t bit_rate,
    const int64_t output_dtype,
    [[maybe_unused]] const bool scale_bias_last) {
  const at::SymIntArrayRef input_sizes = input.sym_sizes();
  const at::SymInt nrows = input_sizes[0];
  // Here we want the number of bytes in a row
  const at::SymInt ncols = nbit_elems_to_bytes_meta(input);
  const at::SymInt num_elem_per_byte = 8 / bit_rate;
  const at::SymInt output_columns =
      (ncols - 2 * sizeof(at::Half)) * num_elem_per_byte;

  SparseType output_sparse_dtype = static_cast<SparseType>(output_dtype);
  switch (output_sparse_dtype) {
    case SparseType::FP32:
      return at::empty_symint(
          {nrows, output_columns}, input.options().dtype(at::kFloat));
    case SparseType::FP16:
      return at::empty_symint(
          {nrows, output_columns}, input.options().dtype(at::kHalf));
    default:
      TORCH_CHECK(false, "Unsupported output dtype ");
  }
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl(
      "FP8RowwiseQuantizedToFloat",
      TORCH_FN(fbgemm_gpu::FP8rowwise_to_float_meta));
  m.impl(
      "FloatToFP8RowwiseQuantized",
      TORCH_FN(fbgemm_gpu::FloatToFP8RowwiseQuantized_meta));
  m.impl(
      "FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf",
      TORCH_FN(fbgemm_gpu::fusednbitrowwise_to_float_or_half_meta));
}
