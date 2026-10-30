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
#include "fbgemm_gpu/utils/grouped_int4_e4m3.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

/// @ingroup quantize-data-meta
///
Tensor FP8rowwise_to_float_meta(
    const Tensor& input,
    bool forward [[maybe_unused]],
    const int64_t output_dtype) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  const at::SymIntArrayRef input_sizes = input.sym_sizes();

  const auto last_dim = input_sizes.size() - 1;
  const at::SymInt& ncols = input_sizes[last_dim];
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
Tensor FloatToFP8RowwiseQuantized_meta(const Tensor& input, bool /*forward*/) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  const at::SymIntArrayRef input_sizes = input.sym_sizes();

  const auto last_dim = input_sizes.size() - 1;
  const at::SymInt& ncols = input_sizes[last_dim];
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
    const bool scale_bias_last [[maybe_unused]]) {
  const at::SymIntArrayRef input_sizes = input.sym_sizes();
  const at::SymInt& nrows = input_sizes[0];
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
    case SparseType::BF16:
      return at::empty_symint(
          {nrows, output_columns}, input.options().dtype(at::kBFloat16));
    default:
      TORCH_CHECK(false, "Unsupported output dtype ");
  }
}

/// @ingroup quantize-data-meta
///
Tensor float_or_half_to_grouped_int4_e4m3_meta(
    const Tensor& input,
    const int64_t group_size) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D, got ", input.dim(), "D");
  TORCH_CHECK(
      group_size > 0 && group_size % 2 == 0,
      "group_size must be a positive even number, got ",
      group_size);
  const at::SymInt& nrows = input.sym_size(0);
  const at::SymInt& dim = input.sym_size(1);
  TORCH_SYM_CHECK(
      dim.sym_gt(0) & (dim % group_size).sym_eq(0),
      "dim must be a positive multiple of group_size ",
      group_size,
      ", got ",
      dim);
  return at::empty_symint(
      {nrows, dim / 2 + dim / group_size}, input.options().dtype(at::kByte));
}

/// @ingroup quantize-data-meta
///
Tensor grouped_int4_e4m3_to_half_meta(
    const Tensor& input,
    const int64_t group_size,
    const int64_t dim,
    const bool as_bytes) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D, got ", input.dim(), "D");
  TORCH_CHECK(
      input.scalar_type() == at::kByte,
      "input must be uint8, got ",
      input.scalar_type());
  grouped_int4_e4m3::check_dims(dim, group_size);
  TORCH_SYM_CHECK(
      input.sym_size(1).sym_ge(
          grouped_int4_e4m3::packed_row_bytes(dim, group_size)),
      "rows of ",
      input.sym_size(1),
      " bytes are too short for dim ",
      dim,
      " and group_size ",
      group_size);
  if (as_bytes) {
    return at::empty_symint(
        {input.sym_size(0), 2 * dim}, input.options().dtype(at::kByte));
  }
  return at::empty_symint(
      {input.sym_size(0), dim}, input.options().dtype(at::kHalf));
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl(
      "FloatOrHalfToGroupedInt4E4M3",
      TORCH_FN(fbgemm_gpu::float_or_half_to_grouped_int4_e4m3_meta));
  m.impl(
      "GroupedInt4E4M3ToHalf",
      TORCH_FN(fbgemm_gpu::grouped_int4_e4m3_to_half_meta));
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
