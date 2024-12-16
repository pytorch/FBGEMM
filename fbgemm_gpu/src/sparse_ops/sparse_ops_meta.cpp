/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "c10/core/SymIntArrayRef.h"
#include "c10/util/DimVector.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor asynchronous_complete_cumsum_meta(const Tensor& t_in) {
  const auto num_dims = t_in.dim();
  TORCH_CHECK(num_dims == 1 || num_dims == 2);

  auto output = num_dims == 1
      ? at::zeros_symint({t_in.sym_numel() + 1}, t_in.options())
      : at::zeros_symint(
            {t_in.sym_size(0), t_in.sym_size(1) + 1}, t_in.options());
  return output;
}

Tensor asynchronous_exclusive_cumsum_meta(const Tensor& t_in) {
  return at::zeros_symint(t_in.sym_sizes(), t_in.options());
}

namespace {

Tensor pack_segments_forward_meta(
    const Tensor& t_in,
    const Tensor& lengths,
    const at::SymInt max_length) {
  at::SymDimVector padded_values_shape({lengths.sym_numel(), max_length});

  for (const auto i : c10::irange(1, t_in.dim())) {
    padded_values_shape.push_back(t_in.sym_size(i));
  }
  return at::empty_symint(padded_values_shape, t_in.options());
}

std::tuple<Tensor, std::optional<Tensor>> pack_segments_forward_meta_v2(
    const Tensor& t_in,
    const Tensor& lengths,
    const at::SymInt max_length,
    const bool pad_minf,
    const bool return_presence_mask) {
  TENSOR_NDIM_IS_GE(t_in, 1);
  TENSOR_NDIM_EQUALS(lengths, 1);
  TORCH_CHECK(
      t_in.dtype() == at::ScalarType::Float ||
          t_in.dtype() == at::ScalarType::Half ||
          t_in.dtype() == at::ScalarType::BFloat16 ||
          t_in.dtype() == at::ScalarType::Int ||
          t_in.dtype() == at::ScalarType::Long,
      "t_in must be of type float, half, bfloat16, int or long");
  TORCH_CHECK_GT(max_length, 0);

  at::SymDimVector padded_values_shape({lengths.sym_numel(), max_length});

  for (const auto i : c10::irange(1, t_in.dim())) {
    padded_values_shape.push_back(t_in.sym_size(i));
  }
  if (return_presence_mask) {
    // Shape of presence is batch_size x max_len
    Tensor presence_mask =
        at::empty_symint({lengths.numel(), max_length}, at::kBool);
    return {
        at::empty_symint(padded_values_shape, t_in.options()), presence_mask};
  }

  return {at::empty_symint(padded_values_shape, t_in.options()), std::nullopt};
}

Tensor pack_segments_backward_meta(
    const at::Tensor& data,
    const at::Tensor& lengths,
    const at::SymInt total_length,
    const at::SymInt max_length) {
  // Create output tensor of appropriate dimensions
  auto shape = data.sym_sizes().vec();
  shape.erase(shape.begin());
  shape[0] = total_length;

  return at::empty_symint(shape, data.options());
}

Tensor offsets_range_meta_symint(const Tensor& offsets, at::SymInt range_size) {
  return at::empty_symint(range_size, offsets.options());
}

Tensor batched_unary_embeddings_forward_meta(
    const Tensor& weight,
    const Tensor& table_offsets,
    const Tensor& offsets,
    const Tensor& /* indices */) {
  at::SymInt N = weight.sym_sizes()[0];
  at::SymInt T = table_offsets.sym_numel() - 1;
  at::SymInt B = (offsets.sym_numel() - 1) / T;
  return at::empty_symint({N, B, T}, weight.options());
}

Tensor asynchronous_inclusive_cumsum_meta(const Tensor& t_in) {
  return at::empty_symint(t_in.sym_sizes(), t_in.options());
}

} // namespace

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("pack_segments", TORCH_FN(fbgemm_gpu::pack_segments_forward_meta));
  m.impl(
      "pack_segments_v2", TORCH_FN(fbgemm_gpu::pack_segments_forward_meta_v2));
  m.impl(
      "pack_segments_backward",
      TORCH_FN(fbgemm_gpu::pack_segments_backward_meta));
  m.impl(
      "asynchronous_inclusive_cumsum",
      TORCH_FN(fbgemm_gpu::asynchronous_inclusive_cumsum_meta));
  m.impl(
      "asynchronous_exclusive_cumsum",
      TORCH_FN(fbgemm_gpu::asynchronous_exclusive_cumsum_meta));
  m.impl(
      "asynchronous_complete_cumsum",
      TORCH_FN(fbgemm_gpu::asynchronous_complete_cumsum_meta));
  m.impl("offsets_range", TORCH_FN(fbgemm_gpu::offsets_range_meta_symint));
  m.impl(
      "batched_unary_embeddings",
      TORCH_FN(fbgemm_gpu::batched_unary_embeddings_forward_meta));
}
