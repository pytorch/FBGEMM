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

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {

Tensor pack_segments_forward_meta(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length) {
  at::DimVector padded_values_shape({lengths.numel(), max_length});
  for (const auto i : c10::irange(1, t_in.dim())) {
    padded_values_shape.push_back(t_in.size(i));
  }
  return at::empty(padded_values_shape, t_in.options());
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

} // namespace

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("pack_segments", TORCH_FN(fbgemm_gpu::pack_segments_forward_meta));
  m.impl(
      "asynchronous_complete_cumsum",
      TORCH_FN(fbgemm_gpu::asynchronous_complete_cumsum_meta));
  m.impl("offsets_range", TORCH_FN(fbgemm_gpu::offsets_range_meta_symint));
  m.impl(
      "batched_unary_embeddings",
      TORCH_FN(fbgemm_gpu::batched_unary_embeddings_forward_meta));
}
