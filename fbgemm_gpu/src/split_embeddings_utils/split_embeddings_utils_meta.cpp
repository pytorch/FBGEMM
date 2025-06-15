/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

using Tensor = at::Tensor;

namespace {

std::tuple<Tensor /*row_output_offsets*/, Tensor /*b_t_map*/>
generate_vbe_metadata_meta(
    const Tensor& B_offsets,
    const Tensor& /*B_offsets_rank_per_feature*/,
    const Tensor& output_offsets_feature_rank,
    const Tensor& /*D_offsets*/,
    const int64_t /*D*/,
    const bool /*nobag*/,
    const c10::SymInt /*max_B_feature_rank*/,
    const int64_t /*info_B_num_bits*/,
    const c10::SymInt total_B) {
  Tensor row_output_offsets =
      at::empty_symint({total_B}, output_offsets_feature_rank.options());
  Tensor b_t_map = at::empty_symint({total_B}, B_offsets.options());
  return {row_output_offsets, b_t_map};
}

std::tuple<int64_t, int64_t>
get_infos_metadata_meta(Tensor /*unused*/, int64_t /*B*/, int64_t /*T*/) {
  return {-1, -1};
}

} // namespace

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("generate_vbe_metadata", &generate_vbe_metadata_meta);
  m.impl("get_infos_metadata", &get_infos_metadata_meta);
}
