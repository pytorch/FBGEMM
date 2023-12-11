/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <torch/library.h>

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

namespace {

std::tuple<Tensor /*row_output_offsets*/, Tensor /*b_t_map*/>
generate_vbe_metadata_meta(
    const Tensor& B_offsets,
    const Tensor& B_offsets_rank_per_feature,
    const Tensor& output_offsets_feature_rank,
    const Tensor& D_offsets,
    const int64_t D,
    const bool nobag,
    const int64_t max_B_feature_rank,
    const int64_t info_B_num_bits,
    const c10::SymInt total_B) {
  Tensor row_output_offsets =
      at::empty_symint({total_B}, output_offsets_feature_rank.options());
  Tensor b_t_map = at::empty_symint({total_B}, B_offsets.options());
  return {row_output_offsets, b_t_map};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "transpose_embedding_input("
      "    Tensor hash_size_cumsum, "
      "    int total_hash_size_bits, "
      "    Tensor indices, "
      "    Tensor offsets, "
      "    bool nobag=False, "
      "    Tensor? vbe_b_t_map=None, "
      "    int info_B_num_bits=26, "
      "    int info_B_mask=0x2FFFFFF, "
      "    int total_unique_indices=-1, "
      "    bool is_index_select=False, "
      "    Tensor? total_L_offsets=None, "
      "    int fixed_L_per_warp=0, "
      "    int num_warps_per_feature=0"
      ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("get_infos_metadata(Tensor unused, int B, int T) -> (int, int)");
  m.def(
      "generate_vbe_metadata("
      "    Tensor B_offsets, "
      "    Tensor B_offsets_rank_per_feature, "
      "    Tensor output_offsets_feature_rank, "
      "    Tensor D_offsets, "
      "    int D, "
      "    bool nobag, "
      "    int max_B_feature_rank, "
      "    int info_B_num_bits, "
      "    SymInt total_B"
      ") -> (Tensor, Tensor)");
  DISPATCH_TO_CUDA("transpose_embedding_input", transpose_embedding_input);
  DISPATCH_TO_CUDA("get_infos_metadata", get_infos_metadata);
  DISPATCH_TO_CUDA("generate_vbe_metadata", generate_vbe_metadata);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("generate_vbe_metadata", &generate_vbe_metadata_meta);
}
