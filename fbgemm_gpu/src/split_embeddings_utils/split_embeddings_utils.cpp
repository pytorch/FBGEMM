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
      "    int total_B"
      ") -> (Tensor, Tensor)");
  DISPATCH_TO_CUDA("transpose_embedding_input", transpose_embedding_input);
  DISPATCH_TO_CUDA("get_infos_metadata", get_infos_metadata);
  DISPATCH_TO_CUDA("generate_vbe_metadata", generate_vbe_metadata);
}
