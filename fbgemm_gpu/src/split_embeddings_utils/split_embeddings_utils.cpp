/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/split_embeddings_utils.cuh" // @manual
#include <ATen/ATen.h>
#include <torch/library.h>
#include "fbgemm_gpu/utils/ops_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // m.def(
  //     "tbe_bwd_indices_preproc("
  //     "    Tensor hash_size_cumsum, "
  //     "    int total_hash_size_bits, "
  //     "    Tensor indices, "
  //     "    Tensor offsets, "
  //     "    int info_B_num_bits=26, "
  //     "    int info_B_mask=0x2FFFFFF, "
  //     "    int total_unique_indices=-1, "
  //     "    Tensor? vbe_b_t_map=None, "
  //     "    bool nobag=False, "
  //     "    bool is_index_select=False"
  //     ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
  //     " "Tensor, Tensor, Tensor, Tensor)");
  // DISPATCH_TO_CUDA("tbe_bwd_indices_preproc", tbe_bwd_indices_preproc_cuda);
  DISPATCH_TO_CUDA("transpose_embedding_input", transpose_embedding_input);
  DISPATCH_TO_CUDA("get_infos_metadata", get_infos_metadata);
  DISPATCH_TO_CUDA("generate_vbe_metadata", generate_vbe_metadata);
}
