/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include "fbgemm_gpu/intraining_embedding_pruning.h"
#include "fbgemm_gpu/utils/ops_utils.h"

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "init_address_lookup(Tensor address_lookups, Tensor buffer_offsets, Tensor emb_sizes) -> ()");
  DISPATCH_TO_CUDA("init_address_lookup", fbgemm_gpu::init_address_lookup_cuda);
  m.def(
      "prune_embedding_tables(int iter, int pruning_interval, Tensor address_lookups, Tensor row_utils, Tensor buffer_offsets, Tensor emb_sizes) -> (Tensor, Tensor, int)");
  DISPATCH_TO_CUDA(
      "prune_embedding_tables", fbgemm_gpu::prune_embedding_tables_cuda);
  // row_util can be updated in remap_indices_update_utils
  m.def(
      "remap_indices_update_utils("
      "    int iter, "
      "    Tensor buffer_idx, "
      "    Tensor feature_lengths, "
      "    Tensor feature_offsets, "
      "    Tensor values, "
      "    Tensor address_lookup, "
      "    Tensor(a!) row_util, "
      "    Tensor buffer_offsets, "
      "    Tensor[]? full_values_list=None, "
      "    bool? update_util=None"
      ") -> Tensor");
  DISPATCH_TO_CUDA(
      "remap_indices_update_utils",
      fbgemm_gpu::remap_indices_update_utils_cuda);
}
