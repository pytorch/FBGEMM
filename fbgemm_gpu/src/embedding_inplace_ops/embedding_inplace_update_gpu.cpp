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
#include "fbgemm_gpu/embedding_inplace_update.h"

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA(
      "emb_inplace_update", fbgemm_gpu::embedding_inplace_update_cuda);
  DISPATCH_TO_CUDA(
      "pruned_array_lookup_from_row_idx",
      fbgemm_gpu::pruned_array_lookup_from_row_idx_cuda);
}
