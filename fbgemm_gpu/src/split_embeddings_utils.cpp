/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "transpose_embedding_input(Tensor hash_size_cumsum, int total_hash_size_bits, Tensor indices, Tensor offsets, bool nobag=False) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  DISPATCH_TO_CUDA("transpose_embedding_input", transpose_embedding_input);
}
