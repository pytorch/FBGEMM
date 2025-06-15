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
  DISPATCH_TO_CUDA("transpose_embedding_input", transpose_embedding_input);
  DISPATCH_TO_CUDA("get_infos_metadata", get_infos_metadata);
  DISPATCH_TO_CUDA("generate_vbe_metadata", generate_vbe_metadata);
}
