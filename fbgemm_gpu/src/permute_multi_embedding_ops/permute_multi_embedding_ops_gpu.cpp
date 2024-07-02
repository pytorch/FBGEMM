/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/permute_multi_embedding_function.h"

// do we really need this file, can we put this into
// permute_multi_embedding_ops.cu?
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // dispatch the forward function to GPU for internal (autograd) usage
  DISPATCH_TO_CUDA(
      "permute_multi_embedding_function",
      fbgemm_gpu::permute_multi_embedding_gpu);
}
