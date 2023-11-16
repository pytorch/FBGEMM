/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops_utils.h"
#include "metric_ops.h"

namespace fbgemm_gpu {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "batch_auc(int num_tasks, Tensor indices, Tensor laebls, Tensor weights) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA("batch_auc", fbgemm_gpu::batch_auc);
}

} // namespace fbgemm_gpu
