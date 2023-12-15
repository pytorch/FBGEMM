/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>
#include "common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "new_unified_tensor(Tensor self, int[] sizes, bool is_host_mapped) -> Tensor");
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CPU("new_unified_tensor", new_unified_tensor_cpu);
}

} // namespace fbgemm_gpu
