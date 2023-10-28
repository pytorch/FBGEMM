/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>
#include "common.cuh"
#include "fbgemm_gpu/ops_utils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm_gpu {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA("uvm_to_cpu", uvm_to_cpu);
  DISPATCH_TO_CUDA("new_managed_tensor", new_managed_tensor);
  DISPATCH_TO_META("new_managed_tensor", new_managed_tensor_meta);
  DISPATCH_TO_CUDA("new_host_mapped_tensor", new_host_mapped_tensor);
  DISPATCH_TO_CUDA("new_unified_tensor", new_unified_tensor);
  DISPATCH_TO_CUDA("new_vanilla_managed_tensor", new_vanilla_managed_tensor);
}

} // namespace fbgemm_gpu
