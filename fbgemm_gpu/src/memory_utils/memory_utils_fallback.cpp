/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

bool is_uvm_tensor(const Tensor&) {
  // memory_utils.cpp is only built in non-GPU builds, so return false here
  return false;
}

Tensor uvm_to_cpu(const Tensor&) {
  // memory_utils.cpp is only built in non-GPU builds, so we always want to
  // throw here
  TORCH_CHECK(false);
  return at::empty({0});
}

} // namespace fbgemm_gpu
