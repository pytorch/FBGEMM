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

Tensor new_unified_tensor_cpu(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes,
    bool is_host_mapped) {
  return at::empty({0}, self.options());
}

} // namespace fbgemm_gpu
