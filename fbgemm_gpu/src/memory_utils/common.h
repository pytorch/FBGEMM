/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include "fbgemm_gpu/cumem_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor new_unified_tensor_cpu(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes,
    bool is_host_mapped);

Tensor uvm_to_cpu_cpu(const Tensor& t);

Tensor uvm_to_cpu_clone_cpu(const Tensor& t);

bool is_uvm_tensor_cpu(const Tensor& t);

} // namespace fbgemm_gpu
