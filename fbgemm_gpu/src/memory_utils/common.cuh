/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

#include "common.h"
#include "fbgemm_gpu/utils/enum_utils.h"

namespace fbgemm_gpu {

FBGEMM_GPU_ENUM_CREATE_TAG(uvm)

// Forward declarations for CUDA dispatch functions
// (only for functions used with DISPATCH_TO_CUDA, not TORCH_FN)
bool is_uvm_tensor_cuda(const Tensor& t);
bool uvm_storage_cuda(const Tensor& t);
Tensor uvm_to_cpu_clone_cuda(const Tensor& t);
Tensor uvm_to_cpu_cuda(const Tensor& t);
Tensor new_managed_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);
Tensor new_host_mapped_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);
Tensor new_unified_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes,
    bool is_host_mapped);
Tensor new_vanilla_managed_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);

} // namespace fbgemm_gpu
