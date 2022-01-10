/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include "fbgemm_gpu/enum_utils.h"

namespace fbgemm_gpu {

using namespace at;

// Allocate the ATen Tensor with unified managed memory (UVM)
// and set both UVM storage preference to CPU and access from self.device
Tensor new_managed_tensor(Tensor self, std::vector<std::int64_t> sizes);

// Allocate the ATen Tensor with unified managed memory (UVM)
Tensor new_vanilla_managed_tensor(Tensor self, std::vector<std::int64_t> sizes);

// Check if a tensor is allocated with UVM
bool uvm_storage(Tensor t);

// Check if a tensor is allocated with UVM *AND* is not on a CPU
bool is_uvm_tensor(Tensor t);

// Convert a UVM tensor to a CPU tensor
Tensor uvm_to_cpu(Tensor t);

// Create a UVM tensor on the same device as prototype sharing
// the same uvm storage as t
Tensor uvm_to_device(Tensor t, Tensor prototype);

// Call cudaMemAdvise on UVM Storage. The hint enum is generated in Python
// (fbgemm,uvm) using data returned from C++ op.
void uvm_cuda_mem_advise(Tensor t, int64_t cudaMemoryAdvise);

// Call cudaMemPrefetchAsync on UVM Storage
void uvm_cuda_mem_prefetch_async(Tensor t, c10::optional<Tensor> device_t);

// Call madvise(..MADV_DONTFORK) on the UVM storage. This is a workaround for
// an issue where the UVM kernel driver unmaps UVM storage pages from the page
// table on fork - causing slowdown on the next access from a CPU.
void uvm_mem_advice_dont_fork(Tensor t);

// Copy a contigious uvm Tensor (uvm_storage(t) is true) into a CPU Tensor
// The copy uses single threaded memcpy
Tensor uvm_to_cpu_clone(Tensor t);

FBGEMM_GPU_ENUM_CREATE_TAG(uvm)

} // namespace fbgemm_gpu
