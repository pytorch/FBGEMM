/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include "fbgemm_gpu/enum_utils.h"

namespace fbgemm_gpu {

using namespace at;

///@defgroup cumem-utils CUDA Memorty Operators
///

///@ingroup cumem-utils
/// Allocate the ATen Tensor with unified managed memory (UVM)
/// and set both UVM storage preference to CPU and access from self.device
Tensor new_managed_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);

///@ingroup cumem-utils
// Allocate the ATen Tensor with host-mapped memory
Tensor new_host_mapped_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);

///@ingroup cumem-utils
// Allocate the ATen Tensor with unified managed memory (UVM) or host-mapped
// memory.
Tensor new_unified_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes,
    bool is_host_mapped);

///@ingroup cumem-utils
/// Allocate the ATen Tensor with unified managed memory (UVM)
Tensor new_vanilla_managed_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);

///@ingroup cumem-utils
/// Check if a tensor is allocated with UVM
bool uvm_storage(const Tensor& t);

///@ingroup cumem-utils
/// Check if a tensor is allocated with UVM *AND* is not on a CPU
bool is_uvm_tensor(const Tensor& t);

///@ingroup cumem-utils
/// Convert a UVM tensor to a CPU tensor
Tensor uvm_to_cpu(const Tensor& t);

///@ingroup cumem-utils
/// Create a UVM tensor on the same device as prototype sharing
/// the same uvm storage as t
Tensor uvm_to_device(const Tensor& t, const Tensor& prototype);

///@ingroup cumem-utils
/// Call cudaMemAdvise on UVM Storage. The hint enum is generated in Python
/// (fbgemm,uvm) using data returned from C++ op.
void uvm_cuda_mem_advise(const Tensor& t, int64_t cuda_memory_advise);

///@ingroup cumem-utils
/// Call cudaMemPrefetchAsync on UVM Storage
void uvm_cuda_mem_prefetch_async(
    const Tensor& t,
    c10::optional<Tensor> device_t);

///@ingroup cumem-utils
/// Call madvise(..MADV_DONTFORK) on the UVM storage. This is a workaround for
/// an issue where the UVM kernel driver unmaps UVM storage pages from the page
/// table on fork - causing slowdown on the next access from a CPU.
void uvm_mem_advice_dont_fork(const Tensor& t);

///@ingroup cumem-utils
/// Copy a contigious uvm Tensor (uvm_storage(t) is true) into a CPU Tensor
/// The copy uses single threaded memcpy
Tensor uvm_to_cpu_clone(const Tensor& t);

FBGEMM_GPU_ENUM_CREATE_TAG(uvm)

} // namespace fbgemm_gpu
