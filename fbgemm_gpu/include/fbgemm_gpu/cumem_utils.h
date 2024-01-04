/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

namespace fbgemm_gpu {

using Tensor = at::Tensor;

/// @defgroup cumem-utils CUDA Memory Operators
///

/// @ingroup cumem-utils
///
/// Allocate an `at::Tensor` with unified managed memory (UVM).  Then set its
/// preferred storage location to CPU (host memory) and establish mappings
/// on the CUDA device to the host memory.
///
/// @param self The input tensor
/// @param sizes The target tensor dimensions
///
/// @return A new tensor backed by UVM
Tensor new_managed_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);

/// @ingroup cumem-utils
///
/// Placeholder operator for the `Meta` dispatch key.
///
/// @param self The input tensor
/// @param sizes The target tensor dimensions
///
/// @return A new empty tensor
Tensor new_managed_tensor_meta(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);

/// @ingroup cumem-utils
///
/// Allocate the `at::Tensor` with host-mapped memory.
///
/// @param self The input tensor
/// @param sizes The target tensor dimensions
///
/// @return A new tensor backed by host-mapped memory
Tensor new_host_mapped_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);

/// @ingroup cumem-utils
///
/// Allocate the `at::Tensor` with either unified managed memory (UVM) or
/// host-mapped memory.
///
/// @param self The input tensor
/// @param sizes The target tensor dimensions
/// @param is_host_mapped Whether to allocate UVM or host-mapped memory
///
/// @return A new tensor backed by UVM or host-mapped memory, depending on the
/// value of `is_host_mapped`
Tensor new_unified_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes,
    bool is_host_mapped);

/// @ingroup cumem-utils
///
/// Allocate an `at::Tensor` with unified managed memory (UVM), but allow for
/// its preferred storage location to be automatically managed.
///
/// @param self The input tensor
/// @param sizes The target tensor dimensions
///
/// @return A new tensor backed by UVM
Tensor new_vanilla_managed_tensor(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes);

/// @ingroup cumem-utils
///
/// Check if a tensor is allocated with UVM (either CPU or GPU tensor).
///
/// @param self The input tensor
///
/// @return `true` if the tensor is allocated with UVM, otherwise `false`
bool uvm_storage(const Tensor& self);

/// @ingroup cumem-utils
///
/// Check if a tensor is allocated with UVM, BUT is not a CPU tensor.
///
/// @param self The input tensor
///
/// @return `true` if the tensor is a non-CPU tensor allocated with UVM,
/// otherwise `false`
bool is_uvm_tensor(const Tensor& self);

/// @ingroup cumem-utils
///
/// Convert a UVM tensor to a CPU tensor.
///
/// @param self The input tensor
///
/// @return A new tensor that is effectively the input moved from UVM to CPU
Tensor uvm_to_cpu(const Tensor& self);

/// @ingroup cumem-utils
///
/// Create a new UVM tensor that shares the same device and UVM storage with
/// `prototype`.
///
/// @param self The input tensor
/// @param prototype The target tensor whose device and and UVM storage will be
///                  shared with the new tensor
///
/// @return A new tensor that shares the same device and UVM storage with
/// `prototype`.
Tensor uvm_to_device(const Tensor& self, const Tensor& prototype);

/// @ingroup cumem-utils
///
/// Call `cudaMemAdvise()` on a UVM tensor's storage. The `cudaMemoryAdvise`
/// enum is available on the Python side in the `fbgemm_gpu.uvm` namespace; see
/// the documentation over there for valid values.
///
/// @param self The input tensor
/// @param cuda_memory_advise The `cudaMemoryAdvise` enum value, as integer
///
/// @see See <a
/// href="https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaMemAdvise">here</a>
/// For more information on the `cudaMemoryAdvise` enum.
void uvm_cuda_mem_advise(const Tensor& self, int64_t cuda_memory_advise);

/// @ingroup cumem-utils
///
/// Call `cudaMemPrefetchAsync()` on a UVM tensor's storage to prefetch memory
/// to a destination device.
///
/// @param self The input tensor
/// @param device_t **[OPTIONAL]** The tensor whose device will be the prefetch
///                 destination
///
/// @see See <a
/// href="https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaMemPrefetchAsync">here</a>
/// For more information on `cudaMemPrefetchAsync()`.
void uvm_cuda_mem_prefetch_async(
    const Tensor& self,
    c10::optional<Tensor> device_t);

/// @ingroup cumem-utils
///
/// Call `madvise(...MADV_DONTFORK)` on a UVM tensor's storage. This is a
/// workaround for an issue where the UVM kernel driver un-maps UVM storage
/// pages from the page table on fork, causing slowdown on the next access from
/// a CPU.
///
/// @param self The input tensor
///
/// @see See <a
/// href="https://man7.org/linux/man-pages/man2/madvise.2.html">here</a> For
/// more information on `madvise()`.
void uvm_mem_advice_dont_fork(const Tensor& self);

/// @ingroup cumem-utils
///
/// Copy a UVM tensor's contiguous storage (uvm_storage(t) is true) into a new
/// CPU Tensor.  The copy operation uses single-threaded `memcpy()`.
///
/// @param self The input tensor
///
/// @return A new CPU tensor containing the data copied from the UVM tensor
Tensor uvm_to_cpu_clone(const Tensor& self);

} // namespace fbgemm_gpu
