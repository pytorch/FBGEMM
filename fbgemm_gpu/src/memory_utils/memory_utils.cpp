/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>
#include "common.h"
#include "fbgemm_gpu/cumem_utils.h"
#include "fbgemm_gpu/utils/ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor new_managed_tensor_meta(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes) {
  return at::empty(sizes, self.options());
}

Tensor new_unified_tensor_meta(
    const Tensor& self,
    const std::vector<std::int64_t>& sizes,
    bool /*is_host_mapped*/) {
  return at::empty(sizes, self.options());
}

Tensor uvm_to_cpu_meta(const Tensor& t) {
  return t;
}

Tensor uvm_to_cpu_clone_meta(const Tensor& t) {
  return t;
}

bool is_uvm_tensor_meta(const Tensor& /*t*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////
// CPU Implementation
////////////////////////////////////////////////////////////////////////////////
Tensor new_unified_tensor_cpu(
    const Tensor& self,
    const std::vector<std::int64_t>& /*sizes*/,
    bool /*is_host_mapped*/) {
  return at::empty({0}, self.options());
}

Tensor uvm_to_cpu_cpu(const Tensor& t) {
  TORCH_CHECK(
      false,
      "Cannot convert CPU tensor to UVM: CPU tensors are never UVM tensors");
  return t;
}

Tensor uvm_to_cpu_clone_cpu(const Tensor& t) {
  TORCH_CHECK(
      false,
      "Cannot clone CPU tensor as UVM: CPU tensors are never UVM tensors");
  return t.clone();
}

bool is_uvm_tensor_cpu(const Tensor& /*t*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////
// Base (Shim) Ops
////////////////////////////////////////////////////////////////////////////////
// These shim functions use the PyTorch dispatcher to dispatch to the
// appropriate device-specific implementation (CPU or CUDA) at runtime. This is
// necessary because some ops are called directly in C++ rather than through
// torch dispatch (e.g., from other FBGEMM GPU kernels).

/// Converts a UVM (Unified Virtual Memory) tensor to a CPU tensor.
/// The returned tensor shares storage with the original UVM tensor.
/// @param t The input UVM tensor
/// @return A CPU tensor view of the UVM tensor's data
Tensor uvm_to_cpu(const Tensor& t) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("fbgemm::uvm_to_cpu", "")
                       .typed<decltype(uvm_to_cpu_cpu)>();
  return op.call(t);
}

/// Converts a UVM tensor to a CPU tensor by cloning the data.
/// Unlike uvm_to_cpu, this creates an independent copy of the data.
/// @param t The input UVM tensor
/// @return A new CPU tensor containing a copy of the UVM tensor's data
Tensor uvm_to_cpu_clone(const Tensor& t) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("fbgemm::uvm_to_cpu_clone", "")
                       .typed<decltype(uvm_to_cpu_clone_cpu)>();
  return op.call(t);
}

/// Checks whether a tensor is a UVM tensor.
/// A UVM tensor is one allocated with CUDA managed memory that can be accessed
/// from both CPU and GPU.
/// @param t The tensor to check
/// @return true if the tensor is a UVM tensor, false otherwise
bool is_uvm_tensor(const Tensor& t) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("fbgemm::is_uvm_tensor", "")
                       .typed<decltype(is_uvm_tensor_cpu)>();
  return op.call(t);
}

/// Checks whether a tensor uses UVM storage.
/// This checks the storage backend rather than the tensor's current device.
/// A tensor may be on CPU but still have UVM storage if it was moved from GPU.
/// @param t The tensor to check
/// @return true if the tensor's storage is UVM-backed, false otherwise
bool uvm_storage(const Tensor& t) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("fbgemm::uvm_storage", "")
                       .typed<bool(const Tensor& t)>();
  return op.call(t);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CPU("uvm_to_cpu", uvm_to_cpu_cpu);
  DISPATCH_TO_CPU("uvm_to_cpu_clone", uvm_to_cpu_clone_cpu);
  DISPATCH_TO_CPU("is_uvm_tensor", is_uvm_tensor_cpu);
  DISPATCH_TO_META("uvm_to_cpu", uvm_to_cpu_meta);
  DISPATCH_TO_META("uvm_to_cpu_clone", uvm_to_cpu_clone_meta);
  DISPATCH_TO_META("is_uvm_tensor", is_uvm_tensor_meta);
  DISPATCH_TO_QUANTIZED_CPU("uvm_to_cpu", uvm_to_cpu_cpu);
  DISPATCH_TO_QUANTIZED_CPU("uvm_to_cpu_clone", uvm_to_cpu_clone_cpu);
  DISPATCH_TO_QUANTIZED_CPU("is_uvm_tensor", is_uvm_tensor_cpu);
}

} // namespace fbgemm_gpu
