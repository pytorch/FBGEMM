/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"
#include "fbgemm_gpu/cumem_utils.h"

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

Tensor new_unified_tensor_cpu(
    const Tensor& self,
    const std::vector<std::int64_t>& /*sizes*/,
    bool /*is_host_mapped*/) {
  return at::empty({0}, self.options());
}

Tensor uvm_to_cpu_cpu(const Tensor& t) {
  // we throw here because cpu tensors are never uvm
  TORCH_CHECK(false);
  return t;
}

Tensor uvm_to_cpu_clone_cpu(const Tensor& t) {
  // we throw here because cpu tensors are never uvm
  TORCH_CHECK(false);
  return t.clone();
}

bool is_uvm_tensor_cpu(const Tensor& /*t*/) {
  return false;
}

bool uvm_storage_cpu(const Tensor& /*t*/) {
  return false;
}

namespace detail {

// Some times libraries use the fbgemm_gpu namespace directly to access these
// rather than go through the torch op registry. In the past this would always
// resolve to the GPU implementation, but now it will default to the CPU version
// unless the GPU library has been loaded.
//
// It's possible this should cover more of the public API but this is the
// minimal set that we have identified as being used for now.
using IsUvmTensorFn = bool (*)(const Tensor&);
using UvmStorageFn = bool (*)(const Tensor&);
using UvmToCpuFn = Tensor (*)(const Tensor&);

static struct UvmFnRegistry {
  IsUvmTensorFn is_uvm_tensor_fn = nullptr;
  UvmStorageFn uvm_storage_fn = nullptr;
  UvmToCpuFn uvm_to_cpu_fn = nullptr;
} global_uvm_fn_registry;

} // namespace detail

void register_uvm_gpu_impl(
    detail::IsUvmTensorFn is_uvm_tensor_fn,
    detail::UvmStorageFn uvm_storage_fn,
    detail::UvmToCpuFn uvm_to_cpu_fn) {
  detail::global_uvm_fn_registry.is_uvm_tensor_fn = is_uvm_tensor_fn;
  detail::global_uvm_fn_registry.uvm_storage_fn = uvm_storage_fn;
  detail::global_uvm_fn_registry.uvm_to_cpu_fn = uvm_to_cpu_fn;
}

bool is_uvm_tensor(const Tensor& self) {
  auto fn = detail::global_uvm_fn_registry.is_uvm_tensor_fn;
  if (fn != nullptr) {
    return fn(self);
  }
  return is_uvm_tensor_cpu(self);
}

bool uvm_storage(const Tensor& self) {
  auto fn = detail::global_uvm_fn_registry.uvm_storage_fn;
  if (fn != nullptr) {
    return fn(self);
  }
  return uvm_storage_cpu(self);
}

Tensor uvm_to_cpu(const Tensor& self) {
  auto fn = detail::global_uvm_fn_registry.uvm_to_cpu_fn;
  if (fn != nullptr) {
    return fn(self);
  }
  return uvm_to_cpu_cpu(self);
}

} // namespace fbgemm_gpu
