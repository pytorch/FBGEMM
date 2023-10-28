/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>
#include "common.cuh"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("is_uvm_tensor(Tensor t) -> bool", TORCH_FN(is_uvm_tensor));
  m.def("uvm_storage(Tensor t) -> bool", TORCH_FN(uvm_storage));
  m.def(
      "uvm_to_device(Tensor self, Tensor prototype) -> Tensor",
      TORCH_FN(uvm_to_device));
  m.def("uvm_to_cpu(Tensor t) -> Tensor");
  m.def("new_managed_tensor(Tensor self, int[] sizes) -> Tensor");
  m.def("new_host_mapped_tensor(Tensor self, int[] sizes) -> Tensor");
  m.def(
      "new_unified_tensor(Tensor self, int[] sizes, bool is_host_mapped) -> Tensor");
  m.def("new_vanilla_managed_tensor(Tensor self, int[] sizes) -> Tensor");
  m.def(
      "cuda_mem_advise(Tensor t, int advice) -> ()",
      TORCH_FN(uvm_cuda_mem_advise));
  m.def(
      "cuda_mem_prefetch_async(Tensor t, Tensor? device_t) -> ()",
      TORCH_FN(uvm_cuda_mem_prefetch_async));
  m.def(
      "uvm_mem_advice_dont_fork(Tensor t) -> ()",
      TORCH_FN(uvm_mem_advice_dont_fork));

  m.def("uvm_to_cpu_clone(Tensor t) -> Tensor", TORCH_FN(uvm_to_cpu_clone));
  m.def(FBGEMM_GPU_ENUM_OP(uvm, fbgemm_gpu_uvm_enum_query));
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CPU("new_unified_tensor", new_unified_tensor_cpu);
  DISPATCH_TO_META("new_managed_tensor", new_managed_tensor_meta);
}

} // namespace fbgemm_gpu
