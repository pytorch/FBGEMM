/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>
#include "common.cuh"
#include "fbgemm_gpu/utils/ops_utils.h"

namespace fbgemm_gpu {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("is_uvm_tensor(Tensor t) -> bool", TORCH_FN(is_uvm_tensor));
  m.def("uvm_storage(Tensor t) -> bool", TORCH_FN(uvm_storage));
  m.def(
      "uvm_to_device(Tensor self, Tensor prototype) -> Tensor",
      TORCH_FN(uvm_to_device));
  m.def(
      "uvm_to_device_d(Tensor self, Device device) -> Tensor",
      TORCH_FN(uvm_to_device_d));

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
  m.def("uvm_to_cpu(Tensor t) -> Tensor", TORCH_FN(uvm_to_cpu));

  m.def(FBGEMM_GPU_ENUM_OP(uvm, fbgemm_gpu_uvm_enum_query));
  m.def("copy_to_shared(Tensor t) -> ()", TORCH_FN(copy_to_shared));
  m.def(
      "initialize_nan_shared_mem(int device_index) -> ()",
      TORCH_FN(initialize_nan_shared_mem));

  DISPATCH_TO_CUDA("new_managed_tensor", new_managed_tensor);
  DISPATCH_TO_CUDA("new_host_mapped_tensor", new_host_mapped_tensor);
  DISPATCH_TO_CUDA("new_unified_tensor", new_unified_tensor);
  DISPATCH_TO_CUDA("new_vanilla_managed_tensor", new_vanilla_managed_tensor);
  DISPATCH_TO_CUDA("copy_to_shared", copy_to_shared);
  DISPATCH_TO_CUDA("initialize_nan_shared_mem", initialize_nan_shared_mem);
}

} // namespace fbgemm_gpu
