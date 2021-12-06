/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <torch/library.h>
#include "fbgemm_gpu/enum_utils.h"

#include "cumem_utils.h"

using namespace at;
namespace fbgemm_gpu {

TORCH_LIBRARY_FRAGMENT(fb, m) {
  m.def("is_uvm_tensor(Tensor t) -> bool", TORCH_FN(is_uvm_tensor));
  m.def("uvm_storage(Tensor t) -> bool", TORCH_FN(uvm_storage));
  m.def(
      "uvm_to_device(Tensor self, Tensor prototype) -> Tensor",
      TORCH_FN(uvm_to_device));
  m.def("uvm_to_cpu(Tensor t) -> Tensor");
  m.impl(
      "uvm_to_cpu",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(uvm_to_cpu)));
  m.def("new_managed_tensor(Tensor self, int[] sizes) -> Tensor");
  m.impl(
      "new_managed_tensor",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(new_managed_tensor)));
  m.def("new_vanilla_managed_tensor(Tensor self, int[] sizes) -> Tensor");
  m.impl(
      "new_vanilla_managed_tensor",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(new_vanilla_managed_tensor)));
  m.def(
      "cuda_mem_advise(Tensor t, int advice) -> ()", TORCH_FN(uvm_cuda_mem_advise));
  m.def(
      "cuda_mem_prefetch_async(Tensor t, Tensor? device_t) -> ()",
      TORCH_FN(uvm_cuda_mem_prefetch_async));
  m.def("uvm_mem_advice_dont_fork(Tensor t) -> ()", TORCH_FN(uvm_mem_advice_dont_fork));

  m.def(FBGEMM_GPU_ENUM_OP(uvm, fbgemm_gpu_uvm_enum_query));
}

} // namespace fbgemm_gpu
