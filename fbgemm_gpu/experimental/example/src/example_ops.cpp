/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

namespace fbgemm_gpu::experimental {

at::Tensor add_tensors_float(const at::Tensor& a, const at::Tensor& b) {
  return a.to(at::kFloat) + b.to(at::kFloat);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("add_tensors_float(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl(
      "add_tensors_float",
      torch::dispatch(
          c10::DispatchKey::CPU,
          TORCH_FN(fbgemm_gpu::experimental::add_tensors_float)));
}

} // namespace fbgemm_gpu::experimental
