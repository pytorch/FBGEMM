/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

namespace fbgemm_gpu {

#ifndef USE_ROCM

at::Tensor gather_along_first_dim(at::Tensor data, at::Tensor index);

void scatter_add_along_first_dim(
    at::Tensor dst,
    at::Tensor src,
    at::Tensor index);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.gather_scatter");
  m.def("gather_along_first_dim(Tensor Data, Tensor Index) -> Tensor");
  m.def(
      "scatter_add_along_first_dim(Tensor Dst, Tensor Src, Tensor Index) -> ()");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("gather_along_first_dim", gather_along_first_dim);
  m.impl("scatter_add_along_first_dim", scatter_add_along_first_dim);
}

#endif

} // namespace fbgemm_gpu
