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

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.gather");
  m.def("gather_along_first_dim(Tensor Data, Tensor Index) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("gather_along_first_dim", gather_along_first_dim);
}

#endif

} // namespace fbgemm_gpu
