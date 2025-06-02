/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/library.h>

namespace fbgemm_gpu {

#ifndef USE_ROCM

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.gather_scatter");
  m.def("gather_along_first_dim(Tensor Data, Tensor Index) -> Tensor");
  m.def(
      "scatter_add_along_first_dim(Tensor Dst, Tensor Src, Tensor Index) -> ()");
}

#endif

} // namespace fbgemm_gpu
