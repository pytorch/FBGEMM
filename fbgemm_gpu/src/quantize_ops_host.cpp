/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor _float_to_fused8bitrowwise_gpu(const at::Tensor& input);

using namespace at;
TORCH_LIBRARY_FRAGMENT(fb, m) {
    m.def("FloatToFused8BitRowwiseQuantized(Tensor t) -> Tensor");
    m.impl("FloatToFused8BitRowwiseQuantized", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(_float_to_fused8bitrowwise_gpu)));
}
