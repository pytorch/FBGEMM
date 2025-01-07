/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

// #include <ATen/cuda/CUDAEvent.h>
// #include <atomic>
// #include <cassert>
// #include <cmath>
// #include <vector>
// #include "c10/util/Exception.h"

namespace fbgemm {

at::Tensor fused_moe_impl(
    const at::Tensor& input, // [tokens, hidden_size]
    const at::Tensor&
        gate_up_weight, // [experts, intermediate_size, hidden_size]
    const at::Tensor& down_weight, // [experts, hidden_size, intermediate_size]
    const at::Tensor& topk_ids, // [tokens, topk]
    const at::Tensor& topk_weights, // [tokens, topk]
    const std::optional<at::Tensor> input_scales = {}, // [tokens]
    const std::optional<at::Tensor> gate_up_scales = {}, // [intermediate_size]
    const std::optional<at::Tensor> down_scales = {}, // [intermediate_size]
    const std::optional<at::Tensor> smooth_scales = {}, // [intermediate_size]
    int64_t block_m = 32,
    bool gate_only = true,
    int64_t fused_quant = 0);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "fused_moe(Tensor input, Tensor gate_up_weight, "
      "Tensor down_weight, Tensor topk_ids, Tensor topk_weights, "
      "Tensor? input_scales=None, Tensor? gate_up_scales=None, "
      "Tensor? down_scales=None, Tensor? smooth_scales=None, "
      "int block_m=32, bool gate_only=True, int fused_quant=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("fused_moe", fused_moe_impl);
}

} // namespace fbgemm
