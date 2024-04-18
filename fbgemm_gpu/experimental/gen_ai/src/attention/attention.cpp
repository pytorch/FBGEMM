/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

namespace fbgemm_gpu::gen_ai::attention {

std::tuple<at::Tensor, at::Tensor, at::Tensor> gqa_attn_splitk_cuda(
    const at::Tensor& XQ,
    const at::Tensor& cache_K,
    const at::Tensor& cache_V,
    const at::Tensor& seq_positions,
    const double qk_scale,
    const int64_t num_split_ks,
    const int64_t num_groups);

} // namespace fbgemm_gpu::gen_ai::attention

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "gqa_attn_splitk("
      "    Tensor XQ, "
      "    Tensor cache_K, "
      "    Tensor cache_V, "
      "    Tensor seq_positions, "
      "    float qk_scale, "
      "    int num_split_ks, "
      "    int num_int4_kv_groups=1"
      ") -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl(
      "gqa_attn_splitk",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fbgemm_gpu::gen_ai::attention::gqa_attn_splitk_cuda)));
}
