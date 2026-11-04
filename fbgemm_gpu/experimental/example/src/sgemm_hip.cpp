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

at::Tensor sgemm_float_hip(
    const double alpha,
    const at::Tensor& TA,
    const at::Tensor& TB,
    const double beta,
    const at::Tensor& TC) {
  TORCH_CHECK_EQ(TA.dim(), 2);
  TORCH_CHECK_EQ(TB.dim(), 2);
  TORCH_CHECK_EQ(TC.dim(), 2);
  TORCH_CHECK_EQ(TB.size(0), TA.size(1));
  TORCH_CHECK_EQ(TC.size(0), TA.size(0));
  TORCH_CHECK_EQ(TC.size(1), TB.size(1));
  TORCH_CHECK(TA.scalar_type() == at::kFloat);
  TORCH_CHECK(TB.scalar_type() == at::kFloat);
  TORCH_CHECK(TC.scalar_type() == at::kFloat);
  return at::addmm(
      TC, TA, TB, static_cast<float>(beta), static_cast<float>(alpha));
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "sgemm_float(float alpha, Tensor TA, Tensor TB, float beta, Tensor TC) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl(
      "sgemm_float",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(fbgemm_gpu::experimental::sgemm_float_hip)));
}

} // namespace fbgemm_gpu::experimental
