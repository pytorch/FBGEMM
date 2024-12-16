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

at::Tensor bf16_gemm(
    at::Tensor A,
    at::Tensor B,
    std::optional<at::Tensor> bias = std::nullopt);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
#ifdef USE_ROCM
  m.def("bf16_gemm(Tensor A, Tensor B, Tensor? bias=None) -> Tensor");
#endif // USE_ROCM
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
#ifdef USE_ROCM
  m.impl("bf16_gemm", bf16_gemm);
#endif // USE_ROCM
}

at::Tensor bf16_gemm_meta(
    at::Tensor A,
    at::Tensor B,
    std::optional<at::Tensor> /* bias */ = std::nullopt) {
  const at::SymInt M = A.sym_size(0);
  const at::SymInt N = B.sym_size(0);
  auto C = at::empty_symint({M, N}, A.options().dtype(at::kBFloat16));
  return C;
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
#ifdef USE_ROCM
  m.impl("bf16_gemm", bf16_gemm_meta);
#endif // USE_ROCM
}

} // namespace fbgemm_gpu
