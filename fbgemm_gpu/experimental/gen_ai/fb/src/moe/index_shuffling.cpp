// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace fbgemm_gpu {

#ifndef USE_ROCM

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch(
    const at::Tensor& scores);

std::tuple<at::Tensor, at::Tensor, at::Tensor> index_shuffling_torch_meta(
    const at::Tensor& scores) {
  int T = scores.size(0);
  int E = scores.size(1);
  at::Tensor counts = at::empty({E + 1}, scores.options().dtype(at::kInt));
  at::Tensor expert_indices = at::empty({T}, scores.options().dtype(at::kInt));
  at::Tensor token_indices = at::empty({T}, scores.options().dtype(at::kInt));
  return {counts, expert_indices, token_indices};
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.experimental.gen_ai.moe");
  m.def("index_shuffling(Tensor Scores) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("index_shuffling", index_shuffling_torch);
}

TORCH_LIBRARY_IMPL(fbgemm, Meta, m) {
  m.impl("index_shuffling", index_shuffling_torch_meta);
}
#endif

} // namespace fbgemm_gpu
