// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAEvent.h>
#include <atomic>
#include <cassert>
#include <cmath>
#include <vector>
#include "c10/util/Exception.h"

namespace fbgemm_gpu {
void moe_align_block_size(
    at::Tensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    at::Tensor sorted_token_ids,
    at::Tensor experts_ids,
    at::Tensor num_tokens_post_pad);

void dynamic_scaled_fp8_quant(
    at::Tensor& out, // [..., d]
    at::Tensor const& input, // [..., d]
    at::Tensor& scales);

void dynamic_per_token_scaled_fp8_quant(
    at::Tensor& out, // [..., d]
    at::Tensor const& input, // [..., d]
    at::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub);

void static_scaled_fp8_quant(
    at::Tensor& out, // [..., d]
    at::Tensor const& input, // [..., d]
    at::Tensor const& scales);

void topk_softmax(
    at::Tensor& topk_weights, // [num_tokens, topk]
    at::Tensor& topk_indices, // [num_tokens, topk]
    at::Tensor& token_expert_indices, // [num_tokens, topk]
    at::Tensor& gating_output); // [num_tokens, num_experts]

void silu_and_mul(
    at::Tensor& out, // [..., d]
    at::Tensor& input); // [..., 2 * d]

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("silu_and_mul(Tensor(a!) out, Tensor input) -> ()");
  m.def(
      "topk_softmax(Tensor(a!) topk_weights, Tensor(b!) topk_indices, Tensor token_expert_indices, Tensor gating_output) -> ()");
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor sorted_token_ids, Tensor experts_ids, Tensor num_tokens_post_pad) -> ()");
#ifndef USE_ROCM
  m.def(
      "dynamic_scaled_fp8_quant(Tensor(a!) out, Tensor input, Tensor(b!) scales) -> ()");
  m.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor(a!) out, Tensor input, Tensor(b!) scales, Tensor? scale_ub=None) -> ()");
  m.def(
      "static_scaled_fp8_quant(Tensor(a!) out, Tensor input, Tensor scales) -> ()");
#endif
}

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  m.impl("topk_softmax", topk_softmax);
  m.impl("silu_and_mul", silu_and_mul);
  m.impl("moe_align_block_size", moe_align_block_size);
#ifndef USE_ROCM
  m.impl("dynamic_scaled_fp8_quant", dynamic_scaled_fp8_quant);
  m.impl(
      "dynamic_per_token_scaled_fp8_quant", dynamic_per_token_scaled_fp8_quant);
  m.impl("static_scaled_fp8_quant", static_scaled_fp8_quant);
#endif
}
} // namespace fbgemm_gpu
