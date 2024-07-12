/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/permute_multi_embedding_function.h"

namespace fbgemm_gpu {

using Tensor = at::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

std::vector<Tensor> permute_multi_embedding_cpu(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& /* in_shapes */,
    const Tensor& /* out_shapes */,
    const std::vector<int64_t>& out_lengths,
    const bool& reverse_permute) {
  std::vector<Tensor> inputs;
  inputs.reserve(pooled_embs.size());
  for (auto i : c10::irange(pooled_embs.size())) {
    Tensor cont_tensor = pooled_embs[i].contiguous();
    inputs.push_back(cont_tensor);
    TENSORS_ON_SAME_DEVICE(cont_tensor, pooled_embs[i]);
    TENSORS_ON_SAME_DEVICE(pooled_embs[i], pooled_embs[0]);
  }
  int32_t B = pooled_embs[0].size(0);
  std::vector<Tensor> outputs;
  outputs.reserve(out_lengths.size());
  for (const auto i : c10::irange(out_lengths.size())) {
    outputs.push_back(at::empty({B, out_lengths[i]}, pooled_embs[0].options()));
    TORCH_CHECK(outputs[i].is_contiguous());
  }
  at::parallel_for(0, B, 0, [&](int32_t start, int32_t end) {
    int32_t in_tensor, out_tensor, in_offset, out_offset, length, next;
    for (const auto i : c10::irange(permutes.size(0))) {
      int32_t* __restrict__ pp = permutes[i].data_ptr<int32_t>();
      if (reverse_permute) {
        out_tensor = pp[PermuteParam::in_tensor];
        in_tensor = pp[PermuteParam::out_tensor];
        out_offset = pp[PermuteParam::in_offset];
        in_offset = pp[PermuteParam::out_offset];
        next = pp[PermuteParam::next];
      } else {
        in_tensor = pp[PermuteParam::in_tensor];
        out_tensor = pp[PermuteParam::out_tensor];
        in_offset = pp[PermuteParam::in_offset];
        out_offset = pp[PermuteParam::out_offset];
      }
      length = pp[PermuteParam::length];
      if (reverse_permute && next < 0) {
        for (auto b : c10::irange(start, end)) {
          auto outp = outputs[out_tensor][b].data_ptr<float>() + out_offset;
          auto inp = inputs[in_tensor][b].data_ptr<float>() + in_offset;
          for (const auto j : c10::irange(length)) {
            outp[j] += inp[j];
          }
        }
      } else {
        for (auto b : c10::irange(start, end)) {
          auto outp = outputs[out_tensor][b].data_ptr<float>() + out_offset;
          auto inp = inputs[in_tensor][b].data_ptr<float>() + in_offset;
          std::memcpy(outp, inp, length * pooled_embs[0].itemsize());
        }
      }
    }
  });
  return outputs;
}

std::vector<Tensor> permute_multi_embedding_meta(
    const at::TensorList& pooled_embs,
    const Tensor& /* permutes */,
    const Tensor& /* in_shapes */,
    const Tensor& /* out_shapes */,
    const std::vector<int64_t>& out_lengths,
    const bool& /* reverse_permute */) {
  int32_t batch_size = pooled_embs[0].size(0);

  std::vector<Tensor> outputs;
  outputs.reserve(out_lengths.size());
  for (const auto i : c10::irange(out_lengths.size())) {
    outputs.push_back(
        at::empty({batch_size, out_lengths[i]}, pooled_embs[0].options()));
  }
  return outputs;
}

/// @ingroup permute pooled embedding function group
///
/// @brief permute and regroup keyed tensors
///
/// We often need to regroup keyed tensors (KTs) in a batch. For example, we
/// have two KTs A and B, where A contains the pooled embeddings of two features
/// (keys) F1 and F2, and B contains the pooled embeddings of two features
/// (keys) F3 and F4. Both KTs have the same batch size.
///
/// We want to permute and regroup the KTs so that in the new KTs, F1 and F3 are
/// grouped together, and F2 and F4 are grouped together.
///
/// **Example:**
/// ```python
/// # input arguments
/// keys = [["F1", "F2"], ["F3", "F4"]]
/// lengths = [[128, 128], [64, 32]]
/// batch_size = 1024
/// values = [torch.randn(batch_size, 256), torch.randn(batch_size, 96)]
///
/// # target output KTs
/// groups = [["F1", "F3"], ["F2", "F4"]]
///
/// # generate permutes
/// permutes, in_shapes, out_shapes, out_lengths = kt_regroup_permutes(keys,
/// lengths, groups)
///
/// # permute and regroup
/// permuted_values = permute_multi_embedding(values, permutes, in_shapes,
/// out_shapes, lengths)
/// ```
///
///
/// @param pooled_embs list of tensors that from KTs' values
/// @param permutes a 2D tensor with each row representing a permute operation.
/// a permute operation is about how to move/copy a feature from the input KT to
/// the output KT. the first column is the input tensor index, and the second
/// column is the output tensor index. the third column is the feature's offset
/// of input tensor, and the fourth column is the feature's offset of output
/// tensor. the fifth column is the length of the feature in a permute, and the
/// last column is a next permute row to operate on (used in backward only).
/// @param in_shapes a 1D tensor with each element representing the length of an
/// input KT.
/// @param out_shapes a 1D tensor with each element representing the length of
/// an output KT.
/// @param out_lengths a 1D vector with each element representing the length of
/// an output KT.
///
/// @return the values of the output KTs.
///
///
/// @note This operator supports autograd, and duplications in the output KTs
/// are supported, such as [["F1", "F3"], ["F2", "F4"], ["F1", "F3"]]
///
/// @warning when a feature is omitted from the output KTs, the gradient of the
/// feature won't be set to 0.
///
std::vector<Tensor> permute_multi_embedding(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const std::vector<int64_t>& out_lengths) {
  return PermuteMultiEmbeddingOp::apply(
      pooled_embs, permutes, in_shapes, out_shapes, out_lengths);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // register the forward function for internal (autograd) usage
  m.def(
      "permute_multi_embedding_function(Tensor[] pooled_embs, Tensor permutes, Tensor in_shapes, Tensor out_shapes, SymInt[] out_lengths, bool reverse=False) -> Tensor[]");

  // register the main function for external usage
  m.def(
      "permute_multi_embedding(Tensor[] pooled_embs,Tensor permutes, Tensor in_shapes, Tensor out_shapes,  SymInt[] out_lengths) -> Tensor[]");

  // dispatch the forward function to CPU for internal (autograd) usage
  DISPATCH_TO_CPU(
      "permute_multi_embedding_function",
      fbgemm_gpu::permute_multi_embedding_cpu);

  // dispatch the forward function to CPU for internal (autograd) usage
  DISPATCH_TO_META(
      "permute_multi_embedding_function",
      fbgemm_gpu::permute_multi_embedding_meta);

  // dispath the main function to Autograd for external usage
  DISPATCH_TO_AUTOGRAD(
      "permute_multi_embedding", fbgemm_gpu::permute_multi_embedding);

  // dispath the main function to Autograd for external usage
  DISPATCH_TO_CUDA(
      "permute_multi_embedding", fbgemm_gpu::permute_multi_embedding);
}
