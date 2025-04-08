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

std::vector<Tensor> permute_multi_embedding_function_cpu(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& /* in_shapes */,
    const Tensor& /* out_shapes */,
    const c10::IntArrayRef out_lengths,
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
  const auto lengths = reinterpret_cast<const int64_t*>(out_lengths.data());
  for (const auto i : c10::irange(out_lengths.size())) {
    outputs.push_back(at::empty({B, lengths[i]}, pooled_embs[0].options()));
    TORCH_CHECK(outputs[i].is_contiguous());
  }
  FBGEMM_DISPATCH_FLOATING_TYPES(
      pooled_embs[0].scalar_type(), "permute_multi_embs_cpu", [&] {
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
                auto outp =
                    outputs[out_tensor][b].data_ptr<scalar_t>() + out_offset;
                auto inp =
                    inputs[in_tensor][b].data_ptr<scalar_t>() + in_offset;
                for (const auto j : c10::irange(length)) {
                  outp[j] += inp[j];
                }
              }
            } else {
              for (auto b : c10::irange(start, end)) {
                auto outp =
                    outputs[out_tensor][b].data_ptr<scalar_t>() + out_offset;
                auto inp =
                    inputs[in_tensor][b].data_ptr<scalar_t>() + in_offset;
                std::memcpy(outp, inp, length * pooled_embs[0].itemsize());
              }
            }
          }
        });
      });
  return outputs;
}

std::vector<Tensor> permute_multi_embedding_function_meta(
    const at::TensorList& pooled_embs,
    const Tensor& /* permutes */,
    const Tensor& /* in_shapes */,
    const Tensor& /* out_shapes */,
    const c10::SymIntArrayRef out_lengths,
    const bool& /* reverse_permute */) {
  auto batch_size = pooled_embs[0].sym_size(0);

  std::vector<Tensor> outputs;
  outputs.reserve(out_lengths.size());
  for (const auto i : c10::irange(out_lengths.size())) {
    outputs.push_back(at::zeros_symint(
        {batch_size, out_lengths[i]}, pooled_embs[0].options()));
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
/// permutes, in_shapes, out_shapes, out_lengths = kt_regroup_arguments(keys,
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
std::vector<Tensor> permute_multi_embedding_autograd(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const c10::SymIntArrayRef out_lengths) {
  return PermuteMultiEmbeddingOp::apply(
      pooled_embs, permutes, in_shapes, out_shapes, out_lengths);
}

std::vector<Tensor> permute_multi_embedding_cpu(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const c10::IntArrayRef out_lengths) {
  return permute_multi_embedding_function_cpu(
      pooled_embs, permutes, in_shapes, out_shapes, out_lengths, false);
}

std::vector<Tensor> permute_multi_embedding_meta(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const c10::SymIntArrayRef out_lengths) {
  return permute_multi_embedding_function_meta(
      pooled_embs, permutes, in_shapes, out_shapes, out_lengths, false);
}

template <typename index_t>
Tensor from_cpu(const std::vector<index_t>& input) {
  Tensor output = at::empty(
      {static_cast<index_t>(input.size())},
      torch::TensorOptions().dtype(torch::kInt32).pinned_memory(false));
  // Ensure that output is contiguous
  TORCH_CHECK(output.is_contiguous());
  std::memcpy(
      output.data_ptr<index_t>(), input.data(), input.size() * sizeof(index_t));
  return output;
}

/// @ingroup permute pooled embedding function group
///
/// @brief generate the permutes arguments for permute_multi_embedding
/// operator
///
/// This is a helper function for the permute_multi_embedding operator. It
/// generates the required arguments for permute_multi_embedding operator.
/// including permutes, in_shapes, out_shapes, and out_lengths.
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
/// permutes, in_shapes, out_shapes, out_lengths = kt_regroup_arguments(keys,
/// lengths, groups)
///
/// # permute and regroup
/// permuted_values = permute_multi_embedding(values, permutes, in_shapes,
/// out_shapes, lengths)
/// ```
///
///
/// @param emb one of the tensors from KTs' values
/// @param keys List[List[str]], each string represents a feature/key in a KT
/// a list of keys represents a KT
/// @param lengths List[List[int64_t]], each int represents the length of a
/// feature/key in a KT, and a list of lengths represents a KT
/// @param groups List[List[str]], each string represents a feature/key in an
/// output KT a list of strings represents one output KT
/// @return tuple of permutes, in_shapes, out_shapes and output_lengths. See the
/// inputs of permute_multi_embedding for more details. The output tensors
/// should be contiguous, and on the same device as the input tensor.
///
/// @note This operator doesn't need autograd since it's purely about index.
///
/// @warning the dispatcher should be able to dispatch meta function for this
/// operator.
///
std::tuple<Tensor, Tensor, Tensor, std::vector<int64_t>>
kt_regroup_arguments_cpu(
    const Tensor& /* embs */,
    const std::vector<std::vector<std::string>>& keys,
    const std::vector<std::vector<int64_t>>& lengths,
    const std::vector<std::vector<std::string>>& groups) {
  auto [permutes, in_lengths, out_lengths] =
      kt_regroup_arguments_impl(keys, lengths, groups);
  auto pt = from_cpu<int32_t>(permutes).view({-1, PermuteParam::size});
  auto in_shapes = from_cpu<int32_t>(in_lengths);
  auto out_shapes = from_cpu<int32_t>(out_lengths);
  std::vector<int64_t> out(out_lengths.begin(), out_lengths.end());
  return {pt, in_shapes, out_shapes, out};
}

std::tuple<Tensor, Tensor, Tensor, std::vector<int64_t>>
kt_regroup_arguments_meta(
    const Tensor& embs,
    const std::vector<std::vector<std::string>>& keys,
    const std::vector<std::vector<int64_t>>& lengths,
    const std::vector<std::vector<std::string>>& groups) {
  const int32_t in_tensors = keys.size();
  const int32_t out_tensors = groups.size();
  int32_t out_num = 0; // total number of features in the output KTs
  for (auto i : c10::irange(out_tensors)) {
    out_num += groups[i].size();
  }

  Tensor permutes = at::empty({out_num, PermuteParam::size}, embs.options());
  Tensor in_shapes = at::empty({in_tensors}, embs.options());
  Tensor out_shapes = at::empty({out_tensors}, embs.options());
  std::vector<int64_t> out_lengths(out_tensors, 0);

  std::unordered_map<std::string, int32_t> lookup;
  for (auto i : c10::irange(in_tensors)) {
    for (auto j : c10::irange(lengths[i].size())) {
      lookup.insert({keys[i][j], lengths[i][j]});
    }
  }

  int64_t* __restrict__ olp = reinterpret_cast<int64_t*>(out_lengths.data());
  for (auto i : c10::irange(out_tensors)) {
    for (auto j : c10::irange(groups[i].size())) {
      auto length = lookup.at(groups[i][j]);
      olp[i] += length;
    }
  }
  return {permutes, in_shapes, out_shapes, out_lengths};
}

std::vector<Tensor> regroup_keyed_tensor_autograd(
    const at::TensorList& pooled_embs,
    const std::vector<std::vector<std::string>>& keys,
    const std::vector<std::vector<int64_t>>& lengths,
    const std::vector<std::vector<std::string>>& groups) {
  const auto op = torch::Dispatcher::singleton()
                      .findSchemaOrThrow("fbgemm::kt_regroup_arguments", "")
                      .typed<decltype(kt_regroup_arguments_cpu)>();
  auto [permutes, in_shapes, out_shapes, out_lengths] =
      op.call(pooled_embs[0], keys, lengths, groups);
  std::vector<at::SymInt> out;
  std::transform(
      out_lengths.begin(),
      out_lengths.end(),
      std::back_inserter(out),
      [](const int32_t v) { return c10::SymInt(v); });
  return PermuteMultiEmbeddingOp::apply(
      pooled_embs, permutes, in_shapes, out_shapes, out);
}

std::vector<Tensor> regroup_keyed_tensor_cpu(
    const at::TensorList& pooled_embs,
    const std::vector<std::vector<std::string>>& keys,
    const std::vector<std::vector<int64_t>>& lengths,
    const std::vector<std::vector<std::string>>& groups) {
  auto [permutes, in_shapes, out_shapes, out_lengths] =
      kt_regroup_arguments_cpu(pooled_embs[0], keys, lengths, groups);
  return permute_multi_embedding_function_cpu(
      pooled_embs, permutes, in_shapes, out_shapes, out_lengths, false);
}

std::vector<Tensor> regroup_keyed_tensor_meta(
    const at::TensorList& pooled_embs,
    const std::vector<std::vector<std::string>>& keys,
    const std::vector<std::vector<int64_t>>& lengths,
    const std::vector<std::vector<std::string>>& groups) {
  auto [permutes, in_shapes, out_shapes, out_lengths] =
      kt_regroup_arguments_meta(pooled_embs[0], keys, lengths, groups);
  std::vector<at::SymInt> out;
  std::transform(
      out_lengths.begin(),
      out_lengths.end(),
      std::back_inserter(out),
      [](const int32_t v) { return c10::SymInt(v); });
  return permute_multi_embedding_function_meta(
      pooled_embs, permutes, in_shapes, out_shapes, out, false);
}
} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.set_python_module("fbgemm_gpu.sparse_ops");
  // register the forward function for internal (autograd) usage
  m.def(
      "permute_multi_embedding_function(Tensor[] pooled_embs, Tensor permutes, Tensor in_shapes, Tensor out_shapes, SymInt[] out_lengths, bool reverse=False) -> Tensor[]");

  // register the main function for external usage
  m.def(
      "permute_multi_embedding(Tensor[] pooled_embs,Tensor permutes, Tensor in_shapes, Tensor out_shapes,  SymInt[] out_lengths) -> Tensor[]");

  // register the permute function
  m.def(
      "kt_regroup_arguments(Tensor embs, str[][] keys, int[][] lengths, str[][] groups) -> (Tensor, Tensor, Tensor, int[])");

  // register the main function for external usage
  m.def(
      "regroup_keyed_tensor(Tensor[] pooled_embs, str[][] keys, int[][] lengths, str[][] groups) -> Tensor[]");

  // dispatch the forward function to CPU for internal (autograd) usage
  DISPATCH_TO_CPU(
      "permute_multi_embedding_function",
      fbgemm_gpu::permute_multi_embedding_function_cpu);

  // dispatch the forward function to CPU for internal (autograd) usage
  DISPATCH_TO_META(
      "permute_multi_embedding_function",
      fbgemm_gpu::permute_multi_embedding_function_meta);

  // dispath the main function to Autograd for external usage
  DISPATCH_TO_AUTOGRAD(
      "permute_multi_embedding", fbgemm_gpu::permute_multi_embedding_autograd);

  // dispath the main function to CPU for external usage
  DISPATCH_TO_CPU(
      "permute_multi_embedding", fbgemm_gpu::permute_multi_embedding_cpu);

  // dispath the main function to CPU for external usage
  DISPATCH_TO_META(
      "permute_multi_embedding", fbgemm_gpu::permute_multi_embedding_meta);

  DISPATCH_TO_CPU("kt_regroup_arguments", fbgemm_gpu::kt_regroup_arguments_cpu);
  DISPATCH_TO_META(
      "kt_regroup_arguments", fbgemm_gpu::kt_regroup_arguments_meta);

  // dispath the main function to Autograd for external usage
  DISPATCH_TO_AUTOGRAD(
      "regroup_keyed_tensor", fbgemm_gpu::regroup_keyed_tensor_autograd);

  // dispath the main function to CPU for external usage
  DISPATCH_TO_CPU("regroup_keyed_tensor", fbgemm_gpu::regroup_keyed_tensor_cpu);

  // dispath the main function to META for external usage
  DISPATCH_TO_META(
      "regroup_keyed_tensor", fbgemm_gpu::regroup_keyed_tensor_meta);
}
