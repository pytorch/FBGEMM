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
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths,
    const bool& reverse_permute) {
  int64_t batch_size = pooled_embs[0].size(0);

  std::vector<Tensor> outputs;
  outputs.reserve(out_lengths.size());
  for (const auto i : c10::irange(out_lengths.size())) {
    outputs.push_back(
        at::empty({batch_size, out_lengths[i]}, pooled_embs[0].options()));
  }

  int64_t in_tensor, out_tensor, in_start, out_start, length, jump;
  const int64_t param = 6;
  for (const auto i : c10::irange(permutes.size() / param)) {
    if (reverse_permute) {
      out_tensor = permutes[i * param];
      in_tensor = permutes[i * param + 1];
      out_start = permutes[i * param + 2];
      in_start = permutes[i * param + 3];
      jump = permutes[i * param + 5];
    } else {
      in_tensor = permutes[i * param];
      out_tensor = permutes[i * param + 1];
      in_start = permutes[i * param + 2];
      out_start = permutes[i * param + 3];
    }
    length = permutes[i * param + 4];
    if (reverse_permute && jump < 0) {
      for (const auto b : c10::irange(batch_size)) {
        for (const auto j : c10::irange(length)) {
          outputs[out_tensor][b][j + out_start] +=
              pooled_embs[in_tensor][b][j + in_start];
        }
      }
    } else {
      for (const auto b : c10::irange(batch_size)) {
        for (const auto j : c10::irange(length)) {
          outputs[out_tensor][b][j + out_start] =
              pooled_embs[in_tensor][b][j + in_start];
        }
      }
    }
  }
  return outputs;
}

std::vector<Tensor> permute_multi_embedding_meta(
    const at::TensorList& pooled_embs,
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths,
    const bool& reverse_permute) {
  int64_t batch_size = pooled_embs[0].size(0);

  std::vector<Tensor> outputs;
  outputs.reserve(out_lengths.size());
  for (const auto i : c10::irange(out_lengths.size())) {
    outputs.push_back(
        at::empty({batch_size, out_lengths[i]}, pooled_embs[0].options()));
  }
  return outputs;
}

std::vector<Tensor> permute_multi_embedding_autograd(
    const at::TensorList& pooled_embs,
    const std::vector<int64_t>& permutes,
    const std::vector<int64_t>& in_lengths,
    const std::vector<int64_t>& out_lengths) {
  return PermuteMultiEmbeddingOp::apply(
      pooled_embs, permutes, in_lengths, out_lengths);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
generate_keyed_tensor_permutes(
    const std::vector<std::string>& keys,
    const std::vector<std::string>& groups,
    const std::vector<int64_t>& lengths,
    const std::vector<int64_t>& in_splits,
    const std::vector<int64_t>& out_splits) {
  const int64_t permute_param = 6;
  const int64_t in_tensors = in_splits.size();
  const int64_t out_tensors = out_splits.size();
  const int64_t in_num = lengths.size();
  const int64_t out_num = groups.size();
  std::vector<int64_t> permutes(out_num * permute_param, 0);
  std::vector<int64_t> in_lengths(in_tensors, 0);
  std::vector<int64_t> out_lengths(out_tensors, 0);

  auto ilp = in_lengths.data();
  int64_t* cumsum = new int64_t[in_num];
  int64_t curr = 0;
  std::unordered_map<std::string, std::tuple<int64_t, int64_t>> lookup;
  for (int32_t in_tensor = 0; in_tensor < in_tensors; in_tensor++) {
    for (int32_t in_key = 0; in_key < in_splits[in_tensor]; in_key++) {
      cumsum[curr] = ilp[in_tensor];
      ilp[in_tensor] += lengths[curr];
      lookup.insert({keys[curr], {in_tensor, curr}});
      curr++;
    }
  }

  auto olp = out_lengths.data();
  auto pp = permutes.data();
  curr = 0;
  std::unordered_map<int64_t, int64_t> last_seen;
  for (int32_t out_tensor = 0; out_tensor < out_tensors; out_tensor++) {
    for (int32_t out_key = 0; out_key < out_splits[out_tensor]; out_key++) {
      auto [in_tensor, idx] = lookup.at(groups[curr]);
      int64_t length = lengths[idx]; // length
      pp[curr * permute_param] = in_tensor;
      pp[curr * permute_param + 1] = out_tensor;

      pp[curr * permute_param + 2] = cumsum[idx]; // in_start
      pp[curr * permute_param + 3] = olp[out_tensor]; // out_start
      pp[curr * permute_param + 4] = length;
      olp[out_tensor] += length;
      if (auto search = last_seen.find(idx); search == last_seen.end()) {
        pp[curr * permute_param + 5] = 0;
        last_seen.insert({idx, curr});
      } else {
        pp[curr * permute_param + 5] = -out_num;
        if (search->second >= 0) {
          pp[search->second * permute_param + 5] = curr;
        } else {
          pp[-search->second * permute_param + 5] = -curr;
        }
        search->second = -curr;
      }
      curr++;
    }
  }
  delete[] cumsum;
  return {permutes, in_lengths, out_lengths};
}

std::vector<Tensor> regroup_keyed_tensor(
    const at::TensorList& pooled_embs,
    const std::vector<std::string>& keys,
    const std::vector<std::string>& groups,
    const std::vector<int64_t>& lengths,
    const std::vector<int64_t>& in_splits,
    const std::vector<int64_t>& out_splits) {
  auto [permutes, in_lengths, out_lengths] = generate_keyed_tensor_permutes(
      keys, groups, lengths, in_splits, out_splits);
  return PermuteMultiEmbeddingOp::apply(
      pooled_embs, permutes, in_lengths, out_lengths);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  // register the forward function for internal (autograd) usage
  m.def(
      "permute_multi_embedding_function(Tensor[] pooled_embs, int[] permutes, SymInt[] in_lengths, SymInt[] out_lengths, bool reverse=False) -> Tensor[]",
      {PT2_COMPLIANT_TAG});

  // register the main function for external usage
  m.def(
      "permute_multi_embedding(Tensor[] pooled_embs, int[] permutes, SymInt[] in_lengths, SymInt[] out_lengths) -> Tensor[]",
      {PT2_COMPLIANT_TAG});

  // register the main function for external usage
  m.def(
      "regroup_keyed_tensor(Tensor[] pooled_embs, str[] keys, str[] groups, int[] lengths, int[] in_splits, int[] out_splits) -> Tensor[]",
      {PT2_COMPLIANT_TAG});

  // register the permute function
  m.def(
      "generate_keyed_tensor_permutes(str[] keys, str[] groups, int[] lengths, int[] in_splits, int[] out_splits) -> (int[], int[], int[])",
      {PT2_COMPLIANT_TAG});

  DISPATCH_TO_ALL(
      "generate_keyed_tensor_permutes",
      fbgemm_gpu::generate_keyed_tensor_permutes);

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
      "permute_multi_embedding", fbgemm_gpu::permute_multi_embedding_autograd);

  // dispath the main function to Autograd for external usage
  DISPATCH_TO_CUDA(
      "permute_multi_embedding", fbgemm_gpu::permute_multi_embedding_autograd);

  // dispath the main function to Autograd for external usage
  DISPATCH_TO_AUTOGRAD(
      "regroup_keyed_tensor", fbgemm_gpu::regroup_keyed_tensor);

  // dispath the main function to Autograd for external usage
  DISPATCH_TO_CUDA("regroup_keyed_tensor", fbgemm_gpu::regroup_keyed_tensor);
}
