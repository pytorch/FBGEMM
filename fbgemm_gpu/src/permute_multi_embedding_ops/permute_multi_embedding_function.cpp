/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/permute_multi_embedding_function.h"
#include <cstdint>
#include <iostream>

namespace fbgemm_gpu {

using Tensor = at::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

variable_list PermuteMultiEmbeddingOp::forward(
    AutogradContext* ctx,
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const c10::SymIntArrayRef out_lengths) {
  ctx->saved_data["permutes"] = permutes;
  ctx->saved_data["in_shapes"] = in_shapes;
  ctx->saved_data["out_shapes"] = out_shapes;

  std::vector<at::SymInt> in_lengths;
  in_lengths.reserve(pooled_embs.size());
  for (auto i : c10::irange(pooled_embs.size())) {
    in_lengths.push_back(pooled_embs[i].sym_size(1));
  }
  ctx->saved_data["in_lengths"] = in_lengths;

  /*
    select the correct dispatched (cpu/gpu) forward function
    the cpu/gup function needs to be registered in the dispatcher,
    e.g., DISPATCH_TO_CPU, DISPATCH_TO_CUDA, etc.
  */
  const auto permute_op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_multi_embedding_function", "")
          .typed<decltype(permute_multi_embedding_function_meta)>();

  return permute_op.call(
      pooled_embs, permutes, in_shapes, out_shapes, out_lengths, false);
}

variable_list PermuteMultiEmbeddingOp::backward(
    AutogradContext* ctx,
    variable_list grad_output) {
  const auto permutes = ctx->saved_data["permutes"].toTensor();
  const auto in_shapes = ctx->saved_data["in_shapes"].toTensor();
  const auto out_shapes = ctx->saved_data["out_shapes"].toTensor();
  const auto in_lengths = ctx->saved_data["in_lengths"].toSymIntVector();

  /*
    select the correct dispatched (cpu/gpu) backward function
    the cpu/gup function needs to be registered in the dispatcher,
    e.g., DISPATCH_TO_CPU, DISPATCH_TO_CUDA, etc.
  */
  const auto permute_op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::permute_multi_embedding_function", "")
          .typed<decltype(permute_multi_embedding_function_meta)>();
  auto grad_input = permute_op.call(
      grad_output, permutes, out_shapes, in_shapes, in_lengths, true);
  grad_input.push_back(torch::autograd::Variable()); // permutes
  grad_input.push_back(torch::autograd::Variable()); // in_shapes
  grad_input.push_back(torch::autograd::Variable()); // out_shapes
  grad_input.push_back(torch::autograd::Variable()); // out_lengths
  return grad_input;
}

/// @ingroup permute pooled embedding function group
///
/// @brief actual implementation of generating permutes arguments
/// for the permute_multi_embedding operator
///
/// This is a helper function for the permute_multi_embedding operator. It
/// generates the vector-form required arguments for permute_multi_embedding
/// operator. including permutes, in_shapes, out_shapes, and out_lengths.
///
/// **example**
/// ```
/// # each row represents a key (feature) permute move, which consists of the
/// following parameters: # [input_tensor_idx, output_tensor_idx, input_key_idx,
/// output_key_idx, key_length, next] permutes = tensor(
///             [
///                 [0, 0, 0, 0, 3, 4],  # f1
///                 [1, 0, 0, 3, 5, 0],  # f3
///                 [0, 1, 3, 0, 4, 0],  # f2
///                 [1, 2, 5, 0, 6, 0],  # f4
///                 [0, 2, 0, 6, 3, -6],  # f1
///                 [2, 2, 0, 9, 8, 0],  # f6
///                 [0, 3, 0, 0, 3, -8],  # f1
///                 [1, 3, 11, 3, 7, 0],  # f5
///             ]
/// )
/// ```
/// # details
/// 1. from the above example usage, we can clearly see that the operatior takes
/// in the following: a) values: List[torch.Tensor], which represents the input
/// KTs. b) permutes: torch.Tensor, which contains the permute information, will
/// be explained later. c) output_lengths_list: List[int], the lengths of the
/// output tensors (KTs), which is needed to allocate memory on device ahead. d)
/// in_lengths: torch.Tensor, lengths of input tensors, which is on device. e)
/// out_lengths: torch.Tensor, lengths of output tensors, which is on device
/// 2. the operator returns a list of tensors, which represents the permuted KTs
/// 3. `permute` is the most critical argument in this operator:
/// a) 2-D tensor
/// b) each row represents a key (feature) permute move
/// c) a permute move = [input_tensor_id, output_tensor_id, input_start_idx,
/// output_start_idx, feature_length, next] d) next is used in backward when a
/// key (feature) from the input tensor is mapped to multiple places in the
/// output tensors
/// 4. The next
/// a) It's only used in the backward computation
/// b) it's usually 0, means no next
/// c) it's non-zero when there is a duplicate in the permute, e.g., the same
/// feature appears more than once in the output.
/// d) the `next` is the next index of the very same feature in the
/// permute sequence with some modifications. e) modification-1: `next` is
/// positive when it's the first of its kind [Start] f) modification-2:
/// `next` is negative when it's not the first of its kind [Continue]. g)
/// modification-3: `next` is the negative value of the length of the
/// permute sequence when it's the last of its kind. [Stop].
///
/// @param keys List[List[str]], each string represents a feature/key in a KT
/// a list of keys represents a KT
/// @param lengths List[List[int64_t]], each int represents the length of a
/// feature/key in a KT, and a list of lengths represents a KT
/// @param groups List[List[str]], each string represents a feature/key in an
/// output KT a list of strings represents one output KT
/// @return tuple of permutes, in_shapes, out_shapes and output_lengths. See the
/// inputs of permute_multi_embedding for more details. The output vector is in
/// int32_t
///
/// @note this function is used internally for the gpu and cpu versions
/// operators
///
std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<int32_t>>
kt_regroup_arguments_impl(
    const std::vector<std::vector<std::string>>& keys,
    const std::vector<std::vector<int64_t>>& lengths,
    const std::vector<std::vector<std::string>>& groups) {
  const int32_t in_tensors = keys.size();
  const int32_t out_tensors = groups.size();
  std::vector<int32_t> in_lengths(in_tensors, 0);
  std::vector<int32_t> out_lengths(out_tensors, 0);
  int32_t out_num = 0;
  // total number of features in the output KTs
  for (auto i : c10::irange(out_tensors)) {
    out_num += groups[i].size();
  }

  // lengths of each input tensor
  int32_t* __restrict__ in_offset = in_lengths.data();
  // build a lookup dictionary: key => (in_tensor, length, in_offset)
  std::unordered_map<std::string, std::tuple<int32_t, int32_t, int32_t>> lookup;
  for (auto i : c10::irange(in_tensors)) {
    for (auto j : c10::irange(lengths[i].size())) {
      // key => (in_tensor, length, in_offset)
      lookup.insert({keys[i][j], {i, lengths[i][j], in_offset[i]}});
      // add up the input tensor length, it's also the offset of the next
      in_offset[i] += lengths[i][j];
    }
  }

  // flattened permutes vector with size of out_num * PermuteParam::size
  std::vector<int32_t> permutes(out_num * PermuteParam::size);
  int32_t* __restrict__ pp = permutes.data();
  // the lengths of each output tensor
  int32_t* __restrict__ out_offset = out_lengths.data();
  // current index of permutes vector: [0, out_num]
  int32_t curr = 0;

  // last_seen is a map of key => +/- permute-index in the output tensor
  // it's negative if it's not the first time appearance
  std::unordered_map<std::string, int32_t> last_seen;
  // build the "permutes" argument: (in_tensor, out_tensor, in_offset,
  // out_offset, length, next)
  // in_tensor is the input tensor index out_tensor is
  // the output tensor index in_offset is the offset from the input tensor
  // out_offset is the offset from the output tensor
  // length is the length of the feature in the output tensor
  // next is the next duplicate feature's permute-index in the output tensor
  for (auto out_tensor : c10::irange(out_tensors)) {
    for (auto key : groups[out_tensor]) {
      // query the loockup dictionary for input tensor index, offset, and length
      auto [in_tensor, length, in_offset] = lookup.at(key);
      int32_t* __restrict__ curr_pp = pp + curr * PermuteParam::size;
      curr_pp[PermuteParam::in_tensor] = in_tensor;
      curr_pp[PermuteParam::out_tensor] = out_tensor;

      curr_pp[PermuteParam::in_offset] = in_offset; // in_offset
      curr_pp[PermuteParam::out_offset] = out_offset[out_tensor]; // out_offset
      curr_pp[PermuteParam::length] = length;
      if (auto search = last_seen.find(key); search == last_seen.end()) {
        curr_pp[PermuteParam::next] = 0;
        last_seen.insert({key, curr});
      } else {
        curr_pp[PermuteParam::next] = -out_num;
        const auto prev_permute = search->second; // index in permutes
        const auto is_prev_frist = (prev_permute >= 0);
        // update the previous permute' next value. the value is positive
        // if it's the first appearance, and negative if it's not the first
        pp[std::abs(prev_permute) * PermuteParam::size + PermuteParam::next] =
            is_prev_frist ? curr : -curr;
        // mark the curr as negative in the last_seen map, so that we can know
        // it's not the first time appearance
        search->second = -curr;
      }
      // add up the output tensor length, it's also the offset of the next
      out_offset[out_tensor] += length;
      curr++;
    }
  }
  return {permutes, in_lengths, out_lengths};
}
} // namespace fbgemm_gpu
