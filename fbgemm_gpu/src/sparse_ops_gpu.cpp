/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <stdexcept> // for logic_error

using Tensor = at::Tensor;

namespace fbgemm_gpu {
// Custom PackSegments operator that is based on the Caffe2 PackSegments and
// UnpackSegments.
// Needed this to support backward pass.
class PackSegments : public torch::autograd::Function<PackSegments> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& t_in,
      const Tensor& lengths,
      const int64_t max_length) {
    const int64_t total_length = t_in.contiguous().size(0);
    ctx->saved_data["max_length"] = max_length;
    ctx->saved_data["total_length"] = total_length;
    ctx->save_for_backward({lengths});

    // Run the forward pass.
    const auto& res = pack_segments_forward_cuda(t_in, lengths, max_length);

    torch::autograd::variable_list outputs(1);
    outputs[0] = res;
    return outputs;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    TORCH_CHECK(grad_output.size() == 2 or grad_output.size() == 1);
    const Tensor& grad = grad_output[0];
    const auto& max_length = ctx->saved_data["max_length"].toInt();
    const auto& total_length = ctx->saved_data["total_length"].toInt();

    // Retrieve saved variables for backward.
    const auto& saved_variables = ctx->get_saved_variables();
    const auto& lengths = saved_variables[0];

    torch::autograd::variable_list grad_inputs(5);
    grad_inputs[0] =
        pack_segments_backward_cuda(grad, lengths, total_length, max_length);
    return grad_inputs;
  }
};

class LookupFunctionBatchedUnaryEmbeddingOp
    : public torch::autograd::Function<LookupFunctionBatchedUnaryEmbeddingOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& weight,
      const Tensor& table_offsets,
      const Tensor& offsets,
      const Tensor& indices) {
    ctx->save_for_backward({weight, table_offsets, offsets, indices});
    auto output = batched_unary_embeddings_forward_cuda(
        weight, table_offsets, offsets, indices);
    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto weight = *savedItr++;
    auto table_offsets = *savedItr++;
    auto offsets = *savedItr++;
    auto indices = *savedItr++;
    TORCH_CHECK(grad_outputs.size() == 1);
    auto grad_weight = batched_unary_embeddings_backward_cuda(
        grad_outputs[0], weight, table_offsets, offsets, indices);
    return {grad_weight, Tensor(), Tensor(), Tensor()};
  }
};

Tensor lookup_batched_unary_embedding_function(
    const Tensor& weight,
    const Tensor& table_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  return LookupFunctionBatchedUnaryEmbeddingOp::apply(
      weight, table_offsets, offsets, indices)[0];
}

class StackedJagged2DToDenseGPUOp
    : public torch::autograd::Function<StackedJagged2DToDenseGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      Tensor values,
      Tensor lengths,
      const std::vector<int64_t>& offset_per_key,
      const std::vector<int64_t>& max_lengths_per_key) {
    int64_t total_L = values.size(0);
    ctx->saved_data["B"] = lengths.size(1);
    ctx->saved_data["D"] = values.size(1);
    ctx->saved_data["total_L"] = total_L;
    ctx->saved_data["offset_per_key"] = offset_per_key;

    auto [padded_values_per_key, offsets_tensor_per_key] =
        stacked_jagged_2d_to_dense_forward_cuda(
            values, lengths, offset_per_key, max_lengths_per_key);
    ctx->saved_data["offsets_tensor_per_key"] = offsets_tensor_per_key;

    return padded_values_per_key;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto B = ctx->saved_data["B"].toInt();
    auto D = ctx->saved_data["D"].toInt();
    auto total_L = ctx->saved_data["total_L"].toInt();
    auto offset_per_key = ctx->saved_data["offset_per_key"].toIntVector();
    auto offsets_tensor_per_key =
        ctx->saved_data["offsets_tensor_per_key"].toTensorVector();

    using torch::autograd::Variable;
    auto grad_values = stacked_jagged_2d_to_dense_backward_cuda(
        B, D, total_L, grad_outputs, offsets_tensor_per_key, offset_per_key);
    return {
        grad_values,
        Variable(), // lengths
        Variable(), // offset_per_key
        Variable() // max_lengths_per_key
    };
  }
};

class IndexSelectDim0GPUOp
    : public torch::autograd::Function<IndexSelectDim0GPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input,
      const Tensor& indices,
      const int consecutive_range_start,
      const int consecutive_range_length) {
    TENSOR_ON_CUDA_GPU(input);
    TENSOR_ON_CUDA_GPU(indices);
    TENSORS_ON_SAME_DEVICE(input, indices);

    // Sort indices to promote locality
    Tensor sorted_indices, orig_indices;
    std::tie(sorted_indices, orig_indices) = indices.sort();

    ctx->save_for_backward({sorted_indices, orig_indices});
    ctx->saved_data["input_shape"] = input.sizes();
    ctx->saved_data["consecutive_range_start"] = consecutive_range_start;
    ctx->saved_data["consecutive_range_length"] = consecutive_range_length;

    return {index_select_with_sorted_indices_cuda(
        input,
        sorted_indices,
        orig_indices,
        consecutive_range_start,
        consecutive_range_length)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    TORCH_CHECK(grad_outputs.size() == 1);
    TENSOR_ON_CUDA_GPU(grad_outputs[0]);

    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    Tensor sorted_indices = *savedItr++;
    Tensor orig_indices = *savedItr++;
    TENSOR_ON_CUDA_GPU(sorted_indices);
    TENSOR_ON_CUDA_GPU(orig_indices);
    Tensor grad_output = grad_outputs[0];
    TENSORS_ON_SAME_DEVICE(grad_output, sorted_indices);
    auto input_shape = ctx->saved_data["input_shape"].toIntVector();
    int consecutive_range_start =
        ctx->saved_data["consecutive_range_start"].toInt();
    int consecutive_range_length =
        ctx->saved_data["consecutive_range_length"].toInt();

    Tensor undef;
    return {
        index_add_with_unique_indices_cuda(
            grad_output,
            sorted_indices,
            orig_indices,
            input_shape,
            consecutive_range_start,
            consecutive_range_length),
        torch::autograd::Variable(), // indices
        undef, // consecutive_range_start
        undef, // consecutive_range_length
    };
  }
};

std::vector<Tensor> stacked_jagged_2d_to_dense_gpu(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(lengths);
  TENSORS_ON_SAME_DEVICE(values, lengths);
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(lengths.dim() == 2);
  return StackedJagged2DToDenseGPUOp::apply(
      values, lengths, offset_per_key, max_lengths_per_key);
}

Tensor pack_segments_cuda(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length) {
  const auto& res = PackSegments::apply(t_in, lengths, max_length);
  return res[0];
}

Tensor index_select_dim0_gpu(
    const Tensor& input,
    const Tensor& indices,
    c10::optional<int64_t> consecutive_range_start,
    c10::optional<int64_t> consecutive_range_length) {
  return IndexSelectDim0GPUOp::apply(
      input,
      indices,
      consecutive_range_start ? *consecutive_range_start : 0,
      consecutive_range_length ? *consecutive_range_length : 0)[0];
}
} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA(
      "permute_sparse_data", fbgemm_gpu::permute_2D_sparse_data_cuda);
  DISPATCH_TO_CUDA(
      "permute_2D_sparse_data", fbgemm_gpu::permute_2D_sparse_data_cuda);
  DISPATCH_TO_CUDA(
      "permute_1D_sparse_data", fbgemm_gpu::permute_1D_sparse_data_cuda);
  DISPATCH_TO_CUDA(
      "expand_into_jagged_permute",
      fbgemm_gpu::expand_into_jagged_permute_cuda);
  DISPATCH_TO_CUDA(
      "block_bucketize_sparse_features",
      fbgemm_gpu::block_bucketize_sparse_features_cuda);
  DISPATCH_TO_CUDA(
      "bucketize_sparse_features", fbgemm_gpu::bucketize_sparse_features_cuda);
  DISPATCH_TO_CUDA(
      "asynchronous_exclusive_cumsum",
      fbgemm_gpu::asynchronous_exclusive_cumsum_gpu);
  DISPATCH_TO_CUDA(
      "asynchronous_complete_cumsum",
      fbgemm_gpu::asynchronous_complete_cumsum_gpu);
  DISPATCH_TO_CUDA(
      "asynchronous_inclusive_cumsum",
      fbgemm_gpu::asynchronous_inclusive_cumsum_gpu);
  DISPATCH_TO_CUDA(
      "reorder_batched_ad_lengths", fbgemm_gpu::reorder_batched_ad_lengths_gpu);
  DISPATCH_TO_CUDA(
      "reorder_batched_ad_indices", fbgemm_gpu::reorder_batched_ad_indices_gpu);
  DISPATCH_TO_CUDA("offsets_range", fbgemm_gpu::offsets_range_cuda);
  DISPATCH_TO_CUDA(
      "batched_unary_embeddings",
      fbgemm_gpu::lookup_batched_unary_embedding_function);
  DISPATCH_TO_CUDA("jagged_1d_to_dense", fbgemm_gpu::jagged_1d_to_dense_gpu);
  DISPATCH_TO_CUDA("jagged_2d_to_dense", fbgemm_gpu::jagged_2d_to_dense_gpu);
  DISPATCH_TO_CUDA(
      "stacked_jagged_1d_to_dense", fbgemm_gpu::stacked_jagged_1d_to_dense_gpu);
  DISPATCH_TO_CUDA(
      "stacked_jagged_2d_to_dense", fbgemm_gpu::stacked_jagged_2d_to_dense_gpu);
  DISPATCH_TO_CUDA(
      "stacked_jagged_2d_to_dense_forward",
      fbgemm_gpu::stacked_jagged_2d_to_dense_forward_cuda);
  DISPATCH_TO_CUDA(
      "stacked_jagged_2d_to_dense_backward",
      fbgemm_gpu::stacked_jagged_2d_to_dense_backward_cuda);
  DISPATCH_TO_CUDA(
      "histogram_binning_calibration",
      fbgemm_gpu::histogram_binning_calibration_cuda);
  DISPATCH_TO_CUDA(
      "histogram_binning_calibration_by_feature",
      fbgemm_gpu::histogram_binning_calibration_by_feature_cuda);
  DISPATCH_TO_CUDA(
      "generic_histogram_binning_calibration_by_feature",
      fbgemm_gpu::generic_histogram_binning_calibration_by_feature_cuda);
  DISPATCH_TO_CUDA("segment_sum_csr", fbgemm_gpu::segment_sum_csr_cuda);
  DISPATCH_TO_CUDA("lengths_range", fbgemm_gpu::lengths_range_cuda);
  DISPATCH_TO_CUDA(
      "permute_sparse_features", fbgemm_gpu::permute_sparse_features_cuda);
  DISPATCH_TO_CUDA(
      "Bfloat16QuantizedToFloat", fbgemm_gpu::_bfloat16_to_float_gpu);
  DISPATCH_TO_CUDA(
      "FloatToBfloat16Quantized", fbgemm_gpu::_float_to_bfloat16_gpu);
  DISPATCH_TO_CUDA(
      "permute102_baddbmm_permute102",
      fbgemm_gpu::permute102_baddbmm_permute102_cuda);
  DISPATCH_TO_CUDA(
      "permute_sequence_embeddings",
      fbgemm_gpu::permute_sequence_embeddings_cuda);
  DISPATCH_TO_CUDA("pack_segments", fbgemm_gpu::pack_segments_cuda);
  DISPATCH_TO_CUDA("index_select_dim0", fbgemm_gpu::index_select_dim0_gpu);
}
