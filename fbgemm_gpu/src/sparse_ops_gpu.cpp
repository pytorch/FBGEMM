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

namespace {
// From https://stackoverflow.com/a/28411055
template <typename T, std::size_t... Indices, typename... Args>
auto vec_to_tup_helper(
    const std::vector<T>& v,
    std::index_sequence<Indices...>) {
  return std::make_tuple(v[Indices]...);
}

template <std::size_t N, typename T>
auto vec_to_tup(const std::vector<T>& v) {
  assert(v.size() >= N);
  return vec_to_tup_helper(v, std::make_index_sequence<N>());
}

template <typename T, typename F>
void apply_(F fn, const std::vector<T>& v) {
  auto size = v.size();
#define APPLY_AUTOGRAD_FN_N(N)          \
  {                                     \
    case N:                             \
      std::apply(fn, vec_to_tup<N>(v)); \
      break;                            \
  }

#define APPLY_AUTOGRAD_FN_2N(N) \
  APPLY_AUTOGRAD_FN_N(N)        \
  APPLY_AUTOGRAD_FN_N(N + 1)

#define APPLY_AUTOGRAD_FN_6N(N) \
  APPLY_AUTOGRAD_FN_2N(N)       \
  APPLY_AUTOGRAD_FN_2N(N + 2)   \
  APPLY_AUTOGRAD_FN_2N(N + 4)

#define APPLY_AUTOGRAD_FN_18N(N) \
  APPLY_AUTOGRAD_FN_6N(N)        \
  APPLY_AUTOGRAD_FN_6N(N + 6)    \
  APPLY_AUTOGRAD_FN_6N(N + 12)

#define APPLY_AUTOGRAD_FN_54N(N) \
  APPLY_AUTOGRAD_FN_18N(N)       \
  APPLY_AUTOGRAD_FN_18N(N + 18)  \
  APPLY_AUTOGRAD_FN_18N(N + 36)

  switch (size) {
    APPLY_AUTOGRAD_FN_54N(1)
    default:
      TORCH_CHECK(false, "size is not supported ", size)
  }
#undef APPLY_AUTOGRAD_FN
}
} // namespace

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
    // .contiguous() is called on the gradient inputs because
    // the batched_unary_embeddings_backward_cuda assumes contiguous inputs.
    // may cause illegal memory access when it is not
    auto grad_output = grad_outputs[0];
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
        grad_output.stride(1) != 1 || grad_output.stride(0) % 4 != 0) {
      grad_output = grad_output.contiguous();
    }
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0) {
      grad_output = at::empty_like(grad_output).copy_(grad_output);
    }
    auto grad_weight = batched_unary_embeddings_backward_cuda(
        grad_output, weight, table_offsets, offsets, indices);
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

class IndexSelectDim0GPUOp
    : public torch::autograd::Function<IndexSelectDim0GPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input,
      const Tensor& indices,
      const int consecutive_range_start,
      const int consecutive_range_length,
      const bool skip_indices_sorting_fwd) {
    TENSOR_ON_CUDA_GPU(input);
    TENSOR_ON_CUDA_GPU(indices);
    TENSORS_ON_SAME_DEVICE(input, indices);
    // Expect a 1D index tensor
    TORCH_CHECK(indices.dim() == 1, "Index tensor must be 1D")

    Tensor sorted_indices, orig_indices;
    if (skip_indices_sorting_fwd) {
      ctx->save_for_backward({indices});
    } else {
      // Sort indices to promote locality
      std::tie(sorted_indices, orig_indices) = indices.sort();
      ctx->save_for_backward({sorted_indices, orig_indices});
    }

    ctx->saved_data["input_shape"] = input.sizes();
    ctx->saved_data["consecutive_range_start"] = consecutive_range_start;
    ctx->saved_data["consecutive_range_length"] = consecutive_range_length;
    ctx->saved_data["skip_indices_sorting_fwd"] = skip_indices_sorting_fwd;

    return {index_select_cuda(
        input,
        skip_indices_sorting_fwd ? indices : sorted_indices,
        orig_indices,
        /*indices_sorted = */ !skip_indices_sorting_fwd)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    TORCH_CHECK(grad_outputs.size() == 1);
    TENSOR_ON_CUDA_GPU(grad_outputs[0]);

    bool skip_indices_sorting_fwd =
        ctx->saved_data["skip_indices_sorting_fwd"].toBool();

    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    Tensor sorted_indices;
    Tensor orig_indices;
    if (skip_indices_sorting_fwd) {
      // Sort indices
      Tensor indices = *savedItr++;
      std::tie(sorted_indices, orig_indices) = indices.sort();
    } else {
      sorted_indices = *savedItr++;
      orig_indices = *savedItr++;
    }
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
        undef, // skip_indices_sorting_fwd
    };
  }
};

class GroupIndexSelectDim0GPUOp
    : public torch::autograd::Function<GroupIndexSelectDim0GPUOp> {
 public:
  template <class... Tensors>
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<Tensor>& indices_group,
      Tensors&&... input_tensors) {
    std::vector<Tensor> input_group = {input_tensors...};
    const int32_t num_groups = input_group.size();
    if (num_groups == 0) {
      return {torch::autograd::Variable()};
    }

    TORCH_CHECK(num_groups == (int32_t)indices_group.size())

    Tensor all_ptrs = at::empty(
        {(int64_t)num_groups * 2},
        at::TensorOptions().dtype(at::kLong).pinned_memory(true));
    int64_t* all_ptrs_buf = all_ptrs.data_ptr<int64_t>();

    auto& first_input = input_group[0];
    auto& first_indices = indices_group[0];

    const int num_output_rows = first_indices.size(0);
    const int num_input_rows = first_input.size(0);
    Tensor input_reshaped = first_input.reshape({num_input_rows, -1});
    const int num_cols = input_reshaped.size(1);

    std::vector<Tensor> saved_list;
    saved_list.reserve(num_groups + 1);
    for (int i = 0; i < num_groups; i++) {
      auto& input = input_group[i];
      auto& indices = indices_group[i];

      // Verify that all tensors are on the same GPU
      TENSOR_ON_CUDA_GPU(input);
      TENSOR_ON_CUDA_GPU(indices);
      TENSORS_ON_SAME_DEVICE(input, indices);

      // Verify that all input tensors have the same shape
      TORCH_CHECK(num_output_rows == indices.size(0))
      TORCH_CHECK(num_input_rows == input.size(0))
      Tensor input_reshaped_ = input.reshape({num_input_rows, -1});
      TORCH_CHECK(num_cols == input_reshaped_.size(1))

      // Put all pointers in an array
      all_ptrs_buf[i] = reinterpret_cast<int64_t>(input.data_ptr());
      all_ptrs_buf[i + num_groups] =
          reinterpret_cast<int64_t>(indices.data_ptr());

      // Save indices for backward
      saved_list.push_back(indices);
    }
    // Store one input tensor to get input tensor property in backward
    saved_list.push_back(input_group[0]);
    // Transfer input pointers to GPU
    all_ptrs = all_ptrs.to(first_input.device(), /*non_blocking=*/true);
    // Save pointers for backward
    saved_list.push_back(all_ptrs);

    ctx->save_for_backward(saved_list);
    ctx->saved_data["input_shape"] = first_input.sizes().vec();
    ctx->saved_data["num_groups"] = num_groups;
    ctx->saved_data["num_cols"] = num_cols;

    auto output_shape = first_input.sizes().vec();
    output_shape[0] = num_groups * num_output_rows;

    return group_index_select_cuda(
        all_ptrs.data_ptr<int64_t>(),
        all_ptrs.data_ptr<int64_t>() + num_groups,
        first_input.options(),
        first_input.scalar_type(),
        first_indices.scalar_type(),
        first_input.device().index(),
        output_shape,
        num_input_rows,
        num_output_rows,
        num_cols,
        num_groups);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_group) {
    const int num_groups = ctx->saved_data["num_groups"].toInt();
    const int num_cols = ctx->saved_data["num_cols"].toInt();
    auto output_shape = ctx->saved_data["input_shape"].toIntVector();
    const int num_output_rows = output_shape[0];
    output_shape[0] *= num_groups;

    TORCH_CHECK((int32_t)grad_output_group.size() == num_groups);

    if (num_groups == 0) {
      return torch::autograd::variable_list();
    }

    const int num_input_rows = grad_output_group[0].size(0);

    const auto saved = ctx->get_saved_variables();
    const auto saved_itr = std::begin(saved);
    Tensor first_indices = *saved_itr;
    Tensor fwd_input = *(saved_itr + num_groups);
    // Get indices pointers from all_ptrs saved in forward
    int64_t* indices_ptrs =
        (saved_itr + num_groups + 1)->data_ptr<int64_t>() + num_groups;

    Tensor grad_output_ptrs = at::empty(
        {num_groups}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
    int64_t* grad_output_ptrs_buf = grad_output_ptrs.data_ptr<int64_t>();
    for (int i = 0; i < num_groups; i++) {
      Tensor& grad = grad_output_group[i];
      TENSOR_ON_CUDA_GPU(grad);
      TENSORS_ON_SAME_DEVICE(grad, first_indices);

      // Put all grad output pointers in an array
      grad_output_ptrs_buf[i] = reinterpret_cast<int64_t>(grad.data_ptr());
    }
    // Transfer grad output pointers to GPU
    grad_output_ptrs =
        grad_output_ptrs.to(first_indices.device(), /*non_blocking=*/true);

    std::vector<Tensor> output_group;
    output_group.reserve(num_groups + 1);
    output_group.push_back(torch::autograd::Variable());

    auto output_index_add_group = group_index_add_cuda(
        grad_output_ptrs.data_ptr<int64_t>(),
        indices_ptrs,
        fwd_input.options(),
        fwd_input.scalar_type(),
        first_indices.scalar_type(),
        fwd_input.device().index(),
        output_shape,
        num_input_rows,
        num_output_rows,
        num_cols,
        num_groups);

    // This can be slow because the complexity is O(num_groups)
    output_group.insert(
        output_group.end(),
        output_index_add_group.begin(),
        output_index_add_group.end());

    TORCH_CHECK((int)output_group.size() == num_groups + 1)

    return output_group;
  }
};

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
    c10::optional<int64_t> consecutive_range_length,
    c10::optional<bool> skip_indices_sorting_fwd) {
  bool user_skip_indices_sorting_fwd =
      skip_indices_sorting_fwd ? *skip_indices_sorting_fwd : false;
  return IndexSelectDim0GPUOp::apply(
      input,
      indices,
      consecutive_range_start ? *consecutive_range_start : 0,
      consecutive_range_length ? *consecutive_range_length : 0,
      // Always skip indices sorting if doing forward only
      user_skip_indices_sorting_fwd && !c10::InferenceMode::is_enabled())[0];
}

std::vector<Tensor> group_index_select_dim0_gpu(
    const std::vector<Tensor>& input_group,
    const std::vector<Tensor>& indices_group) {
  std::vector<Tensor> output_group;
  apply_(
      [&](auto&&... args) {
        output_group = GroupIndexSelectDim0GPUOp::apply(indices_group, args...);
      },
      input_group);
  return output_group;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA(
      "permute_sparse_data", fbgemm_gpu::permute_2D_sparse_data_cuda);
  DISPATCH_TO_CUDA(
      "permute_2D_sparse_data", fbgemm_gpu::permute_2D_sparse_data_cuda);
  DISPATCH_TO_CUDA(
      "permute_1D_sparse_data", fbgemm_gpu::permute_1D_sparse_data_cuda);
  DISPATCH_TO_CUDA("invert_permute", fbgemm_gpu::invert_permute_cuda);
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
  DISPATCH_TO_CUDA(
      "group_index_select_dim0", fbgemm_gpu::group_index_select_dim0_gpu);
}
