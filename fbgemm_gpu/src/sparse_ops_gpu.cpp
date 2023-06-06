/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
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
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input, indices);
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
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(sorted_indices, orig_indices);
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
    constexpr int group_size = sizeof...(Tensors);

    if (group_size == 0) {
      return {torch::autograd::Variable()};
    }

    TORCH_CHECK(group_size == static_cast<int32_t>(indices_group.size()));

    struct GroupIndexSelectArgs {
      int64_t input_ptrs[group_size];
      int64_t output_ptrs[group_size];
      int64_t indices_ptrs[group_size];
      int64_t warp_offsets_group[group_size + 1];
      int32_t num_cols_group[group_size];
    };

    // Allocate memory for GroupIndexSelectArgs
    Tensor args_tensor = at::empty(
        {sizeof(GroupIndexSelectArgs)},
        at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    struct GroupIndexSelectArgs* args =
        reinterpret_cast<struct GroupIndexSelectArgs*>(args_tensor.data_ptr());

    auto& first_input = input_group[0];
    auto& first_indices = indices_group[0];

    const int input_dim = first_input.dim();
    const int num_output_rows = first_indices.size(0);
    const int num_input_rows = first_input.size(0);
    Tensor input_reshaped = first_input.reshape({num_input_rows, -1});
    const int num_cols = input_reshaped.size(1);
    const int cols_per_warp = get_group_index_select_cols_per_warp();
    int64_t warp_offset = 0;
    bool use_var_cols = false;

    std::vector<Tensor> outputs;
    outputs.reserve(group_size);
    std::vector<int64_t> input_shape_group;
    input_shape_group.reserve(group_size * input_dim);
    for (const auto i : c10::irange(group_size)) {
      auto& input = input_group[i];
      auto& indices = indices_group[i];

      // Verify that all input tensors have the same number of dimensions
      TORCH_CHECK(
          input_dim == input.dim(),
          "All inputs in group_index_select must have the same number of dimensions");

      // Verify that all tensors are on the same GPU
      TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(input, indices);

      auto num_output_rows_ = indices.size(0);

      // Verify that all input tensors have the same shape[0]
      TORCH_CHECK(
          num_output_rows == num_output_rows_,
          "The number of indices to be selected must be the same for the entire group");
      TORCH_CHECK(
          num_input_rows == input.size(0),
          "The number of rows in the input must be the same for the entire group");
      Tensor input_reshaped_ = input.reshape({num_input_rows, -1});

      // Number of columns can be different
      auto num_cols_ = input_reshaped_.size(1);
      auto warps_per_row = (num_cols_ + cols_per_warp - 1) / cols_per_warp;

      if (num_cols != num_cols_) {
        use_var_cols = true;
      }

      // Copy input shape
      auto input_shape = input.sizes().vec();
      input_shape_group.insert(
          input_shape_group.end(), input_shape.begin(), input_shape.end());

      // Create output pointers
      input_shape[0] = num_output_rows_;
      Tensor output = at::empty(input_shape, input.options());
      outputs.push_back(output);

      // Store args
      args->input_ptrs[i] = reinterpret_cast<int64_t>(input.data_ptr());
      args->output_ptrs[i] = reinterpret_cast<int64_t>(output.data_ptr());
      args->indices_ptrs[i] = reinterpret_cast<int64_t>(indices.data_ptr());
      args->warp_offsets_group[i] = warp_offset;
      args->num_cols_group[i] = num_cols_;

      warp_offset += warps_per_row * num_output_rows;
    }
    // Store the last offset
    args->warp_offsets_group[group_size] = warp_offset;
    // Transfer args tensor to GPU
    args_tensor = args_tensor.to(first_input.device(), /*non_blocking=*/true);

    TORCH_CHECK(group_size * input_dim == (int)input_shape_group.size())

    struct GroupIndexSelectArgs* gpu_args =
        static_cast<struct GroupIndexSelectArgs*>(args_tensor.data_ptr());

    // Need to store args_tensor for backward to keep indices_ptrs alive
    ctx->save_for_backward({indices_group[0], input_group[0], args_tensor});
    ctx->saved_data["input_dim"] = input_dim;
    ctx->saved_data["input_shape_group"] = input_shape_group;
    ctx->saved_data["group_size"] = group_size;
    ctx->saved_data["use_var_cols"] = use_var_cols;
    ctx->saved_data["indices_ptrs"] =
        reinterpret_cast<int64_t>(gpu_args->indices_ptrs);
    ctx->saved_data["warp_offsets_group"] =
        reinterpret_cast<int64_t>(gpu_args->warp_offsets_group);
    ctx->saved_data["num_cols_group"] =
        reinterpret_cast<int64_t>(gpu_args->num_cols_group);
    ctx->saved_data["total_num_warps"] = warp_offset;

    group_index_select_or_add_cuda(
        gpu_args->input_ptrs,
        gpu_args->output_ptrs,
        gpu_args->indices_ptrs,
        gpu_args->warp_offsets_group,
        gpu_args->num_cols_group,
        first_input.scalar_type(),
        first_indices.scalar_type(),
        first_input.device().index(),
        num_input_rows,
        num_output_rows,
        /*total_num_warps=*/warp_offset,
        group_size,
        /*use_index_select=*/true,
        use_var_cols);

    return outputs;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_group) {
    const int group_size = ctx->saved_data["group_size"].toInt();

    if (group_size == 0) {
      return torch::autograd::variable_list();
    }

    const int output_dim = ctx->saved_data["input_dim"].toInt();
    std::vector<int64_t> output_shape_group =
        ctx->saved_data["input_shape_group"].toIntVector();
    const bool use_var_cols = ctx->saved_data["use_var_cols"].toBool();
    int64_t* indices_ptrs =
        reinterpret_cast<int64_t*>(ctx->saved_data["indices_ptrs"].toInt());
    int64_t* warp_offsets_group = reinterpret_cast<int64_t*>(
        ctx->saved_data["warp_offsets_group"].toInt());
    int32_t* num_cols_group =
        reinterpret_cast<int32_t*>(ctx->saved_data["num_cols_group"].toInt());
    auto total_num_warps = ctx->saved_data["total_num_warps"].toInt();

    TORCH_CHECK(static_cast<int32_t>(grad_output_group.size()) == group_size);

    // We checked in forward that all output rows are the same for all member
    // in the group
    const int num_output_rows = output_shape_group[0];
    const int num_input_rows = grad_output_group[0].size(0);

    const auto saved = ctx->get_saved_variables();
    const auto saved_itr = std::begin(saved);
    Tensor first_indices = *saved_itr;
    Tensor fwd_input = *(saved_itr + 1);

    std::vector<Tensor> output_group;
    output_group.reserve(group_size + 1);
    output_group.push_back(torch::autograd::Variable());

    Tensor args_tensor = at::empty(
        {group_size * 2},
        at::TensorOptions().dtype(at::kLong).pinned_memory(true));
    int64_t* grad_output_ptrs = args_tensor.data_ptr<int64_t>();
    int64_t* grad_input_ptrs = args_tensor.data_ptr<int64_t>() + group_size;
    for (const auto i : c10::irange(group_size)) {
      Tensor& grad = grad_output_group[i];
      TENSOR_ON_CUDA_GPU(grad);
      TENSORS_ON_SAME_DEVICE(grad, first_indices);

      auto grad_input_shape = std::vector<int64_t>(
          output_shape_group.begin() + i * output_dim,
          output_shape_group.begin() + (i + 1) * output_dim);
      Tensor grad_input = at::zeros(grad_input_shape, fwd_input.options());
      output_group.push_back(grad_input);

      // Put all grad output/input pointers in an array
      grad_output_ptrs[i] = reinterpret_cast<int64_t>(grad.data_ptr());
      grad_input_ptrs[i] = reinterpret_cast<int64_t>(grad_input.data_ptr());
    }
    // Transfer grad output pointers to GPU
    args_tensor = args_tensor.to(first_indices.device(), /*non_blocking=*/true);

    group_index_select_or_add_cuda(
        args_tensor.data_ptr<int64_t>(),
        args_tensor.data_ptr<int64_t>() + group_size,
        indices_ptrs,
        warp_offsets_group,
        num_cols_group,
        fwd_input.scalar_type(),
        first_indices.scalar_type(),
        fwd_input.device().index(),
        num_output_rows,
        num_input_rows,
        total_num_warps,
        group_size,
        /*use_index_select=*/false,
        use_var_cols);

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
  const auto group_size = input_group.size();
  std::vector<Tensor> output_group;
  // We use the APPLY_AUTOGRAD_FN macros to instantiate
  // GroupIndexSelectDim0GPUOp for different group sizes.  We only instantiate
  // up to group size of 54.
  constexpr size_t max_group_size = 54;
  // Specialize this path to avoid copy
  if (group_size <= max_group_size) {
    apply_(
        [&](auto&&... args) {
          output_group =
              GroupIndexSelectDim0GPUOp::apply(indices_group, args...);
        },
        input_group);
    return output_group;
  }

  const auto input_itr = input_group.begin();
  const auto indices_itr = indices_group.begin();

  for (size_t start = 0; start < group_size; start += max_group_size) {
    const auto end = std::min(start + max_group_size, group_size);
    std::vector<Tensor> input_subgroup(input_itr + start, input_itr + end);
    std::vector<Tensor> indices_subgroup(
        indices_itr + start, indices_itr + end);
    std::vector<Tensor> output_subgroup;
    apply_(
        [&](auto&&... args) {
          output_subgroup =
              GroupIndexSelectDim0GPUOp::apply(indices_subgroup, args...);
        },
        input_subgroup);
    output_group.insert(
        output_group.end(), output_subgroup.begin(), output_subgroup.end());
  }
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
