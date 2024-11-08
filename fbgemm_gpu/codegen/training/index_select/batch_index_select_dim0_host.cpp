/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

Tensor batch_index_select_dim0_codegen_forward_cuda(
    const Tensor& dev_weights,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt max_D,
    const Tensor& indices,
    const int64_t output_dtype,
    const Tensor& output_offsets,
    const Tensor& total_L_offsets,
    const int64_t output_size,
    const int32_t fixed_L_per_warp,
    const int32_t num_warps_per_feature,
    const bool permute_output_dim_0_1);

Tensor batch_index_select_dim0_codegen_backward_cuda(
    const Tensor& grad_output,
    const Tensor& dev_weights,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt max_D,
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const int64_t max_segment_length_per_warp,
    const Tensor& grad_offsets,
    const Tensor& total_L_offsets,
    const int32_t fixed_L_per_warp,
    const int32_t num_warps_per_feature,
    const bool permute_output_dim_0_1);

class BatchIndexSelectDim0GPUOp
    : public torch::autograd::Function<BatchIndexSelectDim0GPUOp> {
 public:
  static constexpr bool is_traceable = true;
  static torch::autograd::variable_list forward_impl(
      Tensor inputs,
      Tensor indices,
      c10::SymIntArrayRef _input_num_indices,
      c10::SymIntArrayRef _input_rows,
      c10::SymIntArrayRef _input_columns,
      // Permute dim 0 and 1 of the output tensor
      const bool permute_output_dim_0_1) {
    auto to_vec_int64 =
        [](const c10::SymIntArrayRef& sym_vec) -> std::vector<int64_t> {
      std::vector<int64_t> vec;
      std::transform(
          sym_vec.begin(),
          sym_vec.end(),
          std::back_inserter(vec),
          [](const auto& symint) {
            return symint.guard_int(__FILE__, __LINE__);
          });
      return vec;
    };
    auto input_num_indices = to_vec_int64(_input_num_indices);
    auto input_rows = to_vec_int64(_input_rows);
    auto input_columns = to_vec_int64(_input_columns);
    TORCH_CHECK(input_num_indices.size() == _input_num_indices.size());

    // From the empirical study, this value provides the best perf
    constexpr int64_t ROWS_PER_WARP = 1;
    const int64_t num_inputs = input_num_indices.size();

    TORCH_CHECK(
        num_inputs == static_cast<int64_t>(input_rows.size()),
        "[batch_index_select_dim0] input_rows must have the same length as "
        "input_num_indices.");
    TORCH_CHECK(
        num_inputs == static_cast<int64_t>(input_columns.size()),
        "[batch_index_select_dim0] input_columns must have the same length as "
        "input_num_indices.");
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(inputs, indices);

    TORCH_CHECK(
        reinterpret_cast<uint64_t>(inputs.data_ptr()) % 16 == 0,
        "Currently batch_index_select only supports 16-byte align input tensors");

    const auto int_opts = torch::TensorOptions().dtype(torch::kInt64);
    const auto num_cols =
        torch::from_blob(input_columns.data(), {num_inputs}, int_opts);
    const auto max_col = num_inputs > 0 ? num_cols.max().item<int64_t>() : 0;
    const auto input_num_rows =
        torch::from_blob(input_rows.data(), {num_inputs}, int_opts);
    const auto output_num_rows =
        torch::from_blob(input_num_indices.data(), {num_inputs}, int_opts);

    if (num_inputs > 0) {
      TORCH_CHECK(
          torch::all(torch::gt(num_cols, 0)).item<bool>(),
          "[batch_index_select_dim0] All input_columns must be the same.");
      TORCH_CHECK(
          torch::all(torch::gt(input_num_rows, 0)).item<bool>(),
          "[batch_index_select_dim0] All input_rows must be the same.");
      if (permute_output_dim_0_1) {
        // All output rows must be the same
        TORCH_CHECK(input_num_indices[0] > 0);
        TORCH_CHECK(
            torch::all(torch::eq(output_num_rows, input_num_indices[0]))
                .item<bool>(),
            "[batch_index_select_dim0] All input_num_indices must be the same if "
            "permute_output_dim_0_1 is true.");
      } else {
        TORCH_CHECK(
            torch::all(torch::gt(output_num_rows, 0)).item<bool>(),
            "[batch_index_select_dim0] All input_num_indices must be greater than zero.");
      }
    }

    const auto max_output_num_rows =
        num_inputs > 0 ? output_num_rows.max().item<int64_t>() : 0;

    const auto input_numels = input_num_rows * num_cols;
    const auto output_numels =
        permute_output_dim_0_1 ? Tensor() : (output_num_rows * num_cols);

    // Takes ~1.2 ms for num_inputs = 1024 on CPU
    auto D_offsets = fbgemm_gpu::asynchronous_complete_cumsum_cpu(num_cols).to(
        torch::kInt32);
    auto input_offsets =
        fbgemm_gpu::asynchronous_complete_cumsum_cpu(input_numels);
    auto input_row_offsets =
        fbgemm_gpu::asynchronous_complete_cumsum_cpu(input_num_rows);
    auto total_L_offsets =
        fbgemm_gpu::asynchronous_complete_cumsum_cpu(output_num_rows);
    int64_t total_hash_size_bits =
        std::log2(static_cast<float>(input_row_offsets[-1].item<int64_t>())) +
        1;
    input_offsets =
        torch::narrow(input_offsets, 0, 0, input_offsets.numel() - 1);
    const int64_t num_warps_per_input =
        (max_output_num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;

    // Transfer helper tensors to GPU
    const auto device = inputs.device();
    constexpr bool non_blocking = true;

    int64_t output_size;
    Tensor output_offsets;
    if (permute_output_dim_0_1) {
      // output_offsets is not required because the output tensor is not jagged
      output_offsets = at::empty({0}, inputs.options().dtype(at::kLong));
      output_size = num_inputs > 0
          ? (input_num_indices[0] * D_offsets[-1].item<int64_t>())
          : 0;
    } else {
      output_offsets =
          fbgemm_gpu::asynchronous_complete_cumsum_cpu(output_numels);
      output_size = output_offsets[-1].item<int64_t>();
      output_offsets = output_offsets.to(device, non_blocking);
    }

    D_offsets = D_offsets.to(device, non_blocking);
    input_offsets = input_offsets.to(device, non_blocking);
    input_row_offsets = input_row_offsets.to(device, non_blocking);
    total_L_offsets = total_L_offsets.to(device, non_blocking);

    const auto sparse_type = fbgemm_gpu::getSparseType(inputs.scalar_type());
    TORCH_CHECK(
        sparse_type == SparseType::FP32 || sparse_type == SparseType::FP16,
        "batch_index_select_dim0 supports only either float or half")

    const auto output_dtype =
        static_cast<int64_t>(fbgemm_gpu::getSparseType(inputs.scalar_type()));

    const auto max_D = max_col;
    const auto fixed_L_per_warp = ROWS_PER_WARP;

    auto output = inputs.numel() > 0
        ? batch_index_select_dim0_codegen_forward_cuda(
              inputs, // dev_weights
              input_offsets, // weights_offsets
              D_offsets,
              max_D,
              indices,
              output_dtype,
              output_offsets,
              total_L_offsets,
              output_size,
              fixed_L_per_warp,
              num_warps_per_input, // num_warps_per_feature
              permute_output_dim_0_1)
        : at::empty({0}, inputs.options());

    int64_t saved_data[] = {
        max_D,
        total_hash_size_bits,
        fixed_L_per_warp,
        num_warps_per_input,
    };

    auto saved_data_tensor = at::empty(
        {sizeof(saved_data) / sizeof(int64_t)},
        at::TensorOptions().dtype(at::kLong));
    TORCH_CHECK(saved_data_tensor.is_contiguous());
    memcpy(
        saved_data_tensor.data_ptr<int64_t>(), saved_data, sizeof(saved_data));

    return {
        output, // 0:op_output
        input_offsets, // 1:weights_offsets
        input_row_offsets, // 2:hash_size_cumsum,
        D_offsets, // 3:D_offsets,
        output_offsets, // 4:output_offsets,
        total_L_offsets, // 5:total_L_offsets
        saved_data_tensor, // 6:saved_data_tensor
    };
  }

  // make scheme the same as main op
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      Tensor inputs,
      Tensor indices,
      c10::SymIntArrayRef input_num_indices,
      c10::SymIntArrayRef input_rows,
      c10::SymIntArrayRef input_columns,
      // Permute dim 0 and 1 of the output tensor
      const bool permute_output_dim_0_1) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto forward_op_impl =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batch_index_select_dim0_forward_cuda_impl", "")
            .typed<decltype(forward_impl)>();

    auto res = forward_op_impl.call(
        inputs,
        indices,
        input_num_indices,
        input_rows,
        input_columns,
        permute_output_dim_0_1);

    // 0:op_output
    // 1:weights_offsets,
    // 2:hash_size_cumsum,
    // 3:D_offsets,
    // 4:output_offsets,
    // 5:total_L_offsets
    // 6:saved_data_tensor = [max_D, total_hash_size_bits, fixed_L_per_warp,
    // num_warps_per_input]

    ctx->saved_data["permute_output_dim_0_1"] = permute_output_dim_0_1;

    ctx->save_for_backward(std::vector<Tensor>{
        inputs, indices, res[1], res[2], res[3], res[4], res[5], res[6]});

    res.resize(1);
    return res;
  }

  static Tensor backward_impl(
      const Tensor& grad_output,
      const Tensor& dev_weights,
      const Tensor& weights_offsets,
      const Tensor& D_offsets,
      const Tensor& hash_size_cumsum,
      const Tensor& indices,
      const int64_t max_segment_length_per_warp,
      const Tensor& grad_offsets,
      const Tensor& total_L_offsets,
      const bool permute_output_dim_0_1,
      const Tensor& saved_tensor) {
    if (dev_weights.numel() == 0) {
      return at::empty({0}, dev_weights.options());
    }

    auto _grad_output = grad_output;
    // FIXME: to support aligned memory access in Vec4T load/store function
    // 16 for FP32 and 8 for FP16
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
        at::has_internal_overlap(grad_output) != at::MemOverlap::No) {
      _grad_output = at::empty_like(grad_output).copy_(grad_output);
    }

    const auto max_D = saved_tensor[0].item<int64_t>();
    const auto total_hash_size_bits = saved_tensor[1].item<int64_t>();
    const auto fixed_L_per_warp = saved_tensor[2].item<int64_t>();
    const auto num_warps_per_feature = saved_tensor[3].item<int64_t>();

    return batch_index_select_dim0_codegen_backward_cuda(
        _grad_output,
        dev_weights,
        weights_offsets,
        D_offsets,
        max_D,
        hash_size_cumsum,
        total_hash_size_bits,
        indices,
        max_segment_length_per_warp,
        grad_offsets,
        total_L_offsets,
        fixed_L_per_warp,
        num_warps_per_feature,
        permute_output_dim_0_1);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto dev_weights = *savedItr++; // inputs
    auto indices = *savedItr++; // indices

    auto weights_offsets = *savedItr++;
    auto hash_size_cumsum = *savedItr++;
    auto D_offsets = *savedItr++;
    auto grad_offsets = *savedItr++;
    auto total_L_offsets = *savedItr++;

    auto saved_tensor = *savedItr++;

    const auto permute_output_dim_0_1 =
        ctx->saved_data["permute_output_dim_0_1"].toBool();

    using torch::autograd::Variable;

    Tensor grad_dev_weights;
    TORCH_CHECK_EQ(grad_outputs.size(), 1);

    constexpr int32_t max_segment_length_per_warp = 32;

    auto grad_output = grad_outputs[0];

    static auto backward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batch_index_select_dim0_backward_cuda_impl", "")
            .typed<decltype(backward_impl)>();

    auto res = backward_op.call(
        grad_output,
        dev_weights,
        weights_offsets,
        D_offsets,
        hash_size_cumsum,
        indices,
        max_segment_length_per_warp,
        grad_offsets,
        total_L_offsets,
        permute_output_dim_0_1,
        saved_tensor);

    return {
        res, // inputs
        Variable(), // indices
        Variable(), // input_num_indices
        Variable(), // input_rows
        Variable(), // input_columns
        Variable(), // permute_output_dim_0_1
    };
  }
};

Tensor batch_index_select_dim0_gpu(
    Tensor inputs,
    Tensor indices,
    c10::SymIntArrayRef input_num_indices,
    c10::SymIntArrayRef input_rows,
    c10::SymIntArrayRef input_columns,
    // Permute dim 0 and 1 of the output tensor
    const bool permute_output_dim_0_1) {
  return BatchIndexSelectDim0GPUOp::apply(
      inputs,
      indices,
      input_num_indices,
      input_rows,
      input_columns,
      permute_output_dim_0_1)[0];
}

class BatchIndexSelectDim0TensorGPUOp
    : public torch::autograd::Function<BatchIndexSelectDim0TensorGPUOp> {
 public:
  static torch::autograd::variable_list forward_impl(
      const Tensor& inputs,
      const Tensor& indices,
      const Tensor& input_num_indices,
      const Tensor& input_rows,
      const Tensor& input_columns,
      // Permute dim 0 and 1 of the output tensor
      const bool permute_output_dim_0_1) {
    // From the empirical study, this value provides the best perf
    constexpr int64_t ROWS_PER_WARP = 1;
    const int64_t num_inputs = input_num_indices.size(0);

    TORCH_CHECK(
        num_inputs == input_rows.size(0),
        "[batch_index_select_dim0] input_rows must have the same length as "
        "input_num_indices.");
    TORCH_CHECK(
        num_inputs == input_columns.size(0),
        "[batch_index_select_dim0] input_columns must have the same length as "
        "input_num_indices.");
    TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(inputs, indices);

    TORCH_CHECK(
        reinterpret_cast<uint64_t>(inputs.data_ptr()) % 16 == 0,
        "Currently batch_index_select only supports 16-byte align input tensors");

    const auto num_cols = input_columns;
    const auto max_col =
        num_inputs > 0 ? input_columns.max().item<int64_t>() : 0;
    const auto input_num_rows = input_rows;
    const auto output_num_rows = input_num_indices;

    if (num_inputs > 0) {
      TORCH_CHECK(
          torch::all(torch::gt(input_columns, 0)).item<bool>(),
          "[batch_index_select_dim0] All input_columns must be the same.");
      TORCH_CHECK(
          torch::all(torch::gt(input_num_rows, 0)).item<bool>(),
          "[batch_index_select_dim0] All input_rows must be the same.");
      if (permute_output_dim_0_1) {
        // All output rows must be the same
        TORCH_CHECK(input_num_indices[0].item<int64_t>() > 0);
        TORCH_CHECK(
            torch::all(torch::eq(output_num_rows, input_num_indices[0]))
                .item<bool>(),
            "[batch_index_select_dim0] All input_num_indices must be the same if "
            "permute_output_dim_0_1 is true.");
      } else {
        TORCH_CHECK(
            torch::all(torch::gt(output_num_rows, 0)).item<bool>(),
            "[batch_index_select_dim0] All input_num_indices must be greater than zero.");
      }
    }

    const auto max_output_num_rows =
        num_inputs > 0 ? output_num_rows.max().item<int64_t>() : 0;

    const auto input_numels = input_num_rows * num_cols;
    const auto output_numels =
        permute_output_dim_0_1 ? Tensor() : (output_num_rows * num_cols);

    // Takes ~1.2 ms for num_inputs = 1024 on CPU
    auto D_offsets = fbgemm_gpu::asynchronous_complete_cumsum_cpu(num_cols).to(
        torch::kInt32);
    auto input_offsets =
        fbgemm_gpu::asynchronous_complete_cumsum_cpu(input_numels);
    auto input_row_offsets =
        fbgemm_gpu::asynchronous_complete_cumsum_cpu(input_num_rows);
    auto total_L_offsets =
        fbgemm_gpu::asynchronous_complete_cumsum_cpu(output_num_rows);
    int64_t total_hash_size_bits =
        std::log2(static_cast<float>(input_row_offsets[-1].item<int64_t>())) +
        1;
    input_offsets =
        torch::narrow(input_offsets, 0, 0, input_offsets.numel() - 1);
    const int64_t num_warps_per_input =
        (max_output_num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;

    // Transfer helper tensors to GPU
    const auto device = inputs.device();
    constexpr bool non_blocking = true;

    int64_t output_size;
    Tensor output_offsets;
    if (permute_output_dim_0_1) {
      // output_offsets is not required because the output tensor is not jagged
      output_offsets = at::empty({0}, inputs.options().dtype(at::kLong));
      output_size = num_inputs > 0 ? (input_num_indices[0].item<int64_t>() *
                                      D_offsets[-1].item<int64_t>())
                                   : 0;
    } else {
      output_offsets =
          fbgemm_gpu::asynchronous_complete_cumsum_cpu(output_numels);
      output_size = output_offsets[-1].item<int64_t>();
      output_offsets = output_offsets.to(device, non_blocking);
    }

    D_offsets = D_offsets.to(device, non_blocking);
    input_offsets = input_offsets.to(device, non_blocking);
    input_row_offsets = input_row_offsets.to(device, non_blocking);
    total_L_offsets = total_L_offsets.to(device, non_blocking);

    const auto sparse_type = fbgemm_gpu::getSparseType(inputs.scalar_type());
    TORCH_CHECK(
        sparse_type == SparseType::FP32 || sparse_type == SparseType::FP16,
        "batch_index_select_dim0 supports only either float or half")

    const auto output_dtype =
        static_cast<int64_t>(fbgemm_gpu::getSparseType(inputs.scalar_type()));
    const auto max_D = max_col;
    const auto fixed_L_per_warp = ROWS_PER_WARP;

    auto output = inputs.numel() > 0
        ? batch_index_select_dim0_codegen_forward_cuda(
              inputs, // dev_weights
              input_offsets, // weights_offsets
              D_offsets,
              max_D,
              indices,
              output_dtype,
              output_offsets,
              total_L_offsets,
              output_size,
              fixed_L_per_warp,
              num_warps_per_input, // num_warps_per_feature
              permute_output_dim_0_1)
        : at::empty({0}, inputs.options());

    int64_t saved_data[] = {
        max_D,
        total_hash_size_bits,
        fixed_L_per_warp,
        num_warps_per_input,
    };

    auto saved_data_tensor = at::empty(
        {sizeof(saved_data) / sizeof(int64_t)},
        at::TensorOptions().dtype(at::kLong));
    TORCH_CHECK(saved_data_tensor.is_contiguous());
    memcpy(
        saved_data_tensor.data_ptr<int64_t>(), saved_data, sizeof(saved_data));

    return {
        output, // 0:op_output
        input_offsets, // 1:weights_offsets
        input_row_offsets, // 2:hash_size_cumsum,
        D_offsets, // 3:D_offsets,
        output_offsets, // 4:output_offsets,
        total_L_offsets, // 5:total_L_offsets
        saved_data_tensor, // 6:saved_data_tensor
    };
  }

  // make scheme the same as main op
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& inputs,
      const Tensor& indices,
      const Tensor& input_num_indices,
      const Tensor& input_rows,
      const Tensor& input_columns,
      // Permute dim 0 and 1 of the output tensor
      const bool permute_output_dim_0_1) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto forward_op_impl =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batch_index_select_dim0_tensor_forward_cuda_impl", "")
            .typed<decltype(forward_impl)>();

    auto res = forward_op_impl.call(
        inputs,
        indices,
        input_num_indices,
        input_rows,
        input_columns,
        permute_output_dim_0_1);

    // 0:op_output
    // 1:weights_offsets,
    // 2:hash_size_cumsum,
    // 3:D_offsets,
    // 4:output_offsets,
    // 5:total_L_offsets
    // 6:saved_data_tensor = [max_D, total_hash_size_bits, fixed_L_per_warp,
    // num_warps_per_input]

    ctx->saved_data["permute_output_dim_0_1"] = permute_output_dim_0_1;

    ctx->save_for_backward(std::vector<Tensor>{
        inputs, indices, res[1], res[2], res[3], res[4], res[5], res[6]});

    // res.resize(1);
    return res;
  }

  static Tensor backward_impl(
      const Tensor& grad_output,
      const Tensor& dev_weights,
      const Tensor& weights_offsets,
      const Tensor& D_offsets,
      const Tensor& hash_size_cumsum,
      const Tensor& indices,
      const int64_t max_segment_length_per_warp,
      const Tensor& grad_offsets,
      const Tensor& total_L_offsets,
      const bool permute_output_dim_0_1,
      const Tensor& saved_tensor) {
    if (dev_weights.numel() == 0) {
      return at::empty({0}, dev_weights.options());
    }

    auto _grad_output = grad_output;
    // FIXME: to support aligned memory access in Vec4T load/store function
    // 16 for FP32 and 8 for FP16
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
        at::has_internal_overlap(grad_output) != at::MemOverlap::No) {
      _grad_output = at::empty_like(grad_output).copy_(grad_output);
    }

    const auto max_D = saved_tensor[0].item<int64_t>();
    const auto total_hash_size_bits = saved_tensor[1].item<int64_t>();
    const auto fixed_L_per_warp = saved_tensor[2].item<int64_t>();
    const auto num_warps_per_feature = saved_tensor[3].item<int64_t>();

    return batch_index_select_dim0_codegen_backward_cuda(
        _grad_output,
        dev_weights,
        weights_offsets,
        D_offsets,
        max_D,
        hash_size_cumsum,
        total_hash_size_bits,
        indices,
        max_segment_length_per_warp,
        grad_offsets,
        total_L_offsets,
        fixed_L_per_warp,
        num_warps_per_feature,
        permute_output_dim_0_1);
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto dev_weights = *savedItr++; // inputs
    auto indices = *savedItr++; // indices

    auto weights_offsets = *savedItr++;
    auto hash_size_cumsum = *savedItr++;
    auto D_offsets = *savedItr++;
    auto grad_offsets = *savedItr++;
    auto total_L_offsets = *savedItr++;

    auto saved_tensor = *savedItr++;

    const auto permute_output_dim_0_1 =
        ctx->saved_data["permute_output_dim_0_1"].toBool();

    constexpr int32_t max_segment_length_per_warp = 32;

    auto grad_output = grad_outputs[0];

    static auto backward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::batch_index_select_dim0_tensor_backward_cuda_impl", "")
            .typed<decltype(backward_impl)>();

    auto res = backward_op.call(
        grad_output,
        dev_weights,
        weights_offsets,
        D_offsets,
        hash_size_cumsum,
        indices,
        max_segment_length_per_warp,
        grad_offsets,
        total_L_offsets,
        permute_output_dim_0_1,
        saved_tensor);

    using torch::autograd::Variable;
    return {
        std::move(res), // inputs
        Variable(), // indices
        Variable(), // input_num_indices
        Variable(), // input_rows
        Variable(), // input_columns
        Variable(), // permute_output_dim_0_1
    };
  }
};

Tensor batch_index_select_dim0_tensor_gpu(
    const Tensor& inputs,
    const Tensor& indices,
    const Tensor& input_num_indices,
    const Tensor& input_rows,
    const Tensor& input_columns,
    // Permute dim 0 and 1 of the output tensor
    const bool permute_output_dim_0_1) {
  return BatchIndexSelectDim0TensorGPUOp::apply(
      inputs,
      indices,
      input_num_indices,
      input_rows,
      input_columns,
      permute_output_dim_0_1)[0];
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
  DISPATCH_TO_CUDA("batch_index_select_dim0", batch_index_select_dim0_gpu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA(
      "batch_index_select_dim0_forward_cuda_impl",
      BatchIndexSelectDim0GPUOp::forward_impl);
  DISPATCH_TO_CUDA(
      "batch_index_select_dim0_backward_cuda_impl",
      BatchIndexSelectDim0GPUOp::backward_impl);
  DISPATCH_TO_AUTOGRAD_CUDA(
      "batch_index_select_dim0", batch_index_select_dim0_gpu);
  DISPATCH_TO_CUDA(
      "batch_index_select_dim0_tensor_forward_cuda_impl",
      BatchIndexSelectDim0TensorGPUOp::forward_impl);
  DISPATCH_TO_CUDA(
      "batch_index_select_dim0_tensor_backward_cuda_impl",
      BatchIndexSelectDim0TensorGPUOp::backward_impl);
  DISPATCH_TO_AUTOGRAD_CUDA(
      "batch_index_select_dim0_tensor", batch_index_select_dim0_tensor_gpu);
}
