/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

Tensor batch_index_select_dim0_codegen_forward_cuda(
    const Tensor& dev_weights,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const int64_t max_D,
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
    const int64_t max_D,
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
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const int64_t output_dtype,
      const Tensor& dev_weights,
      const Tensor& weights_offsets,
      const Tensor& hash_size_cumsum,
      const int64_t total_hash_size_bits,
      const Tensor& indices,
      const Tensor& D_offsets,
      const int64_t max_D,
      const Tensor& output_offsets,
      const Tensor& total_L_offsets,
      const int64_t output_size,
      const int64_t fixed_L_per_warp,
      const int64_t num_warps_per_feature,
      const bool permute_output_dim_0_1) {
    ctx->save_for_backward(
        {dev_weights,
         weights_offsets,
         hash_size_cumsum,
         indices,
         D_offsets,
         output_offsets,
         total_L_offsets});

    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
    ctx->saved_data["fixed_L_per_warp"] = fixed_L_per_warp;
    ctx->saved_data["num_warps_per_feature"] = num_warps_per_feature;
    ctx->saved_data["permute_output_dim_0_1"] = permute_output_dim_0_1;

    // Early exit
    if (dev_weights.numel() == 0) {
      return {at::empty({0}, dev_weights.options())};
    }

    return {batch_index_select_dim0_codegen_forward_cuda(
        dev_weights,
        weights_offsets,
        D_offsets,
        max_D,
        indices,
        output_dtype,
        output_offsets,
        total_L_offsets,
        output_size,
        fixed_L_per_warp,
        num_warps_per_feature,
        permute_output_dim_0_1)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto dev_weights = *savedItr++;
    auto weights_offsets = *savedItr++;
    auto hash_size_cumsum = *savedItr++;
    auto indices = *savedItr++;
    auto D_offsets = *savedItr++;
    auto grad_offsets = *savedItr++;
    auto total_L_offsets = *savedItr++;

    const auto max_D = ctx->saved_data["max_D"].toInt();
    const auto total_hash_size_bits =
        ctx->saved_data["total_hash_size_bits"].toInt();
    const auto fixed_L_per_warp = ctx->saved_data["fixed_L_per_warp"].toInt();
    const auto num_warps_per_feature =
        ctx->saved_data["num_warps_per_feature"].toInt();
    const auto permute_output_dim_0_1 =
        ctx->saved_data["permute_output_dim_0_1"].toBool();

    using torch::autograd::Variable;

    Tensor grad_dev_weights;
    if (dev_weights.numel() == 0) {
      grad_dev_weights = at::empty({0}, dev_weights.options());
    } else {
      TORCH_CHECK_EQ(grad_outputs.size(), 1);

      constexpr int32_t max_segment_length_per_warp = 32;

      auto grad_output = grad_outputs[0];
      // FIXME: to support aligned memory access in Vec4T load/store function
      // 16 for FP32 and 8 for FP16
      if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0) {
        grad_output = at::empty_like(grad_output).copy_(grad_output);
      }

      grad_dev_weights = batch_index_select_dim0_codegen_backward_cuda(
          grad_output,
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

    return {
        Variable(), // output_dtype
        grad_dev_weights, // grad_dev_weights
        Variable(), // weights_offsets
        Variable(), // hash_size_cumsum
        Variable(), // total_hash_size_bits
        Variable(), // indices
        Variable(), // D_offsets
        Variable(), // max_D
        Variable(), // output_offsets
        Variable(), // total_L_offsets
        Variable(), // output_size
        Variable(), // fixed_L_per_warp
        Variable(), // num_warps_per_feature
        Variable(), // permute_output_dim_0_1
    };
  }
};

Tensor batch_index_select_dim0_gpu(
    Tensor inputs,
    Tensor indices,
    std::vector<int64_t> input_num_indices,
    std::vector<int64_t> input_rows,
    std::vector<int64_t> input_columns,
    // Permute dim 0 and 1 of the output tensor
    const bool permute_output_dim_0_1) {
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
  auto D_offsets =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(num_cols).to(torch::kInt32);
  auto input_offsets =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(input_numels);
  auto input_row_offsets =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(input_num_rows);
  auto total_L_offsets =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(output_num_rows);
  int64_t total_hash_size_bits =
      std::log2(static_cast<float>(input_row_offsets[-1].item<int64_t>())) + 1;
  input_offsets = torch::narrow(input_offsets, 0, 0, input_offsets.numel() - 1);

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

  // Call TBE
  return BatchIndexSelectDim0GPUOp::apply(
      static_cast<int64_t>(fbgemm_gpu::getSparseType(inputs.scalar_type())),
      inputs,
      input_offsets,
      input_row_offsets,
      total_hash_size_bits,
      indices,
      D_offsets,
      max_col,
      output_offsets,
      total_L_offsets,
      output_size,
      ROWS_PER_WARP, // fixed_L_per_warp
      num_warps_per_input,
      permute_output_dim_0_1)[0];
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
  DISPATCH_TO_CUDA("batch_index_select_dim0", batch_index_select_dim0_gpu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA("batch_index_select_dim0", batch_index_select_dim0_gpu);
}
