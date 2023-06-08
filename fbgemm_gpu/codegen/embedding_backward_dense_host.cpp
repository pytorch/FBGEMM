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
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

Tensor dense_embedding_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t output_dtype,
    bool is_experimental);

Tensor dense_embedding_codegen_forward_weighted_cuda(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t output_dtype,
    bool is_experimental);

Tensor dense_embedding_codegen_grad_indice_weights_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    Tensor feature_requires_grad);

Tensor split_embedding_backward_codegen_dense_unweighted_exact_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t BT_block_size,
    int64_t max_segment_length_per_warp,
    double unused);

Tensor split_embedding_backward_codegen_dense_weighted_exact_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int64_t BT_block_size,
    int64_t max_segment_length_per_warp,
    double unused);

class SplitLookupFunction_Dense_Op
    : public torch::autograd::Function<SplitLookupFunction_Dense_Op> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      int64_t output_dtype,
      Tensor dev_weights,
      Tensor weights_offsets,
      Tensor D_offsets,
      int64_t total_D,
      int64_t max_D,
      Tensor hash_size_cumsum,
      int64_t total_hash_size_bits,
      Tensor indices,
      Tensor offsets,
      int64_t pooling_mode,
      c10::optional<Tensor> indice_weights,
      c10::optional<Tensor> feature_requires_grad) {
    ctx->save_for_backward({
        dev_weights,
        weights_offsets,
        D_offsets,
        hash_size_cumsum,
        indices,
        offsets,
        indice_weights.value_or(Tensor()),
        feature_requires_grad.value_or(Tensor()),
    });

    ctx->saved_data["total_D"] = total_D;
    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
    ctx->saved_data["pooling_mode"] = pooling_mode;

    if (!indice_weights.has_value()) {
      return {dense_embedding_codegen_forward_unweighted_cuda(
          dev_weights,
          weights_offsets,
          D_offsets,
          total_D,
          max_D,
          indices,
          offsets,
          pooling_mode,
          output_dtype,
          /*is_experimental=*/false)};
    } else {
      return {dense_embedding_codegen_forward_weighted_cuda(
          dev_weights,
          weights_offsets,
          D_offsets,
          total_D,
          max_D,
          indices,
          offsets,
          pooling_mode,
          indice_weights.value(),
          output_dtype,
          /*is_experimental=*/false)};
    }
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto dev_weights = *savedItr++;
    auto weights_offsets = *savedItr++;
    auto D_offsets = *savedItr++;
    auto hash_size_cumsum = *savedItr++;
    auto indices = *savedItr++;
    auto offsets = *savedItr++;
    auto indice_weights = *savedItr++;
    auto feature_requires_grad = *savedItr++;

    auto total_D = ctx->saved_data["total_D"].toInt();
    auto max_D = ctx->saved_data["max_D"].toInt();
    auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();
    auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();

    TORCH_CHECK_EQ(grad_outputs.size(), 1);

#ifdef __HIP_PLATFORM_HCC__
    constexpr int32_t BT_block_size = 64;
    constexpr int32_t max_segment_length_per_warp = 64;
#else
    constexpr int32_t BT_block_size = 32;
    constexpr int32_t max_segment_length_per_warp = 32;
#endif
    using torch::autograd::Variable;

    auto grad_output = grad_outputs[0];
    // FIXME: to support aligned memory access in Vec4T load/store function
    // 16 for FP32 and 8 for FP16
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
        grad_output.stride(1) != 1 || grad_output.stride(0) % 4 != 0) {
      grad_output = grad_output.contiguous();
    }
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0) {
      grad_output = at::empty_like(grad_output).copy_(grad_output);
    }

    if (!indice_weights.defined()) {
      auto grad_dev_weights =
          split_embedding_backward_codegen_dense_unweighted_exact_cuda(
              grad_output,
              dev_weights,
              weights_offsets,
              D_offsets,
              max_D,
              hash_size_cumsum,
              total_hash_size_bits,
              indices,
              offsets,
              pooling_mode,
              BT_block_size,
              max_segment_length_per_warp,
              /* unused=*/0.0);
      return {
          Variable(), // output_dtype
          grad_dev_weights,
          Variable(), // weights_offsets
          Variable(), // D_offsets
          Variable(), // total_D
          Variable(), // max_D
          Variable(), // hash_size_cumsum
          Variable(), // total_hash_size_bits
          Variable(), // indices
          Variable(), // offsets
          Variable(), // pooling_mode
          Variable(), // indice_weights
          Variable(), // feature_requires_grad
      };
    } else {
      auto grad_indice_weights =
          dense_embedding_codegen_grad_indice_weights_cuda(
              grad_output,
              dev_weights,
              weights_offsets,
              D_offsets,
              max_D,
              indices,
              offsets,
              feature_requires_grad);
      auto grad_dev_weights =
          split_embedding_backward_codegen_dense_weighted_exact_cuda(
              grad_output,
              dev_weights,
              weights_offsets,
              D_offsets,
              max_D,
              hash_size_cumsum,
              total_hash_size_bits,
              indices,
              offsets,
              pooling_mode,
              indice_weights,
              BT_block_size,
              max_segment_length_per_warp,
              /* unused=*/0.0);
      return {
          Variable(), // output_dtype
          grad_dev_weights,
          Variable(), // weights_offsets
          Variable(), // D_offsets
          Variable(), // total_D
          Variable(), // max_D
          Variable(), // hash_size_cumsum
          Variable(), // total_hash_size_bits
          Variable(), // indices
          Variable(), // offsets
          Variable(), // pooling_mode
          grad_indice_weights,
          Variable(), // feature_requires_grad
      };
    }
  }
};

/******** nobag ops ********/
Tensor dense_embedding_nobag_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor weights_offsets,
    int64_t D,
    Tensor indices,
    Tensor offsets,
    int64_t output_dtype,
    bool is_experimental);

Tensor split_embedding_nobag_backward_codegen_dense_unweighted_exact_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor weights_offsets,
    int64_t D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t BT_block_size,
    int64_t max_segment_length_per_warp,
    double unused);

class SplitNoBagLookupFunction_Dense_Op
    : public torch::autograd::Function<SplitNoBagLookupFunction_Dense_Op> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      int64_t output_dtype,
      Tensor dev_weights,
      Tensor weights_offsets,
      int64_t D,
      Tensor hash_size_cumsum,
      int64_t total_hash_size_bits,
      Tensor indices,
      Tensor offsets) {
    ctx->save_for_backward({
        dev_weights,
        weights_offsets,
        hash_size_cumsum,
        indices,
        offsets,
    });

    ctx->saved_data["D"] = D;
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;

    return {dense_embedding_nobag_codegen_forward_unweighted_cuda(
        dev_weights,
        weights_offsets,
        D,
        indices,
        offsets,
        output_dtype,
        /*is_experimental*/ false)};
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
    auto offsets = *savedItr++;

    auto D = ctx->saved_data["D"].toInt();
    auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();

    TORCH_CHECK_EQ(grad_outputs.size(), 1);

    constexpr int32_t BT_block_size = 32;
    constexpr int32_t max_segment_length_per_warp = 32;
    using torch::autograd::Variable;

    auto grad_output = grad_outputs[0];
    // FIXME: to support aligned memory access in Vec4T load/store function
    // 16 for FP32 and 8 for FP16
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
        grad_output.stride(1) != 1 || grad_output.stride(0) % 4 != 0) {
      grad_output = grad_output.contiguous();
    }
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0) {
      grad_output = at::empty_like(grad_output).copy_(grad_output);
    }

    auto grad_dev_weights =
        split_embedding_nobag_backward_codegen_dense_unweighted_exact_cuda(
            grad_output,
            dev_weights,
            weights_offsets,
            D,
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            BT_block_size,
            max_segment_length_per_warp,
            0);
    return {
        Variable(), // output_dtype
        grad_dev_weights, // grad_dev_weights
        Variable(), // weights_offsets
        Variable(), // D
        Variable(), // hash_size_cumsum
        Variable(), // total_hash_size_bits
        Variable(), // indices
        Variable(), // offsets
    };
  }
};

Tensor split_embedding_codegen_lookup_dense_function(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    c10::optional<Tensor> feature_requires_grad,
    int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)) {
  if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
    return SplitNoBagLookupFunction_Dense_Op::apply(
        output_dtype,
        dev_weights,
        weights_offsets,
        max_D,
        hash_size_cumsum,
        total_hash_size_bits,
        indices,
        offsets)[0];
  } else {
    return SplitLookupFunction_Dense_Op::apply(
        output_dtype,
        dev_weights,
        weights_offsets,
        D_offsets,
        total_D,
        max_D,
        hash_size_cumsum,
        total_hash_size_bits,
        indices,
        offsets,
        pooling_mode,
        indice_weights,
        feature_requires_grad)[0];
  }
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
  DISPATCH_TO_CUDA(
      "dense_embedding_codegen_lookup_function",
      split_embedding_codegen_lookup_dense_function);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA(
      "dense_embedding_codegen_lookup_function",
      split_embedding_codegen_lookup_dense_function);
}
