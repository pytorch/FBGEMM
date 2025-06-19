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
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/tensor_utils.h"

#include "fbgemm_gpu/config/feature_gates.h"
#include "fbgemm_gpu/embedding_common.h"

using Tensor = at::Tensor;

/// @defgroup embedding-cuda Embedding CUDA Operators

void _bounds_check_indices_cuda_v1(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    fbgemm_gpu::BoundsCheckMode bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& B_offsets,
    int64_t max_B,
    const std::optional<Tensor>& b_t_map,
    int32_t info_B_num_bits,
    uint32_t info_B_mask,
    int64_t T,
    int64_t B,
    int64_t total_B,
    bool vbe,
    bool prefetch_pipeline);

void _bounds_check_indices_cuda_v2(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    fbgemm_gpu::BoundsCheckMode bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& B_offsets,
    int64_t max_B,
    const std::optional<Tensor>& b_t_map,
    int32_t info_B_num_bits,
    uint32_t info_B_mask,
    int64_t T,
    int64_t B,
    int64_t total_B,
    bool vbe,
    bool prefetch_pipeline);

///@ingroup embedding-cuda
void bounds_check_indices_cuda(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    int64_t bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& B_offsets,
    int64_t max_B,
    const std::optional<Tensor>& b_t_map,
    int64_t info_B_num_bits,
    int64_t info_B_mask,
    int8_t bounds_check_version,
    bool prefetch_pipeline) {
  TORCH_CHECK(bounds_check_version == 1 || bounds_check_version == 2);
  const static bool use_v2_jk = fbgemm_gpu::config::is_feature_enabled(
      fbgemm_gpu::config::FeatureGateName::BOUNDS_CHECK_INDICES_V2);
  const auto bounds_check_indices_fn = (use_v2_jk || bounds_check_version == 2)
      ? _bounds_check_indices_cuda_v2
      : _bounds_check_indices_cuda_v1;
  const auto bounds_check_mode_ =
      static_cast<fbgemm_gpu::BoundsCheckMode>(bounds_check_mode);

  TORCH_CHECK(
      bounds_check_mode_ == fbgemm_gpu::BoundsCheckMode::WARNING ||
          bounds_check_mode_ == fbgemm_gpu::BoundsCheckMode::FATAL ||
          bounds_check_mode_ == fbgemm_gpu::BoundsCheckMode::IGNORE,
      "bounds_check_indices: bounds_check_mode=",
      bounds_check_mode,
      " is not supported");

  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      rows_per_table, indices, offsets, warning, weights, B_offsets, b_t_map);

  TENSOR_NDIM_EQUALS(rows_per_table, 1);
  TENSOR_NDIM_EQUALS(indices, 1);
  TENSOR_NDIM_EQUALS(offsets, 1);
  TENSOR_NDIM_EQUALS(warning, 1);

  const auto T = rows_per_table.size(0);
  const auto total_B = offsets.size(0) - 1;
  const auto B = total_B / T;
  if (total_B == 0 || T == 0) {
    return;
  }

  const auto vbe = B_offsets.has_value();
  if (vbe) {
    TENSOR_NDIM_EQUALS(B_offsets.value(), 1);
    TORCH_CHECK(max_B >= 0);
  } else if (!vbe) {
    TORCH_CHECK(
        offsets.size(0) == B * T + 1,
        "offsets size " + std::to_string(offsets.size(0)) +
            " is not equal to B (" + std::to_string(B) + ") * T (" +
            std::to_string(T) + ") + 1");
  }
  if (weights.has_value() && weights->numel() != 0) {
    const auto num_indices = indices.size(0);
    TORCH_CHECK(
        weights->size(0) == num_indices,
        "weights size " + std::to_string(weights->size(0)) +
            " is not equal to indices size " + std::to_string(num_indices));
  }

  bounds_check_indices_fn(
      rows_per_table,
      indices,
      offsets,
      bounds_check_mode_,
      warning,
      weights,
      B_offsets,
      max_B,
      b_t_map,
      static_cast<int32_t>(info_B_num_bits),
      static_cast<uint32_t>(info_B_mask),
      T,
      B,
      total_B,
      vbe,
      prefetch_pipeline);
}
// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
  DISPATCH_TO_CUDA("bounds_check_indices", bounds_check_indices_cuda);
#ifdef HAS_IMPL_ABSTRACT_PYSTUB
  m.impl_abstract_pystub(
      "fbgemm_gpu.sparse_ops",
      "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_py");
#endif
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA("bounds_check_indices", bounds_check_indices_cuda);
#ifdef HAS_IMPL_ABSTRACT_PYSTUB
  m.impl_abstract_pystub(
      "fbgemm_gpu.sparse_ops",
      "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_py");
#endif
}
