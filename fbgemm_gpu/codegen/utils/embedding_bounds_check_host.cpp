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

#include "fbgemm_gpu/config/feature_gates.h"

using Tensor = at::Tensor;

/// @defgroup embedding-cuda Embedding CUDA Operators

void _bounds_check_indices_cuda_v1(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    int64_t bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& B_offsets,
    const int64_t max_B,
    const std::optional<Tensor>& b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask);

void _bounds_check_indices_cuda_v2(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    int64_t bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& B_offsets,
    const int64_t max_B,
    const std::optional<Tensor>& b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask);

///@ingroup embedding-cuda
void bounds_check_indices_cuda(
    Tensor& rows_per_table,
    Tensor& indices,
    Tensor& offsets,
    int64_t bounds_check_mode,
    Tensor& warning,
    const std::optional<Tensor>& weights,
    const std::optional<Tensor>& B_offsets,
    const int64_t max_B,
    const std::optional<Tensor>& b_t_map,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask,
    const int8_t bounds_check_version) {
  TORCH_CHECK(bounds_check_version == 1 || bounds_check_version == 2);
  const static bool use_v2 =
      fbgemm_gpu::config::is_feature_enabled(
          fbgemm_gpu::config::FeatureGateName::BOUNDS_CHECK_INDICES_V2) ||
      bounds_check_version == 2;
  const auto bounds_check_indices_fn =
      use_v2 ? _bounds_check_indices_cuda_v2 : _bounds_check_indices_cuda_v1;
  bounds_check_indices_fn(
      rows_per_table,
      indices,
      offsets,
      bounds_check_mode,
      warning,
      weights,
      B_offsets,
      max_B,
      b_t_map,
      static_cast<int32_t>(info_B_num_bits),
      static_cast<uint32_t>(info_B_mask));
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
