/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/torch.h>
#include <tuple>
#include "indices_estimator.h"

namespace fbgemm_gpu::tbe {

/// @ingroup tbe-eeg
///
/// @brief Torch interface function for estimating the distribution of TBE
/// indices from a given set of indices.
///
/// @param indices A tensor of either int32_t or int64_t indices
///
/// @return A tuple of parameters that describe the distribution of TBE indices
///     - A tensor of probabilities describing the heavy hitters
///     - The Q parameter of a Zipfian distribution
///     - The S parameter of a Zipfian distribution
///     - The maximum index value
///     - The number of indices
std::tuple<torch::Tensor, double, double, int64_t, int64_t>
estimate_indices_distribution(const at::Tensor& indices) {
  TORCH_CHECK(
      indices.numel() > 0, "indices numel is ", indices.numel(), "(< 1)");
  TORCH_CHECK(
      indices.dtype() == at::kInt || indices.dtype() == at::kLong,
      "indices dtype is ",
      indices.dtype(),
      "(!= I32 or I64)");

  const auto params = *(IndicesEstimator(indices).estimate());

  const auto hitters = torch::from_blob(
      (void*)params.heavyHitters.data(),
      params.heavyHitters.size(),
      torch::TensorOptions().dtype(at::kDouble));

  return {
      hitters,
      params.zipfParams.q,
      params.zipfParams.s,
      params.maxIndex,
      params.numIndices,
  };
}

} // namespace fbgemm_gpu::tbe

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "tbe_estimate_indices_distribution("
      "Tensor indices)"
      "-> (Tensor, float, float, int, int)",
      TORCH_FN(fbgemm_gpu::tbe::estimate_indices_distribution));
}
