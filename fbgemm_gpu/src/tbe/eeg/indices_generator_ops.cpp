/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/torch.h>
#include "indices_generator.h"

namespace fbgemm_gpu::tbe {

at::Tensor generate_indices_from_distribution(
    at::Tensor heavy_hitters,
    double zipf_q,
    double zipf_s,
    int64_t max_index,
    int64_t num_indices) {
  TORCH_CHECK(
      heavy_hitters.dim() == 1,
      "heavy_hitters dim is ",
      heavy_hitters.dim(),
      "(!= 1)");
  TORCH_CHECK(
      heavy_hitters.dtype() == at::kFloat ||
          heavy_hitters.dtype() == at::kDouble,
      "heavy_hitters dtype is ",
      heavy_hitters.dtype(),
      "(!= F32 or F64)");

  // Convert to std::vector<double>
  auto tmp = heavy_hitters.cpu().to(at::kDouble).contiguous();
  const auto heavy_hitters_ = std::vector<double>{
      tmp.data_ptr<double>(), tmp.data_ptr<double>() + tmp.numel()};

  // Build parameters
  const auto params = IndicesDistributionParameters(
      heavy_hitters_, ZipfParameters(zipf_q, zipf_s), max_index, num_indices);

  // Generate and return indices
  return IndicesGenerator(params).generate();
}

} // namespace fbgemm_gpu::tbe

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "tbe_generate_indices_from_distribution("
      "Tensor heavy_hitters, "
      "float zipf_q, "
      "float zipf_s, "
      "int max_index, "
      "int num_indices)"
      "-> Tensor",
      TORCH_FN(fbgemm_gpu::tbe::generate_indices_from_distribution));
}
