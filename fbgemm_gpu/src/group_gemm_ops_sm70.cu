/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/group_gemm_ops.cuh"

namespace fbgemm_gpu {

using namespace cutlass;
using Tensor = at::Tensor;

#define INSTANTIATE_GEMM_GROUPED(scalar_t, LayoutB) \
  template std::vector<Tensor>                      \
  gemm_grouped_cuda<scalar_t, LayoutB, arch::Sm70>( \
      const std::vector<Tensor>& a_group,           \
      const std::vector<Tensor>& b_group,           \
      const c10::optional<std::vector<Tensor>>& c_group);

#define INSTANTIATE_LAYOUT(scalar_t)                   \
  INSTANTIATE_GEMM_GROUPED(scalar_t, layout::RowMajor) \
  INSTANTIATE_GEMM_GROUPED(scalar_t, layout::ColumnMajor)

INSTANTIATE_LAYOUT(double)
INSTANTIATE_LAYOUT(float)
INSTANTIATE_LAYOUT(at::Half)

} // namespace fbgemm_gpu
