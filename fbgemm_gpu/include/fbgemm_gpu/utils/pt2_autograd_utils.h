/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>

using Tensor = at::Tensor;

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////

Tensor reshape_vbe_output(
    const Tensor& grad_output,
    const int64_t max_B,
    const Tensor& B_offsets_rank_per_feature,
    const Tensor& D_offsets);

template <typename index_t>
Tensor reshape_vbe_offsets(
    const Tensor& offsets,
    const Tensor& B_offsets_rank_per_feature,
    const int64_t max_B,
    const int32_t T);
} // namespace fbgemm_gpu
