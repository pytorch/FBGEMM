/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

DLL_PUBLIC Tensor linearize_cache_indices_cpu(
    const Tensor& /*cache_hash_size_cumsum*/,
    const Tensor& indices,
    const Tensor& /*offsets*/,
    const c10::optional<Tensor>& /*B_offsets*/,
    const int64_t /*max_B*/) {
  return at::empty_like(indices);
}

DLL_PUBLIC Tensor linearize_cache_indices_from_row_idx_cpu(
    Tensor /*cache_hash_size_cumsum*/,
    Tensor /*update_table_indices*/,
    Tensor update_row_indices) {
  return at::empty_like(update_row_indices);
}

} // namespace fbgemm_gpu
