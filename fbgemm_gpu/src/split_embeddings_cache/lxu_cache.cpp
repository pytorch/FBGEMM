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

DLL_PUBLIC Tensor lxu_cache_lookup_cpu(
    Tensor linear_cache_indices,
    Tensor /* lxu_cache_state */,
    int64_t /* invalid_index */,
    bool /* gather_cache_stats */,
    c10::optional<Tensor> /* uvm_cache_stats */,
    c10::optional<Tensor> /* num_uniq_cache_indices */,
    c10::optional<Tensor> lxu_cache_locations_output) {
  return lxu_cache_locations_output.value_or(empty_like(
      linear_cache_indices, linear_cache_indices.options().dtype(at::kInt)));
}

DLL_PUBLIC Tensor direct_mapped_lxu_cache_lookup_cpu(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    int64_t invalid_index,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats) {
  return empty_like(
      linear_cache_indices, linear_cache_indices.options().dtype(at::kInt));
}

} // namespace fbgemm_gpu
