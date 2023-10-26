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

DLL_PUBLIC void lru_cache_populate_byte_cpu(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    int64_t row_alignment,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats) {
  return;
}

DLL_PUBLIC void direct_mapped_lru_cache_populate_byte_cpu(
    Tensor weights,
    Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    Tensor cache_index_table_map,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    Tensor lxu_cache_weights,
    int64_t time_stamp,
    Tensor lru_state,
    Tensor lxu_cache_miss_timestamp,
    int64_t row_alignment,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats) {
  return;
}

} // namespace fbgemm_gpu
