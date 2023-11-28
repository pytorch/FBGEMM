/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <limits>
#include <mutex>

#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "fbgemm_gpu/ops_utils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

Tensor linearize_cache_indices_cpu(
    Tensor cache_hash_size_cumsum,
    Tensor indices,
    Tensor offsets);

Tensor linearize_cache_indices_from_row_idx_cpu(
    Tensor cache_hash_size_cumsum,
    Tensor update_table_indices,
    Tensor update_row_indices);

void lru_cache_populate_byte_cpu(
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
    c10::optional<Tensor> uvm_cache_stats);

void direct_mapped_lru_cache_populate_byte_cpu(
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
    c10::optional<Tensor> uvm_cache_stats);

void lfu_cache_populate_byte_cpu(
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
    Tensor lfu_state,
    int64_t row_alignment);

Tensor lxu_cache_lookup_cpu(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    int64_t invalid_index,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats,
    c10::optional<Tensor> num_uniq_cache_indices,
    c10::optional<Tensor> lxu_cache_locations_output);

Tensor direct_mapped_lxu_cache_lookup_cpu(
    Tensor linear_cache_indices,
    Tensor lxu_cache_state,
    int64_t invalid_index,
    bool gather_cache_stats,
    c10::optional<Tensor> uvm_cache_stats);

} // namespace fbgemm_gpu
