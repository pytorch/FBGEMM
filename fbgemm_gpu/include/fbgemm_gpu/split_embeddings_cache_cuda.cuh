/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ATen/ATen.h>

///@defgroup table-batched-embed-cuda CUDA Operators
/// The following are CUDA Operators

///@ingroup table-batched-embed-cuda
/// Deduplicate indices.
std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
get_unique_indices_cuda(
    at::Tensor linear_indices,
    int64_t max_indices,
    bool compute_count);

///@ingroup table-batched-embed-cuda
/// Lookup LRU cache to find uncached indices, and then sort them based on the
/// set.
std::pair<at::Tensor, at::Tensor> lru_cache_find_uncached_cuda(
    at::Tensor unique_indices,
    at::Tensor unique_indices_length,
    int64_t max_indices,
    at::Tensor lxu_cache_state,
    int64_t time_stamp,
    at::Tensor lru_state,
    bool gather_cache_stats,
    at::Tensor uvm_cache_stats);

///@ingroup table-batched-embed-cuda
/// Map index to cache_set. h_in: linear_indices; C: #cache_sets.
int64_t host_lxu_cache_slot(int64_t h_in, int64_t C);

///@ingroup table-batched-embed-cuda
/// Linearize the indices of all tables to make it be unique
at::Tensor linearize_cache_indices_cuda(
    at::Tensor cache_hash_size_cumsum,
    at::Tensor indices,
    at::Tensor offsets);

///@ingroup table-batched-embed-cuda
/// Linearize the indices of all tables to make it be unique.
/// Note the update_table_indices and update_row_indices are
/// from the row indices format for inplace update.
at::Tensor linearize_cache_indices_from_row_idx_cuda(
    at::Tensor cache_hash_size_cumsum,
    at::Tensor update_table_indices,
    at::Tensor update_row_indices);

///@ingroup table-batched-embed-cuda
/// LRU cache: fetch the rows corresponding to `linear_cache_indices` from
///`weights`, and insert them into the cache at timestep `time_stamp`.
void lru_cache_populate_cuda(
    at::Tensor weights,
    at::Tensor hash_size_cumsum,
    int64_t total_cache_hash_size,
    at::Tensor cache_index_table_map,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    at::Tensor linear_cache_indices,
    at::Tensor lxu_cache_state,
    at::Tensor lxu_cache_weights,
    int64_t time_stamp,
    at::Tensor lru_state,
    bool stochastic_rounding,
    bool gather_cache_stats,
    c10::optional<at::Tensor> uvm_cache_stats);

///@ingroup table-batched-embed-cuda
/// LRU cache: fetch the rows corresponding to `linear_cache_indices` from
///`weights`, and insert them into the cache at timestep `time_stamp`.
/// weights and lxu_cache_weights have "uint8_t" byte elements
void lru_cache_populate_byte_cuda(
    at::Tensor weights,
    at::Tensor hash_size_cumsum,
    int64_t total_cache_hash_size,
    at::Tensor cache_index_table_map,
    at::Tensor weights_offsets,
    at::Tensor weights_tys,
    at::Tensor D_offsets,
    at::Tensor linear_cache_indices,
    at::Tensor lxu_cache_state,
    at::Tensor lxu_cache_weights,
    int64_t time_stamp,
    at::Tensor lru_state,
    int64_t row_alignment,
    bool gather_cache_stats,
    c10::optional<at::Tensor> uvm_cache_stats);

///@ingroup table-batched-embed-cuda
/// Direct-mapped (assoc=1) variant of lru_cache_populate_byte_cuda
void direct_mapped_lru_cache_populate_byte_cuda(
    at::Tensor weights,
    at::Tensor hash_size_cumsum,
    int64_t total_cache_hash_size,
    at::Tensor cache_index_table_map,
    at::Tensor weights_offsets,
    at::Tensor weights_tys,
    at::Tensor D_offsets,
    at::Tensor linear_cache_indices,
    at::Tensor lxu_cache_state,
    at::Tensor lxu_cache_weights,
    int64_t time_stamp,
    at::Tensor lru_state,
    at::Tensor lxu_cache_miss_timestamp,
    int64_t row_alignment);

///@ingroup table-batched-embed-cuda
/// LFU cache: fetch the rows corresponding to `linear_cache_indices` from
///`weights`, and insert them into the cache.
void lfu_cache_populate_cuda(
    at::Tensor weights,
    at::Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    at::Tensor cache_index_table_map,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    at::Tensor linear_cache_indices,
    at::Tensor lxu_cache_state,
    at::Tensor lxu_cache_weights,
    at::Tensor lfu_state,
    bool stochastic_rounding);

///@ingroup table-batched-embed-cuda
/// LFU cache: fetch the rows corresponding to `linear_cache_indices` from
///`weights`, and insert them into the cache.
/// weights and lxu_cache_weights have "uint8_t" byte elements
void lfu_cache_populate_byte_cuda(
    at::Tensor weights,
    at::Tensor cache_hash_size_cumsum,
    int64_t total_cache_hash_size,
    at::Tensor cache_index_table_map,
    at::Tensor weights_offsets,
    at::Tensor weights_tys,
    at::Tensor D_offsets,
    at::Tensor linear_cache_indices,
    at::Tensor lxu_cache_state,
    at::Tensor lxu_cache_weights,
    at::Tensor lfu_state,
    int64_t row_alignment);

///@ingroup table-batched-embed-cuda
/// Lookup the LRU/LFU cache: find the cache weights location for all indices.
/// Look up the slots in the cache corresponding to `linear_cache_indices`, with
/// a sentinel value for missing.
at::Tensor lxu_cache_lookup_cuda(
    at::Tensor linear_cache_indices,
    at::Tensor lxu_cache_state,
    int64_t invalid_index,
    bool gather_cache_stats,
    c10::optional<at::Tensor> uvm_cache_stats);

///@ingroup table-batched-embed-cuda
/// Lookup the LRU/LFU cache: find the cache weights location for all indices.
/// Look up the slots in the cache corresponding to `linear_cache_indices`, with
/// a sentinel value for missing.
at::Tensor direct_mapped_lxu_cache_lookup_cuda(
    at::Tensor linear_cache_indices,
    at::Tensor lxu_cache_state,
    int64_t invalid_index);

//////@ingroup table-batched-embed-cuda
/// Flush the cache: store the weights from the cache to the backing storage.
void lxu_cache_flush_cuda(
    at::Tensor uvm_weights,
    at::Tensor cache_hash_size_cumsum,
    at::Tensor cache_index_table_map,
    at::Tensor weights_offsets,
    at::Tensor D_offsets,
    int64_t total_D,
    at::Tensor lxu_cache_state,
    at::Tensor lxu_cache_weights,
    bool stochastic_rounding);

///@ingroup table-batched-embed-cuda
// For in-training embedding pruning
// Function to reset the weight and momentum of pruned indices
void reset_weight_momentum_cuda(
    at::Tensor dev_weights,
    at::Tensor uvm_weights,
    at::Tensor lxu_cache_weights,
    at::Tensor weights_placements,
    at::Tensor weights_offsets,
    at::Tensor momentum1_dev,
    at::Tensor momentum1_uvm,
    at::Tensor momentum1_placements,
    at::Tensor momentum1_offsets,
    at::Tensor D_offsets,
    at::Tensor pruned_indices,
    at::Tensor pruned_indices_offsets,
    at::Tensor logical_table_ids,
    at::Tensor buffer_ids,
    at::Tensor cache_hash_size_cumsum,
    at::Tensor lxu_cache_state,
    int64_t total_cache_hash_size);
