/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

namespace {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "linearize_cache_indices(Tensor cache_hash_size_cumsum, Tensor indices, Tensor offsets, Tensor? B_offsets=None, int max_B=-1) -> Tensor");
  m.def(
      "linearize_cache_indices_from_row_idx(Tensor cache_hash_size_cumsum, Tensor update_table_indices, Tensor update_row_indices) -> Tensor");
  m.def(
      "lru_cache_populate(Tensor weights, Tensor hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, int time_stamp, Tensor(c!) lru_state, bool stochastic_rounding, bool gather_cache_stats=False, Tensor(d!)? uvm_cache_stats=None, bool lock_cache_line=False, Tensor(e!)? lxu_cache_locking_counter=None) -> ()");
  m.def(
      "lru_cache_populate_byte(Tensor weights, Tensor hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, int time_stamp, Tensor(c!) lru_state, int row_alignment=16, bool gather_cache_stats=False, Tensor(d!)? uvm_cache_stats=None) -> ()");
  m.def(
      "direct_mapped_lru_cache_populate_byte(Tensor weights, Tensor hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, int time_stamp, Tensor(c!) lru_state, Tensor(d!) lxu_cache_miss_timestamp, int row_alignment=16, bool gather_cache_stats=False, Tensor(e!)? uvm_cache_stats=None) -> ()");
  m.def(
      "lfu_cache_populate(Tensor weights, Tensor cache_hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, Tensor(c!) lfu_state, bool stochastic_rounding) -> ()");
  m.def(
      "lfu_cache_populate_byte(Tensor weights, Tensor cache_hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, Tensor(c!) lfu_state, int row_alignment=16) -> ()");
  m.def(
      "lxu_cache_lookup(Tensor linear_cache_indices, Tensor lxu_cache_state, int invalid_index = -1, bool gather_cache_stats=False, Tensor(a!)? uvm_cache_stats=None, Tensor? num_uniq_cache_indices=None, Tensor(b!)? lxu_cache_locations_output=None) -> Tensor");
  m.def(
      "direct_mapped_lxu_cache_lookup(Tensor linear_cache_indices, Tensor lxu_cache_state, int invalid_index = -1, bool gather_cache_stats=False, Tensor(a!)? uvm_cache_stats=None) -> Tensor");
  m.def(
      "lxu_cache_flush(Tensor(a!) uvm_weights, Tensor cache_hash_size_cumsum, Tensor cache_index_table_map, Tensor weights_offsets, Tensor D_offsets, int total_D, Tensor(b!) lxu_cache_state, Tensor(c!) lxu_cache_weights, bool stochastic_rounding) -> ()");
  m.def("lxu_cache_slot(int h_in, int C) -> int");
  m.def(
      "reset_weight_momentum(Tensor dev_weights, Tensor uvm_weights, Tensor lxu_cache_weights, Tensor weights_placements, Tensor weights_offsets, Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets, Tensor D_offsets, Tensor pruned_indices, Tensor pruned_indices_offsets, Tensor logical_table_ids, Tensor buffer_ids, Tensor cache_hash_size_cumsum, Tensor lxu_cache_state, int total_cache_hash_size) -> ()");
  m.def(
      "lxu_cache_locking_counter_decrement(Tensor(a!) lxu_cache_locking_counter, Tensor lxu_cache_locations) -> ()");
  m.def(
      "lxu_cache_locations_update(Tensor(a!) lxu_cache_locations, Tensor lxu_cache_locations_new, Tensor? num_uniq_cache_indices=None) -> ()");
  m.def(
      "get_unique_indices(Tensor linear_indices, int max_indices, bool compute_count) -> (Tensor, Tensor, Tensor?)");
}

using namespace fbgemm_gpu;

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CPU("linearize_cache_indices", linearize_cache_indices_cpu);
  DISPATCH_TO_CPU(
      "linearize_cache_indices_from_row_idx",
      linearize_cache_indices_from_row_idx_cpu);
  DISPATCH_TO_CPU("lru_cache_populate_byte", lru_cache_populate_byte_cpu);
  DISPATCH_TO_CPU(
      "direct_mapped_lru_cache_populate_byte",
      direct_mapped_lru_cache_populate_byte_cpu);
  DISPATCH_TO_CPU("lfu_cache_populate_byte", lfu_cache_populate_byte_cpu);
  DISPATCH_TO_CPU("lxu_cache_lookup", lxu_cache_lookup_cpu);
  DISPATCH_TO_CPU(
      "direct_mapped_lxu_cache_lookup", direct_mapped_lxu_cache_lookup_cpu);
}

} // namespace
