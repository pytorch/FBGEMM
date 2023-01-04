/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"

namespace {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "linearize_cache_indices(Tensor cache_hash_size_cumsum, Tensor indices, Tensor offsets) -> Tensor");
  DISPATCH_TO_CUDA("linearize_cache_indices", linearize_cache_indices_cuda);
  m.def(
      "linearize_cache_indices_from_row_idx(Tensor cache_hash_size_cumsum, Tensor update_table_indices, Tensor update_row_indices) -> Tensor");
  DISPATCH_TO_CUDA(
      "linearize_cache_indices_from_row_idx",
      linearize_cache_indices_from_row_idx_cuda);
  m.def(
      "lru_cache_populate(Tensor weights, Tensor hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, int time_stamp, Tensor(c!) lru_state, bool stochastic_rounding, bool gather_cache_stats=False, Tensor(d!)? uvm_cache_stats=None) -> ()");
  DISPATCH_TO_CUDA("lru_cache_populate", lru_cache_populate_cuda);
  m.def(
      "lru_cache_populate_byte(Tensor weights, Tensor hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, int time_stamp, Tensor(c!) lru_state, int row_alignment=16, bool gather_cache_stats=False, Tensor(d!)? uvm_cache_stats=None) -> ()");
  DISPATCH_TO_CUDA("lru_cache_populate_byte", lru_cache_populate_byte_cuda);
  m.def(
      "direct_mapped_lru_cache_populate_byte(Tensor weights, Tensor hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, int time_stamp, Tensor(c!) lru_state, Tensor(d!) lxu_cache_miss_timestamp, int row_alignment=16) -> ()");
  DISPATCH_TO_CUDA(
      "direct_mapped_lru_cache_populate_byte",
      direct_mapped_lru_cache_populate_byte_cuda);
  m.def(
      "lfu_cache_populate(Tensor weights, Tensor cache_hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, Tensor(c!) lfu_state, bool stochastic_rounding) -> ()");
  DISPATCH_TO_CUDA("lfu_cache_populate", lfu_cache_populate_cuda);
  m.def(
      "lfu_cache_populate_byte(Tensor weights, Tensor cache_hash_size_cumsum, int total_cache_hash_size, Tensor cache_index_table_map, Tensor weights_offsets, Tensor weights_tys, Tensor D_offsets, Tensor linear_cache_indices, Tensor(a!) lxu_cache_state, Tensor(b!) lxu_cache_weights, Tensor(c!) lfu_state, int row_alignment=16) -> ()");
  DISPATCH_TO_CUDA("lfu_cache_populate_byte", lfu_cache_populate_byte_cuda);
  m.def(
      "lxu_cache_lookup(Tensor linear_cache_indices, Tensor lxu_cache_state, int invalid_index = -1, bool gather_cache_stats=False, Tensor(a!)? uvm_cache_stats=None) -> Tensor");
  DISPATCH_TO_CUDA("lxu_cache_lookup", lxu_cache_lookup_cuda);
  m.def(
      "direct_mapped_lxu_cache_lookup(Tensor linear_cache_indices, Tensor lxu_cache_state, int invalid_index = -1) -> Tensor");
  DISPATCH_TO_CUDA(
      "direct_mapped_lxu_cache_lookup", direct_mapped_lxu_cache_lookup_cuda);
  m.def(
      "lxu_cache_flush(Tensor(a!) uvm_weights, Tensor cache_hash_size_cumsum, Tensor cache_index_table_map, Tensor weights_offsets, Tensor D_offsets, int total_D, Tensor(b!) lxu_cache_state, Tensor(c!) lxu_cache_weights, bool stochastic_rounding) -> ()");
  DISPATCH_TO_CUDA("lxu_cache_flush", lxu_cache_flush_cuda);
  m.def("lxu_cache_slot(int h_in, int C) -> int");
  DISPATCH_TO_ALL("lxu_cache_slot", host_lxu_cache_slot);
  m.def(
      "reset_weight_momentum(Tensor dev_weights, Tensor uvm_weights, Tensor lxu_cache_weights, Tensor weights_placements, Tensor weights_offsets, Tensor momentum1_dev, Tensor momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets, Tensor D_offsets, Tensor pruned_indices, Tensor pruned_indices_offsets, Tensor logical_table_ids, Tensor buffer_ids, Tensor cache_hash_size_cumsum, Tensor lxu_cache_state, int total_cache_hash_size) -> ()");
  DISPATCH_TO_CUDA("reset_weight_momentum", reset_weight_momentum_cuda);
}

} // namespace
