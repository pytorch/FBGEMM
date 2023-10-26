/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

namespace {

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA("linearize_cache_indices", linearize_cache_indices_cuda);
  DISPATCH_TO_CUDA(
      "linearize_cache_indices_from_row_idx",
      linearize_cache_indices_from_row_idx_cuda);
  DISPATCH_TO_CUDA("lru_cache_populate", lru_cache_populate_cuda);
  DISPATCH_TO_CUDA("lru_cache_populate_byte", lru_cache_populate_byte_cuda);
  DISPATCH_TO_CUDA(
      "direct_mapped_lru_cache_populate_byte",
      direct_mapped_lru_cache_populate_byte_cuda);
  DISPATCH_TO_CUDA("lfu_cache_populate", lfu_cache_populate_cuda);
  DISPATCH_TO_CUDA("lfu_cache_populate_byte", lfu_cache_populate_byte_cuda);
  DISPATCH_TO_CUDA("lxu_cache_lookup", lxu_cache_lookup_cuda);
  DISPATCH_TO_CUDA(
      "direct_mapped_lxu_cache_lookup", direct_mapped_lxu_cache_lookup_cuda);
  DISPATCH_TO_CUDA("lxu_cache_flush", lxu_cache_flush_cuda);
  DISPATCH_TO_ALL("lxu_cache_slot", host_lxu_cache_slot);
  DISPATCH_TO_CUDA("reset_weight_momentum", reset_weight_momentum_cuda);
  DISPATCH_TO_CUDA(
      "lxu_cache_locking_counter_decrement",
      lxu_cache_locking_counter_decrement_cuda);
  DISPATCH_TO_CUDA(
      "lxu_cache_locations_update", lxu_cache_locations_update_cuda);
}

} // namespace
