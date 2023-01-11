/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include "c10/core/ScalarType.h"
#ifdef FBCODE_CAFFE2
#include "common/stats/Stats.h"
#endif
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

#ifdef FBCODE_CAFFE2
// Fraction of unique indices per batch.
// # unique indices / # requested indices.
DEFINE_quantile_stat(
    tbe_uvm_cache_unique_rate,
    "tbe_uvm_cache_unique_rate_per_mille",
    facebook::fb303::ExportTypeConsts::kNone,
    std::array<double, 4>{{.25, .50, .75, .99}});

// Miss rate: # unique index misses / # requested indices
DEFINE_quantile_stat(
    tbe_uvm_cache_unique_miss_rate,
    "tbe_uvm_cache_unique_miss_rate_per_mille",
    facebook::fb303::ExportTypeConsts::kNone,
    std::array<double, 4>{{.25, .50, .75, .99}});

// Miss rate due to conflict in cache associativity.
// # unique misses due to conflict / # requested indices.
DEFINE_quantile_stat(
    tbe_uvm_cache_conflict_unique_miss_rate,
    "tbe_uvm_cache_conflict_unique_miss_rate_per_mille",
    facebook::fb303::ExportTypeConsts::kNone,
    std::array<double, 4>{{.25, .50, .75, .99}});

// FLAGs to control UVMCacheStats.
DEFINE_int32(
    tbe_uvm_cache_stat_report,
    -1,
    "If set to a positive number, it enables UVMCacheStats reporting, and this FLAG value is "
    "stats collecting period");

DEFINE_int32(
    tbe_uvm_cache_stats_print_out_period,
    -1,
    "If tbe_uvm_cache_stat_report is enabled, more detailed raw stats will be printed with this "
    "period. This should be an integer multiple of tbe_uvm_cache_stat_report.");

// TODO: align this with uvm_cache_stats_index in
// split_embeddings_cache_cuda.cu.
const int kUvmCacheStatsSize = 6;

namespace {

// Processes UVMCacheStats from one batch of TBE op call.
// Args:
//  * signature: unique id for TBE op.
//  * total_cache_hash_size: num_embeddding_rows in the whole TBE op.
//  * Per-batch UVMCacheStats.
void process_uvm_cache_stats(
    const size_t signature,
    const int64_t total_cache_hash_size,
    const int64_t call_count,
    const bool gather_uvm_stats,
    const Tensor& uvm_cache_stats) {
  if (gather_uvm_stats) {
    // Export cache stats.
    auto uvm_cache_stats_cpu = uvm_cache_stats.cpu();
    auto* uvm_cache_stats_ptr = uvm_cache_stats_cpu.data_ptr<int32_t>();
    if (uvm_cache_stats_ptr[1] > 0) {
      // Report cache stats in per-mille.
      double num_requested_indices =
          static_cast<double>(uvm_cache_stats_ptr[1]);
      double unique_rate = static_cast<double>(uvm_cache_stats_ptr[2] * 1000) /
          num_requested_indices;
      double unique_miss_rate =
          static_cast<double>(uvm_cache_stats_ptr[3] * 1000) /
          num_requested_indices;
      double unique_conflict_miss_rate =
          static_cast<double>(uvm_cache_stats_ptr[4] * 1000) /
          num_requested_indices;
      STATS_tbe_uvm_cache_unique_rate.addValue(unique_rate);
      STATS_tbe_uvm_cache_unique_miss_rate.addValue(unique_miss_rate);
      STATS_tbe_uvm_cache_conflict_unique_miss_rate.addValue(
          unique_conflict_miss_rate);
    }
    if (call_count % FLAGS_tbe_uvm_cache_stats_print_out_period == 0) {
      LOG(INFO) << "$Stats [" << signature << "] "
                << " hash_size: " << total_cache_hash_size
                << ", call_count: " << call_count
                << ", N_requested_indices: " << uvm_cache_stats_ptr[1]
                << ", N_unique_indices: " << uvm_cache_stats_ptr[2]
                << ", N_unique_misses: " << uvm_cache_stats_ptr[3]
                << ", N_conflict_unique_misses: " << uvm_cache_stats_ptr[4]
                << ", N_conflict_misses: " << uvm_cache_stats_ptr[5];
    }
  }
}

} // namespace
#endif

///@defgroup embedding-cuda Embedding CUDA Operators
///

Tensor int_nbit_split_embedding_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t row_alignment,
    int64_t output_dtype,
    Tensor lxu_cache_weights,
    Tensor lxu_cache_locations,
    int64_t max_float8_D,
    int64_t fp8_exponent_bits,
    int64_t fp8_exponent_bias);

Tensor int_nbit_split_embedding_codegen_forward_weighted_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    int64_t row_alignment,
    Tensor indice_weights,
    int64_t output_dtype,
    Tensor lxu_cache_weights,
    Tensor lxu_cache_locations,
    int64_t max_float8_D,
    int64_t fp8_exponent_bits,
    int64_t fp8_exponent_bias);

Tensor int_nbit_split_embedding_nobag_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    int64_t D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t row_alignment,
    int64_t output_dtype,
    Tensor lxu_cache_weights,
    Tensor lxu_cache_locations,
    int64_t max_float8_D,
    int64_t fp8_exponent_bits,
    int64_t fp8_exponent_bias);

///@ingroup embedding-cuda
Tensor int_nbit_split_embedding_codegen_lookup_function(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    int64_t output_dtype,
    c10::optional<Tensor> lxu_cache_weights,
    c10::optional<Tensor> lxu_cache_locations,
    c10::optional<int64_t> row_alignment,
    c10::optional<int64_t> max_float8_D,
    c10::optional<int64_t> fp8_exponent_bits,
    c10::optional<int64_t> fp8_exponent_bias) {
  if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
    std::vector<int64_t> max_D_list{
        max_int2_D,
        max_int4_D,
        max_int8_D,
        max_float8_D ? *max_float8_D : 0,
        max_float16_D,
        max_float32_D};
    int64_t max_D = *std::max_element(max_D_list.begin(), max_D_list.end());
    return int_nbit_split_embedding_nobag_codegen_forward_unweighted_cuda(
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        weights_tys,
        max_D,
        max_int2_D,
        max_int4_D,
        max_int8_D,
        max_float16_D,
        max_float32_D,
        indices,
        offsets,
        row_alignment ? *row_alignment : 16,
        output_dtype,
        lxu_cache_weights.value_or(at::empty({0, 0}, at::kByte)),
        lxu_cache_locations.value_or(at::empty({0}, at::kInt)),
        max_float8_D ? *max_float8_D : 0,
        fp8_exponent_bits ? *fp8_exponent_bits : -1,
        fp8_exponent_bias ? *fp8_exponent_bias : -1);
  }
  if (!indice_weights) {
    return int_nbit_split_embedding_codegen_forward_unweighted_cuda(
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        weights_tys,
        D_offsets,
        total_D,
        max_int2_D,
        max_int4_D,
        max_int8_D,
        max_float16_D,
        max_float32_D,
        indices,
        offsets,
        pooling_mode,
        row_alignment ? *row_alignment : 16,
        output_dtype,
        lxu_cache_weights.value_or(at::empty({0, 0}, at::kByte)),
        lxu_cache_locations.value_or(at::empty({0}, at::kInt)),
        max_float8_D ? *max_float8_D : 0,
        fp8_exponent_bits ? *fp8_exponent_bits : -1,
        fp8_exponent_bias ? *fp8_exponent_bias : -1);
  }
  return int_nbit_split_embedding_codegen_forward_weighted_cuda(
      dev_weights,
      uvm_weights,
      weights_placements,
      weights_offsets,
      weights_tys,
      D_offsets,
      total_D,
      max_int2_D,
      max_int4_D,
      max_int8_D,
      max_float16_D,
      max_float32_D,
      indices,
      offsets,
      pooling_mode,
      row_alignment ? *row_alignment : 16,
      *indice_weights,
      output_dtype,
      lxu_cache_weights.value_or(at::empty({0, 0}, at::kByte)),
      lxu_cache_locations.value_or(at::empty({0}, at::kInt)),
      max_float8_D ? *max_float8_D : 0,
      fp8_exponent_bits ? *fp8_exponent_bits : -1,
      fp8_exponent_bias ? *fp8_exponent_bias : -1);
}

///@ingroup embedding-cuda
/// Simlar to int_nbit_split_embedding_codegen_lookup_function, but it does
/// UVM_CACHING lookup.
Tensor int_nbit_split_embedding_uvm_caching_codegen_lookup_function(
    // First args should be the same to those of
    // int_nbit_split_embedding_codegen_lookup_function.
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    int64_t output_dtype,
    c10::optional<Tensor> lxu_cache_weights,
    c10::optional<Tensor> lxu_cache_locations,
    c10::optional<int64_t> row_alignment,
    c10::optional<int64_t> max_float8_D,
    c10::optional<int64_t> fp8_exponent_bits,
    c10::optional<int64_t> fp8_exponent_bias,
    // Additional args for UVM_CACHING.
    // cache_hash_size_cumsum: cumulative sum of # embedding rows of all the
    // tables. 1D tensor, dtype=int64.
    c10::optional<Tensor> cache_hash_size_cumsum,
    // total_cache_hash_size: sum of # embedding rows of all the tables.
    c10::optional<int64_t> total_cache_hash_size,
    // cache_index_table_map: (linearized) index to table number map.
    // 1D tensor, dtype=int32.
    c10::optional<Tensor> cache_index_table_map,
    // lxu_cache_state: Cache state (cached idnex, or invalid).
    // 2D tensor: # sets x assoc. dtype=int64.
    c10::optional<Tensor> lxu_cache_state,
    // lxu_state: meta info for replacement (time stamp for LRU).
    // 2D tensor: # sets x assoc. dtype=int64.
    c10::optional<Tensor> lxu_state) {
  // This function does prefetch() and foward() methods in
  // IntNBitTableBatchedEmbeddingBagsCodegen, but run them in sequence.
  // Prefetching of multiple batches of requests is not yet supported.

#ifdef FBCODE_CAFFE2
  static std::mutex uvm_cache_stats_mutex;
  static std::unordered_map<size_t, int64_t> tbe_call_count;
#endif
  static std::atomic<int64_t> time_stamp = 0; // for LRU replacement.
  int64_t curr_time_stamp = -1;

  // UVM_CACHING if lxu_cache_weights are valid.
  if (lxu_cache_weights.has_value() && lxu_cache_weights.value().numel() > 0) {
    curr_time_stamp = ++time_stamp; // increment everytime it's called.
    // Use the copied curr_time_stamp so that we use a consistent value even
    // if it's changed in other threads.

    // Linearize indices.
    auto linear_cache_indices = linearize_cache_indices_cuda(
        cache_hash_size_cumsum.value(), indices, offsets);

    bool gather_uvm_stats = false;
    Tensor uvm_cache_stats =
        at::empty({0}, lxu_cache_weights.value().options().dtype(at::kInt));
#ifdef FBCODE_CAFFE2
    size_t signature = reinterpret_cast<size_t>(uvm_weights.data_ptr());
    int64_t call_count = 0;
    {
      std::lock_guard<std::mutex> guard(uvm_cache_stats_mutex);
      if (tbe_call_count.count(signature) == 0) {
        tbe_call_count[signature] = 0;
      }
      tbe_call_count[signature]++;
      call_count = tbe_call_count[signature];
    }

    if (call_count % FLAGS_tbe_uvm_cache_stat_report == 0) {
      gather_uvm_stats = true;
      uvm_cache_stats = at::zeros(
          {kUvmCacheStatsSize},
          lxu_cache_weights.value().options().dtype(at::kInt));
    }
#endif

    // Lookup and fetch data from UVM: supporting only lru; no lfu currently.
    lru_cache_populate_byte_cuda(
        uvm_weights,
        cache_hash_size_cumsum.value(),
        total_cache_hash_size.value(),
        cache_index_table_map.value(),
        weights_offsets,
        weights_tys,
        D_offsets,
        linear_cache_indices,
        lxu_cache_state.value(),
        lxu_cache_weights.value(),
        curr_time_stamp,
        lxu_state.value(),
        row_alignment ? *row_alignment : 16,
        gather_uvm_stats,
        uvm_cache_stats);

    // Update lxu_cache_locations.
    lxu_cache_locations = lxu_cache_lookup_cuda(
        linear_cache_indices,
        lxu_cache_state.value(),
        total_cache_hash_size.value(),
        gather_uvm_stats,
        uvm_cache_stats);

#ifdef FBCODE_CAFFE2
    process_uvm_cache_stats(
        signature,
        total_cache_hash_size.value(),
        call_count,
        gather_uvm_stats,
        uvm_cache_stats);
#endif
  }

  return int_nbit_split_embedding_codegen_lookup_function(
      dev_weights,
      uvm_weights,
      weights_placements,
      weights_offsets,
      weights_tys,
      D_offsets,
      total_D,
      max_int2_D,
      max_int4_D,
      max_int8_D,
      max_float16_D,
      max_float32_D,
      indices,
      offsets,
      pooling_mode,
      indice_weights,
      output_dtype,
      lxu_cache_weights,
      lxu_cache_locations,
      row_alignment,
      max_float8_D,
      fp8_exponent_bits,
      fp8_exponent_bias);
}

///@ingroup embedding-cuda
Tensor pruned_hashmap_lookup_unweighted_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets);

///@ingroup embedding-cuda
Tensor pruned_array_lookup_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor index_remappings,
    Tensor index_remappings_offsets);

///@ingroup embedding-cuda
Tensor pruned_array_lookup_from_row_idx_cuda(
    Tensor update_row_indices,
    Tensor update_table_indices,
    Tensor index_remappings,
    Tensor index_remappings_offsets);

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  DISPATCH_TO_CUDA(
      "int_nbit_split_embedding_codegen_lookup_function",
      int_nbit_split_embedding_codegen_lookup_function);
  DISPATCH_TO_CUDA(
      "int_nbit_split_embedding_uvm_caching_codegen_lookup_function",
      int_nbit_split_embedding_uvm_caching_codegen_lookup_function);
  DISPATCH_TO_CUDA(
      "pruned_hashmap_lookup", pruned_hashmap_lookup_unweighted_cuda);

  DISPATCH_TO_CUDA("pruned_array_lookup", pruned_array_lookup_cuda);
  DISPATCH_TO_CUDA(
      "pruned_array_lookup_from_row_idx",
      pruned_array_lookup_from_row_idx_cuda);
}
