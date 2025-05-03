/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/stochastic_rounding.cuh"
#include "fbgemm_gpu/utils/vec4.cuh"

namespace fbgemm_gpu {

template <typename T, typename... Ts>
constexpr inline bool is_one_of_v = (std::is_same_v<T, Ts> || ...);

////////////////////////////////////////////////////////////////////////////////
// Quantized Load and Store
////////////////////////////////////////////////////////////////////////////////

template <typename dst_t, typename src_t>
DEVICE_INLINE void quantize_store(
    dst_t* output,
    const Vec4T<src_t>& value,
    StochasticRoundingRNGState* state,
    const float2 qparams) {
  if (!state) {
    nearest_rounding_vector<dst_t, src_t>(output, value, qparams);
  } else {
    stochastic_rounding_vector<dst_t, src_t>(output, value, *state, qparams);
  }
}

template <typename dst_t, typename src_t>
DEVICE_INLINE Vec4T<dst_t> dequantize_load(
    const src_t* value,
    [[maybe_unused]] const float2 qparams) {
  if constexpr (
      std::is_same_v<src_t, uint8_t> && is_one_of_v<dst_t, float, at::Half>) {
    Vec4T<dst_t> out;
    out.acc.x = value[0] * qparams.x + qparams.y;
    out.acc.y = value[1] * qparams.x + qparams.y;
    out.acc.z = value[2] * qparams.x + qparams.y;
    out.acc.w = value[3] * qparams.x + qparams.y;
    return out;

  } else {
    return Vec4T<dst_t>(value);
  }
}

template <typename emb_t>
DEVICE_INLINE float2 load_qparams_from_row(emb_t* qparam_ptr) {
  float2 qparams;
  float* qparams_fp_ptr = reinterpret_cast<float*>(qparam_ptr);
  qparams.x = qparams_fp_ptr[0];
  qparams.y = qparams_fp_ptr[1];
  return qparams;
}

DEVICE_INLINE void store_qparams_to_row(uint8_t* ptr, float2 qparams) {
  auto ptr_as_uint = reinterpret_cast<uintptr_t>(ptr);
  if (ptr_as_uint % 8 == 0) {
    *reinterpret_cast<float2*>(ptr) = qparams;

  } else if (ptr_as_uint % 4 == 0) {
    auto* ptr_float = reinterpret_cast<float*>(ptr);
    auto* qparam_ptr = reinterpret_cast<const float*>(&qparams.x);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      ptr_float[i] = qparam_ptr[i];
    }

  } else if (ptr_as_uint % 2 == 0) {
    auto* ptr_16bit = reinterpret_cast<uint16_t*>(ptr);
    auto* qparam_ptr = reinterpret_cast<const uint16_t*>(&qparams.x);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ptr_16bit[i] = qparam_ptr[i];
    }

  } else {
    auto* qparam_ptr = reinterpret_cast<const uint8_t*>(&qparams.x);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      ptr[i] = qparam_ptr[i];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Weight Row
//
// This is a memory accessor around a row of dim_ number of embedding weights.
// It provides for loading and storing of 4 elements at a time (Vec4T<dst_t>)
// from and to the embedding table or cache.  It also provides for quantization
// and de-quantization of the data.  The cache row pointer is optional, and if
// not provided, then the embedding table is assumed to be the source of truth.
//
// Template parameters:
//  emb_t   : The type of the embedding table (e.g. uint8_t, float, at::Half)
//  cache_t : The type of the cache
//  dst_t   : The type of the registers
////////////////////////////////////////////////////////////////////////////////

template <typename emb_t, typename cache_t, typename dst_t>
// TODO: pass in dimension info and calculate qparams for rowwise integer
// quantization
class WeightRow {
 public:
  // Constructor for no stochastic rounding
  DEVICE_INLINE WeightRow(emb_t* row, cache_t* cache_row, int dim)
      : row_(row),
        cache_row_(cache_row),
        dim_(dim),
        stoc_rounding_state_ptr_(nullptr) {}

  // Constructor for stochastic rounding
  DEVICE_INLINE WeightRow(
      emb_t* row,
      cache_t* cache_row,
      int dim,
      bool stochastic_rounding,
      const at::PhiloxCudaState* stochastic_rounding_philox_args,
      const uint64_t salt_value)
      : row_(row), cache_row_(cache_row), dim_(dim) {
    stoc_rounding_state_ptr_ = nullptr;
    if constexpr (!std::is_same_v<emb_t, float>) {
      if (stochastic_rounding) {
        stoc_rounding_state_.init(*stochastic_rounding_philox_args, salt_value);
        // Store the pointer here to avoid an if-else cond during load/store
        stoc_rounding_state_ptr_ = &stoc_rounding_state_;
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Load 4 elements from the table row at element offset d into a register
  // variable (Vec4T<dst_t>)
  //
  // If the cache row pointer is valid, then data will be read from the cache
  // instead of embedding table.
  //////////////////////////////////////////////////////////////////////////////

  DEVICE_INLINE Vec4T<dst_t> load(const int32_t d, const float2 qparams) const {
    // Load from the cache if resident; else load from the embedding table.
    //
    // Note: This method assumes that dst_t is of higher precision than cache_t
    // and emb_t
    if (cache_row_) {
      return dequantize_load<dst_t, cache_t>(cache_row_ + d, qparams);
    } else {
      return dequantize_load<dst_t, emb_t>(row_ + d, qparams);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Store regster variable of 4 elements (Vec4T<dst_t>) back into the table
  // into the table row at element offset d
  //
  // If the cache row pointer is valid, then data will be written to the cache
  // instead of embedding table.
  //////////////////////////////////////////////////////////////////////////////

  DEVICE_INLINE void
  store(const Vec4T<dst_t>& v, const int32_t d, const float2 qparams) {
    // Write back weight (high precision) to cache if resident; else write to
    // embedding table.
    //
    // Note: This method assumes that dst_t is of higher precision than cache_t
    // and emb_t
    if (cache_row_) {
      quantize_store(cache_row_ + d, v, stoc_rounding_state_ptr_, qparams);
    } else {
      quantize_store(row_ + d, v, stoc_rounding_state_ptr_, qparams);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Fetch the quantization parameters of the table row
  //
  // Qparams are fetched from the end of the row in the embedding table, not the
  // cache.
  //////////////////////////////////////////////////////////////////////////////

  DEVICE_INLINE float2 load_qparams() const {
    if constexpr (std::is_same_v<emb_t, uint8_t>) {
      return load_qparams_from_row<emb_t>(row_ + dim_);
    } else {
      return make_float2(0.0f, 0.0f);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Update the quantization parameters of the table row
  //
  // Qparams are stored at the end of the row in the embedding table, not the
  // cache.
  //////////////////////////////////////////////////////////////////////////////

  template <typename T = emb_t>
  DEVICE_INLINE auto store_qparams(const float2 qparams) const
      -> std::enable_if_t<std::is_same_v<T, uint8_t>, void> {
    store_qparams_to_row(row_ + dim_, qparams);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Load the row from the embedding table into the cache
  //
  // De-quantization will be applied if the embedding table type is uint8_t (low
  // prec -> high prec).
  //////////////////////////////////////////////////////////////////////////////

  DEVICE_INLINE void warp_copy_to_cache(
      cache_t* dst_row,
      const uint32_t dim_length,

      const uint32_t num_lanes,
      const uint32_t lane_id) {
    if constexpr (std::is_same_v<emb_t, cache_t>) {
      // If the embedding table and cache types are the same, then simply copy
      // data from cache to embedding table.
      for (auto d = lane_id * 4; d < dim_length; d += num_lanes * 4) {
        same_type_vector_copy(
            dst_row + d, reinterpret_cast<const cache_t*>(row_ + d));
      }
    } else {
      // Load quantization params from embedding row
      const auto qparams = load_qparams();

      // Copy over for each warp-sized slice of Vec4's
      // Does 2-step conversion: weight_t -> FP32 -> cache_t
      for (auto d = lane_id * 4; d < dim_length; d += num_lanes * 4) {
        const auto slice = load(d, qparams);
        quantize_store(dst_row + d, slice, stoc_rounding_state_ptr_, qparams);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Copy the row from the embedding table into the cache
  //////////////////////////////////////////////////////////////////////////////

  DEVICE_INLINE void evict_cache(const uint32_t d, const float2 qparams) {
    if constexpr (std::is_same_v<emb_t, cache_t>) {
      // If the embedding table and cache types are the same, then simply copy
      // data from cache to embedding table.
      same_type_vector_copy(
          reinterpret_cast<emb_t*>(row_ + d),
          reinterpret_cast<const cache_t*>(cache_row_ + d));
    } else {
      // Else, do 2-step conversion: cache_t -> FP32 (register) -> weight_t
      const auto cache_slice = load(d, qparams);
      quantize_store(row_ + d, cache_slice, stoc_rounding_state_ptr_, qparams);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Evict the row from the cache and into the embedding table.
  //
  // Quantization will be applied if the embedding table type is uint8_t (high
  // prec -> low prec).
  //////////////////////////////////////////////////////////////////////////////

  DEVICE_INLINE void warp_evict_cache(
      const uint32_t dim_length,
      const uint32_t num_lanes,
      const uint32_t lane_id) {
    float2 qparams;

    if constexpr (std::is_same_v<emb_t, uint8_t>) {
      auto local_min = std::numeric_limits<at::acc_type<cache_t, true>>::max();
      auto local_max =
          std::numeric_limits<at::acc_type<cache_t, true>>::lowest();

      // Compute the qparams from the cache row (not embedding row) weights
      for (auto d = lane_id; d * 4 < dim_length; d += num_lanes) {
        const auto cache_slice = load(d * 4, qparams); // qparams not used
        local_max = max(local_max, cache_slice.vmax());
        local_min = min(local_min, cache_slice.vmin());
      }

      // Compute the max and min across the warps
      qparams = warp_find_qparams(local_min, local_max);

      if (lane_id == 0) {
        // Store the qparams into the embedding row
        store_qparams(qparams);
      }
    }

    for (auto d = lane_id * 4; d < dim_length; d += num_lanes * 4) {
      // Evict the slice into the embedding row
      evict_cache(d, qparams);
    }
  }

 private:
  // The pointer to the row of weights in the embedding table
  emb_t* const row_;

  // The pointer to the row of weights in the cache
  cache_t* const cache_row_;

  // The number of elements per table row
  int32_t const dim_;

  // The state for stochastic rounding
  StochasticRoundingRNGState stoc_rounding_state_;
  StochasticRoundingRNGState* stoc_rounding_state_ptr_;

  //////////////////////////////////////////////////////////////////////////////
  // Copy 4 elements (float or at::Half) from src_vec to dst_vec
  //
  // Reinterpret cast to float4* or float2* for mass copy
  //////////////////////////////////////////////////////////////////////////////

  template <
      typename T,
      typename = std::enable_if_t<is_one_of_v<T, float, at::Half>>>
  DEVICE_INLINE void same_type_vector_copy(T* dst_vec, const T* src_vec) {
    // Copy vector from src_vec to dst_vec (both are float)
    using ptr_t = std::conditional_t<std::is_same_v<T, float>, float4, float2>;
    *reinterpret_cast<ptr_t*>(dst_vec) =
        *reinterpret_cast<const ptr_t*>(src_vec);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Weight Row Accessor
//
// This is a lightweight memory accessor around a row of dim_ number of
// embedding weights of type row_t (can be HBM or UVM), and provides for loading
// 4 elements at a time into Vec4T<dst_t> with de-quantization support.  Unlike
// the heavyweight WeightRow class, this accessor is for reading values only,
// and does not handle embedding vs cache tables, etc.
//
// Template parameters:
//  row_t   : The type of the table row (e.g. uint8_t, float, at::Half)
//  dst_t   : The type of the registers
////////////////////////////////////////////////////////////////////////////////

template <typename row_t, typename dst_t>
class WeightRowAccessor {
  // The pointer to the row of weights in the table
  const row_t* const row_;

  // The number of elements per table row.
  //
  // This is NOT necessarily equivalent to the row stride D_emb, as there may be
  // quantization parameters and optimizer states packed into the back of the
  // row.
  //
  // dim_ is presumed to be a multiple of 4, since it loads data into Vec4T for
  // max register occupancy.
  const int32_t dim_;

  // [OPTIONAL] The quantization parameters for the row.  If the row type is not
  // uint8_t, i.e. not quantized, then it is set to (0.0f, 0.0f).
  float2 qparams_ = make_float2(0.0f, 0.0f);

 public:
  DEVICE_INLINE
  WeightRowAccessor(const row_t* const row, const int32_t dim)
      : row_(row), dim_(dim) {
    if constexpr (std::is_same_v<row_t, uint8_t>) {
      qparams_ = qparams();
    }
  }

  template <typename T = row_t>
  DEVICE_INLINE auto qparams() const
      -> std::enable_if_t<std::is_same_v<T, uint8_t>, float2> {
    return load_qparams_from_row<row_t>(row_ + dim_);
  }

  DEVICE_INLINE Vec4T<dst_t> load(const int32_t d) const {
    return dequantize_load<dst_t, row_t>(row_ + d, qparams_);
  }
};

} // namespace fbgemm_gpu
