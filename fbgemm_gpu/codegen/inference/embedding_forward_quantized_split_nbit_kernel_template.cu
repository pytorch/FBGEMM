/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{% set wdesc =  "weighted" if weighted else "unweighted" %}
#include "fbgemm_gpu/embedding_forward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor.h"

using namespace fbgemm_gpu;
using Tensor = at::Tensor;

namespace nbit {

// TODO: increase code sharing (templates for accumulator_ty, accumulation, outputs per thread, etc?)
template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows, bool DeviceOnly, bool PackedMode>
__launch_bounds__(WarpsPerBlock * kWarpSize)
__global__ void {{ emb_weight_type.enum_name }}_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L(
  const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> dev_weights,
  const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> uvm_weights,
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
  const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
  const pta::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits> weights_tys,
  {% if not nobag %}
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
  {% else %}
  const int64_t D,
  {% endif %}
  FixedDivisor fd_B, // FixedDivisor(div_round_up(B, OutputRowsPerThread))
  const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
  const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
  {% if not nobag %}
  const int64_t pooling_mode,
  {% endif %}
  const int64_t row_alignment,
  {% if weighted %}
  pta::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> indice_weights,
  {% endif %}
  {% if emb_weight_type.enum_name == "FP8" %}
  const int exponent_bits,
  const int exponent_bias,
  {% endif %}
  const int32_t num_packed_bags,
  pta::PackedTensorAccessor32<output_t, 2, at::RestrictPtrTraits> output, // [B][total_D],
  const pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations
  ) {
  const int32_t T = weights_offsets.size(0);
  {% if not nobag %}
  const bool mean_pooling = static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN;
  const int32_t B = output.size(0);
  {% else %}
  const int32_t B = (offsets.size(0) - 1) / T;
  {% endif %}
  const int32_t bb_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (bb_t >= fd_B.D() * T) {
    return;
  }
  static_assert(
    std::is_same_v<output_t, float> || std::is_same_v<output_t, at::BFloat16> || std::is_same_v<output_t, at::Half> || std::is_same_v<output_t, uint8_t>,
    "output_t can only be float or half or bytes now"
  );

  int32_t t;
  int32_t bb;
  fd_B.DivMod(bb_t, &t, &bb);

  {% if not nobag %}
  const int32_t D_start = D_offsets[t];
  const int32_t D_end = D_offsets[t + 1];
  const int32_t D = D_end - D_start;
  {% endif %}
  SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
  if (weight_ty != SparseType::{{ emb_weight_type.enum_name }}) {
      return;
  }

  // default to 16 byte alignment for GPU TBE
  const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty, row_alignment);

  if (D_bytes <= MinNum128BRows * 128 || D_bytes > MaxNum128BRows * 128) {
    return;
  }


  const int64_t weights_offset = weights_offsets[t];
  const int32_t D_total = padded_D(D, weight_ty);
  const int32_t D_padding = D_total - D;

  uint32_t warp_idx = threadIdx.y;
  int32_t indices_starts[OutputRowsPerThread];
  int32_t Ls[OutputRowsPerThread];
  int32_t max_Ls = 0;
  constexpr size_t kOutputsPerThread = {{ (32 // emb_weight_type.bit_width) }};

  constexpr uint32_t NumUint4LoadsPerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    const uint32_t packed_bag_idx = PackedMode ? (threadIdx.x % NumUint4LoadsPerRow) / uint4_loads_per_row : 0;
    uint32_t b = min(static_cast<uint32_t>(bb * num_packed_bags * OutputRowsPerThread + i * num_packed_bags + packed_bag_idx), static_cast<uint32_t>(B - 1));
    int32_t indices_start = offsets[t * B + b];
    int32_t indices_end = offsets[t * B + b + 1];
    indices_starts[i] = indices_start;
    Ls[i] = indices_end - indices_start;
    max_Ls = max(max_Ls, Ls[i]);
  }
  const index_t* indices_ = &indices[0];

  const uint8_t* __restrict__ weights;
  const auto placement = DeviceOnly ? PlacementType::DEVICE : static_cast<PlacementType>(weights_placements[t]);
  if (placement == PlacementType::DEVICE) {
      weights = &dev_weights[weights_offset];
  } else {
      weights = &uvm_weights[weights_offset];
  }

  {% if not nobag %}
  VecNT<{{ (32 // emb_weight_type.bit_width) }}, PrimitiveType::{{ emb_weight_type.primitive_type }}> accumulators[OutputRowsPerThread][MaxNum128BRows];
  {% endif %}

  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);

    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4LoadsPerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][PackedMode ? NumUint4LoadsPerRow : 0];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * NumUint4LoadsPerRow; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % NumUint4LoadsPerRow;
      if constexpr (PackedMode) {
        row_load_idx %= uint4_loads_per_row;
      }
      uint32_t input_row_idx = (load_idx / NumUint4LoadsPerRow);
      const uint32_t packed_bag_idx = PackedMode ? (load_idx % NumUint4LoadsPerRow) / uint4_loads_per_row : 0;
      bool load_idx_valid = PackedMode ? packed_bag_idx < num_packed_bags : row_load_idx < uint4_loads_per_row;
      {%- if is_rocm %}
      constexpr uint32_t kMaxRowUnroll = 4;
      constexpr uint32_t kRowUnroll = OutputRowsPerThread < kMaxRowUnroll ? OutputRowsPerThread : kMaxRowUnroll;

      #pragma unroll
      for (uint32_t outer_i = 0; outer_i < OutputRowsPerThread - OutputRowsPerThread % kRowUnroll; outer_i += kRowUnroll) {
        uint4 row_data_v[kRowUnroll];
        const uint4* row_v[kRowUnroll];
        int32_t idx_v[kRowUnroll];
        int32_t cache_idx_v[kRowUnroll];
        #pragma unroll
        for (uint32_t inner_i = 0; inner_i < kRowUnroll; ++inner_i) {
          uint32_t i = outer_i + inner_i;
          bool valid = load_idx_valid && L_start + input_row_idx < Ls[i];
          bool cache_valid = !DeviceOnly && (placement == PlacementType::MANAGED_CACHING && valid);
          idx_v[inner_i] = valid ? indices_[indices_starts[i] + L_start + input_row_idx] : -1;
          cache_idx_v[inner_i] = (!DeviceOnly && cache_valid) ? lxu_cache_locations[indices_starts[i] + L_start + input_row_idx] : -1;
        }


        #pragma unroll
        for (uint32_t inner_i = 0; inner_i < kRowUnroll; ++inner_i) {
          uint32_t i = outer_i + inner_i;
          bool valid = load_idx_valid && L_start + input_row_idx < Ls[i];
          bool cache_valid = !DeviceOnly && (placement == PlacementType::MANAGED_CACHING && valid);
          valid = valid && (idx_v[inner_i] != -1);
          if (!DeviceOnly && cache_valid && cache_idx_v[inner_i] != kCacheLocationMissing) {
            row_v[inner_i] = reinterpret_cast<const uint4*>(&lxu_cache_weights[static_cast<int64_t>(cache_idx_v[inner_i])][0]);
          } else
          if (valid) {
            row_v[inner_i] = reinterpret_cast<const uint4*>(&weights[static_cast<int64_t>(idx_v[inner_i]) * D_bytes]);
          } else {
            row_v[inner_i] = reinterpret_cast<const uint4*>(&weights[0]);
          }
        }
        #pragma unroll
        for (uint32_t inner_i = 0; inner_i < kRowUnroll; inner_i++) {
          uint32_t i = outer_i + inner_i;
          row_data_v[inner_i] = row_v[inner_i][row_load_idx];
        }
        uint4 zeros = {0, 0, 0, 0};
        #pragma unroll
        for (uint32_t inner_i = 0; inner_i < kRowUnroll; inner_i++) {
          uint32_t i = outer_i + inner_i;
          bool valid = load_idx_valid && (L_start + input_row_idx < Ls[i]) && (idx_v[inner_i] != -1);
          uint4 data = valid ? row_data_v[inner_i] : zeros;
          if constexpr (PackedMode) {
            buffers[warp_idx][i][input_row_idx][row_load_idx + uint4_loads_per_row * packed_bag_idx] = data;
          } else {
            buffers[warp_idx][i][input_row_idx][row_load_idx] = data;
          }
          {% if weighted %}
          buffers_indice_weights[warp_idx][i][input_row_idx][packed_bag_idx] = valid ? indice_weights[indices_starts[i] + L_start + input_row_idx] : 0.0;
          {% endif %}
        }
      }
      {%- endif %}
      
      {%- if is_rocm %}
      if constexpr (OutputRowsPerThread % kRowUnroll)
      {
      #pragma unroll
      for (uint32_t i = OutputRowsPerThread - OutputRowsPerThread % kRowUnroll; i < OutputRowsPerThread; ++i) {
      {%- else %}
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
      {%- endif %}
        bool valid = load_idx_valid && L_start + input_row_idx < Ls[i];
        bool cache_valid = !DeviceOnly && (placement == PlacementType::MANAGED_CACHING && valid);
        int32_t idx = valid ? indices_[indices_starts[i] + L_start + input_row_idx] : -1;
        int32_t cache_idx = (!DeviceOnly && cache_valid) ? lxu_cache_locations[indices_starts[i] + L_start + input_row_idx] : -1;
        valid = valid && (idx != -1);
        const uint4* row;
        if (!DeviceOnly && cache_valid && cache_idx != kCacheLocationMissing) {
          row = reinterpret_cast<const uint4*>(&lxu_cache_weights[static_cast<int64_t>(cache_idx)][0]);
        } else if (valid) {
          row = reinterpret_cast<const uint4*>(&weights[static_cast<int64_t>(idx) * D_bytes]);
        } else {
          row = reinterpret_cast<const uint4*>(&weights[0]);
        }
        if constexpr (PackedMode) {
          cp_async_zfill_cg<sizeof(uint4)>(&buffers[warp_idx][i][input_row_idx][row_load_idx + uint4_loads_per_row * packed_bag_idx], &row[row_load_idx], valid);
        } else {
          cp_async_zfill_cg<sizeof(uint4)>(&buffers[warp_idx][i][input_row_idx][row_load_idx], &row[row_load_idx], valid);
        }
        {% if weighted %}
        buffers_indice_weights[warp_idx][i][input_row_idx][packed_bag_idx] = valid ? indice_weights[indices_starts[i] + L_start + input_row_idx] : 0.0;
        {% endif %}
      }
      {%- if is_rocm %}
      } // constexpr if (OutputRowsPerThread % kRowUnroll)
      {%- endif %}
    }
    // equivalent to fence + wait.
    cp_async_wait<0>();
    syncwarp();
    const int32_t uints_per_row = 4 * uint4_loads_per_row;
    if constexpr (PackedMode) {
      input_rows_in_flight = shfl_sync(input_rows_in_flight, threadIdx.x / uints_per_row % num_packed_bags * uint4_loads_per_row);

      #pragma unroll OutputRowsPerThread
      for(uint32_t i = 0; i < OutputRowsPerThread; ++i)
      {
        Ls[i] = shfl_sync(Ls[i], threadIdx.x / uints_per_row % num_packed_bags * uint4_loads_per_row);
      }
    }
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        if (!valid) {
          continue;
        }
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);
        const int32_t packed_bag_idx = PackedMode ? (threadIdx.x / uints_per_row) % num_packed_bags : 0;
        // scale and bias are at the beginning of each row.
        // rationale: have scale/shift at start since these get loaded first
        // and then broadcasted around so it might speed up the first cache miss.
        {% if emb_weight_type.primitive_type == "INT" %}
        half2 shift_scale = reinterpret_cast<const half2*>(row)[packed_bag_idx * uints_per_row];
        {% endif %}

        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx][packed_bag_idx];
        {% endif %}

        using scalar_t = {{ emb_weight_type.cpp_type_name }};

        {% if not nobag %}
        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          scalar_t v = reinterpret_cast<const scalar_t*>(row)[kWarpSize * j + threadIdx.x];
          {% if weighted %}
          accumulators[i][j].fma(v, {% if emb_weight_type.primitive_type == "INT" %} shift_scale, {% elif emb_weight_type.enum_name == "FP8" %} exponent_bits, exponent_bias, {% endif %} row_weight);
          {% else %}
          accumulators[i][j].add(v{% if emb_weight_type.primitive_type == "INT" %}, shift_scale {% elif emb_weight_type.enum_name == "FP8" %}, exponent_bits, exponent_bias {% endif %});
          {% endif %}
        }
        {% else %}
        const int32_t output_j = indices_starts[i] + L_start + input_row_idx;
        if constexpr (std::is_same_v<output_t, float> || std::is_same_v<output_t, at::Half> || std::is_same_v<output_t, at::BFloat16>) {
          #pragma unroll MaxNum128BRows
          for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
            // Read the uint8/4/2 values: note that first 4 Bytes will be ditched later:
            // We shift back by 4/8/16 elements to remove the first 4 Bytes (which is garbage due to
            // the scale/shift handling).
            // Reason: to avoid divergence the first thread in the warp computes garbage.
            const int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
            scalar_t v = reinterpret_cast<const scalar_t*>(row)[kWarpSize * j + threadIdx.x];
            if (output_d >= 0 && output_d < D) {
              const int num_valid_outputs = min(static_cast<int>(D - output_d), static_cast<int>({{ (32 // emb_weight_type.bit_width) }}));
              VecNT<{{ (32 // emb_weight_type.bit_width) }}, PrimitiveType::{{ emb_weight_type.primitive_type }}> acc(v{% if emb_weight_type.primitive_type == "INT" %}, shift_scale {% elif emb_weight_type.enum_name == "FP8" %}, exponent_bits, exponent_bias {% endif %});
              acc.store(&output[output_j][output_d], num_valid_outputs);
            }
          }
        } else if constexpr (std::is_same_v<output_t, uint8_t>) {
          // INT8:
          // apply per feature row-wise int8
          auto thread_local_min = std::numeric_limits<float>::max();
          auto thread_local_max = std::numeric_limits<float>::lowest();
          float2 qparams;
          #pragma unroll MaxNum128BRows
          for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
            int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
            scalar_t v = reinterpret_cast<const scalar_t*>(row)[kWarpSize * j + threadIdx.x];
            VecNT<{{ (32 // emb_weight_type.bit_width) }}, PrimitiveType::{{ emb_weight_type.primitive_type }}> acc(v{% if emb_weight_type.primitive_type == "INT" %}, shift_scale {% elif emb_weight_type.enum_name == "FP8" %}, exponent_bits, exponent_bias {% endif %});
            if (output_d >= 0 && output_d < D) {
              thread_local_max = max(thread_local_max, float{{ (32 // emb_weight_type.bit_width) }}_max(acc.acc));
              thread_local_min = min(thread_local_min, float{{ (32 // emb_weight_type.bit_width) }}_min(acc.acc));
            }
          }
          qparams = warp_find_qparams(thread_local_min, thread_local_max);
          #pragma unroll MaxNum128BRows
          for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
            const int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
            scalar_t v = reinterpret_cast<const scalar_t*>(row)[kWarpSize * j + threadIdx.x];
            if (output_d >= 0 && output_d < D) {
              const int num_valid_outputs = min(static_cast<int>(D - output_d), static_cast<int>({{ (32 // emb_weight_type.bit_width) }}));
              VecNT<{{ (32 // emb_weight_type.bit_width) }}, PrimitiveType::{{ emb_weight_type.primitive_type }}> acc(v{% if emb_weight_type.primitive_type == "INT" %}, shift_scale {% elif emb_weight_type.enum_name == "FP8" %}, exponent_bits, exponent_bias {% endif %});
              acc.store(&output[output_j][output_d], qparams, num_valid_outputs);
            }
          }
          if (threadIdx.x == 0) {
            store_qparams_to_row(&output[output_j][D], qparams);
          }
        }
        {% endif %}
      }
    }
  }

  {% if not nobag %}
  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    const int32_t num_stores_with_padding_per_row = 4 * uint4_loads_per_row; 
    const int32_t packed_bag_idx = PackedMode ? threadIdx.x / num_stores_with_padding_per_row : 0;
    const uint32_t b = min(static_cast<uint32_t>(bb * num_packed_bags * OutputRowsPerThread + i * num_packed_bags + packed_bag_idx), static_cast<uint32_t>(B - 1));
    const float inv_L = (mean_pooling && Ls[i] != 0) ? static_cast<float>(1.0) / Ls[i] : static_cast<float>(1.0);

    if constexpr (std::is_same_v<output_t, float> || std::is_same_v<output_t, at::Half> || std::is_same_v<output_t, at::BFloat16>) {
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        if constexpr (PackedMode) {
          output_d -= packed_bag_idx * kOutputsPerThread * num_stores_with_padding_per_row; 
        }
        accumulators[i][j].mul(inv_L);

        if (output_d >= 0 && output_d < D && (!PackedMode || packed_bag_idx < num_packed_bags)) {
          const int num_valid_outputs = min(static_cast<int>(D - output_d), static_cast<int>({{ (32 // emb_weight_type.bit_width) }}));
          accumulators[i][j].store(&output[b][D_start + output_d], num_valid_outputs);
        }

      }
    } else if constexpr (std::is_same_v<output_t, uint8_t>) {
      // INT8:
      // apply per feature row-wise int8
      float thread_local_min = std::numeric_limits<float>::max();
      float thread_local_max = std::numeric_limits<float>::lowest();
      float2 qparams;
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        accumulators[i][j].mul(inv_L);
        if (output_d >= 0 && output_d < D) {
          thread_local_max = max(thread_local_max, float{{ (32 // emb_weight_type.bit_width) }}_max(accumulators[i][j].acc));
          thread_local_min = min(thread_local_min, float{{ (32 // emb_weight_type.bit_width) }}_min(accumulators[i][j].acc));
        }
      }

      qparams = warp_find_qparams(thread_local_min, thread_local_max);
      const int output_D_start = D_start + t * 8;
      const int output_D_end = output_D_start + D;
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        const int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        if (output_d >= 0 && output_d < D) {
          const int num_valid_outputs = min(static_cast<int>(D - output_d), static_cast<int>({{ (32 // emb_weight_type.bit_width) }}));
          accumulators[i][j].store(&output[b][output_D_start + output_d], qparams, num_valid_outputs);
        }
      }
      if (threadIdx.x == 0) {
        store_qparams_to_row(&output[b][output_D_end], qparams);
      }
    } else {
      // INT4: not implemented yet
    }
  }
  {% endif %}
}

// kWarpsPerBlock is defined in embedding_forward_quantized_split_nbit_host_template.cu
{% set warps_per_block = '4' %}

{% for packed_mode in ['true', 'false'] %}
{% for device_only in ['true', 'false'] %}
{% for output_type in ['at::Half', 'at::BFloat16', 'float', 'uint8_t'] %}
{% for index_type in ['int32_t', 'int64_t'] %}
{% for params in emb_weight_type.template_params %}

{% if output_type == 'at::BFloat16' %}
#if defined(USE_ROCM) || !(                             \
    ((defined(CUDA_VERSION) && CUDA_VERSION < 11000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
{% endif %}

template __launch_bounds__({{ warps_per_block }} * kWarpSize) __global__
void {{ emb_weight_type.enum_name }}_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L
< {{ index_type }},
  {{ output_type }},
  {{ params.output_rows_per_thread }},
  {{ warps_per_block }},
  {{ params.input_rows_in_flight }},
  {{ params.min_128b_rows }},
  {{ params.max_128b_rows }},
  {{ device_only }},
  {{ packed_mode }} > (
  const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> dev_weights,
  const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> uvm_weights,
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
  const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
  const pta::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits> weights_tys,
  {% if not nobag %}
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
  {% else %}
  const int64_t D,
  {% endif %}
  FixedDivisor fd_B, // FixedDivisor(div_round_up(B, OutputRowsPerThread))
  const pta::PackedTensorAccessor32<{{ index_type }}, 1, at::RestrictPtrTraits> indices,
  const pta::PackedTensorAccessor32<{{ index_type }}, 1, at::RestrictPtrTraits> offsets,
  {% if not nobag %}
  const int64_t pooling_mode,
  {% endif %}
  const int64_t row_alignment,
  {% if weighted %}
  pta::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> indice_weights,
  {% endif %}
  {% if emb_weight_type.enum_name == "FP8" %}
  const int exponent_bits,
  const int exponent_bias,
  {% endif %}
  const int32_t num_packed_bags,
  pta::PackedTensorAccessor32<{{ output_type }}, 2, at::RestrictPtrTraits> output, // [B][total_D],
  const pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations
  );

{% if output_type == 'at::BFloat16' %}
#endif
{% endif %}

{% endfor %} // for params in emb_weight_type.template_params
{% endfor %} // for index_type in ['int32_t', 'int64_t']
{% endfor %} // for output_type in [True, False]
{% endfor %} // device_only in [True, False]
{% endfor %} // packed_bags in ['true', 'false']

}

                                      // clang-format on
