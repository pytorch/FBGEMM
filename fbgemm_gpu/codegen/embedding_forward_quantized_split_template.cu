/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// clang-format off
{% set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"

using namespace fbgemm_gpu;

using Tensor = at::Tensor;

namespace nbit {

constexpr int32_t kCacheLocationMissing = -1;

// "Effective" number of elements in the row when we include the row-wise quantization parameters.
__device__ inline int32_t padded_D(int32_t dim, SparseType weight_ty) {
    if (weight_ty == SparseType::FP32) { return dim; }
    if (weight_ty == SparseType::FP16) { return dim; }
    if (weight_ty == SparseType::FP8) { return dim; }
    if (weight_ty == SparseType::INT8) { return dim + 4; }
    if (weight_ty == SparseType::INT4) { return dim + 8; }
    if (weight_ty == SparseType::INT2) { return dim + 16; }
    return 0;
}

// ---------------------- start cp.async helpers, copied from CUTLASS

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void *ptr) {

// We prefer to use the new CVTA intrinsics if they are available, otherwise we will fall back to
// the previous internal intrinsics if they are available.
#if (! defined (__clang__) && defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)
  //
  // This NVVM intrinsic converts an address in shared memory to a plain
  // unsigned integer. This is necessary to pass to shared memory instructions
  // in inline PTX.
  //
  // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only available in 10.2].
  //
  //__device__ size_t __cvta_generic_to_shared(void* ptr);
  /// CUTLASS helper to get SMEM pointer
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
#elif (! defined (__clang__) && defined(__CUDA_ARCH__) &&  __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)
  return __nvvm_get_smem_pointer(ptr);
#elif defined(__CUDA_ARCH__)
  uint32_t smem_ptr;
  asm(
  "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
    : "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
#else
    return 0;
#endif
}

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void const *ptr) {
  return cutlass_get_smem_pointer(const_cast<void *>(ptr));
}

__device__ __forceinline__ void cp_async_fence() {
  #if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
  #endif
}

/// Partial specialization

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  #if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  #endif
}

/// Blocks until all previous cp.async.commit_group operations have committed.
template <>
__device__ __forceinline__ void cp_async_wait<0>() {
  #if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
  #endif
}

/// Partial specialization
template <int SizeInBytes>
__device__ __forceinline__
void cp_async_zfill_cg(void *smem_ptr, void const *global_ptr, bool pred_guard) {
#if __CUDA_ARCH__ >= 800
    static_assert(SizeInBytes == 16,
    "cp.async only supports CacheOperation::Global when access size is 16B.");

    unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
    int src_in_bytes = (pred_guard ? SizeInBytes : 0);
    asm volatile(
    "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
    "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));
#else
    static_assert(SizeInBytes == 16, "");
    using AccessType = uint4;
    if (pred_guard) {
      *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    } else {
      AccessType zeros;
      zeros.x = 0;
      zeros.y = 0;
      zeros.z = 0;
      zeros.w = 0;
      *static_cast<AccessType *>(smem_ptr) = zeros;
    }
#endif
}


/// Copy with zero fill
template <int SizeInBytes>
__device__ __forceinline__
void cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard) {
#if __CUDA_ARCH__ >= 800
    // Make sure the size is supported.
    static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
            "Size is not supported");

    unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
    const int src_in_bytes = pred_guard ? SizeInBytes : 0;

    asm volatile(
    "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
    "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));
#else
    static_assert(SizeInBytes == 16, "");
    using AccessType = uint4;
    if (pred_guard) {
      *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    } else {
      AccessType zeros;
      zeros.x = 0;
      zeros.y = 0;
      zeros.z = 0;
      zeros.w = 0;
      *static_cast<AccessType *>(smem_ptr) = zeros;
    }
#endif
}

{% for nobag in [True, False] %}
{% if not nobag or not weighted %}
// TODO: increase code sharing (templates for accumulator_ty, accumulation, outputs per thread, etc?)
{% for emb_weight_type in ["FP32", "FP16", "FP8", "INT8", "INT4", "INT2"] %}
template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows, bool DeviceOnly>
__launch_bounds__(WarpsPerBlock * kWarpSize)
__global__ void {{ type_map[emb_weight_type].enum_name }}_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L(
  const at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> dev_weights,
  const at::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> uvm_weights,
  const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
  const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
  const at::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits> weights_tys,
  {% if not nobag %}
  const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
  {% else %}
  const int64_t D,
  {% endif %}
  FixedDivisor fd_B, // FixedDivisor(div_round_up(B, OutputRowsPerThread))
  const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
  const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
  {% if not nobag %}
  const int64_t pooling_mode,
  {% endif %}
  const int64_t row_alignment,
  {% if weighted %}
  at::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits>
      indice_weights,
  {% endif %}
  {% if type_map[emb_weight_type].enum_name == "FP8" %}
  const int exponent_bits,
  const int exponent_bias,
  {% endif %}
  at::PackedTensorAccessor32<output_t, 2, at::RestrictPtrTraits>
      output, // [B][total_D],
  const at::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
  const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations
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
    std::is_same<output_t, float>::value || std::is_same<output_t, at::BFloat16>::value || std::is_same<output_t, at::Half>::value || std::is_same<output_t, uint8_t>::value,
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
  if (weight_ty != SparseType::{{ type_map[emb_weight_type].enum_name }}) {
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

  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
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
  constexpr size_t kOutputsPerThread = {{ (32 // type_map[emb_weight_type].bit_width) }};

  constexpr uint32_t NumUint4LoadsPerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  {% if not nobag %}
  VecNT<{{ (32 // type_map[emb_weight_type].bit_width) }}, PrimitiveType::{{ type_map[emb_weight_type].primitive_type }}> accumulators[OutputRowsPerThread][MaxNum128BRows];
  {% endif %}

  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);

    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4LoadsPerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * NumUint4LoadsPerRow; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % NumUint4LoadsPerRow;
      uint32_t input_row_idx = (load_idx / NumUint4LoadsPerRow);
      bool load_idx_valid = row_load_idx < uint4_loads_per_row;
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
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
        cp_async_zfill_cg<sizeof(uint4)>(&buffers[warp_idx][i][input_row_idx][row_load_idx], &row[row_load_idx], valid);

        {% if weighted %}
        buffers_indice_weights[warp_idx][i][input_row_idx] = valid ? indice_weights[indices_starts[i] + L_start + input_row_idx] : 0.0;
        {% endif %}
      }
    }
    // equivalent to fence + wait.
    cp_async_wait<0>();
    syncwarp();
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        if (!valid) {
          continue;
        }
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);
        // scale and bias are at the beginning of each row.
        // rationale: have scale/shift at start since these get loaded first
        // and then broadcasted around so it might speed up the first cache miss.
        {% if type_map[emb_weight_type].primitive_type == "INT" %}
        half2 shift_scale = reinterpret_cast<const half2*>(row)[0];
        {% endif %}

        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx];
        {% endif %}

        using scalar_t = {{ type_map[emb_weight_type].cpp_type_name }};

        {% if not nobag %}
        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          scalar_t v = reinterpret_cast<const scalar_t*>(row)[kWarpSize * j + threadIdx.x];
          {% if weighted %}
          accumulators[i][j].fma(v, {% if type_map[emb_weight_type].primitive_type == "INT" %} shift_scale, {% elif type_map[emb_weight_type].enum_name == "FP8" %} exponent_bits, exponent_bias, {% endif %} row_weight);
          {% else %}
          accumulators[i][j].add(v{% if type_map[emb_weight_type].primitive_type == "INT" %}, shift_scale {% elif type_map[emb_weight_type].enum_name == "FP8" %}, exponent_bits, exponent_bias {% endif %});
          {% endif %}
        }
        {% else %}
        const int32_t output_j = indices_starts[i] + L_start + input_row_idx;
        if (std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value || std::is_same<output_t, at::BFloat16>::value) {
          #pragma unroll MaxNum128BRows
          for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
            // Read the uint8/4/2 values: note that first 4 Bytes will be ditched later:
            // We shift back by 4/8/16 elements to remove the first 4 Bytes (which is garbage due to
            // the scale/shift handling).
            // Reason: to avoid divergence the first thread in the warp computes garbage.
            const int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
            scalar_t v = reinterpret_cast<const scalar_t*>(row)[kWarpSize * j + threadIdx.x];
            if (output_d >= 0 && output_d < D) {
              const int num_valid_outputs = min(static_cast<int>(D - output_d), static_cast<int>({{ (32 // type_map[emb_weight_type].bit_width) }}));
              VecNT<{{ (32 // type_map[emb_weight_type].bit_width) }}, PrimitiveType::{{ type_map[emb_weight_type].primitive_type }}> acc(v{% if type_map[emb_weight_type].primitive_type == "INT" %}, shift_scale {% elif type_map[emb_weight_type].enum_name == "FP8" %}, exponent_bits, exponent_bias {% endif %});
              acc.store(&output[output_j][output_d], num_valid_outputs);
            }
          }
        } else if (std::is_same<output_t, uint8_t>::value) {
          // INT8:
          // apply per feature row-wise int8
          float thread_local_min = std::numeric_limits<float>::max();
          float thread_local_max = std::numeric_limits<float>::lowest();
          float2 qparams;
          #pragma unroll MaxNum128BRows
          for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
            int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
            scalar_t v = reinterpret_cast<const scalar_t*>(row)[kWarpSize * j + threadIdx.x];
            VecNT<{{ (32 // type_map[emb_weight_type].bit_width) }}, PrimitiveType::{{ type_map[emb_weight_type].primitive_type }}> acc(v{% if type_map[emb_weight_type].primitive_type == "INT" %}, shift_scale {% elif type_map[emb_weight_type].enum_name == "FP8" %}, exponent_bits, exponent_bias {% endif %});
            if (output_d >= 0 && output_d < D) {
              thread_local_max = max(thread_local_max, float{{ (32 // type_map[emb_weight_type].bit_width) }}_max(acc.acc));
              thread_local_min = min(thread_local_min, float{{ (32 // type_map[emb_weight_type].bit_width) }}_min(acc.acc));
            }
          }
          qparams = warp_find_qparams(thread_local_min, thread_local_max);
          #pragma unroll MaxNum128BRows
          for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
            const int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
            scalar_t v = reinterpret_cast<const scalar_t*>(row)[kWarpSize * j + threadIdx.x];
            if (output_d >= 0 && output_d < D) {
              const int num_valid_outputs = min(static_cast<int>(D - output_d), static_cast<int>({{ (32 // type_map[emb_weight_type].bit_width) }}));
              VecNT<{{ (32 // type_map[emb_weight_type].bit_width) }}, PrimitiveType::{{ type_map[emb_weight_type].primitive_type }}> acc(v{% if type_map[emb_weight_type].primitive_type == "INT" %}, shift_scale {% elif type_map[emb_weight_type].enum_name == "FP8" %}, exponent_bits, exponent_bias {% endif %});
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
    const uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    const float inv_L = (mean_pooling && Ls[i] != 0) ? static_cast<float>(1.0) / Ls[i]: static_cast<float>(1.0);

    if (std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value || std::is_same<output_t, at::BFloat16>::value) {
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        const int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        accumulators[i][j].mul(inv_L);

        if (output_d >= 0 && output_d < D) {
          const int num_valid_outputs = min(static_cast<int>(D - output_d), static_cast<int>({{ (32 // type_map[emb_weight_type].bit_width) }}));
          accumulators[i][j].store(&output[b][D_start + output_d], num_valid_outputs);
        }

      }
    } else if (std::is_same<output_t, uint8_t>::value) {
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
          thread_local_max = max(thread_local_max, float{{ (32 // type_map[emb_weight_type].bit_width) }}_max(accumulators[i][j].acc));
          thread_local_min = min(thread_local_min, float{{ (32 // type_map[emb_weight_type].bit_width) }}_min(accumulators[i][j].acc));
        }
      }

      qparams = warp_find_qparams(thread_local_min, thread_local_max);
      const int output_D_start = D_start + t * 8;
      const int output_D_end = output_D_start + D;
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        const int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        if (output_d >= 0 && output_d < D) {
          const int num_valid_outputs = min(static_cast<int>(D - output_d), static_cast<int>({{ (32 // type_map[emb_weight_type].bit_width) }}));
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
{% endfor %} // for emb_weight_type in ["FP32", "FP16", "FP8", "INT8", "INT4", "INT2"]
{% endif %} // if not nobag or not weighted
{% endfor %} // for nobag in [True, False]

__device__ inline uint32_t pruned_hash_function(uint32_t h) {
    // MurmorHash3 32-bit mixing function.
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

__global__ __launch_bounds__(kMaxThreads) void int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_{{ wdesc }}_kernel(
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> offsets,
    const at::PackedTensorAccessor64<int32_t, 2, at::RestrictPtrTraits> hash_table,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_table_offsets,
    const int32_t B,
    const int32_t T,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dense_indices) {
    // uint32_t capacity = hash_table.size(0);
    const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    const int32_t t = b_t / B;
    const int32_t b = b_t % B;
    if (b_t >= B * T) {
        return;
    }
    const int32_t indices_start = offsets[t * B + b];
    const int32_t indices_end = offsets[t * B + b + 1];
    const int32_t L = indices_end - indices_start;

    const int64_t table_start = hash_table_offsets[t];
    const int64_t table_end = hash_table_offsets[t + 1];
    const int64_t capacity = table_end - table_start;

    if (capacity == 0) {
      // No pruning applied on the indices associated with this table.
      for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
        dense_indices[indices_start + l] = indices[indices_start + l];
      }
      return;
    }

    const uint32_t subwarp_id = threadIdx.x / 4;
    const uint32_t subwarp_tid = threadIdx.x % 4;
#ifdef __HIP_PLATFORM_HCC__
    const uint64_t subwarp_mask = static_cast<uint64_t>(0xF) << (4 * subwarp_id);
#else
    const uint32_t subwarp_mask = static_cast<uint32_t>(0xF) << (4 * subwarp_id);
#endif
    for (int32_t l_start = 0; l_start + subwarp_id < L; l_start += kWarpSize / 4) {
        const int32_t idx = indices[indices_start + l_start + subwarp_id];
        uint32_t slot_start = pruned_hash_function(static_cast<uint32_t>(idx)) % capacity;
        while (true) {
            const uint32_t slot = (slot_start + subwarp_tid) % capacity;
            const int2 val = *reinterpret_cast<const int2*>(&hash_table[table_start + static_cast<int64_t>(slot)][0]);
            const int32_t slot_sparse_idx = val.x;
            const int32_t slot_dense_idx = val.y;

            bool found = false;
            bool empty = false;
            if (slot_sparse_idx == -1) {
                empty = true;
            } else if (slot_sparse_idx == idx) {
                found = true;
                dense_indices[indices_start + l_start + subwarp_id] = slot_dense_idx;
            }
            if (__any_sync(subwarp_mask, found)) {
                break;
            } else if (__any_sync(subwarp_mask, empty)) {
                dense_indices[indices_start + l_start + subwarp_id] = -1;
                break;
            }
            slot_start += 4;
        }
    }
}

{% if not weighted %}
__global__ __launch_bounds__(kMaxThreads) void int_nbit_split_embedding_codegen_forward_pruned_array_lookup_kernel(
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> offsets,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> index_remappings,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> index_remappings_offsets,
    const int32_t B,
    const int32_t T,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> dense_indices) {
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t t = b_t / B;
  const int32_t b = b_t % B;
  if (b_t >= B * T) {
      return;
  }
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  const int32_t L = indices_end - indices_start;

  const int64_t index_remappings_start = index_remappings_offsets[t];
  const int64_t index_remappings_end = index_remappings_offsets[t + 1];
  const int64_t capacity = index_remappings_end - index_remappings_start;

  if (capacity > 0) {
    for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
        int32_t idx = indices[indices_start + l];
        dense_indices[indices_start + l] = index_remappings[index_remappings_start + idx];
    }
  } else {
    for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
        dense_indices[indices_start + l] = indices[indices_start + l];
    }
  }
}
{% endif %}

{% if not weighted %}
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void int_nbit_split_embedding_codegen_forward_pruned_array_lookup_from_row_idx_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> update_row_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> update_table_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> index_remappings,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> index_remappings_offsets,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> dense_indices) {

  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= update_row_indices.size(0)) {
    return;
  }
  const int table_idx = update_table_indices[idx];
  const auto row_idx = update_row_indices[idx];

  const int64_t index_remappings_start = index_remappings_offsets[table_idx];
  const int64_t index_remappings_end = index_remappings_offsets[table_idx + 1];
  const int64_t capacity = index_remappings_end - index_remappings_start;

  if (capacity > 0) {
    dense_indices[idx] = index_remappings[index_remappings_start + row_idx];
  } else {
    dense_indices[idx] = row_idx;
  }
}
{% endif %}



}

{% for nobag in [True, False] %}
{% if not nobag or not weighted %}
Tensor int_nbit_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    {% if not nobag %}
    Tensor D_offsets,
    const int64_t total_D,
    {% else %}
    const int64_t D,
    {% endif %}
    const int64_t max_int2_D,
    const int64_t max_int4_D,
    const int64_t max_int8_D,
    const int64_t max_float16_D,
    const int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    {% if not nobag %}
    const int64_t pooling_mode,
    {% endif %}
    const int64_t row_alignment,
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    const int64_t output_dtype,
    Tensor lxu_cache_weights,
    Tensor lxu_cache_locations,
    const int64_t max_float8_D,
    const int64_t fp8_exponent_bits,
    const int64_t fp8_exponent_bias
) {
    TENSOR_ON_CUDA_GPU(dev_weights);
    TENSORS_ON_SAME_DEVICE(uvm_weights, dev_weights);
    TENSORS_ON_SAME_DEVICE(weights_placements, dev_weights);
    TENSORS_ON_SAME_DEVICE(weights_offsets, dev_weights);
    TENSORS_ON_SAME_DEVICE(weights_tys, dev_weights);
    {% if not nobag %}
    TENSORS_ON_SAME_DEVICE(D_offsets, dev_weights);
    {% endif %}
    TENSORS_ON_SAME_DEVICE(indices, dev_weights);
    TENSORS_ON_SAME_DEVICE(offsets, dev_weights);
    {% if weighted %}
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(indice_weights, dev_weights);
    {% endif %}
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(lxu_cache_weights, dev_weights);
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(lxu_cache_locations, dev_weights);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    // kernels assume indices are contiguous.
    indices = indices.contiguous();

    {% if not nobag %}
    const int32_t T = D_offsets.numel() - 1;
    {% else %}
    const int32_t total_L = indices.numel();
    const int32_t T = weights_offsets.numel();
    {% endif %}
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    const int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);

    {% if not nobag %}
    TORCH_CHECK(total_D > 0);
    {% else %}
    TORCH_CHECK(D > 0);
    {% endif %}

    Tensor output;
    const int kINT8QparamsBytes = 8;
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 || o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    {% if not nobag %}
    int64_t total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        total_adjusted_D += T * kINT8QparamsBytes;
    }
    if (indices.numel() == 0) {
      output = at::zeros({B, total_adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    }
    else {
      output = at::empty({B, total_adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    }
    {% else %}
    int64_t adjusted_D = D;
    if (o_dtype == SparseType::INT8) {
        adjusted_D += T * kINT8QparamsBytes;
    }
    if (total_L == 0) {
      output = at::zeros({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    }
    else {
      output = at::empty({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    }

    {% endif %}

    if (B == 0 || indices.numel() == 0) {
      return output;
    }

    using index_t = int32_t;

    constexpr int32_t kWarpsPerBlock = 4;

    const auto device_only = lxu_cache_weights.numel() == 0 && uvm_weights.numel() == 0;
    #define Y(...) \
      if (device_only) { \
        X(true, __VA_ARGS__) \
      } else { \
        X(false, __VA_ARGS__) \
      };

    // launch 2-bit kernel
    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::INT2_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "int2_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_int2_D > 0) {
        auto max_int2_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int2_D, SparseType::INT2, row_alignment), 128);
        TORCH_CHECK(max_int2_128b_rows <= 2);
        if (max_int2_128b_rows > 0) {
          Y(2, 16, 0, 1);
        }
        if (max_int2_128b_rows > 1) {
          Y(2, 8, 1, 2);
        }
      }
    }));
    #undef X


    // launch 4-bit kernel
    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::INT4_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "int4_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_int4_D > 0) {
        auto max_int4_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int4_D, SparseType::INT4, row_alignment), 128);
        TORCH_CHECK(max_int4_128b_rows <= 4);
        if (max_int4_128b_rows > 0) {
          Y(4, 8, 0, 1);
        }
        if (max_int4_128b_rows > 1) {
          Y(2, 8, 1, 2);
        }
        if (max_int4_128b_rows > 2) {
          Y(1, 4, 2, 4);
        }
      }
    }));
    #undef X

    // launch 8-bit int kernel
    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::INT8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "int8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_int8_D > 0) {
        auto max_int8_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int8_D, SparseType::INT8, row_alignment), 128);
        TORCH_CHECK(max_int8_128b_rows <= 8);
        if (max_int8_128b_rows > 0) {
          Y(2, 8, 0, 1);
        }
        if (max_int8_128b_rows > 1) {
          Y(2, 4, 1, 2);
        }
        if (max_int8_128b_rows > 2) {
          Y(2, 4, 2, 4);
        }
        if (max_int8_128b_rows > 4) {
          Y(2, 4, 4, 8);
        }
      }
    }));
    #undef X

    // launch 8-bit float kernel
    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::FP8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        fp8_exponent_bits, \
        fp8_exponent_bias, \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "fp8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_float8_D > 0) {
        auto max_fp8_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float8_D, SparseType::FP8, row_alignment), 128);
        TORCH_CHECK(max_fp8_128b_rows <= 8);
        if (max_fp8_128b_rows > 0) {
          Y(2, 8, 0, 1);
        }
        if (max_fp8_128b_rows > 1) {
          Y(2, 4, 1, 2);
        }
        if (max_fp8_128b_rows > 2) {
          Y(2, 4, 2, 4);
        }
        if (max_fp8_128b_rows > 4) {
          Y(2, 4, 4, 8);
        }
      }
    }));
    #undef X

    // launch 16-bit kernel
    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::FP16_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "fp16_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_float16_D > 0) {
        auto max_fp16_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float16_D, SparseType::FP16, row_alignment), 128);
        TORCH_CHECK(max_fp16_128b_rows <= 16);
        if (max_fp16_128b_rows > 0) {
          Y(2, 8, 0, 2);
        }
        if (max_fp16_128b_rows > 2) {
          Y(2, 8, 2, 4);
        }
        if (max_fp16_128b_rows > 4) {
          Y(2, 4, 4, 8);
        }
        if (max_fp16_128b_rows > 8) {
          Y(2, 2, 8, 16);
        }
      }
    }));
    #undef X

    // launch 32-bit kernel
    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::FP32_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "fp32_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_float32_D > 0) {
        auto max_fp32_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float32_D, SparseType::FP32, row_alignment), 128);
        TORCH_CHECK(max_fp32_128b_rows <= 32);
        if (max_fp32_128b_rows > 0) {
          Y(2, 4, 0, 4);
        }
        if (max_fp32_128b_rows > 4) {
          Y(2, 2, 4, 16);
        }
        if (max_fp32_128b_rows > 16) {
          Y(1, 1, 16, 32);
        }
      }
    }));
    #undef X

    return output;
}
{% endif %}  // if not nobag or not weighted
{% endfor %}  // for nobag in [True, False]

Tensor pruned_hashmap_lookup_{{ wdesc }}_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets) {

    TENSOR_ON_CUDA_GPU(indices);
    TENSOR_ON_CUDA_GPU(offsets);
    TENSOR_ON_CUDA_GPU(hash_table);
    TENSOR_ON_CUDA_GPU(hash_table_offsets);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(indices.get_device());
    auto dense_indices = at::empty_like(indices);
    const int32_t T = hash_table_offsets.size(0) - 1;
    const int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    TORCH_CHECK(hash_table.size(0) < std::numeric_limits<int32_t>::max());
    constexpr size_t kForwardMaxThreads = 256;
    nbit::int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_{{ wdesc }}_kernel<<<
        nbit::div_round_up(B * T + 1, kForwardMaxThreads / kWarpSize),
        dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
        0,
        at::cuda::getCurrentCUDAStream()>>>(
            indices.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            hash_table.packed_accessor64<int32_t, 2, at::RestrictPtrTraits>(),
            hash_table_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            B,
            T,
            dense_indices.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dense_indices;
}

{% if not weighted %}
Tensor pruned_array_lookup_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor index_remappings,
    Tensor index_remappings_offsets) {

  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(offsets);
  TENSOR_ON_CUDA_GPU(index_remappings);
  TENSOR_ON_CUDA_GPU(index_remappings_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());
  auto dense_indices = at::empty_like(indices);
  const int32_t T = index_remappings_offsets.size(0) - 1;
  TORCH_CHECK(
      (offsets.size(0) - 1) % T == 0,
      "offsets.size() - 1 is not divisible by T! offsets.size: ",
      offsets.size(0),
      "T: ",
      T
  );
  const int32_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B > 0, "offsets.size(): ", offsets.size(0), ", T: ", T, ", B: ", B);
  TORCH_CHECK(index_remappings.size(0) < std::numeric_limits<int64_t>::max());
  TORCH_CHECK(indices.dim() == 1, "Tensor dim: ", indices.dim());
  TORCH_CHECK(offsets.dim() == 1, "Tensor dim: ", offsets.dim());
  TORCH_CHECK(index_remappings.dim() == 1, "Tensor dim: ", index_remappings.dim());
  TORCH_CHECK(index_remappings_offsets.dim() == 1, "Tensor dim: ", index_remappings_offsets.dim());
  TORCH_CHECK(dense_indices.dim() == 1, "Tensor dim: ", dense_indices.dim());
  constexpr size_t kForwardMaxThreads = 256;
  nbit::int_nbit_split_embedding_codegen_forward_pruned_array_lookup_kernel<<<
      nbit::div_round_up(offsets.size(0), kForwardMaxThreads / kWarpSize),
      dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
          indices.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
          offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
          index_remappings.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
          index_remappings_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
          B,
          T,
          dense_indices.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dense_indices;
}

Tensor pruned_array_lookup_from_row_idx_cuda(
    Tensor update_row_indices,
    Tensor update_table_indices,
    Tensor index_remappings,
    Tensor index_remappings_offsets) {

  TENSOR_ON_CUDA_GPU(update_row_indices);
  TENSOR_ON_CUDA_GPU(update_table_indices);
  TENSOR_ON_CUDA_GPU(index_remappings);
  TENSOR_ON_CUDA_GPU(index_remappings_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(update_table_indices.get_device());
  auto dense_indices = at::empty_like(update_row_indices);
  const int32_t T = index_remappings_offsets.size(0) - 1;

  const auto num_indices = update_row_indices.numel();
  if (num_indices == 0) {
    return dense_indices;
  }

  TORCH_CHECK(index_remappings.size(0) < std::numeric_limits<int64_t>::max());
  TORCH_CHECK(update_row_indices.dim() == 1, "Tensor dim: ", update_row_indices.dim());
  TORCH_CHECK(update_table_indices.dim() == 1, "Tensor dim: ", update_table_indices.dim());
  TORCH_CHECK(index_remappings.dim() == 1, "Tensor dim: ", index_remappings.dim());
  TORCH_CHECK(index_remappings_offsets.dim() == 1, "Tensor dim: ", index_remappings_offsets.dim());
  TORCH_CHECK(dense_indices.dim() == 1, "Tensor dim: ", dense_indices.dim());
  constexpr size_t kForwardMaxThreads = 256;

  AT_DISPATCH_INDEX_TYPES(
      update_row_indices.scalar_type(), "embedding_inplace_update_kernel", [&] {
        nbit::int_nbit_split_embedding_codegen_forward_pruned_array_lookup_from_row_idx_kernel<<<
            nbit::div_round_up(num_indices, kForwardMaxThreads),
            kForwardMaxThreads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
                update_row_indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                update_table_indices.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                index_remappings.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                index_remappings_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                dense_indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return dense_indices;
}
{% endif %}

                                    // clang-format on
