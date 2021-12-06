/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// clang-format off
{% set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"

using namespace fbgemm_gpu;

namespace nbit {

using namespace at;

constexpr int32_t kCacheLocationMissing = -1;

__forceinline__ __host__ __device__ uint32_t round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b) * b;
}


__forceinline__ __host__ __device__ uint32_t div_round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b);
}

__host__ __device__ inline int32_t unpadded_row_size_in_bytes(int32_t dim, SparseType weight_ty) {
    if (weight_ty == SparseType::FP32) { return dim * 4; }
    if (weight_ty == SparseType::FP16) { return dim * 2; }
    if (weight_ty == SparseType::INT8) { return dim + 4; }
    if (weight_ty == SparseType::INT4) { return dim / 2 + 4; }
    if (weight_ty == SparseType::INT2) { return dim / 4 + 4; }
    return 0;
}

__host__ __device__ inline int32_t padded_row_size_in_bytes(int32_t dim, SparseType weight_ty) {
  auto r = unpadded_row_size_in_bytes(dim, weight_ty);
  return round_up(r, 16);
}

// "Effective" number of elements in the row when we include the row-wise quantization parameters.
__device__ inline int32_t padded_D(int32_t dim, SparseType weight_ty) {
    if (weight_ty == SparseType::FP32) { return dim; }
    if (weight_ty == SparseType::FP16) { return dim; }
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
    int src_in_bytes = (pred_guard ? SizeInBytes : 0);

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

// TODO: increase code sharing (templates for accumulator_ty, accumulation, outputs per thread, etc?)
template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows>
__launch_bounds__(WarpsPerBlock * 32)
__global__ void fp32_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L(
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> uvm_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
  const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
  const PackedTensorAccessor32<uint8_t, 1, RestrictPtrTraits> weights_tys,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
  int64_t pooling_mode,
  {% if weighted %}
  PackedTensorAccessor32<float, 1, RestrictPtrTraits>
      indice_weights,
  {% endif %}
  PackedTensorAccessor32<output_t, 2, RestrictPtrTraits>
      output, // [B][total_D],
  const PackedTensorAccessor64<uint8_t, 2, RestrictPtrTraits> lxu_cache_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations
  ) {
  int32_t B = output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t bb_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (bb_t >= div_round_up(B, OutputRowsPerThread) * T) {
      return;
  }
  static_assert(
    std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value || std::is_same<output_t, uint8_t>::value,
    "output_t can only be float or half or bytes now"
  );

  uint32_t t = bb_t / div_round_up(B, OutputRowsPerThread);

  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
  if (weight_ty != SparseType::FP32) {
      return;
  }

  const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty);

  if (D_bytes <= MinNum128BRows * 128 || D_bytes > MaxNum128BRows * 128) {
    return;
  }

  uint32_t bb = bb_t % div_round_up(B, OutputRowsPerThread);

  int64_t weights_offset = weights_offsets[t];
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

  const uint8_t* __restrict__ weights;
  const auto placement = static_cast<PlacementType>(weights_placements[t]);
  if (placement == PlacementType::DEVICE) {
      weights = &dev_weights[weights_offset];
  } else {
      weights = &uvm_weights[weights_offset];
  }
  constexpr size_t kOutputsPerThread = 1;

  constexpr uint32_t NumUint4PerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  VecNT<1> accumulators[OutputRowsPerThread][MaxNum128BRows];

  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);
    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4PerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * uint4_loads_per_row; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % uint4_loads_per_row;
      uint32_t input_row_idx = (load_idx / uint4_loads_per_row);

      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        bool cache_valid = (placement == PlacementType::MANAGED_CACHING && valid);
        int32_t idx = valid ? indices[indices_starts[i] + L_start + input_row_idx] : -1;
        int32_t cache_idx = cache_valid ? lxu_cache_locations[indices_starts[i] + L_start + input_row_idx] : -1;
        valid = valid && (idx != -1);
        const uint4* row;
        if (cache_valid && cache_idx != kCacheLocationMissing) {
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
    __syncwarp();
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);
        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx];
        {% endif %}
        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          float v = reinterpret_cast<const float*>(row)[kWarpSize * j + threadIdx.x];
          if (valid) {
            {% if weighted %}
            accumulators[i][j].fma(v, row_weight);
            {% else %}
            accumulators[i][j].add(v);
            {% endif %}
          }
        }
      }
    }
  }
  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    float inv_L = 1.0 / Ls[i];

    if (std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value) {
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && Ls[i] != 0) {
            accumulators[i][j].mul(inv_L);
        }
        if (output_d >= 0 && output_d < D) {
          accumulators[i][j].store(&output[b][D_start + output_d]);
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
        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && Ls[i] != 0) {
            accumulators[i][j].mul(inv_L);
        }
        if (output_d >= 0 && output_d < D) {
          thread_local_max = max(thread_local_max, accumulators[i][j].acc);
          thread_local_min = min(thread_local_min, accumulators[i][j].acc);
        }
      }
      qparams = warp_find_qparams(thread_local_min, thread_local_max);
      int output_D_start = D_start + t * 8;
      int output_D_end = output_D_start + D;
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        if (output_d >= 0 && output_d < D) {
          accumulators[i][j].store(&output[b][output_D_start + output_d], qparams);
        }
      }
      if (threadIdx.x == 0) {
        store_qparams_to_row(&output[b][output_D_end], qparams);
      }
    } else {
      // INT4: not implemented yet
    }
  }
}

// TODO: increase code sharing (templates for accumulator_ty, accumulation, outputs per thread, etc?)
template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows>
__launch_bounds__(WarpsPerBlock * 32)
__global__ void fp16_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L(
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> uvm_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
  const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
  const PackedTensorAccessor32<uint8_t, 1, RestrictPtrTraits> weights_tys,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
  int64_t pooling_mode,
  {% if weighted %}
  PackedTensorAccessor32<float, 1, RestrictPtrTraits>
      indice_weights,
  {% endif %}
  PackedTensorAccessor32<output_t, 2, RestrictPtrTraits>
      output, // [B][total_D],
  const PackedTensorAccessor64<uint8_t, 2, RestrictPtrTraits> lxu_cache_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations
  ) {
  int32_t B = output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t bb_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (bb_t >= div_round_up(B, OutputRowsPerThread) * T) {
      return;
  }
  static_assert(
    std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value || std::is_same<output_t, uint8_t>::value,
    "output_t can only be float or half or bytes now"
  );

  uint32_t t = bb_t / div_round_up(B, OutputRowsPerThread);

  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
  if (weight_ty != SparseType::FP16) {
      return;
  }

  const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty);

  if (D_bytes <= MinNum128BRows * 128 || D_bytes > MaxNum128BRows * 128) {
    return;
  }

  uint32_t bb = bb_t % div_round_up(B, OutputRowsPerThread);

  int64_t weights_offset = weights_offsets[t];
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

  const uint8_t* __restrict__ weights;
  const auto placement = static_cast<PlacementType>(weights_placements[t]);
  if (placement == PlacementType::DEVICE) {
      weights = &dev_weights[weights_offset];
  } else {
      weights = &uvm_weights[weights_offset];
  }
  constexpr size_t kOutputsPerThread = 2;

  constexpr uint32_t NumUint4PerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  VecNT<2> accumulators[OutputRowsPerThread][MaxNum128BRows];

  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);

    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4PerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * uint4_loads_per_row; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % uint4_loads_per_row;
      uint32_t input_row_idx = (load_idx / uint4_loads_per_row);

      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        bool cache_valid = (placement == PlacementType::MANAGED_CACHING && valid);
        int32_t idx = valid ? indices[indices_starts[i] + L_start + input_row_idx] : -1;
        int32_t cache_idx = cache_valid ? lxu_cache_locations[indices_starts[i] + L_start + input_row_idx] : -1;
        valid = valid && (idx != -1);
        const uint4* row;
        if (cache_valid && cache_idx != kCacheLocationMissing) {
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
    __syncwarp();
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);

        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx];
        {% endif %}

        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          __half2 v = reinterpret_cast<const __half2*>(row)[kWarpSize * j + threadIdx.x];

          if (valid) {
            {% if weighted %}
            accumulators[i][j].fma(v, row_weight);
            {% else %}
            accumulators[i][j].add(v);
            {% endif %}
          }
        }
      }
    }
  }

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    float inv_L = 1.0 / Ls[i];

    if (std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value) {

      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && Ls[i] != 0) {
            accumulators[i][j].mul(inv_L);
        }
        if (output_d >= 0 && output_d < D) {
          accumulators[i][j].store(&output[b][D_start + output_d]);
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
        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && Ls[i] != 0) {
            accumulators[i][j].mul(inv_L);
        }
        if (output_d >= 0 && output_d < D) {
          thread_local_max = max(thread_local_max, max(accumulators[i][j].acc.x, accumulators[i][j].acc.y));
          thread_local_min = min(thread_local_min, min(accumulators[i][j].acc.x, accumulators[i][j].acc.y));
        }
      }

      qparams = warp_find_qparams(thread_local_min, thread_local_max);
      int output_D_start = D_start + t * 8;
      int output_D_end = output_D_start + D;
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        if (output_d >= 0 && output_d < D) {
          accumulators[i][j].store(&output[b][output_D_start + output_d], qparams);
        }
      }
      if (threadIdx.x == 0) {
        store_qparams_to_row(&output[b][output_D_end], qparams);
      }
    } else {
      // INT4: not implemented yet
    }
  }
}

template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows>
__launch_bounds__(WarpsPerBlock * 32)
__global__ void int_8bit_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L(
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> uvm_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
  const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
  const PackedTensorAccessor32<uint8_t, 1, RestrictPtrTraits> weights_tys,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
  int64_t pooling_mode,
  {% if weighted %}
  PackedTensorAccessor32<float, 1, RestrictPtrTraits>
      indice_weights,
  {% endif %}
  PackedTensorAccessor32<output_t, 2, RestrictPtrTraits>
      output, // [B][total_D]
  const PackedTensorAccessor64<uint8_t, 2, RestrictPtrTraits> lxu_cache_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations
  ) {
  int32_t B = output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t bb_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (bb_t >= div_round_up(B, OutputRowsPerThread) * T) {
      return;
  }
  static_assert(
    std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value || std::is_same<output_t, uint8_t>::value,
    "output_t can only be float or half or bytes now"
  );

  uint32_t t = bb_t / div_round_up(B, OutputRowsPerThread);

  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
  if (weight_ty != SparseType::INT8) {
      return;
  }

  const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty);

  if (D_bytes <= MinNum128BRows * 128 || D_bytes > MaxNum128BRows * 128) {
    return;
  }

  uint32_t bb = bb_t % div_round_up(B, OutputRowsPerThread);

  int64_t weights_offset = weights_offsets[t];
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

  const uint8_t* __restrict__ weights;
  const auto placement = static_cast<PlacementType>(weights_placements[t]);
  if (placement == PlacementType::DEVICE) {
      weights = &dev_weights[weights_offset];
  } else {
      weights = &uvm_weights[weights_offset];
  }
  constexpr size_t kOutputsPerThread = 4;

  constexpr uint32_t NumUint4PerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  VecNT<4> accumulators[OutputRowsPerThread][MaxNum128BRows];

  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);

    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4PerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * uint4_loads_per_row; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % uint4_loads_per_row;
      uint32_t input_row_idx = (load_idx / uint4_loads_per_row);

      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        bool cache_valid = (placement == PlacementType::MANAGED_CACHING && valid);
        int32_t idx = valid ? indices[indices_starts[i] + L_start + input_row_idx] : -1;
        int32_t cache_idx = cache_valid ? lxu_cache_locations[indices_starts[i] + L_start + input_row_idx] : -1;
        valid = valid && (idx != -1);
        const uint4* row;
        if (cache_valid && cache_idx != kCacheLocationMissing) {
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
    __syncwarp();
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);
        half2 shift_scale = reinterpret_cast<const half2*>(row)[0];

        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx];
        {% endif %}

        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          uint32_t v = reinterpret_cast<const uint32_t*>(row)[kWarpSize * j + threadIdx.x];
          if (valid) {
            {% if weighted %}
            accumulators[i][j].fma(v, shift_scale, row_weight);
            {% else %}
            accumulators[i][j].add(v, shift_scale);
            {% endif %}
          }
        }
      }
    }
  }

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    float inv_L = 1.0 / Ls[i];

    if (std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value) {
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;

        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && Ls[i] != 0) {
          accumulators[i][j].mul(inv_L);
        }

        if (output_d >= 0 && output_d < D) {
          accumulators[i][j].store(&output[b][D_start + output_d]);
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
        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && Ls[i] != 0) {
          accumulators[i][j].mul(inv_L);
        }
        if (output_d >= 0 && output_d < D) {
          thread_local_max = max(thread_local_max, float4_max(accumulators[i][j].acc));
          thread_local_min = min(thread_local_min, float4_min(accumulators[i][j].acc));
        }
      }

      qparams = warp_find_qparams(thread_local_min, thread_local_max);
      int output_D_start = D_start + t * 8;
      int output_D_end = output_D_start + D;
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        if (output_d >= 0 && output_d < D) {
          accumulators[i][j].store(&output[b][output_D_start + output_d], qparams);
        }
      }
      if (threadIdx.x == 0) {
        store_qparams_to_row(&output[b][output_D_end], qparams);
      }
    } else {
      // INT4: not implemented yet
    }
  }
}

template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows>
__launch_bounds__(WarpsPerBlock * 32)
__global__ void int_4bit_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L(
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
  const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> uvm_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
  const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
  const PackedTensorAccessor32<uint8_t, 1, RestrictPtrTraits> weights_tys,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
  const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
  int64_t pooling_mode,
  {% if weighted %}
  PackedTensorAccessor32<float, 1, RestrictPtrTraits>
      indice_weights,
  {% endif %}
  PackedTensorAccessor32<output_t, 2, RestrictPtrTraits>
      output, // [B][total_D],
  const PackedTensorAccessor64<uint8_t, 2, RestrictPtrTraits> lxu_cache_weights,
  const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations
  ) {
  int32_t B = output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t bb_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (bb_t >= div_round_up(B, OutputRowsPerThread) * T) {
    return;
  }
  static_assert(
    std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value || std::is_same<output_t, uint8_t>::value,
    "output_t can only be float or half or bytes now"
  );

  uint32_t t = bb_t / div_round_up(B, OutputRowsPerThread);

  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
  if (weight_ty != SparseType::INT4) {
      return;
  }

  const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty);

  if (D_bytes <= MinNum128BRows * 128 || D_bytes > MaxNum128BRows * 128) {
    return;
  }

  uint32_t bb = bb_t % div_round_up(B, OutputRowsPerThread);

  int64_t weights_offset = weights_offsets[t];
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

  const uint8_t* __restrict__ weights;
  const auto placement = static_cast<PlacementType>(weights_placements[t]);
  if (placement == PlacementType::DEVICE) {
      weights = &dev_weights[weights_offset];
  } else {
      weights = &uvm_weights[weights_offset];
  }
  constexpr size_t kOutputsPerThread = 8;

  constexpr uint32_t NumUint4PerRow = MaxNum128BRows * 128 / sizeof(uint4);
  const uint32_t uint4_loads_per_row = div_round_up(D_bytes, sizeof(uint4));

  VecNT<8> accumulators[OutputRowsPerThread][MaxNum128BRows];

  for (uint32_t L_start = 0; L_start < max_Ls; L_start += InputRowsInFlight) {
    uint32_t input_rows_in_flight = min(static_cast<uint32_t>(InputRowsInFlight), max_Ls - L_start);

    typedef uint4 AllBuffers[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight][NumUint4PerRow];
    __shared__ AllBuffers buffers;

    {% if weighted %}
    typedef float AllIndiceWeights[WarpsPerBlock][OutputRowsPerThread][InputRowsInFlight];
    __shared__ AllIndiceWeights buffers_indice_weights;
    {% endif %}

    for (uint32_t load_idx = threadIdx.x; load_idx < input_rows_in_flight * uint4_loads_per_row; load_idx += kWarpSize) {
      uint32_t row_load_idx = load_idx % uint4_loads_per_row;
      uint32_t input_row_idx = (load_idx / uint4_loads_per_row);

      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        bool cache_valid = (placement == PlacementType::MANAGED_CACHING && valid);
        int32_t idx = valid ? indices[indices_starts[i] + L_start + input_row_idx] : -1;
        int32_t cache_idx = cache_valid ? lxu_cache_locations[indices_starts[i] + L_start + input_row_idx] : -1;
        valid = valid && (idx != -1);
        const uint4* row;
        if (cache_valid && cache_idx != kCacheLocationMissing) {
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
    __syncwarp();
    for (uint32_t input_row_idx = 0; input_row_idx < input_rows_in_flight; ++input_row_idx) {
      #pragma unroll OutputRowsPerThread
      for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
        bool valid = L_start + input_row_idx < Ls[i];
        const uint32_t* row = reinterpret_cast<const uint32_t*>(&buffers[warp_idx][i][input_row_idx][0]);
        half2 shift_scale = reinterpret_cast<const half2*>(row)[0];

        {% if weighted %}
        float row_weight = buffers_indice_weights[warp_idx][i][input_row_idx];
        {% endif %}

        #pragma unroll MaxNum128BRows
        for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
          uint32_t v = reinterpret_cast<const uint32_t*>(row)[kWarpSize * j + threadIdx.x];
          if (valid) {
            {% if weighted %}
            accumulators[i][j].fma(v, shift_scale, row_weight);
            {% else %}
            accumulators[i][j].add(v, shift_scale);
            {% endif %}
          }
        }
      }
    }
  }

  #pragma unroll OutputRowsPerThread
  for (uint32_t i = 0; i < OutputRowsPerThread; ++i) {
    uint32_t b = min(static_cast<uint32_t>(bb * OutputRowsPerThread + i), static_cast<uint32_t>(B - 1));
    float inv_L = 1.0 / Ls[i];

    if (std::is_same<output_t, float>::value || std::is_same<output_t, at::Half>::value) {
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;

        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && Ls[i] != 0) {
          accumulators[i][j].mul(inv_L);
        }

        if (output_d >= 0 && output_d < D) {
          accumulators[i][j].store(&output[b][D_start + output_d]);
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
        if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN && Ls[i] != 0) {
          accumulators[i][j].mul(inv_L);
        }
        if (output_d >= 0 && output_d < D) {
          thread_local_max = max(thread_local_max, float8_max(accumulators[i][j].acc));
          thread_local_min = min(thread_local_min, float8_min(accumulators[i][j].acc));
        }
      }

      qparams = warp_find_qparams(thread_local_min, thread_local_max);
      int output_D_start = D_start + t * 8;
      int output_D_end = output_D_start + D;
      #pragma unroll MaxNum128BRows
      for (uint32_t j = 0; j < MaxNum128BRows; ++j) {
        int32_t output_d = kWarpSize * j * kOutputsPerThread + threadIdx.x * kOutputsPerThread - D_padding;
        if (output_d >= 0 && output_d < D) {
          accumulators[i][j].store(&output[b][output_D_start + output_d], qparams);
        }
      }
      if (threadIdx.x == 0) {
        store_qparams_to_row(&output[b][output_D_end], qparams);
      }
    } else {
      // INT4: not implemented yet
    }
  }
}

__device__ inline uint32_t pruned_hash_function(uint32_t h) {
    // MurmorHash3 32-bit mixing function.
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

__global__ void int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_{{ wdesc }}_kernel(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor64<int32_t, 2, RestrictPtrTraits> hash_table,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_table_offsets,
    int32_t B,
    int32_t T,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> dense_indices) {
    // uint32_t capacity = hash_table.size(0);
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t t = b_t / B;
    int32_t b = b_t % B;
    if (b_t >= B * T) {
        return;
    }
    int32_t indices_start = offsets[t * B + b];
    int32_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;

    int64_t table_start = hash_table_offsets[t];
    int64_t table_end = hash_table_offsets[t + 1];
    int64_t capacity = table_end - table_start;

    if (capacity == 0) {
      // No pruning applied on the indices associated with this table.
      for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
        dense_indices[indices_start + l] = indices[indices_start + l];
      }
      return;
    }

    uint32_t subwarp_id = threadIdx.x / 4;
    uint32_t subwarp_tid = threadIdx.x % 4;
    uint32_t subwarp_mask = static_cast<uint32_t>(0xF) << (4 * subwarp_id);
    for (int32_t l_start = 0; l_start + subwarp_id < L; l_start += kWarpSize / 4) {
        int32_t idx = indices[indices_start + l_start + subwarp_id];
        uint32_t slot_start = pruned_hash_function(static_cast<uint32_t>(idx)) % capacity;
        while (true) {
            uint32_t slot = (slot_start + subwarp_tid) % capacity;
            int2 val = *reinterpret_cast<const int2*>(&hash_table[table_start + static_cast<int64_t>(slot)][0]);
            int32_t slot_sparse_idx = val.x;
            int32_t slot_dense_idx = val.y;

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
__global__ void int_nbit_split_embedding_codegen_forward_pruned_array_lookup_kernel(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> index_remappings,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> index_remappings_offsets,
    int32_t B,
    int32_t T,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> dense_indices) {
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t t = b_t / B;
  int32_t b = b_t % B;
  if (b_t >= B * T) {
      return;
  }
  int32_t indices_start = offsets[t * B + b];
  int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  int64_t index_remappings_start = index_remappings_offsets[t];
  int64_t index_remappings_end = index_remappings_offsets[t + 1];
  int64_t capacity = index_remappings_end - index_remappings_start;

  for (int32_t l = threadIdx.x; l < L; l += blockDim.x) {
    int32_t idx = indices[indices_start + l];
    dense_indices[indices_start + l] = capacity ? index_remappings[index_remappings_start + idx] : idx;
  }
}
{% endif %}

}

at::Tensor int_nbit_split_embedding_codegen_forward_{{ wdesc }}_cuda(
    at::Tensor dev_weights,
    at::Tensor uvm_weights,
    at::Tensor weights_placements,
    at::Tensor weights_offsets,
    at::Tensor weights_tys,
    at::Tensor D_offsets,
    int64_t total_D,
    int64_t max_int2_D,
    int64_t max_int4_D,
    int64_t max_int8_D,
    int64_t max_float16_D,
    int64_t max_float32_D,
    at::Tensor indices,
    at::Tensor offsets,
    int64_t pooling_mode,
    {% if weighted %}
    at::Tensor indice_weights,
    {% endif %}
    int64_t output_dtype,
    at::Tensor lxu_cache_weights,
    at::Tensor lxu_cache_locations,
    int64_t unused
) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    int32_t T = D_offsets.numel() - 1;
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);

    TORCH_CHECK(total_D > 0);
    TORCH_CHECK(max_int2_D == 0);

    at::Tensor output;
    const int kINT8QparamsBytes = 8;
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 || o_dtype == SparseType::INT8);
    if (o_dtype == SparseType::FP32) {
        output = at::empty({B, total_D}, dev_weights.options().dtype(at::kFloat));
    } else if (o_dtype == SparseType::FP16) {
        output = at::empty({B, total_D}, dev_weights.options().dtype(at::kHalf));
    } else if (o_dtype == SparseType::INT8) {
        output = at::empty({B, total_D + T * kINT8QparamsBytes}, dev_weights.options().dtype(at::kByte));
    }

    if (B == 0) {
      return output;
    }

    using index_t = int32_t;

    // launch 4-bit kernel
    constexpr int32_t kWarpsPerBlock = 4;

    #define X(OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::int_4bit_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        pooling_mode, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.type(), "int4_split_embedding_codegen_forward_kernel", ([&] {
      if (max_int4_D > 0) {
        auto max_int4_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int4_D, SparseType::INT4), 128);
        TORCH_CHECK(max_int4_128b_rows <= 4);
        if (max_int4_128b_rows > 0) {
          X(2, 8, 0, 1);
        }
        if (max_int4_128b_rows > 1) {
          X(2, 4, 1, 2);
        }
        if (max_int4_128b_rows > 2) {
          X(1, 4, 2, 4);
        }
      }
    }));
    #undef X


    #define X(OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::int_8bit_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        pooling_mode, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.type(), "int8_split_embedding_codegen_forward_kernel", ([&] {
      if (max_int8_D > 0) {
        auto max_int8_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int8_D, SparseType::INT8), 128);
        TORCH_CHECK(max_int8_128b_rows <= 8);
        if (max_int8_128b_rows > 0) {
          X(2, 8, 0, 1);
        }
        if (max_int8_128b_rows > 1) {
          X(2, 4, 1, 2);
        }
        if (max_int8_128b_rows > 2) {
          X(2, 4, 2, 4);
        }
        if (max_int8_128b_rows > 4) {
          X(2, 4, 4, 8);
        }
      }
    }));
    #undef X

    #define X(OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::fp16_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        pooling_mode, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.type(), "fp16_split_embedding_codegen_forward_kernel", ([&] {
      if (max_float16_D > 0) {
        auto max_fp16_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float16_D, SparseType::FP16), 128);
        TORCH_CHECK(max_fp16_128b_rows <= 16);
        if (max_fp16_128b_rows > 0) {
          X(2, 8, 0, 2);
        }
        if (max_fp16_128b_rows > 2) {
          X(2, 8, 2, 4);
        }
        if (max_fp16_128b_rows > 4) {
          X(2, 4, 4, 8);
        }
        if (max_fp16_128b_rows > 8) {
          X(2, 2, 8, 16);
        }
      }
    }));
    #undef X

    #define X(OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::fp32_split_embedding_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        dev_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        uvm_weights.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(), \
        weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(), \
        weights_tys.packed_accessor32<uint8_t, 1, at::RestrictPtrTraits>(), \
        D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(), \
        indices.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        offsets.packed_accessor32<index_t, 1, at::RestrictPtrTraits>(), \
        pooling_mode, \
        {% if weighted %} indice_weights.packed_accessor32<float, 1, at::RestrictPtrTraits>(), {% endif %} \
        output.packed_accessor32<output_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_weights.packed_accessor64<uint8_t, 2, at::RestrictPtrTraits>(), \
        lxu_cache_locations.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>() \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.type(), "fp32_split_embedding_codegen_forward_kernel", ([&] {
      if (max_float32_D > 0) {
        auto max_fp32_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float32_D, SparseType::FP32), 128);
        TORCH_CHECK(max_fp32_128b_rows <= 32);
        // FP32 is used for numerical validations and tiny embeddings tables.
        // We haven't carefully tuned the perf of FP32 embeddings.
        X(1, 1, 0, 32);
      }
    }));
    #undef X

    // TODO: 2-bit kernels.
    return output;
}

at::Tensor pruned_hashmap_lookup_{{ wdesc }}_cuda(
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor hash_table,
    at::Tensor hash_table_offsets) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(indices.get_device());
    auto dense_indices = at::empty_like(indices);
    int32_t T = hash_table_offsets.size(0) - 1;
    int32_t B = (offsets.size(0) - 1) / T;
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
at::Tensor pruned_array_lookup_cuda(
    at::Tensor indices,
    at::Tensor offsets,
    at::Tensor index_remappings,
    at::Tensor index_remappings_offsets) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());
  auto dense_indices = at::empty_like(indices);
  int32_t T = index_remappings_offsets.size(0) - 1;
  TORCH_CHECK(
      (offsets.size(0) - 1) % T == 0,
      "offsets.size() - 1 is not divisible by T! offsets.size: ",
      offsets.size(0),
      "T: ",
      T
  );
  int32_t B = (offsets.size(0) - 1) / T;
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
{% endif %}
// clang-format on
