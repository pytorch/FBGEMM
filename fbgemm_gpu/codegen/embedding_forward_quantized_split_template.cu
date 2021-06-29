/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
{% set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"

enum {
  DEVICE = 0,
  MANAGED = 1,
  MANAGED_CACHING = 2,
};

constexpr size_t kForwardMaxThreads = 256;


struct __align__(8) half4 {
  __host__ __device__ half4() {}
  __host__ __device__ half4(const half4 &other) : v(other.v) {}
  __host__ __device__ half4 &operator=(const half4 &other) {
    v = other.v;
    return *this;
  }
  union {
    half2 vals[2];
    int2 v;
  };
};

__device__ __forceinline__ half2 make_half2(float x, float y) {
  half2 t;

  t.x = __float2half_rn(x);
  t.y = __float2half_rn(y);

  return t;
}

__forceinline__ __device__ half2 make_zero_half2() {
  return make_half2(0.0, 0.0);
}

__forceinline__ __device__ half4 make_zero_half4() {
  half4 result;
  result.vals[0] = make_zero_half2();
  result.vals[1] = make_zero_half2();
  return result;
}

struct __align__(16) half8 {
  __host__ __device__ half8() {}
  __host__ __device__ half8(const half8 &other) : v(other.v) {}
  __host__ __device__ half8 &operator=(const half8 &other) {
    v = other.v;
    return *this;
  }
  union {
    half4 vals[2];
    int4 v;
  };
};

__forceinline__ __device__ half8 make_zero_half8() {
  half8 result;
  result.vals[0] = make_zero_half4();
  result.vals[1] = make_zero_half4();
  return result;
}

struct __align__(32) float8 {
  __host__ __device__ float8() {}
  float4 vals[2];
};

__device__ __forceinline__ float8 make_zero_float8() {
  float8 t;
  t.vals[0] = make_float4(0, 0, 0, 0);
  t.vals[1] = make_float4(0, 0, 0, 0);
  return t;
}

__device__ __forceinline__ half8 to_half8(float8 v) {
  half8 t;
  t.vals[0].vals[0] = __float22half2_rn(make_float2(v.vals[0].x, v.vals[0].y));
  t.vals[0].vals[1] = __float22half2_rn(make_float2(v.vals[0].z, v.vals[0].w));
  t.vals[1].vals[0] = __float22half2_rn(make_float2(v.vals[1].x, v.vals[1].y));
  t.vals[1].vals[1] = __float22half2_rn(make_float2(v.vals[1].z, v.vals[1].w));
  return t;
}

__device__ __forceinline__ __half hbfe(uint32_t val, uint32_t pos, uint32_t len) {
  uint32_t ret;
  // Get the bit field of [pos, pos+len) bits from val:
  // (val >> pos) && ( (1u << len) - 1u )
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
  return __uint2half_rn(ret);
}

__forceinline__ __device__ half2 hfma2(const half2 a, const half2 b, const half2 c) {
  // TODO: We might need to use FMA with FP16 input and FP32 output.
#if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
  return __hfma2(a, b, c);
#else
  float2 fa, fb, fc;
  fa = __half22float2(a);
  fb = __half22float2(b);
  fc = __half22float2(c);
  fc.x = fa.x * fb.x + fc.x;
  fc.y = fa.y * fb.y + fc.y;
  return __float22half2_rn(fc);
#endif
}

__forceinline__ __device__ half8  dequantize_int4(uint32_t packedVals, __half2 shift_scale) {
  half8 res;
  res.vals[0].vals[0].x =
      hbfe(packedVals, 0,
           4); // __short2half_rn((uint8_t)((packedVals >> 0) & 0x0F));
  res.vals[0].vals[0].y =
      hbfe(packedVals, 4,
           4); // __short2half_rn((uint8_t)((packedVals >> 4) & 0x0F));
  res.vals[0].vals[1].x =
      hbfe(packedVals, 8,
           4); // __short2half_rn((uint8_t)((packedVals >> 8) & 0x0F));
  res.vals[0].vals[1].y =
      hbfe(packedVals, 12,
           4); // __short2half_rn((uint8_t)((packedVals >> 12) & 0x0F));
  res.vals[1].vals[0].x =
      hbfe(packedVals, 16,
           4); // __short2half_rn((uint8_t)((packedVals >> 16) & 0x0F));
  res.vals[1].vals[0].y =
      hbfe(packedVals, 20,
           4); // __short2half_rn((uint8_t)((packedVals >> 20) & 0x0F));
  res.vals[1].vals[1].x =
      hbfe(packedVals, 24,
           4); // __short2half_rn((uint8_t)((packedVals >> 24) & 0x0F));
  res.vals[1].vals[1].y =
      hbfe(packedVals, 28,
           4); // __short2half_rn((uint8_t)((packedVals >> 28) & 0x0F));

  res.vals[0].vals[0] =
      hfma2(res.vals[0].vals[0], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  res.vals[0].vals[1] =
      hfma2(res.vals[0].vals[1], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  res.vals[1].vals[0] =
      hfma2(res.vals[1].vals[0], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  res.vals[1].vals[1] =
      hfma2(res.vals[1].vals[1], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  return res;
}

__forceinline__ __device__ half8  dequantize_int2(uint16_t packedVals_, __half2 shift_scale) {
  uint32_t packedVals = static_cast<uint32_t>(packedVals_);
  half8 res;
  res.vals[0].vals[0].x =
      hbfe(packedVals, 0,
           2);
  res.vals[0].vals[0].y =
      hbfe(packedVals, 2,
           2);
  res.vals[0].vals[1].x =
      hbfe(packedVals, 4,
           2);
  res.vals[0].vals[1].y =
      hbfe(packedVals, 6,
           2);
  res.vals[1].vals[0].x =
      hbfe(packedVals, 8,
           2);
  res.vals[1].vals[0].y =
      hbfe(packedVals, 10,
           2);
  res.vals[1].vals[1].x =
      hbfe(packedVals, 12,
           2);
  res.vals[1].vals[1].y =
      hbfe(packedVals, 14,
           2);

  res.vals[0].vals[0] =
      hfma2(res.vals[0].vals[0], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  res.vals[0].vals[1] =
      hfma2(res.vals[0].vals[1], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  res.vals[1].vals[0] =
      hfma2(res.vals[1].vals[0], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  res.vals[1].vals[1] =
      hfma2(res.vals[1].vals[1], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  return res;
}

__forceinline__ __device__ half8  dequantize_int8(uint64_t packedVals, __half2 shift_scale) {
  half8 res;
  uint32_t packedVals_l = static_cast<uint32_t>(packedVals);
  uint32_t packedVals_h = static_cast<uint32_t>(packedVals >> 32);

  res.vals[0].vals[0].x =
      hbfe(packedVals_l, 0,
           8);
  res.vals[0].vals[0].y =
      hbfe(packedVals_l, 8,
           8);
  res.vals[0].vals[1].x =
      hbfe(packedVals_l, 16,
           8);
  res.vals[0].vals[1].y =
      hbfe(packedVals_l, 24,
           8);
  res.vals[1].vals[0].x =
      hbfe(packedVals_h, 0,
           8);
  res.vals[1].vals[0].y =
      hbfe(packedVals_h, 8,
           8);
  res.vals[1].vals[1].x =
      hbfe(packedVals_h, 16,
           8);
  res.vals[1].vals[1].y =
      hbfe(packedVals_h, 24,
           8);

  res.vals[0].vals[0] =
      hfma2(res.vals[0].vals[0], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  res.vals[0].vals[1] =
      hfma2(res.vals[0].vals[1], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  res.vals[1].vals[0] =
      hfma2(res.vals[1].vals[0], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  res.vals[1].vals[1] =
      hfma2(res.vals[1].vals[1], __half2(shift_scale.x, shift_scale.x),
              __half2(shift_scale.y, shift_scale.y));
  return res;
}

__forceinline__ __device__ float8 accumulate_packed_int4(float8 acc,
                                                         uint32_t packedVals,
                                                         __half2 shift_scale) {
  half8 res = dequantize_int4(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0].vals[0]);
  float2 v1 = __half22float2(res.vals[0].vals[1]);
  float2 v2 = __half22float2(res.vals[1].vals[0]);
  float2 v3 = __half22float2(res.vals[1].vals[1]);
  acc.vals[0].x += v0.x;
  acc.vals[0].y += v0.y;
  acc.vals[0].z += v1.x;
  acc.vals[0].w += v1.y;
  acc.vals[1].x += v2.x;
  acc.vals[1].y += v2.y;
  acc.vals[1].z += v3.x;
  acc.vals[1].w += v3.y;
  return acc;
}

__forceinline__ __device__ float8 accumulate_packed_int2(float8 acc,
                                                         uint16_t packedVals,
                                                         __half2 shift_scale) {
  half8 res = dequantize_int2(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0].vals[0]);
  float2 v1 = __half22float2(res.vals[0].vals[1]);
  float2 v2 = __half22float2(res.vals[1].vals[0]);
  float2 v3 = __half22float2(res.vals[1].vals[1]);
  acc.vals[0].x += v0.x;
  acc.vals[0].y += v0.y;
  acc.vals[0].z += v1.x;
  acc.vals[0].w += v1.y;
  acc.vals[1].x += v2.x;
  acc.vals[1].y += v2.y;
  acc.vals[1].z += v3.x;
  acc.vals[1].w += v3.y;
  return acc;
}

__forceinline__ __device__ float8 accumulate_packed_int8(float8 acc,
                                                         uint64_t packedVals,
                                                         __half2 shift_scale) {
  half8 res = dequantize_int8(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0].vals[0]);
  float2 v1 = __half22float2(res.vals[0].vals[1]);
  float2 v2 = __half22float2(res.vals[1].vals[0]);
  float2 v3 = __half22float2(res.vals[1].vals[1]);
  acc.vals[0].x += v0.x;
  acc.vals[0].y += v0.y;
  acc.vals[0].z += v1.x;
  acc.vals[0].w += v1.y;
  acc.vals[1].x += v2.x;
  acc.vals[1].y += v2.y;
  acc.vals[1].z += v3.x;
  acc.vals[1].w += v3.y;
  return acc;
}

__forceinline__ __device__ float8 weighted_accumulate_packed_int4(float8 acc,
                                                        uint32_t packedVals,
                                                        __half2 shift_scale,
                                                        float weight) {
  half8 res = dequantize_int4(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0].vals[0]);
  float2 v1 = __half22float2(res.vals[0].vals[1]);
  float2 v2 = __half22float2(res.vals[1].vals[0]);
  float2 v3 = __half22float2(res.vals[1].vals[1]);

  acc.vals[0].x = fmaf(v0.x, weight, acc.vals[0].x);
  acc.vals[0].y = fmaf(v0.y, weight, acc.vals[0].y);
  acc.vals[0].z = fmaf(v1.x, weight, acc.vals[0].z);
  acc.vals[0].w = fmaf(v1.y, weight, acc.vals[0].w);

  acc.vals[1].x = fmaf(v2.x, weight, acc.vals[1].x);
  acc.vals[1].y = fmaf(v2.y, weight, acc.vals[1].y);
  acc.vals[1].z = fmaf(v3.x, weight, acc.vals[1].z);
  acc.vals[1].w = fmaf(v3.y, weight, acc.vals[1].w);

  return acc;
}

__forceinline__ __device__ float8 weighted_accumulate_packed_int2(float8 acc,
                                                        uint16_t packedVals,
                                                        __half2 shift_scale,
                                                        float weight) {
  half8 res = dequantize_int2(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0].vals[0]);
  float2 v1 = __half22float2(res.vals[0].vals[1]);
  float2 v2 = __half22float2(res.vals[1].vals[0]);
  float2 v3 = __half22float2(res.vals[1].vals[1]);

  acc.vals[0].x = fmaf(v0.x, weight, acc.vals[0].x);
  acc.vals[0].y = fmaf(v0.y, weight, acc.vals[0].y);
  acc.vals[0].z = fmaf(v1.x, weight, acc.vals[0].z);
  acc.vals[0].w = fmaf(v1.y, weight, acc.vals[0].w);

  acc.vals[1].x = fmaf(v2.x, weight, acc.vals[1].x);
  acc.vals[1].y = fmaf(v2.y, weight, acc.vals[1].y);
  acc.vals[1].z = fmaf(v3.x, weight, acc.vals[1].z);
  acc.vals[1].w = fmaf(v3.y, weight, acc.vals[1].w);

  return acc;
}

__forceinline__ __device__ float8 weighted_accumulate_packed_int8(float8 acc,
                                                        uint64_t packedVals,
                                                        __half2 shift_scale,
                                                        float weight) {
  half8 res = dequantize_int8(packedVals, shift_scale);
  // Accumulate in float32.
  float2 v0 = __half22float2(res.vals[0].vals[0]);
  float2 v1 = __half22float2(res.vals[0].vals[1]);
  float2 v2 = __half22float2(res.vals[1].vals[0]);
  float2 v3 = __half22float2(res.vals[1].vals[1]);

  acc.vals[0].x = fmaf(v0.x, weight, acc.vals[0].x);
  acc.vals[0].y = fmaf(v0.y, weight, acc.vals[0].y);
  acc.vals[0].z = fmaf(v1.x, weight, acc.vals[0].z);
  acc.vals[0].w = fmaf(v1.y, weight, acc.vals[0].w);

  acc.vals[1].x = fmaf(v2.x, weight, acc.vals[1].x);
  acc.vals[1].y = fmaf(v2.y, weight, acc.vals[1].y);
  acc.vals[1].z = fmaf(v3.x, weight, acc.vals[1].z);
  acc.vals[1].w = fmaf(v3.y, weight, acc.vals[1].w);

  return acc;
}

using namespace at;
using namespace fbgemm_gpu;

// Keep in sync with split_embedding_configs.py:SparseType
enum class SparseType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT4 = 3,
    INT2 = 4,
};

__device__ inline int32_t row_size_in_bytes(int32_t dim, SparseType weight_ty) {
    if (weight_ty == SparseType::INT8) { return dim + 8; }
    if (weight_ty == SparseType::INT4) { return dim / 2 + 4; }
    if (weight_ty == SparseType::INT2) { return dim / 4 + 4; }
    return 0;
}

// "Effective" number of elements in the row when we include the row-wise quantization parameters.
__device__ inline int32_t padded_D(int32_t dim, SparseType weight_ty) {
    if (weight_ty == SparseType::INT8) { return dim + 8; }
    if (weight_ty == SparseType::INT4) { return dim + 8; }
    if (weight_ty == SparseType::INT2) { return dim + 16; }
    return 0;
}


template<typename index_t, size_t kMaxVecsPerThread>
__launch_bounds__(kForwardMaxThreads)
__global__ void float16_split_embedding_codegen_forward_{{ wdesc }}_kernel(
    const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
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
    PackedTensorAccessor32<Half, 2, RestrictPtrTraits>
        output // [B][total_D],
    ) {
    int32_t B = output.size(0);
    int32_t T = D_offsets.size(0) - 1;
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= B * T) {
        return;
    }
    int32_t t = b_t / B;
    int32_t b = b_t % B;
    SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);

    if (weight_ty != SparseType::FP16) {
        return;
    }
    int64_t weights_offset = weights_offsets[t];
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;

    int64_t indices_start = offsets[t * B + b];
    int64_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    const auto* __restrict__ weights = reinterpret_cast<const Half*>(&dev_weights[weights_offset]);


    Vec4T<Half> accumulators[kMaxVecsPerThread];

    for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {% if weighted %}
        acc_type<Half, true> idx_weight = l < L ? indice_weights[indices_start + l] : 0;
        {% endif %}

        for (auto j = 0; j < kWarpSize && l_start + j < L; ++j) {
            int64_t idx_j = __shfl_sync(0xFFFFFFFF, idx, j);

            {% if weighted %}
            acc_type<Half, true> idx_weight_j = __shfl_sync(0xFFFFFFFF, idx_weight, j);
            {% endif %}

            // Handle pruned-out indices.
            if (idx_j != -1) {
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                    ++i) {
                    int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
                    Vec4T<Half> weight(&weights[idx_j * D + d]);
                    {% if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {% else %}
                    accumulators[i].acc.x += weight.acc.x;
                    accumulators[i].acc.y += weight.acc.y;
                    accumulators[i].acc.z += weight.acc.z;
                    accumulators[i].acc.w += weight.acc.w;
                    {% endif %}
                }
            }
        }
    }
#pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
        if (pooling_mode == MEAN && L != 0) {
            accumulators[i].acc.x /= L;
            accumulators[i].acc.y /= L;
            accumulators[i].acc.z /= L;
            accumulators[i].acc.w /= L;
        }
        accumulators[i].store(&output[b][D_start + d]);
    }
}

template<typename index_t, size_t kMaxVecsPerThread, size_t kThreadsPerRow>
__launch_bounds__(kForwardMaxThreads)
__global__ void int_nbit_split_embedding_codegen_forward_{{ wdesc }}_kernel(
    const PackedTensorAccessor64<uint8_t, 1, RestrictPtrTraits> dev_weights,
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
    PackedTensorAccessor32<Half, 2, RestrictPtrTraits>
        output // [B][total_D],
    ) {
    int32_t B = output.size(0);
    int32_t T = D_offsets.size(0) - 1;
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= B * T) {
        return;
    }
    int32_t t = b_t / B;
    int32_t b = b_t % B;

    SparseType weight_ty = static_cast<SparseType>(weights_tys[t]);
    if (weight_ty == SparseType::FP16) {
        return;
    }

    constexpr int32_t kRowsPerWarp = 32 / kThreadsPerRow;
    const int32_t row_in_warp = threadIdx.y % kRowsPerWarp;
    uint32_t subwarp_mask;
    if (kThreadsPerRow == 8) {
        subwarp_mask = uint32_t(0xFF) << row_in_warp;
    }
    if (kThreadsPerRow == 16) {
        subwarp_mask = uint32_t(0xFFFF) << row_in_warp;
    }
    if (kThreadsPerRow == 32) {
        subwarp_mask = uint32_t(0xFFFFFFFF) << row_in_warp;
    }
    int64_t weights_offset = weights_offsets[t];
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;

    const int32_t D_total = padded_D(D, weight_ty);
    const int32_t D_padding = D_total - D;
    const int32_t D_bytes = row_size_in_bytes(D, weight_ty);

    int64_t indices_start = offsets[t * B + b];
    int64_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    const uint8_t* __restrict__ weights = &dev_weights[weights_offset];

    float8 accumulators[kMaxVecsPerThread];
    for (auto i = 0; i < kMaxVecsPerThread; ++i) {
        accumulators[i] = make_zero_float8();
    }

    for (int32_t l_start = 0; l_start < L; l_start += kThreadsPerRow) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        int32_t JLim = L - l_start < kThreadsPerRow ? L - l_start : kThreadsPerRow;

        // negative indices represent "pruned out" values.
        const uint8_t* __restrict__ row = idx >= 0 ? &weights[idx * D_bytes] : nullptr;

        {% if weighted %}
        float idx_weight = l < L ? indice_weights[indices_start + l] : 0.0;
        {% endif %}

        int32_t j = 0;
        constexpr size_t kUnroll = 4;
        int32_t jUnroll = (JLim / kUnroll) * kUnroll;
        for (; j < jUnroll; j += kUnroll) {
            const uint32_t *row_j0 = reinterpret_cast<const uint32_t *>(__shfl_sync(subwarp_mask, intptr_t(row), j + 0 + row_in_warp * kThreadsPerRow));
            const uint32_t *row_j1 = reinterpret_cast<const uint32_t *>(__shfl_sync(subwarp_mask, intptr_t(row), j + 1 + row_in_warp * kThreadsPerRow));
            const uint32_t *row_j2 = reinterpret_cast<const uint32_t *>(__shfl_sync(subwarp_mask, intptr_t(row), j + 2 + row_in_warp * kThreadsPerRow));
            const uint32_t *row_j3 = reinterpret_cast<const uint32_t *>(__shfl_sync(subwarp_mask, intptr_t(row), j + 3 + row_in_warp * kThreadsPerRow));

            // scale and bias are at the beginning of each row.
            // rationale: have scale/shift at start since these get loaded first
            // and then broadcasted around so it might speed up the first cache
            // miss.
            half2 shift_scale_j0 = row ? (reinterpret_cast<const half2*>(row_j0))[0] : make_half2(0.0, 0.0);
            half2 shift_scale_j1 = row ? (reinterpret_cast<const half2*>(row_j1))[0] : make_half2(0.0, 0.0);
            half2 shift_scale_j2 = row ? (reinterpret_cast<const half2*>(row_j2))[0] : make_half2(0.0, 0.0);
            half2 shift_scale_j3 = row ? (reinterpret_cast<const half2*>(row_j3))[0] : make_half2(0.0, 0.0);

            {% if weighted %}
            float idx_weight_j0 = __shfl_sync(subwarp_mask, idx_weight, j + 0 + row_in_warp * kThreadsPerRow);
            float idx_weight_j1 = __shfl_sync(subwarp_mask, idx_weight, j + 1 + row_in_warp * kThreadsPerRow);
            float idx_weight_j2 = __shfl_sync(subwarp_mask, idx_weight, j + 2 + row_in_warp * kThreadsPerRow);
            float idx_weight_j3 = __shfl_sync(subwarp_mask, idx_weight, j + 3 + row_in_warp * kThreadsPerRow);

            {% endif %}

            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && 8 * kThreadsPerRow * i + threadIdx.x * 8 < D_total;
                ++i) {
                // Read the rowwise-quantized int values: note that first D_padding elements will be ditched later:
                // Reason: to avoid divergence the first thread in the warp computes garbage.
                if (weight_ty == SparseType::INT4) {
                    uint32_t v0 = row_j0 ? reinterpret_cast<const uint32_t*>(row_j0)[kThreadsPerRow * i + threadIdx.x] : 0;
                    uint32_t v1 = row_j1 ? reinterpret_cast<const uint32_t*>(row_j1)[kThreadsPerRow * i + threadIdx.x] : 0;
                    uint32_t v2 = row_j2 ? reinterpret_cast<const uint32_t*>(row_j2)[kThreadsPerRow * i + threadIdx.x] : 0;
                    uint32_t v3 = row_j3 ? reinterpret_cast<const uint32_t*>(row_j3)[kThreadsPerRow * i + threadIdx.x] : 0;

                    {% if weighted %}
                    accumulators[i] = weighted_accumulate_packed_int4(
                        accumulators[i], v0,
                        shift_scale_j0, idx_weight_j0);
                    accumulators[i] = weighted_accumulate_packed_int4(
                        accumulators[i], v1,
                        shift_scale_j1, idx_weight_j1);
                    accumulators[i] = weighted_accumulate_packed_int4(
                        accumulators[i], v2,
                        shift_scale_j2, idx_weight_j2);
                    accumulators[i] = weighted_accumulate_packed_int4(
                        accumulators[i], v3,
                        shift_scale_j3, idx_weight_j3);

                    {% else %}
                    accumulators[i] = accumulate_packed_int4(
                        accumulators[i], v0,
                        shift_scale_j0);
                    accumulators[i] = accumulate_packed_int4(
                        accumulators[i], v1,
                        shift_scale_j1);
                    accumulators[i] = accumulate_packed_int4(
                        accumulators[i], v2,
                        shift_scale_j2);
                    accumulators[i] = accumulate_packed_int4(
                        accumulators[i], v3,
                        shift_scale_j3);
                    {% endif %}
                } else if (weight_ty == SparseType::INT2) {
                    uint16_t v0 = row_j0 ? reinterpret_cast<const uint16_t*>(row_j0)[kThreadsPerRow * i + threadIdx.x] : 0;
                    uint16_t v1 = row_j1 ? reinterpret_cast<const uint16_t*>(row_j1)[kThreadsPerRow * i + threadIdx.x] : 0;
                    uint16_t v2 = row_j2 ? reinterpret_cast<const uint16_t*>(row_j2)[kThreadsPerRow * i + threadIdx.x] : 0;
                    uint16_t v3 = row_j3 ? reinterpret_cast<const uint16_t*>(row_j3)[kThreadsPerRow * i + threadIdx.x] : 0;

                    {% if weighted %}
                    accumulators[i] = weighted_accumulate_packed_int2(
                        accumulators[i], v0,
                        shift_scale_j0, idx_weight_j0);
                    accumulators[i] = weighted_accumulate_packed_int2(
                        accumulators[i], v1,
                        shift_scale_j1, idx_weight_j1);
                    accumulators[i] = weighted_accumulate_packed_int2(
                        accumulators[i], v2,
                        shift_scale_j2, idx_weight_j2);
                    accumulators[i] = weighted_accumulate_packed_int2(
                        accumulators[i], v3,
                        shift_scale_j3, idx_weight_j3);

                    {% else %}
                    accumulators[i] = accumulate_packed_int2(
                        accumulators[i], v0,
                        shift_scale_j0);
                    accumulators[i] = accumulate_packed_int2(
                        accumulators[i], v1,
                        shift_scale_j1);
                    accumulators[i] = accumulate_packed_int2(
                        accumulators[i], v2,
                        shift_scale_j2);
                    accumulators[i] = accumulate_packed_int2(
                        accumulators[i], v3,
                        shift_scale_j3);
                    {% endif %}
                } else if (weight_ty == SparseType::INT8) {
                    uint64_t v0 = row_j0 ? reinterpret_cast<const uint64_t*>(row_j0)[kThreadsPerRow * i + threadIdx.x] : 0;
                    uint64_t v1 = row_j1 ? reinterpret_cast<const uint64_t*>(row_j1)[kThreadsPerRow * i + threadIdx.x] : 0;
                    uint64_t v2 = row_j2 ? reinterpret_cast<const uint64_t*>(row_j2)[kThreadsPerRow * i + threadIdx.x] : 0;
                    uint64_t v3 = row_j3 ? reinterpret_cast<const uint64_t*>(row_j3)[kThreadsPerRow * i + threadIdx.x] : 0;

                    {% if weighted %}
                    accumulators[i] = weighted_accumulate_packed_int8(
                        accumulators[i], v0,
                        shift_scale_j0, idx_weight_j0);
                    accumulators[i] = weighted_accumulate_packed_int8(
                        accumulators[i], v1,
                        shift_scale_j1, idx_weight_j1);
                    accumulators[i] = weighted_accumulate_packed_int8(
                        accumulators[i], v2,
                        shift_scale_j2, idx_weight_j2);
                    accumulators[i] = weighted_accumulate_packed_int8(
                        accumulators[i], v3,
                        shift_scale_j3, idx_weight_j3);

                    {% else %}
                    accumulators[i] = accumulate_packed_int8(
                        accumulators[i], v0,
                        shift_scale_j0);
                    accumulators[i] = accumulate_packed_int8(
                        accumulators[i], v1,
                        shift_scale_j1);
                    accumulators[i] = accumulate_packed_int8(
                        accumulators[i], v2,
                        shift_scale_j2);
                    accumulators[i] = accumulate_packed_int8(
                        accumulators[i], v3,
                        shift_scale_j3);
                    {% endif %}
                }
            }
        }
        for (; j < JLim; ++j) {
            const uint32_t *row_j0 = reinterpret_cast<const uint32_t *>(__shfl_sync(subwarp_mask, intptr_t(row), j + 0 + row_in_warp * kThreadsPerRow));
            half2 shift_scale_j0 = row ? (reinterpret_cast<const half2*>(row_j0))[0] : make_half2(0.0, 0.0);

            {% if weighted %}
            float idx_weight_j = __shfl_sync(subwarp_mask, idx_weight, j + row_in_warp * kThreadsPerRow);
            {% endif %}
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && 8 * kThreadsPerRow * i + threadIdx.x * 8 < D_total;
                ++i) {
                if (weight_ty == SparseType::INT4) {
                    uint32_t v0 = row_j0 ? reinterpret_cast<const uint32_t*>(row_j0)[kThreadsPerRow * i + threadIdx.x] : 0;

                    {% if weighted %}
                    accumulators[i] = weighted_accumulate_packed_int4(
                        accumulators[i], v0,
                        shift_scale_j0, idx_weight_j);
                    {% else %}
                    accumulators[i] = accumulate_packed_int4(
                        accumulators[i], v0,
                        shift_scale_j0);
                    {% endif %}
                } else if (weight_ty == SparseType::INT2) {
                    uint16_t v0 = row_j0 ? reinterpret_cast<const uint16_t*>(row_j0)[kThreadsPerRow * i + threadIdx.x] : 0;

                    {% if weighted %}
                    accumulators[i] = weighted_accumulate_packed_int2(
                        accumulators[i], v0,
                        shift_scale_j0, idx_weight_j);
                    {% else %}
                    accumulators[i] = accumulate_packed_int2(
                        accumulators[i], v0,
                        shift_scale_j0);
                    {% endif %}
                } else if (weight_ty == SparseType::INT8) {
                    uint64_t v0 = row_j0 ? reinterpret_cast<const uint64_t*>(row_j0)[kThreadsPerRow * i + threadIdx.x] : 0;

                    {% if weighted %}
                    accumulators[i] = weighted_accumulate_packed_int8(
                        accumulators[i], v0,
                        shift_scale_j0, idx_weight_j);
                    {% else %}
                    accumulators[i] = accumulate_packed_int8(
                        accumulators[i], v0,
                        shift_scale_j0);
                    {% endif %}
                }

            }
        }
    }

#pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 8 * kThreadsPerRow * i + threadIdx.x * 8 < D_total;
        ++i) {
        // We shift back by a fixed number of elements to remove the first group of elements (which is
        // garbage due to the scale/shift handling)
        int32_t d = 8 * kThreadsPerRow * i + threadIdx.x * 8 - D_padding;
        if (pooling_mode == MEAN && L != 0) {
            float inv_L = 1.0 / L;
            accumulators[i].vals[0].x *= inv_L;
            accumulators[i].vals[0].y *= inv_L;
            accumulators[i].vals[0].z *= inv_L;
            accumulators[i].vals[0].w *= inv_L;
            accumulators[i].vals[1].x *= inv_L;
            accumulators[i].vals[1].y *= inv_L;
            accumulators[i].vals[1].z *= inv_L;
            accumulators[i].vals[1].w *= inv_L;
        }
        if (d >= 0 && d < D) {
            *(half8 *)(&output[b][D_start + d]) = to_half8(accumulators[i]);
        }
    }
}

Tensor int_nbit_split_embedding_codegen_forward_{{ wdesc }}_cuda(
    Tensor dev_weights,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_effective_D,
    int64_t max_float16_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    int64_t unused
) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    int32_t T = D_offsets.numel() - 1;
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    TORCH_CHECK(total_D > 0);
    TORCH_CHECK(total_D % 4 == 0);
    TORCH_CHECK(max_effective_D <= {{ max_embedding_dim }});
    auto output = empty({B, total_D}, dev_weights.options().dtype(at::kHalf));

    int32_t kThreads = 128;
    using index_t = int32_t;

    if (max_float16_D) {
        [&](){
            {% for kMaxVecsPerThread in range(1, 9) %}
            if (max_float16_D <= {{ 128 * kMaxVecsPerThread }}) {
                float16_split_embedding_codegen_forward_{{ wdesc }}_kernel<index_t, {{ kMaxVecsPerThread }} ><<<
                    div_round_up((B * T), kForwardMaxThreads / kWarpSize),
                    dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                        dev_weights.packed_accessor64<uint8_t, 1, RestrictPtrTraits>(),
                        weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                        weights_tys.packed_accessor32<uint8_t, 1, RestrictPtrTraits>(),
                        D_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                        indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                        offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                        pooling_mode,
                        {% if weighted %}
                        indice_weights.packed_accessor32<float, 1, RestrictPtrTraits>(),
                        {% endif %}
                        output.packed_accessor32<Half, 2, RestrictPtrTraits>()
                    );
                    return;
            }
            {% endfor %}
            TORCH_CHECK(false, "Unhandled max_float16_D:", max_float16_D);
        }();
    }

    // AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "int_nbit_split_embedding_codegen_forward_", [&] () {
    if (max_effective_D <= 64) {
        constexpr size_t kThreadsPerRow = 8;
        int_nbit_split_embedding_codegen_forward_{{ wdesc }}_kernel<index_t, 1, kThreadsPerRow><<<
            div_round_up((B * T), kThreads / kThreadsPerRow),
            dim3(kThreadsPerRow, kThreads / kThreadsPerRow),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<uint8_t, 1, RestrictPtrTraits>(),
            weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            pooling_mode,
            {% if weighted %}
            indice_weights.packed_accessor32<float, 1, RestrictPtrTraits>(),
            {% endif %}
            output.packed_accessor32<Half, 2, RestrictPtrTraits>()
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;

    }
    if (max_effective_D <= 128) {
        constexpr size_t kThreadsPerRow = 16;
        int_nbit_split_embedding_codegen_forward_{{ wdesc }}_kernel<index_t, 1, kThreadsPerRow><<<
            div_round_up((B * T), kThreads / kThreadsPerRow),
            dim3(kThreadsPerRow, kThreads / kThreadsPerRow),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<uint8_t, 1, RestrictPtrTraits>(),
            weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            pooling_mode,
            {% if weighted %}
            indice_weights.packed_accessor32<float, 1, RestrictPtrTraits>(),
            {% endif %}
            output.packed_accessor32<Half, 2, RestrictPtrTraits>()
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;

    }
    if (max_effective_D <= 256) {
        constexpr size_t kThreadsPerRow = 32;
        int_nbit_split_embedding_codegen_forward_{{ wdesc }}_kernel<index_t, 1, kThreadsPerRow><<<
            div_round_up((B * T), kThreads / kThreadsPerRow),
            dim3(kThreadsPerRow, kThreads / kThreadsPerRow),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<uint8_t, 1, RestrictPtrTraits>(),
            weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            pooling_mode,
            {% if weighted %}
            indice_weights.packed_accessor32<float, 1, RestrictPtrTraits>(),
            {% endif %}
            output.packed_accessor32<Half, 2, RestrictPtrTraits>()
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    if (max_effective_D <= 512) {
        constexpr size_t kThreadsPerRow = 32;
        int_nbit_split_embedding_codegen_forward_{{ wdesc }}_kernel<index_t, 2, kThreadsPerRow><<<
            div_round_up((B * T), kThreads / kThreadsPerRow),
            dim3(kThreadsPerRow, kThreads / kThreadsPerRow),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<uint8_t, 1, RestrictPtrTraits>(),
            weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            pooling_mode,
            {% if weighted %}
            indice_weights.packed_accessor32<float, 1, RestrictPtrTraits>(),
            {% endif %}
            output.packed_accessor32<Half, 2, RestrictPtrTraits>()
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    if (max_effective_D <= 1024) {
        constexpr size_t kThreadsPerRow = 32;
        int_nbit_split_embedding_codegen_forward_{{ wdesc }}_kernel<index_t, 2, kThreadsPerRow><<<
            div_round_up((B * T), kThreads / kThreadsPerRow),
            dim3(kThreadsPerRow, kThreads / kThreadsPerRow),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            dev_weights.packed_accessor64<uint8_t, 1, RestrictPtrTraits>(),
            weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
            weights_tys.packed_accessor32<uint8_t, 1, RestrictPtrTraits>(),
            D_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            pooling_mode,
            {% if weighted %}
            indice_weights.packed_accessor32<float, 1, RestrictPtrTraits>(),
            {% endif %}
            output.packed_accessor32<Half, 2, RestrictPtrTraits>()
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    TORCH_CHECK(false, "Unhandled max_effective_D:", max_effective_D);
    return output;
}

#define BIG_CONSTANT(x) (x##LLU)

__device__ inline uint32_t pruned_hash_function(int32_t key, int32_t table) {
    uint64_t k = (static_cast<uint64_t>(key) << 32) | static_cast<uint64_t>(table);
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return static_cast<uint32_t>(k >> 32);
}

__global__ void int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_{{ wdesc }}_kernel(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor64<int32_t, 2, RestrictPtrTraits> hash_table,
    int32_t B,
    int32_t T,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> dense_indices) {
    uint32_t capacity = hash_table.size(0);
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t t = b_t / B;
    int32_t b = b_t % B;
    if (b_t >= B * T) {
        return;
    }
    int32_t indices_start = offsets[t * B + b];
    int32_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    uint32_t subwarp_id = threadIdx.x / 4;
    uint32_t subwarp_tid = threadIdx.x % 4;
    uint32_t subwarp_mask = static_cast<uint32_t>(0xF) << (4 * subwarp_id);
    for (int32_t l_start = 0; l_start + subwarp_id < L; l_start += kWarpSize / 4) {
        int32_t idx = indices[indices_start + l_start + subwarp_id];
        uint32_t slot_start = static_cast<uint32_t>(pruned_hash_function(idx, t));
        while (true) {
            uint32_t slot = (slot_start + subwarp_tid) % capacity;
            int32_t sidx = hash_table[slot][0];
            int32_t stable = hash_table[slot][1];
            bool found = false;
            bool empty = false;
            if (sidx == -1) {
                empty = true;
            } else if (sidx == idx && stable == t) {
                found = true;
                dense_indices[indices_start + l_start + subwarp_id] = hash_table[slot][2];
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

Tensor pruned_hashmap_lookup_{{ wdesc }}_cuda(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    int64_t T) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(indices.get_device());
    auto dense_indices = empty_like(indices);
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    TORCH_CHECK(hash_table.size(0) < std::numeric_limits<int32_t>::max());
    int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_{{ wdesc }}_kernel<<<
        div_round_up(B * T + 1, kForwardMaxThreads / kWarpSize),
        dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
        0,
        at::cuda::getCurrentCUDAStream()>>>(
            indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
            hash_table.packed_accessor64<int32_t, 2, RestrictPtrTraits>(),
            B,
            T,
            dense_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dense_indices;
}
