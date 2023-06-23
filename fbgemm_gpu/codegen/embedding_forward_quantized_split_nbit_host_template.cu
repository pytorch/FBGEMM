/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{% set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"

using namespace fbgemm_gpu;
using Tensor = at::Tensor;

namespace nbit {

/*
  Looping over the weight types is required to generate all the C++ template
  declarations (not definitions) that will be invoked by the function
  `Tensor int_nbit_split_embedding*_codegen_forward_*_cuda(...)` later in the
  same generated source file.
*/
{% for emb_weight_type in ["FP32", "FP16", "FP8", "INT8", "INT4", "INT2"] %}
template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows, bool DeviceOnly>
__launch_bounds__(WarpsPerBlock * kWarpSize)
__global__ void {{ type_map[emb_weight_type].enum_name }}_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L(
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
  {% if type_map[emb_weight_type].enum_name == "FP8" %}
  const int exponent_bits,
  const int exponent_bias,
  {% endif %}
  pta::PackedTensorAccessor32<output_t, 2, at::RestrictPtrTraits> output, // [B][total_D],
  const pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations
  );
{% endfor %} // for emb_weight_type in ["FP32", "FP16", "FP8", "INT8", "INT4", "INT2"]

}

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


    ////////////////////////////////////////////////////////////////////////////
    // Launch 2-bit int kernel
    ////////////////////////////////////////////////////////////////////////////

#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name1 = "nbit::INT2_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L";
#endif

    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::INT2_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        MAKE_PTA_WITH_NAME(func_name1, dev_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name1, uvm_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name1, weights_placements, int32_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name1, weights_offsets, int64_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name1, weights_tys, uint8_t, 1, 32), \
        {% if not nobag %} \
        MAKE_PTA_WITH_NAME(func_name1, D_offsets, int32_t, 1, 32), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        MAKE_PTA_WITH_NAME(func_name1, indices, index_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name1, offsets, index_t, 1, 32), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} MAKE_PTA_WITH_NAME(func_name1, indice_weights, float, 1, 32), {% endif %} \
        MAKE_PTA_WITH_NAME(func_name1, output, output_t, 2, 32), \
        MAKE_PTA_WITH_NAME(func_name1, lxu_cache_weights, uint8_t, 2, 64), \
        MAKE_PTA_WITH_NAME(func_name1, lxu_cache_locations, int32_t, 1, 32) \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "int2_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_int2_D > 0) {
        auto max_int2_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int2_D, SparseType::INT2, row_alignment), 128);
        TORCH_CHECK(max_int2_128b_rows <= 4);
        if (max_int2_128b_rows > 0) {
          Y(2, 16, 0, 1);
        }
        if (max_int2_128b_rows > 1) {
          Y(2, 8, 1, 2);
        }
        if (max_int2_128b_rows > 2) {
          Y(2, 8, 2, 4);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch 4-bit int kernel
    ////////////////////////////////////////////////////////////////////////////

#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name2 = "nbit::INT4_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L";
#endif

    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::INT4_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        MAKE_PTA_WITH_NAME(func_name2, dev_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name2, uvm_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name2, weights_placements, int32_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name2, weights_offsets, int64_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name2, weights_tys, uint8_t, 1, 32), \
        {% if not nobag %} \
        MAKE_PTA_WITH_NAME(func_name2, D_offsets, int32_t, 1, 32), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        MAKE_PTA_WITH_NAME(func_name2, indices, index_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name2, offsets, index_t, 1, 32), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} MAKE_PTA_WITH_NAME(func_name2, indice_weights, float, 1, 32), {% endif %} \
        MAKE_PTA_WITH_NAME(func_name2, output, output_t, 2, 32), \
        MAKE_PTA_WITH_NAME(func_name2, lxu_cache_weights, uint8_t, 2, 64), \
        MAKE_PTA_WITH_NAME(func_name2, lxu_cache_locations, int32_t, 1, 32) \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "int4_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_int4_D > 0) {
        auto max_int4_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int4_D, SparseType::INT4, row_alignment), 128);
        TORCH_CHECK(max_int4_128b_rows <= 8);
        if (max_int4_128b_rows > 0) {
          Y(4, 8, 0, 1);
        }
        if (max_int4_128b_rows > 1) {
          Y(2, 8, 1, 2);
        }
        if (max_int4_128b_rows > 2) {
          Y(1, 4, 2, 4);
        }
        if (max_int4_128b_rows > 4) {
          Y(1, 4, 4, 8);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch 8-bit int kernel
    ////////////////////////////////////////////////////////////////////////////

#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name3 = "nbit::INT8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L";
#endif

    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::INT8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        MAKE_PTA_WITH_NAME(func_name3, dev_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name3, uvm_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name3, weights_placements, int32_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name3, weights_offsets, int64_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name3, weights_tys, uint8_t, 1, 32), \
        {% if not nobag %} \
        MAKE_PTA_WITH_NAME(func_name3, D_offsets, int32_t, 1, 32), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        MAKE_PTA_WITH_NAME(func_name3, indices, index_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name3, offsets, index_t, 1, 32), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} MAKE_PTA_WITH_NAME(func_name3, indice_weights, float, 1, 32), {% endif %} \
        MAKE_PTA_WITH_NAME(func_name3, output, output_t, 2, 32), \
        MAKE_PTA_WITH_NAME(func_name3, lxu_cache_weights, uint8_t, 2, 64), \
        MAKE_PTA_WITH_NAME(func_name3, lxu_cache_locations, int32_t, 1, 32) \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "int8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_int8_D > 0) {
        auto max_int8_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_int8_D, SparseType::INT8, row_alignment), 128);
        TORCH_CHECK(max_int8_128b_rows <= 16);
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
        if (max_int8_128b_rows > 8) {
          Y(2, 2, 8, 16);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch 8-bit float kernel
    ////////////////////////////////////////////////////////////////////////////

#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name4 = "nbit::FP8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L";
#endif

    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::FP8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        MAKE_PTA_WITH_NAME(func_name4, dev_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name4, uvm_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name4, weights_placements, int32_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name4, weights_offsets, int64_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name4, weights_tys, uint8_t, 1, 32), \
        {% if not nobag %} \
        MAKE_PTA_WITH_NAME(func_name4, D_offsets, int32_t, 1, 32), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        MAKE_PTA_WITH_NAME(func_name4, indices, index_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name4, offsets, index_t, 1, 32), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} MAKE_PTA_WITH_NAME(func_name4, indice_weights, float, 1, 32), {% endif %} \
        fp8_exponent_bits, \
        fp8_exponent_bias, \
        MAKE_PTA_WITH_NAME(func_name4, output, output_t, 2, 32), \
        MAKE_PTA_WITH_NAME(func_name4, lxu_cache_weights, uint8_t, 2, 64), \
        MAKE_PTA_WITH_NAME(func_name4, lxu_cache_locations, int32_t, 1, 32) \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "fp8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_float8_D > 0) {
        auto max_fp8_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float8_D, SparseType::FP8, row_alignment), 128);
        TORCH_CHECK(max_fp8_128b_rows <= 16);
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
        if (max_fp8_128b_rows > 8) {
          Y(2, 2, 4, 8);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch 16-bit float kernel
    ////////////////////////////////////////////////////////////////////////////

#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name5 = "nbit::FP16_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L";
#endif

    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::FP16_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        MAKE_PTA_WITH_NAME(func_name5, dev_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name5, uvm_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name5, weights_placements, int32_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name5, weights_offsets, int64_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name5, weights_tys, uint8_t, 1, 32), \
        {% if not nobag %} \
        MAKE_PTA_WITH_NAME(func_name5, D_offsets, int32_t, 1, 32), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        MAKE_PTA_WITH_NAME(func_name5, indices, index_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name5, offsets, index_t, 1, 32), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} MAKE_PTA_WITH_NAME(func_name5, indice_weights, float, 1, 32), {% endif %} \
        MAKE_PTA_WITH_NAME(func_name5, output, output_t, 2, 32), \
        MAKE_PTA_WITH_NAME(func_name5, lxu_cache_weights, uint8_t, 2, 64), \
        MAKE_PTA_WITH_NAME(func_name5, lxu_cache_locations, int32_t, 1, 32) \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "fp16_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_float16_D > 0) {
        auto max_fp16_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float16_D, SparseType::FP16, row_alignment), 128);
        TORCH_CHECK(max_fp16_128b_rows <= 32);
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
        if (max_fp16_128b_rows > 16) {
          Y(2, 1, 16, 32);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch 32-bit float kernel
    ////////////////////////////////////////////////////////////////////////////

#ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name6 = "nbit::FP32_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L";
#endif

    #define X(DeviceOnly, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    nbit::FP32_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        MAKE_PTA_WITH_NAME(func_name6, dev_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name6, uvm_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name6, weights_placements, int32_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name6, weights_offsets, int64_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name6, weights_tys, uint8_t, 1, 32), \
        {% if not nobag %} \
        MAKE_PTA_WITH_NAME(func_name6, D_offsets, int32_t, 1, 32), \
        {% else %} \
        D, \
        {% endif %} \
        FixedDivisor(div_round_up(B, OutputRowsPerThread)), \
        MAKE_PTA_WITH_NAME(func_name6, indices, index_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name6, offsets, index_t, 1, 32), \
        {% if not nobag %} \
        pooling_mode, \
        {% endif %} \
        row_alignment, \
        {% if weighted %} MAKE_PTA_WITH_NAME(func_name6, indice_weights, float, 1, 32), {% endif %} \
        MAKE_PTA_WITH_NAME(func_name6, output, output_t, 2, 32), \
        MAKE_PTA_WITH_NAME(func_name6, lxu_cache_weights, uint8_t, 2, 64), \
        MAKE_PTA_WITH_NAME(func_name6, lxu_cache_locations, int32_t, 1, 32) \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "fp32_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_float32_D > 0) {
        auto max_fp32_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_float32_D, SparseType::FP32, row_alignment), 128);
        TORCH_CHECK(max_fp32_128b_rows <= 64);
        if (max_fp32_128b_rows > 0) {
          Y(2, 4, 0, 4);
        }
        if (max_fp32_128b_rows > 4) {
          Y(2, 2, 4, 16);
        }
        if (max_fp32_128b_rows > 16) {
          Y(1, 1, 16, 32);
        }
        if (max_fp32_128b_rows > 32) {
          Y(1, 1, 32, 64);
        }
      }
    }));
    #undef X

    return output;
}

// clang-format on
