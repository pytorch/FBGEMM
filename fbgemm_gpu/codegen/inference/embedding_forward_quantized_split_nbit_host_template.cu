/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{%- set wdesc =  "weighted" if weighted else "unweighted" %}
#include "fbgemm_gpu/embedding_forward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor.h"

using namespace fbgemm_gpu;
using Tensor = at::Tensor;

namespace nbit {

/*
  Looping over the weight types is required to generate all the C++ template
  declarations (not definitions) that will be invoked by the function
  `Tensor int_nbit_split_embedding*_codegen_forward_*_cuda(...)` later in the
  same generated source file.
*/
{%- for emb_weight_type in ["FP32", "FP16", "FP8", "INT8", "INT4", "INT2"] %}
template<typename index_t, typename output_t, size_t OutputRowsPerThread, size_t WarpsPerBlock, size_t InputRowsInFlight, size_t MinNum128BRows, size_t MaxNum128BRows, bool DeviceOnly, bool PackedMode>
__launch_bounds__(WarpsPerBlock * kWarpSize)
__global__ void {{ type_map[emb_weight_type].enum_name }}_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel_small_L(
  const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> dev_weights,
  const pta::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> uvm_weights,
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
  const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
  const pta::PackedTensorAccessor32<uint8_t, 1, at::RestrictPtrTraits> weights_tys,
  {%- if not nobag %}
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
  {%- else %}
  const int64_t D,
  {%- endif %}
  FixedDivisor fd_B, // FixedDivisor(div_round_up(B, OutputRowsPerThread))
  const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
  const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
  {%- if not nobag %}
  const int64_t pooling_mode,
  {%- endif %}
  const int64_t row_alignment,
  {%- if weighted %}
  pta::PackedTensorAccessor32<float, 1, at::RestrictPtrTraits> indice_weights,
  {%- endif %}
  {%- if type_map[emb_weight_type].enum_name == "FP8" %}
  const int fp8_exponent_bits,
  const int fp8_exponent_bias,
  {%- endif %}
  const int32_t num_packed_bags,
  pta::PackedTensorAccessor32<output_t, 2, at::RestrictPtrTraits> output, // [B][total_D],
  const pta::PackedTensorAccessor64<uint8_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
  const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations
  );
{%- endfor %} // for emb_weight_type in ["FP32", "FP16", "FP8", "INT8", "INT4", "INT2"]

}

{%- macro define_kernel_invocation(emb_weight_type) %}
    {%- set func_name = "nbit::" + emb_weight_type + "_split_embedding" + ("_nobag" if nobag else "") + "_codegen_forward_" + wdesc + "_kernel_small_L" %}

    #ifdef FBGEMM_GPU_MEMCHECK
    const auto func_name_{{ emb_weight_type }} = "{{ func_name }}_{{ emb_weight_type }}";
    #endif

    #ifdef X
    #undef X
    #endif

    // Define {{ emb_weight_type }} kernel invocation macro
    #define X(DeviceOnly, PackedMode, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
    {{ func_name }}<index_t, output_t, OutputRowsPerThread, kWarpsPerBlock, InputRowsInFlight, MinNum128BRows, MaxNum128BRows, DeviceOnly, PackedMode><<< \
        nbit::div_round_up(T * nbit::div_round_up(B, num_packed_bags * OutputRowsPerThread), kWarpsPerBlock), \
        dim3(kWarpSize, kWarpsPerBlock), \
        0, \
        at::cuda::getCurrentCUDAStream()>>>( \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, dev_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, uvm_weights, uint8_t, 1, 64), \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, weights_placements, int32_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, weights_offsets, int64_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, weights_tys, uint8_t, 1, 32), \
        {%- if not nobag %}
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, D_offsets, int32_t, 1, 32), \
        {%- else %}
        D, \
        {%- endif %}
        FixedDivisor(div_round_up(B, num_packed_bags * OutputRowsPerThread)), \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, indices, index_t, 1, 32), \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, offsets, index_t, 1, 32), \
        {%- if not nobag %}
        pooling_mode, \
        {%- endif %}
        row_alignment, \
        {%- if weighted %}
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, indice_weights, float, 1, 32), \
        {%- endif %}
        {%- if emb_weight_type == "FP8" %}
        fp8_exponent_bits, \
        fp8_exponent_bias, \
        {%- endif %}
        num_packed_bags, \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, output, output_t, 2, 32), \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, lxu_cache_weights, uint8_t, 2, 64), \
        MAKE_PTA_WITH_NAME(func_name_{{ emb_weight_type }}, lxu_cache_locations, int32_t, 1, 32) \
    ); \
    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
{%- endmacro %}

{%- macro construct_and_return_output_tensor() %}
    // kernels assume indices are contiguous.
    indices = indices.contiguous();

    {%- if not nobag %}
    const int32_t T = D_offsets.numel() - 1;
    {%- else %}
    const int32_t total_L = indices.numel();
    const int32_t T = weights_offsets.numel();
    {%- endif %}

    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    const int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);

    {%- if not nobag %}
    TORCH_CHECK(total_D > 0);
    {%- else %}
    TORCH_CHECK(D > 0);
    {%- endif %}

    // Construct output tensor
    Tensor output;
    const int kINT8QparamsBytes = 8;

    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 || o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);

    {%- if not nobag %}

    int64_t total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        total_adjusted_D += T * kINT8QparamsBytes;
    }

    if (indices.numel() == 0) {
      output = at::zeros({B, total_adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    } else {
      output = at::empty({B, total_adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    }

    {%- else %}

    int64_t adjusted_D = D;
    if (o_dtype == SparseType::INT8) {
        adjusted_D += T * kINT8QparamsBytes;
    }

    if (total_L == 0) {
      output = at::zeros({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    } else {
      output = at::empty({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    }

    {%- endif %}

    if (B == 0 || indices.numel() == 0) {
      return output;
    }
{%- endmacro %}

template <typename index_t>
Tensor int_nbit_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_cuda_impl(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    {%- if not nobag %}
    Tensor D_offsets,
    const int64_t total_D,
    {%- else %}
    const int64_t D,
    {%- endif %}
    const int64_t max_int2_D,
    const int64_t max_int4_D,
    const int64_t max_int8_D,
    const int64_t max_float16_D,
    const int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    {%- endif %}
    const int64_t row_alignment,
    {%- if weighted %}
    Tensor indice_weights,
    {%- endif %}
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
    {%- if not nobag %}
    TENSORS_ON_SAME_DEVICE(D_offsets, dev_weights);
    {%- endif %}
    TENSORS_ON_SAME_DEVICE(indices, dev_weights);
    TENSORS_ON_SAME_DEVICE(offsets, dev_weights);
    {%- if weighted %}
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(indice_weights, dev_weights);
    {%- endif %}
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(lxu_cache_weights, dev_weights);
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(lxu_cache_locations, dev_weights);

    CUDA_DEVICE_GUARD(dev_weights);

    {{- construct_and_return_output_tensor() }}

    constexpr int32_t kWarpsPerBlock = 4;
    const auto device_only = lxu_cache_weights.numel() == 0 && uvm_weights.numel() == 0;
    #define PACKED_MODE_SWITCH(dev_only, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows) \
      {%-if is_rocm and not nobag %}
      const int32_t num_uint4_loads_per_row = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_D, sparse_type, row_alignment), sizeof(uint4)); \
      constexpr int32_t NumUint4LoadsPerRow = MaxNum128BRows * 128 / sizeof(uint4); \
      const int32_t num_packed_bags = NumUint4LoadsPerRow > num_uint4_loads_per_row && !std::is_same_v<output_t, uint8_t> && sparse_type != SparseType::FP32 ? NumUint4LoadsPerRow / num_uint4_loads_per_row : 1; \
      {%- else %}
      const int32_t num_packed_bags = 1; \
      {%- endif %}
      if (num_packed_bags > 1) {              \
        X(dev_only, true, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows)        \
      } else {                                \
        X(dev_only, false, OutputRowsPerThread, InputRowsInFlight, MinNum128BRows, MaxNum128BRows)       \
      };

    #define Y(...) \
      if (device_only) { \
        PACKED_MODE_SWITCH(true, __VA_ARGS__) \
      } else { \
        PACKED_MODE_SWITCH(false, __VA_ARGS__) \
      };


    ////////////////////////////////////////////////////////////////////////////
    // Launch INT2 kernel
    ////////////////////////////////////////////////////////////////////////////

    {{- define_kernel_invocation("INT2") }}

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "int2_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_int2_D > 0) {
        const auto max_D = max_int2_D;
        constexpr auto sparse_type = SparseType::INT2;
        auto max_int2_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_D, sparse_type, row_alignment), 128);
        TORCH_CHECK(max_int2_128b_rows <= 8);
        if (max_int2_128b_rows > 0) {
          Y(2, 16, 0, 1);
        }
        if (max_int2_128b_rows > 1) {
          Y(2, 8, 1, 2);
        }
        if (max_int2_128b_rows > 2) {
          Y(2, 8, 2, 4);
        }
        if (max_int2_128b_rows > 4) {
          Y(2, 4, 4, 8);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch INT4 kernel
    ////////////////////////////////////////////////////////////////////////////

    {{- define_kernel_invocation("INT4") }}

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "int4_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_int4_D > 0) {
        const auto max_D = max_int4_D;
        constexpr auto sparse_type = SparseType::INT4;
        auto max_int4_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_D, sparse_type, row_alignment), 128);
        TORCH_CHECK(max_int4_128b_rows <= 16);
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
        if (max_int4_128b_rows > 8) {
          Y(1, 4, 8, 16);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch INT8 kernel
    ////////////////////////////////////////////////////////////////////////////

    {{- define_kernel_invocation("INT8") }}

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "int8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_int8_D > 0) {
        const auto max_D = max_int8_D;
        constexpr auto sparse_type = SparseType::INT8;
        auto max_int8_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_D, sparse_type, row_alignment), 128);
        TORCH_CHECK(max_int8_128b_rows <= 32);
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
        if (max_int8_128b_rows > 16) {
          Y(1, 2, 16, 32);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch FP8 kernel
    ////////////////////////////////////////////////////////////////////////////

    {{- define_kernel_invocation("FP8") }}

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "fp8_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_float8_D > 0) {
        const auto max_D = max_float8_D;
        constexpr auto sparse_type = SparseType::FP8;
        auto max_fp8_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_D, sparse_type, row_alignment), 128);
        TORCH_CHECK(max_fp8_128b_rows <= 32);
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
          Y(2, 2, 8, 16);
        }
        if (max_fp8_128b_rows > 16) {
          Y(1, 2, 16, 32);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch FP16 kernel
    ////////////////////////////////////////////////////////////////////////////

    {{- define_kernel_invocation("FP16") }}

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "fp16_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_float16_D > 0) {
        const auto max_D = max_float16_D;
        constexpr auto sparse_type = SparseType::FP16;
        auto max_fp16_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_D, sparse_type, row_alignment), 128);
        TORCH_CHECK(max_fp16_128b_rows <= 64);
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
          Y(1, 2, 16, 32);
        }
        if (max_fp16_128b_rows > 32) {
          Y(1, 1, 32, 64);
        }
      }
    }));
    #undef X


    ////////////////////////////////////////////////////////////////////////////
    // Launch FP32 kernel
    ////////////////////////////////////////////////////////////////////////////

    {{- define_kernel_invocation("FP32") }}

    DISPATCH_OUTPUT_TYPES(output.scalar_type(), "fp32_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_kernel", ([&] {
      if (max_float32_D > 0) {
        const auto max_D = max_float32_D;
        constexpr auto sparse_type = SparseType::FP32;
        auto max_fp32_128b_rows = nbit::div_round_up(nbit::padded_row_size_in_bytes(max_D, sparse_type, row_alignment), 128);
        TORCH_CHECK(max_fp32_128b_rows <= 64); // 128 doesn't fit in 48KB SM, so FP32 TBE supports a smaller dimension than others
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

Tensor int_nbit_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    {%- if not nobag %}
    Tensor D_offsets,
    const int64_t total_D,
    {%- else %}
    const int64_t D,
    {%- endif %}
    const int64_t max_int2_D,
    const int64_t max_int4_D,
    const int64_t max_int8_D,
    const int64_t max_float16_D,
    const int64_t max_float32_D,
    Tensor indices,
    Tensor offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    {%- endif %}
    const int64_t row_alignment,
    {%- if weighted %}
    Tensor indice_weights,
    {%- endif %}
    const int64_t output_dtype,
    Tensor lxu_cache_weights,
    Tensor lxu_cache_locations,
    const int64_t max_float8_D,
    const int64_t fp8_exponent_bits,
    const int64_t fp8_exponent_bias
) {
    // All argument tensors need to be on the same CUDA device
    TENSOR_ON_CUDA_GPU(dev_weights);
    TENSORS_ON_SAME_DEVICE(uvm_weights, dev_weights);
    TENSORS_ON_SAME_DEVICE(weights_placements, dev_weights);
    TENSORS_ON_SAME_DEVICE(weights_offsets, dev_weights);
    TENSORS_ON_SAME_DEVICE(weights_tys, dev_weights);
    {%- if not nobag %}
    TENSORS_ON_SAME_DEVICE(D_offsets, dev_weights);
    {%- endif %}
    TENSORS_ON_SAME_DEVICE(indices, dev_weights);
    TENSORS_ON_SAME_DEVICE(offsets, dev_weights);
    {%- if weighted %}
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(indice_weights, dev_weights);
    {%- endif %}
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(lxu_cache_weights, dev_weights);
    TENSORS_EMPTY_OR_ON_SAME_DEVICE(lxu_cache_locations, dev_weights);

    // indices and offsets need to have the same scalar type
    TENSORS_HAVE_SAME_TYPE(indices, offsets);
    // Only int32_t and int64_t indices are supported at the moment
    TENSOR_SCALAR_TYPE_IS_ONE_OF(indices, at::ScalarType::Long, at::ScalarType::Int);

    CUDA_DEVICE_GUARD(dev_weights);

    // Create output tensor ref
    Tensor output;

    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "{{ 'int_nbit_split_embedding' + ('_nobag' if nobag else '') + '_codegen_forward_' + wdesc + '_cuda' }}", [&] {
      output = int_nbit_split_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_cuda_impl<index_t>(
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        weights_tys,
        {%- if not nobag %}
        D_offsets,
        total_D,
        {%- else %}
        D,
        {%- endif %}
        max_int2_D,
        max_int4_D,
        max_int8_D,
        max_float16_D,
        max_float32_D,
        indices,
        offsets,
        {%- if not nobag %}
        pooling_mode,
        {%- endif %}
        row_alignment,
        {%- if weighted %}
        indice_weights,
        {%- endif %}
        output_dtype,
        lxu_cache_weights,
        lxu_cache_locations,
        max_float8_D,
        fp8_exponent_bits,
        fp8_exponent_bias);
    });

    return output;
}

        // clang-format on
