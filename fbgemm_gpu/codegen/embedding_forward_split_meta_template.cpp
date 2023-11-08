/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

{#
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
// Note: clang-format off doesn't work with this templaterized code,
// so we need to keep lint-ignore-every.
// See https://fburl.com/dw9ljh4h
#}

// Companion template is embedding_forward_split_template.cu

{%- set ddesc =  "dense" if dense else "split" %}
{%- set wdesc =  "weighted" if weighted else "unweighted" %}
{%- set vdesc = "_vbe" if vbe else "" %}

////////////////////////////////////////////////////////////////////////////////
// Required for op registrations
#include "codegen/embedding_op_registration.h"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/embedding_common.h"
////////////////////////////////////////////////////////////////////////////////

using namespace fbgemm_gpu;
using Tensor = at::Tensor;

static constexpr float kINT8QparamsBytes = 8;

////////////////////////////////////////////////////////////////////////////////
// Kernel Definitions
////////////////////////////////////////////////////////////////////////////////

{%- for nobag in [True, False] %}
{%- set ndesc = "_nobag" if nobag else "" %}
{%- if (not nobag or (not weighted and not vbe)) %}
{%- set has_experimental = (not dense and not nobag and not vbe) %}

Tensor
{{ ddesc }}_embedding{{ ndesc }}_codegen_forward_{{ wdesc }}{{ vdesc }}_meta(
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    {%- if not nobag %}
    const Tensor& D_offsets,
    {%- else %}
    const int64_t D,
    {%- endif %}
    {%- if not nobag %}
    const int64_t total_D,
    {%- endif %}
    {%- if not nobag %}
    const int64_t max_D,
    {% endif %}
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    const Tensor& indice_weights,
    {%- endif %}
    {%- if not dense %}
    const Tensor& lxu_cache_locations,
    {%- endif %}
    const int64_t output_dtype,
    {%- if vbe %}
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const int64_t vbe_output_size,
    const int64_t info_B_num_bits, // int32_t
    const int64_t info_B_mask_int64, // uint32_t
    {%- endif %}
    const bool is_experimental
) {
    // NB: omitted the device tests TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL

    {%- if not nobag %}
    auto T = D_offsets.sym_numel() - 1;
    {%- else %}
    auto total_L = indices.sym_numel();
    auto T = weights_offsets.sym_numel();
    {%- endif %}
    TORCH_CHECK_GT(T, 0);
    // offsets = [B x T  + 1]
    {%- if is_index_select %}
    const auto total_B = num_warps_per_feature * T;
    const auto B = num_warps_per_feature;
    {%- else %}
    const auto total_B = offsets.sym_size(0) - 1;
    const auto B = total_B / T;
    {%- endif %}
    TORCH_CHECK_GE(B, 0);
    {%- if not nobag or is_index_select %}
    {%- if not nobag %}
    TORCH_CHECK_GT(total_D, 0);
    TORCH_CHECK_EQ(total_D % 4, 0);
    {%- endif %}
    TORCH_CHECK_LE(max_D, {{ max_embedding_dim }});
    {%- elif not is_index_select %}
    TORCH_CHECK_GT(D, 0);
    TORCH_CHECK_EQ(D % 4, 0);
    {%- endif %}
    {%- if vbe %}
    TORCH_CHECK_EQ(vbe_row_output_offsets.sym_numel(), total_B);
    TENSORS_HAVE_SAME_NUMEL(vbe_row_output_offsets, vbe_b_t_map);
    TORCH_CHECK_GE(vbe_output_size, 0);

    // Cast info_B_mask from int64_t to uint32_t
    const uint32_t info_B_mask = info_B_mask_int64;
    {%- endif %}

    Tensor output;
    {%- if nobag %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    {%- if is_index_select %}
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16);

    TORCH_CHECK_GT(fixed_L_per_warp, 0);
    TORCH_CHECK_GT(num_warps_per_feature, 0);
    if (!permute_output_dim_0_1) {
        TORCH_CHECK_GE(output_size, 0);
        TORCH_CHECK_GT(output_offsets.sym_numel(), 0);
    }

    // If permute_output_dim_0_1 is true, output shape is (batch_size * total_D)
    // Else, output shape is (output_size)
    output = at::empty_symint({output_size}, dev_weights.options().dtype(getScalarType(o_dtype)));
    {%- else %}
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);

    c10::SymInt adjusted_D = D;
    if (o_dtype == SparseType::INT8) {
        adjusted_D += T * int64_t(kINT8QparamsBytes);
    }

    output = at::empty_symint({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype)));
    {%- endif %}
    {%- else %}
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);
    c10::SymInt total_adjusted_D = total_D;
    if (o_dtype == SparseType::INT8) {
        // TODO: Why is kINT8QparamsBytes a float
        total_adjusted_D += T * int64_t(kINT8QparamsBytes);
    }

    {%- if vbe %}
    // Use a 2D tensor to make it compatible with 2D PackedTensorsAccessor of other output
    output = at::empty_symint(
        {1, vbe_output_size},
        dev_weights.options().dtype(getScalarType(o_dtype))
    );
    {%- else %}
    output = at::empty_symint(
        {B, total_adjusted_D},
        dev_weights.options().dtype(getScalarType(o_dtype))
    );
    {%- endif %}
    {%- endif %} // if nobag

    if (B == 0) {
        {%- if vbe %}
        output = output.reshape({-1});
        {%- endif %}
        return output;
    }

    return output;
}

////////////////////////////////////////////////////////////////////////////////
// Op registrations
////////////////////////////////////////////////////////////////////////////////
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    // NB: yes cuda here
    {%- set embedding_codegen_forward_op =
        "{}_embedding{}_codegen_forward_{}{}_cuda".format(
            ddesc, ndesc, wdesc, vdesc
        )
    %}
    m.impl("{{ embedding_codegen_forward_op }}", torch::dispatch(c10::DispatchKey::Meta, TORCH_FN({{ ddesc }}_embedding{{ ndesc }}_codegen_forward_{{ wdesc }}{{ vdesc }}_meta)));
}
{%- endif %} {#/* if (not nobag or (not weighted and not vbe)) */#}
{%- endfor %} {#-/* for nobag */#}
    // clang-format on
