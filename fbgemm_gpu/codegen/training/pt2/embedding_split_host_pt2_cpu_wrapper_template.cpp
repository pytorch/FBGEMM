/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{#-/* This file contains a template for implementation of pt2 wrapper
    functions for CPU and respective op registration to PyTorch
    dispatcher.
    Codegen file output:
    gen_embedding_forward_split_pt2_cpu_wrapper.cpp
    gen_embedding_backward_split_{optimizer}_pt2_cpu_wrapper.cpp

    [PT2 Autograd] --Torch dispatch-->                   |
        [PT2 wrapper] --Torch dispatch--> [CPU backend] | <<<
            --Fn call--> [CPU kernel]                   |
*/#}
{%- if has_cpu_support %}
////////////////////////////////////////////////////////////////////////////////
// Required for op registrations and dispatchers
////////////////////////////////////////////////////////////////////////////////
#include <torch/script.h>
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
// #include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
{%- if has_vbe_support %}
#include "fbgemm_gpu/utils/pt2_autograd_utils.h"
{%- endif %}

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{%- for vbe in ([True, False] if has_vbe_support else [False]) %}
{%- set vdesc = "_vbe" if vbe else "" %}

{%- if is_forward %}
{#-/* PT2 wrapper function for backward grad_indice_weights CPU */#}
Tensor split_embedding_codegen_grad_indice_weights{{ vdesc }}_pt2_cpu_wrapper(
    const Tensor& grad_output,
    const Tensor& host_weights,
    const Tensor& /*dev_weights*/,
    const Tensor& /*uvm_weights*/,
    const Tensor& /*lxu_cache_weights*/,
    const Tensor& /*weights_placements*/,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt /*max_D*/,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& /*lxu_cache_locations*/,
    {%- if vbe %}
    const Tensor& feature_requires_grad,
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    const Tensor& vbe_B_offsets_rank_per_feature,
    const c10::SymInt max_B
    {%- else %}
    const Tensor& feature_requires_grad
    {%- endif %}
) {
    {%- if vbe %}
    Tensor offsets_;
    const int64_t max_B_int = max_B.guard_int(__FILE__, __LINE__);
    AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "reshape_vbe_offsets_cpu_grad_indices", [&]() {
        offsets_ = reshape_vbe_offsets<index_t>(
            offsets,
            vbe_B_offsets_rank_per_feature,
            max_B_int,
            D_offsets.numel() - 1
        );
    });
    const auto grad_output_ = reshape_vbe_output(
        grad_output,
        max_B_int,
        vbe_B_offsets_rank_per_feature,
        D_offsets
    );
    {%- endif %}
  static auto op =
      torch::Dispatcher::singleton()
        .findSchemaOrThrow(
              "fbgemm::split_embedding_codegen_grad_indice_weights_cpu", "")
        .typed<Tensor(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor)>();
  return op.call(
      {{ "grad_output_" if vbe else "grad_output" }},
      host_weights,
      weights_offsets,
      D_offsets,
      indices,
      {{ "offsets_" if vbe else "offsets" }},
      feature_requires_grad);
}
{%- endif %}
{%- for weighted in [True, False] %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}

{% if is_forward %}
{#-/* PT2 wrapper function for forward CPU */#}
Tensor split_embedding_codegen_forward_{{ wdesc }}{{ vdesc }}_pt2_cpu_wrapper(
    const Tensor& host_weights,
    const Tensor& /*dev_weights*/,
    const Tensor& /*uvm_weights*/,
    const Tensor& /*lxu_cache_weights*/,
    const Tensor& /*weights_placements*/,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt /*max_D*/,
    const Tensor& hash_size_cumsum,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t pooling_mode,
    const Tensor& indice_weights,
    const Tensor& /*lxu_cache_locations*/,
    const Tensor& /*uvm_cache_stats*/,
    {%- if vbe %}
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const c10::SymInt vbe_output_size,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    const Tensor& vbe_B_offsets_rank_per_feature,
    const Tensor& vbe_output_offsets_feature_rank,
    const c10::SymInt max_B,
    {%- endif %}
    const bool /*is_experimental = false*/,
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)) {
    Tensor offsets_;
    {%- if vbe %}
    const int64_t max_B_int = max_B.guard_int(__FILE__, __LINE__);
    AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "reshape_vbe_offsets_cpu_forward", [&]() {
        offsets_ = reshape_vbe_offsets<index_t>(offsets, vbe_B_offsets_rank_per_feature, max_B_int, D_offsets.numel() - 1);
    });
    {%- endif %}
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::split_embedding_codegen_forward_cpu", "")
            .typed<Tensor(
                    Tensor, Tensor, Tensor, c10::SymInt, Tensor, Tensor, Tensor, int64_t, Tensor, int64_t
            )>();
    {%- if vbe %}
    // TODO: remove this after vbe is implemented for CPU kernel
    const auto output = op.call(
        host_weights,
        weights_offsets,
        D_offsets,
        total_D,
        hash_size_cumsum,
        indices,
        offsets_,
        pooling_mode,
        indice_weights,
        output_dtype);
    auto options = at::TensorOptions()
        .dtype(output.options().dtype())
        .device(host_weights.options().device());
    const int64_t vbe_output_size_ = vbe_output_size.guard_int(__FILE__, __LINE__);
    Tensor output_new = at::empty({vbe_output_size_}, options);
    const int32_t T = D_offsets.numel() - 1;
    const int32_t R = vbe_B_offsets_rank_per_feature.size(1) - 1;

    for (int32_t r = 0; r < R; r++){
        auto D_offset = 0;
        for (int32_t t = 0; t < T; t++){
            const int32_t o_begin = vbe_output_offsets_feature_rank[r * T + t].item<int32_t>();
            const int32_t o_end = vbe_output_offsets_feature_rank[r * T + t + 1].item<int32_t>();
            const int32_t D = D_offsets[t + 1].item<int32_t>() - D_offsets[t].item<int32_t>();
            const int32_t b_begin = vbe_B_offsets_rank_per_feature[t][r].item<int32_t>();
            const int32_t b_end = vbe_B_offsets_rank_per_feature[t][r + 1].item<int32_t>();
            
            TORCH_CHECK((o_end - o_begin) == ((b_end - b_begin) * D));           
            auto values = output.index({torch::indexing::Slice(b_begin, b_end), torch::indexing::Slice(D_offset, D_offset + D)}).flatten();
            output_new.index_put_({torch::indexing::Slice(o_begin, o_end)}, values);
            D_offset += D;
        }
    }
    return output_new;
    {%- else %}
    return op.call(
        host_weights,
        weights_offsets,
        D_offsets,
        total_D,
        hash_size_cumsum,
        indices,
        offsets,
        pooling_mode,
        indice_weights,
        output_dtype);
    {%- endif %}
    }
{% else %}
{#-/* PT2 wrapper function for backward CPU */#}
Tensor split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_pt2_cpu_wrapper(
    const Tensor& grad_output,
    const Tensor& host_weights,
    const Tensor& /*dev_weights*/,
    const Tensor& /*uvm_weights*/,
    const Tensor& /*lxu_cache_weights*/,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt max_D,
    const bool mixed_D,
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t pooling_mode,
    const Tensor& indice_weights,
    const Tensor& /*lxu_cache_locations*/,
    const int64_t /*BT_block_size*/,
    const int64_t /*max_segment_length_per_warp*/,
    const bool stochastic_rounding,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    {%- if vbe %}
    const Tensor& B_offsets,
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const Tensor& vbe_B_offsets_rank_per_feature,
    const c10::SymInt max_B,
    {%- endif %}
    const bool /*use_uniq_cache_locations*/,
    const bool /*use_homogeneous_placements*/,
    {{ args_pt2.split_function_args | join(", ") }}
    {%- if not nobag %}
    , const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)
    {%- endif %})
    {
        {%- if vbe %}
        const int64_t max_B_int = max_B.guard_int(__FILE__, __LINE__);
        Tensor offsets_;
        AT_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "reshape_vbe_offsets_cpu_backward", [&]() {
            offsets_ = reshape_vbe_offsets<index_t>(offsets, vbe_B_offsets_rank_per_feature, max_B_int, D_offsets.numel() - 1);
        });
        const auto grad_output_ = reshape_vbe_output(grad_output, max_B_int, vbe_B_offsets_rank_per_feature, D_offsets);
        {%- endif %}
        {%- set backward_op = "split_embedding_backward_codegen_{}_cpu".format(
                optimizer
            )
        %}
        static auto op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::{{ backward_op }}", "")
                .typed<void(
                    Tensor, /* grad_output */
                    Tensor, /* host_weights */
                    Tensor, /* weights_placements */
                    Tensor, /* weights_offsets */
                    Tensor, /* D_offsets */
                    int64_t, /* max_D */
                    Tensor, /* hash_size_cumsum */
                    int64_t, /* total_hash_size_bits */
                    Tensor, /* indices */
                    Tensor, /* offsets */
                    int64_t, /* pooling_mode */
                    Tensor, /* indices_weights */
                    bool, /* stochastic_rounding */
                    {%- for arg_type in args.split_function_args %}
                    {{ arg_type.split(' ')[0]}}{%- if not loop.last %}{{ "," }}{%- endif %}
                    {%- endfor %}
                    {%- if not nobag %}
                    , int64_t
                    {%- endif %}
                )>();

        op.call(
            {{ "grad_output_" if vbe else "grad_output" }},
            host_weights,
            weights_placements,
            weights_offsets,
            D_offsets,
            max_D.guard_int(__FILE__, __LINE__),
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            {{ "offsets_" if vbe else "offsets" }},
            pooling_mode,
            indice_weights,
            stochastic_rounding,
            {{ args.split_function_arg_names | join(", ") }}
            {%- if not nobag %}
            , output_dtype
            {%- endif %}
            );
        return Tensor();
    }
{% endif %}
{%- endfor %} {#-/*for weighted*/#}


namespace {
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    {%- for weighted in [True, False] %}
    {%- set wdesc = "weighted" if weighted else "unweighted" %}
    
    {%- if is_forward %}
    {%- set embedding_codegen_forward_op = "split_embedding_codegen_forward_{}{}_pt2".format(
        wdesc, vdesc
        )
    %}
    m.def("{{ embedding_codegen_forward_op }}_wrapper("
        "    Tensor host_weights, "
        "    Tensor dev_weights, "
        "    Tensor uvm_weights, "
        "    Tensor lxu_cache_weights, "
        "    Tensor weights_placements, "
        "    Tensor weights_offsets, "
        {%- if nobag %}
        "    SymInt D, "
        {%- else %}
        "    Tensor D_offsets, "
        "    SymInt total_D, "
        "    SymInt max_D, "
        {%- endif %}
        "    Tensor hash_size_cumsum, "
        "    Tensor indices, "
        "    Tensor offsets, "
        {%- if not nobag %}
        "    int pooling_mode, "
        "    Tensor indice_weights, "
        {%- endif %}
        "    Tensor lxu_cache_locations, "
        "    Tensor{{ schema_annotation['uvm_cache_stats'] }} uvm_cache_stats, "
        {%- if vbe %}
        "    Tensor vbe_row_output_offsets, "
        "    Tensor vbe_b_t_map, "
        "    SymInt vbe_output_size, "
        "    int info_B_num_bits, "
        "    int info_B_mask_int64, "
        "    Tensor vbe_B_offsets_rank_per_feature, "
        "    Tensor vbe_output_offsets_feature_rank, "
        "    SymInt max_B, "
        {%- endif %}
        "    bool is_experimental, "
        "    int output_dtype "
        ") -> Tensor"
        {%- if not nobag and not vbe %}
          // only split_embedding_codegen_forward_[un]weighted_cuda
          // are tested to be PT2 compliant
        , {PT2_COMPLIANT_TAG}
        {%- endif %}
        );
    DISPATCH_TO_CPU("{{ embedding_codegen_forward_op }}_wrapper", {{ embedding_codegen_forward_op }}_cpu_wrapper);

    {%- else %} {#-/* backward */#}
    {%- set embedding_codegen_backward_op = "split_embedding_backward_codegen_{}_{}{}_pt2".format(
        optimizer, wdesc, vdesc
        )
    %}
    m.def("{{ embedding_codegen_backward_op }}_wrapper("
        "    Tensor grad_output, "
        "    Tensor{{ schema_annotation['weights_host'] }} host_weights, "
        "    Tensor{{ schema_annotation['weights_dev'] }} dev_weights, "
        "    Tensor{{ schema_annotation['weights_uvm'] }} uvm_weights, "
        "    Tensor{{ schema_annotation['weights_lxu_cache'] }} lxu_cache_weights, "
        "    Tensor weights_placements, "
        "    Tensor weights_offsets, "
        {%- if nobag %}
        "    SymInt D, "
        {%- else %}
        "    Tensor D_offsets, "
        "    SymInt max_D, "
        "    bool mixed_D, "
        {%- endif %}
        "    Tensor hash_size_cumsum, "
        "    int total_hash_size_bits, "
        "    Tensor indices, "
        "    Tensor offsets, "
        {%- if not nobag %}
        "    int pooling_mode, "
        "    Tensor indice_weights, "
        {%- endif %}
        "    Tensor lxu_cache_locations, "
        "    int BT_block_size, "
        "    int max_segment_length_per_warp, "
        {%- if optimizer != "none" %}
        "    bool stochastic_rounding, "
        {%- endif %}
        "    int info_B_num_bits, "
        "    int info_B_mask_int64, "
        {%- if vbe %}
        "    Tensor B_offsets, "
        "    Tensor vbe_row_output_offsets, "
        "    Tensor vbe_b_t_map, "
        "    Tensor vbe_B_offsets_rank_per_feature, "
        "    SymInt max_B, "
        {%- endif %}
        "    bool use_uniq_cache_locations, "
        "    bool use_homogeneous_placements,"
        "    {{ args_pt2.split_function_schemas | join(", ") }} "
        {%- if not nobag %}
        "    , int output_dtype=0 "
        {%- endif %}
        ") -> Tensor");
    DISPATCH_TO_CPU("{{ embedding_codegen_backward_op }}_wrapper", {{ embedding_codegen_backward_op }}_cpu_wrapper);
    {%- endif %} {#-/*if is_forward*/#}
    {%- endfor %} {#-/*for weighted*/#}

    {%- if is_forward %}
    {%- set embedding_codegen_grad_indice_weights_op =
        "split_embedding_codegen_grad_indice_weights{}_pt2".format(
            vdesc
        )
    %}
    m.def("{{ embedding_codegen_grad_indice_weights_op }}_wrapper("
        "    Tensor grad_output, "
        "    Tensor host_weights, "
        "    Tensor dev_weights, "
        "    Tensor uvm_weights, "
        "    Tensor lxu_cache_weights, "
        "    Tensor weights_placements, "
        "    Tensor weights_offsets, "
        "    Tensor D_offsets, "
        "    SymInt max_D, "
        "    Tensor indices, "
        "    Tensor offsets, "
        "    Tensor lxu_cache_locations, "
        {%- if vbe %}
        "    Tensor feature_requires_grad, "
        "    Tensor vbe_row_output_offsets, "
        "    Tensor vbe_b_t_map, "
        "    int info_B_num_bits, "
        "    int info_B_mask_int64, "
        "    Tensor vbe_B_offsets_rank_per_feature, "
        "    SymInt max_B "
        {%- else %}
        "    Tensor feature_requires_grad"
        {%- endif %}
        ") -> Tensor");

    DISPATCH_TO_CPU(
        "{{ embedding_codegen_grad_indice_weights_op }}_wrapper",
        {{ embedding_codegen_grad_indice_weights_op }}_cpu_wrapper);
    {%- endif %}
}
} // namespace
{%- endfor %} {#-/* for vbe in [True, False] */#}

{% endif %} {#/* if has_cpu_support */#}
// clang-format on
