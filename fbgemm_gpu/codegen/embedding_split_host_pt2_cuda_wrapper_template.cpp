/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{#-/* This file contains a template for implementation of pt2 wrapper
    functions for CUDA and respective op registration to PyTorch
    dispatcher
    Codegen file output:
    gen_embedding_forward_split_pt2_cuda_wrapper.cpp
    gen_embedding_backward_split_{optimizer}_pt2_cuda_wrapper.cpp

    [PT2 Autograd] --Torch dispatch-->                   |
        [PT2 wrapper] --Torch dispatch--> [CUDA backend] | <<<
            --kernel dispatch--> [CUDA kernel]           |
*/#}
{%- if has_gpu_support %}
////////////////////////////////////////////////////////////////////////////////
// Required for op registrations and dispatchers
#include "fbgemm_gpu/embedding_op_registration.h"
#include <torch/script.h>
#include "fbgemm_gpu/dispatch_macros.h"
////////////////////////////////////////////////////////////////////////////////
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"



using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{%- for vbe in ([True, False] if has_vbe_support else [False]) %}
{%- set vdesc = "_vbe" if vbe else "" %}

{%- for weighted in [True, False] %}
{%- for nobag in ([False] if (weighted or vbe) else [True, False]) %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}
{%- set ndesc = "_nobag" if nobag else "" %}

{%- if is_forward %}
{#-/* PT2 wrapper function for forward CUDA */#}
Tensor split_embedding{{ ndesc }}_codegen_forward_{{ wdesc }}{{ vdesc }}_pt2_cuda_wrapper(
    const Tensor& /*host_weights*/,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    {%- if nobag %}
    const c10::SymInt D,
    {%- else %}
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    {%- endif %}
    const Tensor& /*hash_size_cumsum*/,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    const Tensor& indice_weights, // CPU always takes indice_weights
    {%- endif %}
    const Tensor& lxu_cache_locations,
    const Tensor& uvm_cache_stats,
    {%- if vbe %}
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const c10::SymInt vbe_output_size,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    {%- endif %}
    const bool is_experimental,
    const int64_t output_dtype
    ){
    {%- set op = "split_embedding{}_codegen_forward_{}{}_cuda".format(
        ndesc, wdesc, vdesc
    )
    %}
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ op }}", "")
            .typed<Tensor(
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                {%- if not nobag %}
                const Tensor&,
                {%- else %}
                const c10::SymInt,
                {%- endif %}
                {%- if not nobag %}
                const c10::SymInt,
                const c10::SymInt,
                {%- endif %}
                const Tensor&,
                const Tensor&,
                {%- if not nobag %}
                const int64_t,
                {%- endif %}
                {%- if weighted %}
                const Tensor&,
                {%- endif %}
                const Tensor&,
                const Tensor&,
                const int64_t,
                {%- if vbe %}
                const Tensor&,
                const Tensor&,
                c10::SymInt,
                const int64_t, // int32_t
                const int64_t, // uint32_t
                {%- endif %}
                const bool
            )>();

    return op.call(
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            {%- if nobag %}
            D,
            {%- else %}
            D_offsets,
            total_D,
            max_D,
            {%- endif %}
            indices,
            offsets,
            {%- if not nobag %}
            pooling_mode,
            {%- endif %}
            {%- if weighted %}
            indice_weights,
            {%- endif %}
            lxu_cache_locations,
            uvm_cache_stats,
            output_dtype,
            {%- if vbe %}
            vbe_row_output_offsets,
            vbe_b_t_map,
            vbe_output_size,
            info_B_num_bits,
            info_B_mask_int64,
            {%- endif %}
            is_experimental
        );
    };

{#-/* PT2 wrapper function for forward META */#}
Tensor split_embedding{{ ndesc }}_codegen_forward_{{ wdesc }}{{ vdesc }}_pt2_meta_wrapper(
    const Tensor& /*host_weights*/,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    {%- if nobag %}
    const c10::SymInt D,
    {%- else %}
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    {%- endif %}
    const Tensor& /*hash_size_cumsum*/,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    const Tensor& indice_weights, // CPU always takes indice_weights
    {%- endif %}
    const Tensor& lxu_cache_locations,
    const Tensor& uvm_cache_stats,
    {%- if vbe %}
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const c10::SymInt vbe_output_size,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    {%- endif %}
    const bool is_experimental,
    const int64_t output_dtype
    ){
    {%- set op = "split_embedding{}_codegen_forward_{}{}_cuda".format(
            ndesc, wdesc, vdesc
        )
    %}
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ op }}", "")
            .typed<Tensor(
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                {%- if not nobag %}
                const Tensor&,
                {%- else %}
                const c10::SymInt,
                {%- endif %}
                {%- if not nobag %}
                const c10::SymInt,
                {%- endif %}
                {%- if not nobag %}
                const c10::SymInt,
                {%- endif %}
                const Tensor&,
                const Tensor&,
                {%- if not nobag %}
                const int64_t,
                {%- endif %}
                {%- if weighted %}
                const Tensor&,
                {%- endif %}
                const Tensor&,
                const Tensor&,
                const int64_t,
                {%- if vbe %}
                const Tensor&,
                const Tensor&,
                c10::SymInt,
                const int64_t,
                const int64_t,
                {%- endif %}
                const bool
            )>();

    return op.call(
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            {%- if not nobag %}
            D_offsets,
            {%- else %}
            D,
            {%- endif %}
            {%- if not nobag %}
            total_D,
            {%- endif %}
            {%- if not nobag %}
            max_D,
            {%- endif %}
            indices,
            offsets,
            {%- if not nobag %}
            pooling_mode,
            {%- endif %}
            {%- if weighted %}
            indice_weights,
            {%- endif %}
            lxu_cache_locations,
            uvm_cache_stats,
            output_dtype,
            {%- if vbe %}
            vbe_row_output_offsets,
            vbe_b_t_map,
            vbe_output_size,
            info_B_num_bits, // int32_t
            info_B_mask_int64, // uint32_t
            {%- endif %}
            is_experimental
        );
    }
{%- else %}

{#-/* PT2 wrapper function for backward CUDA */#}
Tensor split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_exact{{ vdesc }}_pt2_cuda_wrapper(
    const Tensor& grad_output,
    const Tensor& /*host_weights*/,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    {%- if nobag %}
    const c10::SymInt D,
    {%- else %}
    const Tensor& D_offsets,
    const c10::SymInt max_D,
    {%- endif %}
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    const Tensor& indice_weights, // currently supports no bag with unweighted
    {%- endif %}
    const Tensor& lxu_cache_locations,
    const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp,
    {%- if optimizer != "none" %}
    const bool stochastic_rounding,
    {%- endif %}
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    {%- if vbe %}
    const Tensor& B_offsets,
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    {%- endif %}
    const bool use_uniq_cache_locations,
    const bool use_homogeneous_placements,
    {{ args_pt2.split_function_args | join(", ") }}
    {%- if not nobag %}
    , const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)
    {%- endif %}){
        {%- set backward_op = "split_embedding{}_backward_codegen_{}_{}_exact{}_cuda".format(
                ndesc, optimizer, wdesc, vdesc
            )
        %}
        static auto op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::{{ backward_op }}", "")
                .typed<Tensor(
                        const Tensor&,
                        const Tensor&,
                        const Tensor&,
                        const Tensor&,
                        const Tensor&,
                        const Tensor&,
                        {%- if nobag %}
                        const c10::SymInt,
                        {%- else %}
                        const Tensor&,
                        const c10::SymInt,
                        {%- endif %}
                        const Tensor&,
                        const int64_t,
                        const Tensor&,
                        const Tensor&,
                        {%- if not nobag %}
                        const int64_t,
                        {%- endif %}
                        {%- if weighted %}
                        const Tensor&,
                        {%- endif %}
                        const Tensor&,
                        const int64_t,
                        const int64_t,
                        {%- if optimizer != "none" %}
                        const bool,
                        {%- endif %}
                        const int64_t,
                        const int64_t,
                        {%- if vbe %}
                        const Tensor&,
                        const Tensor&,
                        const Tensor&,
                        {%- endif %}
                        const bool,
                        const bool,
                        {%- for arg_type in args.split_function_args %}
                        {{ arg_type.split(' ')[0]}}{%- if not loop.last %}{{ "," }}{%- endif %}
                        {%- endfor %}
                )>();

        return op.call(
            grad_output,
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            {%- if nobag %}
            D,
            {%- else %}
            D_offsets,
            max_D,
            {%- endif %}
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            {%- if not nobag %}
            pooling_mode,
            {%- endif %}
            {%- if weighted %}
            indice_weights,
            {%- endif %}
            lxu_cache_locations,
            BT_block_size,
            max_segment_length_per_warp,
            {%- if optimizer != "none" %}
            stochastic_rounding,
            {%- endif %}
            info_B_num_bits,
            info_B_mask_int64,
            {%- if vbe %}
            B_offsets,
            vbe_row_output_offsets,
            vbe_b_t_map,
            {%- endif %}
            use_uniq_cache_locations,
            use_homogeneous_placements,
            {{ args.split_function_arg_names | join(", ") }}
        );
    }
{%- endif %}
{%- endfor %} {#-/*for nobag*/#}
{%- endfor %} {#-/*for weighted*/#}

{%- if is_forward %}
{#-/* PT2 wrapper function for backward grad_indice_weights CUDA */#}
Tensor split_embedding_codegen_grad_indice_weights{{ vdesc }}_pt2_cuda_wrapper(
    const Tensor& grad_output,
    const Tensor& /*host_weights*/,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt max_D,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& lxu_cache_locations,
    {%- if vbe %}
    const Tensor& feature_requires_grad,
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64
    {%- else %}
    const Tensor& feature_requires_grad
    {%- endif %}
){
    {%- set op = "split_embedding_codegen_grad_indice_weights{}_cuda".format(
            vdesc
        )
    %}
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ op }}", "")
            .typed<Tensor(
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const c10::SymInt,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                {%- if vbe %}
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const int64_t,
                const int64_t
                {%- else %}
                const Tensor&
                {%- endif %}
            )>();

    return op.call(
        grad_output,
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D_offsets,
        max_D,
        indices,
        offsets,
        lxu_cache_locations,
        {%- if vbe %}
        feature_requires_grad,
        vbe_row_output_offsets,
        vbe_b_t_map,
        info_B_num_bits,
        info_B_mask_int64
        {%- else %}
        feature_requires_grad
        {%- endif %}
        );
}
{%- endif %}
////////////////////////////////////////////////////////////////////////////////
// Op registrations
////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {

    {%- for weighted in [True, False] %}
    {%- for nobag in ([False] if (weighted or vbe) else [True, False]) %}
    {%- set wdesc = "weighted" if weighted else "unweighted" %}
    {%- set ndesc = "_nobag" if nobag else "" %}
    {%- if is_forward %}
    {%- set embedding_codegen_forward_op = "split_embedding{}_codegen_forward_{}{}_pt2".format(
      ndesc, wdesc, vdesc
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
        "    Tensor uvm_cache_stats, "
        {%- if vbe %}
        "    Tensor vbe_row_output_offsets, "
        "    Tensor vbe_b_t_map, "
        "    SymInt vbe_output_size, "
        "    int info_B_num_bits, "
        "    int info_B_mask_int64, "
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

    DISPATCH_TO_CUDA(
      "{{ embedding_codegen_forward_op }}_wrapper",
      {{ embedding_codegen_forward_op }}_cuda_wrapper
    );
    m.impl("{{ embedding_codegen_forward_op }}_wrapper", torch::dispatch(c10::DispatchKey::Meta, TORCH_FN({{ embedding_codegen_forward_op }}_meta_wrapper)));
    {%- else %}
    {%- set embedding_codegen_backward_op = "split_embedding{}_backward_codegen_{}_{}_exact{}_pt2".format(
        ndesc, optimizer, wdesc, vdesc
        )
    %}
    m.def("{{ embedding_codegen_backward_op }}_wrapper("
        "    Tensor grad_output, "
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
        "    SymInt max_D, "
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
        {%- endif %}
        "    bool use_uniq_cache_locations, "
        "    bool use_homogeneous_placements,"
        "    {{ args_pt2.split_function_schemas | join(", ") }} "
        {%- if not nobag %}
        "    , int output_dtype=0 "
        {%- endif %}
        ") -> Tensor");
    DISPATCH_TO_CUDA(
        "{{ embedding_codegen_backward_op }}_wrapper",
        {{ embedding_codegen_backward_op }}_cuda_wrapper
    );
    {%- endif %}
    {%- endfor %} {#-/*for nobag*/#}
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
        "    int info_B_mask_int64"
        {%- else %}
        "    Tensor feature_requires_grad"
        {%- endif %}
        ") -> Tensor");

    DISPATCH_TO_CUDA(
        "{{ embedding_codegen_grad_indice_weights_op }}_wrapper",
        {{ embedding_codegen_grad_indice_weights_op }}_cuda_wrapper
    );
    {%- endif %}

}
{%- endfor %} {#-/* for vbe in [True, False] */#}

{%- endif %} {#/* if has_gpu_support */#}
// clang-format on
