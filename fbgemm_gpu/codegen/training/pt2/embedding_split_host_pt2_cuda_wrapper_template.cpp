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
////////////////////////////////////////////////////////////////////////////////
#include "fbgemm_gpu/utils/ops_utils.h"
#include <torch/script.h>
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{#/* Module description */#}
{%- set fwd_mdesc = "ssd" if ssd else ("dense" if dense else "split") %}
{%- set bwd_mdesc = "ssd" if ssd else "split" %}

{%- if ssd %}
enum SSDTensor {
  {%- for tensor in ssd_tensors %}
  {{ tensor | upper }} = {{ loop.index - 1 }},
  {%- endfor %}
};
{%- endif %}

{%- for vbe in ([True, False] if has_vbe_support else [False]) %}
{%- set vdesc = "_vbe" if vbe else "" %}

{%- for dispatch_type in ["cuda", "meta"] %}
{%- for weighted in [True, False] %}
{%- for nobag in ([False] if (weighted or vbe) else [True, False]) %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}
{%- set ndesc = "_nobag" if nobag else "" %}
{%- for is_gwd in ([True, False]
    if is_valid_gwd_config(
      dense,
      nobag,
      vbe,
      is_index_select,
      True,
      ssd)
      else [False]) %}
{%- set gwddesc = "_gwd" if is_gwd else "" %}
{%- set desc_suffix = wdesc + vdesc + gwddesc %}

{#-/* PT2 wrapper function for forward CUDA */#}
{%- if is_forward %}
Tensor {{ fwd_mdesc }}_embedding{{ ndesc }}_codegen_forward_{{ desc_suffix }}_pt2_{{ dispatch_type }}_wrapper(
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
    const Tensor& hash_size_cumsum,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    const Tensor& indice_weights, // CPU always takes indice_weights
    {%- endif %}
    const Tensor& {{ "ssd_row_addrs" if ssd else "lxu_cache_locations" }},
    const Tensor& uvm_cache_stats,
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
    {%- if is_gwd %}
    const Tensor& prev_iter_dev,
    const Tensor& learning_rate_tensor,
    const double weight_decay,
    const int64_t iter,
    const double gwd_lower_bound,
    {%- endif %}
    const bool is_experimental,
    const int64_t output_dtype
    ){
    {%- set op = "{}_embedding{}_codegen_forward_{}_cuda".format(
        fwd_mdesc, ndesc, desc_suffix
    )
    %}
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ op }}", "")
            .typed<Tensor(
                const Tensor& /*host_weights*/,
                const Tensor& /*dev_weights*/,
                const Tensor& /*uvm_weights*/,
                const Tensor& /*lxu_cache_weights*/,
                const Tensor& /*weights_placements*/,
                {%- if not nobag %}
                const Tensor& /*D_offsets*/,
                const c10::SymInt /*total_D*/,
                const c10::SymInt /*max_D*/,
                {%- else %}
                const c10::SymInt /*D*/,
                {%- endif %}
                const Tensor& /*indices*/,
                const Tensor& /*offsets*/,
                {%- if not nobag %}
                const int64_t /*pooling_mode*/,
                {%- endif %}
                {%- if weighted %}
                const Tensor& /*indice_weights*/,
                {%- endif %}
                const Tensor& /*row_addrs or lxu_cache_locations*/,
                const Tensor& /*uvm_cache_stats_*/,
                const int64_t /*output_dtype*/,
                {%- if vbe %}
                const Tensor& /*vbe_row_output_offsets*/,
                const Tensor& /*vbe_b_t_map*/,
                c10::SymInt /*vbe_output_size*/,
                const int64_t /*info_B_num_bits*/, // int32_t
                const int64_t /*info_B_num_bits*/, // uint32_t
                {%- endif %}
                {%- if is_gwd %}
                const Tensor& /*hash_size_cumsum*/,
                const Tensor& /*prev_iter_dev*/,
                const Tensor& /*learning_rate_tensor*/,
                const double /*weight_decay*/,
                const int64_t /*iter*/,
                const double /*gwd_lower_bound*/,
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
            {%- if ssd %}
            ssd_row_addrs,
            {%- else %}
            lxu_cache_locations,
            {%- endif %}
            uvm_cache_stats,
            output_dtype,
            {%- if vbe %}
            vbe_row_output_offsets,
            vbe_b_t_map,
            vbe_output_size,
            info_B_num_bits,
            info_B_mask_int64,
            {%- endif %}
            {%- if is_gwd %}
            hash_size_cumsum,
            prev_iter_dev,
            learning_rate_tensor,
            weight_decay,
            iter,
            gwd_lower_bound,
            {%- endif %} {# /* if is_gwd */ #}
            is_experimental
        );
    };
{%- else %}

{#-/* PT2 wrapper function for backward CUDA */#}
Tensor {{ bwd_mdesc }}_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ desc_suffix }}_pt2_{{ dispatch_type }}_wrapper(
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
    const bool mixed_D,
    {%- endif %}
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    const Tensor& indice_weights, // currently supports no bag with unweighted
    {%- endif %}
    {%- if ssd %}
    const Tensor& ssd_row_addrs,
    {%- elif not dense %}
    const Tensor& lxu_cache_locations,
    {%- endif %}
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
    const Tensor& vbe_B_offsets_rank_per_feature,
    const c10::SymInt max_B,
    {%- endif %}
    const bool use_uniq_cache_locations,
    const bool use_homogeneous_placements,
    {%- if is_gwd %}
    {%- if "prev_iter_dev" not in args.split_function_arg_names %}
    const Tensor& prev_iter_dev,
    {%- endif %}
    {%- if "iter" not in args.split_function_arg_names %}
    const int64_t iter,
    {%- endif %}
    const double gwd_lower_bound,
    {%- endif %}
    {{ args_pt2.split_function_args | join(", ") }}
    {%- if not nobag %}
    , const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)
    {%- endif %}){
        {%- set backward_op = "{}_embedding{}_backward_codegen_{}_{}_exact_cuda".format(
                bwd_mdesc, ndesc, optimizer, desc_suffix
            )
        %}
        static auto op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::{{ backward_op }}", "")
                .typed<Tensor(
                        const Tensor& /*grad_output*/,
                        const Tensor& /*dev_weights*/,
                        const Tensor& /*uvm_weights*/,
                        const Tensor& /*lxu_cache_weights*/,
                        const Tensor& /*weights_placements*/,
                        const Tensor& /*weights_offsets*/,
                        {%- if nobag %}
                        const c10::SymInt /*D*/,
                        {%- else %}
                        const Tensor& /*D_offsets*/,
                        const c10::SymInt /*max_D*/,
                        const bool /*mixed_D*/,
                        {%- endif %}
                        const Tensor& /*hash_size_cumsum*/,
                        const int64_t /*total_hash_size_bits*/,
                        const Tensor& /*indices*/,
                        const Tensor& /*offsets*/,
                        {%- if not nobag %}
                        const int64_t /*pooling_mode*/,
                        {%- endif %}
                        {%- if weighted %}
                        const Tensor& /*indice_weights*/,
                        {%- endif %}
                        const Tensor& /*ssd_row_addrs or lxu_cache_locations*/,
                        const int64_t /*BT_block_size*/,
                        const int64_t /*max_segment_length_per_warp*/,
                        {%- if optimizer != "none" %}
                        const bool /*stochastic_rounding*/,
                        {%- endif %}
                        const int64_t /*info_B_num_bits*/,
                        const int64_t /*info_B_mask_int64*/,
                        {%- if vbe %}
                        const Tensor& /*B_offsets*/,
                        const Tensor& /*vbe_row_output_offsets*/,
                        const Tensor& /*vbe_b_t_map*/,
                        {%- endif %}
                        const bool /*use_uniq_cache_locations*/,
                        const bool /*use_homogeneous_placements*/,
                        {%- if is_gwd %}
                        {%- if "prev_iter_dev" not in args.split_function_arg_names %}
                        const Tensor& /*prev_iter_dev*/,
                        {%- endif %}
                        {%- if "iter" not in args.split_function_arg_names %}
                        const int64_t /*iter*/,
                        {%- endif %}
                        const double /*gwd_lower_bound*/,
                        {%- endif %}
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
            mixed_D,
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
            {%- if ssd %}
            ssd_row_addrs,
            {%- else %}
            lxu_cache_locations,
            {%- endif %}
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
            {%- if is_gwd %}
            {%- if "prev_iter_dev" not in args.split_function_arg_names %}
            prev_iter_dev,
            {%- endif %}
            {%- if "iter" not in args.split_function_arg_names %}
            iter,
            {%- endif %}
            gwd_lower_bound,
            {%- endif %}
            {{ args.split_function_arg_names | join(", ") }}
        );
    }

{%- endif %}
{%- endfor %} {#-/*for is_gwd*/#}
{%- endfor %} {#-/*for nobag*/#}
{%- endfor %} {#-/*for weighted*/#}


{%- if is_forward %}
{#-/* PT2 wrapper function for backward grad_indice_weights CUDA */#}
Tensor {{ fwd_mdesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_pt2_{{ dispatch_type }}_wrapper(
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
    {%- if ssd %}
    const Tensor& ssd_row_addrs,
    {%- else %}
    const Tensor& lxu_cache_locations,
    {%- endif %}
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
){
    {%- set op = "{}_embedding_codegen_grad_indice_weights{}_cuda".format(
            fwd_mdesc, vdesc
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
        {%- if ssd %}
        ssd_row_addrs,
        {%- else %}
        lxu_cache_locations,
        {%- endif %}
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
{%- endfor %} {#-/*for dispatch_type*/#}

////////////////////////////////////////////////////////////////////////////////
// Op registrations
////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    {%- for weighted in [True, False] %}
    {%- for nobag in ([False] if (weighted or vbe) else [True, False]) %}
    {%- set wdesc = "weighted" if weighted else "unweighted" %}
    {%- set ndesc = "_nobag" if nobag else "" %}
    {%- for is_gwd in ([True, False]
    if is_valid_gwd_config(
      dense,
      nobag,
      vbe,
      is_index_select,
      True,
      ssd)
      else [False]) %}
    {%- set gwddesc = "_gwd" if is_gwd else "" %}
    {%- set desc_suffix = wdesc + vdesc + gwddesc %}

    {%- if is_forward %}
    {%- set embedding_codegen_forward_op = "{}_embedding{}_codegen_forward_{}_pt2".format(
      fwd_mdesc, ndesc, desc_suffix
      )
    %}
    {%- if ssd or is_gwd or nobag %}
    /* Register scehema for wrappers with GPU-only support */
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
        {%- if ssd %}
        "    Tensor ssd_row_addrs, "
        {%- else %}
        "    Tensor lxu_cache_locations, "
        {%- endif %}
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
        {%- if is_gwd %}
        "    Tensor{{ schema_annotation['prev_iter_dev'] }} prev_iter_dev, "
        "    Tensor learning_rate_tensor, "
        "    float weight_decay, "
        "    int iter, "
        "    float gwd_lower_bound, "
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
    {%- endif %}
    DISPATCH_TO_CUDA(
      "{{ embedding_codegen_forward_op }}_wrapper",
      {{ embedding_codegen_forward_op }}_cuda_wrapper
    );
    m.impl("{{ embedding_codegen_forward_op }}_wrapper", torch::dispatch(c10::DispatchKey::Meta, TORCH_FN({{ embedding_codegen_forward_op }}_meta_wrapper)));
    
    {%- else %} {#-/* backward */#}
    {%- set embedding_codegen_backward_op = "{}_embedding{}_backward_codegen_{}_{}_pt2".format(
        bwd_mdesc, ndesc, optimizer, desc_suffix
        )
    %}
    {%- if ssd or is_gwd or nobag or not has_cpu_support %}
    /* Register scehema for wrappers with GPU-only support */
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
        {%- if ssd %}
        "    Tensor ssd_row_addrs, "
        {%- else %}
        "    Tensor lxu_cache_locations, "
        {%- endif %}
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
        {%- if is_gwd %}
        {%- if "prev_iter_dev" not in args.split_function_arg_names %}
        "    Tensor{{ schema_annotation['prev_iter_dev'] }} prev_iter_dev, "
        {%- endif %}
        {%- if "iter" not in args.split_function_arg_names %}
        "    int iter, "
        {%- endif %}
        "    float gwd_lower_bound, "
        {%- endif %}
        "    {{ args_pt2.split_function_schemas | join(", ") }} "
        {%- if not nobag %}
        "    , int output_dtype=0 "
        {%- endif %}
        ") -> Tensor");
    {%- endif %}
    DISPATCH_TO_CUDA(
        "{{ embedding_codegen_backward_op }}_wrapper",
        {{ embedding_codegen_backward_op }}_cuda_wrapper
    );
    m.impl("{{ embedding_codegen_backward_op }}_wrapper", torch::dispatch(c10::DispatchKey::Meta, TORCH_FN({{ embedding_codegen_backward_op }}_meta_wrapper)));
    {%- endif %} {#-/* if is_forward */#}
    {%- endfor %} {#-/*for is_gwd*/#}
    {%- endfor %} {#-/*for nobag*/#}
    {%- endfor %} {#-/*for weighted*/#}
    {%- if is_forward %}
    {%- set embedding_codegen_grad_indice_weights_op =
        "{}_embedding_codegen_grad_indice_weights{}_pt2".format(
            fwd_mdesc, vdesc
        )
    %}
    {%- if ssd %}
    /* Register scehema for wrappers with GPU-only support */
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
        {%- if ssd %}
        "    Tensor ssd_row_addrs, "
        {%- else %}
        "    Tensor lxu_cache_locations, "
        {%- endif %}
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
    {%- endif %}
    DISPATCH_TO_CUDA(
        "{{ embedding_codegen_grad_indice_weights_op }}_wrapper",
        {{ embedding_codegen_grad_indice_weights_op }}_cuda_wrapper
    );
    m.impl("{{ embedding_codegen_grad_indice_weights_op }}_wrapper", torch::dispatch(c10::DispatchKey::Meta, TORCH_FN({{ embedding_codegen_grad_indice_weights_op }}_meta_wrapper)));
    {%- endif %}

}
{%- endfor %} {#-/* for vbe in [True, False] */#}

{%- endif %} {#/* if has_gpu_support */#}
// clang-format on
