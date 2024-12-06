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

// Companion template is embedding_backward_split_template.cu

{%- set mdesc = "ssd" if ssd else "split" %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}
{%- set vdesc = "_vbe" if vbe else "" %}
{%- set ndesc = "_nobag" if nobag else "" %}

{%- set locs_or_addrs_tensor = "ssd_row_addrs" if ssd else "lxu_cache_locations" %}

////////////////////////////////////////////////////////////////////////////////
// Required for op registrations
////////////////////////////////////////////////////////////////////////////////
#include <torch/library.h>
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/utils/tensor_utils.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using namespace fbgemm_gpu;
using Tensor = at::Tensor;

////////////////////////////////////////////////////////////////////////////////
// Kernel Definitions
////////////////////////////////////////////////////////////////////////////////

{%- for is_gwd in ([True, False]
    if is_valid_gwd_config(
        dense,
        nobag,
        vbe,
        is_index_select,
        has_global_weight_decay_support,
        ssd=False
    ) else [False])
%}

{%- set gwddesc = "_gwd" if is_gwd else "" %}
{%- set desc_suffix = wdesc + vdesc + gwddesc %}

{%- if is_index_select %}
Tensor batch_index_select_dim0_codegen_backward_meta(
{%- else %}
Tensor {{ mdesc }}_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ desc_suffix }}_meta(
{%- endif %}
    const Tensor& grad_output,
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    {%- if not nobag or is_index_select %}
    const Tensor& D_offsets,
    const c10::SymInt max_D,
    {%- else %}
    const c10::SymInt D,
    {%- endif %}
    {%- if not nobag and not is_index_select %}
    const bool mixed_D,
    {%- endif %}
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    {%- if not is_index_select %}
    const Tensor& offsets,
    {%- endif %}
    {%- if not nobag %}
    const int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    const Tensor& indice_weights,
    {%- endif %}
    {%- if not dense %}
    const Tensor& {{ locs_or_addrs_tensor }},
    {%- endif %}
    {%- if not is_index_select %}
    const int64_t unused_,
    {%- endif %}
    const int64_t max_segment_length_per_warp,
    {%- if not dense %}
    {%- if optimizer != "none" %}
    const bool stochastic_rounding,
    {%- endif %}
    const int64_t info_B_num_bits_int64, // int32_t
    const int64_t info_B_mask_int64, // uint32_t
    {%- endif %}
    {%- if vbe %}
    const Tensor& B_offsets,
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    {%- endif %}
    {%- if not is_index_select and not dense %}
    const bool use_uniq_cache_locations,
    const bool use_homogeneous_placements,
    {%- endif %}
    {%- if is_index_select %}
    const Tensor& grad_offsets,
    const Tensor& total_L_offsets,
    const int32_t fixed_L_per_warp,
    const int32_t num_warps_per_feature,
    const bool permute_output_dim_0_1
    {%- elif optimizer != "none" %}
    {%- if is_gwd %}
    const Tensor& prev_iter_dev,
    {%- if "iter" not in args.split_function_arg_names %}
    const int64_t iter,
    {%- endif %}
    const double gwd_lower_bound,
    {%- endif -%}
    {{ args.split_function_args_no_defaults | join(", ") }}
    {%- else %}
    // This is actually passed via args.split_function_args_no_defaults but explicitly list
    // it here for code readability
    int64_t total_hash_size,
    c10::SymInt total_unique_indices
    {%- endif %}
) {

    // NB: Should we have something for aligning memory like we do on the tensor kernels?
    // There it checks if the pointer to the tensor is not divisble by 16, along with
    // some stride checks.

    {%- if nobag and not is_index_select %}
    auto max_D = D;
    {%- endif %}
    {%- if not is_index_select %}
    TORCH_SYM_CHECK(max_D.sym_le({{ max_embedding_dim }}), "");
    {%- endif %}

    {%- if optimizer == "none" %}
    // grad_dev_weights has emb_t type
    auto grad_dev_weights = at::empty_symint({total_unique_indices * max_D}, dev_weights.options());
    {%- else %}
    // Set total_unique_indices to total num indices by default
    const auto total_unique_indices = indices.sym_numel();
    {%- if dense %}
    auto grad_dev_weights = at::zeros_like(dev_weights);
    {%- endif %}
    {%- endif %}

    // short-circuit if there are zero indices.
    if (TORCH_GUARD_SIZE_OBLIVIOUS(indices.sym_numel().sym_eq(0))) {
        {%- if dense %}
        return grad_dev_weights;
        {%- elif optimizer == "none" %}
        return at::_sparse_coo_tensor_unsafe_symint(
            at::empty_symint({1, 0}, indices.options()),
            grad_dev_weights.reshape_symint({0, max_D}),
            {total_hash_size, max_D},
            dev_weights.options().layout(at::kSparse)
        );
        {%- else %}
        return Tensor();
        {%- endif %}
    }

    {%- if not nobag %}
    auto T = D_offsets.sym_numel() - 1;
    {%- else %}
    auto T = weights_offsets.sym_numel();
    {%- endif %}

    TORCH_SYM_CHECK(T.sym_gt(0), "");
    // offsets = [B x T  + 1]
    {%- if is_index_select %}
    const auto total_B = num_warps_per_feature * T;
    {%- else %}
    const auto total_B = offsets.sym_size(0) - 1;
    {%- endif %}
    TORCH_SYM_CHECK(total_B.sym_gt(0), "");

    {%- if vbe %}
    TORCH_SYM_CHECK(B_offsets.sym_numel().sym_eq(T + 1), "");
    TORCH_SYM_CHECK(vbe_row_output_offsets.sym_numel().sym_eq(total_B), "");
    TENSORS_HAVE_SAME_SYM_NUMEL(vbe_row_output_offsets, vbe_b_t_map);
    {%- endif %}

    {%- if dense %}

    auto max_B = total_B / T;
    int32_t info_B_num_bits = 22;
    uint32_t info_B_mask = (1u << info_B_num_bits) - 1;
    if (!max_B.is_symbolic()) {
        std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(max_B.guard_int(__FILE__, __LINE__), T.guard_int(__FILE__, __LINE__));
    }

    {%- endif %}

    {%- if dense %}
    return grad_dev_weights;

    {%- elif optimizer == "none" %}

    // Took allocation from https://www.internalfb.com/code/fbsource/fbcode/deeplearning/fbgemm/fbgemm_gpu/src/split_embeddings_utils.cu?lines=339-347
    Tensor sorted_linear_indices_run;
    if (TORCH_GUARD_SIZE_OBLIVIOUS(total_unique_indices.sym_gt(0))) {
        sorted_linear_indices_run = at::empty_symint({total_unique_indices}, indices.options());
    } else {
        sorted_linear_indices_run = at::empty_like(indices);
    }

    // originally this was sparse_coo_tensor
    return at::_sparse_coo_tensor_unsafe_symint(
        sorted_linear_indices_run.unsqueeze(0),
        grad_dev_weights.reshape_symint({total_unique_indices, max_D}),
        {total_hash_size, max_D},
        dev_weights.options().layout(at::kSparse));

    {%- else %}
    return Tensor();
    {%- endif %}
}

////////////////////////////////////////////////////////////////////////////////
// Op registrations
////////////////////////////////////////////////////////////////////////////////
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    // NB: yes cuda here
    {%- set embedding_codegen_backward_op =
        "{}_embedding{}_backward_codegen_{}_{}_exact_cuda".format(
            mdesc, ndesc, optimizer, desc_suffix
        )
    %}
    m.impl("{{ embedding_codegen_backward_op }}", torch::dispatch(c10::DispatchKey::Meta, TORCH_FN({{ mdesc }}_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ desc_suffix }}_meta)));
    {%- if is_index_select %}
    m.impl("batch_index_select_dim0_codegen_backward_cuda", torch::dispatch(c10::DispatchKey::Meta, TORCH_FN(batch_index_select_dim0_codegen_backward_meta)));
    {%- endif %}
}
{%- endfor %}
// clang-format on
