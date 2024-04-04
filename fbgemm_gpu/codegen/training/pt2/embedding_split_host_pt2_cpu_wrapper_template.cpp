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
#include "fbgemm_gpu/embedding_op_registration.h"
#include <torch/script.h>
#include "fbgemm_gpu/dispatch_macros.h"
////////////////////////////////////////////////////////////////////////////////
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{%- if is_forward %}
{#-/* PT2 wrapper function for backward grad_indice_weights CPU */#}
Tensor split_embedding_codegen_grad_indice_weights_pt2_cpu_wrapper(
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
    const Tensor& feature_requires_grad) {
  static auto op =
      torch::Dispatcher::singleton()
        .findSchemaOrThrow(
              "fbgemm::split_embedding_codegen_grad_indice_weights_cpu", "")
        .typed<Tensor(Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor)>();
  return op.call(
      grad_output,
      host_weights,
      weights_offsets,
      D_offsets,
      indices,
      offsets,
      feature_requires_grad);
}
{%- else %}
{%- endif %}
{%- for weighted in [True, False] %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}

{% if is_forward %}
{#-/* PT2 wrapper function for forward CPU */#}
Tensor split_embedding_codegen_forward_{{ wdesc }}_pt2_cpu_wrapper(
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
    const bool /*is_experimental = false*/,
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)) {
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::split_embedding_codegen_forward_cpu", "")
          .typed<Tensor(
                Tensor, Tensor, Tensor, c10::SymInt, Tensor, Tensor, Tensor, int64_t, Tensor, int64_t
          )>();

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
}
{% else %}
{#-/* PT2 wrapper function for backward CPU */#}
Tensor split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_exact_pt2_cpu_wrapper(
    const Tensor& grad_output,
    const Tensor& host_weights,
    const Tensor& /*dev_weights*/,
    const Tensor& /*uvm_weights*/,
    const Tensor& /*lxu_cache_weights*/,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const int64_t max_D,
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
    const int64_t /*info_B_num_bits*/,
    const int64_t /*info_B_mask_int64*/,
    const bool /*use_uniq_cache_locations*/,
    const bool /*use_homogeneous_placements*/,
    {{ args_pt2.split_function_args | join(", ") }}
    {%- if not nobag %}
    , const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)
    {%- endif %})
    {
        {%- set backward_op = "split_embedding_backward_codegen_{}_cpu".format(
                optimizer
            )
        %}
        static auto op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::{{ backward_op }}", "")
                .typed<void(
                    Tensor,
                    Tensor,
                    Tensor,
                    Tensor,
                    Tensor,
                    int64_t,
                    Tensor,
                    int64_t,
                    Tensor,
                    Tensor,
                    int64_t,
                    Tensor,
                    bool,
                    {%- for arg_type in args.split_function_args %}
                    {{ arg_type.split(' ')[0]}}{%- if not loop.last %}{{ "," }}{%- endif %}
                    {%- endfor %}
                    {%- if not nobag %}
                    , int64_t
                    {%- endif %}
                )>();

        op.call(
            grad_output,
            host_weights,
            weights_placements,
            weights_offsets,
            D_offsets,
            max_D,
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            pooling_mode,
            indice_weights,
            stochastic_rounding,
            {{ args.split_function_arg_names | join(", ") }}
            {%- if not nobag %}
            , output_dtype
            {%- endif %}
            );
        return grad_output;
    }
{% endif %}
{%- endfor %} {#-/*for weighted*/#}


namespace {
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    {%- if is_forward %}
    DISPATCH_TO_CPU(
        "split_embedding_codegen_grad_indice_weights_pt2_wrapper",
        split_embedding_codegen_grad_indice_weights_pt2_cpu_wrapper);
    {%- endif %}

    {%- for weighted in [True, False] %}
    {%- set wdesc = "weighted" if weighted else "unweighted" %}
    {%- if is_forward %}
    {%- set embedding_codegen_forward_op = "split_embedding_codegen_forward_{}_pt2".format(
        wdesc
        )
    %}
    DISPATCH_TO_CPU("{{ embedding_codegen_forward_op }}_wrapper", {{ embedding_codegen_forward_op }}_cpu_wrapper);
    {%- else %}

    {%- set embedding_codegen_backward_op = "split_embedding_backward_codegen_{}_{}_exact_pt2".format(
        optimizer, wdesc
        )
    %}
    DISPATCH_TO_CPU("{{ embedding_codegen_backward_op }}_wrapper", {{ embedding_codegen_backward_op }}_cpu_wrapper);
    {%- endif %}
    {%- endfor %} {#-/*for weighted*/#}
}

} // namespace
{% endif %} // if has_cpu_support
// clang-format on
