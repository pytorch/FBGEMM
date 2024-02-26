/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

#include "codegen/embedding_forward_split_cpu.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{% if has_cpu_support %}
/// @defgroup embedding-cpu Embedding CPU Operators

void split_embedding_backward_codegen_{{ optimizer }}_cpu(
    Tensor grad_output,
    Tensor host_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }},
    int64_t output_dtype = static_cast<int64_t>(SparseType::FP32));

{%- for weighted in [True, False] %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}
Tensor split_embedding_codegen_forward_{{ wdesc }}_pt2_cpu(
    const Tensor& host_weights,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const int64_t total_D,
    const int64_t max_D,
    const Tensor& hash_size_cumsum,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t pooling_mode,
    const Tensor& indice_weights,
    const Tensor& lxu_cache_locations,
    const Tensor& uvm_cache_stats,
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const bool is_experimental = false);

Tensor split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_exact_pt2_cpu(
    const Tensor& grad_output,
    const Tensor& host_weights,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
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
    const Tensor& lxu_cache_locations,
    const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp,
    const bool stochastic_rounding,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    const bool use_uniq_cache_locations,
    const bool use_homogeneous_placements,
    {{ args_pt2.split_function_args | join(", ") }},
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32))
    {
        split_embedding_backward_codegen_{{ optimizer }}_cpu(
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
        {{ args.split_function_arg_names | join(", ") }},
        output_dtype);
        return grad_output;
    }
{%- endfor %} {#-/*for weighted*/#}

Tensor split_embedding_codegen_grad_indice_weights_pt2_cpu(
    const Tensor& grad_output,
    const Tensor& host_weights,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const int64_t max_D,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& lxu_cache_locations,
    const Tensor& feature_requires_grad
);

namespace {
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    {%- for weighted in [True, False] %}
        {%- set wdesc = "weighted" if weighted else "unweighted" %}
        {%- set embedding_codegen_backward_op = "split_embedding_backward_codegen_{}_{}_exact_pt2".format(
            optimizer, wdesc
            )
        %}
        DISPATCH_TO_CPU(
            "{{ embedding_codegen_backward_op }}",
            {{ embedding_codegen_backward_op }}_cpu
        );
    {%- endfor %} {#-/*for weighted*/#}
}

} // namespace
{% endif %} // if has_cpu_support
// clang-format on
