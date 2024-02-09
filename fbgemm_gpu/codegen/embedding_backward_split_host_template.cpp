/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

#include "fbgemm_gpu/dispatch_macros.h"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

{%- if has_gpu_support %}
/// @defgroup embedding-cuda Embedding CUDA Operators

{%- for vbe in ([True, False] if has_vbe_support else [False]) %}
{%- set vdesc = "_vbe" if vbe else "" %}

{%- for weighted in [True, False] %}
{%- for nobag in ([False] if (weighted or vbe) else [True, False]) %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}
{%- set ndesc = "_nobag" if nobag else "" %}

Tensor split_embedding{{ ndesc }}_codegen_forward_{{ wdesc }}{{ vdesc }}_cuda(
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    {%- if nobag %}
    const int64_t D,
    {%- else %}
    const Tensor& D_offsets,
    const int64_t total_D,
    const int64_t max_D,
    {%- endif %}
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    const Tensor& indice_weights,
    {%- endif %}
    const Tensor& lxu_cache_locations,
    const Tensor& uvm_cache_stats,
    const int64_t output_dtype,
    {%- if vbe %}
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const int64_t vbe_output_size,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    {%- endif %}
    const bool is_experimental);

Tensor split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_exact{{ vdesc }}_cuda(
    const Tensor& grad_output,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    {%- if nobag %}
    const int64_t D,
    {%- else %}
    const Tensor& D_offsets,
    const int64_t max_D,
    {%- endif %}
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    const Tensor& indice_weights,
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
    {{ args.split_function_args | join(", ") }});

{%- endfor %} {#-/*for nobag*/#}
{%- endfor %} {#-/*for weighted*/#}

Tensor split_embedding_codegen_grad_indice_weights{{ vdesc }}_cuda(
    const Tensor& grad_output,
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
    {%- if vbe %}
    const Tensor& feature_requires_grad,
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64
    {%- else %}
    const Tensor& feature_requires_grad
    {%- endif %}
);

{%- for nobag in [True, False] %}
{%- if not nobag or not vbe %} {#-/* nobag does not support vbe */#}
{%- set autograd_func = "Split{}{}LookupFunction_{}_Op".format(
      "NoBag" if nobag else "",
      "VBE" if vbe else "",
      optimizer
    )
%}
class {{ autograd_func }} :
    public torch::autograd::Function<{{ autograd_func }}> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const Tensor& placeholder_autograd_tensor,
    const int64_t output_dtype,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    {%- if not nobag %}
    const Tensor& D_offsets,
    const int64_t total_D,
    const int64_t max_D,
    {%- else %}
    const int64_t D,
    {%- endif %}
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    const c10::optional<Tensor>& indice_weights,
    const c10::optional<Tensor>& feature_requires_grad,
    {%- endif %}
    const Tensor& lxu_cache_locations,
    c10::optional<Tensor> uvm_cache_stats,
    {%- if optimizer != "none" %}
    const bool gradient_clipping,
    const double max_gradient,
    const bool stochastic_rounding,
    {%- endif %}
    {%- if vbe %}
    const c10::optional<Tensor>& B_offsets,
    const c10::optional<Tensor>& vbe_output_offsets_feature_rank,
    const c10::optional<Tensor>& vbe_B_offsets_rank_per_feature,
    const int32_t max_B,
    const int32_t max_B_feature_rank,
    const int64_t vbe_output_size,
    {%- endif %}
    const bool is_experimental,
    const bool use_uniq_cache_locations_bwd,
    const bool use_homogeneous_placements,
    {{ args.split_function_args | join(", ") }}) {

    const auto T = weights_offsets.sym_numel();
    {%- if vbe %}
    const auto B_offsets_ = B_offsets.value_or(Tensor());
    const auto vbe_output_offsets_feature_rank_ = vbe_output_offsets_feature_rank.value_or(Tensor());
    const auto vbe_B_offsets_rank_per_feature_ = vbe_B_offsets_rank_per_feature.value_or(Tensor());

    const c10::SymInt max_B_ = max_B;
    {%- else %}
    const auto max_B_ = offsets.sym_size(0) / T;
    {%- endif %}

    // NOTE: The `local_uvm_cache_stats` variable held by the nn.Module has dtype int32_t
    // TODO: Hook up with frontend code
    const auto uvm_cache_stats_ = uvm_cache_stats
      .value_or(at::empty({0}, uvm_weights.options().dtype(at::kInt)));

    // TODO: don't guard here
    auto [info_B_num_bits, info_B_mask] = adjust_info_B_num_bits(max_B_.guard_int(__FILE__, __LINE__), T.guard_int(__FILE__, __LINE__));

    {%- if vbe %}
    static auto generate_vbe_metadata_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::generate_vbe_metadata", "")
            .typed<decltype(generate_vbe_metadata)>();

    auto [
        vbe_row_output_offsets,
        vbe_b_t_map
    ] = generate_vbe_metadata_op.call(
        B_offsets_,
        vbe_B_offsets_rank_per_feature_,
        vbe_output_offsets_feature_rank_,
        {%- if not nobag %}
        D_offsets,
        /*D=*/-1,
        /*nobag=*/false,
        {%- else %}
        // weights_placements has the same options as D_offsets
        at::empty({0}, weights_placements.options()),
        D,
        /*nobag=*/true,
        {%- endif %}
        max_B_feature_rank,
        info_B_num_bits,
        /*total_B=*/offsets.size(0) - 1
        );
    {%- endif %}

    ctx->save_for_backward({
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        {%- if not nobag %}
        D_offsets,
        {%- endif %}
        hash_size_cumsum,
        indices,
        offsets,
        {%- if not nobag %}
        indice_weights.value_or(Tensor()),
        feature_requires_grad.value_or(Tensor()),
        {%- endif %}
        lxu_cache_locations,
        {%- if vbe %}
        B_offsets_,
        vbe_row_output_offsets,
        vbe_b_t_map,
        {%- endif %}
        {{ args.split_saved_tensors | join(", ") }}
    });

    {%- if not nobag %}
    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["pooling_mode"] = pooling_mode;
    {%- else %}
    ctx->saved_data["D"] = D;
    {%- endif %}
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;

    {%- if optimizer != "none" %}
    ctx->saved_data["gradient_clipping"] = gradient_clipping;
    ctx->saved_data["max_gradient"] = max_gradient;
    ctx->saved_data["stochastic_rounding"] = stochastic_rounding;
    {%- endif %} {#-/* if optimizer != "none" */#}
    ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
    const auto info_B_mask_int64 = static_cast<int64_t>(info_B_mask);
    ctx->saved_data["info_B_mask"] = info_B_mask_int64;
    ctx->saved_data["use_uniq_cache_locations_bwd"] = use_uniq_cache_locations_bwd;
    ctx->saved_data["use_homogeneous_placements"] = use_homogeneous_placements;

    {%- for (var, _) in args.saved_data %}
    ctx->saved_data["{{ var }}"] = {{ var }};
    {%- endfor %}

    {%- if optimizer == "none" %}
    // Flatten
    const auto& flatten_dev_weights = dev_weights.flatten();
    {%- else %}
    const auto& flatten_dev_weights = dev_weights;
    {%- endif %}

    {%- if not nobag %}
    {%- for weighted in [False, True] %}
    {%- set wdesc = "weighted" if weighted else "unweighted" %}
    {%- if not weighted %}
    if (!indice_weights) {
    {%- else %}
    else {
    {%- endif %}
        {%- set forward_op = "split_embedding_codegen_forward_{}{}_cuda".format(
                wdesc, vdesc
            )
        %}
        static auto split_embedding_codegen_forward_op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::{{ forward_op }}", "")
                .typed<decltype({{ forward_op }})>();

        return {
          split_embedding_codegen_forward_op.call(
            flatten_dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            D_offsets,
            total_D,
            max_D,
            indices,
            offsets,
            pooling_mode,
            {%- if weighted %}
            *indice_weights,
            {%- endif %}
            lxu_cache_locations,
            uvm_cache_stats_,
            output_dtype,
            {%- if vbe %}
            vbe_row_output_offsets,
            vbe_b_t_map,
            vbe_output_size,
            info_B_num_bits,
            info_B_mask_int64,
            {%- endif %}
            is_experimental
          )
        };
    }
    {%- endfor %}
    {%- else %}
    {%- set forward_nobag_op = "split_embedding_nobag_codegen_forward_unweighted_cuda" %}
    static auto split_embedding_codegen_forward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ forward_nobag_op }}", "")
            .typed<decltype({{ forward_nobag_op }})>();
    return {
      split_embedding_codegen_forward_op.call(
        flatten_dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D,
        indices,
        offsets,
        lxu_cache_locations,
        uvm_cache_stats_,
        output_dtype,
        /*is_experimental=*/false
      )
    };
    {%- endif %}
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto dev_weights = *savedItr++;
    auto uvm_weights = *savedItr++;
    auto lxu_cache_weights = *savedItr++;
    auto weights_placements = *savedItr++;
    auto weights_offsets = *savedItr++;
    {%- if not nobag %}
    auto D_offsets = *savedItr++;
    {%- endif %}
    auto hash_size_cumsum = *savedItr++;
    auto indices = *savedItr++;
    auto offsets = *savedItr++;
    {%- if not nobag %}
    auto indice_weights = *savedItr++;
    auto feature_requires_grad = *savedItr++;
    {%- endif %}
    auto lxu_cache_locations = *savedItr++;
    {%- if vbe %}
    auto B_offsets = *savedItr++;
    auto vbe_row_output_offsets = *savedItr++;
    auto vbe_b_t_map = *savedItr++;
    {%- endif %}

    {%- for tensor in args.split_saved_tensors %}
    auto {{ tensor }} = *savedItr++;
    {%- endfor %}

    {%- if not nobag %}
    auto max_D = ctx->saved_data["max_D"].toInt();
    auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();
    {%- else %}
    auto D = ctx->saved_data["D"].toInt();
    {%- endif %}
    auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();

    {%- if optimizer != "none" %}
    auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
    auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
    auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();
    {%- endif %} {#-/* if optimizer != "none" */#}
    const int32_t info_B_num_bits = ctx->saved_data["info_B_num_bits"].toInt();
    const int64_t info_B_mask_int64 = ctx->saved_data["info_B_mask"].toInt();
    const auto use_uniq_cache_locations_bwd =
      ctx->saved_data["use_uniq_cache_locations_bwd"].toBool();
    const auto use_homogeneous_placements =
      ctx->saved_data["use_homogeneous_placements"].toBool();

    {%- for (var, ivalue_cast) in args.saved_data %}
    auto {{ var }} = ctx->saved_data["{{ var }}"].{{ ivalue_cast }}();
    {%- endfor %}

    TORCH_CHECK_EQ(grad_outputs.size(), 1);

#ifdef USE_ROCM
    constexpr int32_t BT_block_size = 64;
    constexpr int32_t max_segment_length_per_warp = 64;
#else
    constexpr int32_t BT_block_size = 32;
    constexpr int32_t max_segment_length_per_warp = 32;
#endif
    using torch::autograd::Variable;

    {%- if optimizer != "none" %}
    auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];
    {%- else %}
    auto& grad_output = grad_outputs[0];
    {%- endif %}

    // FIXME: to support aligned memory access in Vec4T load/store function
    // 16 for FP32 and 8 for FP16
    if (grad_output.dim() > 1 &&
        (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
        grad_output.stride(1) != 1 || grad_output.stride(0) % 4 != 0)) {
        grad_output = grad_output.contiguous();
    }
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0) {
        grad_output = at::empty_like(grad_output).copy_(grad_output);
    }

    {%- if not nobag %}
    {%- if optimizer == "none" %}
    // Flatten (dev_weights is used in
    // split_embedding_codegen_grad_indice_weights{{ vdesc }}_cuda)
    dev_weights = dev_weights.flatten();
    {%- endif %}

    {%- set grad_indice_weights_op =
        "split_embedding_codegen_grad_indice_weights{}_cuda".format(vdesc)
    %}
    static auto split_embedding_codegen_grad_indice_weights_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ grad_indice_weights_op }}", "")
            .typed<decltype({{ grad_indice_weights_op }})>();

    const auto grad_indice_weights = !indice_weights.defined() ?
      Variable() :
      split_embedding_codegen_grad_indice_weights_op.call(
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

    {%- for weighted in [False, True] %}
    {%- set wdesc = "weighted" if weighted else "unweighted" %}
    {%- set backward_op = "split_embedding_backward_codegen_{}_{}_exact{}_cuda".format(
            optimizer, wdesc, vdesc
        )
    %}
    static auto split_embedding_codegen_{{ wdesc }}_backward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ backward_op }}", "")
            .typed<decltype({{ backward_op }})>();
    {%- endfor %} {#-/* for weighted */#}

    const auto grad_dev_weights = !indice_weights.defined() ?
      {%- for weighted in [False, True] %}
      {%- set wdesc = "weighted" if weighted else "unweighted" %}
      split_embedding_codegen_{{ wdesc }}_backward_op.call(
          grad_output,
          dev_weights,
          uvm_weights,
          lxu_cache_weights,
          weights_placements,
          weights_offsets,
          D_offsets,
          max_D,
          hash_size_cumsum,
          total_hash_size_bits,
          indices,
          offsets,
          pooling_mode,
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
          use_uniq_cache_locations_bwd,
          use_homogeneous_placements,
          {{ args.split_function_arg_names | join(", ") }}
      ) {{ ":" if not weighted else ";" }}
      {%- endfor %} {#-/* for weighted in [False, True] */#}
    return {
        Tensor(), // placeholder autograd tensor
        Variable(), // output_dtype
        grad_dev_weights, // dev_weights
        Variable(), // uvm_weights
        Variable(), // lxu_cache_weights
        Variable(), // weights_placements
        Variable(), // weights_offsets
        Variable(), // D_offsets
        Variable(), // total_D
        Variable(), // max_D
        Variable(), // hash_size_cumsum
        Variable(), //total_hash_size_bits
        Variable(), // indices
        Variable(), // offsets
        Variable(), // pooling_mode
        grad_indice_weights, // indice_weights
        Variable(), // feature_requires_grad
        Variable(), // lxu_cache_locations
        Variable(), // uvm_cache_stats
        {%- if optimizer != "none" %}
        Variable(), // gradient_clipping
        Variable(), // max_gradient
        Variable(), // stochastic_rounding
        {%- endif %}
        {%- if vbe %}
        Variable(), // B_offsets
        Variable(), // vbe_output_offsets_feature_rank
        Variable(), // vbe_B_offsets_rank_per_feature
        Variable(), // max_B
        Variable(), // max_B_feature_rank
        Variable(), // vbe_output_size
        {%- endif %}
        Variable(), // is_experimental
        Variable(), // use_uniq_cache_locations_bwd
        Variable(), // use_homogeneous_placements
        {{ args.split_variables | join(", ") }}
    };
    {%- else %}
    {%- set backward_nobag_op =
        "split_embedding_nobag_backward_codegen_{}_unweighted_exact_cuda".format(
           optimizer
        )
    %}

    static auto split_embedding_nobag_codegen_backward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ backward_nobag_op }}", "")
            .typed<decltype({{ backward_nobag_op }})>();

    const auto grad_dev_weights = split_embedding_nobag_codegen_backward_op.call(
        grad_output,
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D,
        hash_size_cumsum,
        total_hash_size_bits,
        indices,
        offsets,
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
        use_uniq_cache_locations_bwd,
        use_homogeneous_placements,
        {{ args.split_function_arg_names | join(", ") }}
    );
    return {
        Tensor(), // placeholder autograd tensor
        Variable(), // output_dtype
        grad_dev_weights, // dev_weights
        Variable(), // uvm_weights
        Variable(), // lxu_cache_weights
        Variable(), // weights_placements
        Variable(), // weights_offsets
        Variable(), // D
        Variable(), // hash_size_cumsum
        Variable(), // total_hash_size_bits
        Variable(), // indices
        Variable(), // offsets
        Variable(), // lxu_cache_locations
        Variable(), // uvm_cache_stats
        {%- if optimizer != "none" %}
        Variable(), // gradient_clipping
        Variable(), // max_gradient
        Variable(), // stochastic_rounding
        {%- endif %}
        {%- if vbe %}
        Variable(), // B_offsets
        Variable(), // vbe_output_offsets_feature_rank
        Variable(), // vbe_B_offsets_rank_per_feature
        Variable(), // max_B
        Variable(), // max_B_feature_rank
        Variable(), // vbe_output_size
        {%- endif %}
        Variable(), // is_experimental
        Variable(), // use_uniq_cache_locations_bwd
        Variable(), // use_homogeneous_placements
        {{ args.split_variables | join(", ") }}
    };
    {%- endif %}
  }
};
{%- endif %} {#-/* if not nobag or not vbe */#}
{%- endfor %} {#-/* for nobag in [True, False] */#}
{%- endfor %} {#-/* for vbe in [True, False] */#}
{%- endif %} {#-/* if has_gpu_support */#}

///@ingroup embedding-cuda
Tensor split_embedding_codegen_lookup_{{ optimizer }}_function(
    const Tensor& placeholder_autograd_tensor,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const int64_t total_D,
    const int64_t max_D,
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t pooling_mode,
    const c10::optional<Tensor>& indice_weights,
    const c10::optional<Tensor>& feature_requires_grad,
    const Tensor& lxu_cache_locations,
    {%- if optimizer != "none" %}
    const bool gradient_clipping,
    const double max_gradient,
    const bool stochastic_rounding,
    {%- endif %}
    {{ args.split_function_args | join(", ") }},
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const c10::optional<Tensor>& B_offsets = c10::optional<Tensor>(),
    const c10::optional<Tensor>& vbe_output_offsets_feature_rank = c10::optional<Tensor>(),
    const c10::optional<Tensor>& vbe_B_offsets_rank_per_feature = c10::optional<Tensor>(),
    const int64_t max_B = -1,
    const int64_t max_B_feature_rank = -1,
    const int64_t vbe_output_size = -1,
    const bool is_experimental = false,
    const bool use_uniq_cache_locations_bwd = false,
    const bool use_homogeneous_placements = false,
    const c10::optional<Tensor>& uvm_cache_stats = c10::optional<Tensor>()
) {
  {%- if has_gpu_support %}
  {%- for vbe in ([True, False] if has_vbe_support else [False]) %}

  {%- if has_vbe_support %}
  {%- if vbe %}
  if (B_offsets.has_value()) {
  {%- else %}
  else { // if (B_offsets.has_value())
  {%- endif %}
  {%- endif %} {#-/* if has_vbe_support */#}
    {%- for nobag in [True, False] %}
    {%- set vbe = False if nobag else vbe %}
    {%- set autograd_func = "Split{}{}LookupFunction_{}_Op".format(
          "NoBag" if nobag else "",
          "VBE" if vbe else "",
          optimizer
        )
    %}
    {%- if nobag %}
    if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
    {%- else %}
    else {
    {%- endif %}
      return {{ autograd_func }}::apply(
          placeholder_autograd_tensor,
          output_dtype,
          dev_weights,
          uvm_weights,
          lxu_cache_weights,
          weights_placements,
          weights_offsets,
          {%- if nobag %}
          max_D,
          {%- else %}
          D_offsets,
          total_D,
          max_D,
          {%- endif %}
          hash_size_cumsum,
          total_hash_size_bits,
          indices,
          offsets,
          {%- if not nobag %}
          pooling_mode,
          indice_weights,
          feature_requires_grad,
          {%- endif %}
          lxu_cache_locations,
          uvm_cache_stats,
          {%- if optimizer != "none" %}
          gradient_clipping,
          max_gradient,
          stochastic_rounding,
          {%- endif %}
          {%- if vbe %}
          B_offsets,
          vbe_output_offsets_feature_rank,
          vbe_B_offsets_rank_per_feature,
          max_B,
          max_B_feature_rank,
          vbe_output_size,
          {%- endif %}
          is_experimental,
          use_uniq_cache_locations_bwd,
          use_homogeneous_placements,
          {{ args.split_function_arg_names | join(", ") }})[0];
    }
    {%- endfor %} {#-/* for nobag */#}
  {%- if has_vbe_support %}
  }
  {%- endif %}
  {%- endfor %}
  {%- else %}
  TORCH_CHECK(false, "split_embedding_codegen_lookup_{{ optimizer }}_function is deprecated. Please see https://github.com/pytorch/FBGEMM/discussions/1727 for more detail.");
  return Tensor();
  {%- endif %} {#-/* if has_gpu_support */#}
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
{%- for lib_name in ["fb", "fbgemm"] %}
TORCH_LIBRARY_FRAGMENT({{ lib_name }}, m) {
    m.def("split_embedding_codegen_lookup_{{ optimizer }}_function("
          "    Tensor placeholder_autograd_tensor, "
          "    Tensor dev_weights, Tensor uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    int total_D, "
          "    int max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor? indice_weights, "
          "    Tensor? feature_requires_grad, "
          "    Tensor lxu_cache_locations, "
          {%- if optimizer != "none" %}
          "    bool gradient_clipping, "
          "    float max_gradient, "
          "    bool stochastic_rounding, "
          {%- endif %}
          "    {{ args.split_function_schemas | join(", ") }}, "
          "    int output_dtype=0, "
          "    Tensor? B_offsets=None, "
          "    Tensor? vbe_output_offsets_feature_rank=None, "
          "    Tensor? vbe_B_offsets_rank_per_feature=None, "
          "    int max_B=-1, "
          "    int max_B_feature_rank=-1, "
          "    int vbe_output_size=-1, "
          "    bool is_experimental=False, "
          "    bool use_uniq_cache_locations_bwd=False, "
          "    bool use_homogeneous_placements=False, "
          "    Tensor? uvm_cache_stats=None"
          ") -> Tensor",
          {PT2_COMPLIANT_TAG});
    // We're playing a funny trick here: we're using the autograd
    // implementation of the operator at all the dispatch keys.  This is OK
    // because autograd.Function works even in a context where there is
    // no autograd enabled, and all of the internal implementations redispatch
    // appropriately
    m.impl(
        "split_embedding_codegen_lookup_{{ optimizer }}_function",
        torch::dispatch(
          c10::DispatchKey::Autograd,
          TORCH_FN(split_embedding_codegen_lookup_{{ optimizer }}_function)));
    m.impl(
        "split_embedding_codegen_lookup_{{ optimizer }}_function",
        torch::dispatch(
          c10::DispatchKey::Meta,
          TORCH_FN(split_embedding_codegen_lookup_{{ optimizer }}_function)));
    DISPATCH_TO_CUDA(
        "split_embedding_codegen_lookup_{{ optimizer }}_function",
        split_embedding_codegen_lookup_{{ optimizer }}_function);
}
{%- endfor %} {#-/* for lib_name */#}
    // clang-format on
