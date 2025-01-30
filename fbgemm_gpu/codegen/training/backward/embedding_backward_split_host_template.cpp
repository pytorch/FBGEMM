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
#include "torch/csrc/autograd/record_function_ops.h"

#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/config/feature_gates.h"

using Tensor = at::Tensor;

using namespace fbgemm_gpu;
namespace profiler = torch::autograd::profiler;

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

////////////////////////////////////////////////////////////////////////////////
// Macro Helper Functions
////////////////////////////////////////////////////////////////////////////////

// TO DO: Refactor
{#
/* This macro generates a code blob for dispatching corresponding weighted and
    unweighted forward op from via Pytorch dispatcher
*/
#}
{%- macro call_forward_op_dispatch(nobag, weighted, vbe, is_gwd) %}
    {%- set forward_op = "{}_embedding{}_codegen_forward_{}{}{}_cuda".format(
            fwd_mdesc,
            "_nobag" if nobag else "",
            "weighted" if weighted else "unweighted",
            "_vbe" if vbe else "",
            "_gwd" if is_gwd else "",
        )
    %}
    {%- set has_experimental = has_experimental_support(
            dense, nobag, vbe, is_index_select=False, ssd=ssd
        ) and not is_gwd
    %}
    static auto embedding_codegen_forward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ forward_op }}", "")
            .typed<decltype({{ forward_op }})>();

    auto output = embedding_codegen_forward_op.call(
      flatten_dev_weights,
      {%- if not dense %}
      uvm_weights,
      lxu_cache_weights,
      weights_placements,
      {%- endif %}
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
      {%- endif %} {# /* if not nobag */ #}
      {%- if weighted %}
      *indice_weights,
      {%- endif %}
      {%- if not dense %}
      {{ "ssd_tensors[SSDTensor::ROW_ADDRS]" if ssd else "lxu_cache_locations" }},
      uvm_cache_stats_,
      {%- endif %}
      output_dtype,
      {%- if not nobag %}
      {%- if vbe %}
      vbe_row_output_offsets,
      vbe_b_t_map,
      vbe_output_size,
      info_B_num_bits,
      info_B_mask_int64,
      {%- endif %} {# /* if vbe */ #}
      {%- if is_gwd %}
      hash_size_cumsum,
      prev_iter_dev_,
      learning_rate_tensor,
      weight_decay,
      iter,
      gwd_lower_bound,
      {%- endif %} {# /* if is_gwd */ #}
      {%- endif %} {# /* if not nobag */ #}
      {{ "is_experimental" if has_experimental else "false" }}
    );

    if (is_annotate_trace_enabled) {
      record_trace->record.end();
    }

    return {output};
{%- endmacro %}

/* This macro generates a code blob for dispatching corresponding weighted and
    unweighted backward op via Pytorch dispatcher
*/
{%- macro call_backward_op_dispatch(nobag, weighted, vbe, is_gwd) %}
  {%- set wdesc = "_weighted" if weighted else "_unweighted" %}
  {%- set backward_op = "{}_embedding{}_backward_codegen_{}{}{}{}_exact_cuda".format(
          bwd_mdesc,
          "_nobag" if nobag else "",
          optimizer,
          wdesc,
          "_vbe" if vbe else "",
          "_gwd" if is_gwd else "",
      )
  %}
    static auto embedding_codegen_{{ wdesc }}_backward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ backward_op }}", "")
            .typed<decltype({{ backward_op }})>();

    grad_dev_weights = embedding_codegen_{{ wdesc }}_backward_op.call(
          grad_output,
          dev_weights,
          {% if not dense %}
          uvm_weights,
          lxu_cache_weights,
          weights_placements,
          {%- endif %}
          weights_offsets,
          {% if nobag %}
          D,
          {%- else %}
          D_offsets,
          max_D,
          mixed_D,
          {%- endif %} {# /* if nobag */ #}
          hash_size_cumsum,
          total_hash_size_bits,
          indices,
          offsets,
          {%- if not nobag %}
          pooling_mode,
          {%- if weighted %}
          indice_weights,
          {%- endif %}
          {%- endif %} {# /* if not nobag */ #}
          {%- if ssd %}
          ssd_row_addrs,
          {%- elif not dense %}
          lxu_cache_locations,
          {%- endif %}
          BT_block_size,
          max_segment_length_per_warp,
          {%- if optimizer != "none" and not dense %}
          stochastic_rounding,
          {%- endif %}
          {%- if not dense %}
          info_B_num_bits,
          info_B_mask_int64,
          {%- endif %}
          {%- if vbe %}
          B_offsets,
          vbe_row_output_offsets,
          vbe_b_t_map,
          {%- endif %} {# /* if vbe */ #}
          {%- if not dense %}
          use_uniq_cache_locations_bwd,
          use_homogeneous_placements,
          {%- endif %}
          {%- if is_gwd %}
          {%- if "prev_iter_dev" not in args.split_function_arg_names %}
          prev_iter_dev,
          {%- endif %}
          {%- if "iter" not in args.split_function_arg_names %}
          iter,
          {%- endif %}
          gwd_lower_bound,
          {%- endif %} {# /* if is_gwd */ #}
          {%- if not dense %}
          {{ args.split_function_arg_names | join(", ") }}
          {%- else %}
          /*unused=*/0
          {%- endif %}
    );

    if (is_annotate_trace_enabled) {
      record_trace->record.end();
    }

    return {
        {%- if not dense %}
        Tensor(), // placeholder autograd tensor
        {%- endif %}
        Variable(), // output_dtype
        grad_dev_weights, // dev_weights
        {%- if not dense %}
        Variable(), // uvm_weights
        Variable(), // lxu_cache_weights
        Variable(), // weights_placements
        {%- endif %}
        Variable(), // weights_offsets
        {%- if nobag %}
        Variable(), // D
        {%- else %}
        Variable(), // D_offsets
        Variable(), // total_D
        Variable(), // max_D
        Variable(), // mixed_D
        {%- endif %}
        Variable(), // hash_size_cumsum
        Variable(), //total_hash_size_bits
        Variable(), // indices
        Variable(), // offsets
        {%- if not nobag %}
        Variable(), // pooling_mode
        grad_indice_weights, // indice_weights
        Variable(), // feature_requires_grad
        {%- endif %}
        {%- if not dense %}
        Variable(), // lxu_cache_locations
        Variable(), // uvm_cache_stats
        {%- endif %}
        {%- if optimizer != "none" and not dense %}
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
        {%- if not dense %}
        Variable(), // is_experimental
        Variable(), // use_uniq_cache_locations_bwd
        Variable(), // use_homogeneous_placements
        {%- endif %}
        {%- if is_gwd %}
        {%- if "prev_iter_dev" not in args.split_function_arg_names %}
        Variable(), // prev_iter_dev
        {%- endif %}
        {%- if "iter" not in args.split_function_arg_names %}
        Variable(), // iter
        {%- endif %}
        Variable(), // gwd_lower_bound
        {%- endif %}
        {%- if ssd %}
        {%- for tensor in ssd_tensors %}
        Variable(), // {{ tensor }}
        {%- endfor %}
        {%- endif %}
        {{ args.split_variables | join(", ") }}
    };
{%- endmacro %}

/* This macro generates a code blob that calls corresponding autograd function
    from lookup_function
*/
{%- macro call_autograd(nobag, vbe, is_gwd) %}
    {%- set autograd_func = "{}{}{}{}LookupFunction_{}_Op".format(
          "SSD" if ssd else "Split",
          "NoBag" if nobag else "",
          "VBE" if vbe else "",
          "GWD" if is_gwd else "",
          optimizer
        )
    %}
      return {{ autograd_func }}::apply(
          {%- if not dense %}
          placeholder_autograd_tensor,
          {%- endif %}
          output_dtype,
          dev_weights,
          {%- if not dense %}
          uvm_weights,
          lxu_cache_weights,
          weights_placements,
          {%- endif %}
          weights_offsets,
          {%- if nobag %}
          max_D,
          {%- else %}
          D_offsets,
          total_D,
          max_D,
          mixed_D,
          {%- endif %}
          hash_size_cumsum,
          total_hash_size_bits,
          indices,
          {%- if not nobag and dense and not vbe %}
          offsets,
          pooling_mode,
          indice_weights,
          feature_requires_grad
          {%- elif not nobag %}
          offsets,
          pooling_mode,
          indice_weights,
          feature_requires_grad,
          {%- elif nobag and dense and not vbe %}
          offsets
          {%- else %}
          offsets,
          {%- endif %}
          {%- if not dense %}
          lxu_cache_locations,
          uvm_cache_stats,
          {%- endif %}
          {%- if optimizer != "none" and not dense %}
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
          {%- endif %}
          {%- if vbe and not dense %}
          vbe_output_size,
          {%- elif vbe and dense %}
          vbe_output_size
          {%- endif %}
          {%- if not dense %}
          is_experimental,
          use_uniq_cache_locations_bwd,
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
          {%- if ssd %}
          ssd_tensors.value(),
          {%- endif  %}
          {{ args.split_function_arg_names_autograd | join(", ") }}
          {%- endif %}
          )[0];
{%- endmacro %}

////////////////////////////////////////////////////////////////////////////////
// External Function Declarations
////////////////////////////////////////////////////////////////////////////////

{%- if has_gpu_support %}
/// @defgroup embedding-cuda Embedding CUDA Operators

{%- for vbe in ([True, False] if has_vbe_support else [False]) %}
{%- set vdesc = "_vbe" if vbe else "" %}

Tensor {{ fwd_mdesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_cuda(
    const Tensor& grad_output,
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt max_D,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if ssd %}
    const Tensor& ssd_row_addrs,
    {%- elif not dense %}
    const Tensor& lxu_cache_locations,
    {%- endif %}
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

{%- for nobag in ([False] if (weighted or vbe) else [True, False]) %}
{%- set ndesc = "_nobag" if nobag else "" %}

{%- for is_gwd in ([True, False]
    if is_valid_gwd_config(
      dense,
      nobag,
      vbe,
      is_index_select,
      has_global_weight_decay_support,
      ssd)
      else [False]) %}
{%- set gwddesc = "_gwd" if is_gwd else "" %}

{%- for weighted in [True, False] %}
{%- set wdesc = "_weighted" if weighted else "_unweighted" %}

{%- set desc_suffix = wdesc + vdesc + gwddesc %}

Tensor {{ fwd_mdesc }}_embedding{{ ndesc }}_codegen_forward{{ desc_suffix }}_cuda(
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    {%- if nobag %}
    const c10::SymInt D,
    {%- else %}
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    {%- endif %}
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    const Tensor& indice_weights,
    {%- endif %}
    {%- if not dense %}
    const Tensor& {{ "ssd_row_addrs" if ssd else "lxu_cache_locations" }},
    const Tensor& uvm_cache_stats,
    {%- endif %}
    const int64_t output_dtype,
    {%- if vbe %}
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    const c10::SymInt vbe_output_size,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    {%- endif %}
    {%- if is_gwd %}
    const Tensor& hash_size_cumsum,
    const Tensor& prev_iter_dev,
    const Tensor& learning_rate_tensor,
    const double weight_decay,
    const int64_t iter,
    const double gwd_lower_bound,
    {%- endif %}
    const bool is_experimental
);

Tensor
{{ bwd_mdesc }}_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}{{ desc_suffix }}_exact_cuda(
    const Tensor& grad_output,
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
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
    {%- endif %}
    {%- if weighted %}
    const Tensor& indice_weights,
    {%- endif %}
    {%- if ssd %}
    const Tensor& ssd_row_addrs,
    {%- elif not dense %}
    const Tensor& lxu_cache_locations,
    {%- endif %}
    const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp,
    {%- if optimizer != "none" and not dense %}
    const bool stochastic_rounding,
    {%- endif %}
    {%- if not dense %}
    const int64_t info_B_num_bits,
    const int64_t info_B_mask_int64,
    {%- endif %}
    {%- if vbe %}
    const Tensor& B_offsets,
    const Tensor& vbe_row_output_offsets,
    const Tensor& vbe_b_t_map,
    {%- endif %}
    {%- if not dense %}
    const bool use_uniq_cache_locations,
    const bool use_homogeneous_placements,
    {%- endif %}
    {%- if is_gwd %}
    {%- if "prev_iter_dev" not in args.split_function_arg_names %}
    const Tensor& prev_iter_dev,
    {%- endif %}
    {%- if "iter" not in args.split_function_arg_names %}
    const int64_t iter,
    {%- endif %}
    const double gwd_lower_bound,
    {%- endif %}
    {{ args.split_function_args | join(", ") }});
{%- endfor %} {#-/* for weighted*/#}

////////////////////////////////////////////////////////////////////////////////
// Autograd Function Declarations
////////////////////////////////////////////////////////////////////////////////

{#- /* Generate a separate autograd function for global weight decay */ #}
{%- set autograd_func = "{}{}{}{}LookupFunction_{}_Op".format(
      "SSD" if ssd else "Split",
      "NoBag" if nobag else "",
      "VBE" if vbe else "",
      "GWD" if is_gwd else "",
      optimizer,
    )
%}

class {{ autograd_func }} :
    public torch::autograd::Function<{{ autograd_func }}> {
 public:
  static constexpr bool is_traceable = true;

  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    {%- if not dense %}
    const Tensor& placeholder_autograd_tensor,
    {%- endif %}
    const int64_t output_dtype,
    const Tensor& dev_weights,
    {%- if not dense %}
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    {%- if not nobag %}
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    const bool mixed_D,
    {%- else %}
    const c10::SymInt D,
    {%- endif %}
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    {%- if not nobag and dense and not vbe %}
    const Tensor& offsets,
    const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights,
    const std::optional<Tensor>& feature_requires_grad
    {%- elif not nobag %}
    const Tensor& offsets,
    const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights,
    const std::optional<Tensor>& feature_requires_grad,
    {%- elif nobag and dense and not vbe %}
    const Tensor& offsets
    {%- else %}
    const Tensor& offsets,
    {%- endif %}
    {%- if not dense %}
    const Tensor& lxu_cache_locations,
    std::optional<Tensor> uvm_cache_stats,
    {%- if optimizer != "none" %}
    const bool gradient_clipping,
    const double max_gradient,
    const bool stochastic_rounding,
    {%- endif %}
    {%- if vbe %}
    const std::optional<Tensor>& B_offsets,
    const std::optional<Tensor>& vbe_output_offsets_feature_rank,
    const std::optional<Tensor>& vbe_B_offsets_rank_per_feature,
    const c10::SymInt max_B,
    const c10::SymInt max_B_feature_rank,
    const c10::SymInt vbe_output_size,
    {%- endif %}
    const bool is_experimental,
    const bool use_uniq_cache_locations_bwd,
    const bool use_homogeneous_placements,
    {%- if is_gwd %}
    {%- if "prev_iter_dev" not in args.split_function_arg_names %}
    const std::optional<Tensor>& prev_iter_dev,
    {%- endif %}
    {%- if "iter" not in args.split_function_arg_names %}
    const int64_t iter,
    {%- endif %}
    const double gwd_lower_bound,
    {%- endif %}
    {%- if ssd %}
    const at::TensorList& ssd_tensors,
    {%- endif %}
    {{ args.split_function_args_autograd | join(", ") }}
    {%- else %}
    {%- if vbe %}
    const std::optional<Tensor>& B_offsets,
    const std::optional<Tensor>& vbe_output_offsets_feature_rank,
    const std::optional<Tensor>& vbe_B_offsets_rank_per_feature,
    const c10::SymInt max_B,
    const c10::SymInt max_B_feature_rank,
    const c10::SymInt vbe_output_size
    {%- endif %}
    {%- endif %}) {

    const auto T = weights_offsets.sym_numel();
    {%- if vbe %}
    const auto B_offsets_ = B_offsets.value_or(Tensor());
    const auto vbe_output_offsets_feature_rank_ = vbe_output_offsets_feature_rank.value_or(Tensor());
    const auto vbe_B_offsets_rank_per_feature_ = vbe_B_offsets_rank_per_feature.value_or(Tensor());

    const c10::SymInt max_B_ = max_B;
    {%- else %}
    const auto max_B_ = offsets.sym_size(0) / T;
    {%- endif %}

    // Annotate Kineto trace
    const static bool is_annotate_trace_enabled = config::is_feature_enabled(
        config::FeatureGateName::TBE_ANNOTATE_KINETO_TRACE);
    std::string op_annotation = "";
    c10::intrusive_ptr<profiler::PythonRecordFunction> record_trace;
    if (is_annotate_trace_enabled) {
      std::stringstream ss;
      ss << "["
        << "weighted={{ "T" if weighted else "F" }},"
        << "pooled={{ "T" if not nobag else "F" }},"
        << "vbe={{ "T" if vbe else "F" }},"
        << "avg_B=" << ({{ "max_B_" if not vbe else "max_B_ / T" }}) << ","
        << "max_B=" << max_B_ << ","
        << "T=" << T << ","
        << "avg_D=" << ({{ "total_D / T" if not nobag else "D" }}) << ","
        << "max_D=" << {{ "max_D" if not nobag else "D" }} << ","
        << "num_indices=" << indices.sym_numel() << ","
        << "avg_pooling_fac=" << (static_cast<c10::SymFloat>(indices.sym_numel()) / T / max_B_)
        << "]";
      op_annotation = ss.str();
      record_trace = profiler::record_function_enter_new(
        "{{ fwd_mdesc }}_tbe_fwd" + op_annotation);
      ctx->saved_data["op_annotation"] = op_annotation;
    }

    {%- if not dense %}
    // NOTE: The `local_uvm_cache_stats` variable held by the nn.Module has dtype int32_t
    // TODO: Hook up with frontend code
    const auto uvm_cache_stats_ = uvm_cache_stats
      .value_or(at::empty({0}, uvm_weights.options().dtype(at::kInt)));
    {%- endif %}

    // Default values for Dynamo tracing
    // SymInt does not support bitshifts operator
    // Constanting info_B_num_bits, info_B_mask for Dynamo for now.
    int32_t info_B_num_bits = DEFAULT_INFO_B_NUM_BITS;
    uint32_t info_B_mask = (1u << info_B_num_bits) - 1;
    if (max_B_.is_symbolic()) {
      info_B_num_bits = 22;
      info_B_mask = (1u << info_B_num_bits) - 1;

      // TODO(ivankobzarev): Guarding Dynamo that T and B fits in constanted number of bits.
      // TORCH_CHECK(max_B_ < 1u << info_B_num_bits)
      // TORCH_CHECK(T < 1u << (DEFAULT_INFO_NUM_BITS - info_B_num_bits))
    } else {
      // TODO: don't guard here
      std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(max_B_.guard_int(__FILE__, __LINE__), T.guard_int(__FILE__, __LINE__));
    }

    {%- if vbe %}
    static auto generate_vbe_metadata_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::generate_vbe_metadata", "")
            .typed<std::tuple<Tensor, Tensor>(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const int64_t, const bool, const c10::SymInt, const int64_t, const c10::SymInt)>();

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
        /*total_B=*/offsets.sym_size(0) - 1
        );
    {%- endif %}

    {%- if is_gwd %}
    const auto prev_iter_dev_ = prev_iter_dev.value_or(Tensor());
    {%- endif %}

    ctx->save_for_backward({
        dev_weights,
        {%- if not dense %}
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        {%- endif %}
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
        {%- if not dense %}
        lxu_cache_locations,
        {%- endif %}
        {%- if vbe %}
        B_offsets_,
        vbe_row_output_offsets,
        vbe_b_t_map,
        {%- endif %}
        {%- if is_gwd and "prev_iter_dev" not in args.split_function_arg_names %}
        prev_iter_dev_,
        {%- endif %}
        {%- if ssd %}
        {%- for tensor in ssd_tensors %}
        ssd_tensors[SSDTensor::{{ tensor | upper }}],
        {%- endfor %}
        {%- endif %}
        {{ args.split_saved_tensors_optional | join(", ") }}
    });

    {%- if not nobag %}
    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["mixed_D"] = mixed_D;
    ctx->saved_data["pooling_mode"] = pooling_mode;
    {%- else %}
    ctx->saved_data["D"] = D;
    {%- endif %}
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;

    {%- if optimizer != "none" and not dense %}
    ctx->saved_data["gradient_clipping"] = gradient_clipping;
    ctx->saved_data["max_gradient"] = max_gradient;
    ctx->saved_data["stochastic_rounding"] = stochastic_rounding;
    {%- endif %} {#-/* if optimizer != "none" */#}
    ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
    const auto info_B_mask_int64 = static_cast<int64_t>(info_B_mask);
    ctx->saved_data["info_B_mask"] = info_B_mask_int64;
    {%- if not dense %}
    ctx->saved_data["use_uniq_cache_locations_bwd"] = use_uniq_cache_locations_bwd;
    ctx->saved_data["use_homogeneous_placements"] = use_homogeneous_placements;
    {%- endif %}
    {%- if is_gwd %}
    {%- if "iter" not in args.split_function_arg_names %}
    ctx->saved_data["iter"] = iter;
    {%- endif %}
    ctx->saved_data["gwd_lower_bound"] = gwd_lower_bound;
    {%- endif %}

    {%- if not dense %}
    {%- for (var, _) in args.saved_data %}
    ctx->saved_data["{{ var }}"] = {{ var }};
    {%- endfor %}
    {%- endif %}

    {%- if optimizer == "none" %}
    // Flatten
    const auto& flatten_dev_weights = dev_weights.flatten();
    {%- else %}
    const auto& flatten_dev_weights = dev_weights;
    {%- endif %}

    {%- if nobag %}
      {{
         call_forward_op_dispatch(
             nobag=True,
             weighted=False,
             vbe=vbe,
             is_gwd=is_gwd,
         )
      }}
    {%- else %}
    if (indice_weights) {
      {{
         call_forward_op_dispatch(
             nobag=False,
             weighted=True,
             vbe=vbe,
             is_gwd=is_gwd,
         )
       }}
    }
    {{
       call_forward_op_dispatch(
           nobag=False,
           weighted=False,
           vbe=vbe,
           is_gwd=is_gwd,
       )
     }}

    {%- endif %} {#-/* if not nobag */ #}
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto dev_weights = *savedItr++;
    {%- if not dense %}
    auto uvm_weights = *savedItr++;
    auto lxu_cache_weights = *savedItr++;
    auto weights_placements = *savedItr++;
    {%- endif %}
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
    {%- if not dense %}
    auto lxu_cache_locations = *savedItr++;
    {%- endif %}
    {%- if vbe %}
    auto B_offsets = *savedItr++;
    auto vbe_row_output_offsets = *savedItr++;
    auto vbe_b_t_map = *savedItr++;
    {%- endif %}
    {%- if is_gwd and "prev_iter_dev" not in args.split_function_arg_names %}
    auto prev_iter_dev = *savedItr++;
    {%- endif %}
    {%- if ssd %}
    {%- for tensor in ssd_tensors %}
    auto ssd_{{ tensor }} = *savedItr++;
    {%- endfor %}
    {%- endif %}

    {%- for tensor in args.split_saved_tensors %}
    auto {{ tensor }} = *savedItr++;
    {%- endfor %}

    {%- if not nobag %}
    auto max_D = ctx->saved_data["max_D"].toSymInt();
    const auto mixed_D = ctx->saved_data["mixed_D"].toBool();
    auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();
    {%- else %}
    auto D = ctx->saved_data["D"].toSymInt();
    {%- endif %}
    auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();

    {%- if optimizer != "none" and not dense %}
    auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
    auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
    auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();
    {%- endif %} {#-/* if optimizer != "none" */#}
    [[maybe_unused]] const int32_t info_B_num_bits = ctx->saved_data["info_B_num_bits"].toInt();
    [[maybe_unused]] const int64_t info_B_mask_int64 = ctx->saved_data["info_B_mask"].toInt();
    {%- if not dense %}
    const auto use_uniq_cache_locations_bwd =
      ctx->saved_data["use_uniq_cache_locations_bwd"].toBool();
    const auto use_homogeneous_placements =
      ctx->saved_data["use_homogeneous_placements"].toBool();
    {%- endif %}

    {%- if is_gwd %}
    {%- if "iter" not in args.split_function_arg_names %}
    const auto iter = ctx->saved_data["iter"].toInt();
    {%- endif %}
    const auto gwd_lower_bound = ctx->saved_data["gwd_lower_bound"].toDouble();
    {%- endif %}

    {%- if not dense%}
    {%- for (var, ivalue_cast) in args.saved_data %}
    auto {{ var }} = ctx->saved_data["{{ var }}"].{{ ivalue_cast }}();
    {%- endfor %}
    {%- endif %}

    const static bool is_annotate_trace_enabled = config::is_feature_enabled(
        config::FeatureGateName::TBE_ANNOTATE_KINETO_TRACE);
    c10::intrusive_ptr<profiler::PythonRecordFunction> record_trace;
    if (is_annotate_trace_enabled) {
      auto& op_annotation = ctx->saved_data["op_annotation"].toStringRef();
      record_trace = profiler::record_function_enter_new(
          "{{ bwd_mdesc }}_tbe_bwd" + op_annotation);
    }

    TORCH_CHECK_EQ(grad_outputs.size(), 1);

#ifdef USE_ROCM
    constexpr int32_t BT_block_size = 64;
    constexpr int32_t max_segment_length_per_warp = 64;
#else
    constexpr int32_t BT_block_size = 32;
    constexpr int32_t max_segment_length_per_warp = 32;
#endif
    using torch::autograd::Variable;

    {%- if optimizer != "none" and not dense %}
    auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];
    {%- else %}
    auto& grad_output = grad_outputs[0];
    {%- endif %}

    {%- if not nobag %}
    {%- if optimizer == "none" %}
    // Flatten (dev_weights is used in
    // {{ fwd_mdesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_cuda)
    dev_weights = dev_weights.flatten();
    {%- endif %}

    {%- set grad_indice_weights_op =
        "{}_embedding_codegen_grad_indice_weights{}_cuda".format(fwd_mdesc, vdesc)
    %}
    static auto embedding_codegen_grad_indice_weights_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ grad_indice_weights_op }}", "")
            .typed<decltype({{ grad_indice_weights_op }})>();

    const auto grad_indice_weights = !indice_weights.defined() ?
      Variable() :
      embedding_codegen_grad_indice_weights_op.call(
        grad_output,
        dev_weights,
        {%- if not dense %}
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        {%- endif %}
        weights_offsets,
        D_offsets,
        max_D,
        indices,
        offsets,
        {%- if ssd %}
        ssd_row_addrs,
        {%- elif not dense %}
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
    Tensor grad_dev_weights;
    if (indice_weights.defined())
    {
      {{
         call_backward_op_dispatch(
             nobag=False,
             weighted=True,
             vbe=vbe,
             is_gwd=is_gwd,
        )
      }}
    }
    {{
       call_backward_op_dispatch(
           nobag=False,
           weighted=False,
           vbe=vbe,
           is_gwd=is_gwd,
      )
    }}
    {%- else %}
    Tensor grad_dev_weights;
      {{
         call_backward_op_dispatch(
             nobag=True,
             weighted=False,
             vbe=vbe,
             is_gwd=is_gwd,
        )
      }}
    {%- endif %}
  }
};
{%- endfor %} {#-/* for is_gwd */#}
{%- endfor %} {#-/* for nobag */#}
{%- endfor %} {#-/* for vbe */#}
{%- endif %} {#-/* if has_gpu_support */#}

///@ingroup embedding-cuda
Tensor {{ bwd_mdesc }}_embedding_codegen_lookup_{{ optimizer }}_function(
    {%- if dense %}
    const Tensor& dev_weights,
    {%- else %}
    const Tensor& placeholder_autograd_tensor,
    const Tensor& dev_weights,
    const Tensor& uvm_weights,
    const Tensor& lxu_cache_weights,
    const Tensor& weights_placements,
    {%- endif %}
    const Tensor& weights_offsets,
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights,
    const std::optional<Tensor>& feature_requires_grad,
    {%- if not dense %}
    const Tensor& lxu_cache_locations,
    {%- if optimizer != "none"%}
    const bool gradient_clipping,
    const double max_gradient,
    const bool stochastic_rounding,
    {%- endif %}
    {{ args.split_function_args_v1 }},
    {%- endif %}
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const std::optional<Tensor>& B_offsets = std::nullopt,
    const std::optional<Tensor>& vbe_output_offsets_feature_rank = std::nullopt,
    const std::optional<Tensor>& vbe_B_offsets_rank_per_feature = std::nullopt,
    const c10::SymInt max_B = -1,
    const c10::SymInt max_B_feature_rank = -1,
    {%- if not dense %}
    const c10::SymInt vbe_output_size = -1,
    const bool is_experimental_tbe = false, // formerly named is_experimental
    const bool use_uniq_cache_locations_bwd = false,
    const bool use_homogeneous_placements = false,
    const std::optional<Tensor>& uvm_cache_stats = std::nullopt,
    {%- if "prev_iter_dev" not in args.split_function_arg_names %}
    const std::optional<Tensor>& prev_iter_dev = std::nullopt,
    {%- endif %}
    {%- if "iter" not in args.split_function_arg_names %}
    const int64_t iter = 0,
    {%- endif %}
    const bool apply_global_weight_decay = false,
    {%- if ssd %}
    const std::optional<at::TensorList>& ssd_tensors = std::nullopt,
    {%- endif %}
    const double gwd_lower_bound = 0,
    {%- else %}
    const c10::SymInt vbe_output_size = -1,
    {%- endif %}
    const bool mixed_D = true
) {
  // TODO: refactor into macro
  {%- if has_gpu_support %}

    {%- if "learning_rate_tensor" in args.split_function_arg_names %}
    // `learning rate` is changed to tensor to prevent recompilation. 
    // This interface (V1) still accepts learning rate as float for backward compatibility, 
    // We convert learning rate to tensor here to work with the backend
    // The unified PT2 interface already accepts learning rate as tensor.
    auto learning_rate_tensor = at::empty({1}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    learning_rate_tensor.fill_(learning_rate);
    {%- endif %}

    {%- if not dense %}
    // Load the config value from JK once
    static auto is_tbev2_enabled = config::is_feature_enabled(config::FeatureGateName::TBE_V2);

    // Set to experimental if either the feature is enabled in JK, or the user specifies to use TBEv2
    const auto is_experimental = is_tbev2_enabled || is_experimental_tbe;
    {%- endif %}

    {%- if ssd %}
    TORCH_CHECK(
        ssd_tensors.value().size() == {{ ssd_tensors | length }},
        "SSD TBE expects {{ ssd_tensors | length }} in ssd_tensors");
    {%- endif %}

    {%- if has_vbe_support %}
    // has vbe support
    if (B_offsets.has_value()) {
      {%- if has_global_weight_decay_support and not ssd %}
        // vbe and has gwd support
        if (apply_global_weight_decay && weight_decay > 0) {
          {{ call_autograd(nobag=False, vbe=True, is_gwd=True) }}
        }
      {%- endif %} {#-/* if has_global_weight_decay_support */ #}
      // vbe and no gwd support
      {{ call_autograd(nobag=False, vbe=True, is_gwd=False) }}
    }
    {%- endif %} {#-/* if has_vbe_support */ #}

    {%- if has_global_weight_decay_support and not ssd %}
    // has gwd support
    if (apply_global_weight_decay && weight_decay > 0) {
      // not vbe and gwd
      {{ call_autograd(nobag=False, vbe=False, is_gwd=True) }}
    }
    {%- endif %} {#-/* if has_global_weight_decay_support */ #}

    if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
      // no bag
      {{ call_autograd(nobag=True, vbe=False, is_gwd=False) }}
    }
    else {
      {{ call_autograd(nobag=False, vbe=False, is_gwd=False) }}
    }
  {%- else %}
  TORCH_CHECK(
      false,
      "{{ bwd_mdesc }}_embedding_codegen_lookup_{{ optimizer }}_function is deprecated. Please see https://github.com/pytorch/FBGEMM/discussions/1727 for more detail."
  );
  return Tensor();
  {%- endif %} {#-/* if has_gpu_support */#}
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
{%- for lib_name in ["fb", "fbgemm"] %}
TORCH_LIBRARY_FRAGMENT({{ lib_name }}, m) {
    {%- set op_name = "{}_embedding_codegen_lookup_{}_function".format(bwd_mdesc, optimizer) %}
    {%- if not dense %}
    m.def("{{ op_name }}("
          "    Tensor placeholder_autograd_tensor, "
          "    Tensor(a!) dev_weights, "
          "    Tensor(b!) uvm_weights, "
          "    Tensor lxu_cache_weights, "
          "    Tensor weights_placements, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    SymInt total_D, "
          "    SymInt max_D, "
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
          "    {{ args.split_function_schemas_v1 }}, "
          "    int output_dtype=0, "
          "    Tensor? B_offsets=None, "
          "    Tensor? vbe_output_offsets_feature_rank=None, "
          "    Tensor? vbe_B_offsets_rank_per_feature=None, "
          "    SymInt max_B=-1, "
          "    SymInt max_B_feature_rank=-1, "
          "    SymInt vbe_output_size=-1, "
          "    bool is_experimental=False, "
          "    bool use_uniq_cache_locations_bwd=False, "
          "    bool use_homogeneous_placements=False, "
          "    Tensor? uvm_cache_stats=None, "
          {%- if "prev_iter_dev" not in args.split_function_arg_names %}
          "    Tensor? prev_iter_dev=None, "
          {%- endif %}
          {%- if "iter" not in args.split_function_arg_names %}
          "    int iter=0, "
          {%- endif %}
          "    bool apply_global_weight_decay=False, "
          {%- if ssd %}
          "    Tensor[]? ssd_tensors=None,"
          {%- endif %}
          "   float gwd_lower_bound=0, "
          "   bool mixed_D=True"
          ") -> Tensor",
          {PT2_COMPLIANT_TAG});

    // We're playing a funny trick here: we're using the autograd
    // implementation of the operator at all the dispatch keys.  This is OK
    // because autograd.Function works even in a context where there is
    // no autograd enabled, and all of the internal implementations redispatch
    // appropriately
    m.impl(
        "{{ op_name }}",
        torch::dispatch(
          c10::DispatchKey::Autograd,
          TORCH_FN({{ op_name }})));
    m.impl(
        "{{ op_name }}",
        torch::dispatch(
          c10::DispatchKey::Meta,
          TORCH_FN({{ op_name }})));
    {%- endif %} {#/* if not dense */#}

    DISPATCH_TO_CUDA(
        {%- if not dense %}
        "{{ op_name }}",
        {%- else %}
        "dense_embedding_codegen_lookup_function",
        {%- endif %}
        {{ op_name }});
}
{%- endfor %} {#-/* for lib_name */#}
    // clang-format on
