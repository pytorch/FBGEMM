/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{#-/* This file contains a template for unified optimizer lookup
      function for Split TBE
      (i.e., split_embedding_codegen_lookup_{optimizer}_function_pt2),
      its op registration to PyTorch dispatcher and its corresponding
      Autograd function
      (i.e., `Split{nobag}{vbe}LookupFunction_{optimizer}_Op_pt2`).
      Codegen file: gen_embedding_split_{optimizer}_pt2_autograd.cpp

    [Lookup function invoker] (Python) --Torch dispatch-->  |
      [PT2 Lookup function] (C++) --apply--> [PT2 Autograd] | <<<
        --Torch dispatch--> [PT2 wrapper function]          | <<<
          --Torch dispatch--> [CPU/CUDA backend]            |
            --Fn call/kernel dispatch--> [CPU/CUDA kernel]  |

    The `_pt2` suffix indicates the unified API, i.e., the operators
    have the same name and function signature and dispatching between
    versions of the same operator is done via the PyTorch dispatcher
    based on input device type.
*/#}
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>

////////////////////////////////////////////////////////////////////////////////
// Required for op registrations and dispatchers
////////////////////////////////////////////////////////////////////////////////
#include "fbgemm_gpu/utils/ops_utils.h"
#include <torch/script.h>
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/split_embeddings_utils.h"
#include "fbgemm_gpu/config/feature_gates.h"
#include "fbgemm_gpu/utils/tensor_utils.h"
#include "torch/csrc/autograd/record_function_ops.h"
#include "torch/csrc/autograd/record_function_ops.h"

#include "pt2_arg_utils.h"

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
    {%- set forward_op = "{}_embedding{}_codegen_forward_{}{}{}_pt2_wrapper".format(
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
            .typed<Tensor(
                const Tensor& /*weights_host*/,
                const Tensor& /*weights_dev*/,
                {%- if not dense %}
                const Tensor& /*weights_uvm*/,
                const Tensor& /*weights_lxu_cache*/,
                const Tensor& /*weights_placements*/,
                {%- endif %}
                const Tensor& /*weights_offsets*/,
                {%- if nobag %}
                const c10::SymInt /*D*/,
                {%- else %}
                const Tensor& /*D_offsets*/,
                const c10::SymInt /*total_D*/,
                const c10::SymInt /*max_D*/,
                {%- endif %}
                const Tensor& /*hash_size_cumsum*/,
                const Tensor& /*indices*/,
                const Tensor& /*offsets*/,
                {%- if not nobag %}
                const int64_t /*pooling_mode*/,
                const Tensor& /*indice_weights*/, // CPU always takes indice_weights
                {%- endif %}
                {%- if not dense %}
                const Tensor& /*ssd_row_addrs or lxu_cache_locations*/,
                const Tensor& /*uvm_cache_stats*/,
                {%- endif %}
                {%- if vbe %}
                const Tensor& /*vbe_row_output_offsets*/,
                const Tensor& /*vbe_b_t_map*/,
                const c10::SymInt /*vbe_output_size*/,
                const int64_t /*info_B_num_bits*/,
                const int64_t /*info_B_mask_int64*/,
                const Tensor& /*vbe_B_offsets_rank_per_feature*/, // for reshaping vbe cpu offsets and output
                const Tensor& /*vbe_output_offsets_feature_rank*/, // for reshaping vbe cpu output
                const c10::SymInt /*max_B*/, // for reshaping vbe cpu offsets
                {%- endif %}
                {%- if is_gwd %}
                const Tensor& /*prev_iter_dev*/,
                const Tensor& /*learning_rate_tensor*/,
                const double /*weight_decay*/,
                const int64_t /*iter*/,
                const double /*gwd_lower_bound*/,
                {%- endif %}
                const bool /*is_experimental*/,
                const int64_t /*output_dtype*/
            )>();

    auto output = embedding_codegen_forward_op.call(
      weights_host,
      flatten_weights_dev,
      weights_uvm,
      weights_lxu_cache,
      weights_placements,
      weights_offsets,
      {%- if nobag %}
      D,
      {%- else %}
      D_offsets,
      total_D,
      max_D,
      {%- endif %}
      hash_size_cumsum,
      indices,
      offsets,
      {%- if not nobag %}
      pooling_mode,
      indice_weights_value,
      {%- endif %} {# /* if not nobag */ #}
      {%- if not dense %}
      {{ "ssd_tensors[SSDTensor::ROW_ADDRS]" if ssd else "lxu_cache_locations" }},
      uvm_cache_stats,
      {%- endif %}
      {%- if not nobag %}
      {%- if vbe %}
      vbe_row_output_offsets,
      vbe_b_t_map,
      vbe_output_size,
      info_B_num_bits,
      info_B_mask_int64,
      vbe_B_offsets_rank_per_feature_, // for reshaping vbe cpu offsets and output
      vbe_output_offsets_feature_rank_, // for reshaping vbe cpu output
      max_B_, // for reshaping vbe cpu offsets
      {%- endif %} {# /* if vbe */ #}
      {%- if is_gwd %}
      prev_iter_dev_,
      learning_rate_tensor,
      weight_decay,
      iter,
      gwd_lower_bound,
      {%- endif %} {# /* if is_gwd */ #}
      {%- endif %} {# /* if not nobag */ #}
      is_experimental,
      output_dtype
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
  {%- set backward_op = "{}_embedding{}_backward_codegen_{}{}{}{}_pt2_wrapper".format(
          bwd_mdesc,
          "_nobag" if nobag else "",
          optimizer,
          wdesc,
          "_vbe" if vbe else "",
          "_gwd" if is_gwd else "",
      )
  %}
    static auto embedding_codegen{{ wdesc }}_backward_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ backward_op }}", "")
            .typed<Tensor(
                const Tensor& /*grad_output*/,
                const Tensor& /*weights_host*/,
                const Tensor& /*weights_dev*/,
                const Tensor& /*weights_uvm*/,
                const Tensor& /*lxu_cache_weight*/,
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
                const Tensor& /*indice_weights*/, // currently supports no bag with unweighted
                {%- endif %}
                {%- if ssd %}
                const Tensor& /*ssd_row_addrs*/,
                {%- else %}
                const Tensor& /*lxu_cache_locations*/,
                {%- endif %}
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
                const Tensor& /*vbe_B_offsets_rank_per_feature*/, // for reshaping vbe cpu offsets and grad output
                const c10::SymInt /*max_B*/, // for reshaping vbe cpu offsets
                {%- endif %}
                const bool /*use_uniq_cache_locations_bwd*/,
                const bool /*use_homogeneous_placements*/,
                {%- if is_gwd %}
                {%- if "prev_iter_dev" not in args_pt2.split_function_arg_names %}
                const Tensor& /*prev_iter_dev*/,
                {%- endif %}
                {%- if "iter" not in args_pt2.split_function_arg_names %}
                const int64_t /*iter*/,
                {%- endif %}
                const double /*gwd_lower_bound*/,
                {%- endif %} {# /* if is_gwd */ #}
                {%- for arg_type in args_pt2.split_function_args %}
                {{ arg_type.split(' ')[0]}}{%- if not loop.last %}{{ "," }}{%- endif %}
                {%- endfor %}
                {%- if not nobag %}
                , const int64_t /*output_dtype*/
                {%- endif %}
            )>();

    grad_weights_dev = embedding_codegen{{ wdesc }}_backward_op.call(
          grad_output,
          {% if dense %}
          dev_weights,
          {% else %}
          weights_host,
          weights_dev,
          weights_uvm,
          weights_lxu_cache,
          weights_placements,
          weights_offsets,
          {% endif %}
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
          indice_weights,
          {%- endif %} {# /* if not nobag */ #}
          {%- if ssd %}
          ssd_row_addrs,
          {%- elif not dense %}
          lxu_cache_locations,
          {%- endif %}
          BT_block_size,
          max_segment_length_per_warp,
          {%- if not dense %}
          {%- if optimizer != "none" %}
          stochastic_rounding,
          {%- endif %}
          info_B_num_bits,
          info_B_mask_int64,
          {%- endif %} {# /* if not dense */ #}
          {%- if vbe %}
          B_offsets,
          vbe_row_output_offsets,
          vbe_b_t_map,
          vbe_B_offsets_rank_per_feature, // for reshaping vbe cpu offsets and grad output
          max_B, // for reshaping vbe cpu offsets
          {%- endif %} {# /* if vbe */ #}
          {%- if not dense %}
          use_uniq_cache_locations_bwd,
          use_homogeneous_placements,
          {%- endif %}
          {%- if is_gwd %}
          {%- if "prev_iter_dev" not in args_pt2.split_function_arg_names %}
          prev_iter_dev,
          {%- endif %}
          {%- if "iter" not in args_pt2.split_function_arg_names %}
          iter,
          {%- endif %}
          gwd_lower_bound,
          {%- endif %} {# /* if is_gwd */ #}
          {%- if dense %}
          /*unused=*/0
          {%- else %}
          {{ args_pt2.split_function_arg_names | join(", ") }}
          {%- endif %}
          {%- if not nobag %}
          , output_dtype
          {%- endif %}
    );

    if (is_annotate_trace_enabled) {
      record_trace->record.end();
    }

    // Number of returned gradients have to match the input to Autograd's forward
    // The number of items in the tensorlist differ between devices and is determined at runtime
    std::vector<Tensor> ret;

    {%- if not dense %}
    ret.push_back(Variable()); // placeholder autograd tensor
    {%- endif %}
    ret.push_back(Variable()); // output_dtype
    {%- if not dense %}
    if (weights_host.numel() > 0) {
      ret.push_back(Tensor()); // host_weights
    }
    else {
      ret.push_back(grad_weights_dev); // dev_weights
      ret.push_back(Variable()); // weights_uvm
      ret.push_back(Variable()); // weights_lxu_cache
    }
    ret.push_back(Variable()); // weights_placement
    {%- endif %}
    ret.push_back(Variable()); // weights_offsets
    {%- if nobag %}
    ret.push_back(Variable()); // D
    {%- else %}
    ret.push_back(Variable()); // D_offsets
    ret.push_back(Variable()); // total_D
    ret.push_back(Variable()); // max_D
    {%- endif %}
    ret.push_back(Variable()); // hash_size_cumsum
    ret.push_back(Variable()); // total_hash_size_bits
    ret.push_back(Variable()); // indices
    ret.push_back(Variable()); // offsets
    {%- if not nobag %}
    ret.push_back(Variable()); // pooling_mode
    ret.push_back(grad_indice_weights); // indice_weights
    ret.push_back(Variable()); // feature_requires_grad
    {%- endif %}
    {%- if vbe %}
    {%- if dense %}
    ret.push_back(Variable()); // B_offsets
    ret.push_back(Variable()); // vbe_output_offsets_feature_rank
    ret.push_back(Variable()); // vbe_B_offsets_rank_per_feature
    {%- endif %} {# /* if dense */ #}
    ret.push_back(Variable()); // max_B
    ret.push_back(Variable()); // max_B_feature_rank
    ret.push_back(Variable()); // vbe_output_size
    {%- endif %} {# /* if vbe */ #}
    {%- if not dense %}
    ret.push_back(Variable()); // aux_tensor
    ret.push_back(Variable()); // aux_int
    ret.push_back(Variable()); // aux_float
    ret.push_back(Variable()); // aux_bool
    {%- endif %}
    {%- if ssd %}
    {%- for tensor in ssd_tensors %}
    ret.push_back(Variable()); // {{ tensor }}
    {%- endfor %}
    {%- endif %}
    {{ args_pt2.unified_pt2.split_variables | join("\n") }}
    return ret;
{%- endmacro %}

/* This macro generates a code blob that calls corresponding autograd function
    from lookup_function
*/
{%- macro call_autograd(nobag, vbe, is_gwd) %}
    {%- set autograd_func = "{}{}{}{}LookupFunction_{}_Op_pt2".format(
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
          {%- if dense %}
          dev_weights,
          weights_offsets,
          {%- else %}
          weights,
          {%- endif %}
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
          {%- if dense and nobag %}
          offsets
          {%- else %}
          offsets,
          {%- endif %}
          {%- if not nobag %}
          pooling_mode,
          indice_weights,
          {%- if dense and not vbe %}
          feature_requires_grad
          {%- else %}
          feature_requires_grad,
          {%- endif %}
          {%- endif %}
          {%- if vbe %}
          {%- if dense %}
          B_offsets,
          vbe_output_offsets_feature_rank,
          vbe_B_offsets_rank_per_feature,
          {%- endif %} {# /* if dense */ #}
          max_B,
          max_B_feature_rank,
          vbe_output_size,
          {%- endif %} {# /* if vbe */ #}
          {%- if not dense %}
          aux_tensor,
          aux_int,
          aux_float,
          aux_bool,
          {%- if ssd %}
          ssd_tensors.value(),
          {%- endif  %}
          {{ args_pt2.unified_pt2.split_function_arg_names | join(", ") }}
          {%- endif %}
          )[0];
{%- endmacro %}

/* This macro generates a code blob for unpacking TensorList
*/
{%- macro unpack_tensorlist(name) %}
    Tensor {{ name }}_host;
    Tensor {{ name }}_dev;
    Tensor {{ name }}_uvm;
    Tensor {{ name }}_placements;
    Tensor {{ name }}_offsets;
    {%- if name == "weights" %}
    Tensor {{ name }}_lxu_cache;
    {%- endif %}

    if ({{ name }}.size() == 3) {
      TENSOR_ON_CPU_OR_MTIA({{ name }}[0]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE({{ name }}[0], {{ name }}[1]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE({{ name }}[0], {{ name }}[2]);
      {{ name }}_host = {{ name }}[0];
      {{ name }}_placements = {{ name }}[1];
      {{ name }}_offsets = {{ name }}[2]; 
    }
    else if ({{ name }}.size() == {{ 5 if name == "weights" else 4 }})  {
      TENSOR_ON_CUDA_GPU({{ name }}[0]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE({{ name }}[0], {{ name }}[1]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE({{ name }}[0], {{ name }}[2]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE({{ name }}[0], {{ name }}[3]);
      {%- if name == "weights" %}
      TENSORS_EMPTY_OR_ON_SAME_DEVICE({{ name }}[0], {{ name }}[4]);
      {%- endif %}
      {{ name }}_dev = {{ name }}[0]; 
      {{ name }}_uvm = {{ name }}[1];
      {{ name }}_placements = {{ name }}[2];
      {{ name }}_offsets = {{ name }}[3];
      {%- if name == "weights" %}
      {{ name }}_lxu_cache = {{ name }}[4];
      {%- endif %}
    }
    else {
      TORCH_CHECK(false, "Invalid size of {{ name }}, expected 3 for CPU or {{ 5 if name == "weights" else 4 }} for CUDA but got ", {{ name }}.size());
    }
{%- endmacro %}

/* This macro generates a code blob for creating tensor that is unpacked from list of optional tensors*/
{%- macro get_optional_optim_tensor(name, suffix, idx, options) %}
  auto {{ name }}_{{ suffix }} = GET_OPTIONAL_TENSOR_VALUE(optim_tensor[{{ idx }}], at::empty({0}, {{ options }}));
{%- endmacro %}

/* This macro generates a code blob for unpacking a list of optional tensors
    We cannot do list of optional tensorlist. We need to pack optimizer optional tensors in a flatten manner.
    For readability and programmability, we pass all unified args (i.e., 5 items), as opposed to passing per device (like above)
    which needs to be determined at runtime.
*/
{%- macro unpack_tensorlist_optional(name, arg_index) %}
  at::TensorOptions options = weights_host.numel() > 0 ? weights_host.options() : weights_dev.options();
  {{ get_optional_optim_tensor(name, "host", arg_index * 5, "options") }}
  {{ get_optional_optim_tensor(name, "dev", arg_index * 5 + 1, "options") }}
  options = weights_host.numel() > 0 ? weights_host.options() : weights_uvm.options();
  {{ get_optional_optim_tensor(name, "uvm", arg_index * 5 + 2, "options") }}
  {{ get_optional_optim_tensor(name, "placements", arg_index * 5 + 3, "weights_placements.options()") }}
  {{ get_optional_optim_tensor(name, "offsets", arg_index * 5 + 4, "weights_offsets.options()") }}
{%- endmacro %}

////////////////////////////////////////////////////////////////////////////////
// MACROS 
////////////////////////////////////////////////////////////////////////////////
#define GET_OPTIONAL_TENSOR_VALUE(name, empty_tensor) name.has_value() ? name.value() : empty_tensor;

////////////////////////////////////////////////////////////////////////////////
// Autograd Function Declarations
////////////////////////////////////////////////////////////////////////////////

{%- if has_gpu_support or has_cpu_support %}

{%- for vbe in ([True, False] if has_vbe_support else [False]) %}
{%- set vdesc = "_vbe" if vbe else "" %}

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

{%- set autograd_func = "{}{}{}{}LookupFunction_{}_Op_pt2".format(
      "SSD" if ssd else "Split",
      "NoBag" if nobag else "",
      "VBE" if vbe else "",
      "GWD" if is_gwd else "",
      optimizer,
    )
%}

{#-/* This unified Autograd function
      `Split{nobag}{vbe}LookupFunction_{optimizer}_Op_pt2` calls the
      wrapper ops (e.g.,
      `split_embedding_codegen_forward_{wdesc}{vbe}_pt2`) and
      dispatches to the version of the op based on input device type.
*/#}
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
    {%- if dense %}
    const Tensor& dev_weights,
    const Tensor& weights_offsets,
    {%- else %}
    const at::TensorList weights,
    {%- endif %}
    {%- if not nobag %}
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    {%- else %}
    const c10::SymInt D,
    {%- endif %}
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    {%- if not nobag %}
    const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights,
    const std::optional<Tensor>& feature_requires_grad,
    {%- endif %}
    {%- if vbe %}
    {%- if dense %}
    const std::optional<Tensor>& B_offsets,
    const std::optional<Tensor>& vbe_output_offsets_feature_rank,
    const std::optional<Tensor>& vbe_B_offsets_rank_per_feature,
    {%- endif %} {# /* if dense */ #}
    const c10::SymInt max_B,
    const c10::SymInt max_B_feature_rank,
    const c10::SymInt vbe_output_size,
    {%- endif %} {# /* if vbe */ #}
    {%- if not dense %}
    std::vector<std::optional<at::Tensor>> aux_tensor,
    std::vector<int64_t> aux_int,
    std::vector<double> aux_float,
    c10::List<bool> aux_bool,
    {%- endif %}
    {%- if ssd %}
    const at::TensorList& ssd_tensors,
    {%- endif %}
    {{ args_pt2.unified_pt2.split_function_args | join(", ") }}) {

    // unpack Tensor lists
    {{ unpack_tensorlist("weights") }}
    {%- for arg_name in args_pt2.unified_pt2.split_saved_tensorlist %}
    {{ unpack_tensorlist(arg_name) }}
    {%- endfor %}
    {%- if "optim_tensor" in args_pt2.unified_pt2.split_function_arg_names %}
    TORCH_CHECK(optim_tensor.size() % 5 == 0);
    {%- endif %}
    {%- for arg_name in args_pt2.unified_pt2.split_saved_tensorlist_optional %}
    {{ unpack_tensorlist_optional(arg_name, loop.index0) }}
    {%- endfor %}

    const auto T = weights_offsets.sym_numel();

    {%- if vbe %}
    const auto B_offsets_ = GET_OPTIONAL_TENSOR_VALUE(aux_tensor[IDX_B_OFFSETS], Tensor());
    const auto vbe_output_offsets_feature_rank_ = GET_OPTIONAL_TENSOR_VALUE(aux_tensor[IDX_VBE_OUTPUT_OFFSETS_FEATURE_RANK], Tensor());
    const auto vbe_B_offsets_rank_per_feature_ = GET_OPTIONAL_TENSOR_VALUE(aux_tensor[IDX_VBE_B_OFFSETS_RANK_PER_FEATURE], Tensor());
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
    at::TensorOptions uvm_options = weights_host.numel() > 0 ? weights_host.options() : weights_dev.options();
    const auto uvm_cache_stats = GET_OPTIONAL_TENSOR_VALUE(aux_tensor[IDX_UVM_CACHE_STATS], at::empty({0}, uvm_options.dtype(at::kInt)));
    TORCH_CHECK(aux_tensor[IDX_LXU_CACHE_LOCATIONS].has_value(), "lxu_cache_locations should have value.");
    const auto lxu_cache_locations = aux_tensor[IDX_LXU_CACHE_LOCATIONS].value();
    const auto is_experimental = aux_bool[IDX_IS_EXPERIMENTAL_TBE];
    {%- endif %}

    // Default values for Dynamo tracing
    // SymInt does not support bitshifts operator
    // Constanting info_B_num_bits, info_B_mask for Dynamo for now.
    const auto info_B_num_bits = static_cast<int32_t>(aux_int[IDX_INFO_B_NUM_BITS]);
    const auto info_B_mask = static_cast<uint32_t>(aux_int[IDX_INFO_B_MASK]);

    {%- if vbe %}
    static auto generate_vbe_metadata_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::generate_vbe_metadata", "")
            .typed<std::tuple<Tensor, Tensor>(
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const Tensor&,
                const int64_t,
                const bool,
                const c10::SymInt,
                const int64_t,
                const c10::SymInt)>();
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
    {%- endif %} // vbe

    {%- if is_gwd %}
    {%- if "prev_iter" in args_pt2.unified_pt2.split_function_arg_names %}
    const auto prev_iter_dev_ = GET_OPTIONAL_TENSOR_VALUE(prev_iter_dev, Tensor());
    {%- else %}
    const auto prev_iter_dev_ = GET_OPTIONAL_TENSOR_VALUE(aux_tensor[IDX_PREV_ITER_DEV], Tensor());
    {%- endif %}
    {%- endif %}

    {%- if not nobag %}
    const auto indice_weights_value = GET_OPTIONAL_TENSOR_VALUE(indice_weights, Tensor());
    {%- endif %}

    ctx->save_for_backward({
        {%- if dense %}
        dev_weights,
        weights_offsets,
        {%- else %}
        weights_host,
        weights_dev,
        weights_uvm,
        weights_lxu_cache,
        weights_placements,
        weights_offsets,
        {%- endif %}
        {%- if not nobag %}
        D_offsets,
        {%- endif %}
        hash_size_cumsum,
        indices,
        offsets,
        {%- if not nobag %}
        indice_weights_value,
        feature_requires_grad.value_or(Tensor()),
        {%- endif %}
        {%- if not dense %}
        lxu_cache_locations,
        {%- endif %}
        {%- if vbe %}
        B_offsets_,
        vbe_row_output_offsets,
        vbe_b_t_map,
        vbe_B_offsets_rank_per_feature_, // for reshaping vbe cpu grad_output
        {%- endif %}
        {%- if is_gwd and "prev_iter_dev" not in args_pt2.split_function_arg_names %}
        prev_iter_dev_,
        {%- endif %}
        {%- if ssd %}
        {%- for tensor in ssd_tensors %}
        ssd_tensors[SSDTensor::{{ tensor | upper }}],
        {%- endfor %}
        {%- endif %}
        {{ args_pt2.split_saved_tensors | join(", ") }}
    });

    {%- if not nobag %}
    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["mixed_D"] = static_cast<bool>(aux_bool[IDX_MIXED_D]);
    ctx->saved_data["pooling_mode"] = pooling_mode;
    {%- else %}
    ctx->saved_data["D"] = D;
    {%- endif %}
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;

    {%- if optimizer != "none" and not dense %}
    ctx->saved_data["gradient_clipping"] = static_cast<bool>(aux_bool[IDX_GRADIENT_CLIPPING]);
    ctx->saved_data["max_gradient"] = aux_float[IDX_MAX_GRADIENT];
    ctx->saved_data["stochastic_rounding"] = static_cast<bool>(aux_bool[IDX_STOCHASTIC_ROUNDING]);
    {%- endif %} {#-/* if optimizer != "none" */#}
    ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
    const auto info_B_mask_int64 = static_cast<int64_t>(info_B_mask);
    ctx->saved_data["info_B_mask"] = info_B_mask_int64;
    {%- if not dense %}
    ctx->saved_data["use_uniq_cache_locations_bwd"] = static_cast<bool>(aux_bool[IDX_USE_UNIQ_CACHE_LOCATIONS_BWD]);
    ctx->saved_data["use_homogeneous_placements"] = static_cast<bool>(aux_bool[IDX_USE_HOMOGENEOUS_PLACEMENTS]);
    {%- endif %}
    const auto iter = aux_int[IDX_ITER];
    ctx->saved_data["iter"] = iter;
    {%- if is_gwd %}
    const auto gwd_lower_bound = aux_float[IDX_GWD_LOWER_BOUND];
    ctx->saved_data["gwd_lower_bound"] = gwd_lower_bound;
    {%- endif %}
    {%- if not nobag %}
    ctx->saved_data["output_dtype"] = output_dtype;
    {%- endif %}
    {%- if vbe %}
    ctx->saved_data["max_B"] = max_B_; // for reshaping vbe cpu offsets and grad_output 
    {%- endif %}

    {%- if not dense %}
    // unpack optim args
    {%- for (var, dict_val, _, type) in args_pt2.unified_pt2.split_saved_data %}
    {%- if type == "bool" %}
    bool {{ var }} = {{ dict_val }};
    {%- elif type != "c10::SymInt" %}
    auto {{ var }} = {{ dict_val }};
    {%- endif %}
    ctx->saved_data["{{ var }}"] = {{ var }};
    {%- endfor %}
    {%- endif %}

    {%- if optimizer == "none" %}
    // Flatten
    const auto& flatten_weights_dev = weights_dev.flatten();
    {%- else %}
    const auto& flatten_weights_dev = weights_dev;
    {%- endif %}
    {%- if nobag %}
    // nobag
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
        // weighted
      {{
         call_forward_op_dispatch(
             nobag=False,
             weighted=True,
             vbe=vbe,
             is_gwd=is_gwd,
         )
       }}
    }
    // unweighted
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
    {%- if dense %}
    auto dev_weights = *savedItr++;
    auto weights_offsets = *savedItr++;
    {%- else %}
    auto weights_host = *savedItr++;
    auto weights_dev = *savedItr++;
    auto weights_uvm = *savedItr++;
    auto weights_lxu_cache = *savedItr++;
    auto weights_placements = *savedItr++;
    auto weights_offsets = *savedItr++;
    {%- endif %}
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
    auto vbe_B_offsets_rank_per_feature = *savedItr++; // for reshaping vbe cpu grad_output
    {%- endif %}
    {%- if is_gwd and "prev_iter_dev" not in args_pt2.split_function_arg_names %}
    auto prev_iter_dev = *savedItr++;
    {%- endif %}
    {%- if ssd %}
    {%- for tensor in ssd_tensors %}
    auto ssd_{{ tensor }} = *savedItr++;
    {%- endfor %}
    {%- endif %}

    {%- for tensor in args_pt2.split_saved_tensors %}
    auto {{ tensor }} = *savedItr++;
    {%- endfor %}

    {%- if not nobag %}
    auto max_D = ctx->saved_data["max_D"].toSymInt();
    const auto mixed_D = ctx->saved_data["mixed_D"].toBool();
    auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();
    {%- else %}
    auto D = ctx->saved_data["D"].toInt();
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
    const auto use_uniq_cache_locations_bwd = ctx->saved_data["use_uniq_cache_locations_bwd"].toBool();
    const auto use_homogeneous_placements = ctx->saved_data["use_homogeneous_placements"].toBool();
    {%- endif %}
    {%- if is_gwd or "iter" in args_pt2.unified_pt2.split_unpacked_arg_names %}
    const auto iter = ctx->saved_data["iter"].toInt();
    {%- endif %}
    {%- if is_gwd %}
    const auto gwd_lower_bound = ctx->saved_data["gwd_lower_bound"].toDouble();
    {%- endif %}
    
    {%- if not nobag %}
    auto output_dtype = ctx->saved_data["output_dtype"].toInt();
    {%- endif %}
    {%- if not dense %}
    {%- if vbe %}
    auto max_B = ctx->saved_data["max_B"].toSymInt(); // for reshaping vbe cpu offsets and grad_output
    {%- endif %}

    {%- for (var, _ , ivalue_cast, type) in args_pt2.unified_pt2.split_saved_data %}
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
    // Flatten (weights_dev is used in
    // {{ fwd_mdesc }}_embedding_codegen_grad_indice_weights{{ vdesc }}_pt2_cuda)
    weights_dev = weights_dev.flatten();
    {%- endif %}

    {%- set grad_indice_weights_op =
        "{}_embedding_codegen_grad_indice_weights{}_pt2_wrapper".format(fwd_mdesc, vdesc)
    %}
    static auto embedding_codegen_grad_indice_weights_op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::{{ grad_indice_weights_op }}", "")
            .typed<Tensor(
                const Tensor& /*grad_output*/,
                {%- if dense %}
                const Tensor& /*dev_weights*/,
                const Tensor& /*weights_offsets*/,
                {%- else %}
                const Tensor& /*weights_host*/,
                const Tensor& /*weights_dev*/,
                const Tensor& /*weights_uvm*/,
                const Tensor& /*weights_lxu_cache*/,
                const Tensor& /*weights_placements*/,
                const Tensor& /*weights_offsets*/,
                 {%- endif %}
                const Tensor& /*D_offsets*/,
                const c10::SymInt /*max_D*/,
                const Tensor& /*indices*/,
                const Tensor& /*offsets*/,
                {%- if ssd %}
                const Tensor& /*ssd_row_addrs*/,
                {%- elif not dense %}
                const Tensor& /*lxu_cache_locations*/,
                {%- endif %}
                {%- if vbe %}
                const Tensor& /*feature_requires_grad*/,
                const Tensor& /*vbe_row_output_offsets*/,
                const Tensor& /*vbe_b_t_map*/,
                const int64_t /*info_B_num_bits*/,
                const int64_t /*info_B_mask_int64*/,
                const Tensor& /*vbe_B_offsets_rank_per_feature*/, // for reshaping vbe cpu grad_output
                const c10::SymInt /*max_B*/ // for reshaping vbe cpu offsets and grad_output
                {%- else %}
                const Tensor& /*feature_requires_grad*/
                {%- endif %}
            )>();

    const auto grad_indice_weights = !indice_weights.defined() ?
      Variable() :
      embedding_codegen_grad_indice_weights_op.call(
        grad_output,
        weights_host,
        weights_dev,
        weights_uvm,
        weights_lxu_cache,
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
        info_B_mask_int64,
        vbe_B_offsets_rank_per_feature,
        max_B
        {%- else %}
        feature_requires_grad
        {%- endif %}
        );
    
    Tensor grad_weights_dev;
    // weighted
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
    // unweighted
    {{
       call_backward_op_dispatch(
           nobag=False,
           weighted=False,
           vbe=vbe,
           is_gwd=is_gwd,
      )
    }}
    {%- else %} {#-/* if not nobag */#}
    // nobag
    Tensor grad_weights_dev;
      {{
         call_backward_op_dispatch(
             nobag=True,
             weighted=False,
             vbe=vbe,
             is_gwd=is_gwd,
        )
      }}
    {%- endif %} {#-/* if not nobag */#}

}
};
{%- endfor %} {#-/* for is_gwd */#}
{%- endfor %} {#-/* for nobag in [True, False] */#}
{%- endfor %} {#-/* for vbe in [True, False] */#}

///@ingroup embedding-cuda
Tensor {{ bwd_mdesc }}_embedding_codegen_lookup_{{ optimizer }}_function_pt2(
    {%- if dense %}
    const Tensor& dev_weights,
    const Tensor& weights_offsets,
    {%- else %}
    const Tensor& placeholder_autograd_tensor,
    const at::TensorList weights,
    {%- endif %}
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
    const int64_t output_dtype,
    {%- if not dense %}
    const std::vector<std::optional<at::Tensor>>& aux_tensor,
    const std::vector<int64_t>& aux_int,
    const std::vector<double>& aux_float,
    c10::List<bool> aux_bool,
    {%- endif %}
    {{ args_pt2.unified_pt2.split_function_args | join(", ") }},
    const c10::SymInt max_B = -1,
    const c10::SymInt max_B_feature_rank = -1,
    {%- if ssd %}
    const c10::SymInt vbe_output_size = -1,
    const std::optional<at::TensorList>& ssd_tensors = std::nullopt
    {%- else %}
    const c10::SymInt vbe_output_size = -1
    {%- endif %}
) {

  {%- if has_gpu_support or has_cpu_support %}

    {%- if not dense %}
    // Load the config value from JK once
    static auto is_tbev2_enabled = config::is_feature_enabled(config::FeatureGateName::TBE_V2);

    // Set to experimental if either the feature is enabled in JK, or the user specifies to use TBEv2
    aux_bool[IDX_IS_EXPERIMENTAL_TBE] = is_tbev2_enabled || aux_bool[IDX_IS_EXPERIMENTAL_TBE];
    {%- endif %}

    {%- if ssd %}
    TORCH_CHECK(
        ssd_tensors.value().size() == {{ ssd_tensors | length }},
        "SSD TBE expects {{ ssd_tensors | length }} in ssd_tensors");
    {%- endif %}

    {%- if has_vbe_support %}
    // has vbe support and on gpu
    if (aux_tensor[IDX_B_OFFSETS].has_value()) {
      {%- if has_global_weight_decay_support and not ssd %}
        // vbe and has gwd support
        // if weight_decay arg is not passed or < 0 even though apply_global_weight_decay is True, we don't do gwd
        // TODO: add check to ensure weight decay exists
        if (aux_bool[IDX_APPLY_GLOBAL_WEIGHT_DECAY] && optim_float[{{args_pt2.unified_pt2.split_args_dict["optim_float"].index("weight_decay")}}] > 0) {
          {{ call_autograd(nobag=False, vbe=True, is_gwd=True) }}
        }
      {%- endif %} {#-/* if has_global_weight_decay_support */ #}
      // vbe and no gwd support
      {{ call_autograd(nobag=False, vbe=True, is_gwd=False) }}
    }
    {%- endif %} {#-/* if has_vbe_support */ #}

    {%- if has_global_weight_decay_support and not ssd %}
    // has gwd support
     if (aux_bool[IDX_APPLY_GLOBAL_WEIGHT_DECAY] && optim_float[{{args_pt2.unified_pt2.split_args_dict["optim_float"].index("weight_decay")}}] > 0) {
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
  {%- endif %} 
}


TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    {%- set op_name = "{}_embedding_codegen_lookup_{}_function_pt2".format(fwd_mdesc, optimizer) %}
    m.def("{{ op_name }}("
        {%- if dense %}
        "    Tensor dev_weights, "
        "    Tensor weights_offsets, "
        {%- else %}
        "    Tensor placeholder_autograd_tensor, "
        "    Tensor[]{{ schema_annotation['weights'] }} weights, "
        {%- endif %}
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
        "    int output_dtype, "
        {%- if not dense %}
        "    Tensor?[]{{ schema_annotation['aux_tensor'] }} aux_tensor, "
        "    int[] aux_int, "
        "    float[] aux_float, "
        "    bool[] aux_bool, "
        "    {{ args_pt2.unified_pt2.split_function_schemas | join(", ") }}, "
        "    SymInt max_B=-1, "
        "    SymInt max_B_feature_rank=-1, "
        {%- if ssd %}
        "    SymInt vbe_output_size=-1, "
        "    Tensor[]? ssd_tensors=None"
        {%- else %}
         "    SymInt vbe_output_size=-1 "
        {%- endif %}
        {%- endif %}
        ") -> Tensor",
        {PT2_COMPLIANT_TAG});
    // We're playing a funny trick here: we're using the autograd
    // implementation of the operator at all the dispatch keys.  This is OK
    // because autograd function works even in a context where there is
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
    DISPATCH_TO_CUDA(
        " {{ op_name }} ",
        {{ op_name }} );
}

{%- endif %} {#-/* if has_gpu_support or has_cpu_support */#}
