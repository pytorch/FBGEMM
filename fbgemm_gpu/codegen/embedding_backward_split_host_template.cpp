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

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{% if has_gpu_support %}
/// @defgroup embedding-cuda Embedding CUDA Operators

{% for vbe in ([True, False] if has_vbe_support else [False]) %}
{% set vbe_desc = "_vbe" if vbe else "" %}
Tensor split_embedding_codegen_forward_unweighted{{ vbe_desc }}_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor lxu_cache_locations,
    int64_t output_dtype,
    {% if vbe %}
    const VBEMetadata& vbe_metadata,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {% endif %}
    bool is_experimental);

Tensor split_embedding_codegen_forward_weighted{{ vbe_desc }}_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    Tensor lxu_cache_locations,
    int64_t output_dtype,
    {% if vbe %}
    const VBEMetadata& vbe_metadata,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {% endif %}
    bool is_experimental);

Tensor split_embedding_codegen_grad_indice_weights{{ vbe_desc }}_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    Tensor lxu_cache_locations,
    {% if vbe %}
    Tensor feature_requires_grad,
    const VBEMetadata& vbe_metadata,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask
    {% else %}
    Tensor feature_requires_grad
    {% endif %}
);

Tensor split_embedding_backward_codegen_{{ optimizer }}_unweighted_exact{{ vbe_desc }}_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor lxu_cache_locations,
    int64_t BT_block_size,
    int64_t max_segment_length_per_warp,
    {% if optimizer != "none" %}
    bool stochastic_rounding,
    {% endif %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {% if vbe %}
    const VBEMetadata& vbe_metadata,
    {% endif %}
    {{ args.split_function_args | join(", ") }});

Tensor split_embedding_backward_codegen_{{ optimizer }}_weighted_exact{{ vbe_desc }}_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
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
    Tensor lxu_cache_locations,
    int64_t BT_block_size,
    int64_t max_segment_length_per_warp,
    {% if optimizer != "none" %}
    bool stochastic_rounding,
    {% endif %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {% if vbe %}
    const VBEMetadata& vbe_metadata,
    {% endif %}
    {{ args.split_function_args | join(", ") }});

{% if not vbe %}
Tensor split_embedding_nobag_codegen_forward_unweighted_cuda(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    int64_t D,
    Tensor indices,
    Tensor offsets,
    Tensor lxu_cache_locations,
    int64_t output_dtype,
    bool is_experimental);

Tensor split_embedding_nobag_backward_codegen_{{ optimizer }}_unweighted_exact_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    int64_t D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    Tensor lxu_cache_locations,
    int64_t BT_block_size,
    int64_t max_segment_length_per_warp,
    {% if optimizer != "none" %}
    bool stochastic_rounding,
    {% endif %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {{ args.split_function_args | join(", ") }});
{% endif %} // if not vbe

{% for nobag in [True, False] %}
{% if not nobag or not vbe %} // nobag does not support vbe
class Split{{ "NoBag" if nobag else "" }}{{ "VBE" if vbe else "" }}LookupFunction_{{ optimizer }}_Op :
    public torch::autograd::Function<
        Split{{ "NoBag" if nobag else "" }}{{ "VBE" if vbe else "" }}LookupFunction_{{ optimizer }}_Op
    > {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    Tensor placeholder_autograd_tensor,
    int64_t output_dtype,
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    {% if not nobag %}
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    {% else %}
    int64_t D,
    {% endif %}
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    {% if not nobag %}
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    c10::optional<Tensor> feature_requires_grad,
    {% endif %}
    Tensor lxu_cache_locations,
    {% if optimizer != "none" %}
    bool gradient_clipping,
    double max_gradient,
    bool stochastic_rounding,
    {% endif %}
    {% if vbe %}
    const c10::optional<Tensor>& B_offsets,
    const c10::optional<Tensor>& vbe_output_offsets_feature_rank,
    const c10::optional<Tensor>& vbe_B_offsets_rank_per_feature,
    const int32_t max_B,
    const int32_t max_B_feature_rank,
    const int64_t vbe_output_size,
    {% endif %}
    bool is_experimental,
    {{ args.split_function_args | join(", ") }}) {

    const auto T = weights_offsets.numel();
    {% if vbe %}
    struct VBEMetadata vbe_metadata = {
      .B_offsets = B_offsets.value_or(Tensor()),
      .output_offsets_feature_rank = vbe_output_offsets_feature_rank.value_or(Tensor()),
      .B_offsets_rank_per_feature = vbe_B_offsets_rank_per_feature.value_or(Tensor()),
      .max_B_feature_rank = max_B_feature_rank,
      .output_size = vbe_output_size
    };
    const auto max_B_ = max_B;
    {% else %}
    const auto max_B_ = offsets.size(0) / T;
    {% endif %}

    int32_t info_B_num_bits;
    uint32_t info_B_mask;
    std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(max_B_, T);

    {% if vbe %}
    populate_vbe_metadata_foreach_sample_inplace(
        vbe_metadata,
        {% if not nobag %}
        D_offsets,
        /*D=*/-1,
        /*nobag=*/false,
        {% else %}
        // weights_placements has the same options as D_offsets
        at::empty({0}, weights_placements.options()),
        D,
        /*nobag=*/true,
        {% endif %}
        info_B_num_bits,
        /*total_B=*/offsets.size(0) - 1
        );
    {% endif %}

    ctx->save_for_backward({
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        {% if not nobag %}
        D_offsets,
        {% endif %}
        hash_size_cumsum,
        indices,
        offsets,
        {% if not nobag %}
        indice_weights.value_or(Tensor()),
        feature_requires_grad.value_or(Tensor()),
        {% endif %}
        lxu_cache_locations,
        {% if vbe %}
        vbe_metadata.B_offsets,
        vbe_metadata.output_offsets,
        vbe_metadata.b_t_map,
        {% endif %}
        {{ args.split_saved_tensors | join(", ") }}
    });

    {% if not nobag %}
    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["pooling_mode"] = pooling_mode;
    {% else %}
    ctx->saved_data["D"] = D;
    {% endif %}
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;

    {% if optimizer != "none" %}
    ctx->saved_data["gradient_clipping"] = gradient_clipping;
    ctx->saved_data["max_gradient"] = max_gradient;
    ctx->saved_data["stochastic_rounding"] = stochastic_rounding;
    {% endif %} // if optimizer != "none"
    ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
    uint64_t info_B_mask_64 = info_B_mask;
    ctx->saved_data["info_B_mask"] = *reinterpret_cast<int64_t*>(&info_B_mask_64);

    {% for (var, _) in args.saved_data %}
    ctx->saved_data["{{ var }}"] = {{ var }};
    {% endfor %}

    {% if optimizer == "none" %}
    // Flatten
    dev_weights = dev_weights.flatten();
    {% endif %}

    {% if not nobag %}
    if (!indice_weights) {
        return {
          split_embedding_codegen_forward_unweighted{{ vbe_desc }}_cuda(
            dev_weights,
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
            lxu_cache_locations,
            output_dtype,
            {% if vbe %}
            vbe_metadata,
            info_B_num_bits,
            info_B_mask,
            {% endif %}
            is_experimental
          )
        };
    } else {
        return {
          split_embedding_codegen_forward_weighted{{ vbe_desc }}_cuda(
            dev_weights,
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
            *indice_weights,
            lxu_cache_locations,
            output_dtype,
            {% if vbe %}
            vbe_metadata,
            info_B_num_bits,
            info_B_mask,
            {% endif %}
            is_experimental
          )
        };
    }
    {% else %}
    return {
      split_embedding_nobag_codegen_forward_unweighted_cuda(
        dev_weights,
        uvm_weights,
        lxu_cache_weights,
        weights_placements,
        weights_offsets,
        D,
        indices,
        offsets,
        lxu_cache_locations,
        output_dtype,
        /*is_experimental=*/false
      )
    };
    {% endif %}
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
    {% if not nobag %}
    auto D_offsets = *savedItr++;
    {% endif %}
    auto hash_size_cumsum = *savedItr++;
    auto indices = *savedItr++;
    auto offsets = *savedItr++;
    {% if not nobag %}
    auto indice_weights = *savedItr++;
    auto feature_requires_grad = *savedItr++;
    {% endif %}
    auto lxu_cache_locations = *savedItr++;
    {% if vbe %}
    auto B_offsets = *savedItr++;
    auto vbe_output_offsets = *savedItr++;
    auto vbe_b_t_map = *savedItr++;
    {% endif %}

    {% for tensor in args.split_saved_tensors %}
    auto {{ tensor }} = *savedItr++;
    {% endfor %}

    {% if not nobag %}
    auto max_D = ctx->saved_data["max_D"].toInt();
    auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();
    {% else %}
    auto D = ctx->saved_data["D"].toInt();
    {% endif %}
    auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();

    {% if optimizer != "none" %}
    auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
    auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
    auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();
    {% endif %} // if optimizer != "none"
    const int32_t info_B_num_bits = ctx->saved_data["info_B_num_bits"].toInt();
    const int64_t info_B_mask_64 = ctx->saved_data["info_B_mask"].toInt();
    const uint32_t info_B_mask = *reinterpret_cast<const uint64_t*>(&info_B_mask_64);

    {% for (var, ivalue_cast) in args.saved_data %}
    auto {{ var }} = ctx->saved_data["{{ var }}"].{{ ivalue_cast }}();
    {% endfor %}

    TORCH_CHECK_EQ(grad_outputs.size(), 1);

#ifdef __HIP_PLATFORM_HCC__
    constexpr int32_t BT_block_size = 64;
    constexpr int32_t max_segment_length_per_warp = 64;
#else
    constexpr int32_t BT_block_size = 32;
    constexpr int32_t max_segment_length_per_warp = 32;
#endif
    using torch::autograd::Variable;

    {% if optimizer != "none" %}
    auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];
    {% else %}
    auto& grad_output = grad_outputs[0];
    {% endif %}

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

    {% if vbe %}
    struct VBEMetadata vbe_metadata = {
      .B_offsets = B_offsets,
      .output_offsets = vbe_output_offsets,
      .b_t_map = vbe_b_t_map,
    };
    {% endif %}

    {% if not nobag %}
    {% if optimizer == "none" %}
    // Flatten (dev_weights is used in
    // split_embedding_codegen_grad_indice_weights{{ vbe_desc }}_cuda)
    dev_weights = dev_weights.flatten();
    {% endif %}
    const auto grad_indice_weights = !indice_weights.defined() ?
      Variable() :
      split_embedding_codegen_grad_indice_weights{{ vbe_desc }}_cuda(
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
        {% if vbe %}
        feature_requires_grad,
        vbe_metadata,
        info_B_num_bits,
        info_B_mask
        {% else %}
        feature_requires_grad
        {% endif %}
        );
    const auto grad_dev_weights = !indice_weights.defined() ?
      {% for weighted in [False, True] %}
      split_embedding_backward_codegen_{{ optimizer }}_{{ "weighted" if weighted else "unweighted" }}_exact{{ vbe_desc }}_cuda(
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
          {% if weighted %}
          indice_weights,
          {% endif %}
          lxu_cache_locations,
          BT_block_size,
          max_segment_length_per_warp,
          {% if optimizer != "none" %}
          stochastic_rounding,
          {% endif %}
          info_B_num_bits,
          info_B_mask,
          {% if vbe %}
          vbe_metadata,
          {% endif %}
          {{ args.split_function_arg_names | join(", ") }}
      ) {{ ":" if not weighted else ";" }}
      {% endfor %} // for weighted in [False, True]
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
        {% if optimizer != "none" %}
        Variable(), // gradient_clipping
        Variable(), // max_gradient
        Variable(), // stochastic_rounding
        {% endif %}
        {% if vbe %}
        Variable(), // B_offsets
        Variable(), // vbe_output_offsets_feature_rank
        Variable(), // vbe_B_offsets_rank_per_feature
        Variable(), // max_B
        Variable(), // max_B_feature_rank
        Variable(), // vbe_output_size
        {% endif %}
        Variable(), // is_experimental
        {{ args.split_variables | join(", ") }}
    };
    {% else %}
    const auto grad_dev_weights = split_embedding_nobag_backward_codegen_{{ optimizer }}_unweighted_exact_cuda(
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
        {% if optimizer != "none" %}
        stochastic_rounding,
        {% endif %}
        info_B_num_bits,
        info_B_mask,
        {% if vbe %}
        vbe_metadata,
        {% endif %}
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
        {% if optimizer != "none" %}
        Variable(), // gradient_clipping
        Variable(), // max_gradient
        Variable(), // stochastic_rounding
        {% endif %}
        {% if vbe %}
        Variable(), // B_offsets
        Variable(), // vbe_output_offsets_feature_rank
        Variable(), // vbe_B_offsets_rank_per_feature
        Variable(), // max_B
        Variable(), // max_B_feature_rank
        Variable(), // vbe_output_size
        {% endif %}
        Variable(), // is_experimental
        {{ args.split_variables | join(", ") }}
    };
    {% endif %}
  }
};
{% endif %} // if not nobag or not vbe
{% endfor %} // for nobag in [True, False]
{% endfor %} // for vbe in [True, False]
{% endif %} // if has_gpu_support

///@ingroup embedding-cuda
Tensor split_embedding_codegen_lookup_{{ optimizer }}_function(
    Tensor placeholder_autograd_tensor,
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    c10::optional<Tensor> indice_weights,
    c10::optional<Tensor> feature_requires_grad,
    Tensor lxu_cache_locations,
    {% if optimizer != "none" %}
    bool gradient_clipping,
    double max_gradient,
    bool stochastic_rounding,
    {% endif %}
    {{ args.split_function_args | join(", ") }},
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const c10::optional<Tensor>& B_offsets = c10::optional<Tensor>(),
    const c10::optional<Tensor>& vbe_output_offsets_feature_rank = c10::optional<Tensor>(),
    const c10::optional<Tensor>& vbe_B_offsets_rank_per_feature = c10::optional<Tensor>(),
    const int64_t max_B = -1,
    const int64_t max_B_feature_rank = -1,
    const int64_t vbe_output_size = -1,
    const bool is_experimental = false
) {
  {% if has_gpu_support %}
  {% for vbe in ([True, False] if has_vbe_support else [False]) %}
  {% set vbe_class_desc = "VBE" if vbe else "" %}

  {% if has_vbe_support %}
  {% if vbe %}
  if (B_offsets.has_value()) {
  {% else %}
  else { // if (B_offsets.has_value())
  {% endif %}
  {% endif %} // if has_vbe_support
    if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
      return SplitNoBagLookupFunction_{{ optimizer }}_Op::apply(
          placeholder_autograd_tensor,
          output_dtype,
          dev_weights,
          uvm_weights,
          lxu_cache_weights,
          weights_placements,
          weights_offsets,
          max_D,
          hash_size_cumsum,
          total_hash_size_bits,
          indices,
          offsets,
          lxu_cache_locations,
          {% if optimizer != "none" %}
          gradient_clipping,
          max_gradient,
          stochastic_rounding,
          {% endif %}
          is_experimental,
          {{ args.split_function_arg_names | join(", ") }})[0];
    } else {
      return Split{{ vbe_class_desc }}LookupFunction_{{ optimizer }}_Op::apply(
          placeholder_autograd_tensor,
          output_dtype,
          dev_weights,
          uvm_weights,
          lxu_cache_weights,
          weights_placements,
          weights_offsets,
          D_offsets,
          total_D,
          max_D,
          hash_size_cumsum,
          total_hash_size_bits,
          indices,
          offsets,
          pooling_mode,
          indice_weights,
          feature_requires_grad,
          lxu_cache_locations,
          {% if optimizer != "none" %}
          gradient_clipping,
          max_gradient,
          stochastic_rounding,
          {% endif %}
          {% if vbe %}
          B_offsets,
          vbe_output_offsets_feature_rank,
          vbe_B_offsets_rank_per_feature,
          max_B,
          max_B_feature_rank,
          vbe_output_size,
          {% endif %}
          is_experimental,
          {{ args.split_function_arg_names | join(", ") }})[0];
    }
  {% if has_vbe_support %}
  }
  {% endif %}
  {% endfor %}
  {% else %}
  TORCH_CHECK(false, "split_embedding_codegen_lookup_{{ optimizer }}_function is deprecated. Please see https://github.com/pytorch/FBGEMM/discussions/1727 for more detail.");
  return Tensor();
  {% endif %} // if has_gpu_support
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
    m.def("split_embedding_codegen_lookup_{{ optimizer }}_function("
          "Tensor placeholder_autograd_tensor, "
          "Tensor dev_weights, Tensor uvm_weights, "
          "Tensor lxu_cache_weights, "
          "Tensor weights_placements, "
          "Tensor weights_offsets, "
          "Tensor D_offsets, "
          "int total_D, "
          "int max_D, "
          "Tensor hash_size_cumsum, "
          "int total_hash_size_bits, "
          "Tensor indices, "
          "Tensor offsets, "
          "int pooling_mode, "
          "Tensor? indice_weights, "
          "Tensor? feature_requires_grad, "
          "Tensor lxu_cache_locations, "
          {% if optimizer != "none" %}
          "bool gradient_clipping, "
          "float max_gradient, "
          "bool stochastic_rounding, "
          {% endif %}
          "{{ args.split_function_schemas | join(", ") }}, "
          "int output_dtype=0, "
          "Tensor? B_offsets=None, "
          "Tensor? vbe_output_offsets_feature_rank=None, "
          "Tensor? vbe_B_offsets_rank_per_feature=None, "
          "int max_B=-1, "
          "int max_B_feature_rank=-1, "
          "int vbe_output_size=-1, "
          "bool is_experimental=False) -> Tensor");
    DISPATCH_TO_CUDA("split_embedding_codegen_lookup_{{ optimizer }}_function", split_embedding_codegen_lookup_{{ optimizer }}_function);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    m.def("split_embedding_codegen_lookup_{{ optimizer }}_function("
          "Tensor placeholder_autograd_tensor, "
          "Tensor dev_weights, Tensor uvm_weights, "
          "Tensor lxu_cache_weights, "
          "Tensor weights_placements, "
          "Tensor weights_offsets, "
          "Tensor D_offsets, "
          "int total_D, "
          "int max_D, "
          "Tensor hash_size_cumsum, "
          "int total_hash_size_bits, "
          "Tensor indices, "
          "Tensor offsets, "
          "int pooling_mode, "
          "Tensor? indice_weights, "
          "Tensor? feature_requires_grad, "
          "Tensor lxu_cache_locations, "
          {% if optimizer != "none" %}
          "bool gradient_clipping, "
          "float max_gradient, "
          "bool stochastic_rounding, "
          {% endif %}
          "{{ args.split_function_schemas | join(", ") }}, "
          "int output_dtype=0, "
          "Tensor? B_offsets=None, "
          "Tensor? vbe_output_offsets_feature_rank=None, "
          "Tensor? vbe_B_offsets_rank_per_feature=None, "
          "int max_B=-1, "
          "int max_B_feature_rank=-1, "
          "int vbe_output_size=-1, "
          "bool is_experimental=False) -> Tensor");
    DISPATCH_TO_CUDA("split_embedding_codegen_lookup_{{ optimizer }}_function", split_embedding_codegen_lookup_{{ optimizer }}_function);
}

  // clang-format on
