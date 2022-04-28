/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
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

using Tensor = at::Tensor;

Tensor split_embedding_codegen_forward_unweighted_cuda(
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
    int64_t BT_block_size);

Tensor split_embedding_codegen_forward_weighted_cuda(
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
    int64_t BT_block_size);

Tensor split_embedding_codegen_grad_indice_weights_cuda(
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
    Tensor feature_requires_grad);

void split_embedding_backward_codegen_{{ optimizer }}_unweighted_exact_cuda(
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
    bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }});

void split_embedding_backward_codegen_{{ optimizer }}_weighted_exact_cuda(
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
    bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }});

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
    int64_t unused);

void split_embedding_nobag_backward_codegen_{{ optimizer }}_unweighted_exact_cuda(
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
    bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }});

{% for nobag in [True, False] %}
class Split{{ "NoBag" if nobag else "" }}LookupFunction_{{ optimizer }}_Op :
    public torch::autograd::Function<Split{{ "NoBag" if nobag else "" }}LookupFunction_{{ optimizer }}_Op> {
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
    bool gradient_clipping,
    double max_gradient,
    bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }}) {
    ctx->save_for_backward({
        dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets, {% if not nobag %} D_offsets, {% endif %} hash_size_cumsum,
        indices, offsets, {% if not nobag %} indice_weights.value_or(Tensor()), feature_requires_grad.value_or(Tensor()), {% endif %} lxu_cache_locations, {{ args.split_saved_tensors | join(", ") }} });

    {% if not nobag %}
    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["pooling_mode"] = pooling_mode;
    {% else %}
    ctx->saved_data["D"] = D;
    {% endif %}
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
    ctx->saved_data["gradient_clipping"] = gradient_clipping;
    ctx->saved_data["max_gradient"] = max_gradient;
    ctx->saved_data["stochastic_rounding"] = stochastic_rounding;

    {% for (var, _) in args.saved_data %}
    ctx->saved_data["{{ var }}"] = {{ var }};
    {% endfor %}

    {% if not nobag %}
#ifdef __HIP_PLATFORM_HCC__
    constexpr int32_t BT_block_size = 64;
#else
    constexpr int32_t BT_block_size = 32;
#endif
    if (!indice_weights) {
        return {split_embedding_codegen_forward_unweighted_cuda(
        dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets,
        D_offsets, total_D, max_D, indices, offsets, pooling_mode, lxu_cache_locations, output_dtype, BT_block_size)};
    }  else {
        return {split_embedding_codegen_forward_weighted_cuda(
        dev_weights, uvm_weights, lxu_cache_weights, weights_placements, weights_offsets,
        D_offsets, total_D, max_D, indices, offsets, pooling_mode, *indice_weights, lxu_cache_locations, output_dtype, BT_block_size)};
    }
    {% else %}
    return {split_embedding_nobag_codegen_forward_unweighted_cuda(
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
      0)};
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
    auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
    auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
    auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();

    {% for (var, ivalue_cast) in args.saved_data %}
    auto {{ var }} = ctx->saved_data["{{ var }}"].{{ ivalue_cast }}();
    {% endfor %}

    TORCH_CHECK(grad_outputs.size() == 1);

#ifdef __HIP_PLATFORM_HCC__
    constexpr int32_t BT_block_size = 64;
    constexpr int32_t max_segment_length_per_warp = 64;
#else
    constexpr int32_t BT_block_size = 32;
    constexpr int32_t max_segment_length_per_warp = 32;
#endif
    using torch::autograd::Variable;

    auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];
    if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
        grad_output.stride(1) != 1 ||
        grad_output.stride(0) % 4 != 0) {
        grad_output = grad_output.contiguous();
    }

    {% if not nobag %}
    if (!indice_weights.defined()) {
      split_embedding_backward_codegen_{{ optimizer }}_unweighted_exact_cuda(
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
          lxu_cache_locations,
          BT_block_size,
          max_segment_length_per_warp,
          stochastic_rounding,
          {{ args.split_function_arg_names | join(", ") }});
      return {
          Tensor(), // placeholder autograd tensor
          Variable(), // output_dtype
          Tensor(), // dev_weights
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
          Variable(), // indice_weights
          Variable(), // feature_requires_grad
          Variable(), // lxu_cache_locations
          Variable(), // gradient_clipping
          Variable(), // max_gradient
          Variable(), // stochastic_rounding
          {{ args.split_variables | join(", ") }}
      };
    } else {
      auto grad_indice_weights = split_embedding_codegen_grad_indice_weights_cuda(
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
          feature_requires_grad);
      split_embedding_backward_codegen_{{ optimizer }}_weighted_exact_cuda(
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
          indice_weights,
          lxu_cache_locations,
          BT_block_size,
          max_segment_length_per_warp,
          stochastic_rounding,
          {{ args.split_function_arg_names | join(", ") }});
      return {
          Tensor(), // placeholder autograd tensor
          Variable(), // output_dtype
          Tensor(), // dev_weights
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
          grad_indice_weights,
          Variable(), // indice_weights
          Variable(), // feature_requires_grad
          Variable(), // lxu_cache_locations
          Variable(), // gradient_clipping
          Variable(), // max_gradient
          Variable(), // stochastic_rounding
          {{ args.split_variables | join(", ") }}
      };
    }
    {% else %}
    split_embedding_nobag_backward_codegen_{{ optimizer }}_unweighted_exact_cuda(
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
        stochastic_rounding,
        {{ args.split_function_arg_names | join(", ") }});
    return {
        Tensor(), // placeholder autograd tensor
        Variable(), // output_dtype
        Tensor(), // dev_weights
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
        Variable(), // gradient_clipping
        Variable(), // max_gradient
        Variable(), // stochastic_rounding
        {{ args.split_variables | join(", ") }}
    };
    {% endif %}
  }
};
{% endfor %}

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
    bool gradient_clipping,
    double max_gradient,
    bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }},
    int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)) {
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
      gradient_clipping,
      max_gradient,
      stochastic_rounding,
      {{ args.split_function_arg_names | join(", ") }})[0];
  } else {
    return SplitLookupFunction_{{ optimizer }}_Op::apply(
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
      gradient_clipping,
      max_gradient,
      stochastic_rounding,
      {{ args.split_function_arg_names | join(", ") }})[0];
  }
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
    m.def("split_embedding_codegen_lookup_{{ optimizer }}_function(Tensor placeholder_autograd_tensor, Tensor dev_weights, Tensor uvm_weights, Tensor lxu_cache_weights, Tensor weights_placements, Tensor weights_offsets, Tensor D_offsets, int total_D, int max_D, Tensor hash_size_cumsum, int total_hash_size_bits, Tensor indices, Tensor offsets, int pooling_mode, Tensor? indice_weights, Tensor? feature_requires_grad, Tensor lxu_cache_locations, bool gradient_clipping, float max_gradient, bool stochastic_rounding, {{ args.split_function_schemas | join(", ") }}, int output_dtype=0) -> Tensor");
    DISPATCH_TO_CUDA("split_embedding_codegen_lookup_{{ optimizer }}_function", split_embedding_codegen_lookup_{{ optimizer }}_function);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    m.def("split_embedding_codegen_lookup_{{ optimizer }}_function(Tensor placeholder_autograd_tensor, Tensor dev_weights, Tensor uvm_weights, Tensor lxu_cache_weights, Tensor weights_placements, Tensor weights_offsets, Tensor D_offsets, int total_D, int max_D, Tensor hash_size_cumsum, int total_hash_size_bits, Tensor indices, Tensor offsets, int pooling_mode, Tensor? indice_weights, Tensor? feature_requires_grad, Tensor lxu_cache_locations, bool gradient_clipping, float max_gradient, bool stochastic_rounding, {{ args.split_function_schemas | join(", ") }}, int output_dtype=0) -> Tensor");
    DISPATCH_TO_CUDA("split_embedding_codegen_lookup_{{ optimizer }}_function", split_embedding_codegen_lookup_{{ optimizer }}_function);
}

// clang-format on
