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
{% endif %}

namespace {

{% if has_cpu_support %}
class SplitLookupFunction_{{ optimizer }}_Op : public torch::autograd::Function<SplitLookupFunction_{{ optimizer }}_Op> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    Tensor host_weights,
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
    bool gradient_clipping,
    double max_gradient,
    bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }},
    int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)) {
    Tensor indice_weights_value = indice_weights.value_or(Tensor());
    Tensor feature_requires_grad_value =
        feature_requires_grad.value_or(Tensor());
    ctx->save_for_backward({
        host_weights, weights_placements, weights_offsets, D_offsets, hash_size_cumsum,
        indices, offsets, indice_weights_value, feature_requires_grad_value, {{ args.split_saved_tensors | join(", ") }} });

    ctx->saved_data["total_D"] = total_D;
    ctx->saved_data["max_D"] = max_D;
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
    ctx->saved_data["pooling_mode"] = pooling_mode;
    ctx->saved_data["gradient_clipping"] = gradient_clipping;
    ctx->saved_data["max_gradient"] = max_gradient;
    ctx->saved_data["stochastic_rounding"] = stochastic_rounding;
    ctx->saved_data["output_dtype"] = output_dtype;

    {% for (var, _) in args.saved_data %}
    ctx->saved_data["{{ var }}"] = {{ var }};
    {% endfor %}

    return {split_embedding_codegen_forward_cpu(
        host_weights,
        weights_offsets,
        D_offsets,
        total_D,
        hash_size_cumsum,
        indices,
        offsets,
        pooling_mode,
        indice_weights_value,
        output_dtype)};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto host_weights = *savedItr++;
    auto weights_placements = *savedItr++;
    auto weights_offsets = *savedItr++;
    auto D_offsets = *savedItr++;
    auto hash_size_cumsum = *savedItr++;
    auto indices = *savedItr++;
    auto offsets = *savedItr++;
    auto indice_weights = *savedItr++;
    auto feature_requires_grad = *savedItr++;

    {% for tensor in args.split_saved_tensors %}
    auto {{ tensor }} = *savedItr++;
    {% endfor %}

    auto total_D = ctx->saved_data["total_D"].toInt();
    auto max_D = ctx->saved_data["max_D"].toInt();
    auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();
    auto pooling_mode = ctx->saved_data["pooling_mode"].toInt();
    auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
    auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
    auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();
    auto output_dtype = ctx->saved_data["output_dtype"].toInt();

    {% for (var, ivalue_cast) in args.saved_data %}
    auto {{ var }} = ctx->saved_data["{{ var }}"].{{ ivalue_cast }}();
    {% endfor %}

    TORCH_CHECK_EQ(grad_outputs.size(), 1);

    using torch::autograd::Variable;

    auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];
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
    // NOTE: MEAN pooling will not work with indice_weights!
    auto grad_indice_weights = indice_weights.defined()
        ? split_embedding_codegen_grad_indice_weights_cpu(
              grad_outputs[0],
              host_weights,
              weights_offsets,
              D_offsets,
              indices,
              offsets,
              feature_requires_grad)
        : Variable();
    return {
        Tensor(), // host_weights
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
        Variable(), // feature_requires_grad
        Variable(), // gradient_clipping
        Variable(), // max_gradient
        Variable(), // stochastic_rounding
        {{ args.split_variables | join(", ") }},
        Variable(), // output_dtype
    };
  }
};
{% endif %} // if has_cpu_support

///@ingroup embedding-cpu
Tensor split_embedding_codegen_lookup_{{ optimizer }}_function_cpu(
    Tensor host_weights,
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
    bool gradient_clipping,
    double max_gradient,
    bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }},
    int64_t output_dtype = static_cast<int64_t>(SparseType::FP32)) {
  {% if has_cpu_support %}
  return SplitLookupFunction_{{ optimizer }}_Op::apply(
      host_weights,
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
      gradient_clipping,
      max_gradient,
      stochastic_rounding,
      {{ args.split_function_arg_names | join(", ") }},
      output_dtype)[0];
  {% else %}
  TORCH_CHECK(false, "split_embedding_codegen_lookup_{{ optimizer }}_function_cpu is deprecated. Please see https://github.com/pytorch/FBGEMM/discussions/1727 for more detail.");
  return Tensor();
  {% endif %} // if has_cpu_support
}

// Deprecated for fb namespace! Please use fbgemm namespace instead!
TORCH_LIBRARY_FRAGMENT(fb, m) {
    m.def("split_embedding_codegen_lookup_{{ optimizer }}_function_cpu(Tensor host_weights, Tensor weights_placements, Tensor weights_offsets, Tensor D_offsets, int total_D, int max_D, Tensor hash_size_cumsum, int total_hash_size_bits, Tensor indices, Tensor offsets, int pooling_mode, Tensor? indice_weights, Tensor? feature_requires_grad, bool gradient_clipping, float max_gradient, bool stochastic_rounding, {{ args.split_function_schemas | join(", ") }}, int output_dtype=0) -> Tensor");
    DISPATCH_TO_CPU(
        "split_embedding_codegen_lookup_{{ optimizer }}_function_cpu",
        split_embedding_codegen_lookup_{{ optimizer }}_function_cpu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    m.def("split_embedding_codegen_lookup_{{ optimizer }}_function_cpu(Tensor host_weights, Tensor weights_placements, Tensor weights_offsets, Tensor D_offsets, int total_D, int max_D, Tensor hash_size_cumsum, int total_hash_size_bits, Tensor indices, Tensor offsets, int pooling_mode, Tensor? indice_weights, Tensor? feature_requires_grad, bool gradient_clipping, float max_gradient, bool stochastic_rounding, {{ args.split_function_schemas | join(", ") }}, int output_dtype=0) -> Tensor");
    DISPATCH_TO_CPU(
        "split_embedding_codegen_lookup_{{ optimizer }}_function_cpu",
        split_embedding_codegen_lookup_{{ optimizer }}_function_cpu);
}

} // namespace
// clang-format on
