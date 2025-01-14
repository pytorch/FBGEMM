/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor.h"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

template <
    typename emb_t,
    typename cache_t,
    {%- for ph_name in args.placeholder_tensor_names %}
    typename {{ ph_name + "_ph_t" }},
    {%- endfor %}
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH
>
__global__ __launch_bounds__(kMaxThreads) void
split_{{ optimizer }}_update_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    // grad_dev_indices is equivalent to sorted_linear_indices_run
    const pta::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> grad_dev_indices,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_lxu_cache_locations,
    const int32_t max_D,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {{ args.split_kernel_args | replace_pta_namespace() | join(",\n  ") }});

////////////////////////////////////////////////////////////////////////////////
// Auto generated placeholder tensor dispatch macros
////////////////////////////////////////////////////////////////////////////////

{%- if args.placeholder_tensor_names | length == 0 %}

#define DISPATCH_PLACEHOLDER_TYPES(NAME, ...) \
  return __VA_ARGS__();

{%- else %}

// TODO: create a macro for this
#define DISPATCH_PLACEHOLDER_TYPES({{ args.placeholder_tensor_names | to_upper_placeholder_types() | join(", ") }}, NAME, ...) \
  {%- for ph_name in args.placeholder_tensor_names %}
  at::ScalarType _{{ph_name}}_t =                                       \
    {{ ph_name.upper() + "_T" }};                                       \
  {%- endfor %}
  {%- for ph_combo in args.placeholder_type_combos %}
  if (                                                                  \
    {%- for ph_name, ph_ty in ph_combo.items() %}
    {{ ph_name.upper() + "_T" }} == {{ ph_ty.scalar_type }} &&          \
    {%- endfor %}
    true                                                                \
  ) {                                                                   \
    {%- for ph_name, ph_ty in ph_combo.items() %}
    using {{ ph_name + "_ph_t" }} = {{ ph_ty.primitive_type }};         \
    {%- endfor  %}
    return __VA_ARGS__();                                               \
  }                                                                     \
  {%- endfor %}
  else {                                                                \
    AT_ERROR(                                                           \
        #NAME, \
        " not implemented for ",                                        \
        {%- for ph_name in args.placeholder_tensor_names %}
        "{{ ph_name + "_ph_t" }} = '", toString(_{{ph_name}}_t) , "' ", \
        {%- endfor %}
        ""                                                              \
    );                                                                  \
  }

{%- endif %}

////////////////////////////////////////////////////////////////////////////////
// Operator implementation
////////////////////////////////////////////////////////////////////////////////

// split_{{ optimizer }}_update assumes that all tables have the same embedding
// dimension.
void split_embedding_{{ optimizer }}_update(
    Tensor& dev_weights,
    Tensor& uvm_weights,
    Tensor& lxu_cache_weights,
    const Tensor& grad_dev_weights,
    const Tensor& grad_dev_indices,
    const Tensor& weights_placements,
    const Tensor& weights_offsets,
    const int64_t max_D,
    const bool stochastic_rounding,
    {{ args.split_function_args | join(", ") }}
) {
    TENSOR_ON_CUDA_GPU(dev_weights);
    TENSOR_ON_CUDA_GPU(uvm_weights);
    TENSOR_ON_CUDA_GPU(lxu_cache_weights);
    TENSOR_ON_CUDA_GPU(grad_dev_weights);
    TENSOR_ON_CUDA_GPU(grad_dev_indices);
    TENSOR_ON_CUDA_GPU(weights_placements);
    TENSOR_ON_CUDA_GPU(weights_offsets);
    {%- for tensor in args.split_tensors %}
    TENSOR_ON_CUDA_GPU({{ tensor }}_dev);
    TENSOR_ON_CUDA_GPU({{ tensor }}_uvm);
    TENSOR_ON_CUDA_GPU({{ tensor }}_placements);
    TENSOR_ON_CUDA_GPU({{ tensor }}_offsets);
    {%- endfor %}

    TENSORS_ON_SAME_DEVICE(dev_weights, uvm_weights);
    TENSORS_ON_SAME_DEVICE(dev_weights, lxu_cache_weights);
    TENSORS_ON_SAME_DEVICE(dev_weights, grad_dev_weights);
    TENSORS_ON_SAME_DEVICE(dev_weights, grad_dev_indices);
    TENSORS_ON_SAME_DEVICE(dev_weights, weights_placements);
    TENSORS_ON_SAME_DEVICE(dev_weights, weights_offsets);
    {%- for tensor in args.split_tensors %}
    TENSORS_ON_SAME_DEVICE(dev_weights, {{ tensor }}_dev);
    TENSORS_ON_SAME_DEVICE(dev_weights, {{ tensor }}_uvm);
    TENSORS_ON_SAME_DEVICE(dev_weights, {{ tensor }}_placements);
    TENSORS_ON_SAME_DEVICE(dev_weights, {{ tensor }}_offsets);
    {%- endfor %}

    TORCH_CHECK_LE(max_D, {{ legacy_max_embedding_dim }});

    if (grad_dev_indices.numel() == 0) {
        return;
    }
    {%- if "learning_rate" in args.split_kernel_arg_constructors %}
    // convert `learning rate` to float since `learning rate` is float in kernels
    const float learning_rate = learning_rate_tensor.item<float>();
    {%- endif %}

    CUDA_DEVICE_GUARD(dev_weights);

    // Flatten dev_weights because it is currrently 2D
    dev_weights = dev_weights.flatten();
    const auto& flatten_grad_dev_weights = grad_dev_weights.flatten();
    const auto& flatten_grad_dev_indices = grad_dev_indices.flatten();

    DISPATCH_EMB_CACHE_TYPES(
        dev_weights.scalar_type(),
        lxu_cache_weights.scalar_type(),
        "split_embedding_{{ optimizer }}_update_kernel",
        [&] {
            TORCH_CHECK(!(std::is_same<emb_t, uint8_t>::value));

            at::PhiloxCudaState rng_engine_inputs;
            if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
                auto gen = at::cuda::detail::getDefaultCUDAGenerator();
                std::lock_guard<std::mutex> lock(gen.mutex());
                rng_engine_inputs =
                    at::check_generator<at::CUDAGeneratorImpl>(gen)
                        ->philox_cuda_state(4);
            }
            {%- for kMaxElemPerThread in range(1, legacy_max_embedding_dim // (items_per_warp // 4) + 1) %}
            {%- if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
            if (max_D <= {{ items_per_warp // 4 * kMaxElemPerThread }}) {
                // hipcc can't use max in constexpr
                constexpr int kMaxVecsPerThread = {{ kMaxElemPerThread }} / 4 >= 1 ? {{ kMaxElemPerThread }} / 4 : 1;
                // If max_D is small, use fewer number of threads than kWarpSize.
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
                constexpr int kThreadGroupSize = kWarpSize / std::max(4 / {{ kMaxElemPerThread }}, 1);
#else
                constexpr int kThreadGroupSize = kWarpSize;
#endif
#ifdef FBGEMM_GPU_MEMCHECK
                const auto func_name = "split_{{ optimizer }}_update_kernel";
#endif

                DISPATCH_PLACEHOLDER_TYPES(
                  {%- for ph_name in args.placeholder_tensor_names %}
                  {{ ph_name + "_dev" }}.scalar_type(),
                  {%- endfor %}
                  "split_embedding_{{ optimizer }}_update_placeholder_type_kernel",
                  [&] {
                    split_{{ optimizer }}_update_kernel<
                        emb_t,
                        cache_t,
                        {%- for ph_name in args.placeholder_tensor_names %}
                        {{ ph_name + "_ph_t" }},
                        {%- endfor %}
                        kMaxVecsPerThread,
                        kThreadGroupSize,
                        4>
                        <<<div_round_up(grad_dev_indices.numel(), kMaxThreads / kThreadGroupSize),
                           dim3(kThreadGroupSize, kMaxThreads / kThreadGroupSize, 1),
                           0, // Shared memory is not needed because uint8_t is not supported
                           at::cuda::getCurrentCUDAStream()
                        >>>
                        (
                            MAKE_PTA_WITH_NAME(func_name, dev_weights, emb_t, 1, 64),
                            MAKE_PTA_WITH_NAME(func_name, uvm_weights, emb_t, 1, 64),
                            MAKE_PTA_WITH_NAME(func_name, lxu_cache_weights, cache_t, 2, 64),
                            MAKE_PTA_WITH_NAME(func_name, flatten_grad_dev_weights, emb_t, 1, 64),
                            MAKE_PTA_WITH_NAME(func_name, flatten_grad_dev_indices, int64_t, 1, 64),
                            MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                            MAKE_PTA_WITH_NAME(func_name, weights_offsets, int64_t, 1, 32),
                            // Use weights_placements instead of
                            // sorted_lxu_cache_locations because LXU cache is not
                            // supported right now
                            MAKE_PTA_WITH_NAME(func_name, weights_placements, int32_t, 1, 32),
                            max_D,
                            stochastic_rounding,
                            rng_engine_inputs,
                            {{ args.split_kernel_arg_constructors | make_pta_acc_format("func_name") | join(", ") }}
                        );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }); // DISPATCH_PLACEHOLDER_TYPES
                return;
            }
            {%- endif %}
            {%- endfor %}
        } // DISPATCH_EMB_CACHE_TYPES
    );
}
