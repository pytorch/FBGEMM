/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

#define GROUP_REDUCE_ALL_SUM(val, ...) \
  warpReduceAllSum<__VA_ARGS__, kThreadGroupSize>(val, shfl_sync_mask)

{%- set mdesc = "ssd" if ssd else "split" %}
{%- set locs_or_addrs_tensor = "ssd_row_addrs" if ssd else "lxu_cache_locations" %}
{%- set locs_or_addrs_type = "int64_t" if ssd else "int32_t" %}
{%- set locs_or_addrs_idx = "row_idx" if ssd else "cache_idx" %}

using namespace fbgemm_gpu;

template <
    typename emb_t,
    typename cache_t,
    {%- for ph_name in args.placeholder_tensor_names %}
    {%- set ph_type = "{}_ph_t".format(ph_name) %}
    typename {{ ph_type }},
    {%- endfor %}
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH,
    bool kUseVecBlocking
>
DEVICE_INLINE void {{ mdesc }}_{{ optimizer }}_table_update_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>& lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& weights_offsets,
    const pta::PackedTensorAccessor32<{{ locs_or_addrs_type }}, 1, at::RestrictPtrTraits>& sorted_{{ locs_or_addrs_tensor }},
    Vec4TAcc<cache_t>* grad_sum,
    Vec4TAcc<cache_t>* smem_grad_sum,
    Vec4TAcc<cache_t>* shared_weight_update_row,
    const bool stochastic_rounding,
    const at::PhiloxCudaState& stochastic_rounding_philox_args,
    const uint32_t run_id,
    const uint32_t cache_loc_run_id,
    const int32_t D,
    const int32_t t,
    const int64_t idx,
    {%- if has_global_weight_decay_support %}
    const float global_weight_decay,
    {%- endif %}
    const uint32_t shfl_sync_mask,
    const int32_t max_vecs_per_thread,
    {{ args.split_ref_kernel_args | replace_pta_namespace() | join(",\n    ") }}
) {
    constexpr auto kIsInt8 = std::is_same_v<emb_t, uint8_t>;
    // Copy value to max_vecs to make max_vecs_per_thread known at compile time
    // when kUseVecBlocking == false
    const int32_t max_vecs =
        kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
    const int64_t weights_offset = weights_offsets[t];
    emb_t* __restrict__ weights {nullptr};
    cache_t* __restrict__ cache_weights {nullptr};
    int32_t D_emb = D;
    if (kIsInt8) {
        D_emb += kINT8QparamsBytes;
    }
    const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
    if (weights_placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset + idx * D_emb];
    } else {
        weights = {{ "nullptr" if ssd else "&uvm_weights[weights_offset + idx * D_emb]" }};
    }
    if (weights_placement == PlacementType::MANAGED_CACHING) {
        const auto {{ locs_or_addrs_idx }} = sorted_{{ locs_or_addrs_tensor }}[cache_loc_run_id];
        {%- if ssd %}
        cache_weights = reinterpret_cast<cache_t*>(
            *reinterpret_cast<const uint64_t*>(&{{ locs_or_addrs_idx }}));
        {%- else %}
        if ({{ locs_or_addrs_idx }} != kCacheLocationMissing) {
          cache_weights = &lxu_cache_weights[{{ locs_or_addrs_idx }}][0];
        }
        {%- endif %}
    }
    {%- for tensor in args.split_tensors %}
    {{ args.split_tensor_types[tensor] }}* __restrict__ {{ tensor }};
    const auto {{ tensor }}_placement = static_cast<PlacementType>({{ tensor }}_placements[t]);
    const int64_t {{ tensor }}_offset = {{ tensor }}_offsets[t];
    if ({{ tensor }}_placement == PlacementType::DEVICE) {
        {{ tensor }} = &{{ tensor }}_dev[{{ tensor }}_offset];
    } else {
        {{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset];
    }
    {%- endfor %}

    auto weight_row_template =
        WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
            weights,
            cache_weights,
            D,
            stochastic_rounding,
            &stochastic_rounding_philox_args,
            threadIdx.x + run_id * blockDim.x);

    float2 qparams_template;
    if constexpr (kIsInt8) {
        if (!cache_weights) {
            qparams_template = weight_row_template.load_qparams();
        }
    }

    {{ split_precomputation }}

    {# /* Note: technically, global weight decay (gwd) compensation should be done before
    `split_precomputation`). But since decouple mode in `rowwise_adagrad` only computes correction,
    the order of applying gwd does not matter. We perform gwd update before `split_weight_update`
    below to minimize number of times to load weights.
    So, note that the behavior may be different if you want to enable gwd for other optimizers
    such as `lamb` or `partial_rowwise_lamb`.
    */#}
    float2 qparams_new;
    {{
       generate_optimized_grad_sum_loop_access(
           """
           Vec4TAcc<cache_t> weight_new = weight_row_template.load(d, qparams_template);
           Vec4TAcc<cache_t>& grad = {grad_vec};
           {global_weight_decay_update}
           {split_weight_update}
           if (kIsInt8 && !cache_weights) {
               shared_weight_update_row[d_vec] = weight_new;
           } else {
               // qparams_new not used if type is not int8
               weight_row_template.store(weight_new, d, qparams_new);
           }
           """,
           other_formats={
               "split_weight_update": split_weight_update,
               "global_weight_decay_update": "weight_new.mul_(global_weight_decay);" if has_global_weight_decay_support else ""
            },
       )
    }}

    if constexpr (kIsInt8) {
        if (!cache_weights) {
            // Calculate new qparams after row update
            qparams_new = thrust_find_qparams<at::acc_type<cache_t, true>>(
                shared_weight_update_row, D);
            weight_row_template.store_qparams(qparams_new);

            // Fetch cached updated row from shared mem and quantize on-the-fly
            // when saving to lowp embedding
            for (int32_t vec = 0;
                (vec * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++vec) {
                const auto d_vec = vec * kThreadGroupSize + threadIdx.x;
                const int32_t d = d_vec * VEC_WIDTH;
                weight_row_template.store(
                    shared_weight_update_row[d_vec],
                    d,
                    qparams_new);
            }
        }
    }

    {{ split_post_update }}
}

// clang-format on
