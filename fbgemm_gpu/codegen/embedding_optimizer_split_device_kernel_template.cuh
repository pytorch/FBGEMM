/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using namespace fbgemm_gpu;

template <
    typename emb_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize,
    int32_t VEC_WIDTH
>
DEVICE_INLINE void split_{{ optimizer }}_table_update_kernel(
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& dev_weights,
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits>& uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits>& lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& weights_placements,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>& weights_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>& sorted_lxu_cache_locations,
    Vec4T<at::acc_type<cache_t, true>>* grad_sum,
    const bool stochastic_rounding,
    const at::PhiloxCudaState& stochastic_rounding_philox_args,
    const uint32_t run_id,
    const uint32_t cache_loc_run_id,
    const int32_t D,
    const int32_t t,
    const int64_t idx,
    const uint32_t shfl_sync_mask,
    const int32_t shared_weight_offset,
    {{ args.split_ref_kernel_args | replace_pta_namespace() | join(",\n    ") }}
) {
    constexpr auto is_int8 = std::is_same<emb_t, uint8_t>::value;
    const int64_t weights_offset = weights_offsets[t];
    emb_t* __restrict__ weights {nullptr};
    cache_t* __restrict__ cache_weights {nullptr};
    int32_t D_emb = D;
    if (is_int8) {
        D_emb += kINT8QparamsBytes;
    }
    const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
    if (weights_placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset + idx * D_emb];
    } else {
        weights = &uvm_weights[weights_offset + idx * D_emb];
    }
    if (weights_placement == PlacementType::MANAGED_CACHING) {
        const int32_t cache_idx = sorted_lxu_cache_locations[cache_loc_run_id];
        if (cache_idx != kCacheLocationMissing) {
            cache_weights = &lxu_cache_weights[cache_idx][0];
        }
    }
    {%- for tensor in args.split_tensors %}
    at::acc_type<cache_t, true>* __restrict__ {{ tensor }};
    const auto {{ tensor }}_placement = static_cast<PlacementType>({{ tensor }}_placements[t]);
    const int64_t {{ tensor }}_offset = {{ tensor }}_offsets[t];
    if ({{ tensor }}_placement == PlacementType::DEVICE) {
        {{ tensor }} = &{{ tensor }}_dev[{{ tensor }}_offset];
    } else {
        {{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset];
    }
    {%- endfor %}

    struct SharedMemory<Vec4T<at::acc_type<cache_t, true>>> weight_update_buffer;
    Vec4T<at::acc_type<cache_t, true>>* shared_weight_update_row =
        is_int8 ? weight_update_buffer.getPointer() : nullptr;

    StochasticRoundingRNGState state;
    auto weight_row_template =
        WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(
            weights,
            cache_weights,
            D,
            stochastic_rounding ? &state : nullptr,
            &stochastic_rounding_philox_args,
            threadIdx.x + run_id * blockDim.x);

    float2 qparams_template;
    if (is_int8 && !cache_weights) {
        qparams_template = weight_row_template.load_qparams();
    }

    {{ split_precomputation }}

    float2 qparams_new;
#pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
        int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        Vec4T<at::acc_type<cache_t, true>> weight_new = weight_row_template.load(d, qparams_template);
        auto& grad = grad_sum[i];
        {{ split_weight_update }}
        if (is_int8 && !cache_weights) {
            shared_weight_update_row[
                threadIdx.x + (i * kThreadGroupSize) + shared_weight_offset] = weight_new;
        } else {
            // qparams_new not used if type is not int8
            weight_row_template.store(weight_new, d, qparams_new);
        }
    }

    if (is_int8 && !cache_weights) {
        // Calculate new qparams after row update
        qparams_new = thrust_find_qparams<at::acc_type<cache_t, true>>(
            &shared_weight_update_row[shared_weight_offset], D);
        weight_row_template.store_qparams(qparams_new);

        // Fetch cached updated row from shared mem and quantize on-the-fly
        // when saving to lowp embedding
#pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            const int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            weight_row_template.store(
                shared_weight_update_row[threadIdx.x + (i * kThreadGroupSize) + shared_weight_offset],
                d,
                qparams_new);
        }
    }

    {{ split_post_update }}
}

// clang-format on
