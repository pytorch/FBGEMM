/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

{#-
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
// Note: clang-format off doesn't work with this templaterized code,
// so we need to keep lint-ignore-every.
// See https://fburl.com/dw9ljh4h
#}

{%- set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    {%- if not dense %}
    bool use_lxu_cache,
    {%- endif %}
    typename index_t,
    {%- if not nobag %}
    size_t kMaxVecsPerThread,
    {%- endif %}
    size_t kThreadGroupSize >
__launch_bounds__(kForwardMaxThreads) __global__
void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel(
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    FixedDivisor fd_B,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    at::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output // [B][total_D],
    ) {
    int32_t T = weights_offsets.size(0);
    {%- if not nobag %}
    const bool mean_pooling = static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN;
    int32_t B = output.size(0);
    {%- else %}
    int32_t B = (offsets.size(0) - 1) / T;
    {%- endif %}
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= B * T) {
        return;
    }
    int32_t t;
    int32_t b;
    fd_B.DivMod(b_t, &t, &b);
    int64_t weights_offset = weights_offsets[t];
    {%- if not nobag %}
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;
    {%- endif %}
    index_t indices_start = offsets[t * B + b];
    index_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    const emb_t* __restrict__ weights;
    {%- if not dense %}
    const auto placement = static_cast<PlacementType>(weights_placements[t]);
    if (placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset];
    } else {
        weights = &uvm_weights[weights_offset];
    }
    {%- else %}
    weights = &dev_weights[weights_offset];
    {%- endif %}

    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }

    constexpr int VEC_WIDTH = 4;
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif

    {%- if not nobag %}
    const float inv_L = (mean_pooling && L != 0) ? static_cast<float>(1.0) / L: static_cast<float>(1.0);
    Vec4T<cache_t> accumulators[kMaxVecsPerThread];
    {%- endif %}
    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {%- if not dense %}
        int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {%- endif %}
        {%- if weighted %}
        at::acc_type<cache_t, true> idx_weight = l < L ? indice_weights[indices_start + l] : 0;
        {%- endif %}
        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
            int64_t idx_j = SHFL_SYNC(idx, j);
            {%- if nobag %}
            int64_t output_j = indices_start + l_start + j;
            {%- endif %}
            {%- if not dense %}
            int32_t cache_idx_j = use_lxu_cache ? SHFL_SYNC(cache_idx, j) : 0;
            {%- endif %}

            {%- if weighted %}
            at::acc_type<cache_t, true> idx_weight_j = SHFL_SYNC(idx_weight, j);
            {%- endif %}

            {%- if not dense %}
            // use_lxu_cache is a compile time condition
            if (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
                    const_cast<emb_t*>(&weights[idx_j * D_emb]),
                    const_cast<cache_t*>(&lxu_cache_weights[cache_idx_j][0]),
                    D,
                    nullptr);
                // assume cache is fp16/fp32 which doesn't require qparams
                float2 qparams_cache = make_float2(0.0f, 0.0f);

                {%- if not nobag %}
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                    {%- if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {%- else %}
                    accumulators[i].add_(weight);
                    {%- endif %}
                }
                {%- else %}
                for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                    int32_t d = i + threadIdx.x * VEC_WIDTH;
                    if (d < D) {
                        Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                        weight.store(&output[output_j][d]);
                    }
                }
                {%- endif %}
            }
            else { // else row is not in cache
            {%- endif %}
                auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
                    const_cast<emb_t*>(&weights[idx_j * D_emb]),
                    nullptr,
                    D,
                    nullptr);
                float2 qparams_emb;
                if (std::is_same<emb_t, uint8_t>::value) {
                    qparams_emb = weight_row_emb.load_qparams();
                }
                {%- if not nobag %}
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    {%- if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {%- else %}
                    accumulators[i].add_(weight);
                    {%- endif %}
                }
                {%- else %}
                for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                    int32_t d = i + threadIdx.x * VEC_WIDTH;
                    if (d < D) {
                        Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                        weight.store(&output[output_j][d]);
                    }
                }
                {%- endif %}
            {%- if not dense %}
            } // else row is not in cache
            {%- endif %}
        }
    }

    {%- if not nobag %}
    if (!std::is_same<output_t, uint8_t>::value) {
        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
            accumulators[i].mul_(inv_L);
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            accumulators[i].store(&output[b][D_start + d]);
        }
    } else {
        // apply per feature row-wise int8
        float thread_local_min = std::numeric_limits<float>::max();
        float thread_local_max = std::numeric_limits<float>::lowest();
        float2 qparams;

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            accumulators[i].mul_(inv_L);
            thread_local_max = max(thread_local_max, vec4_max(accumulators[i]));
            thread_local_min = min(thread_local_max, vec4_min(accumulators[i]));
        }

        qparams = warp_find_qparams(thread_local_min, thread_local_max);
        int output_D_start = D_start + t * 8;
        int output_D_end = output_D_start + D;

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            nearest_rounding_vector<output_t, cache_t>(&output[b][output_D_start + d], accumulators[i], qparams);
        }
        if (threadIdx.x == 0) {
            store_qparams_to_row(&output[b][output_D_end], qparams);
        }

    }
    {%- endif %}
}

/*
    Explicitly instantiate the kernel function template.  The instantiations are
    based on the types enumerated by DISPATCH_EMB_GRAD_CACHE_TYPES macro used in
    embedding_forward_split_template.cu
*/

{%- for output_type in ['uint8_t', 'at::Half', 'float'] %}
{%- for emb_type in ['uint8_t', 'float', 'at::Half'] %}
{%- for cache_type in ['float', 'at::Half'] %}

////////////////////////////////////////////////////////////////////////////////
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Compute the Cartesian product of (use_cache, kMaxVecsPerThread, kThreadGroupSize)
    in the FBGEMM_USE_SUBWARP_SHUFFLE case

    constexpr int kMaxVecsPerThread = std::max({{ kMaxElemPerThread }} / 4, 1);
    constexpr int kThreadGroupSize = kWarpSize / std::max(4 / {{ kMaxElemPerThread }}, 1);

    This is needed to compute the unique tuples to use for explicit instantiation,
    so that we can avoid duplicate template instantiations.
*/ #}
{%- set tuples = [] %}
{%- for use_cache in ['true', 'false'] %}
{%- for kMaxElemPerThread in range(1, max_embedding_dim // (items_per_warp // 4) + 1) %}
{%- if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
    {%- set t0 = use_cache if not dense else "NULL" %}
    {%- set t1 = [ (kMaxElemPerThread // 4), 1 ] | max if not nobag else "NULL" %}
    {%- set t2 = [ 4 // kMaxElemPerThread, 1] | max %}
    {%- set temp = tuples.append((t0, t1, "(kWarpSize / " ~ t2 ~ ")")) %}
{%- endif %}
{%- endfor %}
{%- endfor %}

{#- /*
    Enumerate over the unique tuples (NULL means the field is not materialized
    for the template context, e.g. where nobag = true):

    (true,·1,·(kWarpSize·/·4))
    (true,·1,·(kWarpSize·/·2))
    (true,·1,·(kWarpSize·/·1))
    (true,·2,·(kWarpSize·/·1))
    (true,·3,·(kWarpSize·/·1))
    (true,·4,·(kWarpSize·/·1))
    (true,·5,·(kWarpSize·/·1))
    (true,·6,·(kWarpSize·/·1))
    (true,·7,·(kWarpSize·/·1))
    (true,·8,·(kWarpSize·/·1))
    (false,·1,·(kWarpSize·/·4))
    (false,·1,·(kWarpSize·/·2))
    (false,·1,·(kWarpSize·/·1))
    (false,·2,·(kWarpSize·/·1))
    (false,·3,·(kWarpSize·/·1))
    (false,·4,·(kWarpSize·/·1))
    (false,·5,·(kWarpSize·/·1))
    (false,·6,·(kWarpSize·/·1))
    (false,·7,·(kWarpSize·/·1))
    (false,·8,·(kWarpSize·/·1))

    (NULL,·1,·(kWarpSize·/·4))
    (NULL,·1,·(kWarpSize·/·2))
    (NULL,·1,·(kWarpSize·/·1))
    (NULL,·2,·(kWarpSize·/·1))
    (NULL,·3,·(kWarpSize·/·1))
    (NULL,·4,·(kWarpSize·/·1))
    (NULL,·5,·(kWarpSize·/·1))
    (NULL,·6,·(kWarpSize·/·1))
    (NULL,·7,·(kWarpSize·/·1))
    (NULL,·8,·(kWarpSize·/·1))

    (true,·NULL,·(kWarpSize·/·4))
    (true,·NULL,·(kWarpSize·/·2))
    (true,·NULL,·(kWarpSize·/·1))
    (false,·NULL,·(kWarpSize·/·4))
    (false,·NULL,·(kWarpSize·/·2))
    (false,·NULL,·(kWarpSize·/·1))

    (NULL,·NULL,·(kWarpSize·/·4))
    (NULL,·NULL,·(kWarpSize·/·2))
    (NULL,·NULL,·(kWarpSize·/·1))
*/ #}
{%- for (use_cache, kMaxVecsPerThread, kThreadGroupSize) in tuples | unique %}

template __launch_bounds__(kForwardMaxThreads) __global__
void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel
<
    {{ emb_type }},
    {{ cache_type }},
    {{ output_type }},
    {%- if not dense %}
    {{ use_cache }},
    {%- endif %}
    int64_t,
    {%- if not nobag %}
    {{- kMaxVecsPerThread }},
    {%- endif %}
    {{ kThreadGroupSize }}
> (
    const at::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const at::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    FixedDivisor fd_B,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    at::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    at::PackedTensorAccessor64<{{ output_type }}, 2, at::RestrictPtrTraits> output);

{%- endfor %}

////////////////////////////////////////////////////////////////////////////////
#else
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Compute the Cartesian product of (use_cache, kMaxVecsPerThread, kThreadGroupSize)
    in the non-FBGEMM_USE_SUBWARP_SHUFFLE case

    constexpr int kMaxVecsPerThread = std::max({{ kMaxElemPerThread }} / 4, 1);
    constexpr int kThreadGroupSize = kWarpSize;
*/ #}
{%- set tuples = [] %}
{%- for use_cache in ['true', 'false'] %}
{%- for kMaxElemPerThread in range(1, max_embedding_dim // (items_per_warp // 4) + 1) %}
{%- if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
    {%- set t0 = use_cache if not dense else "NULL" %}
    {%- set t1 = [ (kMaxElemPerThread // 4), 1 ] | max if not nobag else "NULL" %}
    {%- set temp = tuples.append((t0, t1, "kWarpSize")) %}
{%- endif %}
{%- endfor %}
{%- endfor %}

{#- /*
    Enumerate over the unique tuples (NULL means the field is not materialized
    for the template context, e.g. where nobag = true):

    (true,·1,·kWarpSize)
    (true,·2,·kWarpSize)
    (true,·3,·kWarpSize)
    (true,·4,·kWarpSize)
    (true,·5,·kWarpSize)
    (true,·6,·kWarpSize)
    (true,·7,·kWarpSize)
    (true,·8,·kWarpSize)
    (false,·1,·kWarpSize)
    (false,·2,·kWarpSize)
    (false,·3,·kWarpSize)
    (false,·4,·kWarpSize)
    (false,·5,·kWarpSize)
    (false,·6,·kWarpSize)
    (false,·7,·kWarpSize)
    (false,·8,·kWarpSize)

    (NULL,·1,·kWarpSize)
    (NULL,·2,·kWarpSize)
    (NULL,·3,·kWarpSize)
    (NULL,·4,·kWarpSize)
    (NULL,·5,·kWarpSize)
    (NULL,·6,·kWarpSize)
    (NULL,·7,·kWarpSize)
    (NULL,·8,·kWarpSize)

    (true,·NULL,·kWarpSize)
    (false,·NULL,·kWarpSize)

    (NULL,·NULL,·kWarpSize)
*/ #}
{%- for (use_cache, kMaxVecsPerThread, kThreadGroupSize) in tuples | unique %}

template __launch_bounds__(kForwardMaxThreads) __global__
void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}_kernel
<
    {{ emb_type }},
    {{ cache_type }},
    {{ output_type }},
    {%- if not dense %}
    {{ use_cache }},
    {%- endif %}
    int64_t,
    {%- if not nobag %}
    {{- kMaxVecsPerThread }},
    {%- endif %}
    {{ kThreadGroupSize }}
> (
    const at::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const at::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    const at::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    FixedDivisor fd_B,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    at::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    at::PackedTensorAccessor64<{{ output_type }}, 2, at::RestrictPtrTraits> output);

{%- endfor %}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////

{%- endfor %}
{%- endfor %}
{%- endfor %}
