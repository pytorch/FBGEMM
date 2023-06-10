/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
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
{%- set vbe_desc = "_vbe" if vbe else "" %}
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
void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}{{ vbe_desc }}_kernel(
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    {%- if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- else %}
    FixedDivisor fd_B,
    {%- endif %}
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    pta::PackedTensorAccessor64<output_t, 2, at::RestrictPtrTraits> output // [B][total_D]
    ) {

// shfl_sync_mask is implicitly used by SHFL_SYNC
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif

    // Elements are processed 4 at a time through fbgemm_gpu::Vec4 (CUDA float4, 16 bytes)
    constexpr int VEC_WIDTH = 4;

    // Determine the linearized warp ID, and exit early if needed
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    if (b_t >= offsets.size(0) - 1) {
        return;
    }

    // Determine the Table and Training Example IDs
    int32_t t;  // Table ID
    int32_t b;  // Training Example ID
    {%- if vbe %}
    const auto info = reinterpret_cast<const uint32_t*>(&b_t_map[b_t])[0];
    reinterpret_cast<uint32_t*>(&t)[0] = info >> info_B_num_bits;
    reinterpret_cast<uint32_t*>(&b)[0] = info | info_B_mask;
    {%- else %}
    fd_B.DivMod(b_t, &t, &b);
    {%- endif %}

    // From the Table ID, fetch its weight tensor offset, locate that position
    // in the input weights tensor, and set the weights table pointer
    const emb_t* __restrict__ weights;
    int64_t weights_offset = weights_offsets[t];
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

    // Get total number of tables
    int32_t T = weights_offsets.size(0);

    // Determine the number of indices (pooling factor) to look up within the bag
    index_t indices_start = offsets[b_t];
    index_t indices_end = offsets[b_t + 1];
    int32_t L = indices_end - indices_start;

    // Get the offsets of the embedding dimensions of the tables and determine D
    {%- if not nobag %}
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;
    {%- endif %}

    // D is computed in the bag case or provided as function arg in the nobag case
    // (nobag only supports the case where the embedding dimensions are the same for all tables)
    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }

    {%- if not nobag %}
    // Determine if we're doing mean pooling
    const bool mean_pooling = static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN;

    // Compute 1/L - this is used to compute the mean later on
    const float inv_L = (mean_pooling && L != 0) ? static_cast<float>(1.0) / L: static_cast<float>(1.0);

    // Set up the accumulator buffer
    Vec4T<cache_t> accumulators[kMaxVecsPerThread];
    {%- endif %}

    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
        // Determine the L index that this thread will load data from in cooperative load
        int32_t l = l_start + threadIdx.x;
        // Cooperatively load the indices
        int64_t idx = l < L ? indices[indices_start + l] : 0;

        {%- if not dense %}
        // Cooperatively load the cache's indices
        int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {%- endif %}

        {%- if weighted %}
        // Cooperatively load the positional weight indices
        at::acc_type<cache_t, true> idx_weight = l < L ? indice_weights[indices_start + l] : 0;
        {%- endif %}

        // Iterate over kThreadGroupSize indices
        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
            // Load index from thread j in the group
            int64_t idx_j = SHFL_SYNC(idx, j);

            {%- if nobag %}
            int64_t output_j = indices_start + l_start + j;
            {%- endif %}

            {%- if not dense %}
            // Load cache's index from thread j in the group
            int32_t cache_idx_j = use_lxu_cache ? SHFL_SYNC(cache_idx, j) : 0;
            {%- endif %}

            {%- if weighted %}
            // Load positional weight index from thread j in the group
            at::acc_type<cache_t, true> idx_weight_j = SHFL_SYNC(idx_weight, j);
            {%- endif %}

            {%- if not dense %}
            // use_lxu_cache is a compile time condition
            if (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                // Load the embedding table row from cache to the buffer
                auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
                    const_cast<emb_t*>(&weights[idx_j * D_emb]),
                    const_cast<cache_t*>(&lxu_cache_weights[cache_idx_j][0]),
                    D,
                    nullptr);

                // Assume cache is fp16/fp32 which doesn't require qparams
                float2 qparams_cache = make_float2(0.0f, 0.0f);

                {%- if not nobag %}
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    // Load Vec4 from cache
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);

                    {%- if weighted %}
                    // Accumulate the weight * positional weight
                    accumulators[i].fma_(weight, idx_weight_j);
                    {%- else %}
                    // Accumulate the weight
                    accumulators[i].add_(weight);
                    {%- endif %}
                }
                {%- else %}
                for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                    int32_t d = i + threadIdx.x * VEC_WIDTH;
                    if (d < D) {
                        // Since there is no pooling, simply copy the weight to output
                        Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                        weight.store(&output[output_j][d]);
                    }
                }
                {%- endif %}

            } else { // else row is not in cache
            {%- endif %}
                // Load the embedding table row from memory to the buffer
                auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
                    const_cast<emb_t*>(&weights[idx_j * D_emb]),
                    nullptr,
                    D,
                    nullptr);

                // Load the two quantization params (scale and bias) from the end of the embedding table row (2 floats)
                [[maybe_unused]] float2 qparams_emb;
                if (std::is_same<emb_t, uint8_t>::value) {
                    qparams_emb = weight_row_emb.load_qparams();
                }

                {%- if not nobag %}
                // Iterate over the row of elements in the weights table, in 4-element strides between adjacent threads
                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    // Figure out the position in the embedding table row to load
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;

                    // Fused load-and-dequantize from the buffer
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);

                    {%- if weighted %}
                    // Accumulate the weight * positional weight
                    accumulators[i].fma_(weight, idx_weight_j);
                    {%- else %}
                    // Accumulate the weight
                    accumulators[i].add_(weight);
                    {%- endif %}
                }
                {%- else %}
                for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                    int32_t d = i + threadIdx.x * VEC_WIDTH;
                    if (d < D) {
                        // Since there is no pooling, simply copy the weight to output
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
    // If weight type is FP32/16
    if constexpr (!std::is_same_v<output_t, uint8_t>) {
        {%- if vbe %}
        output_t* output_ = &output[0][output_offsets[b_t]];
        {%- else %}
        output_t* output_ = &output[b][D_start];
        {%- endif %}

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
             i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
             ++i) {
            // Compute the mean (for mean pooling) and store directly to memory as is
            accumulators[i].mul_(inv_L);
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            accumulators[i].store(output_ + d);
        }

    } else {
        // Else weight type is INT8
        float thread_local_min = std::numeric_limits<float>::max();
        float thread_local_max = std::numeric_limits<float>::lowest();
        float2 qparams;

        // Accumulate the min and max values
        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            // Simultaneously multiply by 1/L to compute the mean
            accumulators[i].mul_(inv_L);
            thread_local_max = max(thread_local_max, vec4_max(accumulators[i]));
            thread_local_min = min(thread_local_max, vec4_min(accumulators[i]));
        }

        // Construct the quantization parameters from the min and max values
        qparams = warp_find_qparams(thread_local_min, thread_local_max);
        int output_D_start = D_start + t * 8;
        int output_D_end = output_D_start + D;

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            // Fused quantize-and-store to memory
            nearest_rounding_vector<output_t, cache_t>(&output[b][output_D_start + d], accumulators[i], qparams);
        }

        // Write out the qparams to the front of the embedding table row
        if (threadIdx.x == 0) {
            store_qparams_to_row(&output[b][output_D_end], qparams);
        }

    }
    {%- endif %}
}


/*
    Explicitly instantiate the kernel function template.  The instantiations are
    based on the types enumerated by DISPATCH_EMB_CACHE_TYPES macro used in
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
void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}{{ vbe_desc }}_kernel
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
    const pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    {%- if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- else %}
    FixedDivisor fd_B,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    pta::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    pta::PackedTensorAccessor64<{{ output_type }}, 2, at::RestrictPtrTraits> output);

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
void {{ "dense" if dense else "split" }}_embedding{{ "_nobag" if nobag else "" }}_codegen_forward_{{ wdesc }}{{ vbe_desc }}_kernel
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
    const pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    {%- if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- else %}
    FixedDivisor fd_B,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> indices,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    pta::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    {%- endif %}
    pta::PackedTensorAccessor64<{{ output_type }}, 2, at::RestrictPtrTraits> output);

{%- endfor %}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////

{%- endfor %}
{%- endfor %}
{%- endfor %}
