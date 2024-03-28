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

{%- set ddesc = "dense" if dense else "split" %}
{%- set wdesc = "weighted" if weighted else "unweighted" %}
{%- set ndesc = "_nobag" if nobag else "" %}
{%- set vdesc = "_vbe" if vbe else "" %}

#include "codegen/embedding_forward_template_helpers.cuh"
#include "fbgemm_gpu/split_embeddings_cache_cuda.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

{#-/*
    This code chunk describes the weights load + accumulate step in the
    forward kernel, containing 3 steps:

    1. Set up the WeightRow
    1. Load the quantization params
    1. Load and accumulate the slices of values from the row

    The main difference is in whether the slices are loaded from the embedding
    table or cache.

    NOTE: The decision was made to define this code chunk as a Jinja macro
    instead of inline C++ function, since the compiler might not be able to
    inline the code.

    In-code variables that are defined outside:
        emb_t, cache_t, cache_t
        idx_j
        D_emb
        lxu_cache_weights
        cache_idx_j
        idx_weight_j
        VEC_WIDTH
        D
        kThreadGroupSize
        output_j
*/#}
{%- macro load_and_accumulate(from_cache) %}
    {#-/* Set the weights row */#}
    const auto weights_row = WeightRowAccessor
        <
            emb_t,
            cache_t,
            cache_t,
            {%- if from_cache %}
            true
            {%- else %}
            false
            {%- endif %}
        >(
        {%- if from_cache %}
        // Pass nullptr to avoid calling &weights[idx_j * D_emb], which loads
        // memory into the registers as a side effect
        nullptr,
        // Load from the cache
        const_cast<cache_t*>(&lxu_cache_weights[cache_idx_j][0]),
        {%- else %}
        // Load from the embedding table
        const_cast<emb_t*>(&weights[idx_j * D_emb]),
        // Pass nullptr bc we are loading from the embedding table
        nullptr,
        {%- endif %}
        D,
        nullptr);

    {#-/* Set the quantization params */#}
    {%- if from_cache %}
    // Assume cache is FP16/FP32, which doesn't require quantization params
    const auto qparams = make_float2(0.0f, 0.0f);
    {%- else %}
    // Load the quantization params from the embedding table row if emb_t == uint8_t
    const auto qparams = weights_row.load_qparams();
    {%- endif %}

    {%- if not nobag %}
    // Iterate over the row in the weights table, in 4-element strides
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
        // Load the slice of the weights
        const int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        const auto weights_slice = weights_row.load(d, qparams);

        {%- if weighted %}
        // Accumulate the weights * positional weight
        accumulators[i].fma_(weights_slice, idx_weight_j);
        {%- else %}
        // Accumulate the weights
        accumulators[i].add_(weights_slice);
        {%- endif %}
    }

    {%- else %}
    for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
        const int32_t d = i + threadIdx.x * VEC_WIDTH;
        if (d < D) {
            // Since there is no pooling, simply copy the weights to output
            const auto weights_slice = weights_row.load(d, qparams);
            {%- if is_index_select %}
            // output is 1D (because the stride can be irregular)
            weights_slice.store(&output[output_offset + output_j * output_stride + d]);
            {%- else %}
            // output is 2D
            weights_slice.store(&output[output_j][d]);
            {%- endif %}
        }
    }
    {%- endif %}
{%- endmacro %}



{#-/*
    This code chunk contains the implementation body of the kernel, and is
    defined as a Jinja macro to be copy-pasted directly into the kernel as
    opposed to a template impl function called by the kernel, because during
    benchmarks, it was found that the extra function-calling resulted in a
    20-100 GB/s bandwidth reduction.
*/#}
{%- macro embedding_pool_or_store(lxu_miss_rate) %}
    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
        // Determine the L index that this thread will load data from in cooperative load
        int32_t l = l_start + threadIdx.x;

        {%- if dense or lxu_miss_rate != "cache_conflict_miss_rate::zero" %}
        // Cooperatively load the indices
        [[maybe_unused]] int64_t idx = l < L ? indices[indices_start + l] : 0;
        {%- endif %}

        {%- if not dense and lxu_miss_rate != "cache_conflict_miss_rate::all" %}
        // Cooperatively load the cache's indices
        [[maybe_unused]] int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {%- endif %}

        {%- if weighted %}
        // Cooperatively load the positional weight indices
        at::acc_type<cache_t, true> idx_weight = l < L ? indice_weights[indices_start + l] : 0;
        {%- endif %}

        // Iterate over kThreadGroupSize indices
        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
            {%- if dense or lxu_miss_rate != "cache_conflict_miss_rate::zero" %}
            // Load index from thread j in the group
            [[maybe_unused]] int64_t idx_j = SHFL_SYNC(idx, j);
            {%- endif %}

            {%- if is_index_select %}
            int64_t output_j = L_start + l_start + j;
            {%- elif nobag %}
            int64_t output_j = indices_start + l_start + j;
            {%- endif %}

            {%- if not dense and lxu_miss_rate != "cache_conflict_miss_rate::all" %}
            // Load cache's index from thread j in the group
            [[maybe_unused]] int32_t cache_idx_j = use_lxu_cache ? SHFL_SYNC(cache_idx, j) : 0;
            {%- endif %}

            {%- if weighted %}
            // Load positional weight index from thread j in the group
            at::acc_type<cache_t, true> idx_weight_j = SHFL_SYNC(idx_weight, j);
            {%- endif %}


            {#/**************************************************************/#}
            {#-/*
                This is the main switch that determines how we are to load and
                accumulate weights, and is determined by Jinja-time, compile-time,
                and run-time variables.
            */#}

            {%- if dense %}
                {#-/* If it's dense, cache is not supported, so load from the embedding table */#}
                {{- load_and_accumulate(false) }}

            {%- elif lxu_miss_rate == "cache_conflict_miss_rate::all" %}
                {#-/* Else if we know we have a 100% miss rate, then always fetch from the embedding table */#}
                {{- load_and_accumulate(false) }}

            {%- elif lxu_miss_rate == "cache_conflict_miss_rate::zero" %}
                {#-/* Else if we know we have a 0% miss rate, then always fetch from the cache */#}
                {{ load_and_accumulate(true) }}
            {%- else %}
                {#-/* Else we defer to run-time selection */#}
                if (placement == PlacementType::MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                    {#-/* If the row is available in the cache, fetch from the cache */#}
                    {{ load_and_accumulate(true) }}
                } else {
                    {#-/* Else fetch from the embedding table */#}
                    {{ load_and_accumulate(false) }}
                }

            {%- endif %}
            {#/**************************************************************/#}
        }
    }
{%- endmacro %}


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
__launch_bounds__(kForwardMaxThreads) __global__ void
{%- if is_index_select %}
batch_index_select_dim0_codegen_forward_kernel(
{%- else %}
{{ ddesc }}_embedding{{ ndesc }}_codegen_forward_{{ wdesc }}{{ vdesc }}_kernel(
{%- endif %}
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    const pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    const pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %} // if nobag
    {%- if vbe %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- else %}
    FixedDivisor fd_B,
    {%- endif %}
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    {%- if not is_index_select %}
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    {%- endif %}
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    /*
      NOTE: We pass in `lxu_cache_conflict_misses =
      uvm_cache_stats[uvm_cache_stats_index::num_conflict_unique_misses]` as a
      run-time argument here instead of passing the cache miss rate as a
      compile-time argument, because `lxu_cache_conflict_misses` is only
      available on the GPU, and invoking a templatized kernel with the cache
      miss rate as a template argument requires this information to first be
      passed back to the host, which is an expensive operation.
    */
    const int32_t* lxu_cache_conflict_misses,
    {%- endif %}
    {%- if is_index_select %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> total_L_offsets,
    const int32_t fixed_L_per_warp,
    const bool permute_output_dim_0_1,
    {%- endif %}
    // If 2D, shape is [B][total_D]
    pta::PackedTensorAccessor64<output_t, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> output
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
    {%- if not is_index_select %}
    if (b_t >= offsets.size(0) - 1) {
        return;
    }
    {%- endif %}

    // Determine the Table and Training Example IDs
    int32_t t;  // Table ID
    int32_t b;  // Training Example ID
    {%- if vbe %}
    const auto info = reinterpret_cast<const uint32_t*>(&b_t_map[b_t])[0];
    reinterpret_cast<uint32_t*>(&t)[0] = info >> info_B_num_bits;
    reinterpret_cast<uint32_t*>(&b)[0] = info & info_B_mask;
    {%- else %}
    fd_B.DivMod(b_t, &t, &b);
    {%- endif %}

    // Get total number of tables
    int32_t T = weights_offsets.size(0);

    {%- if is_index_select %}
    index_t indices_start;
    int32_t L;
    int32_t L_start;
    if (t >= T) {
        return;
    }
    const auto total_L_start = total_L_offsets[t];
    const auto total_L = total_L_offsets[t + 1] - total_L_start;
    L_start = b * fixed_L_per_warp;
    if (L_start >= total_L) {
        return;
    }
    indices_start = total_L_start + L_start;
    L = (total_L - L_start >= fixed_L_per_warp) ? fixed_L_per_warp : (total_L - L_start);
    {%- else %}
    // Determine the number of indices (pooling factor) to look up within the bag
    index_t indices_start = offsets[b_t];
    int32_t L = offsets[b_t + 1] - indices_start;
    {%- endif %}

    // Get the offsets of the embedding dimensions of the tables and determine D
    {%- if not nobag or is_index_select %}
    const auto D_start = D_offsets[t];
    const auto D_end = D_offsets[t + 1];
    const auto D = D_end - D_start;
    {%- endif %}

    {%- if is_index_select %}
    // Check D in the kernel to avoid iterating through the list on host
    CUDA_KERNEL_ASSERT(D % 4 == 0 && "The column size must be multiple of 4");
    const auto output_offset = permute_output_dim_0_1 ? D_start : output_offsets[t];
    const auto output_stride = permute_output_dim_0_1 ? D_offsets[T] : D;
    {%- endif %}

    // From the Table ID, fetch its weight tensor offset, locate that position
    // in the input weights tensor, and set the weights table pointer
    int64_t weights_offset = weights_offsets[t];
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

    {%- if dense %}
    {{ embedding_pool_or_store("NULL") }}

    {%- else %}
    if constexpr (! use_lxu_cache) {
        // If use_lxu_cache is false, then the cache conflict miss rate is
        // effectively 100%
        {{ embedding_pool_or_store("cache_conflict_miss_rate::all") }}

    } else {
        if (placement != PlacementType::MANAGED_CACHING) {
            // Load every row from HBM or UVM
            {{ embedding_pool_or_store("cache_conflict_miss_rate::all") }}
        }
        else if (lxu_cache_conflict_misses && *lxu_cache_conflict_misses == 0) {
            // If the UVM cache stats tensor is valid and tell us there are no
            // conflict unique misses, then the miss rate is effectively 0%
            {{ embedding_pool_or_store("cache_conflict_miss_rate::zero") }}

        } else {
            // Else, the cache conflict miss rate is mixed
            {{ embedding_pool_or_store("cache_conflict_miss_rate::mixed") }}
        }
    }
    {%- endif %}

    {%- if not nobag %}
    // If weight type is FP32/16
    if constexpr (!std::is_same_v<output_t, uint8_t>) {
        {%- if vbe %}
        output_t* output_ = &output[0][row_output_offsets[b_t]];
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


////////////////////////////////////////////////////////////////////////////////
// Explicit Template Instantiations
////////////////////////////////////////////////////////////////////////////////

/*
    Explicitly instantiate the kernel function template.  The instantiations are
    based on the types enumerated by DISPATCH_EMB_CACHE_TYPES macro used in
    embedding_forward_split_template.cu
*/

{%- macro template_instantiation(emb_type, cache_type, output_type, use_cache, kMaxVecsPerThread, kThreadGroupSize) %}
template __launch_bounds__(kForwardMaxThreads) __global__ void
{%- if is_index_select %}
batch_index_select_dim0_codegen_forward_kernel
{%- else %}
{{ ddesc }}_embedding{{ ndesc }}_codegen_forward_{{ wdesc }}{{ vdesc }}_kernel
{%- endif %}
<
    {{ emb_type }},
    {{ cache_type }},
    {{ output_type }},
    {%- if not dense %}
    {{ use_cache }},
    {%- endif %}
    int64_t,
    {%- if not nobag %}
    {{ kMaxVecsPerThread }},
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
    {%- if not nobag or is_index_select %}
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
    {%- if not is_index_select %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    {%- endif %}
    {%- if not nobag %}
    int64_t pooling_mode,
    {%- endif %}
    {%- if weighted %}
    pta::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> indice_weights,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> lxu_cache_locations,
    const int32_t* lxu_cache_conflict_misses,
    {%- endif %}
    {%- if is_index_select %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> output_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> total_L_offsets,
    const int32_t fixed_L_per_warp,
    const bool permute_output_dim_0_1,
    {%- endif %}
    pta::PackedTensorAccessor64<{{ output_type }}, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> output);
{%- endmacro %}

{%- macro bulk_template_instantiations(use_cache, kMaxVecsPerThread, kThreadGroupSize) %}
    {%- for emb_type in ['float', 'at::Half'] %}
    {%- for cache_type in ['float', 'at::Half'] %}
    {%- for output_type in ['float', 'at::Half', 'at::BFloat16'] %}
        {{ template_instantiation(emb_type, cache_type, output_type, use_cache, kMaxVecsPerThread, kThreadGroupSize) }}
    {%- endfor %}
    {%- endfor %}
    {%- endfor %}
{%- endmacro %}

{%- macro instantiate_templates(use_subwarp_shuffle) %}
{%- set has_experimental =
      has_experimental_support(dense, nobag, vbe, is_index_select, is_rocm)
%}
{%- set max_forward_embedding_dim =
      legacy_max_embedding_dim if has_experimental else max_embedding_dim
%}
{%- for use_cache in (["true", "false"] if not dense else ["NULL"]) %}
{%- for (kMaxVecsPerThread, kThreadGroupSize, use_blocking)
    in get_max_vecs_template_configs(
        items_per_warp,
        fixed_max_vecs_per_thread=max_forward_embedding_dim // items_per_warp,
        use_subwarp_shuffle=use_subwarp_shuffle,
        use_vec_blocking=False,
    )
%}
    {#-/* nobag does not have kMaxVecsPerThread as a template arg */#}
    {%- if not nobag or kMaxVecsPerThread <= 1 %}
        {{
           bulk_template_instantiations(
               use_cache,
               kMaxVecsPerThread,
               kThreadGroupSize
           )
        }}
    {%- endif %}
{%- endfor %}
{%- endfor %}
{%- endmacro %}

////////////////////////////////////////////////////////////////////////////////
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Explicitly instantiate kernels for the FBGEMM_USE_SUBWARP_SHUFFLE case

    Please see get_max_vecs_template_configs in
    codegen/embedding_common_code_generator.py for more details
*/ #}

{{ instantiate_templates(use_subwarp_shuffle=True) }}

////////////////////////////////////////////////////////////////////////////////
#else
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Explicitly instantiate kernels for the non-FBGEMM_USE_SUBWARP_SHUFFLE case

    Please see get_max_vecs_template_configs in
    codegen/embedding_common_code_generator.py for more details
*/ #}

{{ instantiate_templates(use_subwarp_shuffle=False) }}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////
