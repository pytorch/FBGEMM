/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
{%- set wdesc = "weighted" if weighted else "unweighted" %}
{%- set vbe_desc = "_vbe" if vbe else "" %}
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
{%- if optimizer != "none" and not dense %}
#include "gen_embedding_optimizer_{{ optimizer }}_split_device_kernel.cuh"
{%- endif %}

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

////////////////////////////////////////////////////////////////////////////////
// Kernel Template Definition
////////////////////////////////////////////////////////////////////////////////

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kBackwardMaxThreads) void
{%- if is_index_select %}
batch_index_select_dim0_codegen_backward_kernel_warp_per_row(
{%- else %}
split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vbe_desc }}_kernel_warp_per_row_1(
{%- endif %}
    const pta::PackedTensorAccessor64<grad_t, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> grad_output,
    {%- if optimizer != "none" %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    {%- endif %}
    {%- if weighted %}
    const pta::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {%- if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    pta::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- if optimizer == "none" %}
    const int32_t max_D,
    {%- endif %}
    {%- endif %} // if not dense and optimizer != "none"
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const bool permute_output_dim_0_1
    {%- else %}
    {{ args.split_kernel_args | replace_pta_namespace() | join(",\n    ") }}
    {%- endif %}
) {
    {%- if not nobag %}
    int32_t T = D_offsets.size(0) - 1;
    {%- else %}
    int32_t T = weights_offsets.size(0);
    {%- endif %}
    const int32_t start_run_id = blockIdx.x * blockDim.y + threadIdx.y;

#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
    const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
    const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
    constexpr int VEC_WIDTH = 4;

    for (uint32_t run_id = start_run_id;
         run_id < sorted_linear_indices_run.size(0) && run_id < sorted_linear_indices_num_runs[0];
             run_id += gridDim.x * blockDim.y) {

        const int64_t linear_index = sorted_linear_indices_run[run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[run_id];
        const int32_t segment_end =
            sorted_linear_indices_cumulative_run_lengths[run_id + 1];
        const int32_t SL = segment_end - segment_start;

        if (SL >= max_segment_length_per_warp) {
            continue;
        }

        // now, each segment corresponds to exactly one table `t` and row in
        // that table (`idx`). Thus, we can hoist out some of the book-keeping.
        {%- if not nobag %}
        const auto info_0 = reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start];
        const auto t_0 = info_0 >> info_B_num_bits;
        {%- else %}
        const auto info_0 = sorted_infos[segment_start];
        int32_t t_0 = info_0 % T;
        {%- endif %}

        int64_t hash_size = hash_size_cumsum[t_0];
        {%- if not nobag or is_index_select %}
        const auto D_start_t0 = D_offsets[t_0];
        // D can be hoisted here because D is the same if features share the
        // same table, but D_start is different
        const int32_t D = D_offsets[t_0 + 1] - D_start_t0;
        {%- if is_index_select %}
        // grad_offset can be hoisted here for batch_index_select because it
        // does not allow multiple features to share a single embedding table
        const auto grad_offset = permute_output_dim_0_1 ? D_start_t0 : grad_offsets[t_0];
        const auto grad_stride = permute_output_dim_0_1 ? D_offsets[T] : D;
        {%- endif %}
        {%- endif %}
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = 0;
        const int32_t sl_end = SL;
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            {%- if not nobag %}
            const auto b_t = sl_j < sl_end ? reinterpret_cast<const uint32_t*>(&sorted_infos[0])[segment_start + sl_j] : 0;
            const auto b = b_t & info_B_mask;
            const auto t = b_t >> info_B_num_bits;
            {%- if vbe %}
            const auto grad_offset = row_output_offsets[B_offsets[t] + b];
            {% else %}
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0;
            {%- endif %} // if vbe
            {%- else %} // if not nobag
            int64_t l_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
            int32_t l = l_t / T;
            {%- endif %} // if not nobag
            {%- if weighted %}
            at::acc_type<cache_t, true> idx_weight = sl_j < sl_end ? sorted_indice_weights[segment_start + sl_j] : 0.0;
            {%- endif %}

            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                {%- if nobag %}
                int32_t l_j = SHFL_SYNC(l, j);
                {%- elif vbe %}
                const auto grad_offset_j = SHFL_SYNC(grad_offset, j);
                {%- else %}
                int32_t b_j = SHFL_SYNC(b, j);
                int32_t D_start_j = SHFL_SYNC(D_start, j);
                {%- endif %}

                {%- if weighted %}
                at::acc_type<cache_t, true> idx_weight_j = SHFL_SYNC(idx_weight, j);
                {%- endif %}

                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                        ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        {%- if nobag and is_index_select %}
                        // grad_output is 1d
                        &grad_output[grad_offset + l_j * grad_stride + d]
                        {%- elif nobag %}
                        &grad_output[l_j][d]
                        {%- elif vbe %}
                        &grad_output[0][grad_offset_j + d]
                        {%- else %}
                        &grad_output[b_j][0] + D_start_j + d
                        {%- endif %}
                    );

                    {%- if weighted %}
                    grad_sum[i].fma_(grad_out_vec, idx_weight_j);
                    {%- else %}
                    grad_sum[i].add_(grad_out_vec);
                    {%- endif %}
                }
            }
        }
        {%- if not dense and optimizer != "none" %}
        split_{{ optimizer }}_table_update_kernel
          <emb_t, cache_t, kMaxVecsPerThread, kThreadGroupSize, VEC_WIDTH>(
              dev_weights,
              uvm_weights,
              lxu_cache_weights,
              weights_placements,
              weights_offsets,
              sorted_lxu_cache_locations,
              grad_sum,
              stochastic_rounding,
              stochastic_rounding_philox_args,
              run_id,
              D,
              t_0,
              idx,
              segment_start,
              shfl_sync_mask,
              threadIdx.y * kMaxVecsPerThread * kThreadGroupSize, // shared_weight_offset
              {{ args.split_function_arg_names | join(", ") }});
        {%- else %}
        // Write deduplicated gradient to grad_dev_weights gradient is sparse
        // for split_embedding and dense for dense_embedding
        {%- if dense %}
        const int64_t weights_offset = weights_offsets[t_0];
        {%- else %}
        // Compute offset of sparse gradient
        const int64_t weights_offset = run_id * max_D;
        idx = 0;
        {%- endif %}
    	#pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            auto& grad = grad_sum[i];
            grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
        }
        {%- endif %} // if not dense and optimizer != "none"

    }
}


////////////////////////////////////////////////////////////////////////////////
// Explicit Template Instantiations
////////////////////////////////////////////////////////////////////////////////

/*
    Explicitly instantiate the kernel function template.  The instantiations are
    based on the types enumerated by DISPATCH_EMB_GRAD_CACHE_TYPES macro used in
    embedding_backward_split_template.cu
*/

{%- macro template_instantiation(emb_type, grad_type, cache_type, kMaxVecsPerThread, kThreadGroupSize) %}
template __global__ __launch_bounds__(kBackwardMaxThreads) void
{%- if is_index_select %}
batch_index_select_dim0_codegen_backward_kernel_warp_per_row
{%- else %}
split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vbe_desc }}_kernel_warp_per_row_1
{%- endif %}
< {{ emb_type }},
  {{ grad_type }},
  {{ cache_type }},
  {{ kMaxVecsPerThread }},
  {{ kThreadGroupSize }}
> (
    const pta::PackedTensorAccessor64<{{ grad_type }}, {{ "1" if is_index_select else "2" }}, at::RestrictPtrTraits> grad_output,
    {%- if optimizer != "none" %}
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    pta::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag or is_index_select %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int64_t D,
    {%- endif %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    {%- if not nobag %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    {%- endif %}
    {%- if weighted %}
    const pta::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {%- if not dense and optimizer != "none" %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    pta::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- if optimizer == "none" %}
    const int32_t max_D,
    {%- endif %}
    {%- endif %} // if not dense and optimizer != "none"
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    {%- if is_index_select %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    const bool permute_output_dim_0_1
    {%- else %}
    {{ args.split_kernel_args_no_defaults | replace_pta_namespace() | join(",\n    ") | replace("cache_t", cache_type) }}
    {%- endif %}
);
{%- endmacro %}

{%- macro bulk_template_instantiations(kMaxVecsPerThread, kThreadGroupSize) %}
    {%- for grad_type in ['float', 'at::Half'] %}
    {%- for emb_type in ['uint8_t', 'float', 'at::Half'] %}
    {%- for cache_type in ['float', 'at::Half'] %}
        {{ template_instantiation(emb_type, grad_type, cache_type, kMaxVecsPerThread, kThreadGroupSize) }}
    {%- endfor %}
    {%- endfor %}
    {%- endfor %}
{%- endmacro %}


{%- if is_experimental_optimizer %}

{{ bulk_template_instantiations(max_embedding_dim // items_per_warp, 'kWarpSize') }}

{%- else %}

////////////////////////////////////////////////////////////////////////////////
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Compute the Cartesian product of (kMaxVecsPerThread, kThreadGroupSize)
    in the FBGEMM_USE_SUBWARP_SHUFFLE case

    constexpr int kMaxVecsPerThread = std::max({{ kMaxElemPerThread }} / 4, 1);
    constexpr int kThreadGroupSize = kWarpSize / std::max(4 / {{ kMaxElemPerThread }}, 1);

    This is needed to compute the unique tuples to use for explicit instantiation,
    so that we can avoid duplicate template instantiations.
*/ #}
{%- set tuples = [] %}
{%- for kMaxElemPerThread in range(1, max_embedding_dim // (items_per_warp // 4) + 1) %}
{%- if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
    {%- set t0 = [ (kMaxElemPerThread // 4), 1 ] | max %}
    {%- set t1 = [ 4 // kMaxElemPerThread, 1] | max %}
    {%- set temp = tuples.append((t0, "(kWarpSize / " ~ t1 ~ ")")) %}
{%- endif %}
{%- endfor %}

{#- /* Enumerate over the unique tuples */ #}
{%- for (kMaxVecsPerThread, kThreadGroupSize) in tuples | unique %}
    {{ bulk_template_instantiations(kMaxVecsPerThread, kThreadGroupSize) }}
{%- endfor %}

////////////////////////////////////////////////////////////////////////////////
#else
////////////////////////////////////////////////////////////////////////////////

{#- /*
    Compute the Cartesian product of (kMaxVecsPerThread, kThreadGroupSize)
    in the non-FBGEMM_USE_SUBWARP_SHUFFLE case

    constexpr int kMaxVecsPerThread = std::max({{ kMaxElemPerThread }} / 4, 1);
    constexpr int kThreadGroupSize = kWarpSize;
*/ #}
{%- set tuples = [] %}
{%- for kMaxElemPerThread in range(1, max_embedding_dim // (items_per_warp // 4) + 1) %}
{%- if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
    {%- set t0 = [ (kMaxElemPerThread // 4), 1 ] | max %}
    {%- set temp = tuples.append((t0, "kWarpSize")) %}
{%- endif %}
{%- endfor %}

{#- /* Enumerate over the unique tuples */ #}
{%- for (kMaxVecsPerThread, kThreadGroupSize) in tuples | unique %}
    {{ bulk_template_instantiations(kMaxVecsPerThread, kThreadGroupSize) }}
{%- endfor %}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////

{%- endif %}
        // clang-format on
