/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// clang-format off
{%- set wdesc = "weighted" if weighted else "unweighted" %}
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize >
__global__ __launch_bounds__(kBackwardMaxThreads) void
split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_warp_per_row_1(
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output,
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int32_t B,
    int64_t D,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_run,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_cumulative_run_lengths,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_lxu_cache_locations,
    {%- endif %}
    {%- if weighted %}
    const at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {%- if not dense %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    at::PackedTensorAccessor64<cache_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %}
    {%- if not nobag %}
    FixedDivisor fd,
    {%- endif %}
    {{ args.split_kernel_args | join(",\n    ") }}) {

    {%- if not nobag %}
    int32_t T = D_offsets.size(0) - 1;
    const int32_t B = grad_output.size(0);
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
    const auto info_0 = sorted_infos[segment_start];

    {%- if not nobag %}
    int32_t t_0 = fd.Div(info_0); // info_0 / B;
    {%- else %}
    int32_t t_0 = info_0 % T;
    {%- endif %}

    int64_t hash_size = hash_size_cumsum[t_0];
    {%- if not nobag %}
    int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
    {%- endif %}
    int64_t idx = linear_index - hash_size;

    const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
    const int32_t sl_start = 0;
    const int32_t sl_end = SL;
    Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
    for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
        int32_t sl_j = sl + threadIdx.x;
        {%- if not nobag %}
        int32_t b_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
        int32_t b; //= b_t % B;
        int32_t t; //= b_t / B;
        fd.DivMod(b_t, &t, &b);
        int32_t D_start = D_offsets[t];
        {%- else %}
        int64_t l_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
        int32_t l = l_t / T;
        {%- endif %}
        {%- if weighted %}
        at::acc_type<cache_t, true> idx_weight = sl_j < sl_end ? sorted_indice_weights[segment_start + sl_j] : 0.0;
        {%- endif %}

        for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
            {%- if not nobag %}
            int32_t b_j = SHFL_SYNC(b, j);
            int32_t D_start_j = SHFL_SYNC(D_start, j);
            {%- else %}
            int32_t l_j = SHFL_SYNC(l, j);
            {%- endif %}
            {%- if weighted %}
            at::acc_type<cache_t, true> idx_weight_j = SHFL_SYNC(idx_weight, j);
            {%- endif %}

            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                {%- if not nobag %}
                Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                    &grad_output[b_j][0] + D_start_j + d);
                {%- else %}
                Vec4T<at::acc_type<grad_t, true>> grad_out_vec(&grad_output[l_j][d]);
                {%- endif %}
                {%- if weighted %}
                grad_sum[i].fma_(grad_out_vec, idx_weight_j);
                {%- else %}
                grad_sum[i].add_(grad_out_vec);
                {%- endif %}
            }
        }
    }
    int64_t weights_offset = weights_offsets[t_0];
    {%- if not dense %}
    emb_t* __restrict__ weights{nullptr};
    cache_t* __restrict__ cache_weights{nullptr};
    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }
    const auto weights_placement = static_cast<PlacementType>(weights_placements[t_0]);
    if (weights_placement == PlacementType::DEVICE) {
        weights = &dev_weights[weights_offset + idx * D_emb];
    } else {
        weights = &uvm_weights[weights_offset + idx * D_emb];
    }
    if (weights_placement == PlacementType::MANAGED_CACHING) {
        int32_t cache_idx = sorted_lxu_cache_locations[segment_start];
        if (cache_idx != kCacheLocationMissing) {
            cache_weights = &lxu_cache_weights[cache_idx][0];
        }
    }
    {%- for tensor in args.split_tensors %}
    at::acc_type<cache_t, true>* __restrict__ {{ tensor }};
    const auto {{ tensor }}_placement = static_cast<PlacementType>({{ tensor }}_placements[t_0]);
    int64_t {{ tensor }}_offset = {{ tensor }}_offsets[t_0];
    if ({{ tensor }}_placement == PlacementType::DEVICE) {
        {{ tensor }} = &{{ tensor }}_dev[{{ tensor }}_offset];
    } else {
        {{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset];
    }
    {%- endfor %}

    struct SharedMemory<Vec4T<at::acc_type<cache_t, true>>> weight_update_buffer;
    Vec4T<at::acc_type<cache_t, true>>* shared_weight_update_row = weight_update_buffer.getPointer();
    auto weight_row_template = WeightRow<emb_t, cache_t, at::acc_type<cache_t, true>>(weights, cache_weights, D, nullptr);
    if (!std::is_same<emb_t, float>::value && stochastic_rounding) {
        StochasticRoundingRNGState state;
        // different for every *run* and every *thread*.
        auto stochastic_rounding_seeds =
            at::cuda::philox::unpack(stochastic_rounding_philox_args);
        stochastic_rounding_init(
            std::get<0>(stochastic_rounding_seeds) ^
                std::get<1>(stochastic_rounding_seeds),
            threadIdx.x + run_id * blockDim.x,
            &state);
        weight_row_template.set_stoc_state(&state);
    }
    float2 qparams_template;
    if (std::is_same<emb_t, uint8_t>::value && !cache_weights){
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
        if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
            shared_weight_update_row[threadIdx.x + (i + threadIdx.y * kMaxVecsPerThread) * kThreadGroupSize] = weight_new;
        } else {
            weight_row_template.store(weight_new, d, qparams_new); // qparams_new not used if type is not int8
        }
    }

    if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
        // calculate new qparams after row update
        qparams_new = thrust_find_qparams<at::acc_type<cache_t, true>>(&shared_weight_update_row[threadIdx.y * kMaxVecsPerThread * kThreadGroupSize], D);
        weight_row_template.store_qparams(qparams_new);

        // fetch cached updated row from shared mem and quantize on-the-fly when saving to lowp embedding
        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            weight_row_template.store(shared_weight_update_row[threadIdx.x + (i + threadIdx.y * kMaxVecsPerThread) * kThreadGroupSize], d, qparams_new);
        }
    }
    {%- else %}
#pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
        int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        auto& grad = grad_sum[i];
        grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
    }
    {%- endif %}

    }
}

/*
    Explicitly instantiate the kernel function template.  The instantiations are
    based on the types enumerated by DISPATCH_EMB_GRAD_CACHE_TYPES macro used in
    embedding_backward_split_template.cu
*/

{%- for grad_type in ['float', 'at::Half'] %}
{%- for emb_type in ['uint8_t', 'float', 'at::Half'] %}
{%- for cache_type in ['float', 'at::Half'] %}

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

template __global__ __launch_bounds__(kBackwardMaxThreads)
void split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_warp_per_row_1
< {{ emb_type }},
  {{ grad_type }},
  {{ cache_type }},
  {{ kMaxVecsPerThread }},
  {{ kThreadGroupSize }}
> (
    const at::PackedTensorAccessor64<{{ grad_type }}, 2, at::RestrictPtrTraits> grad_output,
    at::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    at::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int32_t B,
    int64_t D,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    {%- endif %}
    {%- if weighted %}
    const at::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {%- if not dense %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    at::PackedTensorAccessor64<{{ cache_type }}, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %}
    {%- if not nobag %}
    FixedDivisor fd,
    {%- endif %}
    {{ args.split_kernel_args_no_defaults | join(",\n    ") | replace("cache_t", cache_type) }});

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
    {%- set t1 = [ 4 // kMaxElemPerThread, 1] | max %}
    {%- set temp = tuples.append((t0, "kWarpSize")) %}
{%- endif %}
{%- endfor %}

{#- /* Enumerate over the unique tuples */ #}
{%- for (kMaxVecsPerThread, kThreadGroupSize) in tuples | unique %}

template __global__ __launch_bounds__(kBackwardMaxThreads)
void split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_warp_per_row_1
< {{ emb_type }},
  {{ grad_type }},
  {{ cache_type }},
  {{ kMaxVecsPerThread }},
  {{ kThreadGroupSize }}
> (
    const at::PackedTensorAccessor64<{{ grad_type }}, 2, at::RestrictPtrTraits> grad_output,
    at::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    at::PackedTensorAccessor64<{{ emb_type }}, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<{{ cache_type }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> weights_placements,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {%- else %}
    int32_t B,
    int64_t D,
    {%- endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> hash_size_cumsum,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_linear_indices_run,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    {%- endif %}
    {%- if weighted %}
    const at::PackedTensorAccessor32<at::acc_type<{{ cache_type }}, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {%- if not dense %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    at::PackedTensorAccessor64<{{ cache_type }}, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %}
    {%- if not nobag %}
    FixedDivisor fd,
    {%- endif %}
    {{ args.split_kernel_args_no_defaults | join(",\n    ") | replace("cache_t", cache_type) }});

{%- endfor %}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////

{%- endfor %}
{%- endfor %}
{%- endfor %}

        // clang-format on
