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
__global__ __launch_bounds__(kMaxThreads) void
split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_cta_per_row_1(
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
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
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_ids,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_long_run_ids,
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
    {%- if not dense %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    at::PackedTensorAccessor64<cache_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %}
    {%- if not nobag %}
    FixedDivisor fd,
    {%- endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    {{ args.split_kernel_args | join(",\n    ") }}) {
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
  const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
  const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
  constexpr int VEC_WIDTH = 4;
  int32_t T = weights_offsets.size(0);
  {%- if not nobag %}
  const int32_t B = grad_output.size(0);
  {%- endif %}
  const int32_t num_long_runs = num_long_run_ids[0];
  for (int32_t long_run_id = blockIdx.x; long_run_id < num_long_runs; long_run_id += gridDim.x) {
        // The first thread block in the really long run has run_id in long_run_ids
        // and the rest have the negative of its offset (see find_long_segments kernel).
        int32_t cta_rank_on_current_run = 0;
        int32_t current_run_id = long_run_ids[long_run_id];
        if (current_run_id < 0) {
            cta_rank_on_current_run = -long_run_ids[long_run_id];
            current_run_id = long_run_ids[long_run_id - cta_rank_on_current_run];
        }
        const int32_t run_length =
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1] -
            sorted_linear_indices_cumulative_run_lengths[current_run_id];
        // This computation must agree with how we compute num_ctas_for_run in
        // find_long_segments kernel!
        const int32_t num_ctas_on_current_run =
            use_deterministic_algorithms ? 1 : div_round_up(run_length, max_segment_length_per_cta);


        const int64_t linear_index = sorted_linear_indices_run[current_run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[current_run_id] +
            cta_rank_on_current_run * max_segment_length_per_cta;
        const int32_t segment_end = std::min(
            use_deterministic_algorithms ? INT_MAX : segment_start + max_segment_length_per_cta,
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1]);
        const int32_t SL = segment_end - segment_start;
        const int32_t warp_id = threadIdx.y;
        const int32_t lane_id = threadIdx.x;

        // Note that with shared embedding tables we can have multiple tables
        // (i.e. different values of `t` sharing the same segment).
        //
        const auto info_0 = sorted_infos[segment_start];

        {%- if not nobag %}
        int32_t t_0 = fd.Div(info_0); //info_0 / B;
        {%- else %}
        int32_t t_0 = info_0 % T;
        {%- endif %}

        int64_t hash_size = hash_size_cumsum[t_0];
        {%- if not nobag %}
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        {%- endif %}
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = SL_per_warp * warp_id;
        const int32_t sl_end = min(SL_per_warp * (warp_id + 1), SL);
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            {%- if not nobag %}
            int32_t b_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
            int32_t b; //= b_t % B;
            int32_t t; //= b_t / B;
            fd.DivMod(b_t, &t, &b);
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0;
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
        // do shared memory reduction only if we used multiple warps.
        if (SL > SL_per_warp) {
            struct SharedMemory<Vec4T<at::acc_type<cache_t, true>>> smem;
            Vec4T<at::acc_type<cache_t, true>>* shared_grad_sums = smem.getPointer();

    #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
            shared_grad_sums
                [lane_id + i * kThreadGroupSize +
                warp_id * kMaxVecsPerThread * kThreadGroupSize] = grad_sum[i];
            }
            __syncthreads();
            if (blockDim.y >= 32) {
            if (warp_id < 16) {
    #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 16) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 16) {
            if (warp_id < 8) {
    #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 8) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 8) {
            if (warp_id < 4) {
    #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 4) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 4) {
            if (warp_id < 2) {
    #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kThreadGroupSize +
                    warp_id * kMaxVecsPerThread * kThreadGroupSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                warp_id * kMaxVecsPerThread * kThreadGroupSize],
                            shared_grad_sums
                                [lane_id + i * kThreadGroupSize +
                                (warp_id + 2) * kMaxVecsPerThread * kThreadGroupSize]);
                }
            }
            __syncthreads();
            }
            if (warp_id == 0) {
    #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                grad_sum[i] = vec4_acc(
                    shared_grad_sums
                        [lane_id + i * kThreadGroupSize +
                        warp_id * kMaxVecsPerThread * kThreadGroupSize],
                    shared_grad_sums
                        [lane_id + i * kThreadGroupSize +
                        (warp_id + 1) * kMaxVecsPerThread * kThreadGroupSize]);
            }
            }
        }

        if (warp_id != 0) {
            continue;
        }

        if (num_ctas_on_current_run > 1) {
            int really_long_run_id = long_run_id_to_really_long_run_ids[long_run_id];
            Vec4T<at::acc_type<cache_t, true>> *temp_grad_accum_ptr =
                reinterpret_cast<Vec4T<at::acc_type<cache_t, true>>*>(&temp_grad_accum[really_long_run_id][0]);
#pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.x, grad_sum[i].acc.x);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.y, grad_sum[i].acc.y);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.z, grad_sum[i].acc.z);
                gpuAtomicAdd(&temp_grad_accum_ptr[lane_id + i * kThreadGroupSize].acc.w, grad_sum[i].acc.w);
            }
            int counter;
            if (threadIdx.x == 0) {
                __threadfence();
                counter = gpuAtomicAdd(&grad_accum_counter[really_long_run_id], -1);
            }
            counter = SHFL_SYNC(counter, 0);
            // Only the thread block accumulated the gradient last does the weight update.
            if (counter > 1) {
                continue;
            }
            CUDA_KERNEL_ASSERT(counter == 1 && "Invalid grad_accum_counter. Race condition?");
#pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                ++i) {
                grad_sum[i] = temp_grad_accum_ptr[lane_id + i * kThreadGroupSize];
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
                threadIdx.x + current_run_id * blockDim.x,
                &state);
            weight_row_template.set_stoc_state(&state);
        }

        float2 qparams_template;
        if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
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
                shared_weight_update_row[lane_id + i * kThreadGroupSize] = weight_new;
            } else {
                weight_row_template.store(weight_new, d, qparams_new); // qparams_new not used if embedding is not int8
            }
        }
        if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
            // calculate qparams from updated weight row
            qparams_new = thrust_find_qparams<at::acc_type<cache_t, true>>(shared_weight_update_row, D);
            weight_row_template.store_qparams(qparams_new);

#pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                weight_row_template.store(shared_weight_update_row[lane_id + i * kThreadGroupSize], d, qparams_new);
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
    } // for each run
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

template __global__ __launch_bounds__(kMaxThreads)
void split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_cta_per_row_1
< {{ emb_type }},
  {{ grad_type }},
  {{ cache_type }},
  {{ kMaxVecsPerThread }},
  {{ kThreadGroupSize }}
> (
    const at::PackedTensorAccessor64<{{ grad_type  }}, 2, at::RestrictPtrTraits> grad_output,
    at::PackedTensorAccessor64<{{ emb_type  }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    at::PackedTensorAccessor64<{{ emb_type  }}, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<{{ cache_type  }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
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
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_ids,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_long_run_ids,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    {%- endif %}
    {%- if weighted %}
    const at::PackedTensorAccessor32<at::acc_type<{{ cache_type  }}, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    {%- if not dense %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    at::PackedTensorAccessor64<{{ cache_type  }}, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %}
    {%- if not nobag %}
    FixedDivisor fd,
    {%- endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    at::PackedTensorAccessor32<at::acc_type<{{ cache_type  }}, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
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

template __global__ __launch_bounds__(kMaxThreads)
void split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_cta_per_row_1
< {{ emb_type }},
  {{ grad_type }},
  {{ cache_type }},
  {{ kMaxVecsPerThread }},
  {{ kThreadGroupSize }}
> (
    const at::PackedTensorAccessor64<{{ grad_type  }}, 2, at::RestrictPtrTraits> grad_output,
    at::PackedTensorAccessor64<{{ emb_type  }}, 1, at::RestrictPtrTraits> dev_weights,
    {%- if not dense %}
    at::PackedTensorAccessor64<{{ emb_type  }}, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<{{ cache_type  }}, 2, at::RestrictPtrTraits> lxu_cache_weights,
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
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_ids,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> num_long_run_ids,
    {%- if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- else %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {%- endif %}
    {%- if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_lxu_cache_locations,
    {%- endif %}
    {%- if weighted %}
    const at::PackedTensorAccessor32<at::acc_type<{{ cache_type  }}, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {%- endif %}
    {%- if not dense %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {%- else %}
    at::PackedTensorAccessor64<{{ cache_type  }}, 1, at::RestrictPtrTraits> grad_dev_weights,
    {%- endif %}
    {%- if not nobag %}
    FixedDivisor fd,
    {%- endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    at::PackedTensorAccessor32<at::acc_type<{{ cache_type  }}, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    {{ args.split_kernel_args_no_defaults | join(",\n    ") | replace("cache_t", cache_type) }});

{%- endfor %}

////////////////////////////////////////////////////////////////////////////////
#endif
////////////////////////////////////////////////////////////////////////////////

{%- endfor %}
{%- endfor %}
{%- endfor %}

        // clang-format on
