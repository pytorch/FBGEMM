/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// clang-format off
{% set wdesc = "weighted" if weighted else "unweighted" %}
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

#define SHFL_SYNC(val, srcLane) shfl_sync(val, srcLane, kThreadGroupSize, shfl_sync_mask)

{% if not dense %}
constexpr int32_t kCacheLocationMissing = -1;
{% endif %}

constexpr size_t kBackwardMaxThreads = 512;

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

namespace {

// Based on the empirical study, max grid size that is 64x larger than the
// number of SMs gives good performance across the board
constexpr int MAX_THREAD_BLOCKS_FACTOR = 64;

int get_max_thread_blocks_() {
  return MAX_THREAD_BLOCKS_FACTOR * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}

} // namespace

__global__ __launch_bounds__(kMaxThreads) void
split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_find_long_segments(
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_run_lengths,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_ids,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_long_run_ids,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        long_run_id_to_really_long_run_ids,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        num_really_long_run_ids,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        grad_accum_counter,
    const int32_t max_segment_length_per_warp,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms) {
  const int32_t num_runs = sorted_linear_indices_num_runs[0];
  for (auto run_id = blockIdx.x * blockDim.x + threadIdx.x; run_id < num_runs; run_id += blockDim.x * gridDim.x) {
    if (sorted_linear_indices_run_lengths[run_id] >= max_segment_length_per_warp) {
        // A segment with length > max_segment_length_per_cta is handled by more than 1 thread block.
        const int num_ctas_for_run =
            use_deterministic_algorithms ? 1 : div_round_up(sorted_linear_indices_run_lengths[run_id], max_segment_length_per_cta);
        const auto long_run_idx = gpuAtomicAdd(&num_long_run_ids[0], num_ctas_for_run);
        // The first thread block in the really long run gets run_id in long_run_ids
        // and the rest get the negative of its offset.
        long_run_ids[long_run_idx] = run_id;
        for (int i = 1; i < num_ctas_for_run; ++i) {
            long_run_ids[long_run_idx + i] = -i;
        }
        if (num_ctas_for_run > 1) {
            const auto really_long_run_idx = gpuAtomicAdd(&num_really_long_run_ids[0], 1);
            grad_accum_counter[really_long_run_idx] = num_ctas_for_run;
            for (int i = 0; i < num_ctas_for_run; ++i) {
                long_run_id_to_really_long_run_ids[long_run_idx + i] = really_long_run_idx;
            }
        }
    }
  }
}

template <typename grad_t>
__global__ __launch_bounds__(kMaxThreads) void grad_mean_kernel(
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output_mean) {
  int32_t B = grad_output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  if (b_t >= B * T) {
    return;
  }
  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  int64_t indices_start = offsets[t * B + b];
  int64_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  if (L != 0) {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<grad_t> grad_out_vec(&grad_output[b][D_start + d * 4]);
      grad_out_vec.mul_(1.0 / L);
      grad_out_vec.store(&grad_output_mean[b][D_start + d * 4]);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<grad_t> grad_out_vec(&grad_output[b][D_start + d * 4]);
      grad_out_vec.store(&grad_output_mean[b][D_start + d * 4]);
    }
  }
}

{% for nobag in [True, False] %}
{% if not nobag or not weighted %}
template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize>
__global__ __launch_bounds__(kMaxThreads) void
split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_cta_per_row_1(
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits> grad_output,
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {% if not dense %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {% else %}
    int32_t B,
    int64_t D,
    {% endif %}
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
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {% else %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {% endif %}
    {% if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_lxu_cache_locations,
    {% endif %}
    {% if weighted %}
    const at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {% endif %}
    {% if not dense %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {% else %}
    at::PackedTensorAccessor64<cache_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {% endif %}
    {% if not nobag %}
    FixedDivisor fd,
    {% endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> long_run_id_to_really_long_run_ids,
    at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits> temp_grad_accum,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> grad_accum_counter,
    const int32_t max_segment_length_per_cta,
    const bool use_deterministic_algorithms,
    {{ args.split_kernel_args | join(", ") }}) {
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
  const unsigned int shfl_sync_mask =
        ((1L << kThreadGroupSize) - 1) <<
        (threadIdx.y % (kWarpSize / kThreadGroupSize) * kThreadGroupSize);
#else
  const unsigned int shfl_sync_mask = 0xffffffffu;
#endif
  constexpr int VEC_WIDTH = 4;
  int32_t T = weights_offsets.size(0);
  {% if not nobag %}
  const int32_t B = grad_output.size(0);
  {% endif %}
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

        {% if not nobag %}
        int32_t t_0 = fd.Div(info_0); //info_0 / B;
        {% else %}
        int32_t t_0 = info_0 % T;
        {% endif %}

        int64_t hash_size = hash_size_cumsum[t_0];
        {% if not nobag %}
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        {% endif %}
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = SL_per_warp * warp_id;
        const int32_t sl_end = min(SL_per_warp * (warp_id + 1), SL);
        Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            int32_t sl_j = sl + threadIdx.x;
            {% if not nobag %}
            int32_t b_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
            int32_t b; //= b_t % B;
            int32_t t; //= b_t / B;
            fd.DivMod(b_t, &t, &b);
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0;
            {% else %}
            int64_t l_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
            int32_t l = l_t / T;
            {% endif %}
            {% if weighted %}
            at::acc_type<cache_t, true> idx_weight = sl_j < sl_end ? sorted_indice_weights[segment_start + sl_j] : 0.0;
            {% endif %}
            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                {% if not nobag %}
                int32_t b_j = SHFL_SYNC(b, j);
                int32_t D_start_j = SHFL_SYNC(D_start, j);
                {% else %}
                int32_t l_j = SHFL_SYNC(l, j);
                {% endif %}

                {% if weighted %}
                at::acc_type<cache_t, true> idx_weight_j = SHFL_SYNC(idx_weight, j);
                {% endif %}

        #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                    int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                    {% if not nobag %}
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                        &grad_output[b_j][0] + D_start_j + d);
                    {% else %}
                    Vec4T<at::acc_type<grad_t, true>> grad_out_vec(&grad_output[l_j][d]);
                    {% endif %}
                    {% if weighted %}
                    grad_sum[i].fma_(grad_out_vec, idx_weight_j);
                    {% else %}
                    grad_sum[i].add_(grad_out_vec);
                    {% endif %}
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
        {% if not dense %}
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
        {% for tensor in args.split_tensors %}
        at::acc_type<cache_t, true>* __restrict__ {{ tensor }};
        const auto {{ tensor }}_placement = static_cast<PlacementType>({{ tensor }}_placements[t_0]);
        int64_t {{ tensor }}_offset = {{ tensor }}_offsets[t_0];
        if ({{ tensor }}_placement == PlacementType::DEVICE) {
            {{ tensor }} = &{{ tensor }}_dev[{{ tensor }}_offset];
        } else {
            {{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset];
        }
        {% endfor %}


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
        {% else %}
#pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
            ++i) {
            int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
            auto& grad = grad_sum[i];
            grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
        }
        {% endif %}
    } // for each run
}


template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    size_t kMaxVecsPerThread,
    int32_t kThreadGroupSize = kWarpSize>
__global__
__launch_bounds__(kBackwardMaxThreads)
void
split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_warp_per_row_1(
    const at::PackedTensorAccessor64<grad_t, 2, at::RestrictPtrTraits>
        grad_output,
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> dev_weights,
    {% if not dense %}
    at::PackedTensorAccessor64<emb_t, 1, at::RestrictPtrTraits> uvm_weights,
    at::PackedTensorAccessor64<cache_t, 2, at::RestrictPtrTraits> lxu_cache_weights,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> weights_offsets,
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> D_offsets,
    {% else %}
    int32_t B,
    int64_t D,
    {% endif %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_run,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_cumulative_run_lengths,
    {% if not nobag %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> sorted_infos,
    {% else %}
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> sorted_infos,
    {% endif %}
    {% if not dense %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_lxu_cache_locations,
    {% endif %}
    {% if weighted %}
    const at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits> sorted_indice_weights,
    {% endif %}
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {% if not dense %}
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    {% else %}
    at::PackedTensorAccessor64<cache_t, 1, at::RestrictPtrTraits> grad_dev_weights,
    {% endif %}
    {% if not nobag %}
    FixedDivisor fd,
    {% endif %}
    {{ args.split_kernel_args | join(", ") }}) {

    {% if not nobag %}
    int32_t T = D_offsets.size(0) - 1;
    const int32_t B = grad_output.size(0);
    {% else %}
    int32_t T = weights_offsets.size(0);
    {% endif %}
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

    {% if not nobag %}
    int32_t t_0 = fd.Div(info_0); // info_0 / B;
    {% else %}
    int32_t t_0 = info_0 % T;
    {% endif %}

    int64_t hash_size = hash_size_cumsum[t_0];
    {% if not nobag %}
    int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
    {% endif %}
    int64_t idx = linear_index - hash_size;

    const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
    const int32_t sl_start = 0;
    const int32_t sl_end = SL;
    Vec4T<at::acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
    for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
        int32_t sl_j = sl + threadIdx.x;
        {% if not nobag %}
        int32_t b_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
        int32_t b; //= b_t % B;
        int32_t t; //= b_t / B;
        fd.DivMod(b_t, &t, &b);
        int32_t D_start = D_offsets[t];
        {% else %}
        int64_t l_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
        int32_t l = l_t / T;
        {% endif %}
        {% if weighted %}
        at::acc_type<cache_t, true> idx_weight = sl_j < sl_end ? sorted_indice_weights[segment_start + sl_j] : 0.0;
        {% endif %}

        for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
            {% if not nobag %}
            int32_t b_j = SHFL_SYNC(b, j);
            int32_t D_start_j = SHFL_SYNC(D_start, j);
            {% else %}
            int32_t l_j = SHFL_SYNC(l, j);
            {% endif %}
            {% if weighted %}
            at::acc_type<cache_t, true> idx_weight_j = SHFL_SYNC(idx_weight, j);
            {% endif %}

            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                    i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
                    ++i) {
                int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
                {% if not nobag %}
                Vec4T<at::acc_type<grad_t, true>> grad_out_vec(
                    &grad_output[b_j][0] + D_start_j + d);
                {% else %}
                Vec4T<at::acc_type<grad_t, true>> grad_out_vec(&grad_output[l_j][d]);
                {% endif %}
                {% if weighted %}
                grad_sum[i].fma_(grad_out_vec, idx_weight_j);
                {% else %}
                grad_sum[i].add_(grad_out_vec);
                {% endif %}
            }
        }
    }
    int64_t weights_offset = weights_offsets[t_0];
    {% if not dense %}
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
    {% for tensor in args.split_tensors %}
    at::acc_type<cache_t, true>* __restrict__ {{ tensor }};
    const auto {{ tensor }}_placement = static_cast<PlacementType>({{ tensor }}_placements[t_0]);
    int64_t {{ tensor }}_offset = {{ tensor }}_offsets[t_0];
    if ({{ tensor }}_placement == PlacementType::DEVICE) {
        {{ tensor }} = &{{ tensor }}_dev[{{ tensor }}_offset];
    } else {
        {{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset];
    }
    {% endfor %}

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
    {% else %}
#pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH < D;
        ++i) {
        int32_t d = (i * kThreadGroupSize + threadIdx.x) * VEC_WIDTH;
        auto& grad = grad_sum[i];
        grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
    }
    {% endif %}

    }
}

{{ "void" if not dense else "Tensor" }} split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_exact_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    {% if not dense %}
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    {% endif %}
    Tensor weights_offsets,
    {% if not nobag %}
    Tensor D_offsets,
    int64_t max_D,
    {% else %}
    int64_t D,
    {% endif %}
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    {% if not nobag %}
    int64_t pooling_mode,
    {% endif %}
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    {% if not dense %}
    Tensor lxu_cache_locations,
    {% endif %}
    int64_t unused_,
    int64_t max_segment_length_per_warp,
    {% if not dense %}
    bool stochastic_rounding,
    {% endif %}
    {{ args.split_function_args | join(", ") }}) {

    TENSOR_ON_CUDA_GPU(grad_output);
    TENSOR_ON_CUDA_GPU(dev_weights);
    {% if not dense %}
    TENSOR_ON_CUDA_GPU(uvm_weights);
    TENSOR_ON_CUDA_GPU(lxu_cache_weights);
    TENSOR_ON_CUDA_GPU(weights_placements);
    {% endif %}
    TENSOR_ON_CUDA_GPU(weights_offsets);
    {% if not nobag %}
    TENSOR_ON_CUDA_GPU(D_offsets);
    {% endif %}
    TENSOR_ON_CUDA_GPU(hash_size_cumsum);
    TENSOR_ON_CUDA_GPU(indices);
    TENSOR_ON_CUDA_GPU(offsets);
    {% if weighted %}
    TENSOR_ON_CUDA_GPU(indice_weights);
    {% endif %}
    {% if not dense %}
    TENSOR_ON_CUDA_GPU(lxu_cache_locations);
    {% endif %}

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    {% if dense %}
    auto grad_dev_weights = zeros_like(dev_weights);
    {% endif %}

    // short-circuit if there are zero indices.
    if (indices.numel() == 0) {
        return {{ "grad_dev_weights" if dense else "" }};
    }

    {% if not nobag %}
    int32_t T = D_offsets.numel() - 1;
    {% else %}
    int32_t T = weights_offsets.numel();
    {% endif %}

    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    const auto B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    auto BT_block_size = kMaxThreads / kWarpSize;
    TORCH_CHECK(BT_block_size * kWarpSize <= kMaxThreads);
    {% if nobag %}
    auto max_D = D;
    {% endif %}
    TORCH_CHECK(max_D <= {{ max_embedding_dim }});

    // V100: 96 KB; A100: 160 KB.
    int max_shared_bytes = 0;
#ifndef __HIP_PLATFORM_HCC__
    cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_weights.get_device());
#else
    // MI100 has 64 KB local memory (shared memory) per workgroup
    max_shared_bytes = 64 << 10;
#endif
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int shared_kb = max_shared_bytes >> 10;
    // V100: 64 KB; A100: 96 KB.
#ifndef __HIP_PLATFORM_HCC__
    // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
    int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
    TORCH_CHECK(used_shared_kb > 0);
#else
    // MI100 has independent shared mem and L1
    int used_shared_kb = shared_kb;
#endif
    int used_shared_bytes = used_shared_kb << 10;

    Tensor linear_indices, linear_indices_sorted;
    Tensor infos_sorted;
    Tensor sorted_linear_indices_run, sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths;
    std::tie(
        linear_indices,
        linear_indices_sorted,
        infos_sorted,
        sorted_linear_indices_run,
        sorted_linear_indices_run_lengths,
        sorted_linear_indices_num_runs,
        sorted_linear_indices_cumulative_run_lengths) =
        transpose_embedding_input(
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            {{"true" if nobag else "false"}});

    {% if not dense %}
    auto lxu_cache_locations_sorted = at::empty_like(lxu_cache_locations);
    if (lxu_cache_locations.size(0) > 0) {
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(radix_sort_pairs(
            nullptr,
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            indices.options().dtype(at::kByte));
        AT_CUDA_CHECK(radix_sort_pairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
    }
    {% endif %}

    DISPATCH_EMB_GRAD_CACHE_TYPES(
        dev_weights.scalar_type(),
        grad_output.scalar_type(),
        {% if not dense %}
        lxu_cache_weights.scalar_type(),
        {% else %}
        dev_weights.scalar_type(),
        {% endif %}
            "split_embedding_backward_{{ optimizer }}_exact_kernel",
        [&] {
            {% if weighted %}
            auto indice_weights_sorted = at::empty_like(indice_weights);
            {
            size_t temp_storage_bytes = 0;
            AT_CUDA_CHECK(radix_sort_pairs(
                nullptr,
                temp_storage_bytes,
                linear_indices.data_ptr<int64_t>(),
                linear_indices_sorted.data_ptr<int64_t>(),
                indice_weights.data_ptr<at::acc_type<cache_t, true>>(),
                indice_weights_sorted.data_ptr<at::acc_type<cache_t, true>>(),
                linear_indices.numel(),
                0,
                total_hash_size_bits,
                at::cuda::getCurrentCUDAStream(),
                false));
            auto temp_storage = at::empty(
                {static_cast<int64_t>(temp_storage_bytes)},
                indices.options().dtype(at::kByte));
            AT_CUDA_CHECK(radix_sort_pairs(
                temp_storage.data_ptr(),
                temp_storage_bytes,
                linear_indices.data_ptr<int64_t>(),
                linear_indices_sorted.data_ptr<int64_t>(),
                indice_weights.data_ptr<at::acc_type<cache_t, true>>(),
                indice_weights_sorted.data_ptr<at::acc_type<cache_t, true>>(),
                linear_indices.numel(),
                0,
                total_hash_size_bits,
                at::cuda::getCurrentCUDAStream(),
                false));
            }
            {% endif %}

            // early memory release
            linear_indices.reset();
            linear_indices_sorted.reset();

            auto grad_output_accessor = grad_output.packed_accessor64<grad_t, 2, at::RestrictPtrTraits>();
            {% if not nobag %}
            Tensor grad_output_mean;
            if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::MEAN) {
              grad_output_mean = at::empty_like(grad_output);
              grad_mean_kernel<grad_t>
                  <<<div_round_up((B * T), kMaxThreads / kWarpSize),
                     dim3(kWarpSize, kMaxThreads / kWarpSize),
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      grad_output_accessor,
                      D_offsets
                          .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                      offsets
                          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                      grad_output_mean.packed_accessor64<
                          grad_t, 2, at::RestrictPtrTraits>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              grad_output_accessor = grad_output_mean.packed_accessor64<
                  grad_t, 2, at::RestrictPtrTraits>();
            }
            {% endif %}

            {% if not dense %}
            at::PhiloxCudaState rng_engine_inputs;
            if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
                auto gen = at::cuda::detail::getDefaultCUDAGenerator();
                std::lock_guard<std::mutex> lock(gen.mutex());
                rng_engine_inputs =
                    at::check_generator<at::CUDAGeneratorImpl>(gen)
                        ->philox_cuda_state(4);
            }
            {% endif %}
            // kMaxElemPerThread is # of elements handled by thread if we use a full warp for a row
            // We consider kMaxElemPerThread 1 and 2, and then a multiple of 4.
            {% for kMaxElemPerThread in range(1, max_embedding_dim // (items_per_warp // 4) + 1) %}
            {% if kMaxElemPerThread in [1, 2] or kMaxElemPerThread % 4 == 0 %}
            if (max_D <= {{ items_per_warp // 4 * kMaxElemPerThread }}) {
            // hipcc can't use max in constexpr
            constexpr int kMaxVecsPerThread = {{ kMaxElemPerThread }} / 4 >= 1 ? {{ kMaxElemPerThread }} / 4 : 1;
            // If max_D is small, use fewer number of threads than kWarpSize.
#ifdef FBGEMM_USE_SUBWARP_SHUFFLE
            constexpr int kThreadGroupSize = kWarpSize / std::max(4 / {{ kMaxElemPerThread }}, 1);
#else
            constexpr int kThreadGroupSize = kWarpSize;
#endif
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<emb_t, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty_like(sorted_linear_indices_run_lengths);
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

            const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
            const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;
            Tensor long_run_id_to_really_long_run_ids;
            if (use_deterministic_algorithms) {
                long_run_id_to_really_long_run_ids =
                    at::empty(0, sorted_linear_indices_run_lengths.options());
            } else {
                long_run_id_to_really_long_run_ids =
                    at::empty_like(sorted_linear_indices_run_lengths);
            }
            auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
            auto grad_accum_counter = at::empty(
                use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                indices.options().dtype(at::kInt));

            split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_find_long_segments<<<
                div_round_up(indices.numel(), kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                sorted_linear_indices_num_runs.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                sorted_linear_indices_run_lengths.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                num_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                long_run_id_to_really_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                num_really_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                grad_accum_counter.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                max_segment_length_per_warp,
                max_segment_length_per_cta,
                use_deterministic_algorithms);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // A temp buffer to accumulate gradients with atomics.
            auto temp_grad_accum = at::zeros(
                {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

            int32_t grid_size = std::min(
                div_round_up(long_run_ids.numel(), kMaxThreads),
                get_max_thread_blocks_());

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".

#ifndef __HIP_PLATFORM_HCC__
            cudaFuncSetAttribute(
                split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB.
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            // dividing by kMaxThreads is a heuristic to avoid num of blocks far exceeding num_long_run_ids[0]
            split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_cta_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, BT_block_size),
                    BT_block_size * sizeof(at::acc_type<cache_t, true>) * 4 * kWarpSize *
                        kMaxVecsPerThread,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    {% if not dense %}
                    dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                    weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% if not nobag %}
                    D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    B,
                    D,
                    {% endif %}
                    hash_size_cumsum.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    sorted_linear_indices_run
                        .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    sorted_linear_indices_cumulative_run_lengths
                        .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    num_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% if not nobag %}
                    infos_sorted.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    infos_sorted.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not dense %}
                    lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if weighted %}
                    indice_weights_sorted.packed_accessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not dense %}
                    stochastic_rounding,
                    rng_engine_inputs,
                    {% else %}
                    grad_dev_weights.packed_accessor64<cache_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not nobag %}
                    FixedDivisor(B),
                    {% endif %}
                    long_run_id_to_really_long_run_ids.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    temp_grad_accum.packed_accessor32<at::acc_type<cache_t, true>, 2, at::RestrictPtrTraits>(),
                    grad_accum_counter.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    max_segment_length_per_cta,
                    use_deterministic_algorithms,
                    {{ args.split_kernel_arg_constructors | join(", ") }});
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            grid_size = std::min(
                div_round_up(sorted_linear_indices_run.numel(), kBackwardMaxThreads / kThreadGroupSize),
                get_max_thread_blocks_());

            // Shared memory is not needed for non uint8_t weights
            size_t shmem_bytes = 0;
            if (std::is_same<emb_t, uint8_t>::value) {
                shmem_bytes = BT_block_size * sizeof(
                    at::acc_type<cache_t, true>) * 4 * kWarpSize * kMaxVecsPerThread;
#ifndef __HIP_PLATFORM_HCC__
                cudaFuncSetAttribute(
                    split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_warp_per_row_1<
                    emb_t,
                    grad_t,
                    cache_t,
                    kMaxVecsPerThread,
                    kThreadGroupSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    used_shared_bytes); // V100: 64 KB; A100: 96 KB.
#endif
            }

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding{{ "_nobag" if nobag else "" }}_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_warp_per_row_1<
                emb_t,
                grad_t,
                cache_t,
                kMaxVecsPerThread,
                kThreadGroupSize>
                <<<grid_size,
                    dim3(kThreadGroupSize, kBackwardMaxThreads / kThreadGroupSize),
                    shmem_bytes,
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    {% if not dense %}
                    dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    uvm_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    lxu_cache_weights.packed_accessor64<cache_t, 2, at::RestrictPtrTraits>(),
                    weights_placements.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    dev_weights.packed_accessor64<emb_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    weights_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% if not nobag %}
                    D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    B,
                    D,
                    {% endif %}
                    hash_size_cumsum.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    sorted_linear_indices_run
                        .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    sorted_linear_indices_cumulative_run_lengths
                        .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% if not nobag %}
                    infos_sorted.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% else %}
                    infos_sorted.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not dense %}
                    lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if weighted %}
                    indice_weights_sorted.packed_accessor32<at::acc_type<cache_t, true>, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    sorted_linear_indices_num_runs
                        .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                    max_segment_length_per_warp,
                    {% if not dense %}
                    stochastic_rounding,
                    rng_engine_inputs,
                    {% else %}
                    grad_dev_weights.packed_accessor64<cache_t, 1, at::RestrictPtrTraits>(),
                    {% endif %}
                    {% if not nobag %}
                    FixedDivisor(B),
                    {% endif %}
                    {{ args.split_kernel_arg_constructors | join(", ") }});
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        {% endif %}
        {% endfor %}
        });

    return {{ "grad_dev_weights" if dense else "" }};
}
{% endif %}
{% endfor %}
// clang-format on
