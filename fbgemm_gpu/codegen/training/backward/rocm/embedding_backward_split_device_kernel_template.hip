/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*******************************************************************************
 * Copyright (c) 2016 - 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <type_traits>

#include <ATen/AccumulateType.h>
// Provides at::PhiloxCudaState + at::cuda::philox::unpack. This is a raw .hip
// translation unit, and the OSS hipify pipeline does NOT rewrite #includes
// inside .hip files. So we must include the ROCm (ATen/hip) header path
// directly: an ATen/cuda/... include would (a) resolve to the CUDA header that
// pulls <cuda.h> (absent in the ROCm toolchain) and (b) define a second,
// CUDA-flavored at::PhiloxCudaState that clashes with the HIP one already in the
// enclosing warp TU. PhiloxUtils.cuh is header-guarded, so it de-dups with the
// warp's copy. (The namespace stays at::cuda::philox even under ROCm.)
#ifdef USE_ROCM
#include <ATen/hip/PhiloxUtils.cuh>
#else
#include <ATen/cuda/PhiloxUtils.cuh>
#endif
#include "fbgemm_gpu/rocm/split_embeddings_common.h"

namespace fbgemm_gpu::rocm {

// Stochastic-rounding primitives, defined locally in fbgemm_gpu::rocm rather
// than pulled from stochastic_rounding.cuh (directly or via
// rocm/stochastic_rounding.h). This device kernel is compiled both standalone
// (gen_..._device_kernel_hip.hip.o) and #included into the warp kernel, and
// neither of those headers is usable from a raw .hip TU:
//   (a) they transitively include cuda_prelude.cuh -> <cuda.h> and
//       <curand_kernel.h>, which do not exist on the raw ROCm compile, and
//   (b) in the #included build they would re-define
//       fbgemm_gpu::StochasticRoundingRNGState against the hipified copy already
//       present in the warp TU.
// Keeping our own copy in the fbgemm_gpu::rocm namespace sidesteps both: it is
// self-contained for the standalone compile and does not collide with the
// fbgemm_gpu:: symbols in the #included compile.

// xorshift* RNG state seeded from a philox state; mirrors
// fbgemm_gpu::StochasticRoundingRNGState.
struct StochasticRoundingRNGState {
  uint64_t state = 0;

  // From https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h
  __device__ __forceinline__ uint64_t splitmix64_stateless(uint64_t index) {
    uint64_t z = (index + UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
  }

  // salt_value should differ for every run and every thread.
  __device__ __forceinline__ void init(
      const at::PhiloxCudaState& philox_state,
      const uint64_t salt_value) {
    const auto [s0, s1] = at::cuda::philox::unpack(philox_state);
    state = splitmix64_stateless(s0 ^ s1) ^ splitmix64_stateless(salt_value);
    // Never allow a zero state (astronomically unlikely, but still).
    if (state == 0) {
      state = 1;
    }
  }

  // xorshift* -- see https://en.wikipedia.org/wiki/Xorshift#xorshift*
  __device__ __forceinline__ uint4 rand4() {
    uint4 random_bits = {0, 0, 0, 0};
    uint64_t x = state; // must be seeded nonzero
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    random_bits.x = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    random_bits.y = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    random_bits.z = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    random_bits.w = (x * UINT64_C(0x2545F4914F6CDD1D)) >> 32;
    state = x;
    return random_bits;
  }
};

// Correct for cases where x is not subnormal; mirrors
// fbgemm_gpu::stochastic_rounding_scalar.
__device__ __forceinline__ __half
stochastic_rounding_scalar(float x, uint32_t random_value) {
  uint32_t w_int = __float_as_uint(x);
  unsigned assembles = (w_int & 0xff800000) | (random_value >> 19);
  unsigned subtract = (w_int & 0xff800000);
  float assemble_float = __uint_as_float(assembles) - __uint_as_float(subtract);
  return __float2half_rz(x + assemble_float);
}

// Writes `thread_length` fp32 accumulator values into reduced-precision `output`
// with stochastic rounding. SR is only representable for at::Half on this scalar
// path (stochastic_rounding_scalar yields __half); any other emb_t falls back to
// a plain round-to-nearest cast.
template <typename emb_t, typename cache_t, int32_t thread_length>
__device__ __forceinline__ void stochastic_rounding_store_vector(
    emb_t* output,
    const cache_t* value,
    StochasticRoundingRNGState& state) {
  if constexpr (std::is_same_v<emb_t, at::Half>) {
#pragma unroll
    for (int32_t i = 0; i < thread_length; i += 4) {
      const uint4 random_bits = state.rand4();
      const uint32_t bits[4] = {
          random_bits.x, random_bits.y, random_bits.z, random_bits.w};
#pragma unroll
      for (int32_t j = 0; j < 4; j++) {
        if (i + j < thread_length) {
          output[i + j] = stochastic_rounding_scalar(
              static_cast<float>(value[i + j]), bits[j]);
        }
      }
    }
  } else {
#pragma unroll
    for (int32_t i = 0; i < thread_length; i++) {
      output[i] = static_cast<emb_t>(value[i]);
    }
  }
}

// Scalar-array weight write with plain round-to-nearest (no SR). Used for the
// SR-disabled case and for fp32 weights (where SR is a no-op).
template <typename emb_t, typename cache_t, int32_t thread_length>
__device__ __forceinline__ void nearest_rounding_store_vector(
    emb_t* output,
    const cache_t* value) {
#pragma unroll
  for (int32_t i = 0; i < thread_length; i++) {
    output[i] = static_cast<emb_t>(value[i]);
  }
}

template <typename cache_t, typename emb_t, typename index_t, int32_t embedding_dim, int32_t weight_decay_mode>
struct rowwise_adagrad_optimizer_t
{
    __device__ rowwise_adagrad_optimizer_t(const rowwise_adagrad_kernel_arg_t& karg_)
        : karg(karg_)
    {
    }

    template <int32_t thread_length, int32_t segment_split>
    __device__ void update(
        cache_t* acc,
        emb_t* weight,
        index_t row_index,
        StochasticRoundingRNGState* sr_state = nullptr)
    {
        if constexpr(segment_split == 0)
        {
            using momentum_t = at::acc_type<cache_t, true>;
            momentum_t* p_momentum = reinterpret_cast<momentum_t*>(karg.p_momentum);
            momentum_t momentum = p_momentum[row_index]; // should be s_load
            // compute per row square sum
            cache_t local_sum_squre = .0f;
            if constexpr(weight_decay_mode == 1)
            {
#pragma unroll
                for(auto i = 0; i < thread_length; i++)
                {
                    cache_t w = static_cast<cache_t>(weight[i]);
                    cache_t a = acc[i] + w * karg.weight_decay;
                    local_sum_squre += a * a;
                }
            }
            else
            {
#pragma unroll
                for(auto i = 0; i < thread_length; i++)
                {
                    cache_t a = acc[i];
                    local_sum_squre += a * a;
                }
            }

            momentum_t avg_square =
                static_cast<momentum_t>(wave_reduce<reduce_op_sum_t<cache_t>, cache_t, AMDGCN_WAVE_SIZE>(local_sum_squre)) /
                embedding_dim;

            momentum_t momentum_new = momentum + avg_square;

            cache_t multiplier = karg.learning_rate / (sqrtf(momentum_new) + karg.eps);
            cache_t correction;

            if constexpr(weight_decay_mode == 1)
            {
                correction = 1.0 - multiplier * karg.weight_decay;
            }
            else if constexpr(weight_decay_mode == 2)
            {
                correction = 1.0 - karg.learning_rate * karg.weight_decay;
            }
            else
            {
                correction = 1.0;
            }

// update new weight value
            cache_t updated_weight[thread_length];
#pragma unroll
            for(auto i = 0; i < thread_length; i++)
            {
                cache_t w = static_cast<cache_t>(weight[i]);
                cache_t a = acc[i];
                updated_weight[i] = correction * w - multiplier * a;
            }
            // Write the fp32 accumulator back to the emb_t weights. When SR is
            // armed (sr_state != nullptr; only the case for at::Half), copy the
            // RNG state into a register-local and apply stochastic rounding;
            // otherwise use plain round-to-nearest. The on/off branch lives here
            // (per row, warp-uniform); the helpers themselves are branch-free.
            if (sr_state != nullptr)
            {
                StochasticRoundingRNGState local_state = *sr_state;
                stochastic_rounding_store_vector<emb_t, cache_t, thread_length>(
                    weight, updated_weight, local_state);
            }
            else
            {
                nearest_rounding_store_vector<emb_t, cache_t, thread_length>(
                    weight, updated_weight);
            }

            p_momentum[row_index] = momentum_new;
        }
    }

    rowwise_adagrad_kernel_arg_t karg;
};

template <typename optimizer_t,
          typename optimizer_karg_t,
          typename emb_t,
          typename cache_t,
          typename grad_t,
          typename index_t,
          int32_t block_size,
          int32_t embedding_dim,
          int32_t segment_prefetch, // 2
          int32_t segment_unroll, // 8
          int32_t segment_split,  // 0-warp per row, 1-cta per row, 2-atomic
          bool    weighted>
__device__ void split_tbe_backward_hip_kernel_{{ kdesc }}(
    const grad_t* p_output_grad,
    emb_t* p_emb_table,
    const int64_t* p_hash_size_cumsum,
    const index_t* p_sorted_linear_indices_run,
    const int32_t* p_sorted_linear_indices_cumulative_run_lengths,
    const int32_t* p_sorted_linear_indices_num_runs,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    const int32_t* p_sorted_infos,
    uint32_t max_segment_length_per_warp,
    uint32_t emb_dim,
    uint32_t num_tables,
    optimizer_karg_t opt_karg,
    bool stochastic_rounding,
    at::PhiloxCudaState stochastic_rounding_philox_args,
    const float * p_sorted_indice_weights = nullptr)
{
    constexpr uint32_t dword_per_row   = (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    constexpr uint32_t waves_per_block = block_size / AMDGCN_WAVE_SIZE;
    constexpr uint32_t length_mask     = ~(segment_unroll - 1);
    const uint32_t wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / AMDGCN_WAVE_SIZE);
    const uint32_t lane_id = threadIdx.x % AMDGCN_WAVE_SIZE;
    const uint32_t run_id  = wave_id + blockIdx.x * waves_per_block;

    if(run_id >= p_sorted_linear_indices_num_runs[0])
    {
        return;
    }

    const auto linear_index  = p_sorted_linear_indices_run[run_id];

    const int32_t segment_start = p_sorted_linear_indices_cumulative_run_lengths[run_id];
    const int32_t segment_end   = p_sorted_linear_indices_cumulative_run_lengths[run_id + 1];

    const auto info_0 = reinterpret_cast<const uint32_t*>(&p_sorted_infos[0])[segment_start];
    const auto t_0 = info_0 >> info_B_num_bits;

    int64_t hash_size = p_hash_size_cumsum[t_0];

    const int64_t emb_idx = linear_index - hash_size;

    p_emb_table += hash_size * emb_dim;
    using momentum_t = at::acc_type<cache_t, true>;
    opt_karg.p_momentum = reinterpret_cast<void*>(reinterpret_cast<momentum_t*>(opt_karg.p_momentum) + hash_size);

    const int32_t segment_length = segment_end - segment_start;

    if(segment_length >= max_segment_length_per_warp)
        return;

    const int32_t segment_length_mod = segment_length & length_mask;

    cache_t grad_acc[dword_per_row];
    uint32_t infos[segment_unroll];
    grad_t grad_data[dword_per_row * segment_prefetch];
    emb_t emb_data[dword_per_row];
    float indice_weights[segment_unroll];

    #pragma unroll
    for(int i=0; i < dword_per_row; i++)
    {
        grad_acc[i] = .0f;
    }

    int itr = 0;
    if(segment_length_mod == 0)
        goto L_tail_grad_acc;

    if constexpr (!weighted) {
        #pragma unroll
        for(int i = 0; i < segment_unroll; i++)
        {
            infos[i] = p_sorted_infos[segment_start + i];
        }
    } else {
        for(int i = 0; i < segment_unroll; i++)
        {
            infos[i] = p_sorted_infos[segment_start + i];
            indice_weights[i] = p_sorted_indice_weights[segment_start + i];
        }
    }

    itr += segment_unroll;
    p_sorted_infos += segment_unroll;

    if constexpr (weighted) {
        p_sorted_indice_weights += segment_unroll;
    }

    uint32_t bag_index;
    uint32_t table_index;

    // LOOP
    for(; itr < segment_length_mod; itr += segment_unroll)
    {
        table_index = infos[0] >> info_B_num_bits;
        bag_index = infos[0] & info_B_mask;

        load_row_per_warp<grad_t, embedding_dim, index_t>::run(
            &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

        table_index = infos[1] >> info_B_num_bits;
        bag_index = infos[1] & info_B_mask;

        load_row_per_warp<grad_t, embedding_dim, index_t>::run(
            &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
        if constexpr (!weighted){
            #pragma unroll
            for(int j = 2; j < segment_unroll; j += 2)
            {
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id);

                table_index = infos[j] >> info_B_num_bits;
                bag_index = infos[j] & info_B_mask;

                load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[dword_per_row], lane_id);

                table_index = infos[j + 1] >> info_B_num_bits;
                bag_index = infos[j + 1] & info_B_mask;

                load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                    &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
            }

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id);
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id);

            #pragma unroll
            for(int i = 0; i < segment_unroll; i++)
            {
                infos[i] = p_sorted_infos[segment_start + i];
            }
            p_sorted_infos += segment_unroll;


        } else {
            #pragma unroll
            for(int j = 2; j < segment_unroll; j += 2)
            {
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id, indice_weights[j-2]);

                table_index = infos[j] >> info_B_num_bits;
                bag_index = infos[j] & info_B_mask;

                load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[j-1]);

                table_index = infos[j + 1] >> info_B_num_bits;
                bag_index = infos[j + 1] & info_B_mask;

                load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                    &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
            }

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id, indice_weights[segment_unroll-2]);
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[segment_unroll-1]);

            #pragma unroll
            for(int i = 0; i < segment_unroll; i++)
            {
                infos[i] = p_sorted_infos[segment_start + i];
                indice_weights[i] = p_sorted_indice_weights[segment_start + i];
            }
            p_sorted_infos += segment_unroll;
            p_sorted_indice_weights += segment_unroll;
        }
    }

    // LAST
    table_index = infos[0] >> info_B_num_bits;
    bag_index = infos[0] & info_B_mask;

    load_row_per_warp<grad_t, embedding_dim, index_t>::run(
        &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

    table_index = infos[1] >> info_B_num_bits;
    bag_index = infos[1] & info_B_mask;

    load_row_per_warp<grad_t, embedding_dim, index_t>::run(
        &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

    if constexpr (!weighted) {
        #pragma unroll
        for(int j = 2; j < segment_unroll; j += 2)
        {
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id);

            table_index = infos[j] >> info_B_num_bits;
            bag_index = infos[j] & info_B_mask;

            load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id);

            table_index = infos[j + 1] >> info_B_num_bits;
            bag_index = infos[j + 1] & info_B_mask;

            load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
        }

        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
            &grad_acc[0], &grad_data[0], lane_id);
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
            &grad_acc[0], &grad_data[dword_per_row], lane_id);
    } else {
        #pragma unroll
        for(int j = 2; j < segment_unroll; j += 2)
        {
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id, indice_weights[j-2]);

            table_index = infos[j] >> info_B_num_bits;
            bag_index = infos[j] & info_B_mask;

            load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[j-1]);

            table_index = infos[j + 1] >> info_B_num_bits;
            bag_index = infos[j + 1] & info_B_mask;

            load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
        }

        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
            &grad_acc[0], &grad_data[0], lane_id, indice_weights[segment_unroll-2]);
        accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
            &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[segment_unroll-1]);
    }

L_tail_grad_acc:
    if(segment_length & (segment_unroll - 1))
    {
        if constexpr (!weighted){
            // last, load one by one
            do
            {
                infos[0] = p_sorted_infos[segment_start];
                p_sorted_infos++;

                table_index = infos[0] >> info_B_num_bits;
                bag_index = infos[0] & info_B_mask;

                load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id);

                itr++;
            } while(itr < segment_length);
        } else {
            do
            {
                infos[0] = p_sorted_infos[segment_start];
                indice_weights[0] = p_sorted_indice_weights[segment_start];
                p_sorted_infos++;
                p_sorted_indice_weights++;

                table_index = infos[0] >> info_B_num_bits;
                bag_index = infos[0] & info_B_mask;

                load_row_per_warp<grad_t, embedding_dim, index_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id, indice_weights[0]);

                itr++;
            } while(itr < segment_length);
        }
    }

    // load the old emb weight data
    load_row_per_warp<emb_t, embedding_dim, index_t>::run(
        &emb_data[0], emb_idx, p_emb_table, lane_id);
    optimizer_t optimizer(opt_karg);

    // Seed a per-thread, per-run stochastic-rounding RNG state when SR is
    // enabled and the weights are at::Half (SR is a no-op for fp32 and is only
    // supported for Half on this scalar write path). The salt mirrors the
    // generic path's per-thread/per-run convention. The philox state is passed
    // as a kernel parameter (rather than via opt_karg) so the broadly-included
    // rocm/split_embeddings_common.h need not depend on ATen.
    StochasticRoundingRNGState sr_state;
    StochasticRoundingRNGState* sr_state_ptr = nullptr;
    if constexpr (std::is_same_v<emb_t, at::Half>) {
        if (stochastic_rounding) {
            sr_state.init(
                stochastic_rounding_philox_args,
                static_cast<uint64_t>(run_id) * AMDGCN_WAVE_SIZE + lane_id);
            sr_state_ptr = &sr_state;
        }
    }
    optimizer.template update<dword_per_row, segment_split>(grad_acc, emb_data, emb_idx, sr_state_ptr);

    store_row_per_warp<emb_t, embedding_dim>::run(&emb_data[0], p_emb_table + emb_idx * embedding_dim, lane_id);
}
} // namespace fbgemm_gpu::rocm
