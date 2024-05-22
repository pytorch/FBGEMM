/*******************************************************************************
 * Copyright (c) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include "split_tbe_common_hip.h"

template <typename cache_t, typename emb_t, int32_t embedding_dim, int32_t weight_decay_mode>
struct rowwise_adagrad_optimizer_t
{
    __device__ rowwise_adagrad_optimizer_t(const rowwise_adagrad_kernel_arg_t& karg_)
        : karg(karg_)
    {
    }

    // template<int32_t acc_length>
    // __device__ static void precompute(float * acc){
    //     // compute per row square sum
    // }
    template <int32_t thread_length, int32_t segment_split>
    __device__ void update(cache_t* acc, emb_t* weight, uint32_t row_index)
    {
        if constexpr(segment_split == 0)
        {
            cache_t * p_momentum = reinterpret_cast<cache_t*>(karg.p_momentum);
            cache_t momentum = p_momentum[row_index]; // should be s_load
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

            cache_t avg_square =
                wave_reduce<reduce_op_sum_t<cache_t>, cache_t, AMDGCN_WAVE_SIZE>(local_sum_squre) /
                embedding_dim;

            cache_t momentum_new = momentum + avg_square;

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
#pragma unroll
            for(auto i = 0; i < thread_length; i++)
            {
                cache_t w = static_cast<cache_t>(weight[i]);
                cache_t a = acc[i];
                w         = correction * w - multiplier * a;
                weight[i] = static_cast<emb_t>(w);
            }

            // printf("momentum_new:%f, avg_square:%f, row_index:%d, momentum:%f\n",  momentum_new,  avg_square, row_index, momentum);
            // printf("momentum_new:%f",  momentum_new);

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
          int32_t block_size,
          int32_t embedding_dim,
          int32_t segment_prefetch, // 2
          int32_t segment_unroll, // 8
          int32_t segment_split,  // 0-warp per row, 1-cta per row, 2-atomic(needed?)
          bool    weighted>
__device__ void split_tbe_backward_hip_kernel(
    const grad_t* p_output_grad,
    emb_t* p_emb_table,
    const int64_t* p_hash_size_cumsum,
    const int64_t* p_sorted_linear_indices_run,
    const int32_t* p_sorted_linear_indices_cumulative_run_lengths,
    const int32_t* p_sorted_linear_indices_num_runs,
    // const int32_t* p_long_run_ids,  // unused
    // const int32_t* p_num_long_run_ids, // unused
    const int32_t* p_sorted_infos,
    magic_div_u32_t batch_mdiv,
    uint32_t max_segment_length_per_warp,
    uint32_t emb_dim,
    uint32_t batch,
    uint32_t num_rows,
    uint32_t num_tables,
    optimizer_karg_t opt_karg,
    const float * p_sorted_indice_weights = nullptr)
{
    constexpr uint32_t dword_per_row   = (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    constexpr uint32_t waves_per_block = block_size / AMDGCN_WAVE_SIZE;
    constexpr uint32_t length_mask     = ~(segment_unroll - 1);
    const uint32_t wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / AMDGCN_WAVE_SIZE);
    const uint32_t lane_id = threadIdx.x % AMDGCN_WAVE_SIZE;
    const uint32_t run_id  = wave_id + blockIdx.x * waves_per_block;

    // printf("wave_id:%d, run_id:%d(%d), batch:%d(%d, %d)\n",
    //     wave_id, run_id, p_sorted_linear_indices_num_runs[0], batch, batch_mdiv.magic, batch_mdiv.shift);

    if(run_id >= p_sorted_linear_indices_num_runs[0])
        return;

    const int64_t linear_index  = p_sorted_linear_indices_run[run_id];

    const int32_t segment_start = p_sorted_linear_indices_cumulative_run_lengths[run_id];
    const int32_t segment_end   = p_sorted_linear_indices_cumulative_run_lengths[run_id + 1];

    int32_t info_0 = p_sorted_infos[segment_start];
    uint32_t t_0 = magic_div_u32_run(batch_mdiv, info_0);
    int64_t hash_size = p_hash_size_cumsum[t_0];

    const int64_t emb_idx       = linear_index - hash_size;

    // printf("[%d] segment_start:%d, info_0:%d, t_0:%d, num_rows:%d, emb_dim:%d, linear_index:%ld\n", wave_id, segment_start, info_0, t_0, num_rows, emb_dim, linear_index);

    // p_output_grad += t_0 * emb_dim;

    p_emb_table += hash_size * emb_dim;
    opt_karg.p_momentum = reinterpret_cast<void*>(reinterpret_cast<cache_t*>(opt_karg.p_momentum) + hash_size);

    const int32_t segment_length = segment_end - segment_start;

    if(segment_length >= max_segment_length_per_warp)
        return;

    // printf("[%d] segment_length:%d\n", wave_id, segment_length);

    const int32_t segment_length_mod = segment_length & length_mask;

    cache_t grad_acc[dword_per_row];
    int32_t infos[segment_unroll];
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
        magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
        load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
            &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

        magic_div_u32_run_with_mod(batch_mdiv, infos[1], batch, table_index, bag_index);
        load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
            &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
        if constexpr (!weighted){
            #pragma unroll
            for(int j = 2; j < segment_unroll; j += 2)
            {
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id);
                magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[dword_per_row], lane_id);
                magic_div_u32_run_with_mod(
                    batch_mdiv, infos[j + 1], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
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
                magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[j-1]);
                magic_div_u32_run_with_mod(
                    batch_mdiv, infos[j + 1], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
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
    magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
    load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
        &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

    magic_div_u32_run_with_mod(batch_mdiv, infos[1], batch, table_index, bag_index);
    load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
        &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

    if constexpr (!weighted) {
        #pragma unroll
        for(int j = 2; j < segment_unroll; j += 2)
        {
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id);
            magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id);
            magic_div_u32_run_with_mod(batch_mdiv, infos[j + 1], batch, table_index, bag_index);
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
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
            magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[j-1]);
            magic_div_u32_run_with_mod(batch_mdiv, infos[j + 1], batch, table_index, bag_index);
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
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

                magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
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

                magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id, indice_weights[0]);

                itr++;
            } while(itr < segment_length);
        }
    }

    // printf("[%d] segment_length:%d ==<< %f, emb_idx:%ld\n", wave_id, segment_length, grad_acc[0], emb_idx);

    // load the old emb weight data
    load_row_per_warp<emb_t, embedding_dim, int64_t>::run(
        &emb_data[0], emb_idx, p_emb_table, lane_id);
    optimizer_t optimizer(opt_karg);
    optimizer.template update<dword_per_row, segment_split>(grad_acc, emb_data, emb_idx);

    // store updated weight to grad ??
    store_row_per_warp<emb_t, embedding_dim, emb_t>::run(&emb_data[0], p_emb_table + emb_idx * embedding_dim, lane_id);
}


// PR23: for the reference, the template is
// hip_split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_kernel_warp_per_row_1(

template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    bool kUseVecBlocking,
    int32_t embedding_dim>
//__global__ __launch_bounds__(kBackwardMaxThreads) void
__global__ void
// {%- if is_index_select %}
// batch_index_select_dim0_codegen_backward_kernel_warp_per_row(
// {%- else %}
split_embedding{{ ndesc }}_backward_codegen_{{ optimizer }}_{{ wdesc }}{{ vdesc }}_kernel_warp_per_row_1(
// {%- endif %}
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
    const bool use_uniq_cache_locations,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> table_unique_indices_offsets,
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
    {%- endif %} // if not dense and optimizer != "none"
    {%- if not nobag and vbe %}
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> B_offsets,
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> row_output_offsets,
    {%- endif %}
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    const int32_t max_D,
    const int32_t max_vecs_per_thread,
    magic_div_u32_t batch_mdiv, // PR23 extra
    const int32_t batch, // PR23 extra
    const int32_t num_rows, // PR23 extra
    const int32_t num_tables, // PR23 extra
    // {%- if is_index_select %}
    // const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> grad_offsets,
    // const bool permute_output_dim_0_1
    // {%- else %}
    {{ args.split_kernel_args | replace_pta_namespace() | join(",\n    ") }}
    // {%- endif %}
) {
    // WIP: test the build system
#if 0
    auto p_output_grad = grad_output.data();
    auto p_emb_table = dev_weights.data();
    auto p_hash_size_cumsum = hash_size_cumsum.data();
    auto p_sorted_linear_indices_run = sorted_linear_indices_run.data();
    auto p_sorted_linear_indices_cumulative_run_lengths = sorted_linear_indices_cumulative_run_lengths.data();
    auto p_sorted_linear_indices_num_runs = sorted_linear_indices_num_runs.data();
    p_sorted_infos = sorted_infos.data();
    // max_segment_length_per_warp = max_segment_length_per_warp;
    // indice_weights_sorted = indice_weights_sorted.packed_accessor32<float, 1, at::RestrictPtrTraits>().data();
    auto emb_dim = embedding_dim;
    // TODO
    // num_rows = dev_weights.numel() / T / max_D;
    // num_tables = T;
    split_tbe_backward_hip_kernel<
        {{optimizer}}_t<cache_t, emb_t, embedding_dim, /* weight_decay_mode */ 0>,
        {{optimizer}}_kernel_arg_t,
        emb_t,
        cache_t, // cache_t
        grad_t,
        BLOCK_SIZE,
        embedding_dim,
        segment_prefetch,
        segment_unroll,
        segment_split,
        false>(p_output_grad,
               p_emb_table,
               p_hash_size_cumsum,
               p_sorted_linear_indices_run,
               p_sorted_linear_indices_cumulative_run_lengths,
               p_sorted_linear_indices_num_runs,
               // p_long_run_ids,  // unused
               // p_num_long_run_ids,  // unused
               p_sorted_infos,
               batch_mdiv,
               max_segment_length_per_warp,
               emb_dim,
               batch,
               num_rows,
               num_tables,
               opt_karg);
#endif
}

#if 0  // Upstream moved to a differnet approach
#define __SPLIT_TBE_BWD_KERNEL(optimizer,                                                                                            \
                                          weight_decay_mode,                                                                         \
                                          segment_split,                                                                             \
                                          emb_prec,                                                                                  \
                                          emb_type,                                                                                  \
                                          grad_prec,                                                                                 \
                                          grad_type,                                                                                 \
                                          embedding_dim,                                                                             \
                                          segment_prefetch,                                                                          \
                                          segment_unroll)                                                                            \
    extern "C" __global__ void                                                                                                       \
        hip_split_tbe_bwd_unweighted_hip_kernel_##optimizer##_w##weight_decay_mode##_s##segment_split##_##emb_prec##_##grad_prec##_e##embedding_dim(   \
            const grad_type* p_output_grad,                                                                                          \
            emb_type* p_emb_table,                                                                                                   \
            const int64_t* p_hash_size_cumsum,                                                                                       \
            const int64_t* p_sorted_linear_indices_run,                                                                              \
            const int32_t* p_sorted_linear_indices_cumulative_run_lengths,                                                           \
            const int32_t* p_sorted_linear_indices_num_runs,                                                                         \
            const int32_t* p_long_run_ids,                                                                                           \
            const int32_t* p_num_long_run_ids,                                                                                       \
            const int32_t* p_sorted_infos,                                                                                           \
            magic_div_u32_t batch_mdiv,                                                                                              \
            uint32_t max_segment_length_per_warp,                                                                                    \
            uint32_t emb_dim,                                                                                                        \
            uint32_t batch,                                                                                                          \
            uint32_t num_rows,                                                                                                       \
            uint32_t num_tables,                                                                                                     \
            optimizer##_kernel_arg_t opt_karg)                                                                                       \
    {                                                                                                                                \
        split_tbe_backward_hip_kernel<                                                                                               \
            optimizer##_optimizer_t<float, emb_type, embedding_dim, weight_decay_mode>,                                              \
            optimizer##_kernel_arg_t,                                                                                                \
            emb_type,                                                                                                                \
            float,                                                                                                                   \
            grad_type,                                                                                                               \
            BLOCK_SIZE,                                                                                                              \
            embedding_dim,                                                                                                           \
            segment_prefetch,                                                                                                        \
            segment_unroll,                                                                                                          \
            segment_split,                                                                                                           \
            false>(p_output_grad,                                                                                                    \
                           p_emb_table,                                                                                              \
                           p_hash_size_cumsum,                                                                                       \
                           p_sorted_linear_indices_run,                                                                              \
                           p_sorted_linear_indices_cumulative_run_lengths,                                                           \
                           p_sorted_linear_indices_num_runs,                                                                         \
                           p_long_run_ids,                                                                                           \
                           p_num_long_run_ids,                                                                                       \
                           p_sorted_infos,                                                                                           \
                           batch_mdiv,                                                                                               \
                           max_segment_length_per_warp,                                                                              \
                           emb_dim,                                                                                                  \
                           batch,                                                                                                    \
                           num_rows,                                                                                                 \
                           num_tables,                                                                                               \
                           opt_karg);                                                                                                \
    }                                                                                                                                \
                                                                                                                                     \
    extern "C" __global__ void                                                                                                       \
        hip_split_tbe_bwd_weighted_hip_kernel_##optimizer##_w##weight_decay_mode##_s##segment_split##_##emb_prec##_##grad_prec##_e##embedding_dim(   \
            const grad_type* p_output_grad,                                                                                          \
            emb_type* p_emb_table,                                                                                                   \
            const int64_t* p_hash_size_cumsum,                                                                                       \
            const int64_t* p_sorted_linear_indices_run,                                                                              \
            const int32_t* p_sorted_linear_indices_cumulative_run_lengths,                                                           \
            const int32_t* p_sorted_linear_indices_num_runs,                                                                         \
            const int32_t* p_long_run_ids,                                                                                           \
            const int32_t* p_num_long_run_ids,                                                                                       \
            const int32_t* p_sorted_infos,                                                                                           \
            magic_div_u32_t batch_mdiv,                                                                                              \
            uint32_t max_segment_length_per_warp,                                                                                    \
            const float * p_indice_weights,                                                                                          \
            uint32_t emb_dim,                                                                                                        \
            uint32_t batch,                                                                                                          \
            uint32_t num_rows,                                                                                                       \
            uint32_t num_tables,                                                                                                     \
            optimizer##_kernel_arg_t opt_karg)                                                                                       \
    {                                                                                                                                \
        split_tbe_backward_hip_kernel<                                                                                               \
            optimizer##_optimizer_t<float, emb_type, embedding_dim, weight_decay_mode>,                                              \
            optimizer##_kernel_arg_t,                                                                                                \
            emb_type,                                                                                                                \
            float,                                                                                                                   \
            grad_type,                                                                                                               \
            BLOCK_SIZE,                                                                                                              \
            embedding_dim,                                                                                                           \
            segment_prefetch,                                                                                                        \
            segment_unroll,                                                                                                          \
            segment_split,                                                                                                           \
            true>(p_output_grad,                                                                                                     \
                           p_emb_table,                                                                                              \
                           p_hash_size_cumsum,                                                                                       \
                           p_sorted_linear_indices_run,                                                                              \
                           p_sorted_linear_indices_cumulative_run_lengths,                                                           \
                           p_sorted_linear_indices_num_runs,                                                                         \
                           p_long_run_ids,                                                                                           \
                           p_num_long_run_ids,                                                                                       \
                           p_sorted_infos,                                                                                           \
                           batch_mdiv,                                                                                               \
                           max_segment_length_per_warp,                                                                              \
                           emb_dim,                                                                                                  \
                           batch,                                                                                                    \
                           num_rows,                                                                                                 \
                           num_tables,                                                                                               \
                           opt_karg,                                                                                                 \
                           p_indice_weights);                                                                                        \
    }

#define SPLIT_TBE_BWD_KERNEL_ALL_WDM(optimizer,                                 \
                                          segment_split,                        \
                                          emb_prec,                             \
                                          emb_type,                             \
                                          grad_prec,                            \
                                          grad_type,                            \
                                          embedding_dim,                        \
                                          segment_prefetch,                     \
                                          segment_unroll)                       \
    __SPLIT_TBE_BWD_KERNEL(optimizer, 0, segment_split, emb_prec, emb_type, grad_prec, grad_type, embedding_dim, segment_prefetch, segment_unroll)  \
    __SPLIT_TBE_BWD_KERNEL(optimizer, 1, segment_split, emb_prec, emb_type, grad_prec, grad_type, embedding_dim, segment_prefetch, segment_unroll)  \
    __SPLIT_TBE_BWD_KERNEL(optimizer, 2, segment_split, emb_prec, emb_type, grad_prec, grad_type, embedding_dim, segment_prefetch, segment_unroll)


#define SPLIT_TBE_BWD_KERNEL(optimizer,                               \
                                segment_split,                        \
                                embedding_dim)                        \
    SPLIT_TBE_BWD_KERNEL_ALL_WDM(optimizer, segment_split, fp32, float, fp32, float, embedding_dim, 2, 8) \
    SPLIT_TBE_BWD_KERNEL_ALL_WDM(optimizer, segment_split, fp32, float, fp16,  half, embedding_dim, 2, 8) \
    SPLIT_TBE_BWD_KERNEL_ALL_WDM(optimizer, segment_split, fp16,  half, fp32, float, embedding_dim, 2, 8) \
    SPLIT_TBE_BWD_KERNEL_ALL_WDM(optimizer, segment_split, fp16,  half, fp16,  half, embedding_dim, 2, 8)

// warp per row
SPLIT_TBE_BWD_KERNEL(rowwise_adagrad, 0, 64)
SPLIT_TBE_BWD_KERNEL(rowwise_adagrad, 0, 128)
SPLIT_TBE_BWD_KERNEL(rowwise_adagrad, 0, 192)
SPLIT_TBE_BWD_KERNEL(rowwise_adagrad, 0, 256)
#endif

