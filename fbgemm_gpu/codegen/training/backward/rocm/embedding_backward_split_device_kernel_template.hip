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

#include "fbgemm_gpu/rocm/split_embeddings_common.h"

template <typename cache_t, typename emb_t, int32_t embedding_dim, int32_t weight_decay_mode>
struct rowwise_adagrad_optimizer_t
{
    __device__ rowwise_adagrad_optimizer_t(const rowwise_adagrad_kernel_arg_t& karg_)
        : karg(karg_)
    {
    }

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
          int32_t segment_split,  // 0-warp per row, 1-cta per row, 2-atomic
          bool    weighted>
__device__ void split_tbe_backward_hip_kernel_{{ kdesc }}(
    const grad_t* p_output_grad,
    emb_t* p_emb_table,
    const int64_t* p_hash_size_cumsum,
    const int64_t* p_sorted_linear_indices_run,
    const int32_t* p_sorted_linear_indices_cumulative_run_lengths,
    const int32_t* p_sorted_linear_indices_num_runs,
    {%- if not nobag %}
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    {%- endif %}
    {%- if not nobag %}
    const int32_t* p_sorted_infos,
    {%- else %}
    const int64_t* p_sorted_infos,
    {%- endif %}
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

    if(run_id >= p_sorted_linear_indices_num_runs[0])
    {
        return;
    }

    const int64_t linear_index  = p_sorted_linear_indices_run[run_id];

    const int32_t segment_start = p_sorted_linear_indices_cumulative_run_lengths[run_id];
    const int32_t segment_end   = p_sorted_linear_indices_cumulative_run_lengths[run_id + 1];

    {%- if nobag %}
    const auto info_0 = p_sorted_infos[segment_start];
    int32_t t_0 = info_0 % num_tables;
    {%- else %}
    const auto info_0 = reinterpret_cast<const uint32_t*>(&p_sorted_infos[0])[segment_start];
    const auto t_0 = info_0 >> info_B_num_bits;
    {%- endif %}
    int64_t hash_size = p_hash_size_cumsum[t_0];

    const int64_t emb_idx = linear_index - hash_size;

    p_emb_table += hash_size * emb_dim;
    opt_karg.p_momentum = reinterpret_cast<void*>(reinterpret_cast<cache_t*>(opt_karg.p_momentum) + hash_size);

    const int32_t segment_length = segment_end - segment_start;

    if(segment_length >= max_segment_length_per_warp)
        return;

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
        {%- if nobag %}
        magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
        {%- else %}
        table_index = infos[0] >> info_B_num_bits;
        bag_index = infos[0] & info_B_mask;
        {%- endif %}
        load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
            &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

        {%- if nobag %}
        magic_div_u32_run_with_mod(batch_mdiv, infos[1], batch, table_index, bag_index);
        {%- else %}
        table_index = infos[1] >> info_B_num_bits;
        bag_index = infos[1] & info_B_mask;
        {%- endif %}
        load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
            &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
        if constexpr (!weighted){
            #pragma unroll
            for(int j = 2; j < segment_unroll; j += 2)
            {
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id);
                {%- if nobag %}
                magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
                {%- else %}
                table_index = infos[j] >> info_B_num_bits;
                bag_index = infos[j] & info_B_mask;
                {%- endif %}
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[dword_per_row], lane_id);
                {%- if nobag %}
                magic_div_u32_run_with_mod(batch_mdiv, infos[j + 1], batch, table_index, bag_index);
                {%- else %}
                table_index = infos[j + 1] >> info_B_num_bits;
                bag_index = infos[j + 1] & info_B_mask;
                {%- endif %}
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
                {%- if nobag %}
                magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
                {%- else %}
                table_index = infos[j] >> info_B_num_bits;
                bag_index = infos[j] & info_B_mask;
                {%- endif %}
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[j-1]);
                {%- if nobag %}
                magic_div_u32_run_with_mod(batch_mdiv, infos[j + 1], batch, table_index, bag_index);
                {%- else %}
                table_index = infos[j + 1] >> info_B_num_bits;
                bag_index = infos[j + 1] & info_B_mask;
                {%- endif %}
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
    {%- if nobag %}
    magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
    {%- else %}
    table_index = infos[0] >> info_B_num_bits;
    bag_index = infos[0] & info_B_mask;
    {%- endif %}
    load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
        &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

    {%- if nobag %}
    magic_div_u32_run_with_mod(batch_mdiv, infos[1], batch, table_index, bag_index);
    {%- else %}
    table_index = infos[1] >> info_B_num_bits;
    bag_index = infos[1] & info_B_mask;
    {%- endif %}
    load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
        &grad_data[dword_per_row], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

    if constexpr (!weighted) {
        #pragma unroll
        for(int j = 2; j < segment_unroll; j += 2)
        {
            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[0], lane_id);
            {%- if nobag %}
            magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
            {%- else %}
            table_index = infos[j] >> info_B_num_bits;
            bag_index = infos[j] & info_B_mask;
            {%- endif %}
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id);
            {%- if nobag %}
            magic_div_u32_run_with_mod(batch_mdiv, infos[j + 1], batch, table_index, bag_index);
            {%- else %}
            table_index = infos[j + 1] >> info_B_num_bits;
            bag_index = infos[j + 1] & info_B_mask;
            {%- endif %}
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
            {%- if nobag %}
            magic_div_u32_run_with_mod(batch_mdiv, infos[j], batch, table_index, bag_index);
            {%- else %}
            table_index = infos[j] >> info_B_num_bits;
            bag_index = infos[j] & info_B_mask;
            {%- endif %}
            load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);

            accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                &grad_acc[0], &grad_data[dword_per_row], lane_id, indice_weights[j-1]);
            {%- if nobag %}
            magic_div_u32_run_with_mod(batch_mdiv, infos[j + 1], batch, table_index, bag_index);
            {%- else %}
            table_index = infos[j + 1] >> info_B_num_bits;
            bag_index = infos[j + 1] & info_B_mask;
            {%- endif %}
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

                {%- if nobag %}
                magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
                {%- else %}
                table_index = infos[0] >> info_B_num_bits;
                bag_index = infos[0] & info_B_mask;
                {%- endif %}
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

                {%- if nobag %}
                magic_div_u32_run_with_mod(batch_mdiv, infos[0], batch, table_index, bag_index);
                {%- else %}
                table_index = infos[0] >> info_B_num_bits;
                bag_index = infos[0] & info_B_mask;
                {%- endif %}
                load_row_per_warp<grad_t, embedding_dim, int32_t>::run(
                    &grad_data[0], bag_index * num_tables, p_output_grad + table_index * embedding_dim, lane_id);
                accumulate_row_per_warp<grad_t, embedding_dim, cache_t, weighted>::run(
                    &grad_acc[0], &grad_data[0], lane_id, indice_weights[0]);

                itr++;
            } while(itr < segment_length);
        }
    }

    // load the old emb weight data
    load_row_per_warp<emb_t, embedding_dim, int64_t>::run(
        &emb_data[0], emb_idx, p_emb_table, lane_id);
    optimizer_t optimizer(opt_karg);
    optimizer.template update<dword_per_row, segment_split>(grad_acc, emb_data, emb_idx);

    store_row_per_warp<emb_t, embedding_dim, emb_t>::run(&emb_data[0], p_emb_table + emb_idx * embedding_dim, lane_id);
}

