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
#ifdef __HIP_PLATFORM_HCC__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "split_tbe_common_hip.h"
#include "../codegen/embedding_forward_template_helpers_hip.cuh"

template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    int32_t embedding_dim,
    int32_t bag_prefetch,
    int32_t bag_unroll,
    bool    weighted>
__device__ void split_tbe_forward_hip_kernel(
    output_t * p_output,
    const emb_t * p_emb_table,
    const index_t * p_indices,
    const index_t * p_offsets,
    const int32_t * D_offsets,
    const int64_t * weights_offsets,
    const int64_t pooling_mode,
    uint32_t batch,
    uint32_t num_rows,
    uint32_t num_tables,
    const float * p_indice_weights = nullptr)
{ 
    const auto emb_dim = D_offsets[blockIdx.y + 1] - D_offsets[blockIdx.y];
    constexpr uint32_t dword_output_per_row = (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    // constexpr uint32_t input_data_per_dword = 4 / sizeof(emb_t);    // TODO: larger than 4 byte
    // constexpr uint32_t dword_input_per_row = (dword_output_per_row + input_data_per_dword - 1) / input_data_per_dword;
    // constexpr uint32_t dword_input_per_row_rounded = dword_input_per_row == 1 ? dword_input_per_row
    //                                  : ((dword_input_per_row + 1) / 2 * 2); // round to 2x
    constexpr uint32_t length_mask = ~(bag_unroll - 1);

    static_assert(bag_prefetch < bag_unroll, "");

    float accumulator[dword_output_per_row];
    index_t indices[bag_unroll];
    float indice_weights[bag_unroll];

    emb_t emb_data[dword_output_per_row * bag_prefetch];

    int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / AMDGCN_WAVE_SIZE);
    int bag_id = (blockIdx.x << 2) | wave_id;
    if(bag_id >= batch)
        return ;
    int lane_id = threadIdx.x & (AMDGCN_WAVE_SIZE - 1);

    p_offsets += blockIdx.y * batch + bag_id;
    index_t indices_start = p_offsets[0];
    index_t indices_end = p_offsets[1];

    p_emb_table += weights_offsets[blockIdx.y];
    p_output += D_offsets[blockIdx.y] + bag_id * D_offsets[num_tables];

    #pragma unroll
    for(int i=0; i < dword_output_per_row; i++)
    {
        accumulator[i] = .0f;
    }
    p_indices += indices_start;

    if constexpr (weighted) {
        p_indice_weights += indices_start;
    }

    int32_t length = indices_end - indices_start;
    int32_t length_mod = length & length_mask;

    int itr = 0;
    if(length_mod == 0)
        goto L_end;

    if constexpr (!weighted) {
        #pragma unroll
        for(int i=0; i < bag_unroll; i++){
            indices[i] = p_indices[i];
        }
    } else {
        #pragma unroll
        for(int i=0; i < bag_unroll; i++){
            indices[i] = p_indices[i];
	    indice_weights[i] = p_indice_weights[i];
        }
    }

    itr += bag_unroll;
    p_indices += bag_unroll;

    if constexpr (weighted) {
        p_indice_weights += bag_unroll;
    }

    // LOOP
    for( ; itr<length_mod; itr += bag_unroll){
        load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id, emb_dim);
        load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[1], p_emb_table, lane_id, emb_dim);

	if constexpr (!weighted) {
            #pragma unroll
            for(int j = 2 ; j < bag_unroll; j += 2){
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id, emb_dim);

                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id, emb_dim);
            }
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0], &emb_data[0], lane_id);
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);

            #pragma unroll
            for(int i=0; i < bag_unroll; i++){
                indices[i] = p_indices[i];
            }
            p_indices += bag_unroll;

        } else {    // row weighted
            #pragma unroll
            for(int j = 2 ; j < bag_unroll; j += 2){
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[j-2]);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id, emb_dim);

                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[j-1]);
                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id, emb_dim);
            }
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0], &emb_data[0], lane_id, indice_weights[bag_unroll-2]);
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[bag_unroll-1]);

            #pragma unroll
            for(int i=0; i < bag_unroll; i++){
                indices[i] = p_indices[i];
                indice_weights[i] = p_indice_weights[i];
            }
            p_indices += bag_unroll;
            p_indice_weights += bag_unroll;
        }
    }
    // LAST
    load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id, emb_dim);
    load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[1], p_emb_table, lane_id, emb_dim);

    if constexpr (!weighted) {
        #pragma unroll
        for(int j = 2 ; j < bag_unroll; j += 2){
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id, emb_dim);

            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id, emb_dim);
        }
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id);

    } else {    // row weighted
        #pragma unroll
        for(int j = 2 ; j < bag_unroll; j += 2){
            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[j-2]);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[j], p_emb_table, lane_id, emb_dim);

            accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[j-1]);
            load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[dword_output_per_row], indices[j+1], p_emb_table, lane_id, emb_dim);
        }
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[bag_unroll-2]);
        accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[dword_output_per_row], lane_id, indice_weights[bag_unroll-1]);
    }

L_end:
    if(length & (bag_unroll - 1)){
        if constexpr (!weighted) {
            // last, load one by one
            do {
                indices[0] = p_indices[0];
                p_indices++;

                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id, emb_dim);
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id);

                itr++;
            } while (itr < length);
        } else {    // row weighted
            do {
                indices[0] = p_indices[0];
                indice_weights[0] = p_indice_weights[0];
                p_indices++;
                p_indice_weights++;

                load_row_per_warp<emb_t, embedding_dim, index_t>::run(&emb_data[0], indices[0], p_emb_table, lane_id, emb_dim);
                accumulate_row_per_warp<emb_t, embedding_dim, output_t, weighted>::run(&accumulator[0],  &emb_data[0], lane_id, indice_weights[0]);

                itr++;
            } while(itr < length);
        }
    }

    if (static_cast<fbgemm_gpu::PoolingMode>(pooling_mode) == fbgemm_gpu::PoolingMode::MEAN && length != 0){
#pragma unroll
        for (int i = 0; i < dword_output_per_row; i++){
            accumulator[i] *= 1.0f / length;
        }
    }

    // store out
    store_row_per_warp<emb_t, embedding_dim, output_t>::run(&accumulator[0], p_output, lane_id, emb_dim);
}


#define SPLIT_TBE_FWD_KERNEL(emb_prec, emb_type, embedding_dim, bag_prefetch, bag_unroll) \
    extern "C" __global__ void split_tbe_fwd_unweighted_hip_kernel_ ## emb_prec ## _e ## embedding_dim ( \
            float * p_output,              \
            const emb_type * p_emb_table,  \
            const int64_t * p_indices,     \
            const int64_t * p_offsets,     \
            const int32_t * D_offsets,     \
            const int64_t * weights_offsets, \
            const int64_t pooling_mode,    \
            uint32_t batch,                \
            uint32_t num_rows,             \
            uint32_t num_tables)           \
    {                                      \
        split_tbe_forward_hip_kernel<emb_type, float, float, int64_t, embedding_dim, bag_prefetch, bag_unroll, false> \
                (p_output, p_emb_table, p_indices, p_offsets, D_offsets, weights_offsets, pooling_mode, batch, num_rows, num_tables); \
    } \
    \
    extern "C" __global__ void split_tbe_fwd_weighted_hip_kernel_ ## emb_prec ## _e ## embedding_dim ( \
            float * p_output,              \
            const emb_type * p_emb_table,  \
            const int64_t * p_indices,     \
            const int64_t * p_offsets,     \
            const int32_t * D_offsets,     \
            const int64_t * weights_offsets, \
            const int64_t pooling_mode,    \
            const float * p_indice_weights,\
            uint32_t batch,                \
            uint32_t num_rows,             \
            uint32_t num_tables)           \
    {                                      \
        split_tbe_forward_hip_kernel<emb_type, float, float, int64_t, embedding_dim, bag_prefetch, bag_unroll, true> \
                (p_output, p_emb_table, p_indices, p_offsets, D_offsets, weights_offsets, pooling_mode, batch, num_rows, num_tables, p_indice_weights); \
    }


SPLIT_TBE_FWD_KERNEL(fp16,  half,  64, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 128, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 192, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 256, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 384, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 512, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 640, 2, 8)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 768, 2, 8)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 896, 2, 4)
SPLIT_TBE_FWD_KERNEL(fp16,  half, 1024, 2, 4)

SPLIT_TBE_FWD_KERNEL(fp32, float,  64, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 128, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 192, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 256, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 384, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 512, 2, 16)
SPLIT_TBE_FWD_KERNEL(fp32, float, 640, 2, 8)
SPLIT_TBE_FWD_KERNEL(fp32, float, 768, 2, 8)
SPLIT_TBE_FWD_KERNEL(fp32, float, 896, 2, 4)
SPLIT_TBE_FWD_KERNEL(fp32, float, 1024, 2, 4)

#endif
