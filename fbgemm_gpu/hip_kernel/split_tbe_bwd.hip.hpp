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
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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
        split_tbe_bwd_unweighted_hip_kernel_##optimizer##_w##weight_decay_mode##_s##segment_split##_##emb_prec##_##grad_prec##_e##embedding_dim(   \
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
            optimizer##_kernel_arg_t opt_karg);                                                                                      \
                                                                                                                                     \
    extern "C" __global__ void                                                                                                       \
        split_tbe_bwd_weighted_hip_kernel_##optimizer##_w##weight_decay_mode##_s##segment_split##_##emb_prec##_##grad_prec##_e##embedding_dim(   \
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
            optimizer##_kernel_arg_t opt_karg);

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
SPLIT_TBE_BWD_KERNEL(rowwise_adagrad, 0, 64);
SPLIT_TBE_BWD_KERNEL(rowwise_adagrad, 0, 128);
SPLIT_TBE_BWD_KERNEL(rowwise_adagrad, 0, 192);
SPLIT_TBE_BWD_KERNEL(rowwise_adagrad, 0, 256);