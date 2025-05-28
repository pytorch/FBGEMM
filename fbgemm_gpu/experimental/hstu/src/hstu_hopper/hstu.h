/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <vector>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Hstu_fwd_params : public Qkv_params {
    // The O matrix (output).
    void * __restrict__ o_ptr;

    // The stride between rows of O.
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded, total_q, total_k;
    int seqlen_i;
    float alpha;
    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    int * __restrict__ num_contexts;
    int * __restrict__ num_targets;

    void* __restrict__ rab_ptr;
    index_t rab_batch_stride;
    index_t rab_row_stride;
    index_t rab_head_stride;
    // The number of rab heads.
    int h_rab;

    // Local window size
    int window_size_left, window_size_right;
    int target_group_size;

    bool has_rab;
    bool is_bf16;
    bool is_e4m3;
    bool is_causal;
    bool is_local;
    bool is_target;
    bool is_delta_q;
    bool is_context;

    int * __restrict__ tile_count_semaphore;
    float * __restrict__ descale_q_ptr;
    float * __restrict__ descale_k_ptr;
    float * __restrict__ descale_v_ptr;

    int arch;
    int num_sm;

    bool is_balance_fwd;
    bool is_balance_bwd;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Hstu_bwd_params : public Hstu_fwd_params {

    // The dO and dQKVRab matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;
    void *__restrict__ drab_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;
    index_t drab_row_stride;
    index_t drab_head_stride;
    index_t drab_batch_stride;

    int *__restrict__ dq_semaphore;

    bool deterministic;
    bool has_drab;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int Arch, typename T, int Headdim, bool Has_rab, bool Is_local,
         bool Is_causal, bool Is_context, bool Is_target, bool Is_delta_q>
void run_hstu_fwd_(Hstu_fwd_params &params, cudaStream_t stream);
template<int Arch, typename T, int Headdim, bool Has_rab, bool Has_drab, bool Is_local,
         bool Is_causal, bool Is_context, bool Is_target, bool Is_delta_q>
void run_hstu_bwd_(Hstu_bwd_params &params, cudaStream_t stream);
