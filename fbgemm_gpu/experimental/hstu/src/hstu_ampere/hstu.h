/*
 * Copyright (c) 2023, Tri Dao.
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

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh> // For at::cuda::philox::unpack

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Hstu_params {
  using index_t = int64_t;
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;

  // The number of heads.
  int h, h_k, h_rab;
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k
  // could be different from nheads (query).
  int h_h_k_ratio; // precompute h / h_k,

  bool is_balance_fwd;
  bool is_balance_bwd;
  int arch;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Hstu_fwd_params : public Hstu_params {
  // The O matrix (output).
  void* __restrict__ o_ptr;

  // The stride between rows of O.
  index_t o_row_stride;
  index_t o_head_stride;

  // array of length b+1 holding starting offset of each sequence.
  int* __restrict__ cu_seqlens_q;
  int* __restrict__ cu_seqlens_k;
  int* __restrict__ seqused_q;
  int* __restrict__ seqused_k;

  int* __restrict__ num_targets;
  int* __restrict__ num_contexts;

  void* __restrict__ func_ptr; // batch_size x heads x n_func x seqlen_q
  index_t func_ids_stride;
  index_t func_head_stride;
  int func_batch;
  int n_func;

  void* __restrict__ rab_ptr;
  index_t rab_seqlen_qk_stride;
  index_t rab_seqlen_q_stride;
  index_t rab_seqlen_k_stride;

  // for paged kv cache
  void* __restrict__ kv_cache_ptr;
  index_t kv_cache_row_stride;
  index_t kv_cache_head_stride;
  index_t kv_cache_page_stride;
  index_t kv_cache_kvtensor_stride;
  int page_size;
  int total_pages;

  int*  __restrict__ page_offsets;
  int*  __restrict__ page_ids;
  int*  __restrict__ last_page_lens;

  // The dimensions.
  int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded;
  int scaling_seqlen;
  float alpha;

  int target_group_size;

  int window_size_left;
  int window_size_right;

  bool has_rab;
  bool is_bf16;
  bool is_causal;
  bool is_local;
  bool is_target;
  bool is_context;
  bool is_paged_kv;
  bool is_arbitrary_mask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Hstu_bwd_params : public Hstu_fwd_params {
  // The dO and dQKV matrices.
  void* __restrict__ dRab_ptr;
  void* __restrict__ do_ptr;
  void* __restrict__ dq_ptr;
  void* __restrict__ dk_ptr;
  void* __restrict__ dv_ptr;

  // The stride between rows of the dO, dQ, dK and dV matrices.
  // TD [2022-04-16]: We're using 32-bit indexing to save registers.
  // The code probably won't work for arrays larger than 2GB.
  index_t do_batch_stride;
  index_t dq_batch_stride;
  index_t dk_batch_stride;
  index_t dv_batch_stride;

  index_t do_row_stride;
  index_t do_head_stride;

  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;

  void* __restrict__ dq_accum_ptr;
  bool deterministic;
  index_t dq_accum_split_stride;
  index_t dq_accum_row_stride;
  index_t dq_accum_head_stride;

  index_t drab_seqlen_qk_stride;
  index_t drab_seqlen_q_stride;
  index_t drab_seqlen_k_stride;

  bool has_drab;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Arch,
    typename T,
    int Headdim,
    bool Has_rab,
    bool Is_local,
    bool Is_causal,
    bool Is_context,
    bool Is_target,
    bool Is_arbitrary,
    int kNFunc>
void run_hstu_fwd_8x(Hstu_fwd_params& params, cudaStream_t stream);

template <
    int Arch, 
    typename T,
    int Headdim,
    bool Has_rab,
    bool Has_drab,
    bool Is_local,
    bool Is_causal,
    bool Is_context,
    bool Is_target,
    bool Is_arbitrary,
    int kNFunc,
    bool Is_deterministic>
void run_hstu_bwd_80(Hstu_bwd_params& params, cudaStream_t stream);
