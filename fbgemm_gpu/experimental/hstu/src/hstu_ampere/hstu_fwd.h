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
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "hstu.h"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "block_info.h"
#include "kernel_traits.h"
#include "static_switch.h"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void hstu_compute_attn_1rowblock(
    const Params& params,
    const int bidb,
    const int bidh,
    int m_block) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const auto tidx = threadIdx.x;
  constexpr bool Is_causal = Kernel_traits::Is_causal;
  constexpr bool Is_target = Kernel_traits::Is_target;
  constexpr bool Is_context = Kernel_traits::Is_context;
  constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
  constexpr int  kNFunc = Kernel_traits::kNFunc;
  constexpr bool Is_local = Kernel_traits::Is_local;

  constexpr bool Has_rab = Kernel_traits::Has_rab;
  constexpr bool Paged_KV = Kernel_traits::Paged_KV;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q) {
    return;
  }

  char *smem_q = reinterpret_cast<char*>(smem_);
  char *smem_func = reinterpret_cast<char*>(smem_) + Kernel_traits::kSmemSizeQKVRabValidBlockIds;

  int *sn_valid_block_max = reinterpret_cast<int*>(smem_func);
  int *sf_min = reinterpret_cast<int*>(sn_valid_block_max) + 1;
  int *sf_max = reinterpret_cast<int*>(sf_min) + (kNFunc/2 + 1);

  const int actual_seqlen_q = binfo.actual_seqlen_q;
  const int actual_seqlen_k = binfo.actual_seqlen_k;
  // Actual target length of this sequence
  const int actual_seqlen_t = Is_target ? binfo.actual_seqlen_t : 0;
  // Actual context length of this sequence
  const int actual_seqlen_c = Is_context ? binfo.actual_seqlen_c : 0;
  // Actual history length of this sequence
  const int actual_seqlen_h =
      Is_target ? actual_seqlen_k - actual_seqlen_t : actual_seqlen_k;
  const int actual_seqlen_offset = actual_seqlen_k - actual_seqlen_q;
  // Paged KV
  const int last_page_seqlen = binfo.last_page_seqlen;
  const int page_offset = binfo.sum_s_page;

  const bool is_jump =
      Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
  const bool is_in_target =
      Is_target && (m_block + 1) * kBlockM + actual_seqlen_offset > actual_seqlen_h;
  const bool is_in_context =
      Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
  const bool is_in_mixed_context = Is_context &&
      (m_block + 1) * kBlockM > actual_seqlen_c &&
      m_block * kBlockM < actual_seqlen_c;

  const bool is_in_paged_target = is_in_target && Paged_KV;
  const int last_page_offset = is_in_paged_target ? kBlockN - last_page_seqlen : 0;

  const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
  const int target_index =
      (m_block * kBlockM - actual_seqlen_h) / params.target_group_size;
  const int n_block_paged = Paged_KV ? n_block_history : 0;
  const int n_block_target = cute::ceil_div(actual_seqlen_t, kBlockN);

  // calculate n_block_min and n_block_max
  int n_block_min = !Is_local ? 0
                              : std::max(
                                    0,
                                    (m_block * kBlockM + actual_seqlen_offset -
                                     params.window_size_left) /
                                        kBlockN);
  int n_block_max = Paged_KV ? n_block_history + n_block_target
                             : cute::ceil_div(actual_seqlen_k, kBlockN);
  if constexpr (Is_causal || Is_local) {
    int offset = (m_block + 1) * kBlockM + actual_seqlen_offset + params.window_size_right;
    if (is_in_paged_target) { offset += last_page_offset; }
    n_block_max = std::min(
        n_block_max,
        cute::ceil_div(offset, kBlockN));
  }
  if constexpr (Is_context) {
    n_block_min = (is_in_context || is_in_mixed_context) ? 0 : n_block_min;
    n_block_max = (is_in_context || is_in_mixed_context)
        ? std::max(n_block_history, n_block_max)
        : n_block_max;
  }

  // calculate n_masking_block_max and n_masking_block_min
  int n_masking_block_max = cute::ceil_div(
      std::min(actual_seqlen_k + last_page_offset, (m_block + 1) * kBlockM + actual_seqlen_offset + last_page_offset),
      kBlockN);
  int n_masking_block_min =
      (m_block * kBlockM + actual_seqlen_offset) / kBlockN;
  if constexpr (Is_target) {
    n_masking_block_min = is_jump ? (actual_seqlen_h + actual_seqlen_offset +
                                     target_index * params.target_group_size + last_page_offset) /
            kBlockN
                                  : n_masking_block_min;
  }
  if constexpr (Is_context) {
    n_masking_block_min =
        is_in_mixed_context ? n_block_min : n_masking_block_min;
    n_masking_block_max =
        is_in_mixed_context ? n_block_max : n_masking_block_max;
  }
  const int n_masking_steps = (!Is_causal || is_in_context)
      ? 0
      : n_masking_block_max - n_masking_block_min;

  // arbitrary func
  Tensor mMaxFunc = make_tensor(make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q),
        make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2 + 1>{}, actual_seqlen_q),
        make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));
  Tensor mMinFunc = make_tensor(make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q + params.func_ids_stride),
      make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2>{}, actual_seqlen_q),
      make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));

  Tensor gMaxFunc = local_tile(mMaxFunc(Int<0>{}, _, _),
                              make_shape(Int<kNFunc/2 + 1>{}, Int<kBlockM>{}),
                              make_coord(Int<0>{}, m_block));
  Tensor gMinFunc = local_tile(mMinFunc(Int<0>{}, _, _),
                              make_shape(Int<kNFunc/2>{}, Int<kBlockM>{}),
                              make_coord(Int<0>{}, m_block));


  // We exit early and write 0 to gO. This also covers the case where
  // actual_seqlen_k == 0. Otherwise we might read OOB elements from gK and gV.
  if ((Is_causal || Is_local || Is_arbitrary) && n_block_max <= n_block_min) {
    Tensor mO = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<Element*>(params.o_ptr) +
            binfo.q_offset(params.o_row_stride)),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(
        mO(_, bidh, _),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(m_block, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    clear(tOrO);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<
        /*Is_even_MN=*/false,
        /*Clear_OOB_MN=*/false,
        /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O,
        tOrO,
        tOgO,
        tOcO,
        actual_seqlen_q - m_block * kBlockM);
    return;
  }

  // We iterate over the blocks in reverse order.
  Tensor mQ = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.q_ptr) +
          binfo.q_offset(params.q_row_stride)),
      make_shape(actual_seqlen_q, params.h, params.d),
      make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(
      mQ(_, bidh, _),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_coord(m_block, 0));
  Tensor mK = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.k_ptr) +
          (Paged_KV ? binfo.kv_cache_offset(params.k_row_stride) : binfo.k_offset(params.k_row_stride))),
      make_shape(Paged_KV ? actual_seqlen_t : actual_seqlen_k, params.h_k, params.d),
      make_stride(params.k_row_stride, params.k_head_stride, _1{}));
  Tensor gK = local_tile(
      mK(_, bidh / params.h_h_k_ratio, _),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_coord(_, 0));
  Tensor mV = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.v_ptr) +
          (Paged_KV ? binfo.kv_cache_offset(params.v_row_stride) : binfo.k_offset(params.v_row_stride))),
      make_shape(Paged_KV ? actual_seqlen_t : actual_seqlen_k, params.h_k, params.d),
      make_stride(params.v_row_stride, params.v_head_stride, _1{}));
  Tensor gV = local_tile(
      mV(_, bidh / params.h_h_k_ratio, _),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_coord(_, 0));

  Tensor mKV_page = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.kv_cache_ptr)),
                                make_shape(params.total_pages, 2, params.page_size, params.h_k, params.d),
                                make_stride(params.kv_cache_kvtensor_stride, params.kv_cache_page_stride, params.kv_cache_head_stride, params.kv_cache_row_stride, _1{}));

  Tensor gK_page = local_tile(mKV_page(_, 0, _, bidh / params.h_h_k_ratio, _),
                        Shape<_1, Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0, 0));

  Tensor gV_page = local_tile(mKV_page(_, 1, _, bidh / params.h_h_k_ratio, _),
                        Shape<_1, Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0, 0));

  const int bidh_rab = (params.h_rab > 1) ? bidh : 0;
  size_t rab_qkv_not_equal_offset = bidb * params.rab_seqlen_qk_stride +
      bidh_rab * params.rab_seqlen_q_stride +
      params.seqlen_k_rounded * actual_seqlen_offset;
  auto mRab = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.rab_ptr) +
          rab_qkv_not_equal_offset),
      make_shape(actual_seqlen_q, params.seqlen_k_rounded),
      make_stride(params.rab_seqlen_k_stride, _1{}));
  auto gRab = local_tile(
      mRab(_, _),
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
      make_coord(m_block, _));

  Tensor sQ = make_tensor(
      make_smem_ptr(reinterpret_cast<Element*>(smem_q)),
      typename Kernel_traits::SmemLayoutQ{});
  // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
  Tensor sK = make_tensor(
      sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
      typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
  Tensor sRab = make_tensor(
      sV.data() + size(sV), typename Kernel_traits::SmemLayoutRab{});

  Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemSizeQKVRab)), typename Kernel_traits::SmemLayoutValidBlockIds{});
  Tensor sFunc_min = make_tensor(make_smem_ptr(reinterpret_cast<int*>(sf_min)), typename Kernel_traits::SmemLayoutMinFunc{});
  Tensor sFunc_max = make_tensor(make_smem_ptr(reinterpret_cast<int*>(sf_max)), typename Kernel_traits::SmemLayoutMaxFunc{});

  if constexpr (Is_arbitrary) {
    const int lane_id = cutlass::canonical_lane_idx();
    const int warp_id = cutlass::canonical_warp_idx_sync();
    // only 1 warp
    if (warp_id == 0)  {
      // init smme
      *sn_valid_block_max = 0;
      sFunc_min[0] = 0;
      __syncwarp();

      int f_min = INT_MAX;
      int f_max = INT_MIN;

      const int base_row = m_block * kBlockM;
      for (int i = 0; i < size<0>(gMinFunc); i++) {
        for (int j = lane_id; j < size<1>(gMinFunc); j+=32) {
          const int row = base_row + j;
          if (row < actual_seqlen_q) {
            if (f_min > gMinFunc(i, j)) {
              f_min = gMinFunc(i, j);
            }
          }
        }
        warpReduce(f_min, MinOp<int>());

        if (lane_id == 0) {
          sFunc_min[i+1] = f_min;
        }
        f_min = INT_MAX;
      }
      for (int i = 0; i < size<0>(gMaxFunc); i++) {
        for (int j = lane_id; j < size<1>(gMaxFunc); j+=32) {
          const int row = base_row + j;
          if (row < actual_seqlen_q) {
            if (f_max < gMaxFunc(i, j)) {
              f_max = gMaxFunc(i, j);
            }
          }
        }
        // warpReduceMax(f_max);
        warpReduce(f_max, MaxOp<int>());
        if (lane_id == 0) {
          sFunc_max[i] = f_max;
        }
        f_max = INT_MIN;
      }
      if (lane_id == 0) {
        for (int n_block = n_block_min; n_block < n_block_max; n_block++) {
          int b_max = (n_block + 1) * kBlockN;
          int b_min = n_block * kBlockN;
          for (int i = 0; i < (kNFunc + 1)/2; i++) {
            int f_min = sFunc_min[i];
            int f_max = sFunc_max[i];
            if (f_max <= f_min) { continue; }

            bool case1 = (f_min <= b_min && f_max > b_min);
            bool case2 = (f_min >= b_min && b_max > f_min);
            bool case3 = (f_min >= b_min && f_max < b_max);

            if (case1 || case2 || case3) {
              sValidBlockIds[*sn_valid_block_max] = n_block;
              (*sn_valid_block_max)++;
              break;
            }
          }
        }
      }
    }
    __syncthreads();
    n_block_max = *sn_valid_block_max;
    n_block_min = 0;
  }

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  typename Kernel_traits::GmemTiledCopyRab gmem_tiled_copy_Rab;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  auto gmem_thr_copy_Rab = gmem_tiled_copy_Rab.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  Tensor tKgK_page = gmem_thr_copy_QKV.partition_S(gK_page(0, _, _, _));
  Tensor tVgV_page = gmem_thr_copy_QKV.partition_S(gV_page(0, _, _, _));

  auto tQgRab = gmem_thr_copy_Rab.partition_S(gRab);
  auto tQsRab = gmem_thr_copy_Rab.partition_D(sRab);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle(_, _, _0{}));
  Tensor acc_o =
      partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  //
  // Copy Atom retiling
  //
  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  auto smem_tiled_copy_rab =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(tidx);
  auto tSsRab = smem_thr_copy_rab.partition_S(sRab);

  //
  // PREDICATES
  //
  // c = coord
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

  auto cRab = make_identity_tensor(make_shape(size<0>(sRab), size<1>(sRab)));
  auto tQcRab = gmem_thr_copy_Rab.partition_S(cRab);

  auto copy_if_g2s_rab = [&](int n_block_id, int buffer_stage) {
    auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
#pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
      if (get<0>(tQcRab(0, m, 0)) < (actual_seqlen_q - m_block * kBlockM)) {
#pragma unroll
        for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
          if (get<1>(tQcRab(0, m, k)) <
                  (actual_seqlen_k - n_block_id * kBlockN)) {
            cute::copy(
                gmem_tiled_copy_Rab, ctQgRab_view(_, m, k), tQsRab(_, m, k, buffer_stage));
          }
        }
      }
    }
  };

  auto copy_g2s_rab = [&](int n_block_id, int buffer_stage) {
    auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
#pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
        if (get<0>(tQcRab(0, m, k)) < (actual_seqlen_q - m_block * kBlockM)) {
          cute::copy(
              gmem_tiled_copy_Rab, ctQgRab_view(_, m, k), tQsRab(_, m, k, buffer_stage));
        }
      }
    }
  };

  // Prologue
  int n_valid_block_max = Is_arbitrary ? *sn_valid_block_max - 1 : 0;
  int n_block = !Is_arbitrary ? n_block_max - 1 : sValidBlockIds[n_valid_block_max];
  int buffer_stage = 0;

  if constexpr (Has_rab) {
    copy_if_g2s_rab(n_block, buffer_stage);
  }

  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  // prefill q
  flash::copy</*Is_even_MN=*/false>(
      gmem_tiled_copy_QKV,
      tQgQ,
      tQsQ,
      tQcQ,
      actual_seqlen_q - m_block * kBlockM);
  if constexpr (Kernel_traits::Is_Q_in_regs) {
    cute::cp_async_fence();
  }

  if constexpr (Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<0>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();
  }

  // prefill k
  auto tKsK_stage_view = tKsK(_, _, _, buffer_stage);
  bool is_paged_tile = (n_block < n_block_paged) && Paged_KV;
  if (!is_paged_tile) {
    flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false>(
        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - n_block_paged), tKsK_stage_view, tKVcKV,
        (Paged_KV ? actual_seqlen_t : actual_seqlen_k) - (n_block - n_block_paged) * kBlockN);
  } else {
    flash::copy</*Is_even_MN=*/true, /*Clear_OOB_MN=*/false>(
        gmem_tiled_copy_QKV, tKgK_page(_, _, _, params.page_ids[page_offset + n_block]), tKsK_stage_view, tKVcKV);
  }
  cute::cp_async_fence();

  if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<1>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
  }

  clear(acc_o);

  auto col_limit_right = [&](int row) {
    return std::min(
        actual_seqlen_k,
        row + 1 + params.window_size_right);
  };
  auto col_limit_left = [&](int row) {
    return std::max(0, row - params.window_size_left);
  };

  auto apply_mask = [&](auto& tSrS, int n_block) {
    static constexpr int Row = 0, Col = 1;
    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);
    const int base_row = m_block * kBlockM + actual_seqlen_offset;
    const int base_col = n_block * kBlockN;

    Tensor tSrS_view = make_tensor(tSrS.data(), group<1, 3>(group<0, 2>(select<1, 2, 0, 3>(flatten(tSrS.layout())))));
    Tensor tScS_view = make_tensor(tScS.data(), group<1, 3>(group<0, 2>(select<1, 2, 0, 3>(flatten(tScS.layout())))));

    #pragma unroll
    for (int mma_row = 0; mma_row < size<0>(tSrS_view); mma_row++) {
      const int block_row = int(get<Row>(tScS_view(mma_row, 0)));
      const int row = block_row + base_row;

      [[maybe_unused]] const int target_index = Is_target ? (row - actual_seqlen_h) / params.target_group_size : 0;
      [[maybe_unused]] const int target_col_limit_left = Is_target ? actual_seqlen_h + target_index * params.target_group_size : 0;

      Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
      Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
      if constexpr (Is_arbitrary) {
        // Below if code introduces BRA. For the forward pass, we don't need to apply a mask in the seq_q direction.
        // The reason for adding the 'if' statement was to guard against gMin and gMax going out of bounds.
        // However, our design ensures that the lengths of the gMin and gMax sequences are both max_seq_q, so there's no need to worry about this issue.
        /*if (row >= actual_seqlen_q) {
          #pragma unroll
          for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
            tSrS_view(mma_row, mma_col) = -INFINITY;
          }
          continue;
        }*/
        col_max(0) = gMaxFunc(0, block_row);
        #pragma unroll
        for (int j = 0; j < size<0>(gMinFunc); ++j) {
          col_min(j)   = gMinFunc(j, block_row);
          col_max(j+1) = gMaxFunc(j+1, block_row);
        }
      }

      #pragma unroll
      for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
        const int block_col = int(get<Col>(tScS_view(mma_row, mma_col)));
        int col = block_col + base_col;
        if (Paged_KV && row >= actual_seqlen_h) {
          col -= last_page_offset;
        }
        if constexpr (!Is_causal && !Is_local && !Is_arbitrary) {
          if (col >= actual_seqlen_k) {
            tSrS_view(mma_row, mma_col) = -INFINITY;
            continue;
          }
        } else {
          if constexpr (Is_context) {
            if (row < actual_seqlen_c && col < actual_seqlen_h) {
                continue;
            }
          }
          // causal mask
          if (col >= col_limit_right(row)) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
              continue;
          }
          if constexpr (Is_local) {
            if (col < col_limit_left(row)) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
              continue;
            }
          }
          if constexpr (Is_target) {
            if (row >= actual_seqlen_h && (col + (Paged_KV ? last_page_offset : 0)) >= actual_seqlen_h && col < target_col_limit_left) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
            }
          }
        }
        if constexpr (Is_arbitrary) {
          bool non_mask = false;
          non_mask = (/*col_min=*/0 <= col) && (col < col_max(0));
          if (non_mask) continue;
          #pragma unroll
          for (int j = 0; j < size<0>(gMinFunc); ++j) {
            non_mask = (col_min(j) <= col) && (col < col_max(j+1));
            if (non_mask) break;
          }
          if (!non_mask) {
            tSrS_view(mma_row, mma_col) = -INFINITY;
          }
        }
      } // col loop
    } // row loop
  };

  auto fwd_step = [&](int n_valid_block, int masking_step) {
    int n_block = !Is_arbitrary ? n_valid_block : sValidBlockIds[n_valid_block];
    // When jumps occur, it is necessary to apply a mask for the mixed situation
    const bool is_masking = masking_step < n_masking_steps ||
        (n_block + 1) * kBlockN > actual_seqlen_h;
    Tensor acc_s = partition_fragment_C(
        tiled_mma,
        Shape<Int<kBlockM>, Int<kBlockN>>{}); // (MMA=4, MMA_M, MMA_N)
    flash::cp_async_wait<0>();
    __syncthreads();

    // async load(v)
    auto tVsV_stage_view = tVsV(_, _, _, buffer_stage);
    bool is_paged_tile = (n_block < n_block_paged) && Paged_KV;
    if (masking_step > 0) {
      flash::copy</*Is_even_MN=*/true>(
          gmem_tiled_copy_QKV, is_paged_tile ? tVgV_page(_, _, _, params.page_ids[page_offset + n_block]) : tVgV(_, _, _, n_block - n_block_paged), tVsV_stage_view, tKVcKV);
    } else {
      if (!is_paged_tile) {
        flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block - n_block_paged), tVsV_stage_view, tKVcKV, (Paged_KV ? actual_seqlen_t : actual_seqlen_k) - (n_block - n_block_paged) * kBlockN);
      } else {
        flash::copy</*Is_even_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV_page(_, _, _, params.page_ids[page_offset + n_block]), tVsV_stage_view, tKVcKV);
      }
    }
    cute::cp_async_fence();

    // compute q @ k + rab
    if constexpr (Has_rab) {
      Tensor rRab = make_tensor<Element>(
          partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
      auto tSrRab_view = smem_thr_copy_rab.retile_D(rRab);
      cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, buffer_stage), tSrRab_view(_, _, _));
      flash::convert_type_safe(rRab, acc_s);
      if (n_valid_block > n_block_min) {
        int n_block_next = !Is_arbitrary ? n_block - 1 : sValidBlockIds[n_valid_block - 1];
        if (is_jump && masking_step == n_masking_steps - 1) {
          n_block_next = std::min(n_block, n_block_history) - 1;
        }
        if (n_block_next >= n_block_min) {
          copy_g2s_rab(n_block_next, buffer_stage);
        }
      }
    } else {
      clear(acc_s);
    }
    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
        acc_s,
        tSrQ,
        tSrK,
        tSsQ,
        tSsK(_, _, _, buffer_stage),
        tiled_mma,
        smem_tiled_copy_Q,
        smem_tiled_copy_K,
        smem_thr_copy_Q,
        smem_thr_copy_K);

    if (Is_arbitrary || Is_local || is_masking) {
      apply_mask(acc_s, n_block);
    }

    flash::cp_async_wait<0>();
    __syncthreads();

    if (n_valid_block > n_block_min) {
      // async load(next(k))
      int n_block_next = !Is_arbitrary ? n_block - 1 : sValidBlockIds[n_valid_block - 1];
      if (is_jump && masking_step == n_masking_steps - 1) {
        n_block_next = std::min(n_block, n_block_history) - 1;
      }
      bool is_paged_tile = (n_block_next < n_block_paged) && Paged_KV;
      auto tKsK_stage_view_next = tKsK(_, _, _, buffer_stage);
      if (n_block_next >= n_block_min) {
        flash::copy</*Is_even_MN=*/true>(
            gmem_tiled_copy_QKV, is_paged_tile ? tKgK_page(_, _, _, params.page_ids[page_offset + n_block_next]) : tKgK(_, _, _, n_block_next - n_block_paged), tKsK_stage_view_next, tKVcKV);
        cute::cp_async_fence();
      }
    }
    for (int i = 0; i < size(acc_s); ++i) {
      acc_s(i) *= params.alpha;
    }
    fast_silu(acc_s);

    // Convert acc_s from fp32 to fp16/bf16
    Tensor rP = make_tensor_like<Element>(acc_s);
    flash::convert_type_safe(acc_s, rP);

    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2) if
    // using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
    // compute qk @ v
    flash::gemm_rs(
        acc_o,
        tOrP,
        tOrVt,
        tOsVt(_, _, _, buffer_stage),
        tiled_mma,
        smem_tiled_copy_V,
        smem_thr_copy_V);
  };

  for (int n_block = n_block_max - 1, masking_step = 0; n_block >= n_block_min; ++masking_step, --n_block) {
    fwd_step(n_block, masking_step);
    if (is_jump && masking_step == n_masking_steps - 1) {
      n_block = std::min(n_block, n_block_history);
    }
  }

  // scale acc_o
  for (int i = 0; i < size(acc_o); ++i) {
    acc_o(i) /= params.seqlen_q;
  }

  // Epilogue
  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = make_tensor_like<Element>(acc_o);
  flash::convert_type_safe(acc_o, rO);
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

  if constexpr (Kernel_traits::Share_Q_K_smem) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor mO = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.o_ptr) +
          binfo.q_offset(params.o_row_stride)),
      make_shape(actual_seqlen_q, params.h, params.d),
      make_stride(params.o_row_stride, params.o_head_stride, _1{}));
  Tensor gO = local_tile(
      mO(_, bidh, _),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_coord(m_block, 0));

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
  // Repeat the partitioning with identity layouts
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy</*Is_even_MN=*/false,
              /*Clear_OOB_MN=*/false,
              /*Clear_OOB_K=*/false>(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, typename Params>
__global__ void hstu_fwd_kernel(Params params) {
  int m_block = gridDim.x - blockIdx.x - 1;
  int bidh = blockIdx.y;
  int bidb = blockIdx.z;

  hstu_compute_attn_1rowblock<Kernel_traits>(params, bidb, bidh, m_block);
}

} // namespace flash

template <
    typename elem_type,
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_causal,
    bool Is_target,
    bool Is_context,
    bool Is_local,
    bool Is_arbitrary,
    int kNFunc,
    bool Has_rab,
    bool Paged_KV,
    bool Is_Q_in_regs = false,
    bool Share_Q_K_smem = false>
void run_hstu_fwd_impl(Hstu_fwd_params& params, cudaStream_t stream) {
  using Kernel_traits = Hstu_fwd_kernel_traits<
      kHeadDim,
      kBlockM,
      kBlockN,
      kNWarps,
      Is_causal,
      Is_target,
      Is_context,
      Is_local,
      Is_arbitrary,
      kNFunc,
      Has_rab,
      Paged_KV,
      Is_Q_in_regs,
      Share_Q_K_smem,
      elem_type>;

  size_t smem_size = Kernel_traits::kSmemSize;
  const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;
  dim3 grid = dim3(num_m_block, params.h, params.b);
  auto kernel = &flash::hstu_fwd_kernel<Kernel_traits, Hstu_fwd_params>;

  if (smem_size >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <
    int Arch,
    typename elem_type,
    int kHeadDim,
    bool Has_rab,
    bool Is_local,
    bool Is_causal,
    bool Is_context,
    bool Is_target,
    bool Is_arbitrary,
    int kNFunc>
void run_hstu_fwd_8x(Hstu_fwd_params& params, cudaStream_t stream) {
  INT_SWITCH(params.page_size, Page_Size, [&] {
    static constexpr bool Paged_KV = Page_Size > 0;
    static constexpr auto tile_size =
        flash::get_tile_size_fwd<Arch, kHeadDim, Has_rab>();
    static constexpr int kBlockM = std::get<0>(tile_size);
    static constexpr int kBlockN = Paged_KV? Page_Size : std::get<1>(tile_size);
    static constexpr int kNWarps = std::get<2>(tile_size);
    static constexpr bool Is_Q_in_regs = kHeadDim <= 128;
    static constexpr bool Share_Q_K_smem = kHeadDim <= 128;
    run_hstu_fwd_impl<
        elem_type,
        kHeadDim,
        kBlockM,
        kBlockN,
        kNWarps,
        Is_causal,
        Is_target,
        Is_context,
        Is_local,
        Is_arbitrary,
        kNFunc,
        Has_rab,
        Paged_KV,
        Is_Q_in_regs,
        Share_Q_K_smem>(params, stream);
  });
}
