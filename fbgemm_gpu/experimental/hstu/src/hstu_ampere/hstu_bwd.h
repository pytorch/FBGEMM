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
#include <cuda_fp16.h>

#include <cub/cub.cuh>

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

template <typename Kernel_traits>
inline __device__ void hstu_compute_dq_dk_dv_1colblock(
    const Hstu_bwd_params& params,
    const int bidb,
    const int bidh,
    const int n_block) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  constexpr bool Is_causal = Kernel_traits::Is_causal;
  constexpr bool Is_local = Kernel_traits::Is_local;
  constexpr bool Has_rab = Kernel_traits::Has_rab;
  constexpr bool Has_drab = Kernel_traits::Has_drab;
  constexpr bool Is_context = Kernel_traits::Is_context;
  constexpr bool Is_target = Kernel_traits::Is_target;
  constexpr bool Is_arbitrary = Kernel_traits::Is_arbitrary;
  constexpr int  kNFunc = Kernel_traits::kNFunc;
  constexpr bool Is_deterministic = Kernel_traits::Is_deterministic;
  constexpr bool Rab_one_head = Kernel_traits::Rab_one_head;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const auto tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int MMA_N_SdP = kBlockN /
      decltype(typename Kernel_traits::TiledMmaSdP{}
                   .template tile_size_mnk<1>())::value;
  constexpr int kStages = Kernel_traits::kStages;
  constexpr bool Is_V_in_regs = Kernel_traits::Is_V_in_regs;

  const HstuBlockInfo<Kernel_traits, Hstu_bwd_params> binfo(params, bidb);
  if (n_block * kBlockN >= binfo.actual_seqlen_k)
    return;

  const int actual_seqlen_q = binfo.actual_seqlen_q;
  const int actual_seqlen_k = binfo.actual_seqlen_k;
  // Actual context length of this sequence
  const int actual_seqlen_c = Is_context ? binfo.actual_seqlen_c : 0;
  // Actual history length of this sequence
  const int actual_seqlen_h =
      Is_target ? actual_seqlen_k - binfo.actual_seqlen_t : actual_seqlen_k;
  const int m_block_context =
      Is_context ? cute::ceil_div(actual_seqlen_c, kBlockM) : 0;

  const bool is_jump = Is_target && n_block * kBlockN >= actual_seqlen_h;
  const bool is_in_context =
      Is_context && actual_seqlen_c > 0 && n_block * kBlockN <= actual_seqlen_h;
  const bool is_in_mixed_context = Is_context && Is_target &&
      n_block * kBlockN < actual_seqlen_h &&
      (n_block + 1) * kBlockN > actual_seqlen_h;

  const int actual_seqlen_offset = actual_seqlen_k - actual_seqlen_q;
  const int target_index = cute::ceil_div(
      (n_block + 1) * kBlockN - actual_seqlen_h, params.target_group_size);

  // calculate m_masking_block_min and m_masking_block_max
  int m_masking_block_max = cute::ceil_div(
      std::max(actual_seqlen_c, (n_block + 1) * kBlockN - actual_seqlen_offset),
      kBlockM);
  if constexpr (Is_target) {
    m_masking_block_max = std::max(
        m_masking_block_max,
        cute::ceil_div(
            actual_seqlen_h - actual_seqlen_offset +
                target_index * params.target_group_size,
            kBlockM));
    m_masking_block_max =
        std::min(m_masking_block_max, cute::ceil_div(actual_seqlen_q, kBlockM));
    const bool is_mixed_target = (n_block + 1) * kBlockN > actual_seqlen_h &&
        n_block * kBlockN < actual_seqlen_h;
    if (is_mixed_target) {
      m_masking_block_max = cute::ceil_div(actual_seqlen_q, kBlockM);
    }
  }
  const int m_masking_block_min =
      std::max(0, n_block * kBlockN - actual_seqlen_offset) / kBlockM;
  const int n_masking_steps =
      Is_causal ? m_masking_block_max - m_masking_block_min : 1;

  // calculate m_block_min and m_block_max
  int m_block_min = (!Is_causal && !Is_local)
      ? 0
      : std::max(
            0,
            (n_block * kBlockN - actual_seqlen_offset -
             params.window_size_right) /
                kBlockM);
  int m_block_max = cute::ceil_div(actual_seqlen_q, kBlockM);
  if constexpr (Is_local) {
    m_block_max = std::min(
        m_block_max,
        cute::ceil_div(
            (n_block + 1) * kBlockN - actual_seqlen_offset +
                params.window_size_left,
            kBlockM));
  }
  if constexpr (Is_target) {
    m_block_max = is_jump ? m_masking_block_max : m_block_max;
  }

  const int m_block_min_casual = m_block_min;
  if constexpr (Is_context) {
    m_block_min = is_in_context ? 0 : m_block_min;
  }
  if constexpr (Is_arbitrary) {
    m_block_min = 0;
    m_block_max = cute::ceil_div(actual_seqlen_q, kBlockM);
  }

  // arbitrary func
  Tensor mMaxFunc = make_tensor(make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q),
      make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2 + 1>{}, actual_seqlen_q),
        make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));
  Tensor mMinFunc = make_tensor(make_gmem_ptr(reinterpret_cast<int*>(params.func_ptr) + binfo.sum_s_q + params.func_ids_stride),
      make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2>{}, actual_seqlen_q),
      make_stride(params.func_head_stride, 2 * params.func_ids_stride, _1{}));

  Tensor gMaxFunc = local_tile(mMaxFunc(Int<0>{}, _, _),
                             make_shape(Int<kNFunc/2 + 1>{}, Int<kBlockM>{}), make_coord(Int<0>{}, _));
  Tensor gMinFunc = local_tile(mMinFunc(Int<0>{}, _, _),
                             make_shape(Int<kNFunc/2>{}, Int<kBlockM>{}), make_coord(Int<0>{}, _));

  Tensor mQ = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.q_ptr) +
          binfo.q_offset(params.q_row_stride)),
      make_shape(actual_seqlen_q, params.h, params.d),
      make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(
      mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, 0));

  Tensor mK = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.k_ptr) +
          binfo.k_offset(params.k_row_stride)),
      make_shape(actual_seqlen_k, params.h, params.d),
      make_stride(params.k_row_stride, params.k_head_stride, _1{}));
  Tensor gK = local_tile(
      mK(_, bidh / params.h_h_k_ratio, _),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_coord(n_block, 0));

  Tensor mV = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.v_ptr) +
          binfo.k_offset(params.v_row_stride)),
      make_shape(actual_seqlen_k, params.h, params.d),
      make_stride(params.v_row_stride, params.v_head_stride, _1{}));
  Tensor gV = local_tile(
      mV(_, bidh / params.h_h_k_ratio, _),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_coord(n_block, 0));

  Tensor mdO = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.do_ptr) +
          binfo.q_offset(params.do_row_stride)),
      make_shape(actual_seqlen_q, params.h, params.d),
      make_stride(params.do_row_stride, params.do_head_stride, _1{}));
  Tensor gdO = local_tile(
      mdO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(_, 0));

  const index_t row_offset_dq_accum1 = binfo.q_offset(params.dq_accum_row_stride) +
      128 * bidb * params.dq_accum_row_stride +
      // If deterministic, each thread block will do atomicAdd to a different
      // dQ_accum buffer.
      +(!Is_deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
  Tensor mdQaccum = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<ElementAccum*>(params.dq_accum_ptr) +
          row_offset_dq_accum1),
      make_shape(actual_seqlen_q, params.h, params.d),
      make_stride(params.dq_accum_row_stride, params.dq_accum_head_stride, _1{}));
  Tensor gdQaccum = local_tile(
      mdQaccum(_, bidh, _),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_coord(_, 0));

  const int bidh_rab = !Rab_one_head ? bidh : 0;
  size_t rab_qkv_not_equal_offset = bidb * params.rab_seqlen_qk_stride +
      bidh_rab * params.rab_seqlen_q_stride +
      params.seqlen_k_rounded * actual_seqlen_offset;
  auto mRab = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.rab_ptr) +
          rab_qkv_not_equal_offset),
      make_shape(actual_seqlen_q, actual_seqlen_k),
      make_stride(params.rab_seqlen_k_stride, _1{}));
  auto gRab = local_tile(
      mRab(_, _),
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
      make_coord(_, n_block));

  size_t dRab_qkv_not_equal_offset = bidb * params.drab_seqlen_qk_stride +
      bidh_rab * params.drab_seqlen_q_stride +
      params.seqlen_k_rounded * actual_seqlen_offset;
  auto mdRab = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<Element*>(params.dRab_ptr) +
          dRab_qkv_not_equal_offset),
      make_shape(actual_seqlen_q, actual_seqlen_k),
      make_stride(params.drab_seqlen_k_stride, _1{}));
  auto gdRab = local_tile(
      mdRab(_, _),
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
      make_coord(_, n_block));

  Tensor sQ = make_tensor(
      make_smem_ptr(reinterpret_cast<Element*>(smem_)),
      typename Kernel_traits::SmemLayoutQ{});
  Tensor sQt =
      make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQtransposed{});
  Tensor sQtNoSwizzle = make_tensor(
      sQ.data(), typename Kernel_traits::SmemLayoutQtransposedNoSwizzle{});
  // Double buffer for sQ
  Tensor sdO = make_tensor(
      sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutdO{});
  Tensor sdOt = make_tensor(
      sdO.data(), typename Kernel_traits::SmemLayoutdOtransposed{});
  Tensor sdOtNoSwizzle = make_tensor(
      sdO.data(), typename Kernel_traits::SmemLayoutdOtransposedNoSwizzle{});
  Tensor sK = make_tensor(
      sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
  Tensor sKt =
      make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
  Tensor sKtNoSwizzle = make_tensor(
      sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sdS = make_tensor(
      !Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
      typename Kernel_traits::SmemLayoutPdS{});
  Tensor sdSt = make_tensor(
      sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
  Tensor sdStNoSwizzle = make_tensor(
      sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
  Tensor sP = make_tensor(
      sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
  Tensor sPt =
      make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
  Tensor sPtNoSwizzle = make_tensor(
      sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
  Tensor sRab = make_tensor(
      sP.data() + size(sP), typename Kernel_traits::SmemLayoutRab{});

  Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemSize1colblock)), typename Kernel_traits::SmemLayoutValidBlockIds{});

  int *sm_valid_block_max = reinterpret_cast<int*>(smem_ + Kernel_traits::kSmemSize1colblock_validblockids);

  if constexpr (Is_arbitrary) {
    const int lane_id = cutlass::canonical_lane_idx();
    const int warp_id = cutlass::canonical_warp_idx_sync();
    // only 1 warp
    if (warp_id == 0)  {
      *sm_valid_block_max = 0;
      int b_min = n_block * kBlockN;
      int b_max = (n_block + 1) * kBlockN;
      #pragma unroll
      for (int m_block = m_block_min; m_block < m_block_max; ++m_block) {
        int base_row = m_block * kBlockM;
        int f_min = 0;
        int f_max = INT_MIN;
        for (int j = lane_id; j < size<1>(gMaxFunc); j+=32) {
          const int row = base_row + j;
          if (row < actual_seqlen_q) {
            if (f_max < gMaxFunc(0, j, m_block)) {
              f_max = gMaxFunc(0, j, m_block);
            }
          }
        }
        warpReduce(f_max, MaxOp<int>());
        bool case1 = (f_min <= b_min && f_max > b_min);
        bool case2 = (f_min >= b_min && b_max > f_min);
        bool case3 = (f_min >= b_min && f_max < b_max);
        bool is_valid = __shfl_sync(0xffffffff, (case1 || case2 || case3) && (f_max > f_min), 0);

        if (is_valid) {
          sValidBlockIds[*sm_valid_block_max] = m_block;
          if (lane_id == 0) {
            (*sm_valid_block_max)++;
          }
          continue;
        }

        #pragma unroll
        for (int i = 0; i < size<0>(gMinFunc); i++) {
          f_min = INT_MAX;
          f_max = INT_MIN;
          #pragma unroll
          for (int j = lane_id; j < size<1>(gMinFunc); j+=32) {
            const int row = base_row + j;
            if (row < actual_seqlen_q) {
              if (f_min > gMinFunc(i, j, m_block)) {
                f_min = gMinFunc(i, j, m_block);
              }
              if (f_max < gMaxFunc(i+1, j, m_block)) {
                f_max = gMaxFunc(i+1, j, m_block);
              }
            }
          }
          warpReduce(f_min, MinOp<int>());
          warpReduce(f_max, MaxOp<int>());
          bool case1 = (f_min <= b_min && f_max > b_min);
          bool case2 = (f_min >= b_min && b_max > f_min);
          bool case3 = (f_min >= b_min && f_max < b_max);
          bool is_valid = __shfl_sync(0xffffffff, (case1 || case2 || case3) && (f_max > f_min), 0);
          if (is_valid) {
            sValidBlockIds[*sm_valid_block_max] = m_block;
            if (lane_id == 0) {
              (*sm_valid_block_max)++;
            }
            break;
          }
        }
      }
    }
    __syncthreads();
    m_block_max = *sm_valid_block_max;
    m_block_min = 0;
  }

  // Copy tiling
  typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
  auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
  Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
  Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);

  typename Kernel_traits::GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
  auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);
  Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
  auto tQgRab = gmem_thr_copy_QKV.partition_S(gRab);
  auto tQsRab = gmem_thr_copy_QKV.partition_D(sRab);

  // MMA tiling
  typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
  auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ(_, _, 0));
  Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
  Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
  Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

  typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
  auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
  Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle);
  Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle(_, _, 0));
  Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);
  Tensor tdVrdOt = thr_mma_dkv.partition_fragment_B(sdOtNoSwizzle);

  Tensor acc_dk =
      partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
  Tensor acc_dv =
      partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

  typename Kernel_traits::TiledMmadQ tiled_mma_dq;
  auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
  Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);
  Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

  //
  // Copy Atom retiling
  //
  auto smem_tiled_copy_QdO =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
  auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
  Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

  auto smem_tiled_copy_KV = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
  auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
  Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

  // Partition sP and sdS to match the accumulator partitioning
  // This has to be tiled_mma_sdp, not tiled_mma_dkv
  auto smem_tiled_copy_PdS = make_tiled_copy_C(
      typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
  auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
  Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);
  Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);

  auto smem_tiled_copy_PdSt = make_tiled_copy_A(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
  auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
  Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
  Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

  auto smem_tiled_copy_QdOt = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
  auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
  Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
  Tensor tdVsdO = smem_thr_copy_QdOt.partition_S(sdO);
  Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

  auto smem_tiled_copy_dS =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
  auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
  Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

  auto smem_tiled_copy_Kt = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
  auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
  Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

  auto smem_tiled_copy_rab = make_tiled_copy_C(
      typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
  auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(tidx);
  auto tSsRab = smem_thr_copy_rab.partition_S(sRab);

  //
  // PREDICATES
  //
  // c = coord
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
  Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

  Tensor cRab = make_identity_tensor(make_shape(size<0>(sRab), size<1>(sRab)));
  Tensor tQcRab = gmem_thr_copy_QKV.partition_D(cRab);

  int m_block = Is_arbitrary ? sValidBlockIds[m_block_max - 1] : m_block_max - 1;

  int buffer_stage = 0;

  auto copy_if_g2s_rab = [&](int m_block_id) {
    auto ctQgRab_view = tQgRab(_, _, _, m_block_id);
#pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
      if (get<0>(tQcRab(0, m, 0)) < actual_seqlen_q - m_block * kBlockM) {
#pragma unroll
        for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
          if (get<1>(tQcRab(0, m, k)) < (actual_seqlen_k - n_block * kBlockN)) {
            cute::copy(
                gmem_tiled_copy_QKV,
                ctQgRab_view(_, m, k),
                tQsRab(_, m, k));
          } else {
            cute::clear(tQsRab(_, m, k));
          }
        }
      } else {
        cute::clear(tQsRab(_, m, _));
      }
    }
  };
  auto copy_g2s_rab = [&](int m_block_id) {
    auto ctQgRab_view = tQgRab(_, _, _, m_block_id);
#pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
#pragma unroll
      for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
        if (get<1>(tQcRab(0, m, k)) < (actual_seqlen_k - n_block * kBlockN)) {
          cute::copy(
              gmem_tiled_copy_QKV,
              ctQgRab_view(_, m, k),
              tQsRab(_, m, k));
        } else {
          cute::clear(tQsRab(_, m, k));
        }
      }
    }
  };

  // Prologue
  // If not local, we're guaranteed that m_block_min <= m_block:
  // We checked earlier that n_block * kBlockN < actual_seqlen_k, so in the
  // causal case, n_block * kBlockN + actual_seqlen_q - actual_seqlen_k
  // < actual_seqlen_q. So m_block_min <= (actual_seqlen_q - 1) / kBlockM.
  // Recall that m_block_max = cute::ceil_div(actual_seqlen_q, kBlockM) =
  // (actual_seqlen_q + kBlockM - 1) / kBlockM. So m_block_max - 1 =
  // (actual_seqlen_q - 1) / kBlockM. We conclude that m_block_min <= m_block,
  // so we will always have at least 1 iteration of the for loop. However, if
  // local, then there're possible some blocks of K & V not attending to any
  // query. We might need to exit early and write 0 to dK and dV for those
  // blocks. Otherwise we get wrong result for the blocks that don't enter
  // the for loop. And we might read OOB elements from gQ and gdO. This also
  // covers the case where actual_seqlen_q == 0.
  if ((Is_causal || Is_local || Is_arbitrary) && m_block_max <= m_block_min) {
    const index_t row_offset_dk = binfo.k_offset(params.dk_row_stride) +
        n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
    const index_t row_offset_dv = binfo.k_offset(params.dv_row_stride) +
        n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
    Tensor gdK = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<Element*>(params.dk_ptr) + row_offset_dk),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(
        make_gmem_ptr(
            reinterpret_cast<Element*>(params.dv_ptr) + row_offset_dv),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(params.dv_row_stride, _1{}));
    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    clear(tdKrdK);
    clear(tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<
        /*Is_even_MN=*/false,
        /*Clear_OOB_MN=*/false,
        /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV,
        tdKrdK,
        tdKgdK,
        tdKVcdKV,
        actual_seqlen_k - n_block * kBlockN);
    flash::copy<
        /*Is_even_MN=*/false,
        /*Clear_OOB_MN=*/false,
        /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV,
        tdVrdV,
        tdVgdV,
        tdKVcdKV,
        actual_seqlen_k - n_block * kBlockN);
    return;
  }

  // prefill v
  flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
      gmem_tiled_copy_QKV,
      tVgV,
      tVsV,
      tKVcKV,
      actual_seqlen_k - n_block * kBlockN);
  if (Is_V_in_regs) {
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
    Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
    CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view)); // M
    cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
  }

  // prefill do
  flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
      gmem_tiled_copy_dO,
      tdOgdO(_, _, _, m_block),
      tdOsdO,
      tQcQ,
      actual_seqlen_q - m_block * kBlockM);

  // prefill q
  auto tQsQ_view = tQsQ(_, _, _, buffer_stage);
  flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
      gmem_tiled_copy_QKV,
      tQgQ(_, _, _, m_block),
      tQsQ_view,
      tQcQ,
      actual_seqlen_q - m_block * kBlockM);

  // prefill k
  flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
      gmem_tiled_copy_QKV,
      tKgK,
      tKsK,
      tKVcKV,
      actual_seqlen_k - n_block * kBlockN);

  if constexpr (Has_rab) {
    copy_if_g2s_rab(m_block);
  }
  cute::cp_async_fence();

  clear(acc_dv);
  clear(acc_dk);

  auto col_limit_right = [&](int row) {
    return std::min(
        actual_seqlen_k,
        row + 1 + params.window_size_right);
  };

  auto col_limit_left = [&](int row) {
    return std::max(0, row - params.window_size_left);
  };


  auto apply_mask = [&](auto& tSrS, int const m_block) {
    static constexpr int Row = 0, Col = 1;
    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma_sdp.partition_C(cS);

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
        // Below if code introduces BRA. For the backward pass, we will apply a mask for seq_q in the non casual/local case.
        // The reason for adding the 'if' statement was to guard against gMin and gMax going out of bounds.
        // However, our design ensures that the lengths of the gMin and gMax sequences are both max_seq_q, so there's no need to worry about this issue.
        /*if (row >= actual_seqlen_q) {
          #pragma unroll
          for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
            tSrS_view(mma_row, mma_col) = -INFINITY;
          }
          continue;
        }*/
        col_max(0) = gMaxFunc(0, block_row, m_block);
        #pragma unroll
        for (int j = 0; j < size<0>(gMinFunc); ++j) {
          col_min(j)   = gMinFunc(j, block_row, m_block);
          col_max(j+1) = gMaxFunc(j+1, block_row, m_block);
        }
      }

      #pragma unroll
      for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
        const int block_col = int(get<Col>(tScS_view(mma_row, mma_col)));
        const int col = block_col + base_col;
        if constexpr (!Is_causal && !Is_local) {
          if (col >= actual_seqlen_k || row >= actual_seqlen_q + actual_seqlen_offset) {
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
          if (col >= col_limit_right(row) || row >= actual_seqlen_q + actual_seqlen_offset) {
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
            if (row >= actual_seqlen_h && col >= actual_seqlen_h && col < target_col_limit_left) {
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

  auto bwd_step = [&](int m_valid_block, auto is_last_step) {
    constexpr bool Is_last_step = decltype(is_last_step)::value;

    int m_block = !Is_arbitrary ? m_valid_block : sValidBlockIds[m_valid_block];
    const bool is_masking = (m_block - m_block_min_casual) < n_masking_steps || m_block == m_block_context - 1 || is_in_mixed_context;
    Tensor acc_s = partition_fragment_C(
        tiled_mma_sdp,
        Shape<Int<kBlockM>, Int<kBlockN>>{}); // (MMA=4, MMA_N, MMA_N)

    cute::cp_async_wait<0>();
    __syncthreads();

    // compute q @ k + rab
    if constexpr (Has_rab) {
      auto rRab = make_tensor<Element>(partition_shape_C(
          tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{}));
      auto tSrRab_view = smem_thr_copy_rab.retile_D(rRab);
      static_assert(size<0>(tSrRab_view) == size<0>(tSsRab));
      static_assert(size<1>(tSrRab_view) == size<1>(tSsRab));
      static_assert(size<2>(tSrRab_view) == size<2>(tSsRab));

      cute::copy(smem_tiled_copy_rab, tSsRab, tSrRab_view);
#pragma unroll
      for (int i = 0; i < size(rRab); ++i) {
        acc_s(i) = static_cast<float>(rRab(i));
      }
    } else {
      clear(acc_s);
    }

    flash::gemm(
        acc_s,
        tSrQ,
        tSrK,
        tSsQ(_, _, _, buffer_stage),
        tSsK,
        tiled_mma_sdp,
        smem_tiled_copy_QdO,
        smem_tiled_copy_KV,
        smem_thr_copy_QdO,
        smem_thr_copy_KV);

    if (Is_arbitrary || Is_local || is_masking) {
      apply_mask(acc_s, m_block);
    }

    for (int i = 0; i < size(acc_s); ++i) {
      acc_s(i) *= params.alpha;
    }
    auto acc_s_silu = make_fragment_like(acc_s);
    silu_bwd(acc_s, acc_s_silu);

    for (int i = 0; i < size(acc_s_silu); ++i) {
      acc_s_silu(i) /= params.seqlen_q;
    }

    if (Has_rab && !Is_last_step) {
      int m_block_next = !Is_arbitrary ? m_block - 1 : sValidBlockIds[m_valid_block - 1];
      if constexpr (Is_context) {
        m_block_next = (m_block == m_block_min_casual && m_block >= m_block_context)
            ? m_block_context - 1
            : m_block_next;
      }
      __syncthreads();
      copy_g2s_rab(m_block_next);
    }

    // convert acc_s_silu from fp32 to fp16/bf16
    Tensor rP = make_tensor_like<Element>(acc_s_silu);
    flash::convert_type_safe(acc_s_silu, rP);
    Tensor tPaP = smem_thr_copy_PdS.retile_S(rP);
    cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);

    Tensor acc_dp = partition_fragment_C(
        tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
    CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));
    CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));
    CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));

    clear(acc_dp);
    // compute do @ v
    flash::gemm</*A_in_regs=*/false, /*B_in_regs=*/Is_V_in_regs>(
        acc_dp,
        tdPrdO,
        tdPrV,
        tdPsdO,
        tdPsV,
        tiled_mma_sdp,
        smem_tiled_copy_QdO,
        smem_tiled_copy_KV,
        smem_thr_copy_QdO,
        smem_thr_copy_KV);

    for (int i = 0; i < size(acc_dp); ++i) {
      acc_dp(i) /= params.seqlen_q;
    }
    dsilu_bwd(acc_dp, acc_s);
    for (int i = 0; i < size(acc_dp); ++i) {
      acc_dp(i) *= params.alpha;
    }

    if (kStages == 2 && !Is_last_step) {
      // async load(next(q))
      int m_block_next = !Is_arbitrary ? m_block - 1 : sValidBlockIds[m_valid_block - 1];
      if constexpr (Is_context) {
        m_block_next = (m_block == m_block_min_casual && m_block >= m_block_context)
            ? m_block_context - 1
            : m_block_next;
      }
      auto tQsQ_next_view = tQsQ(_, _, _, buffer_stage ^ 1);
      flash::copy</*Is_even_MN=*/true>(
          gmem_tiled_copy_QKV, tQgQ(_, _, _, m_block_next), tQsQ_next_view, tQcQ);
      cute::cp_async_fence();
    }

    // dS
    // Convert dS from fp32 to fp16/bf16
    Tensor tdSrdS = make_tensor_like<Element>(acc_dp);
    flash::convert_type_safe(acc_dp, tdSrdS);
    Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);
    cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
    __syncthreads();

    if constexpr (Has_drab) {
      if constexpr (!Rab_one_head) { // rab heads = qkv heads
        typename Kernel_traits::GmemTiledCopydRab gmem_tiled_copy_dRab;
        auto gmem_thr_copy_dRab = gmem_tiled_copy_dRab.get_thread_slice(tidx);
        Tensor tdRabsdRab = gmem_thr_copy_dRab.partition_S(sdS);
        Tensor tdRabgdRab =
            gmem_thr_copy_dRab.partition_D(gdRab(_, _, m_block));
        Tensor tdRabrdRab = make_tensor<Element>(shape(tdRabgdRab));
        cute::copy(gmem_tiled_copy_dRab, tdRabsdRab, tdRabrdRab);

        auto tdRabcRab = gmem_thr_copy_dRab.partition_D(cRab);
#pragma unroll
        for (int m = 0; m < size<1>(tdRabrdRab); m++) {
          if (get<0>(tdRabcRab(0, m, 0)) <
              (binfo.actual_seqlen_q - m_block * kBlockM)) {
#pragma unroll
            for (int k = 0; k < size<2>(tdRabrdRab); k++) {
              if (get<1>(tdRabcRab(0, m, k)) <
                      (binfo.actual_seqlen_k - n_block * kBlockN)) {
                cute::copy(
                    gmem_tiled_copy_dRab,
                    tdRabrdRab(_, m, k),
                    tdRabgdRab(_, m, k));
              }
            }
          }
        }
      } else { // rab heads = 1 && rab heads != qkv heads
        typename Kernel_traits::GmemTiledAtomdRab gmem_tiled_copy_dRab;
        using ElementV2 = typename Kernel_traits::ElementV2;
        auto gmem_thr_copy_dRab = gmem_tiled_copy_dRab.get_thread_slice(tidx);
        Tensor tdRabsdRab = gmem_thr_copy_dRab.partition_S(sdS);
        Tensor tdRabgdRab =
            gmem_thr_copy_dRab.partition_D(gdRab(_, _, m_block));
        Tensor tdRabsdRab_atomic = recast<ElementV2>(tdRabsdRab);
        Tensor tdRabgdRab_atomic = recast<ElementV2>(tdRabgdRab);
        auto tdRabcRab = gmem_thr_copy_dRab.partition_D(cRab);
#pragma unroll
        for (int m = 0; m < size<1>(tdRabgdRab_atomic); m++) {
#pragma unroll
          for (int k = 0; k < size<2>(tdRabgdRab_atomic); k++) {
            if (get<0>(tdRabcRab(0, m, k)) <
                    (binfo.actual_seqlen_q - m_block * kBlockM) &&
                get<1>(tdRabcRab(0, m, k)) <
                    (binfo.actual_seqlen_k - n_block * kBlockN)) {
              atomicAdd(
                  &tdRabgdRab_atomic(0, m, k), tdRabsdRab_atomic(0, m, k));
            }
          }
        }
      }
    }

    // calculate Pt @ dOt
    flash::gemm(
        acc_dv,
        tdVrPt,
        tdVrdOt,
        tdVsPt,
        tdVsdOt,
        tiled_mma_dkv,
        smem_tiled_copy_PdSt,
        smem_tiled_copy_QdOt,
        smem_thr_copy_PdSt,
        smem_thr_copy_QdOt);
    // Need syncthreads since we're writing to the same sdO location
    __syncthreads();

    if (!Is_last_step) {
      // async load(next(dO))
      int m_block_next = !Is_arbitrary ? m_block - 1 : sValidBlockIds[m_valid_block - 1];
      if constexpr (Is_context) {
        m_block_next = (m_block == m_block_min_casual && m_block >= m_block_context)
            ? m_block_context - 1
            : m_block_next;
      }
      flash::copy</*Is_even_MN=*/true>(
          gmem_tiled_copy_dO, tdOgdO(_, _, _, m_block_next), tdOsdO, tQcQ);
      cute::cp_async_fence();
    }

    // calculate dS @ Kt
    Tensor acc_dq = partition_fragment_C(
        tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_dq);
    flash::gemm(
        acc_dq,
        tdQrdS,
        tdQrKt,
        tdQsdS,
        tdQsKt,
        tiled_mma_dq,
        smem_tiled_copy_dS,
        smem_tiled_copy_Kt,
        smem_thr_copy_dS,
        smem_thr_copy_Kt);

    auto tdQgdQaccum_view = tdQgdQaccum(_, _, _, m_block);
    CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum_view));
#pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) {
      atomicAdd(&tdQgdQaccum_view(i), acc_dq(i));
    }

    // calculate dSt @ Qt
    flash::gemm(
        acc_dk,
        tdKrdSt,
        tdKrQt,
        tdKsdSt,
        tdKsQt(_, _, _, buffer_stage),
        tiled_mma_dkv,
        smem_tiled_copy_PdSt,
        smem_tiled_copy_QdOt,
        smem_thr_copy_PdSt,
        smem_thr_copy_QdOt);

    if constexpr (kStages == 2) { // Double buffer for sQ
      buffer_stage ^= 1;
    } else if (!Is_last_step) {
      __syncthreads();
      // async load(next(q))
      int m_block_next = !Is_arbitrary ? m_block - 1 : sValidBlockIds[m_valid_block - 1];
      if constexpr (Is_context) {
        m_block_next = (m_block == m_block_min_casual && m_block >= m_block_context)
            ? m_block_context - 1
            : m_block_next;
      }
      auto tQsQ_next_view = tQsQ(_, _, _, buffer_stage);
      flash::copy</*Is_even_MN=*/true>(
          gmem_tiled_copy_QKV, tQgQ(_, _, _, m_block_next), tQsQ_next_view, tQcQ);
      cute::cp_async_fence();
    }
  };

  for (m_block = m_block_max - 1; m_block > m_block_min; --m_block) {
    bwd_step(m_block, /*is_last_step=*/cute::bool_constant<false>{});
    if constexpr (Is_context) {
      if (m_block == m_block_min_casual && is_in_context) {
        m_block = std::min(m_block_context, m_block);
      }
    }
  }
  bwd_step(m_block_min, /*is_last_step=*/cute::bool_constant<true>{});

  // Epilogue
  // Convert acc_dk and acc_dv from fp32 to fp16/bf16
  Tensor rdK = make_tensor_like<Element>(acc_dk);
  flash::convert_type_safe(acc_dk, rdK);
  Tensor rdV = make_tensor_like<Element>(acc_dv);
  flash::convert_type_safe(acc_dv, rdV);

  Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});
  Tensor sdV = make_tensor(
      sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{});

  // Partition sdV and sdK to match the accumulator partitioning
  auto smem_tiled_copy_dKV = make_tiled_copy_C(
      typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
  auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
  Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);
  Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);
  Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);
  Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);

  // We need syncthreads here since we're writing to the same location as sK and
  // sV. Without syncthreads, some thread might modify the location of sK while
  // another thread is reading it for dQ gemm, leading to a race condition.
  __syncthreads();

  cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
  cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

  __syncthreads();

  const index_t row_offset_dk = binfo.k_offset(params.dk_row_stride) +
      n_block * kBlockN * params.dk_row_stride + bidh * params.dk_head_stride;
  const index_t row_offset_dv = binfo.k_offset(params.dv_row_stride) +
      n_block * kBlockN * params.dv_row_stride + bidh * params.dv_head_stride;
  Tensor gdK = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element*>(params.dk_ptr) + row_offset_dk),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.dk_row_stride, _1{}));
  Tensor gdV = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element*>(params.dv_ptr) + row_offset_dv),
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      make_stride(params.dv_row_stride, _1{}));

  typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
  auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
  Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);
  Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
  Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);
  Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

  Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
  cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
  Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
  cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);

  Tensor cdKV = make_identity_tensor(
      make_shape(size<0>(sdK), size<1>(sdK))); // (BLK_N,BLK_K) -> (blk_n,blk_k)
  Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::
      copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dKV,
          tdKrdK,
          tdKgdK,
          tdKVcdKV,
          actual_seqlen_k - n_block * kBlockN);
  flash::
      copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dKV,
          tdVrdV,
          tdVgdV,
          tdKVcdKV,
          actual_seqlen_k - n_block * kBlockN);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
__global__ void hstu_bwd_compute_dq_dk_dv_kernel(Hstu_bwd_params params) {
  int n_block = blockIdx.x;
  int bidh = blockIdx.z;
  int bidb = blockIdx.y;
  // If deterministic, each thread block will do atomicAdd to a different
  // dQ_accum buffer.
  int max_n_block =
      (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
  for (; n_block < max_n_block; n_block += gridDim.x) {
    hstu_compute_dq_dk_dv_1colblock<Kernel_traits>(params, bidb, bidh, n_block);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits>
__global__ void hstu_bwd_convert_dq_kernel(
    const Hstu_bwd_params params,
    const int nsplits) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;
  // Shared memory.
  extern __shared__ char smem_[];

  const auto m_block = blockIdx.x;
  // The block index for the batch.
  const auto bidb = blockIdx.y;
  // The block index for the head.
  const auto bidh = blockIdx.z;
  // The thread index.
  const auto tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  const HstuBlockInfo<Kernel_traits, Hstu_bwd_params> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q)
    return;

  const index_t row_offset_dq = binfo.q_offset(params.dq_row_stride) +
      m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
  const index_t row_offset_dq_accum = binfo.q_offset(params.dq_accum_row_stride) +
      (m_block * kBlockM + (params.cu_seqlens_q == nullptr ? 0 : 128 * bidb)) *
          params.dq_accum_row_stride +
      bidh * params.dq_accum_head_stride;

  Tensor gdQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element*>(params.dq_ptr) + row_offset_dq),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_stride(params.dq_row_stride, _1{}));
  Tensor gdQaccum = make_tensor(
      make_gmem_ptr(
          reinterpret_cast<ElementAccum*>(params.dq_accum_ptr) +
          row_offset_dq_accum),
      Shape<Int<kBlockM>, Int<kHeadDim>>{},
      make_stride(params.dq_accum_row_stride, _1{}));

  Tensor sdQ = make_tensor(
      make_smem_ptr(reinterpret_cast<Element*>(smem_)),
      typename Kernel_traits::SmemLayoutdQ{});

  typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
  auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
  typename Kernel_traits::GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
  auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

  typename Kernel_traits::TiledMmadQ tiled_mma_dq;
  auto smem_tiled_copy_dQ =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
  auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
  Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);

  Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);
  Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
  Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum);

  Tensor acc_dq =
      partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));

  Tensor tdQrdQaccum = make_fragment_like(tdQgdQaccum);
  clear(acc_dq);
  for (int s = 0; s < nsplits; ++s) {
    cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, tdQrdQaccum);
#pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) {
      acc_dq(i) += tdQrdQaccum(i);
    }
    tdQgdQaccum.data() = tdQgdQaccum.data() + params.dq_accum_split_stride;
  }

  // Convert acc_dq from fp32 to fp16/bf16
  Tensor rdQ = make_tensor_like<Element>(acc_dq);
  flash::convert_type_safe(acc_dq, rdQ);
  Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
  cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
  __syncthreads();
  Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
  cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);

  Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
  Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
  Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));

  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::
      copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dQ,
          tdQrdQ,
          tdQgdQ,
          tdQcdQ,
          binfo.actual_seqlen_q - m_block * kBlockM);
}

} // namespace flash

template <
    typename elem_type,
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    bool Is_causal,
    bool Is_target,
    bool Is_context,
    bool Is_local,
    bool Is_arbitrary,
    int kNFunc,
    bool Is_deterministic,
    bool Has_rab,
    bool Has_drab,
    bool Rab_one_head,
    int kNWarps,
    int AtomLayoutMSdP,
    int AtomLayoutNdKV,
    int AtomLayoutMdQ,
    bool Is_V_in_regs>
void run_hstu_bwd_impl_(Hstu_bwd_params& params, cudaStream_t stream) {
  using Kernel_traits = Hstu_bwd_kernel_traits<
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
      Is_deterministic,
      Has_rab,
      Has_drab,
      Rab_one_head,
      AtomLayoutMSdP,
      AtomLayoutNdKV,
      AtomLayoutMdQ,
      Is_V_in_regs,
      elem_type>;

  int gridDimx = (params.seqlen_k + kBlockN - 1) / kBlockN;
  if constexpr (Is_deterministic) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    gridDimx = (dprops->multiProcessorCount + params.b * params.h - 1) /
        (params.b * params.h);
  }
  dim3 grid_n = dim3(gridDimx, params.b, params.h);
  auto kernel = &flash::hstu_bwd_compute_dq_dk_dv_kernel<Kernel_traits>;
  constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize;
  if constexpr (smem_size_dq_dk_dv >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_dq_dk_dv));
  }
  kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(
      params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;
  dim3 grid_m(num_m_block, params.b, params.h);
  auto kernel_dq = &flash::hstu_bwd_convert_dq_kernel<Kernel_traits>;
  constexpr int smem_size_dq = Kernel_traits::kSmemdQSize;
  if constexpr (smem_size_dq >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq));
  }
  kernel_dq<<<grid_m, Kernel_traits::kNThreads, smem_size_dq, stream>>>(
      params, Is_deterministic ? gridDimx : 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <
    typename elem_type,
    int kHeadDim,
    bool Has_rab,
    bool Has_drab,
    bool Is_local,
    bool Is_causal,
    bool Is_context,
    bool Is_target,
    bool Is_arbitrary,
    int kNFunc,
    bool Is_deterministic>
void run_hstu_bwd_80(Hstu_bwd_params& params, cudaStream_t stream) {
  const bool rab_one_head = params.h_rab != params.h && params.h_rab == 1;
  BOOL_SWITCH(rab_one_head, Rab_one_head, [&] {
    static constexpr auto tile_size =
        flash::get_tile_size_bwd<kHeadDim, Has_rab>();
    static constexpr int kBlockM = std::get<0>(tile_size);
    static constexpr int kBlockN = std::get<1>(tile_size);
    static constexpr int kNWarps = std::get<2>(tile_size);
    run_hstu_bwd_impl_<
        elem_type,
        kHeadDim,
        kBlockM,
        kBlockN,
        Is_causal,
        Is_target,
        Is_context,
        Is_local,
        Is_arbitrary,
        kNFunc,
        Is_deterministic,
        Has_rab,
        Has_drab,
        Rab_one_head,
        kNWarps,
        4,
        kHeadDim <= 64 ? 4: 2,
        kHeadDim <= 64 ? 4: 2,
        false>(params, stream);
    });
}
