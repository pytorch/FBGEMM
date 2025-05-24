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

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/barrier.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "seq_len.h"
#include "utils.h"

namespace flash {

template <bool A, class Mma, class Tensor0>
CUTLASS_DEVICE
auto mma_partition_fragment_AB(Mma const& mma, Tensor0 const& tensor0) {
  if constexpr (A) {
    return mma.partition_fragment_A(tensor0);
  } else {
    return mma.partition_fragment_B(tensor0);
  }
}

using namespace cute;

template <typename Ktraits, typename Seqlen_traits>
struct CollectiveMainloopBwd {
  using Element = typename Ktraits::Element;
  using ElementAccum = typename Ktraits::ElementAccum;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape;

  static constexpr bool Is_causal = Ktraits::Is_causal;
  static constexpr bool Is_target = Ktraits::Is_target;
  static constexpr bool Is_context = Ktraits::Is_context;
  static constexpr bool Is_delta_q = Ktraits::Is_delta_q;
  static constexpr bool Is_local = Ktraits::Is_local;
  static constexpr bool Has_rab = Ktraits::Has_rab;
  static constexpr bool Has_drab = Ktraits::Has_drab;
  static constexpr bool Is_deterministic = Ktraits::Is_deterministic;

  static constexpr int kStages = Ktraits::kStages;
  static constexpr int kStages_dO = Ktraits::kStages_dO;
  static constexpr int kStages_dS = Ktraits::kStages_dS;
  static_assert(kStages >= kStages_dO);
  static_assert(kStages_dS == kStages);

  static constexpr bool SdP_swapAB = Ktraits::SdP_swapAB;
  static constexpr bool dKV_swapAB = Ktraits::dKV_swapAB;
  static constexpr bool dQ_swapAB = Ktraits::dQ_swapAB;

  static constexpr int AtomLayoutMSdP = Ktraits::AtomLayoutMSdP;
  static constexpr int AtomLayoutNdKV = Ktraits::AtomLayoutNdKV;
  static constexpr int AtomLayoutMdQ = Ktraits::AtomLayoutMdQ;

  static constexpr bool Q_dO_same_stages = kStages == kStages_dO;

  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr int kBlockN = Ktraits::kBlockN;
  static constexpr int kHeadDim = Ktraits::kHeadDim;

  static constexpr int NumMmaWarpGroups = Ktraits::NumMmaWarpGroups;
  static constexpr int NumdQWarpGroups = Ktraits::NumdQWarpGroups;
  static constexpr int NumMmaThreads = Ktraits::NumMmaThreads;
  static constexpr int kNThreadsdQ = Ktraits::kNThreadsdQ;

  static_assert(get<0>(ClusterShape{}) == 1 && get<2>(ClusterShape{}) == 1);

  static constexpr bool Mma_dKV_is_RS = Ktraits::Mma_dKV_is_RS;

  using TileShapeAtomSdP = typename Ktraits::TileShapeAtomSdP;
  using AtomLayoutSdP = typename Ktraits::AtomLayoutSdP;
  using TiledMmaSdP = typename Ktraits::TiledMmaSdP;

  using TileShapeAtomdKV = typename Ktraits::TileShapeAtomdKV;
  using AtomLayoutdKV = typename Ktraits::AtomLayoutdKV;
  using TiledMmadKV = typename Ktraits::TiledMmadKV;

  using TileShapeAtomdQ = typename Ktraits::TileShapeAtomdQ;
  using AtomLayoutdQ = typename Ktraits::AtomLayoutdQ;
  using TiledMmadQ = typename Ktraits::TiledMmadQ;

  using SmemLayoutAtomQdO = typename Ktraits::SmemLayoutAtomQdO;
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutdO = typename Ktraits::SmemLayoutdO;

  using SmemLayoutAtomRab = typename Ktraits::SmemLayoutAtomRab;
  using SmemLayoutRab = typename Ktraits::SmemLayoutRab;
  using SmemLayoutRabt = typename Ktraits::SmemLayoutRabt;

  using SmemLayoutAtomK = typename Ktraits::SmemLayoutAtomK;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;

  using SmemLayoutAtomV = typename Ktraits::SmemLayoutAtomV;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;

  using SmemLayoutAtomPdS = typename Ktraits::SmemLayoutAtomPdS;
  using SmemLayoutPdS = typename Ktraits::SmemLayoutPdS;

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutQt = typename Ktraits::SmemLayoutQt;
  using SmemLayoutdOt = typename Ktraits::SmemLayoutdOt;
  using SmemLayoutKt = typename Ktraits::SmemLayoutKt;
  using SmemLayoutPdSt = typename Ktraits::SmemLayoutPdSt;

  using R2SLayoutAtomdQaccum = typename Ktraits::R2SLayoutAtomdQaccum;
  using R2STiledCopydQaccum = typename Ktraits::R2STiledCopydQaccum;
  using SmemLayoutdQaccum = typename Ktraits::SmemLayoutdQaccum;
  using SmemLayoutAtomdQaccumTMA = typename Ktraits::SmemLayoutAtomdQaccumTMA;
  using SmemLayoutdQaccumTMA = typename Ktraits::SmemLayoutdQaccumTMA;
  using SmemLayoutdQaccumTMANoSwizzle = typename Ktraits::SmemLayoutdQaccumTMANoSwizzle;

  using SmemCopyAtomPdS = typename Ktraits::SmemCopyAtomPdS;
  using SmemCopyAtomRab = typename Ktraits::SmemCopyAtomRab;

  static constexpr bool dQacc_use_TMA = Ktraits::dQacc_use_TMA;
  using GmemLayoutAtomdQaccum = typename Ktraits::GmemLayoutAtomdQaccum;
  using GmemTiledCopydQaccumAtomic = typename Ktraits::GmemTiledCopydQaccumAtomic;

  using GmemTiledCopyQdO = typename Ktraits::GmemTiledCopyQdO;
  using GmemTiledCopyRab = typename Ktraits::GmemTiledCopyRab;
  using GmemTiledCopyKV = typename Ktraits::GmemTiledCopyKV;
  using GmemTiledCopydQaccum = typename Ktraits::GmemTiledCopydQaccum;
  using GmemTiledCopydRab = typename Ktraits::GmemTiledCopydRab;

  using TMA_QdO = decltype(make_tma_copy(
      GmemTiledCopyQdO{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
          typename Seqlen_traits::StrideT{}),
      take<0, 2>(SmemLayoutQ{}),
      select<0, 2>(TileShape_MNK{}),
      size<1>(ClusterShape{}))); // mcast along N mode for this M load, if any

  using TMA_Rab = decltype(make_tma_copy(
      GmemTiledCopyRab{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const *>(nullptr)),
          repeat_like(typename Seqlen_traits::StrideRabT{}, int32_t(0)),
          typename Seqlen_traits::StrideRabT{}),
      take<0, 2>(SmemLayoutRab{}),
      select<0, 1>(TileShape_MNK{}),
        _1{})); // no mcast for Rab

  using TMA_K = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const *>(nullptr)),
          repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
          typename Seqlen_traits::StrideT{}),
      SmemLayoutK{},
      select<1, 2>(TileShape_MNK{}),
      _1{})); // no mcast for KV

  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const *>(nullptr)),
          repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
          typename Seqlen_traits::StrideT{}),
      SmemLayoutV{},
      select<1, 2>(TileShape_MNK{}),
      _1{})); // no mcast for KV

  using TMA_add_dQ = decltype(make_tma_copy(
      GmemTiledCopydQaccum{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementAccum const *>(nullptr)),
          repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
          typename Seqlen_traits::StrideT{}),
      SmemLayoutdQaccumTMA{},
      select<0, 2>(TileShape_MNK{}),
      _1{})); // no mcast for dQ

  using TMA_store_dRab = decltype(make_tma_copy(
      GmemTiledCopydRab{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const *>(nullptr)),
          repeat_like(typename Seqlen_traits::StrideRabT{}, int32_t(0)),
          typename Seqlen_traits::StrideRabT{}),
      take<0, 2>(SmemLayoutPdS{}),
      select<0, 1>(TileShape_MNK{}),
      _1{})); // no mcast for dRab

  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipeline_dO = typename Ktraits::MainloopPipeline_dO;
  using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
  using MainloopPipeline_dRab = typename Ktraits::MainloopPipeline_dRab;
  using PipelineState_dRab = typename MainloopPipeline_dRab::PipelineState;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesRab = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutRab{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(SmemLayoutV{}) * cutlass::sizeof_bits_v<Element> / 8);

  // These are tuned for speed. They don't affect correctness.
  // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
  // this helps quite a bit to not have to do causal masking for most of the iterations.
  static constexpr bool SeparateMaskingIterations = kHeadDim <= 64;
  // For hdim256, we want to slice the dQ MMA (64 x 256 on 2 WGs) into two (64 x 128 on 2 WGs) so that we can
  // do atomic add on one half before doing the other half of the MMA, to reduce register pressure.
  static constexpr bool Slice_dQKV_Mma = kHeadDim == 256 && dQ_swapAB && AtomLayoutMdQ == 1 && NumMmaWarpGroups == 2;
  static_assert(!(Is_deterministic && Slice_dQKV_Mma), "Deterministic mode not supported with Slice_dQKV_Mma");

  // Host side kernel arguments
  struct Arguments {
    Element const* ptr_Q;
    typename Seqlen_traits::LayoutT layout_Q;
    Element const* ptr_Rab;
    typename Seqlen_traits::LayoutRabT layout_Rab;
    Element const* ptr_K;
    typename Seqlen_traits::LayoutT layout_K;
    Element const* ptr_V;
    typename Seqlen_traits::LayoutT layout_V;
    Element const* ptr_dO;
    typename Seqlen_traits::LayoutT layout_dO;
    ElementAccum* ptr_dQaccum;
    typename Seqlen_traits::LayoutT layout_dQaccum;
    Element * ptr_dRab;
    typename Seqlen_traits::LayoutRabT layout_dRab;
    const int num_batch;
    const int window_size_left;
    const int window_size_right;
    const int target_group_size;
    const float target_group_size_inv;
    const float alpha;
    int* dq_semaphore;
  };

  // Device side kernel params
  struct Params {
    typename Seqlen_traits::LayoutT layout_Q;
    typename Seqlen_traits::LayoutRabT layout_Rab;
    typename Seqlen_traits::LayoutT layout_K;
    typename Seqlen_traits::LayoutT layout_V;
    ElementAccum* ptr_dQaccum;
    typename Seqlen_traits::LayoutT layout_dQaccum;
    typename Seqlen_traits::LayoutRabT layout_dRab;
    cutlass::FastDivmod qhead_per_khead_divmod;
    cutlass::FastDivmod qhead_per_rabhead_divmod;
    TMA_QdO tma_load_Q, tma_load_dO;
    TMA_Rab tma_load_Rab;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    TMA_add_dQ tma_add_dQ;
    TMA_store_dRab tma_store_dRab;
    const int num_batch;
    const int window_size_left;
    const int window_size_right;
    const int target_group_size;
    const float target_group_size_inv;
    const float alpha;
    int* dq_semaphore;
  };

  static Params
  to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.layout_Q);
    TMA_QdO tma_load_Q = make_tma_copy(
        GmemTiledCopyQdO{},
        mQ,
        SmemLayoutQ{}(_, _, _0{}),
        select<0, 2>(TileShape_MNK{}),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    Tensor mRab = make_tensor(make_gmem_ptr(args.ptr_Rab), args.layout_Rab);
    TMA_Rab tma_load_Rab = make_tma_copy(
        GmemTiledCopyRab{},
        mRab,
        SmemLayoutRab{}(_, _, _0{}),
        select<0, 1>(TileShape_MNK{}),
        _1{}); // no mcast for Rab
    Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.layout_dO);
    TMA_QdO tma_load_dO = make_tma_copy(
        GmemTiledCopyQdO{},
        mdO,
        SmemLayoutdO{}(_, _, _0{}),
        select<0, 2>(TileShape_MNK{}),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.layout_K);
    TMA_K tma_load_K = make_tma_copy(
        GmemTiledCopyKV{},
        mK,
        SmemLayoutK{},
        select<1, 2>(TileShape_MNK{}),
        _1{}); // no mcast for KV
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.layout_V);
    TMA_V tma_load_V = make_tma_copy(
        GmemTiledCopyKV{},
        mV,
        SmemLayoutV{},
        select<1, 2>(TileShape_MNK{}),
        _1{}); // no mcast for KV
    Tensor mdQaccum = make_tensor(make_gmem_ptr(args.ptr_dQaccum), args.layout_dQaccum);
    TMA_add_dQ tma_add_dQ = make_tma_copy(
        GmemTiledCopydQaccum{},
        mdQaccum,
        SmemLayoutdQaccumTMA{},
        select<0, 2>(TileShape_MNK{}),
        _1{}); // no mcast for dQaccum
    Tensor mdRab = make_tensor(make_gmem_ptr(args.ptr_dRab), args.layout_dRab);
    TMA_store_dRab tma_store_dRab = make_tma_copy(
        GmemTiledCopydRab{},
        mdRab,
        SmemLayoutPdS{}(_, _, _0{}),
        select<0, 1>(TileShape_MNK{}),
        _1{}); // no mcast for dRab
    if constexpr (Is_deterministic) { assert(args.dq_semaphore != nullptr); }
    return {args.layout_Q, args.layout_Rab, args.layout_K, args.layout_V, args.ptr_dQaccum, args.layout_dQaccum, args.layout_dRab,
            cutlass::FastDivmod(cute::ceil_div(get<2>(args.layout_Q.shape()), get<2>(args.layout_K.shape()))),
            cutlass::FastDivmod(cute::ceil_div(get<2>(args.layout_Q.shape()), get<2>(args.layout_Rab.shape()))),
            tma_load_Q, tma_load_dO, tma_load_Rab, tma_load_K, tma_load_V, tma_add_dQ, tma_store_dRab,
            args.num_batch, args.window_size_left, args.window_size_right,
            args.target_group_size, args.target_group_size_inv, args.alpha, args.dq_semaphore};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_add_dQ.get_tma_descriptor());
    if (Has_rab){
      cute::prefetch_tma_descriptor(params.tma_load_Rab.get_tma_descriptor());
    }
  }


  template <typename SchedulerPrefetch, typename SharedStorage>
  CUTLASS_DEVICE void
  load(Params const& mainloop_params,
        MainloopPipeline pipeline_q,
        MainloopPipeline pipeline_rab,
        MainloopPipeline_dO pipeline_do,
        PipelineState& smem_pipe_write,
        PipelineState_dO& smem_pipe_write_do,
        SharedStorage &shared_storage,
        SchedulerPrefetch const& scheduler_prefetch,
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        int m_block_min,
        int m_block_max,
        bool is_in_context,
        int m_block_context,
        const Seqlen_traits &seqlen_traits_q,
        const Seqlen_traits &seqlen_traits_k) {

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sRab = make_tensor(make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    auto [n_block, bidh, bidb] = block_coord;
    int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);
    int bidh_rab = mainloop_params.qhead_per_rabhead_divmod.divide(bidh);

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape()); // (_, _, bidh);
    Tensor mRab = mainloop_params.tma_load_Rab.get_tma_tensor(mainloop_params.layout_Rab.shape())(_, _, bidh_rab, bidb);
    Tensor mdO = mainloop_params.tma_load_dO.get_tma_tensor(mainloop_params.layout_Q.shape())(_, _, bidh);
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape())(_, _, bidh_kv);
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape())(_, _, bidh_kv);

    int const offset_q = seqlen_traits_q.cu_seq_len[bidb];
    int const offset_k = seqlen_traits_k.cu_seq_len[bidb];
    // Tensor gQ = local_tile(domain_offset(make_coord(offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
    Tensor gQ = seqlen_traits_q.get_local_tile_tensor(mQ, select<0, 2>(TileShape_MNK{}), bidh, bidb);
    int actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int offset_rab = actual_seqlen_k - actual_seqlen_q;
    Tensor gRab = local_tile(domain_offset(make_coord(offset_rab, _0{}), mRab), select<0, 1>(TileShape_MNK{}),
            make_coord(_, n_block));

    Tensor gdO = local_tile(domain_offset(make_coord(offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
    Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
    Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)

    Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
    Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
    Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
    Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));

    auto block_tma_Q = mainloop_params.tma_load_Q.get_slice(cluster_local_block_id.y);
    auto block_tma_Rab = mainloop_params.tma_load_Rab.get_slice(cluster_local_block_id.y);
    auto block_tma_dO = mainloop_params.tma_load_dO.get_slice(cluster_local_block_id.y);
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
    Tensor tRabgRab = group_modes<0, 3>(block_tma_Rab.partition_S(gRab));
    Tensor tRabsRab = group_modes<0, 3>(block_tma_Rab.partition_D(sRab));
    Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
    Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
    auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));  // (TMA), (TMA)
    auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));  // (TMA), (TMA)

    uint16_t mcast_mask_qdo = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_qdo |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, _0{}));
      }
    }

    int m_block = is_in_context ? 0 : m_block_min;

    int lane_predicate = cute::elect_one_sync();
    // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
    // cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::KVEmpty) /*id*/);
    if (lane_predicate) {
      // Copy K tile and V tile from GMEM to SMEM.
      shared_storage.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
      copy(mainloop_params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_KV), 0 /*mcast_mask*/), tKgK, tKsK);
      copy(mainloop_params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_KV), 0 /*mcast_mask*/), tVgV, tVsV);
    }

    if (lane_predicate) {
        pipeline_q.producer_acquire(smem_pipe_write);
        copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
                tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
        if (Has_rab){
            pipeline_rab.producer_acquire(smem_pipe_write);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0), tRabgRab(_, m_block), tRabsRab(_, smem_pipe_write.index()));
        }
        if constexpr (Is_context) {
            #pragma unroll (kHeadDim < 256 ? 2 : 1)
            for (; m_block < m_block_context; ++m_block) {
                // If Q and dO have the same number of stages, we can use the same pipeline state variable
                // to reduce registers
                PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
                pipeline_do.producer_acquire(smem_pipe_write_do_cur);
                copy(mainloop_params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo),
                tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
                if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
                ++smem_pipe_write;
                int m_block_next = m_block + 1;
                if (m_block == m_block_context - 1) {
                  m_block_next = std::max(m_block_next, m_block_min);
                }
                if (m_block_next < m_block_max) {
                  pipeline_q.producer_acquire(smem_pipe_write);
                  copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
                      tQgQ(_, m_block_next), tQsQ(_, smem_pipe_write.index()));
                  if (Has_rab){
                      pipeline_rab.producer_acquire(smem_pipe_write);
                      copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0), tRabgRab(_, m_block_next), tRabsRab(_, smem_pipe_write.index()));
                  }
                }
            }
            m_block = std::max(m_block, m_block_min);
        }
        #pragma unroll (kHeadDim < 256 ? 2 : 1)
        for (; m_block < m_block_max - 1; ++m_block) {
          // If Q and dO have the same number of stages, we can use the same pipeline state variable
          // to reduce registers
          PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
          pipeline_do.producer_acquire(smem_pipe_write_do_cur);
          copy(mainloop_params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo),
              tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
          if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
          ++smem_pipe_write;
          pipeline_q.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
              tQgQ(_, m_block + 1), tQsQ(_, smem_pipe_write.index()));
        if (Has_rab){
            pipeline_rab.producer_acquire(smem_pipe_write);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0), tRabgRab(_, m_block + 1), tRabsRab(_, smem_pipe_write.index()));
        }
      }
      if(m_block < m_block_max) {
        PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
        pipeline_do.producer_acquire(smem_pipe_write_do_cur);
        copy(mainloop_params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo),
            tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
        if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
        ++smem_pipe_write;
      }
    }
    scheduler_prefetch();
    if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline_q, MainloopPipeline pipeline_rab, MainloopPipeline_dO pipeline_do,
            PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do) {
    PipelineState smem_pipe_write_rab = smem_pipe_write;
    int lane_predicate = cute::elect_one_sync();
    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
      * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
      * then would just be acquired since the phase was still inverted from make_producer_start_state
      */
      pipeline_q.producer_tail(smem_pipe_write);
      pipeline_do.producer_tail(smem_pipe_write_do);
      if (Has_rab){
          pipeline_rab.producer_tail(smem_pipe_write_rab);
      }
    }
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void
  store_dq(Params const& mainloop_params,
            SharedStorage &shared_storage,
            cute::tuple<int32_t, int32_t, int32_t> block_coord,
            const int m_block_min,
            const int m_block_max,
            const bool is_in_context,
            const int m_block_context,
            const Seqlen_traits &seqlen_traits_q) {
    if constexpr (!dQacc_use_TMA) { return; }

    Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccumTMA{});
    Tensor sdQnoswizzle = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccumTMANoSwizzle{});
    auto [n_block, bidh, bidb] = block_coord;

    int const offset_padded = (seqlen_traits_q.cu_seq_len[bidb] + bidb * kBlockM) / kBlockM * kBlockM;
    // Prepare the TMA loads
    Tensor mdQaccum = mainloop_params.tma_add_dQ.get_tma_tensor(mainloop_params.layout_dQaccum.shape())(_, _, bidh);
    Tensor gdQaccum = local_tile(domain_offset(make_coord(offset_padded, _0{}), mdQaccum), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
    auto block_tma_dQ = mainloop_params.tma_add_dQ.get_slice(_0{});
    Tensor tdQgdQ = block_tma_dQ.partition_D(gdQaccum);  // (TMA, TMA_M, TMA_K)
    Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ); // (TMA, TMA_M, TMA_K)

    int m_block = is_in_context ? 0 : m_block_min;

    int const num_batch = mainloop_params.num_batch;
    int const num_head = get<2>(mainloop_params.layout_Q.shape());
    int *lock_ptr = !Is_deterministic ? nullptr : mainloop_params.dq_semaphore + bidb * num_head + bidh;
    using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
    int lane_predicate = cute::elect_one_sync();

    if constexpr (Is_context) {
      #pragma unroll 2
      for (; m_block < m_block_context; ++m_block) {
        if constexpr (Is_deterministic) {
          Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head, n_block);
        }
        cutlass::arch::NamedBarrier::sync(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQFull) /*id*/);  // sdQ full, to be written to gmem
        if (lane_predicate) {
          cute::copy(mainloop_params.tma_add_dQ, tdQsdQ, tdQgdQ(_, _, _, m_block));
          tma_store_arrive();
        }
        tma_store_wait<0>();
        if constexpr (Is_deterministic) {
          Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
        }
        cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
      }
      m_block = std::max(m_block, m_block_min);
    }

    #pragma unroll 2
    for (; m_block < m_block_max; ++m_block) {
      if constexpr (Is_deterministic) {
        Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head, n_block);
      }
      cutlass::arch::NamedBarrier::sync(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQFull) /*id*/);  // sdQ full, to be written to gmem
      if (lane_predicate) {
        cute::copy(mainloop_params.tma_add_dQ, tdQsdQ, tdQgdQ(_, _, _, m_block));
        tma_store_arrive();
      }
      tma_store_wait<0>();
      if constexpr (Is_deterministic) {
        Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
      }
      cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
    }
    if constexpr (Is_local && Is_deterministic) {
      constexpr int kBlockM = get<0>(TileShape_MNK{});
      int const actual_seqlen_q = seqlen_traits_q.actual_seq_len;
      int const m_block_global_max = cute::ceil_div(actual_seqlen_q, kBlockM);
      #pragma unroll 2
      for (; m_block < m_block_global_max; ++m_block) {
        Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
      }
    }
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void
  store_drab(Params const& mainloop_params,
            MainloopPipeline_dRab pipeline_drab,
            PipelineState_dRab& smem_pipe_read_drab,
            SharedStorage &shared_storage,
            cute::tuple<int32_t, int32_t, int32_t> block_coord,
            const int m_block_min,
            const int m_block_max,
            const bool is_in_context,
            const int m_block_context,
            const Seqlen_traits &seqlen_traits_q,
            const Seqlen_traits &seqlen_traits_k) {
    Tensor sdRab = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdS{});
    Tensor sdRab_pi = cute::as_position_independent_swizzle_tensor(sdRab);
    Tensor sdRabt = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdSt{});
    Tensor sdRabt_pi = cute::as_position_independent_swizzle_tensor(sdRabt);
    auto [n_block, bidh, bidb] = block_coord;
    int bidh_drab = mainloop_params.qhead_per_rabhead_divmod.divide(bidh);

    // Prepare the TMA stores
    int offset_rab = Is_delta_q ? seqlen_traits_k.actual_seq_len - seqlen_traits_q.actual_seq_len : 0;
    Tensor mdRab = mainloop_params.tma_store_dRab.get_tma_tensor(mainloop_params.layout_dRab.shape())(_, _, bidh_drab, bidb);
    Tensor gdRab = local_tile(domain_offset(make_coord(offset_rab, _0{}), mdRab), select<0, 1>(TileShape_MNK{}),
            make_coord(_, n_block));

    auto block_tma_dRab = mainloop_params.tma_store_dRab.get_slice(_0{});
    Tensor tdRabgdRab = block_tma_dRab.partition_D(gdRab);
    Tensor tdRabsdRab = block_tma_dRab.partition_S(sdRab);

    int m_block = is_in_context ? 0 : m_block_min;

    if constexpr (Is_context) {
      #pragma unroll 2
      for (; m_block < m_block_context; ++m_block) {
        pipeline_drab.consumer_wait(smem_pipe_read_drab);
        if (cute::elect_one_sync()) {
          cute::copy(mainloop_params.tma_store_dRab, tdRabsdRab(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read_drab.index())), tdRabgdRab(_, _, _, m_block));
          tma_store_arrive();
        }
        tma_store_wait<0>();
        pipeline_drab.consumer_release(smem_pipe_read_drab);
        ++smem_pipe_read_drab;
      }
      m_block = std::max(m_block, m_block_min);
    }

    #pragma unroll 2
    for (; m_block < m_block_max; ++m_block) {
      pipeline_drab.consumer_wait(smem_pipe_read_drab);
      if (cute::elect_one_sync()) {
        cute::copy(mainloop_params.tma_store_dRab, tdRabsdRab(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read_drab.index())), tdRabgdRab(_, _, _, m_block));
        tma_store_arrive();
      }
      tma_store_wait<0>();
      pipeline_drab.consumer_release(smem_pipe_read_drab);
      ++smem_pipe_read_drab;
    }
  }

  CUTLASS_DEVICE void
  mma_init() {
    // Tell producer (warp 0) that smem_k and smem_v are ready
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::KVEmpty) /*id*/);
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if constexpr (dQacc_use_TMA) {
      if (cutlass::canonical_warp_group_idx() == 1 && warp_idx_in_warpgroup == 0) {
        cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
      }
    }
  }

  template <typename SharedStorage, typename FrgTensordKV>
  CUTLASS_DEVICE void
  mma(Params const& mainloop_params,
      MainloopPipeline pipeline_q,
      MainloopPipeline pipeline_rab,
      MainloopPipeline_dO pipeline_do,
      MainloopPipeline_dRab pipeline_drab,
      PipelineState& smem_pipe_read,
      PipelineState_dO& smem_pipe_read_do,
      PipelineState_dRab& smem_pipe_write_drab,
      FrgTensordKV& tdKrdK,
      FrgTensordKV& tdVrdV,
      int m_block_min,
      int m_block_max,
      int m_masking_steps,
      bool is_in_context,
      int m_block_context,
      int thread_idx,
      int work_idx,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      SharedStorage& shared_storage,
      const Seqlen_traits &seqlen_traits_q,
      const Seqlen_traits &seqlen_traits_k) {
      static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sRab = make_tensor(make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sRabt = make_tensor(make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRabt{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
    Tensor sQt = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQt{});
    Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), SmemLayoutdOt{});
    Tensor sKt = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutKt{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutPdS{});
    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
    Tensor sPt = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutPdSt{});
    Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
    Tensor sdS = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdS{});
    Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
    Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdSt{});
    Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
    Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccum{});

    static_assert(stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
                  stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
                  size<0>(typename TiledMmaSdP::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                  size<0>(typename TiledMmaSdP::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
    constexpr int MmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                  make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
    Layout warp_group_thread_layout_dq = make_layout(make_shape(Int<NumdQWarpGroups>{}),
                                                  make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    TiledMmaSdP tiled_mma_SdP;
    TiledMmadKV tiled_mma_dKV;
    TiledMmadQ tiled_mma_dQ;

    auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
    auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dQ = tiled_mma_dQ.get_slice(thread_idx);

    auto smem_tiled_copy_PdS = make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);

    R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
    auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(NumdQWarpGroups == 2 ? thread_idx : thread_idx % cutlass::NumThreadsPerWarpGroup);
    Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);

    // Allocate "fragments/descriptors"
    Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
    Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
    Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
    Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sV);
    Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);
    Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
    Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
    Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

    Tensor tPsP = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP_pi, sPt_pi));
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_pi, sdSt_pi));
    // For Rab
    auto smem_tiled_copy_rab = make_tiled_copy_C(SmemCopyAtomRab{}, tiled_mma_SdP);
    auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(thread_idx);
    Tensor tSsRab = smem_thr_copy_rab.partition_S(cute::conditional_return<!SdP_swapAB>(sRab, sRabt));
    Tensor tSrRab = make_tensor<Element>(partition_shape_C(tiled_mma_SdP, cute::conditional_return<!SdP_swapAB>(select<0, 1>(TileShape_MNK{}), select<1, 0>(TileShape_MNK{}))));
    Tensor tSrRab_copy_view = smem_thr_copy_rab.retile_D(tSrRab);
    Tensor tSrRab_accum = make_tensor_like<ElementAccum>(tSrRab);

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
        auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
        pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    auto [n_block, bidh, bidb] = block_coord;
    constexpr int kBlockM = get<0>(TileShape_MNK{});
    constexpr int kBlockN = get<1>(TileShape_MNK{});

    int const actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int const actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int const actual_seqlen_h = Is_target ? seqlen_traits_k.actual_seq_len_h : actual_seqlen_k;
    int const actual_seqlen_c = Is_context ? seqlen_traits_k.actual_seq_len_c : 0;

    int const actual_seqlen_offset = Is_delta_q ? actual_seqlen_k - actual_seqlen_q : 0;
    int const max_seq_len_q = seqlen_traits_q.max_seq_len;
    int m_block = is_in_context ? 0 : m_block_min;

    Tensor mdQaccum = make_tensor(make_gmem_ptr(mainloop_params.ptr_dQaccum), mainloop_params.layout_dQaccum)(_, _, bidh);
    int const offset_padded = (seqlen_traits_q.cu_seq_len[bidb] + bidb * kBlockM) / kBlockM * kBlockM;
    Tensor gdQaccum = local_tile(domain_offset(make_coord(offset_padded, _0{}), mdQaccum), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));

    GmemTiledCopydQaccumAtomic gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    auto col_limit_right = [&](int row) {
        return std::min(
            actual_seqlen_k,
            row + 1 + actual_seqlen_offset + mainloop_params.window_size_right
        );
    };

    auto col_limit_left = [&](int row) {
        return std::max(
            0,
            row + actual_seqlen_offset - mainloop_params.window_size_left
        );
    };

    auto apply_mask = [&](auto& tSrS, const int m_block, const int n_block, auto is_causal_type, auto is_local_type, auto is_context_type) {
      static constexpr int Row = !SdP_swapAB ? 0 : 1;
      static constexpr int Col = !SdP_swapAB ? 1 : 0;
      Tensor cS = cute::make_identity_tensor(select<Row, Col>(TileShape_MNK{}));
      Tensor tScS = thread_mma_SdP.partition_C(cS);
      constexpr bool Is_in_causal = decltype(is_causal_type)::value;
      constexpr bool Is_in_local = decltype(is_local_type)::value;
      constexpr bool Is_in_context = decltype(is_context_type)::value;
      if (threadIdx.x > 9999) {
          printf("This would never run. But this is neccessary to avoid register overflow.\n");
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSrS); ++i) {
        int row = int(get<Row>(tScS(i))) + m_block * kBlockM;
        int col = int(get<Col>(tScS(i))) + n_block * kBlockN;
        if constexpr (!Is_in_causal && !Is_in_local) {
          if (col >= actual_seqlen_k || row >= actual_seqlen_q) {
              tSrS(i) = -INFINITY;
          }
        } else {
          if constexpr (Is_in_context) {
            if (row < actual_seqlen_c && col < actual_seqlen_h) {
              continue;
            }
          }
          // causal mask
          if (col >= col_limit_right(row) || row >= actual_seqlen_q) {
            tSrS(i) = -INFINITY;
            continue;
          }
          if constexpr (Is_in_local) {
            if (col < col_limit_left(row)) {
              tSrS(i) = -INFINITY;
              continue;
            }
          }
          if constexpr (Is_target) {
            const int target_index = (row - actual_seqlen_h) * mainloop_params.target_group_size_inv;
            const int target_col_limit_left = actual_seqlen_h + target_index * mainloop_params.target_group_size;
            if (row >= actual_seqlen_h && col >= actual_seqlen_h) {
              if (col < target_col_limit_left) {
                tSrS(i) = -INFINITY;
              }
            }
          }
        }
      }
    };

    clear(tdKrdK);
    clear(tdVrdV);

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_KV.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.barrier_KV.wait(work_idx % 2); }

    auto bwd_step = [&](int m_block, auto mask_fn) {
      Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
      pipeline_q.consumer_wait(smem_pipe_read);
      flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
      if constexpr (Has_rab){
        pipeline_rab.consumer_wait(smem_pipe_read);
        cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, smem_pipe_read.index()), tSrRab_copy_view);
        flash::convert_type_safe(tSrRab, tSrRab_accum);
      }
      Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
      PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
      pipeline_do.consumer_wait(smem_pipe_read_do_cur);
      flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);
      if constexpr (Has_rab){
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) += tSrRab_accum(i);
        }
        pipeline_rab.consumer_release(smem_pipe_read);
      }
      for (int i = 0; i < size(tSrS); ++i) {
        tSrS(i) *= mainloop_params.alpha;
      }

      mask_fn(tSrS, m_block);
      auto tSrS_silu = make_fragment_like(tSrS);
      silu_bwd(tSrS, tSrS_silu);
      for (int i = 0; i < size(tSrS_silu); ++i) {
        tSrS_silu(i) /= max_seq_len_q;
      }
      // Convert scores from fp32 to fp16/bf16
      Tensor rP = make_tensor_like<Element>(tSrS_silu);
      flash::convert_type_safe(tSrS_silu, rP);

      if constexpr (!Slice_dQKV_Mma && Mma_dKV_is_RS) {
        Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
        flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
      } else {
        warpgroup_wait<0>();
      }
      for (int i = 0; i < size(tdPrdP); ++i) {
        tdPrdP(i) /= max_seq_len_q;
      }
      dsilu_bwd(tdPrdP, tSrS);
      for (int i = 0; i < size(tdPrdP); ++i) {
        tdPrdP(i) *= mainloop_params.alpha;
      }

      if constexpr (!Mma_dKV_is_RS) {
        Tensor tPaP = smem_thr_copy_PdS.retile_S(rP);
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())));
        cutlass::arch::fence_view_async_shared();
      }
      Tensor rdS = make_tensor_like<Element>(tdPrdP);
      flash::convert_type_safe(tdPrdP, rdS);
      Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);
      // If there's double buffering on dS, we don't need to sync here.
      // Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
      // But because both WGs have to sync at the end of the loop and double buffering,
      // this race condition is not possible.
      // This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
      // (2) dS is already read by the Mma in the previous iteration in case of Mma_dKV_is_RS.
      if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS) /*id*/);
      }
      if constexpr (Has_drab) {
        pipeline_drab.producer_acquire(smem_pipe_write_drab);
      }
      cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())));

      if constexpr (!Slice_dQKV_Mma) {
        // Most cases take this path, except for hdim256 where we want to slice to reduce register pressure
        if constexpr (Mma_dKV_is_RS) {
          // If dKV is RS, it's slightly faster to kick off dK Mma before dQ_Mma
          Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
        } else {
          Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
          Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
        }
        pipeline_do.consumer_release(smem_pipe_read_do_cur);

        // SMEM fence to make sure sdS is written before it's read by WGMMA
        cutlass::arch::fence_view_async_shared();
        if constexpr (Has_drab) {
          pipeline_drab.producer_commit(smem_pipe_write_drab);
        }
        if constexpr (dQacc_use_TMA) {
          cutlass::arch::NamedBarrier::sync(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty));
        } else {
          cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS));
        }
        Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        if constexpr (Mma_dKV_is_RS) { pipeline_q.consumer_release(smem_pipe_read); }
        if constexpr (dQacc_use_TMA) {
          Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
          cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
          cutlass::arch::fence_view_async_shared();
          cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQFull));
        } else {
          Tensor tdQrdQ_atomic = recast<float4>(tdQrdQ);
          Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
        }

        if constexpr (!Mma_dKV_is_RS) {
          Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
          Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
        }

      } else {  // Slice_dQKV_Mma

        Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
        Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);

        cutlass::arch::fence_view_async_shared();
        if constexpr (Has_drab) {
          pipeline_drab.producer_commit(smem_pipe_write_drab);
        }
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS));
        Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/dQ_swapAB, /*M_slice=*/0>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
        Tensor tdQrdQ_atomic = recast<float4>(tdQrdQ);
        Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

        Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
        Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
        flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
        pipeline_do.consumer_release(smem_pipe_read_do_cur);

        flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB, /*M_slice=*/1>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        CUTLASS_PRAGMA_UNROLL
        for (int i = size(tdQrdQ_atomic) / 2;  i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

        flash::gemm</*zero_init=*/false, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
      }

      if constexpr (!Mma_dKV_is_RS) { pipeline_q.consumer_release(smem_pipe_read); }
      ++smem_pipe_read;
      ++smem_pipe_write_drab;
      if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
    };

    if constexpr (Is_context) {
      auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<Is_causal>{}, cute::bool_constant<Is_local>{}, cute::bool_constant<Is_context>{}); };
      for (; m_block < m_block_context; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
      m_block = std::max(m_block, m_block_min);
    }

    if constexpr ((Is_causal || Is_local) && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<Is_causal>{}, cute::bool_constant<Is_local>{}, cute::bool_constant<false>{}); };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < std::min(m_block_max, m_block_min + m_masking_steps); ++m_block) {
          bwd_step(m_block, mask_fn);
      }
    }

    static constexpr int n_local_bottom_steps = (!Is_local || !SeparateMaskingIterations) ? 0 : cute::ceil_div(kBlockN, kBlockM) + 1;
    auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<Is_causal && !SeparateMaskingIterations>{}, cute::bool_constant<Is_local>{}, cute::bool_constant<false>{}); };
    CUTLASS_PRAGMA_NO_UNROLL
    for (; m_block < m_block_max - n_local_bottom_steps; ++m_block) {
      bwd_step(m_block, mask_fn);
    }

    if constexpr (Is_local && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<false>{}, cute::bool_constant<Is_local>{}, cute::bool_constant<false>{}); };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
  }

};

} // namespace flash
