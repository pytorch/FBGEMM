/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cutlass/array.h>
#include <cutlass/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "seq_len.h"
#include "utils.h"
#define NO_FP8_COLUMN_PERMUTE

namespace flash {

template <bool A, class Mma, class Tensor0>
CUTLASS_DEVICE auto mma_partition_fragment_AB(
    Mma const& mma,
    Tensor0 const& tensor0) {
  if constexpr (A) {
    return mma.partition_fragment_A(tensor0);
  } else {
    return mma.partition_fragment_B(tensor0);
  }
}

using namespace cute;

template <typename Ktraits, typename Seqlen_traits>
struct CollectiveMainloopBwd {
  using index_t = typename Ktraits::index_t;
  using Element = typename Ktraits::Element;
  using ElementAccum = typename Ktraits::ElementAccum;
  using ElementRab = typename Ktraits::ElementRab;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape;

  static constexpr bool Is_causal = Ktraits::Is_causal;
  static constexpr bool Is_target = Ktraits::Is_target;
  static constexpr bool Is_context = Ktraits::Is_context;
  static constexpr bool Is_local = Ktraits::Is_local;
  static constexpr bool Is_arbitrary = Ktraits::Is_arbitrary;
  static constexpr int  kNFunc = Ktraits::kNFunc;

  static constexpr bool Has_rab = Ktraits::Has_rab;
  static constexpr bool Has_drab = Ktraits::Has_drab;
  static constexpr bool Is_deterministic = Ktraits::Is_deterministic;
  static constexpr int Quant_mode = Ktraits::Quant_mode;
  static constexpr bool Is_fp8 = Ktraits::Is_fp8;

  static constexpr int kStages = Ktraits::kStages;
  static constexpr int kStages_dO = Ktraits::kStages_dO;
  static constexpr int kStages_dS = Ktraits::kStages_dS;
  static_assert(kStages >= kStages_dO);
  static_assert(kStages_dS == kStages);
  static constexpr int kMBlock_shared = Ktraits::kMBlock_shared;
  static constexpr int kNBlock_shared = Ktraits::kNBlock_shared;

  static constexpr bool SdP_swapAB = Ktraits::SdP_swapAB;
  static constexpr bool dKV_swapAB = Ktraits::dKV_swapAB;
  static constexpr bool dQ_swapAB = Ktraits::dQ_swapAB;

  static constexpr int AtomLayoutMdQ = Ktraits::AtomLayoutMdQ;

  static constexpr bool Q_dO_same_stages = kStages == kStages_dO;

  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr int kBlockN = Ktraits::kBlockN;
  static constexpr int kHeadDim = Ktraits::kHeadDim;

  static constexpr int NumMmaWarps = Ktraits::NumMmaWarps;
  static constexpr int NumMmaWarpGroups = Ktraits::NumMmaWarpGroups;
  static constexpr int NumdQWarpGroups = Ktraits::NumdQWarpGroups;
  static constexpr int NumMmaThreads = Ktraits::NumMmaThreads;
  static constexpr int kNThreadsdQ = Ktraits::kNThreadsdQ;

  static_assert(get<0>(ClusterShape{}) == 1 && get<2>(ClusterShape{}) == 1);

  static constexpr bool Mma_dKV_is_RS = Ktraits::Mma_dKV_is_RS;

  using ShapeQKVdO = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, d, head)
  using StrideQKVdO = cute::Stride<int64_t, _1, int64_t>;
  // using StrideQKVdOt = cute::Stride<_1, int64_t, int64_t>;

  using ShapeRab = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, seqlen, head, batch)
  using StrideRab = cute::Stride<int64_t, _1, int64_t, int64_t>;

  using LayoutQKVdO = cute::Layout<ShapeQKVdO, StrideQKVdO>;
  using LayoutRab = cute::Layout<ShapeRab, StrideRab>;

  using TiledMmaSdP = typename Ktraits::TiledMmaSdP;
  using TiledMmadKV = typename Ktraits::TiledMmadKV;
  using TiledMmadQ = typename Ktraits::TiledMmadQ;

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutdO = typename Ktraits::SmemLayoutdO;
  using SmemLayoutRab = typename Ktraits::SmemLayoutRab;
  using SmemLayoutRabt = typename Ktraits::SmemLayoutRabt;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutPdS = typename Ktraits::SmemLayoutPdS;

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutQt = typename Ktraits::SmemLayoutQt;
  using SmemLayoutdOt = typename Ktraits::SmemLayoutdOt;
  using SmemLayoutKt = typename Ktraits::SmemLayoutKt;
  using SmemLayoutPdSt = typename Ktraits::SmemLayoutPdSt;

  using R2STiledCopydQaccum = typename Ktraits::R2STiledCopydQaccum;
  using SmemLayoutdQaccum = typename Ktraits::SmemLayoutdQaccum;
  using SmemLayoutdQaccumTMA = typename Ktraits::SmemLayoutdQaccumTMA;
  using SmemLayoutdQaccumTMANoSwizzle =
      typename Ktraits::SmemLayoutdQaccumTMANoSwizzle;

  using SmemCopyAtomRab = typename Ktraits::SmemCopyAtomRab;

  static constexpr bool dQacc_use_TMA = Ktraits::dQacc_use_TMA;
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
          ShapeQKVdO{},
          StrideQKVdO{}),
      take<0, 2>(SmemLayoutQ{}),
      select<0, 2>(TileShape_MNK{}),
      size<1>(ClusterShape{}))); // mcast along N mode for this M load, if any

  using TMA_Rab = decltype(make_tma_copy(
      GmemTiledCopyRab{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementRab const*>(nullptr)),
          ShapeRab{},
          StrideRab{}),
      take<0, 2>(SmemLayoutRab{}),
      select<0, 1>(TileShape_MNK{}),
      _1{})); // no mcast for Rab

  using TMA_K = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKVdO{},
          StrideQKVdO{}),
      SmemLayoutK{},
      select<1, 2>(TileShape_MNK{}),
      _1{})); // no mcast for KV

  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKVdO{},
          StrideQKVdO{}),
      SmemLayoutV{},
      select<1, 2>(TileShape_MNK{}),
      _1{})); // no mcast for KV

  using TMA_add_dQ = decltype(make_tma_copy(
      GmemTiledCopydQaccum{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementAccum const*>(nullptr)),
          ShapeQKVdO{},
          StrideQKVdO{}),
      SmemLayoutdQaccumTMA{},
      select<0, 2>(TileShape_MNK{}),
      _1{})); // no mcast for dQ

  using TMA_store_dRab = decltype(make_tma_copy(
      GmemTiledCopydRab{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementRab const*>(nullptr)),
          ShapeRab{},
          StrideRab{}),
      take<0, 2>(SmemLayoutPdS{}),
      select<0, 1>(TileShape_MNK{}),
      _1{})); // no mcast for dRab

  using TMA_QdOt = std::conditional_t<Is_fp8,
    decltype(make_tma_copy(
      GmemTiledCopyQdO{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKVdO{},
          StrideQKVdO{}),
      take<0, 2>(SmemLayoutQt{}),
      select<2, 0>(TileShape_MNK{}),
      size<1>(ClusterShape{}))),
    int>; // mcast along N mode for this M load, if any

  using TMA_Kt = std::conditional_t<Is_fp8,
    decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const *>(nullptr)),
          ShapeQKVdO{},
          StrideQKVdO{}),
      take<0, 2>(SmemLayoutKt{}),
      select<2, 1>(TileShape_MNK{}),
      _1{})),
    int>; // no mcast for KV

  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipelinedO = typename Ktraits::MainloopPipelinedO;
  using PipelineStatedO = typename MainloopPipelinedO::PipelineState;
  using MainloopPipelinedRab = typename Ktraits::MainloopPipelinedRab;
  using PipelineStatedRab = typename MainloopPipelinedRab::PipelineState;

  // Set the bytes transferred in this TMA transaction (may involve multiple
  // issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesRab = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutRab{})) * cutlass::sizeof_bits_v<ElementRab> / 8);
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
      size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
      size(SmemLayoutV{}) * cutlass::sizeof_bits_v<Element> / 8);

  // These are tuned for speed. They don't affect correctness.
  // We have separate iterations with causal masking. Not necessary for hdim 128
  // but for hdim 64 this helps quite a bit to not have to do causal masking for
  // most of the iterations.
  static constexpr bool SeparateMaskingIterations = kHeadDim <= 64;
  // For hdim256, we want to slice the dQ MMA (64 x 256 on 2 WGs) into two (64 x
  // 128 on 2 WGs) so that we can do atomic add on one half before doing the
  // other half of the MMA, to reduce register pressure.
  static constexpr bool Slice_dQKV_Mma = kHeadDim == 256 && dQ_swapAB &&
      AtomLayoutMdQ == 1 && NumMmaWarpGroups == 2;
  static_assert(
      !(Is_deterministic && Slice_dQKV_Mma),
      "Deterministic mode not supported with Slice_dQKV_Mma");

  // Host side kernel arguments
  struct Arguments {
    Element const* ptr_Q;
    LayoutQKVdO layout_Q;
    ElementRab const* ptr_Rab;
    LayoutRab layout_Rab;
    Element const* ptr_K;
    LayoutQKVdO layout_K;
    Element const* ptr_V;
    LayoutQKVdO layout_V;
    Element const* ptr_dO;
    LayoutQKVdO layout_dO;
    ElementAccum* ptr_dQaccum;
    LayoutQKVdO layout_dQaccum;
    ElementRab * ptr_dRab;
    LayoutRab layout_dRab;
    Element const* ptr_Qt;
    LayoutQKVdO layout_Qt;
    Element const* ptr_dOt;
    LayoutQKVdO layout_dOt;
    Element const* ptr_Kt;
    LayoutQKVdO layout_Kt;
    const int num_batch;
    float const *descale_q_ptr;
    index_t descale_q_head_stride;
    float const *descale_qt_ptr;
    index_t descale_qt_head_stride;
    index_t descale_qt_row_stride;
    float const *descale_k_ptr;
    index_t descale_k_head_stride;
    float const *descale_kt_ptr;
    index_t descale_kt_head_stride;
    index_t descale_kt_row_stride;
    float const *descale_v_ptr;
    index_t descale_v_head_stride;
    float const *descale_do_ptr;
    index_t descale_do_head_stride;
    float const *descale_dot_ptr;
    index_t descale_dot_head_stride;
    index_t descale_dot_row_stride;
    int const *cu_seqlens_descale_qt_ptr;
    int const *cu_seqlens_descale_kt_ptr;
    int const* cu_seqlens_q_block_descale;
    int const* cu_seqlens_kv_block_descale;
    int const *func_ptr;
    const index_t func_ids_stride;
    const int window_size_left;
    const int window_size_right;
    const int target_group_size;
    const float alpha;
    int* dq_semaphore;
    const index_t q_block_descale_head_stride;
    const index_t kv_block_descale_head_stride;
  };

  // Device side kernel params
  struct Params {
    LayoutQKVdO layout_Q;
    LayoutRab layout_Rab;
    LayoutQKVdO layout_K;
    LayoutQKVdO layout_V;
    LayoutQKVdO layout_Qt;
    LayoutQKVdO layout_dOt;
    LayoutQKVdO layout_Kt;
    ElementAccum* ptr_dQaccum;
    LayoutQKVdO layout_dQaccum;
    LayoutRab layout_dRab;
    cutlass::FastDivmod qhead_per_khead_divmod;
    cutlass::FastDivmod qhead_per_rabhead_divmod;
    TMA_QdO tma_load_Q, tma_load_dO;
    TMA_Rab tma_load_Rab;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    TMA_add_dQ tma_add_dQ;
    TMA_store_dRab tma_store_dRab;
    TMA_QdOt tma_load_Qt, tma_load_dOt;
    TMA_Kt tma_load_Kt;
    const int num_batch;
    float const *descale_q_ptr;
    index_t descale_q_head_stride;
    float const *descale_qt_ptr;
    index_t descale_qt_head_stride;
    index_t descale_qt_row_stride;
    float const *descale_k_ptr;
    index_t descale_k_head_stride;
    float const *descale_kt_ptr;
    index_t descale_kt_head_stride;
    index_t descale_kt_row_stride;
    float const *descale_v_ptr;
    index_t descale_v_head_stride;
    float const *descale_do_ptr;
    index_t descale_do_head_stride;
    float const *descale_dot_ptr;
    index_t descale_dot_head_stride;
    index_t descale_dot_row_stride;
    int const *cu_seqlens_descale_qt_ptr;
    int const *cu_seqlens_descale_kt_ptr;
    int const* cu_seqlens_q_block_descale;
    int const* cu_seqlens_kv_block_descale;
    int const *func_ptr;
    const index_t func_ids_stride;
    const int window_size_left;
    const int window_size_right;
    const int target_group_size;
    const float alpha;
    int* dq_semaphore;
    const index_t q_block_descale_head_stride;
    const index_t kv_block_descale_head_stride;
  };

  static Params to_underlying_arguments(Arguments const& args) {
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
    Tensor mdQaccum =
        make_tensor(make_gmem_ptr(args.ptr_dQaccum), args.layout_dQaccum);
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

    TMA_QdOt tma_load_Qt{};
    TMA_QdOt tma_load_dOt{};
    TMA_Kt tma_load_Kt{};
    if constexpr (Is_fp8) {
      Tensor mQt = make_tensor(make_gmem_ptr(args.ptr_Qt), args.layout_Qt);
      tma_load_Qt = make_tma_copy(
          GmemTiledCopyQdO{},
          mQt,
          SmemLayoutQt{}(_, _, _0{}),
          select<2, 0>(TileShape_MNK{}),
          size<1>(ClusterShape{})); // mcast along N mode for this M load, if any

      Tensor mdOt = make_tensor(make_gmem_ptr(args.ptr_dOt), args.layout_dOt);
      tma_load_dOt = make_tma_copy(
          GmemTiledCopyQdO{},
          mdOt,
          SmemLayoutdOt{}(_, _, _0{}),
          select<2, 0>(TileShape_MNK{}),
          size<1>(ClusterShape{})); // mcast along N mode for this M load, if any

      Tensor mKt = make_tensor(make_gmem_ptr(args.ptr_Kt), args.layout_Kt);
      tma_load_Kt = make_tma_copy(
          GmemTiledCopyKV{},
          mKt,
          SmemLayoutKt{},
          select<2, 1>(TileShape_MNK{}),
          _1{}); // no mcast for KV
    }

    if constexpr (Is_deterministic) { assert(args.dq_semaphore != nullptr); }
    return {args.layout_Q, args.layout_Rab, args.layout_K, args.layout_V, args.layout_Qt, args.layout_dOt, args.layout_Kt,
            args.ptr_dQaccum, args.layout_dQaccum, args.layout_dRab,
            cutlass::FastDivmod(cute::ceil_div(get<2>(args.layout_Q.shape()), get<2>(args.layout_K.shape()))),
            cutlass::FastDivmod(cute::ceil_div(get<2>(args.layout_Q.shape()), get<2>(args.layout_Rab.shape()))),
            tma_load_Q, tma_load_dO, tma_load_Rab, tma_load_K, tma_load_V, tma_add_dQ, tma_store_dRab,
            tma_load_Qt, tma_load_dOt, tma_load_Kt,
            args.num_batch,
            args.descale_q_ptr, args.descale_q_head_stride, args.descale_qt_ptr, args.descale_qt_head_stride, args.descale_qt_row_stride,
            args.descale_k_ptr, args.descale_k_head_stride, args.descale_kt_ptr, args.descale_kt_head_stride, args.descale_kt_row_stride,
            args.descale_v_ptr, args.descale_v_head_stride,
            args.descale_do_ptr, args.descale_do_head_stride, args.descale_dot_ptr, args.descale_dot_head_stride, args.descale_dot_row_stride,
            args.cu_seqlens_descale_qt_ptr, args.cu_seqlens_descale_kt_ptr, args.cu_seqlens_q_block_descale, args.cu_seqlens_kv_block_descale,
            args.func_ptr, args.func_ids_stride, args.window_size_left, args.window_size_right,
            args.target_group_size, args.alpha, args.dq_semaphore, args.q_block_descale_head_stride, args.kv_block_descale_head_stride};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
  /// performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_add_dQ.get_tma_descriptor());
    if constexpr (Has_rab) {
      cute::prefetch_tma_descriptor(params.tma_load_Rab.get_tma_descriptor());
    }
  }

  template <typename SchedulerPrefetch, typename SharedStorage>
  CUTLASS_DEVICE void load(
      Params const& mainloop_params,
      MainloopPipeline pipeline_q,
      MainloopPipeline pipeline_rab,
      MainloopPipelinedO pipeline_do,
      PipelineState& smem_pipe_write,
      PipelineStatedO& smem_pipe_write_do,
      SharedStorage& shared_storage,
      SchedulerPrefetch const& scheduler_prefetch,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      int m_block_min,
      int m_block_max,
      bool is_in_context,
      int m_block_context,
      const Seqlen_traits& seqlen_traits_q,
      const Seqlen_traits& seqlen_traits_k) {
    Tensor sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sRab = make_tensor(
        make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sdO = make_tensor(
        make_smem_ptr(shared_storage.smem_do.data()), SmemLayoutdO{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV =
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    auto [n_block, bidh, bidb] = block_coord;
    int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);
    int bidh_rab = mainloop_params.qhead_per_rabhead_divmod.divide(bidh);

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {
        block_rank_in_cluster % cluster_shape_x,
        block_rank_in_cluster / cluster_shape_x};
    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(
        mainloop_params.layout_Q.shape()); // (_, _, bidh);
    Tensor mRab = mainloop_params.tma_load_Rab.get_tma_tensor(
        mainloop_params.layout_Rab.shape())(_, _, bidh_rab, bidb);
    Tensor mdO = mainloop_params.tma_load_dO.get_tma_tensor(
        mainloop_params.layout_Q.shape())(_, _, bidh);
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(
        mainloop_params.layout_K.shape())(_, _, bidh_kv);
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(
        mainloop_params.layout_V.shape())(_, _, bidh_kv);

    int const offset_q = seqlen_traits_q.cu_seq_len[bidb];
    int const offset_k = seqlen_traits_k.cu_seq_len[bidb];
    // Tensor gQ = local_tile(domain_offset(make_coord(offset_q, _0{}), mQ),
    // select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
    Tensor gQ = seqlen_traits_q.get_local_tile_tensor(
        mQ, select<0, 2>(TileShape_MNK{}), bidh, bidb);
    int actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int offset_rab = actual_seqlen_k - actual_seqlen_q;
    Tensor gRab = local_tile(
        domain_offset(make_coord(offset_rab, _0{}), mRab),
        select<0, 1>(TileShape_MNK{}),
        make_coord(_, n_block));

    Tensor gdO = local_tile(
        domain_offset(make_coord(offset_q, _0{}), mdO),
        select<0, 2>(TileShape_MNK{}),
        make_coord(_, _0{})); // (M, K, _)
    Tensor gK = local_tile(
        domain_offset(make_coord(offset_k, _0{}), mK),
        select<1, 2>(TileShape_MNK{}),
        make_coord(n_block, _0{})); // (N, K)
    Tensor gV = local_tile(
        domain_offset(make_coord(offset_k, _0{}), mV),
        select<1, 2>(TileShape_MNK{}),
        make_coord(n_block, _0{})); // (N, K)

    Tensor sK_x =
        make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
    Tensor gK_x =
        make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
    Tensor sV_x =
        make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
    Tensor gV_x =
        make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));

    auto block_tma_Q =
        mainloop_params.tma_load_Q.get_slice(cluster_local_block_id.y);
    auto block_tma_Rab =
        mainloop_params.tma_load_Rab.get_slice(cluster_local_block_id.y);
    auto block_tma_dO =
        mainloop_params.tma_load_dO.get_slice(cluster_local_block_id.y);
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
    Tensor tRabgRab = group_modes<0, 3>(block_tma_Rab.partition_S(gRab));
    Tensor tRabsRab = group_modes<0, 3>(block_tma_Rab.partition_D(sRab));
    Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
    Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
    auto [tKgK, tKsK] = tma_partition(
        mainloop_params.tma_load_K,
        _0{},
        Layout<_1>{},
        group_modes<0, 2>(sK_x),
        group_modes<0, 2>(gK_x)); // (TMA), (TMA)
    auto [tVgV, tVsV] = tma_partition(
        mainloop_params.tma_load_V,
        _0{},
        Layout<_1>{},
        group_modes<0, 2>(sV_x),
        group_modes<0, 2>(gV_x)); // (TMA), (TMA)

    uint16_t mcast_mask_qdo = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_qdo |=
            (uint16_t(1) << block_layout(cluster_local_block_id.x, n, _0{}));
      }
    }
    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});
    int lane_predicate = cute::elect_one_sync();
    // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
    if (lane_predicate) {
      // Copy K tile and V tile from GMEM to SMEM.
      shared_storage.barrier_KV.arrive_and_expect_tx(
          TmaTransactionBytesK + TmaTransactionBytesV);
      copy(
          mainloop_params.tma_load_K.with(
              reinterpret_cast<
                  cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                  shared_storage.barrier_KV),
              0 /*mcast_mask*/),
          tKgK,
          tKsK);
      copy(
          mainloop_params.tma_load_V.with(
              reinterpret_cast<
                  cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                  shared_storage.barrier_KV),
              0 /*mcast_mask*/),
          tVgV,
          tVsV);
    }

    auto load_step = [&](int m_block_valid, int m_block_next_valid) {
      int m_block      = !Is_arbitrary ? m_block_valid : sValidBlockIds[m_block_valid];
      int m_block_next = !Is_arbitrary ? m_block_next_valid : sValidBlockIds[m_block_next_valid];
      // If Q and dO have the same number of stages, we can use the same pipeline state variable
      // to reduce registers
      PipelineStatedO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
      pipeline_do.producer_acquire(smem_pipe_write_do_cur);
      copy(mainloop_params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo),
          tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
      if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
      ++smem_pipe_write;
      if (!Is_context || m_block_next < m_block_max) {
        pipeline_q.producer_acquire(smem_pipe_write);
        copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
            tQgQ(_, m_block_next), tQsQ(_, smem_pipe_write.index()));
        if (Has_rab) {
            pipeline_rab.producer_acquire(smem_pipe_write);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0), tRabgRab(_, m_block_next), tRabsRab(_, smem_pipe_write.index()));
        }
      }
    };

    int m_block = is_in_context ? 0 : m_block_min;
    m_block = !Is_arbitrary ? m_block : sValidBlockIds[m_block];

    if (lane_predicate) {
      pipeline_q.producer_acquire(smem_pipe_write);
      copy(
          mainloop_params.tma_load_Q.with(
              *pipeline_q.producer_get_barrier(smem_pipe_write),
              mcast_mask_qdo),
          tQgQ(_, m_block),
          tQsQ(_, smem_pipe_write.index()));
      if (Has_rab) {
        pipeline_rab.producer_acquire(smem_pipe_write);
        copy(
            mainloop_params.tma_load_Rab.with(
                *pipeline_rab.producer_get_barrier(smem_pipe_write), 0),
            tRabgRab(_, m_block),
            tRabsRab(_, smem_pipe_write.index()));
      }
      m_block = !Is_arbitrary ? m_block : m_block_min; // back to m_block_min for arbitrary mask

      if constexpr (Is_context) {
#pragma unroll(kHeadDim < 256 ? 2 : 1)
        for (; m_block < m_block_context; ++m_block) {
          int m_block_next = m_block + 1;
          if (m_block == m_block_context - 1) {
            m_block_next = std::max(m_block_next, m_block_min);
          }
          load_step(m_block, m_block_next);
        }
        m_block = std::max(m_block, m_block_min);
      }
#pragma unroll(kHeadDim < 256 ? 2 : 1)
      for (; m_block < m_block_max - 1; ++m_block) {
        int m_block_next = m_block + 1;
        load_step(m_block, m_block_next);
      }

      if (m_block < m_block_max) {
        m_block = !Is_arbitrary ? m_block : sValidBlockIds[m_block];
        PipelineStatedO smem_pipe_write_do_cur =
            cute::conditional_return<Q_dO_same_stages>(
                smem_pipe_write, smem_pipe_write_do);
        pipeline_do.producer_acquire(smem_pipe_write_do_cur);
        copy(
            mainloop_params.tma_load_dO.with(
                *pipeline_do.producer_get_barrier(smem_pipe_write_do_cur),
                mcast_mask_qdo),
            tdOgdO(_, m_block),
            tdOsdO(_, smem_pipe_write_do_cur.index()));
        if constexpr (!Q_dO_same_stages) {
          ++smem_pipe_write_do;
        }
        ++smem_pipe_write;
      }
    }
    scheduler_prefetch();
    if constexpr (Q_dO_same_stages) {
      smem_pipe_write_do = smem_pipe_write;
    }
  }

  template <typename SchedulerPrefetch, typename SharedStorage>
  CUTLASS_DEVICE void
  load_fp8(Params const& mainloop_params,
        MainloopPipeline pipeline_q,
        MainloopPipeline pipeline_qt,
        MainloopPipeline pipeline_rab,
        MainloopPipelinedO pipeline_do,
        MainloopPipelinedO pipeline_dot,
        PipelineState& smem_pipe_write,
        PipelineState& smem_pipe_read,
        PipelineStatedO& smem_pipe_write_do,
        PipelineStatedO& smem_pipe_read_do,
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
    Tensor sQt = make_tensor(make_smem_ptr(shared_storage.smem_qt.data()), SmemLayoutQt{});
    Tensor sKt = make_tensor(make_smem_ptr(shared_storage.smem_kt.data()), SmemLayoutKt{});
    Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.smem_dot.data()), SmemLayoutdOt{});

    auto [n_block, bidh, bidb] = block_coord;
    int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);
    int bidh_rab = mainloop_params.qhead_per_rabhead_divmod.divide(bidh);

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
    Tensor mRab = mainloop_params.tma_load_Rab.get_tma_tensor(mainloop_params.layout_Rab.shape())(_, _, bidh_rab, bidb);
    Tensor mdO = mainloop_params.tma_load_dO.get_tma_tensor(mainloop_params.layout_Q.shape())(_, _, bidh);
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape())(_, _, bidh_kv);
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape())(_, _, bidh_kv);
    Tensor mQt = mainloop_params.tma_load_Qt.get_tma_tensor(mainloop_params.layout_Qt.shape());
    Tensor mdOt = mainloop_params.tma_load_dOt.get_tma_tensor(mainloop_params.layout_dOt.shape())(_, _, bidh);
    Tensor mKt = mainloop_params.tma_load_Kt.get_tma_tensor(mainloop_params.layout_Kt.shape())(_, _, bidh_kv);

    int const offset_q = seqlen_traits_q.cu_seq_len[bidb];
    int const offset_k = seqlen_traits_k.cu_seq_len[bidb];
    Tensor gQ = seqlen_traits_q.get_local_tile_tensor(mQ, select<0, 2>(TileShape_MNK{}), bidh, bidb);
    Tensor gQt = seqlen_traits_q.get_local_tile_tensorT(mQt, select<2, 0>(TileShape_MNK{}), bidh, bidb);
    int offset_rab = seqlen_traits_k.actual_seq_len - seqlen_traits_q.actual_seq_len;
    Tensor gRab = local_tile(domain_offset(make_coord(offset_rab, _0{}), mRab), select<0, 1>(TileShape_MNK{}),
            make_coord(_, n_block));

    Tensor gdO = local_tile(domain_offset(make_coord(offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
    Tensor gdOt = local_tile(domain_offset(make_coord(_0{}, offset_q), mdOt), select<2, 0>(TileShape_MNK{}), make_coord(_0{}, _));  // (M, K, _)
    Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
    Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
    Tensor gKt = local_tile(domain_offset(make_coord(_0{}, offset_k), mKt), select<2, 1>(TileShape_MNK{}), make_coord(_0{}, n_block));  // (N, K)

    Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
    Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
    Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
    Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
    Tensor sKt_x = make_tensor(sKt.data(), make_layout(sKt.layout(), Layout<_1>{}));
    Tensor gKt_x = make_tensor(gKt.data(), make_layout(gKt.layout(), Layout<_1>{}));

    auto block_tma_Q = mainloop_params.tma_load_Q.get_slice(cluster_local_block_id.y);
    auto block_tma_Rab = mainloop_params.tma_load_Rab.get_slice(cluster_local_block_id.y);
    auto block_tma_dO = mainloop_params.tma_load_dO.get_slice(cluster_local_block_id.y);
    auto block_tma_Qt = mainloop_params.tma_load_Qt.get_slice(cluster_local_block_id.y);
    auto block_tma_dOt = mainloop_params.tma_load_dOt.get_slice(cluster_local_block_id.y);
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
    Tensor tQtgQt = group_modes<0, 3>(block_tma_Qt.partition_S(gQt));
    Tensor tQtsQt = group_modes<0, 3>(block_tma_Qt.partition_D(sQt));
    Tensor tRabgRab = group_modes<0, 3>(block_tma_Rab.partition_S(gRab));
    Tensor tRabsRab = group_modes<0, 3>(block_tma_Rab.partition_D(sRab));
    Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
    Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
    Tensor tdOtgdOt = group_modes<0, 3>(block_tma_dOt.partition_S(gdOt));
    Tensor tdOtsdOt = group_modes<0, 3>(block_tma_dOt.partition_D(sdOt));
    auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));  // (TMA), (TMA)
    auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));  // (TMA), (TMA)
    auto [tKtgKt, tKtsKt] = tma_partition(mainloop_params.tma_load_Kt, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sKt_x), group_modes<0, 2>(gKt_x));  // (TMA), (TMA)

    uint16_t mcast_mask_qdo = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_qdo |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, _0{}));
      }
    }
    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});

    int m_block = is_in_context ? 0 : m_block_min;
    m_block = !Is_arbitrary ? m_block : sValidBlockIds[m_block];

    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);

    auto load_step = [&](int m_block_valid, int m_block_next_valid) {
      int m_block      = !Is_arbitrary ? m_block_valid : sValidBlockIds[m_block_valid];
      int m_block_next = !Is_arbitrary ? m_block_next_valid : sValidBlockIds[m_block_next_valid];
      // If Q and dO have the same number of stages, we can use the same pipeline state variable
      // to reduce registers
      PipelineStatedO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
      pipeline_do.producer_acquire(smem_pipe_write_do_cur);
      copy(mainloop_params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo),
          tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));

      if constexpr (Quant_mode == 1) {
        pipeline_qt.producer_acquire(smem_pipe_write);
        copy(mainloop_params.tma_load_Qt.with(*pipeline_qt.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
            tQtgQt(_, m_block), tQtsQt(_, smem_pipe_write.index()));

        PipelineStatedO smem_pipe_write_dot_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
        pipeline_dot.producer_acquire(smem_pipe_write_dot_cur);
        copy(mainloop_params.tma_load_dOt.with(*pipeline_dot.producer_get_barrier(smem_pipe_write_dot_cur), mcast_mask_qdo),
            tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_dot_cur.index()));
      }
      if constexpr (!Q_dO_same_stages) {
        ++smem_pipe_write_do;
      }
      ++smem_pipe_write; // write next smem

      if (!Is_context || m_block_next < m_block_max) {
        pipeline_q.producer_acquire(smem_pipe_write);
        copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
            tQgQ(_, m_block_next), tQsQ(_, smem_pipe_write.index()));
        if (Has_rab) {
            pipeline_rab.producer_acquire(smem_pipe_write);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0), tRabgRab(_, m_block_next), tRabsRab(_, smem_pipe_write.index()));
        }
      }
    };

    if (lane_predicate && warp_idx_in_warpgroup == 0) {
      pipeline_q.producer_acquire(smem_pipe_write);
      copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
            tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
      if constexpr (Has_rab) {
        pipeline_rab.producer_acquire(smem_pipe_write);
        copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0), tRabgRab(_, m_block), tRabsRab(_, smem_pipe_write.index()));
      }

      // Copy K tile and V tile from GMEM to SMEM.
      const int TmaTransactionBytes = TmaTransactionBytesK + TmaTransactionBytesV + (Quant_mode == 1 ? TmaTransactionBytesK : 0);
      shared_storage.barrier_KV.arrive_and_expect_tx(TmaTransactionBytes);
      copy(mainloop_params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_KV), 0 /*mcast_mask*/), tKgK, tKsK);
      copy(mainloop_params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_KV), 0 /*mcast_mask*/), tVgV, tVsV);

      if constexpr (Quant_mode == 1) {
        copy(mainloop_params.tma_load_Kt.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_KV), 0 /*mcast_mask*/), tKtgKt, tKtsKt);
      }
      m_block = !Is_arbitrary ? m_block : m_block_min;

      if constexpr (Is_context) {
        #pragma unroll (kHeadDim < 256 ? 2 : 1)
        for (; m_block < m_block_context; ++m_block) {
          int m_block_next = m_block + 1;
          if (m_block == m_block_context - 1) {
            m_block_next = std::max(m_block_next, m_block_min);
          }
          load_step(m_block, m_block_next);
        }
        m_block = std::max(m_block, m_block_min);
      }

      #pragma unroll (kHeadDim < 256 ? 2 : 1)
      for (; m_block < m_block_max - 1; ++m_block) {
        int m_block_next = m_block + 1;
        load_step(m_block, m_block_next);
      }
    }

    if (m_block < m_block_max) {
      if (lane_predicate && warp_idx_in_warpgroup == 0) {
        m_block = !Is_arbitrary ? m_block : sValidBlockIds[m_block];
        PipelineStatedO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
        pipeline_do.producer_acquire(smem_pipe_write_do_cur);
        copy(mainloop_params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo),
            tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));

        if constexpr (Quant_mode == 1) {
          pipeline_qt.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_Qt.with(*pipeline_qt.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
              tQtgQt(_, m_block), tQtsQt(_, smem_pipe_write.index()));

          PipelineStatedO smem_pipe_write_dot_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
          pipeline_dot.producer_acquire(smem_pipe_write_dot_cur);
          copy(mainloop_params.tma_load_dOt.with(*pipeline_dot.producer_get_barrier(smem_pipe_write_dot_cur), mcast_mask_qdo),
              tdOtgdOt(_, m_block), tdOtsdOt(_, smem_pipe_write_dot_cur.index()));
        }
      }

      // trans dO
      if constexpr (!Q_dO_same_stages) {
        ++smem_pipe_write_do;
        ++smem_pipe_read_do;
      }
      ++smem_pipe_read; // read next smem
      ++smem_pipe_write; // write next smem
    }
    scheduler_prefetch();
    if constexpr (Q_dO_same_stages) {
      smem_pipe_write_do = smem_pipe_write;
      smem_pipe_read_do = smem_pipe_read;
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(
      MainloopPipeline pipeline_q,
      MainloopPipeline pipeline_rab,
      MainloopPipelinedO pipeline_do,
      PipelineState& smem_pipe_write,
      PipelineStatedO& smem_pipe_write_do) {
    PipelineState smem_pipe_write_rab = smem_pipe_write;
    int lane_predicate = cute::elect_one_sync();
    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all Consumer UNLOCKs), or
       * if the stage was never used then would just be acquired since the phase
       * was still inverted from make_producer_start_state
       */
      pipeline_q.producer_tail(smem_pipe_write);
      pipeline_do.producer_tail(smem_pipe_write_do);
      if constexpr (Has_rab) {
        pipeline_rab.producer_tail(smem_pipe_write_rab);
      }
    }
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void store_dq(
      Params const& mainloop_params,
      SharedStorage& shared_storage,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      const int m_block_min,
      const int m_block_max,
      const bool is_in_context,
      const int m_block_context,
      const Seqlen_traits& seqlen_traits_q) {
    if constexpr (!dQacc_use_TMA) {
      return;
    }

    Tensor sdQ = make_tensor(
        make_smem_ptr(shared_storage.smem_dqacc.data()),
        SmemLayoutdQaccumTMA{});
    Tensor sdQnoswizzle = make_tensor(
        make_smem_ptr(shared_storage.smem_dqacc.data()),
        SmemLayoutdQaccumTMANoSwizzle{});
    auto [n_block, bidh, bidb] = block_coord;

    int const offset_padded =
        (seqlen_traits_q.cu_seq_len[bidb] + bidb * kBlockM) / kBlockM * kBlockM;
    // Prepare the TMA loads
    Tensor mdQaccum = mainloop_params.tma_add_dQ.get_tma_tensor(
        mainloop_params.layout_dQaccum.shape())(_, _, bidh);
    Tensor gdQaccum = local_tile(
        domain_offset(make_coord(offset_padded, _0{}), mdQaccum),
        select<0, 2>(TileShape_MNK{}),
        make_coord(_, _0{})); // (M, K, _)
    auto block_tma_dQ = mainloop_params.tma_add_dQ.get_slice(_0{});
    Tensor tdQgdQ = block_tma_dQ.partition_D(gdQaccum); // (TMA, TMA_M, TMA_K)
    Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ); // (TMA, TMA_M, TMA_K)
    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});

    int m_block = is_in_context ? 0 : m_block_min;

    int const num_batch = mainloop_params.num_batch;
    int const num_head = get<2>(mainloop_params.layout_Q.shape());
    int* lock_ptr = !Is_deterministic
        ? nullptr
        : mainloop_params.dq_semaphore + bidb * num_head + bidh;
    using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
    int lane_predicate = cute::elect_one_sync();

    auto store_dq_step = [&](int m_block_valid) {
      int m_block = !Is_arbitrary ? m_block_valid : sValidBlockIds[m_block_valid];
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
    };

    if constexpr (Is_context) {
      for (; m_block < m_block_context; ++m_block) {
        store_dq_step(m_block);
      }
      m_block = std::max(m_block, m_block_min);
    }

    for (; m_block < m_block_max; ++m_block) {
      store_dq_step(m_block);
    }
    if constexpr (Is_local && Is_deterministic) {
      constexpr int kBlockM = get<0>(TileShape_MNK{});
      int const actual_seqlen_q = seqlen_traits_q.actual_seq_len;
      int const m_block_global_max = cute::ceil_div(actual_seqlen_q, kBlockM);
#pragma unroll 2
      for (; m_block < m_block_global_max; ++m_block) {
        Barrier::arrive_inc(
            lock_ptr,
            threadIdx.x % cutlass::NumThreadsPerWarp,
            m_block * num_batch * num_head);
      }
    }
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void store_drab(
      Params const& mainloop_params,
      MainloopPipelinedRab pipeline_drab,
      PipelineStatedRab& smem_pipe_read_drab,
      SharedStorage& shared_storage,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      const int m_block_min,
      const int m_block_max,
      const bool is_in_context,
      const int m_block_context,
      const Seqlen_traits& seqlen_traits_q,
      const Seqlen_traits& seqlen_traits_k) {
    Tensor sdRab = make_tensor(
        make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdS{});
    Tensor sdRab_pi = cute::as_position_independent_swizzle_tensor(sdRab);
    Tensor sdRabt = make_tensor(
        make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdSt{});
    Tensor sdRabt_pi = cute::as_position_independent_swizzle_tensor(sdRabt);
    auto [n_block, bidh, bidb] = block_coord;
    int bidh_drab = mainloop_params.qhead_per_rabhead_divmod.divide(bidh);

    // Prepare the TMA stores
    int offset_rab = seqlen_traits_k.actual_seq_len - seqlen_traits_q.actual_seq_len;
    Tensor mdRab = mainloop_params.tma_store_dRab.get_tma_tensor(
        mainloop_params.layout_dRab.shape())(_, _, bidh_drab, bidb);
    Tensor gdRab = local_tile(
        domain_offset(make_coord(offset_rab, _0{}), mdRab),
        select<0, 1>(TileShape_MNK{}),
        make_coord(_, n_block));

    auto block_tma_dRab = mainloop_params.tma_store_dRab.get_slice(_0{});
    Tensor tdRabgdRab = block_tma_dRab.partition_D(gdRab);
    Tensor tdRabsdRab = block_tma_dRab.partition_S(sdRab);

    int m_block = is_in_context ? 0 : m_block_min;
    int lane_predicate = cute::elect_one_sync();
    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});

    auto store_drab_step = [&](int m_block_valid) {
      int m_block = !Is_arbitrary ? m_block_valid : sValidBlockIds[m_block_valid];
      pipeline_drab.consumer_wait(smem_pipe_read_drab);
      if (lane_predicate) {
        cute::copy(mainloop_params.tma_store_dRab, tdRabsdRab(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read_drab.index())), tdRabgdRab(_, _, _, m_block));
        tma_store_arrive();
      }
      tma_store_wait<0>();
      pipeline_drab.consumer_release(smem_pipe_read_drab);
      ++smem_pipe_read_drab;
    };
    if constexpr (Is_context) {
      for (; m_block < m_block_context; ++m_block) {
        store_drab_step(m_block);
      }
      m_block = std::max(m_block, m_block_min);
    }

    for (; m_block < m_block_max; ++m_block) {
      store_drab_step(m_block);
    }
  }

  CUTLASS_DEVICE void mma_init() {
    // Tell producer (warp 0) that smem_k and smem_v are ready
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<int>(BwdNamedBarriers::KVEmpty) /*id*/);
    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if constexpr (dQacc_use_TMA) {
      if (cutlass::canonical_warp_group_idx() == 1 &&
          warp_idx_in_warpgroup == 0) {
        cutlass::arch::NamedBarrier::arrive(
            kNThreadsdQ + cutlass::NumThreadsPerWarp,
            static_cast<int>(
                BwdNamedBarriers::dQEmpty) /*id*/); // sdQ empty, ready to be
                                                    // written to
      }
    }
  }

  template <typename SharedStorage, typename FrgTensordKV>
  CUTLASS_DEVICE void mma(
      Params const& mainloop_params,
      MainloopPipeline pipeline_q,
      MainloopPipeline pipeline_rab,
      MainloopPipelinedO pipeline_do,
      MainloopPipelinedRab pipeline_drab,
      PipelineState& smem_pipe_read,
      PipelineStatedO& smem_pipe_read_do,
      PipelineStatedRab& smem_pipe_write_drab,
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
      const Seqlen_traits& seqlen_traits_q,
      const Seqlen_traits& seqlen_traits_k) {
    using SmemCopyAtomPdS = typename Ktraits::SmemCopyAtomPdS;
    static_assert(
        is_rmem<FrgTensordKV>::value,
        "dK and dV tensor must be rmem resident.");

    Tensor sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sRab = make_tensor(
        make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sRabt = make_tensor(
        make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRabt{});
    Tensor sdO = make_tensor(
        make_smem_ptr(shared_storage.smem_do.data()), SmemLayoutdO{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV =
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
    Tensor sQt = make_tensor(
        make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQt{});
    Tensor sdOt = make_tensor(
        make_smem_ptr(shared_storage.smem_do.data()), SmemLayoutdOt{});
    Tensor sKt = make_tensor(
        make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutKt{});
    Tensor sP = make_tensor(
        make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutPdS{});
    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
    Tensor sPt = make_tensor(
        make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutPdSt{});
    Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
    Tensor sdS = make_tensor(
        make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdS{});
    Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
    Tensor sdSt = make_tensor(
        make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdSt{});
    Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
    Tensor sdQ = make_tensor(
        make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccum{});

    static_assert(
        stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
            stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
            size<0>(typename TiledMmaSdP::ALayout{}) ==
                cutlass::NumThreadsPerWarpGroup and
            size<0>(typename TiledMmaSdP::BLayout{}) ==
                cutlass::NumThreadsPerWarpGroup,
        "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
    constexpr int MmaWarpGroups =
        NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(
        make_shape(Int<MmaWarpGroups>{}),
        make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
    Layout warp_group_thread_layout_dq = make_layout(
        make_shape(Int<NumdQWarpGroups>{}),
        make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    int warp_group_idx = __shfl_sync(
        0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    TiledMmaSdP tiled_mma_SdP;
    TiledMmadKV tiled_mma_dKV;
    TiledMmadQ tiled_mma_dQ;

    auto wg_mma_SdP =
        tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
    auto wg_mma_dKV =
        tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dQ = tiled_mma_dQ.get_slice(thread_idx);

    auto smem_tiled_copy_PdS =
        make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);

    R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
    auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(
        NumdQWarpGroups == 2 ? thread_idx
                             : thread_idx % cutlass::NumThreadsPerWarpGroup);
    Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);

    // Allocate "fragments/descriptors"
    Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
    Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
    Tensor tdPrdO =
        mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
    Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sV);
    Tensor tdVrdO =
        mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);
    Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
    Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
    Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

    Tensor tPsP = smem_thr_copy_PdS.partition_D(
        cute::conditional_return<!SdP_swapAB>(sP_pi, sPt_pi));
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(
        cute::conditional_return<!SdP_swapAB>(sdS_pi, sdSt_pi));
    // For Rab
    auto smem_tiled_copy_rab =
        make_tiled_copy_C(SmemCopyAtomRab{}, tiled_mma_SdP);
    auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(thread_idx);
    Tensor tSsRab = smem_thr_copy_rab.partition_S(
        cute::conditional_return<!SdP_swapAB>(sRab, sRabt));
    Tensor tSrRab = make_tensor<Element>(partition_shape_C(
        tiled_mma_SdP,
        cute::conditional_return<!SdP_swapAB>(
            select<0, 1>(TileShape_MNK{}), select<1, 0>(TileShape_MNK{}))));
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
    int const actual_seqlen_h =
        Is_target ? seqlen_traits_k.actual_seq_len_h : actual_seqlen_k;
    int const actual_seqlen_c =
        Is_context ? seqlen_traits_k.actual_seq_len_c : 0;

    int const actual_seqlen_offset = actual_seqlen_k - actual_seqlen_q;
    float param_seqlen_q = static_cast<float>(seqlen_traits_q.max_seq_len);
    int m_block = is_in_context ? 0 : m_block_min;

    Tensor mdQaccum = make_tensor(
        make_gmem_ptr(mainloop_params.ptr_dQaccum),
        mainloop_params.layout_dQaccum)(_, _, bidh);
    int const offset_padded =
        (seqlen_traits_q.cu_seq_len[bidb] + bidb * kBlockM) / kBlockM * kBlockM;
    Tensor gdQaccum = local_tile(
        domain_offset(make_coord(offset_padded, _0{}), mdQaccum),
        select<0, 2>(TileShape_MNK{}),
        make_coord(_, _0{}));

    GmemTiledCopydQaccumAtomic gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum =
        gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    // arbitrary func
    Tensor mMaxFunc = make_tensor(make_gmem_ptr(mainloop_params.func_ptr + seqlen_traits_q.offset),
        make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2 + 1>{}, actual_seqlen_q),
        make_stride(/*mainloop_params.func_head_stride*/Int<0>{}, 2 * mainloop_params.func_ids_stride, _1{}));
    Tensor mMinFunc = make_tensor(make_gmem_ptr(mainloop_params.func_ptr + seqlen_traits_q.offset + mainloop_params.func_ids_stride),
        make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2>{}, actual_seqlen_q),
        make_stride(/*mainloop_params.func_head_stride*/Int<0>{}, 2 * mainloop_params.func_ids_stride, _1{}));

    Tensor gMaxFunc = local_tile(mMaxFunc(Int<0>{}, _, _),
        make_shape(Int<kNFunc/2 + 1>{}, Int<kBlockM>{}), make_coord(Int<0>{}, _));
    Tensor gMinFunc = local_tile(mMinFunc(Int<0>{}, _, _),
        make_shape(Int<kNFunc/2>{}, Int<kBlockM>{}), make_coord(Int<0>{}, _));

    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});
    auto col_limit_right = [&](int row) {
      return std::min(
          actual_seqlen_k,
          row + 1 + mainloop_params.window_size_right);
    };

    auto col_limit_left = [&](int row) {
      return std::max(0, row - mainloop_params.window_size_left);
    };

    auto apply_mask = [&](auto& tSrS,
                          const int m_block,
                          const int n_block,
                          auto is_causal_type,
                          auto is_local_type,
                          auto is_context_type,
                          auto is_arbitrary_type) {
      static constexpr int Row = !SdP_swapAB ? 0 : 1;
      static constexpr int Col = !SdP_swapAB ? 1 : 0;

      constexpr bool Is_in_causal = decltype(is_causal_type)::value;
      constexpr bool Is_in_local = decltype(is_local_type)::value;
      constexpr bool Is_in_context = decltype(is_context_type)::value;
      constexpr bool Is_in_arbitrary = decltype(is_arbitrary_type)::value;

      Tensor cS = cute::make_identity_tensor(select<Row, Col>(TileShape_MNK{}));
      Tensor tScS = thread_mma_SdP.partition_C(cS);

      const int base_row = m_block * kBlockM + actual_seqlen_offset;
      const int base_col = n_block * kBlockN;

      Tensor tSrS_view = make_tensor(tSrS.data(),
        cute::conditional_return<!SdP_swapAB>(
          group<1, 4>(group<0, 2>(select<1, 3, 0, 2, 4>(flatten(tSrS.layout())))),
          group<1, 3>(group<0, 3>(select<0, 2, 4, 1, 3>(flatten(tSrS.layout()))))
        )
      );

      Tensor tScS_view = make_tensor(tScS.data(),
        cute::conditional_return<!SdP_swapAB>(
          group<1, 4>(group<0, 2>(select<1, 3, 0, 2, 4>(flatten(tScS.layout())))),
          group<1, 3>(group<0, 3>(select<0, 2, 4, 1, 3>(flatten(tScS.layout()))))
        )
      );

      if (threadIdx.x > 9999) { // This would never run. But this is neccessary to avoid register overflow
          __syncwarp();
      }

      CUTLASS_PRAGMA_UNROLL
      for (int mma_row = 0; mma_row < size<0>(tSrS_view); mma_row++) {
        const int block_row = int(get<Row>(tScS_view(mma_row, 0)));
        const int row = block_row + base_row;

        [[maybe_unused]] const int target_index = Is_target ? (row - actual_seqlen_h) / mainloop_params.target_group_size : 0;
        [[maybe_unused]] const int target_col_limit_left = Is_target ? actual_seqlen_h + target_index * mainloop_params.target_group_size : 0;

        Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
        Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
        if constexpr (Is_in_arbitrary) {
          // Below if code introduces BRA. For the backward pass, we will apply a mask for seq_q in the non casual/local case.
          // The reason for adding the 'if' statement was to guard against gMin and gMax going out of bounds.
          // However, our design ensures that the lengths of the gMin and gMax sequences are both max_seq_q, so there's no need to worry about this issue.
          /*if (row >= actual_seqlen_q) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
            }
            continue;
          }*/
          col_max(0) = gMaxFunc(0, block_row, m_block);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(gMinFunc); ++j) {
            col_min(j)   = gMinFunc(j, block_row, m_block);
            col_max(j+1) = gMaxFunc(j+1, block_row, m_block);
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
          int block_col = int(get<Col>(tScS_view(mma_row, mma_col)));
          int col = block_col + base_col;
          if constexpr (!Is_in_causal && !Is_in_local) {
            if (col >= actual_seqlen_k || row >= actual_seqlen_q + actual_seqlen_offset) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
              continue;
            }
          } else {
            if constexpr (Is_in_context) {
              if (row < actual_seqlen_c && col < actual_seqlen_h) {
                  continue;
              }
            }
            // causal mask
            if (col >= col_limit_right(row) || row >= actual_seqlen_q + actual_seqlen_offset) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
              continue;
            }
            if constexpr (Is_in_local) {
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
          if constexpr (Is_in_arbitrary) {
            bool non_mask = false;
            non_mask = (/*col_min=*/0 <= col) && (col < col_max(0));
            if (non_mask) continue;
            CUTLASS_PRAGMA_UNROLL
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

    clear(tdKrdK);
    clear(tdVrdV);

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(
        shared_storage.barrier_KV.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.barrier_KV.wait(work_idx % 2);
    }

    auto bwd_step = [&](int m_block_valid, auto mask_fn) {
      int m_block = !Is_arbitrary ? m_block_valid : sValidBlockIds[m_block_valid];
      Tensor tSrS = partition_fragment_C(
          tiled_mma_SdP,
          select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
      pipeline_q.consumer_wait(smem_pipe_read);
      flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(
          tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
      if constexpr (Has_rab) {
        pipeline_rab.consumer_wait(smem_pipe_read);
        cute::copy(
            smem_tiled_copy_rab,
            tSsRab(_, _, _, smem_pipe_read.index()),
            tSrRab_copy_view);
        flash::convert_type_safe(tSrRab, tSrRab_accum);
      }
      Tensor tdPrdP = partition_fragment_C(
          tiled_mma_SdP,
          select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
      PipelineStatedO smem_pipe_read_do_cur =
          cute::conditional_return<Q_dO_same_stages>(
              smem_pipe_read, smem_pipe_read_do);
      pipeline_do.consumer_wait(smem_pipe_read_do_cur);
      flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/SdP_swapAB>(
          tiled_mma_SdP,
          tdPrdO(_, _, _, smem_pipe_read_do_cur.index()),
          tdPrV,
          tdPrdP);
      if constexpr (Has_rab) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) += tSrRab_accum(i);
        }
        pipeline_rab.consumer_release(smem_pipe_read);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSrS); ++i) {
        tSrS(i) *= mainloop_params.alpha;
      }

      mask_fn(tSrS, m_block);
      auto tSrS_silu = make_fragment_like(tSrS);
      silu_bwd(tSrS, tSrS_silu);

      // Convert scores from fp32 to fp16/bf16
      Tensor rP = make_tensor_like<Element>(tSrS_silu);
      flash::convert_type_safe(tSrS_silu, rP);

      if constexpr (!Slice_dQKV_Mma && Mma_dKV_is_RS) {
        Tensor tdVrP = make_tensor(
            rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
        flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(
            tiled_mma_dKV,
            tdVrP,
            tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
            tdVrdV);
      } else {
        warpgroup_wait<0>();
      }
      if constexpr (Has_drab) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdPrdP); ++i) {
          tdPrdP(i) /= param_seqlen_q;
          tdPrdP(i) *= mainloop_params.alpha;
        }
      }
      dsilu_bwd(tdPrdP, tSrS);

      if constexpr (!Mma_dKV_is_RS) {
        Tensor tPaP = smem_thr_copy_PdS.retile_S(rP);
        cute::copy(
            smem_tiled_copy_PdS,
            tPaP,
            tPsP(
                _,
                _,
                _,
                cute::conditional_return<kStages_dS == 1>(
                    _0{}, smem_pipe_read.index())));
        cutlass::arch::fence_view_async_shared();
      }
      Tensor rdS = make_tensor_like<Element>(tdPrdP);
      flash::convert_type_safe(tdPrdP, rdS);
      Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);
      // If there's double buffering on dS, we don't need to sync here.
      // Otherwise we might have WG1 writing to dS before WG2 is done reading
      // from it during MmadQ. But because both WGs have to sync at the end of
      // the loop and double buffering, this race condition is not possible.
      // This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
      // (2) dS is already read by the Mma in the previous iteration in case of
      // Mma_dKV_is_RS.
      if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
        cutlass::arch::NamedBarrier::sync(
            NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS) /*id*/);
      }
      if constexpr (Has_drab) {
        pipeline_drab.producer_acquire(smem_pipe_write_drab);
      }
      cute::copy(
          smem_tiled_copy_PdS,
          tdSadS,
          tdSsdS(
              _,
              _,
              _,
              cute::conditional_return<kStages_dS == 1>(
                  _0{}, smem_pipe_read.index())));

      if constexpr (!Slice_dQKV_Mma) {
        // Most cases take this path, except for hdim256 where we want to slice
        // to reduce register pressure
        if constexpr (Mma_dKV_is_RS) {
          // If dKV is RS, it's slightly faster to kick off dK Mma before dQ_Mma
          Tensor tdKrdS = make_tensor(
              rdS.data(),
              convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(
              tiled_mma_dKV,
              tdKrdS,
              tdKrQ(_, _, _, smem_pipe_read.index()),
              tdKrdK);
        } else {
          Tensor tdVrP =
              mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
          Tensor tdVrP_cur = tdVrP(
              _,
              _,
              _,
              cute::conditional_return<kStages_dS == 1>(
                  _0{}, smem_pipe_read.index()));
          flash::
              gemm</*zero_init=*/false, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB>(
                  tiled_mma_dKV,
                  tdVrP_cur,
                  tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
                  tdVrdV);
        }
        pipeline_do.consumer_release(smem_pipe_read_do_cur);

        // SMEM fence to make sure sdS is written before it's read by WGMMA
        cutlass::arch::fence_view_async_shared();
        if constexpr (Has_drab) {
          pipeline_drab.producer_commit(smem_pipe_write_drab);
        }
        if constexpr (dQacc_use_TMA) {
          cutlass::arch::NamedBarrier::sync(
              kNThreadsdQ + cutlass::NumThreadsPerWarp,
              static_cast<int>(BwdNamedBarriers::dQEmpty));
        } else {
          cutlass::arch::NamedBarrier::sync(
              NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS));
        }
        Tensor tdQrdQ = partition_fragment_C(
            tiled_mma_dQ,
            select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB>(
            tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        if constexpr (Mma_dKV_is_RS) {
          pipeline_q.consumer_release(smem_pipe_read);
        }
        if constexpr (dQacc_use_TMA) {
          Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
          cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
          cutlass::arch::fence_view_async_shared();
          cutlass::arch::NamedBarrier::arrive(
              kNThreadsdQ + cutlass::NumThreadsPerWarp,
              static_cast<int>(BwdNamedBarriers::dQFull));
        } else {
          Tensor tdQrdQ_atomic = recast<float4>(tdQrdQ);
          Tensor tdQgdQaccum_atomic =
              recast<float4>(tdQgdQaccum(_, _, _, m_block));
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) {
            atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
          }
        }

        if constexpr (!Mma_dKV_is_RS) {
          Tensor tdKrdS =
              mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
          Tensor tdKrdS_cur = tdKrdS(
              _,
              _,
              _,
              cute::conditional_return<kStages_dS == 1>(
                  _0{}, smem_pipe_read.index()));
          flash::
              gemm</*zero_init=*/false, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB>(
                  tiled_mma_dKV,
                  tdKrdS_cur,
                  tdKrQ(_, _, _, smem_pipe_read.index()),
                  tdKrdK);
        }

      } else { // Slice_dQKV_Mma

        Tensor tdVrP =
            mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
        Tensor tdVrP_cur = tdVrP(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        flash::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/-1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/0>(
            tiled_mma_dKV,
            tdVrP_cur,
            tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
            tdVrdV);

        cutlass::arch::fence_view_async_shared();
        if constexpr (Has_drab) {
          pipeline_drab.producer_commit(smem_pipe_write_drab);
        }
        cutlass::arch::NamedBarrier::sync(
            NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS));
        Tensor tdQrdQ = partition_fragment_C(
            tiled_mma_dQ,
            select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        flash::gemm<
            /*zero_init=*/true,
            /*wg_wait=*/-1,
            /*SwapAB=*/dQ_swapAB,
            /*M_slice=*/0>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        flash::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/1>(
            tiled_mma_dKV,
            tdVrP_cur,
            tdVrdO(_, _, _, smem_pipe_read_do_cur.index()),
            tdVrdV);
        Tensor tdQrdQ_atomic = recast<float4>(tdQrdQ);
        Tensor tdQgdQaccum_atomic =
            recast<float4>(tdQgdQaccum(_, _, _, m_block));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) {
          atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
        }

        Tensor tdKrdS =
            mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
        Tensor tdKrdS_cur = tdKrdS(
            _,
            _,
            _,
            cute::conditional_return<kStages_dS == 1>(
                _0{}, smem_pipe_read.index()));
        flash::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/1,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/0>(
            tiled_mma_dKV,
            tdKrdS_cur,
            tdKrQ(_, _, _, smem_pipe_read.index()),
            tdKrdK);
        pipeline_do.consumer_release(smem_pipe_read_do_cur);

        flash::gemm<
            /*zero_init=*/true,
            /*wg_wait=*/0,
            /*SwapAB=*/dQ_swapAB,
            /*M_slice=*/1>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
        CUTLASS_PRAGMA_UNROLL
        for (int i = size(tdQrdQ_atomic) / 2; i < size(tdQrdQ_atomic); ++i) {
          atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
        }

        flash::gemm<
            /*zero_init=*/false,
            /*wg_wait=*/0,
            /*SwapAB=*/dKV_swapAB,
            /*M_slice=*/1>(
            tiled_mma_dKV,
            tdKrdS_cur,
            tdKrQ(_, _, _, smem_pipe_read.index()),
            tdKrdK);
      }

      if constexpr (!Mma_dKV_is_RS) {
        pipeline_q.consumer_release(smem_pipe_read);
      }
      ++smem_pipe_read;
      ++smem_pipe_write_drab;
      if constexpr (!Q_dO_same_stages) {
        ++smem_pipe_read_do;
      }
    };

    if constexpr (Is_arbitrary) {
      auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<false>{}, cute::bool_constant<false>{}, cute::bool_constant<false>{}, cute::bool_constant<true>{}); };
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    if constexpr (Is_context) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        apply_mask(
            tSrS,
            m_block,
            n_block,
            cute::bool_constant<Is_causal>{},
            cute::bool_constant<Is_local>{},
            cute::bool_constant<Is_context>{},
            cute::bool_constant<false>{});
      };
      for (; m_block < m_block_context; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
      m_block = std::max(m_block, m_block_min);
    }

    if constexpr ((Is_causal || Is_local) && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        apply_mask(
            tSrS,
            m_block,
            n_block,
            cute::bool_constant<Is_causal>{},
            cute::bool_constant<Is_local>{},
            cute::bool_constant<false>{},
            cute::bool_constant<false>{});
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < std::min(m_block_max, m_block_min + m_masking_steps);
           ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    static constexpr int n_local_bottom_steps =
        (!Is_local || !SeparateMaskingIterations)
        ? 0
        : cute::ceil_div(kBlockN, kBlockM) + 1;
    auto mask_fn = [&](auto& tSrS, int m_block) {
      apply_mask(
          tSrS,
          m_block,
          n_block,
          cute::bool_constant < Is_causal && !SeparateMaskingIterations > {},
          cute::bool_constant<Is_local>{},
          cute::bool_constant<false>{},
          cute::bool_constant<false>{});
    };
    CUTLASS_PRAGMA_NO_UNROLL
    for (; m_block < m_block_max - n_local_bottom_steps; ++m_block) {
      bwd_step(m_block, mask_fn);
    }

    if constexpr (Is_local && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        apply_mask(
            tSrS,
            m_block,
            n_block,
            cute::bool_constant<false>{},
            cute::bool_constant<Is_local>{},
            cute::bool_constant<false>{},
            cute::bool_constant<false>{});
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tdVrdV); ++i) {
      tdVrdV(i) /= param_seqlen_q;
    }

    if constexpr (!Has_drab) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tdKrdK); ++i) {
        tdKrdK(i) /= param_seqlen_q;
        tdKrdK(i) *= mainloop_params.alpha;
      }
    }

    if constexpr (Q_dO_same_stages) {
      smem_pipe_read_do = smem_pipe_read;
    }
  }

  template <typename SharedStorage, typename FrgTensordKV>
  CUTLASS_DEVICE void
  mma_fp8(Params const& mainloop_params,
      MainloopPipeline pipeline_q,
      MainloopPipeline pipeline_qt,
      MainloopPipeline pipeline_rab,
      MainloopPipelinedO pipeline_do,
      MainloopPipelinedO pipeline_dot,
      MainloopPipelinedRab pipeline_drab,
      PipelineState& smem_pipe_read,
      PipelineStatedO& smem_pipe_read_do,
      PipelineStatedRab& smem_pipe_write_drab,
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
    using SmemLayoutdSfp8 = typename Ktraits::SmemLayoutdSfp8;
    using SmemLayoutdStfp8 = typename Ktraits::SmemLayoutdStfp8;
    using SmemLayoutTranspose = typename Ktraits::SmemLayoutTranspose;
    using SmemLayoutTransposeK = typename Ktraits::SmemLayoutTransposeK;
    using SmemLayoutTransposeT = typename Ktraits::SmemLayoutTransposeT;
    using SmemLayoutTransposeKt = typename Ktraits::SmemLayoutTransposeKt;
    using SmemCopyAtomdSStore = typename Ktraits::SmemCopyAtomdSStore;
    using SmemCopyAtomdSfp8Store = typename Ktraits::SmemCopyAtomdSfp8Store;
    using SmemCopyAtomRab_LDSM = typename Ktraits::SmemCopyAtomRab_LDSM;
    using TiledMmaQKdescale = typename Ktraits::TiledMmaQKdescale;
    using TiledMmadKVdescale = typename Ktraits::TiledMmadKVdescale;
    using TiledMmadQdescale = typename Ktraits::TiledMmadQdescale;

    static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sRab = make_tensor(make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sRabt = make_tensor(make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRabt{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sK_pi = cute::as_position_independent_swizzle_tensor(sK);
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
    Tensor sQt = make_tensor(make_smem_ptr(shared_storage.smem_qt.data()), SmemLayoutQt{});
    Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.smem_dot.data()), SmemLayoutdOt{});
    Tensor sKt = make_tensor(make_smem_ptr(shared_storage.smem_kt.data()), SmemLayoutKt{});
    Tensor sdS = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdS{});
    Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
    Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.smem_ds.data()), SmemLayoutPdSt{});
    Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
    Tensor sdSfp8 = make_tensor(make_smem_ptr(shared_storage.smem_ds_fp8.data()), SmemLayoutdSfp8{});
    Tensor sdSfp8_pi = cute::as_position_independent_swizzle_tensor(sdSfp8);
    Tensor sdStfp8 = make_tensor(make_smem_ptr(shared_storage.smem_ds_fp8.data()), SmemLayoutdStfp8{});
    Tensor sdStfp8_pi = cute::as_position_independent_swizzle_tensor(sdStfp8);
    Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccum{});
    ElementAccum *smem_reduce_max = shared_storage.smem_reduce_max.data();

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
    TiledMmaQKdescale tiled_mma_QKdescale;
    TiledMmadKVdescale tiled_mma_dKVdescale;
    TiledMmadQdescale tiled_mma_dQdescale;

    auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
    auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dQ = tiled_mma_dQ.get_slice(thread_idx);
    auto thr_mma_QKdescale = tiled_mma_QKdescale.get_thread_slice(thread_idx);
    auto thr_mma_dKVdescale = tiled_mma_dKVdescale.get_thread_slice(thread_idx);
    auto thr_mma_dQdescale = tiled_mma_dQdescale.get_thread_slice(thread_idx);

    auto smem_tiled_copy_dS_store = make_tiled_copy_C(SmemCopyAtomdSStore{}, tiled_mma_SdP);
    auto smem_thr_copy_dS_store = smem_tiled_copy_dS_store.get_thread_slice(thread_idx);
    auto smem_tiled_copy_dSfp8_store = make_tiled_copy_C(SmemCopyAtomdSfp8Store{}, tiled_mma_SdP);
    auto smem_thr_copy_dSfp8_store = smem_tiled_copy_dSfp8_store.get_thread_slice(thread_idx);

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
    Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdSfp8);
    Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

    Tensor tdSsdS_d = smem_thr_copy_dS_store.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_pi, sdSt_pi));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdSfp8 = smem_thr_copy_dSfp8_store.partition_D(cute::conditional_return<!SdP_swapAB>(sdSfp8_pi, sdStfp8_pi));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    // For Rab
    auto smem_tiled_copy_rab = make_tiled_copy_C(SmemCopyAtomRab{}, tiled_mma_SdP);
    auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(thread_idx);
    Tensor tSsRab = smem_thr_copy_rab.partition_S(cute::conditional_return<!SdP_swapAB>(sRab, sRabt)); // (CPY, CPY_M, CPY_N, PIPE)
    Tensor tSrRab = make_tensor<Element>(partition_shape_C(tiled_mma_SdP, cute::conditional_return<!SdP_swapAB>(select<0, 1>(TileShape_MNK{}), select<1, 0>(TileShape_MNK{}))));
    Tensor tSrRab_copy_view = smem_thr_copy_rab.retile_D(tSrRab); // (CPY, CPY_M, CPY_N)
    Tensor tSrRab_accum = make_tensor_like<ElementAccum>(tSrRab);

    auto smem_tiled_copy_rab_ldsm = make_tiled_copy_C(SmemCopyAtomRab_LDSM{}, tiled_mma_SdP);
    auto smem_thr_copy_rab_ldsm = smem_tiled_copy_rab_ldsm.get_thread_slice(thread_idx);
    Tensor tSsRab_ldsm = smem_thr_copy_rab_ldsm.partition_S(cute::conditional_return<!SdP_swapAB>(sRab, sRabt)); // (CPY, CPY_M, CPY_N, PIPE)
    Tensor tSrRab_ldsm = make_tensor<Element>(partition_shape_C(tiled_mma_SdP, cute::conditional_return<!SdP_swapAB>(select<0, 1>(TileShape_MNK{}), select<1, 0>(TileShape_MNK{}))));
    Tensor tSrRab_copy_view_ldsm = smem_thr_copy_rab_ldsm.retile_D(tSrRab_ldsm); // (CPY, CPY_M, CPY_N)
    Tensor tSrRab_accum_ldsm = make_tensor_like<ElementAccum>(tSrRab_ldsm);

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    auto [n_block, bidh, bidb] = block_coord;
    int const actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int const actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int const actual_seqlen_h = Is_target ? seqlen_traits_k.actual_seq_len_h : actual_seqlen_k;
    int const actual_seqlen_c = Is_context ? seqlen_traits_k.actual_seq_len_c : 0;

    int const actual_seqlen_offset = actual_seqlen_k - actual_seqlen_q;
    float param_seqlen_q = static_cast<float>(seqlen_traits_q.max_seq_len);
    int m_block = is_in_context ? 0 : m_block_min;

    Tensor mdQaccum = make_tensor(make_gmem_ptr(mainloop_params.ptr_dQaccum), mainloop_params.layout_dQaccum)(_, _, bidh);
    int const offset_padded = (seqlen_traits_q.cu_seq_len[bidb] + bidb * kBlockM) / kBlockM * kBlockM;
    Tensor gdQaccum = local_tile(domain_offset(make_coord(offset_padded, _0{}), mdQaccum), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)

    GmemTiledCopydQaccumAtomic gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    // Quantization Mode == 1: 1xDIM & 128x1 scale
    int const num_heads = get<2>(mainloop_params.layout_Q.shape());
    Tensor mQ_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr + seqlen_traits_q.cu_seq_len[bidb]),
                                    make_shape(num_heads, actual_seqlen_q, _1{}), make_stride(mainloop_params.descale_q_head_stride, _1{}, _0{}));
    Tensor gQ_descale = local_tile(mQ_descale(bidh, _, _), Shape<Int<kBlockM>, Int<1>>{}, make_coord(_, _0{}));

    Tensor mK_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr + seqlen_traits_k.cu_seq_len[bidb]),
                                    make_shape(num_heads, actual_seqlen_k, _1{}), make_stride(mainloop_params.descale_k_head_stride, _1{}, _0{}));
    Tensor gK_descale = local_tile(mK_descale(bidh, _, _), Shape<Int<kBlockN>, Int<1>>{}, make_coord(n_block, _0{}));

    Tensor mV_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_v_ptr + seqlen_traits_k.cu_seq_len[bidb]),
                                    make_shape(num_heads, actual_seqlen_k, _1{}), make_stride(mainloop_params.descale_v_head_stride, _1{}, _0{}));
    Tensor gV_descale = local_tile(mV_descale(bidh, _, _), Shape<Int<kBlockN>, Int<1>>{}, make_coord(n_block, _0{}));

    Tensor mdO_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_do_ptr + seqlen_traits_q.cu_seq_len[bidb]),
                                    make_shape(num_heads, actual_seqlen_q, _1{}), make_stride(mainloop_params.descale_do_head_stride, _1{}, _0{}));
    Tensor gdO_descale = local_tile(mdO_descale(bidh, _, _), Shape<Int<kBlockM>, Int<1>>{}, make_coord(_, _0{}));

    Tensor mQt_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_qt_ptr + mainloop_params.cu_seqlens_descale_qt_ptr[bidb] * mainloop_params.descale_qt_row_stride),
                                     make_shape(num_heads, Int<kHeadDim>{}, mainloop_params.cu_seqlens_descale_qt_ptr[bidb+1] - mainloop_params.cu_seqlens_descale_qt_ptr[bidb]),
                                     make_stride(mainloop_params.descale_qt_head_stride, _1{}, mainloop_params.descale_qt_row_stride));
    Tensor gQt_descale = local_tile(mQt_descale(bidh, _, _), Shape<Int<kHeadDim>, Int<1>>{}, make_coord(_, _));

    Tensor mKt_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_kt_ptr + mainloop_params.cu_seqlens_descale_kt_ptr[bidb] * mainloop_params.descale_kt_row_stride),
                                     make_shape(num_heads, Int<kHeadDim>{}, mainloop_params.cu_seqlens_descale_kt_ptr[bidb+1] - mainloop_params.cu_seqlens_descale_kt_ptr[bidb]),
                                     make_stride(mainloop_params.descale_kt_head_stride, _1{}, mainloop_params.descale_kt_row_stride));
    Tensor gKt_descale = local_tile(mKt_descale(bidh, _, _), Shape<Int<kHeadDim>, Int<1>>{}, make_coord(_, n_block));

    Tensor mdOt_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_dot_ptr + mainloop_params.cu_seqlens_descale_qt_ptr[bidb] * mainloop_params.descale_dot_row_stride),
                                      make_shape(num_heads, Int<kHeadDim>{}, mainloop_params.cu_seqlens_descale_qt_ptr[bidb+1] - mainloop_params.cu_seqlens_descale_qt_ptr[bidb]),
                                      make_stride(mainloop_params.descale_dot_head_stride, _1{}, mainloop_params.descale_dot_row_stride));
    Tensor gdOt_descale = local_tile(mdOt_descale(bidh, _, _), Shape<Int<kHeadDim>, Int<1>>{}, make_coord(_, _));

    // Quantization Mode == 2: block scale
    auto mQ_block_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr + mainloop_params.cu_seqlens_q_block_descale[bidb]),
                                make_shape(num_heads, mainloop_params.cu_seqlens_q_block_descale[bidb + 1] - mainloop_params.cu_seqlens_q_block_descale[bidb]),
                                make_stride(mainloop_params.q_block_descale_head_stride,  _1{}));
    auto gQ_block_descale = local_tile(mQ_block_descale(bidh, _), Shape<Int<1>>{}, make_coord(_)); //slice all Q

    auto mdO_block_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_do_ptr + mainloop_params.cu_seqlens_q_block_descale[bidb]),
                                make_shape(num_heads, mainloop_params.cu_seqlens_q_block_descale[bidb + 1] - mainloop_params.cu_seqlens_q_block_descale[bidb]),
                                make_stride(mainloop_params.q_block_descale_head_stride,  _1{}));
    auto gdO_block_descale = local_tile(mdO_block_descale(bidh, _), Shape<Int<1>>{}, make_coord(_)); //slice all dO

    auto mK_block_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr + mainloop_params.cu_seqlens_kv_block_descale[bidb]),
                                make_shape(num_heads, mainloop_params.cu_seqlens_kv_block_descale[bidb + 1] - mainloop_params.cu_seqlens_kv_block_descale[bidb]),
                                make_stride(mainloop_params.kv_block_descale_head_stride,  _1{}));
    auto gK_block_descale = local_tile(mK_block_descale(bidh, _), Shape<Int<1>>{}, make_coord(n_block)); //slice current K

    auto mV_block_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_v_ptr + mainloop_params.cu_seqlens_kv_block_descale[bidb]),
                                make_shape(num_heads, mainloop_params.cu_seqlens_kv_block_descale[bidb + 1] - mainloop_params.cu_seqlens_kv_block_descale[bidb]),
                                make_stride(mainloop_params.kv_block_descale_head_stride,  _1{}));
    auto gV_block_descale = local_tile(mV_block_descale(bidh, _), Shape<Int<1>>{}, make_coord(n_block)); //slice current V

    // Quantization Mode == 3: head scale
    int const num_batch = mainloop_params.num_batch;
    auto mQ_head_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr), make_shape(num_batch, num_heads), make_stride(num_heads, _1{}));
    auto mK_head_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr), make_shape(num_batch, num_heads), make_stride(num_heads, _1{}));
    auto mV_head_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_v_ptr), make_shape(num_batch, num_heads), make_stride(num_heads, _1{}));
    auto mdO_head_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_do_ptr), make_shape(num_batch, num_heads), make_stride(num_heads, _1{}));

    // Quantization Mode == 4: batch scale
    auto mQ_batch_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr), make_shape(num_batch), make_stride(_1{}));
    auto mK_batch_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr), make_shape(num_batch), make_stride(_1{}));
    auto mV_batch_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_v_ptr), make_shape(num_batch), make_stride(_1{}));
    auto mdO_batch_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_do_ptr), make_shape(num_batch), make_stride(_1{}));

    // Quantization Mode == 5: tensor scale
    auto mQ_tensor_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr), make_shape(_1{}), make_stride(_0{}));
    auto mK_tensor_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr), make_shape(_1{}), make_stride(_0{}));
    auto mV_tensor_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_v_ptr), make_shape(_1{}), make_stride(_0{}));
    auto mdO_tensor_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_do_ptr), make_shape(_1{}), make_stride(_0{}));

    // arbitrary func
    Tensor mMaxFunc = make_tensor(make_gmem_ptr(mainloop_params.func_ptr + seqlen_traits_q.offset),
        make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2 + 1>{}, actual_seqlen_q),
        make_stride(/*mainloop_params.func_head_stride*/Int<0>{}, 2 * mainloop_params.func_ids_stride, _1{}));
    Tensor mMinFunc = make_tensor(make_gmem_ptr(mainloop_params.func_ptr + seqlen_traits_q.offset + mainloop_params.func_ids_stride),
        make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2>{}, actual_seqlen_q),
        make_stride(/*mainloop_params.func_head_stride*/Int<0>{}, 2 * mainloop_params.func_ids_stride, _1{}));

    Tensor gMaxFunc = local_tile(mMaxFunc(Int<0>{}, _, _),
        make_shape(Int<kNFunc/2 + 1>{}, Int<kBlockM>{}), make_coord(Int<0>{}, _));
    Tensor gMinFunc = local_tile(mMinFunc(Int<0>{}, _, _),
        make_shape(Int<kNFunc/2>{}, Int<kBlockM>{}), make_coord(Int<0>{}, _));

    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});

    auto col_limit_right = [&](int row) {
      return std::min(
          actual_seqlen_k,
          row + 1 + mainloop_params.window_size_right
      );
    };

    auto col_limit_left = [&](int row) {
      return std::max(
          0,
          row - mainloop_params.window_size_left
      );
    };

    auto apply_mask = [&](auto& tSrS, const int m_block, const int n_block, auto is_causal_type, auto is_local_type, auto is_context_type, auto is_arbitrary_type) {
      static constexpr int Row = !SdP_swapAB ? 0 : 1;
      static constexpr int Col = !SdP_swapAB ? 1 : 0;

      constexpr bool Is_in_causal = decltype(is_causal_type)::value;
      constexpr bool Is_in_local = decltype(is_local_type)::value;
      constexpr bool Is_in_context = decltype(is_context_type)::value;
      constexpr bool Is_in_arbitrary = decltype(is_arbitrary_type)::value;

      Tensor cS = cute::make_identity_tensor(select<Row, Col>(TileShape_MNK{}));
      Tensor tScS = thread_mma_SdP.partition_C(cS);

      const int base_row = m_block * kBlockM + actual_seqlen_offset;
      const int base_col = n_block * kBlockN;

      Tensor tSrS_view = make_tensor(tSrS.data(),
        cute::conditional_return<!SdP_swapAB>(
          group<1, 4>(group<0, 2>(select<1, 3, 0, 2, 4>(flatten(tSrS.layout())))),
          group<1, 3>(group<0, 3>(select<0, 2, 4, 1, 3>(flatten(tSrS.layout()))))
        )
      );

      Tensor tScS_view = make_tensor(tScS.data(),
        cute::conditional_return<!SdP_swapAB>(
          group<1, 4>(group<0, 2>(select<1, 3, 0, 2, 4>(flatten(tScS.layout())))),
          group<1, 3>(group<0, 3>(select<0, 2, 4, 1, 3>(flatten(tScS.layout()))))
        )
      );

      if (threadIdx.x > 9999) { // This would never run. But this is neccessary to avoid register overflow
          __syncwarp();
      }

      CUTLASS_PRAGMA_UNROLL
      for (int mma_row = 0; mma_row < size<0>(tSrS_view); mma_row++) {
        const int block_row = int(get<Row>(tScS_view(mma_row, 0)));
        const int row = block_row + base_row;

        [[maybe_unused]] const int target_index = Is_target ? (row - actual_seqlen_h) / mainloop_params.target_group_size : 0;
        [[maybe_unused]] const int target_col_limit_left = Is_target ? actual_seqlen_h + target_index * mainloop_params.target_group_size : 0;

        Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
        Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
        if constexpr (Is_in_arbitrary) {
          // Below if code introduces BRA. For the backward pass, we will apply a mask for seq_q in the non casual/local case.
          // The reason for adding the 'if' statement was to guard against gMin and gMax going out of bounds.
          // However, our design ensures that the lengths of the gMin and gMax sequences are both max_seq_q, so there's no need to worry about this issue.
          /*if (row >= actual_seqlen_q) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
            }
            continue;
          }*/
          col_max(0) = gMaxFunc(0, block_row, m_block);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(gMinFunc); ++j) {
            col_min(j)   = gMinFunc(j, block_row, m_block);
            col_max(j+1) = gMaxFunc(j+1, block_row, m_block);
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
          int block_col = int(get<Col>(tScS_view(mma_row, mma_col)));
          int col = block_col + base_col;
          if constexpr (!Is_in_causal && !Is_in_local) {
            if (col >= actual_seqlen_k || row >= actual_seqlen_q + actual_seqlen_offset) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
              continue;
            }
          } else {
            if constexpr (Is_in_context) {
              if (row < actual_seqlen_c && col < actual_seqlen_h) {
                  continue;
              }
            }
            // causal mask
            if (col >= col_limit_right(row) || row >= actual_seqlen_q + actual_seqlen_offset) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
              continue;
            }
            if constexpr (Is_in_local) {
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
          if constexpr (Is_in_arbitrary) {
            bool non_mask = false;
            non_mask = (/*col_min=*/0 <= col) && (col < col_max(0));
            if (non_mask) continue;
            CUTLASS_PRAGMA_UNROLL
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

    clear(tdKrdK);
    clear(tdVrdV);

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_KV.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.barrier_KV.wait(work_idx % 2); }

    Tensor sdO_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_do.data()), SmemLayoutTranspose{}));
    Tensor sdOt_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_dot.data()), SmemLayoutTransposeT{}));

    Tensor sQ_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutTranspose{}));
    Tensor sQt_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_qt.data()), SmemLayoutTransposeT{}));

    Tensor sK_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutTransposeK{}));
    Tensor sKt_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_kt.data()), SmemLayoutTransposeKt{}));

    auto smem_transpose_dO = Ktraits::SmemTransposeFp8_64x64();
    auto do_transpose_dO = [&](int stage) {
      if (warp_group_idx == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < shape<2>(SmemLayoutTranspose{}); ++j) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < shape<1>(SmemLayoutTranspose{}); ++i) {
            smem_transpose_dO(threadIdx.x - 128, flatten(sdO_divide(_, i, j, stage)),
                        flatten(sdOt_divide(_, i, j, stage)));
          }
        }
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::ConsumerWG) /*id*/);
    };

    auto smem_transpose_Q = Ktraits::SmemTransposeFp8_64x64();
    auto do_transpose_Q = [&](int stage) {
      if (warp_group_idx == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < shape<2>(SmemLayoutTranspose{}); ++j) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < shape<1>(SmemLayoutTranspose{}); ++i) {
            smem_transpose_Q(threadIdx.x - 128, flatten(sQ_divide(_, i, j, stage)),
                        flatten(sQt_divide(_, i, j, stage)));
          }
        }
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::ConsumerWG) /*id*/);
    };

    auto smem_transpose_K = Ktraits::SmemTransposeKFp8_64x64();
    auto do_transpose_K = [&]() {
      if (warp_group_idx == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < shape<2>(SmemLayoutTransposeK{}); ++j) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < shape<1>(SmemLayoutTransposeK{}); ++i) {
            smem_transpose_K(threadIdx.x - 128, flatten(sK_divide(_, i, j)),
                        flatten(sKt_divide(_, i, j)));
          }
        }
      }
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::ConsumerWG) /*id*/);
    };

    auto block_reduce_max = [&](auto& tensor) {
      ElementAccum thread_max = EPSILON;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tensor); ++i) {
        thread_max = fmax(thread_max, fabsf(tensor(i)));
      }
      __syncwarp();
      ElementAccum warp_max = thread_max;
      CUTLASS_PRAGMA_UNROLL
      for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmax(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, offset));
      }

      if (thread_idx % 32 == 0) {
        smem_reduce_max[thread_idx / 32] = warp_max;
      }
      cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::ReduceMaxWG) /*id*/);
      if (thread_idx / 32 == 0) {
        ElementAccum val = (thread_idx % 32 < NumMmaWarps) ? smem_reduce_max[thread_idx % 32] : 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset > 0; offset >>= 1) {
          val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (thread_idx % 32 == 0) {
          smem_reduce_max[0] = val;
        }
      }
      cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::ReduceMaxWG) /*id*/);
      return smem_reduce_max[0] / std::numeric_limits<Element>::max();
    };

    if constexpr (Quant_mode != 1) {
      do_transpose_K(); // TODO: move to load_fp8
    }

    auto bwd_step = [&](int m_block_valid, auto mask_fn) {
      int m_block = Is_arbitrary ? sValidBlockIds[m_block_valid] : m_block_valid;
      Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
      pipeline_q.consumer_wait(smem_pipe_read);
      flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);

      Tensor tSrK_descale = thr_mma_QKdescale.partition_fragment_A(gK_descale);
      Tensor tSrQ_descale = thr_mma_QKdescale.partition_fragment_B(gQ_descale(_, _, m_block));
      Tensor tSrS_descale = partition_fragment_C(tiled_mma_QKdescale, Shape<Int<kBlockN>, Int<kBlockM>>{});
      auto smem_tiled_copy_K_descale = make_tiled_copy_A(typename Ktraits::DescaleCopyAtom{}, tiled_mma_QKdescale);
      auto smem_thr_copy_K_descale = smem_tiled_copy_K_descale.get_thread_slice(thread_idx);
      Tensor tSgK_descale = smem_thr_copy_K_descale.partition_S(gK_descale);
      Tensor tSrK_descale_view = smem_thr_copy_K_descale.retile_D(tSrK_descale);

      auto smem_tiled_copy_Q_descale = make_tiled_copy_B(typename Ktraits::DescaleCopyAtom{}, tiled_mma_QKdescale);
      auto smem_thr_copy_Q_descale = smem_tiled_copy_Q_descale.get_thread_slice(thread_idx);
      Tensor tSgQ_descale = smem_thr_copy_Q_descale.partition_S(gQ_descale(_, _, m_block));
      Tensor tSrQ_descale_view = smem_thr_copy_Q_descale.retile_D(tSrQ_descale);
      if constexpr (Quant_mode == 1) {
        cute::copy(smem_tiled_copy_K_descale, tSgK_descale, tSrK_descale_view);
        cute::copy(smem_tiled_copy_Q_descale, tSgQ_descale, tSrQ_descale_view);
        cute::gemm(tiled_mma_QKdescale, tSrK_descale, tSrQ_descale, tSrS_descale);
      }

      Tensor tSrS_descale_view = make_tensor(tSrS_descale.data(), composition(tSrS_descale.layout(), Layout<Shape<_2, _2, Int<kBlockM/8>>, Stride<_2, _1, _4>>{}));

      if constexpr (Has_rab) {
        pipeline_rab.consumer_wait(smem_pipe_read);
        cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, smem_pipe_read.index()), tSrRab_copy_view);
        flash::convert_type_safe(tSrRab, tSrRab_accum);
      }
      Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
      PipelineStatedO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
      pipeline_do.consumer_wait(smem_pipe_read_do_cur);

      flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);
      if constexpr (Quant_mode == 1) {
        pipeline_q.consumer_release(smem_pipe_read);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSrS); i++) {
        if constexpr (Quant_mode == 1) {
          tSrS(i) *= tSrS_descale_view(i);
        } else if constexpr (Quant_mode == 2) {
          tSrS(i) *= gQ_block_descale(m_block) * gK_block_descale(0);
        } else if constexpr (Quant_mode == 3) {
          tSrS(i) *= mQ_head_descale(bidb, bidh) * mK_head_descale(bidb, bidh);
        } else if constexpr (Quant_mode == 4) {
          tSrS(i) *= mQ_batch_descale(bidb) * mK_batch_descale(bidb);
        } else if constexpr (Quant_mode == 5) {
          tSrS(i) *= mQ_tensor_descale(0) * mK_tensor_descale(0);
        }
      }
      if constexpr (Has_rab) {
        pipeline_rab.consumer_release(smem_pipe_read);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) += tSrRab_accum(i);
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSrS); ++i) {
        tSrS(i) *= mainloop_params.alpha;
      }

      mask_fn(tSrS, m_block);
      auto tSrS_silu = make_fragment_like(tSrS);
      silu_bwd(tSrS, tSrS_silu);

      ElementAccum P_block_max = block_reduce_max(tSrS_silu);
      if constexpr (Quant_mode >= 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS_silu); ++i) {
          tSrS_silu(i) = tSrS_silu(i) / P_block_max;
        }
      }

      // Convert scores from fp32 to fp8
      Tensor rP = make_tensor_like<Element>(tSrS_silu);
      flash::convert_type_safe(tSrS_silu, rP);
      Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
      permute_regs_C_to_A<Quant_mode != 1>(tdVrP);
      if constexpr (Quant_mode != 1) {
        do_transpose_dO(smem_pipe_read_do_cur.index()); // TODO: move to load_fp8
      } else {
        pipeline_dot.consumer_wait(smem_pipe_read_do_cur);
      }
      Tensor tdVrdV_tmp = make_tensor_like(tdVrdV);
      if constexpr (Quant_mode >= 1) {
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma_dKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV_tmp);
      } else {
        flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_dKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
      }

      Tensor tdPrV_descale = thr_mma_QKdescale.partition_fragment_A(gV_descale);
      Tensor tdPrdO_descale = thr_mma_QKdescale.partition_fragment_B(gdO_descale(_, _, m_block));
      Tensor tdPrdP_descale = partition_fragment_C(tiled_mma_QKdescale, Shape<Int<kBlockN>, Int<kBlockM>>{});
      auto smem_tiled_copy_V_descale = make_tiled_copy_A(typename Ktraits::DescaleCopyAtom{}, tiled_mma_QKdescale);
      auto smem_thr_copy_V_descale = smem_tiled_copy_V_descale.get_thread_slice(thread_idx);
      Tensor tdPgV_descale = smem_thr_copy_V_descale.partition_S(gV_descale);
      Tensor tdPrV_descale_view = smem_thr_copy_V_descale.retile_D(tdPrV_descale);

      auto smem_tiled_copy_dO_descale = make_tiled_copy_B(typename Ktraits::DescaleCopyAtom{}, tiled_mma_QKdescale);
      auto smem_thr_copy_dO_descale = smem_tiled_copy_dO_descale.get_thread_slice(thread_idx);
      Tensor tdPgdO_descale = smem_thr_copy_dO_descale.partition_S(gdO_descale(_, _, m_block));
      Tensor tdPrdO_descale_view = smem_thr_copy_dO_descale.retile_D(tdPrdO_descale);

      if constexpr (Quant_mode == 1) {
        cute::copy(smem_tiled_copy_V_descale, tdPgV_descale, tdPrV_descale_view);
        cute::copy(smem_tiled_copy_dO_descale, tdPgdO_descale, tdPrdO_descale_view);
        cute::gemm(tiled_mma_QKdescale, tdPrV_descale, tdPrdO_descale, tdPrdP_descale);
        Tensor tdPrdP_descale_view = make_tensor(tdPrdP_descale.data(), composition(tdPrdP_descale.layout(), Layout<Shape<_2, _2, Int<kBlockM/8>>, Stride<_2, _1, _4>>{}));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdPrdP); ++i) {
          tdPrdP(i) *= tdPrdP_descale_view(i);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdPrdP); ++i) {
          if constexpr (Quant_mode == 2) {
            tdPrdP(i) *= gdO_block_descale(m_block) * gV_block_descale(0);
          } else if constexpr (Quant_mode == 3) {
            tdPrdP(i) *= mdO_head_descale(bidb, bidh) * mV_head_descale(bidb, bidh);
          } else if constexpr (Quant_mode == 4) {
            tdPrdP(i) *= mdO_batch_descale(bidb) * mV_batch_descale(bidb);
          } else if constexpr (Quant_mode == 5) {
            tdPrdP(i) *= mdO_tensor_descale(0) * mV_tensor_descale(0);
          }
        }
      }

      Tensor tdVrP_descale = thr_mma_dKVdescale.partition_fragment_A(gK_descale);
      Tensor tdVrdOt_descale = thr_mma_dKVdescale.partition_fragment_B(gdOt_descale(_, _, 0, m_block / kMBlock_shared));
      Tensor tdVrdV_descale = partition_fragment_C(tiled_mma_dKVdescale, Shape<Int<kBlockN>, Int<kHeadDim>>{});

      auto smem_tiled_copy_dOt_descale = make_tiled_copy_B(typename Ktraits::DescaleCopyAtom{}, tiled_mma_dKVdescale);
      auto smem_thr_copy_dOt_descale = smem_tiled_copy_dOt_descale.get_thread_slice(thread_idx);
      Tensor tdVgdOt_descale = smem_thr_copy_dOt_descale.partition_S(gdOt_descale(_, _, 0, m_block / kMBlock_shared));
      Tensor tdVrdOt_descale_view = smem_thr_copy_dOt_descale.retile_D(tdVrdOt_descale);

      if constexpr (Quant_mode == 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdVrP_descale); ++i) {
          tdVrP_descale(i) = P_block_max;
        }
        cute::copy(smem_tiled_copy_dOt_descale, tdVgdOt_descale, tdVrdOt_descale_view);
        cute::gemm(tiled_mma_dKVdescale, tdVrP_descale, tdVrdOt_descale, tdVrdV_descale);
        Tensor tdVrdV_descale_view = make_tensor(tdVrdV_descale.data(), composition(tdVrdV_descale.layout(), Layout<Shape<_2, _2, Int<kBlockM/8>>, Stride<_2, _1, _4>>{}));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdVrdV); ++i) {
          tdVrdV(i) += tdVrdV_tmp(i) * tdVrdV_descale_view(i);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdVrdV); ++i) {
          if constexpr (Quant_mode == 2) {
            tdVrdV(i) += tdVrdV_tmp(i) * P_block_max * gdO_block_descale(m_block);
          } else if constexpr (Quant_mode == 3) {
            tdVrdV(i) += tdVrdV_tmp(i) * P_block_max * mdO_head_descale(bidb, bidh);
          } else if constexpr (Quant_mode == 4) {
            tdVrdV(i) += tdVrdV_tmp(i) * P_block_max * mdO_batch_descale(bidb);
          } else if constexpr (Quant_mode == 5) {
            tdVrdV(i) += tdVrdV_tmp(i) * P_block_max * mdO_tensor_descale(0);
          }
        }
      }

      pipeline_do.consumer_release(smem_pipe_read_do_cur);  // release dO, after dOt gemm and trans over
      if constexpr (Quant_mode == 1) {
        pipeline_dot.consumer_release(smem_pipe_read_do_cur);
      }
      if constexpr (Has_drab) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdPrdP); ++i) {
          tdPrdP(i) /= param_seqlen_q;
          tdPrdP(i) *= mainloop_params.alpha;
        }
      }
      dsilu_bwd(tdPrdP, tSrS);

      Tensor rdS_fp16 = make_tensor_like<ElementRab>(tdPrdP);
      flash::convert_type_safe(tdPrdP, rdS_fp16);
      ElementAccum dS_block_max = block_reduce_max(tdPrdP);
      if constexpr (Quant_mode >= 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdPrdP); ++i) {
          tdPrdP(i) = tdPrdP(i) / dS_block_max;
        }
      }

      Tensor rdS_fp8 = make_tensor_like<Element>(tdPrdP);
      flash::convert_type_safe(tdPrdP, rdS_fp8);
      // If there's double buffering on dS, we don't need to sync here.
      // Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
      // But because both WGs have to sync at the end of the loop and double buffering,
      // this race condition is not possible.
      // This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
      // (2) dS is already read by the Mma in the previous iteration in case of Mma_dKV_is_RS.
      if constexpr (kStages_dS == 1) {
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS) /*id*/);
      }
      // For hdim 64, It's faster to write to smem_dS first before the dV gemm
      if constexpr (Has_drab) {
        pipeline_drab.producer_acquire(smem_pipe_write_drab);
      }

      // Store dS to smem
      Tensor tdSadS_fp16 = smem_thr_copy_dS_store.retile_S(rdS_fp16);     // ((Atom,AtomNum), MMA_N, MMA_N)
      cute::copy(smem_tiled_copy_dS_store, tdSadS_fp16, tdSsdS_d(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()))); // have error

      Tensor tdSadS_fp8 = smem_thr_copy_dSfp8_store.retile_S(rdS_fp8);     // ((Atom,AtomNum), MMA_N, MMA_N)
      cute::copy(smem_tiled_copy_dSfp8_store, tdSadS_fp8, tdSsdSfp8(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()))); // have error

      // Most cases take this path, except for hdim256 where we want to slice to reduce register pressure
      Tensor tdKrdS = make_tensor(rdS_fp8.data(), convert_layout_acc_Aregs_fp8(tdPrdP.layout()));
      permute_regs_C_to_A<Quant_mode != 1>(tdKrdS);
      if constexpr (Quant_mode == 1) {
        pipeline_qt.consumer_wait(smem_pipe_read);
      } else {
        do_transpose_Q(smem_pipe_read.index()); // TODO: move to load_fp8
        pipeline_q.consumer_release(smem_pipe_read); // release Q, after QK gemm and trans over
      }
      Tensor tdKrdK_tmp = make_tensor_like(tdKrdK);
      if constexpr (Quant_mode >= 1) {
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma_dKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK_tmp);
      } else {
        flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_dKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
      }

      if constexpr (Quant_mode == 1) {
        pipeline_qt.consumer_release(smem_pipe_read);
      }

      Tensor tdKrP_descale = thr_mma_dKVdescale.partition_fragment_A(gK_descale);
      Tensor tdKrQt_descale = thr_mma_dKVdescale.partition_fragment_B(gQt_descale(_, _, 0, m_block / kMBlock_shared));
      Tensor tdKrdK_descale = partition_fragment_C(tiled_mma_dKVdescale, Shape<Int<kBlockN>, Int<kHeadDim>>{});
      auto smem_tiled_copy_Qt_descale = make_tiled_copy_B(typename Ktraits::DescaleCopyAtom{}, tiled_mma_dKVdescale);
      auto smem_thr_copy_Qt_descale = smem_tiled_copy_Qt_descale.get_thread_slice(thread_idx);
      Tensor tdKgQt_descale = smem_thr_copy_Qt_descale.partition_S(gQt_descale(_, _, 0, m_block / kMBlock_shared));
      Tensor tdKrQt_descale_view = smem_thr_copy_Qt_descale.retile_D(tdKrQt_descale);

      if constexpr (Quant_mode == 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdKrP_descale); ++i) {
            tdKrP_descale(i) = dS_block_max;
        }
        cute::copy(smem_tiled_copy_Qt_descale, tdKgQt_descale, tdKrQt_descale_view);
        cute::gemm(tiled_mma_dKVdescale, tdKrP_descale, tdKrQt_descale, tdKrdK_descale);
        Tensor tdKrdK_descale_view = make_tensor(tdKrdK_descale.data(), composition(tdKrdK_descale.layout(), Layout<Shape<_2, _2, Int<kBlockM/8>>, Stride<_2, _1, _4>>{}));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdKrdK); ++i) {
          tdKrdK(i) += tdKrdK_tmp(i) * tdKrdK_descale_view(i);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdKrdK); ++i) {
          if constexpr (Quant_mode == 2) {
            tdKrdK(i) += tdKrdK_tmp(i) * dS_block_max * gQ_block_descale(m_block);
          } else if constexpr (Quant_mode == 3) {
            tdKrdK(i) += tdKrdK_tmp(i) * dS_block_max * mQ_head_descale(bidb, bidh);
          } else if constexpr (Quant_mode == 4) {
            tdKrdK(i) += tdKrdK_tmp(i) * dS_block_max * mQ_batch_descale(bidb);
          } else if constexpr (Quant_mode == 5) {
            tdKrdK(i) += tdKrdK_tmp(i) * dS_block_max * mQ_tensor_descale(0);
          }
        }
      }

      // SMEM fence to make sure sdS is written before it's read by WGMMA
      cutlass::arch::fence_view_async_shared();
      if constexpr (Has_drab) {
        pipeline_drab.producer_commit(smem_pipe_write_drab);
      }
      if constexpr (dQacc_use_TMA) {
        cutlass::arch::NamedBarrier::sync(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
      } else {
        cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS) /*id*/);
      }

      Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
      flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB>(tiled_mma_dQ, tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())), tdQrK, tdQrdQ);

      Tensor tdQrP_descale = thr_mma_dQdescale.partition_fragment_A(gQ_descale(_, _, m_block));
      Tensor tdQrKt_descale = thr_mma_dQdescale.partition_fragment_B(gKt_descale(_, _, 0));
      Tensor tdQrdQ_descale = partition_fragment_C(tiled_mma_dQdescale, Shape<Int<kBlockM>, Int<kHeadDim>>{});
      auto smem_tiled_copy_Kt_descale = make_tiled_copy_B(typename Ktraits::DescaleCopyAtom{}, tiled_mma_dQdescale);
      auto smem_thr_copy_Kt_descale = smem_tiled_copy_Kt_descale.get_thread_slice(thread_idx);
      Tensor tdQgKt_descale = smem_thr_copy_Kt_descale.partition_S(gKt_descale(_, _, 0));
      Tensor tdQrKt_descale_view = smem_thr_copy_Kt_descale.retile_D(tdQrKt_descale);

      if constexpr (Quant_mode == 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdQrP_descale); ++i) {
          tdQrP_descale(i) = dS_block_max;
        }
        cute::copy(smem_tiled_copy_Kt_descale, tdQgKt_descale, tdQrKt_descale_view);
        cute::gemm(tiled_mma_dQdescale, tdQrP_descale, tdQrKt_descale, tdQrdQ_descale);
        Tensor tdQrdQ_descale_view = make_tensor(tdQrdQ_descale.data(), composition(tdQrdQ_descale.layout(), Layout<Shape<_2, _2, Int<kBlockM/8>>, Stride<_2, _1, _4>>{}));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdQrdQ); ++i) {
          tdQrdQ(i) *= tdQrdQ_descale_view(i);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdQrdQ); ++i) {
          if constexpr (Quant_mode == 2) {
            tdQrdQ(i) *= dS_block_max * gK_block_descale(0);
          } else if constexpr (Quant_mode == 3) {
            tdQrdQ(i) *= dS_block_max * mK_head_descale(bidb, bidh);
          } else if constexpr (Quant_mode == 4) {
            tdQrdQ(i) *= dS_block_max * mK_batch_descale(bidb);
          } else if constexpr (Quant_mode == 5) {
            tdQrdQ(i) *= dS_block_max * mK_tensor_descale(0);
          }
        }
      }

      if constexpr (dQacc_use_TMA) {
        Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);        // ((Atom,AtomNum), MMA_M, MMA_N)
        cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQFull) /*id*/);  // sdQ full, to be written to gmem
      } else {
        Tensor tdQrdQ_atomic = recast<float4>(tdQrdQ);
        Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
      }

      ++smem_pipe_read;
      if constexpr (Has_drab) { ++smem_pipe_write_drab; }
      if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
    };

    if constexpr (Is_arbitrary) {
      auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<false>{}, cute::bool_constant<false>{}, cute::bool_constant<false>{}, cute::bool_constant<true>{}); };
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    if constexpr (Is_context) {
      auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<Is_causal>{}, cute::bool_constant<Is_local>{}, cute::bool_constant<Is_context>{}, cute::bool_constant<false>{}); };
      for (; m_block < m_block_context; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
      m_block = std::max(m_block, m_block_min);
    }

    if constexpr ((Is_causal || Is_local) && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<Is_causal>{}, cute::bool_constant<Is_local>{}, cute::bool_constant<false>{}, cute::bool_constant<false>{}); };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < std::min(m_block_max, m_block_min + m_masking_steps); ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    static constexpr int n_local_bottom_steps = (!Is_local || !SeparateMaskingIterations) ? 0 : cute::ceil_div(kBlockN, kBlockM) + 1;
    auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<Is_causal && !SeparateMaskingIterations>{}, cute::bool_constant<Is_local>{}, cute::bool_constant<false>{}, cute::bool_constant<false>{}); };
    CUTLASS_PRAGMA_NO_UNROLL
    for (; m_block < m_block_max - n_local_bottom_steps; ++m_block) {
      bwd_step(m_block, mask_fn);
    }

    if constexpr (Is_local && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) { apply_mask(tSrS, m_block, n_block, cute::bool_constant<false>{}, cute::bool_constant<Is_local>{}, cute::bool_constant<false>{}, cute::bool_constant<false>{}); };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tdVrdV); ++i) {
      tdVrdV(i) /= param_seqlen_q;
    }

    if constexpr (!Has_drab) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tdKrdK); ++i) {
        tdKrdK(i) /= param_seqlen_q;
        tdKrdK(i) *= mainloop_params.alpha;
      }
    }

    if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
  }

};

} // namespace flash
