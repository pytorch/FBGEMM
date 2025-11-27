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
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "seq_len.h"
#include "utils.h"

namespace flash {

using namespace cute;

// 4 warps
struct SmemTransposeFp8_64x64 {
  using Element = cutlass::float_e4m3_t;

  using ldsm_thread_shape = Shape<_4, _1, _8, _4>;
  using ldsm_value_shape = Shape<_2, _8, _2, _1>;
  using ldsm_value_stride = Stride<_2, _4, _1, _0>;
  using TiledCopyLDSM = decltype(make_tiled_copy(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
      Layout<ldsm_thread_shape>{},
      Layout<ldsm_value_shape, ldsm_value_stride>{}));
  TiledCopyLDSM tiled_copy_ldsm;

  using stsm_thread_shape = Shape<_4, _1, _8, _4>;
#ifndef NO_FP8_COLUMN_PERMUTE
  using stsm_value_shape = Shape<_4, _4, _1, _2>;
  using stsm_value_stride = Stride<_1, _8, _0, _4>;
#else
  using stsm_value_shape = Shape<_4, _4, _2, _1>;
  using stsm_value_stride = Stride<_1, _8, _4, _0>;
#endif

  using TiledCopySTSM = decltype(make_tiled_copy(
      Copy_Atom<SM90_U32x4_STSM_N, Element>{},
      Layout<stsm_thread_shape>{},
      Layout<stsm_value_shape, stsm_value_stride>{}));
  TiledCopySTSM tiled_copy_stsm;

  template <class SmemTensor, class SmemTensorOut>
  CUTLASS_DEVICE void operator()(SmemTensor&& s_in, SmemTensorOut&& s_out) {
    using namespace cute;

    auto tid = threadIdx.x;
    auto thr_copy_ldsm = tiled_copy_ldsm.get_thread_slice(tid);
    auto thr_copy_stsm = tiled_copy_stsm.get_thread_slice(tid);

    auto tXsX = thr_copy_ldsm.partition_S(s_in);
    auto tXrX = make_tensor<Element>(shape(tXsX));
    auto tXsX_out = thr_copy_stsm.partition_D(s_out);

    cute::copy(tiled_copy_ldsm, tXsX, tXrX);

    auto data = tXrX.data();
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size(tXrX); n += 8) {
      uint32_t* data_32bit = reinterpret_cast<uint32_t*>(&data[n]);
      auto upper = data_32bit[0];
      auto lower = data_32bit[1];
      data_32bit[0] = __byte_perm(upper, lower, 0x6420);
      data_32bit[1] = __byte_perm(upper, lower, 0x7531);
    }

    cute::copy(tiled_copy_stsm, tXrX, tXsX_out);
  }
};

template <typename Ktraits, typename Seqlen_traits>
struct CollectiveMainloopFwd {
  using index_t = typename Ktraits::index_t;
  using Element = typename Ktraits::Element;
  using ElementAccum = typename Ktraits::ElementAccum;
  using ElementRab = cute::conditional_t<
      cutlass::sizeof_bits_v<Element> == 8,
      typename Ktraits::OutputType,
      typename Ktraits::Element>;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using TileShape_MNK_PV = typename Ktraits::TileShape_MNK_PV;
  static constexpr int kNBlock_shared = Ktraits::kNBlock_shared;

  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr bool Is_causal = Ktraits::Is_causal;
  static constexpr bool Is_target = Ktraits::Is_target;
  static constexpr bool Is_context = Ktraits::Is_context;
  static constexpr bool Is_local = Ktraits::Is_local;
  static constexpr bool Is_arbitrary = Ktraits::Is_arbitrary;
  static constexpr int  kNFunc = Ktraits::kNFunc;
  static constexpr bool Has_rab = Ktraits::Has_rab;

  static constexpr int kStages = Ktraits::kStages;
  static constexpr int kHeadDim = Ktraits::kHeadDim;
  static constexpr int Quant_mode = Ktraits::Quant_mode;

  using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t>;  // (seqlen, d, head)
  using StrideQKV = cute::Stride<int64_t, _1, int64_t>;
  using StrideV = std::conditional_t<Quant_mode == 1, cute::Stride<int64_t, _1, int64_t>, cute::Stride<_1, int64_t, int64_t>>;

  using ShapeRab = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
  using StrideRab = cute::Stride<int64_t, _1, int64_t, int64_t>;

  using LayoutQKV = cute::Layout<ShapeQKV, StrideQKV>;
  using LayoutV = cute::Layout<ShapeQKV, StrideV>;
  using LayoutRab = cute::Layout<ShapeRab, StrideRab>;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyRab = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV =
      decltype(cutlass::gemm::collective::detail::
                   sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutRab = typename Ktraits::SmemLayoutRab;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
  using SmemLayoutVtMma = typename Ktraits::SmemLayoutVtMma;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKV{},
          StrideQKV{}),
      SmemLayoutQ{},
      select<0, 2>(TileShape_MNK{}),
      _1{})); // no mcast for Q

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
          ShapeQKV{},
          StrideQKV{}),
      take<0, 2>(SmemLayoutK{}),
      select<1, 2>(TileShape_MNK{}),
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  // TMA_V may differ from TMA_K for fp8 kernel (e.g. swizzling mode)
  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKV{},
          StrideV{}),
      take<0, 2>(SmemLayoutVt{}),
      select<1, 2>(TileShape_MNK_PV{}),
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  static constexpr int NumMmaThreads = Ktraits::NumMmaThreads;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using MainloopPipelineNoTMA = typename Ktraits::MainloopPipelineNoTMA;
  using MainloopPipelineVt = std::conditional_t<Quant_mode == 1, MainloopPipeline, MainloopPipelineNoTMA>;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  // Set the bytes transferred in this TMA transaction (may involve multiple
  // issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
      size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesRab = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutRab{})) * cutlass::sizeof_bits_v<ElementRab> /
      8);
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);

  static constexpr bool UseSchedulerBarrier = true;
  static constexpr bool IntraWGOverlap =
      cutlass::sizeof_bits_v<Element> == 16 ? kHeadDim < 128 : true;

  // Host side kernel arguments
  struct Arguments {
    Element const* ptr_Q;
    LayoutQKV layout_Q;
    ElementRab const* ptr_Rab;
    LayoutRab layout_Rab;
    Element const* ptr_K;
    LayoutQKV layout_K;
    Element const* ptr_V;
    LayoutV layout_V;
    int const num_batch;
    float const *descale_q_ptr;
    float const *descale_k_ptr;
    float const *descale_v_ptr;
    float const *descale_vt_ptr;
    const index_t descale_q_head_stride;
    const index_t descale_k_head_stride;
    const index_t descale_v_head_stride;
    const index_t descale_vt_head_stride;
    const index_t descale_vt_row_stride;
    int const *func_ptr;
    const index_t func_ids_stride;
    const int window_size_left;
    const int window_size_right;
    const int target_group_size;
    const float alpha;
    int const* cu_seqlens_vt_descale;
    int const* cu_seqlens_q_block_descale;
    int const* cu_seqlens_kv_block_descale;
    const index_t q_block_descale_head_stride;
    const index_t kv_block_descale_head_stride;
  };

  // Device side kernel params
  struct Params {
    LayoutQKV layout_Q;
    LayoutRab layout_Rab;
    LayoutQKV layout_K;
    LayoutV layout_V;
    cutlass::FastDivmod qhead_per_khead_divmod;
    cutlass::FastDivmod qhead_per_rabhead_divmod;
    TMA_Q tma_load_Q;
    TMA_Rab tma_load_Rab;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    int const num_batch;
    float const* descale_q_ptr;
    float const* descale_k_ptr;
    float const* descale_v_ptr;
    float const* descale_vt_ptr;
    const index_t descale_q_head_stride;
    const index_t descale_k_head_stride;
    const index_t descale_v_head_stride;
    const index_t descale_vt_head_stride;
    const index_t descale_vt_row_stride;
    int const* func_ptr;
    const index_t func_ids_stride;
    const int window_size_left;
    const int window_size_right;
    const int target_group_size;
    const float alpha;
    int const* cu_seqlens_vt_descale;
    int const* cu_seqlens_q_block_descale;
    int const* cu_seqlens_kv_block_descale;
    const index_t q_block_descale_head_stride;
    const index_t kv_block_descale_head_stride;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.layout_Q);
    TMA_Q tma_load_Q = make_tma_copy(
        GmemTiledCopyQ{},
        mQ,
        SmemLayoutQ{},
        select<0, 2>(TileShape_MNK{}),
        _1{}); // no mcast for Q
    Tensor mRab = make_tensor(make_gmem_ptr(args.ptr_Rab), args.layout_Rab);
    TMA_Rab tma_load_Rab = make_tma_copy(
        GmemTiledCopyRab{},
        mRab,
        SmemLayoutRab{}(_, _, _0{}),
        select<0, 1>(TileShape_MNK{}),
        _1{}); // no mcast for Rab
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.layout_K);
    TMA_K tma_load_K = make_tma_copy(
        GmemTiledCopyKV{},
        mK,
        SmemLayoutK{}(_, _, _0{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.layout_V);
    TMA_V tma_load_V = make_tma_copy(
        GmemTiledCopyKV{},
        mV,
        SmemLayoutVt{}(_, _, _0{}),
        select<1, 2>(TileShape_MNK_PV{}),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
    return {
        args.layout_Q,
        args.layout_Rab,
        args.layout_K,
        args.layout_V,
        cutlass::FastDivmod(
            cute::ceil_div(
                get<2>(args.layout_Q.shape()), get<2>(args.layout_K.shape()))),
        cutlass::FastDivmod(
            cute::ceil_div(
                get<2>(args.layout_Q.shape()),
                get<2>(args.layout_Rab.shape()))),
        tma_load_Q,
        tma_load_Rab,
        tma_load_K,
        tma_load_V,
        args.num_batch,
        args.descale_q_ptr,
        args.descale_k_ptr,
        args.descale_v_ptr,
        args.descale_vt_ptr,
        args.descale_q_head_stride,
        args.descale_k_head_stride,
        args.descale_v_head_stride,
        args.descale_vt_head_stride,
        args.descale_vt_row_stride,
        args.func_ptr,
        args.func_ids_stride,
        args.window_size_left,
        args.window_size_right,
        args.target_group_size,
        args.alpha,
        args.cu_seqlens_vt_descale,
        args.cu_seqlens_q_block_descale,
        args.cu_seqlens_kv_block_descale,
        args.q_block_descale_head_stride,
        args.kv_block_descale_head_stride};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
  /// performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_V.get_tma_descriptor());
    if (Has_rab) {
      cute::prefetch_tma_descriptor(
          mainloop_params.tma_load_Rab.get_tma_descriptor());
    }
  }

  template <typename Scheduler, typename SharedStorage>
  CUTLASS_DEVICE void load(
      Params const& mainloop_params,
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_rab,
      MainloopPipeline pipeline_v,
      PipelineState& smem_pipe_write_k,
      PipelineState& smem_pipe_write_rab,
      PipelineState& smem_pipe_write_v,
      int n_block_max,
      int n_block_min,
      int n_masking_steps,
      bool is_jump,
      int n_block_history,
      SharedStorage& shared_storage,
      Scheduler& scheduler,
      typename Scheduler::Params const& scheduler_params,
      typename Scheduler::WorkTileInfo& work_tile_info,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      int work_idx,
      const Seqlen_traits& seqlen_traits_q,
      const Seqlen_traits& seqlen_traits_k) {
    Tensor sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sRab = make_tensor(
        make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sVt =
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});

    auto [m_block, bidh, bidb] = block_coord;
    int actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int offset_rab = actual_seqlen_k - actual_seqlen_q;

    int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);
    int bidh_rab = mainloop_params.qhead_per_rabhead_divmod.divide(bidh);

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(
        mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(
        mainloop_params.layout_K.shape());
    Tensor mRab = mainloop_params.tma_load_Rab.get_tma_tensor(
        mainloop_params.layout_Rab.shape())(_, _, bidh_rab, bidb);
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(
        mainloop_params.layout_V.shape());

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {
        block_rank_in_cluster % cluster_shape_x,
        block_rank_in_cluster / cluster_shape_x};
    Tensor gQ = seqlen_traits_q.get_local_tile_tensor(
        mQ, select<0, 2>(TileShape_MNK{}), bidh, bidb)(_, _, m_block);
    Tensor gK = seqlen_traits_k.get_local_tile_tensor(
        mK, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);
    Tensor gV = seqlen_traits_k.get_local_tile_tensorT(
        mV, select<1, 2>(TileShape_MNK_PV{}), bidh_kv, bidb);
    Tensor gRab = local_tile(
        domain_offset(make_coord(offset_rab, _0{}), mRab),
        select<0, 1>(TileShape_MNK{}),
        make_coord(m_block, _));

    auto block_tma_Q = mainloop_params.tma_load_Q.get_slice(_0{});
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));

    auto block_tma_K = mainloop_params.tma_load_K.get_slice(block_rank_in_cluster);
    Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK));
    Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK));

    auto block_tma_V = mainloop_params.tma_load_V.get_slice(block_rank_in_cluster);
    Tensor tVgV = group_modes<0, 3>(block_tma_V.partition_S(gV));
    Tensor tVsV = group_modes<0, 3>(block_tma_V.partition_D(sVt));

    auto block_tma_Rab = mainloop_params.tma_load_Rab.get_slice(_0{});
    Tensor tRabgRab = group_modes<0, 3>(block_tma_Rab.partition_S(gRab));
    Tensor tRabsRab = group_modes<0, 3>(block_tma_Rab.partition_D(sRab));

    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_kv |=
            (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
      }
    }

    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});

    int n_block = n_block_max - 1;
    n_block = !Is_arbitrary ? n_block : sValidBlockIds[n_block];

    int lane_predicate = cute::elect_one_sync();
    if constexpr (IntraWGOverlap) {
      if (lane_predicate) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(
            mainloop_params.tma_load_K.with(
                *pipeline_k.producer_get_barrier(smem_pipe_write_k),
                mcast_mask_kv),
            tKgK(_, n_block),
            tKsK(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;

        if (Has_rab) {
          pipeline_rab.producer_acquire(smem_pipe_write_rab);
          copy(
              mainloop_params.tma_load_Rab.with(
                  *pipeline_rab.producer_get_barrier(smem_pipe_write_rab), 0),
              tRabgRab(_, n_block),
              tRabsRab(_, smem_pipe_write_rab.index()));
          ++smem_pipe_write_rab;
        }
      }
    }

    // Wait for the MMA warpgroups to say that smem_q is ready
    cutlass::arch::NamedBarrier::sync(
        NumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<int>(FwdNamedBarriers::QueryEmpty));
    if (lane_predicate) {
      shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
      copy(
          mainloop_params.tma_load_Q.with(
              reinterpret_cast<
                  cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                  shared_storage.barrier_Q),
              0 /*mcast_mask*/),
          tQgQ,
          tQsQ);
    }

    // Wait for warp 1 to signal that smem_v are ready and V can be copied from
    // gmem Need ClusterBarrier, not just NamedBarrier. Otherwise we might have
    // CTA 0 finishing the TMA store on O first, call TMA multicast load on V,
    // before CTA 1 can finishing TMA store on O.
    shared_storage.barrier_O.wait((work_idx + 1) % 2);
    if (lane_predicate) {
      if constexpr (IntraWGOverlap) {
        auto load_step = [&](int n_valid_block, int n_valid_block_prev, int masking_step) {
          int n_block_prev = !Is_arbitrary ? n_valid_block_prev : sValidBlockIds[n_valid_block];
          int n_block_next = !Is_arbitrary ? n_valid_block - 1  : sValidBlockIds[n_valid_block - 1];
          if (n_block_next >= n_block_min) {
            pipeline_k.producer_acquire(smem_pipe_write_k);
            copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                  tKgK(_, n_block_next), tKsK(_, smem_pipe_write_k.index()));
            ++smem_pipe_write_k;
          }

          if (Has_rab && n_block_next >= n_block_min) {
            pipeline_rab.producer_acquire(smem_pipe_write_rab);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write_rab), 0),
                  tRabgRab(_, n_block_next), tRabsRab(_, smem_pipe_write_rab.index()));
            ++smem_pipe_write_rab;
          }

          pipeline_v.producer_acquire(smem_pipe_write_v);
          copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                  tVgV(_, n_block_prev), tVsV(_, smem_pipe_write_v.index()));
          ++smem_pipe_write_v;
        };
        n_block = n_block_max - 1;
#pragma unroll 2
        for (int masking_step = 0; n_block > n_block_min;
             ++masking_step, --n_block) {
          int n_block_prev = n_block;
          if (is_jump && masking_step == n_masking_steps - 1) {
            n_block = std::min(n_block, n_block_history);
          }
          load_step(n_block, n_block_prev, masking_step);
        }
      } else {
        auto load_step = [&](int n_valid_block, int masking_step) {
          int n_block = !Is_arbitrary ? n_valid_block : sValidBlockIds[n_valid_block];
          pipeline_k.producer_acquire(smem_pipe_write_k);
          copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
          ++smem_pipe_write_k;

          if (Has_rab) {
            pipeline_rab.producer_acquire(smem_pipe_write_rab);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write_rab), 0),
                  tRabgRab(_, n_block), tRabsRab(_, smem_pipe_write_rab.index()));
            ++smem_pipe_write_rab;
          }

          pipeline_v.producer_acquire(smem_pipe_write_v);
          copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
          ++smem_pipe_write_v;
        };
        n_block = n_block_max - 1;
#pragma unroll 2
        for (int masking_step = 0; n_block >= n_block_min;
             ++masking_step, --n_block) {
          load_step(n_block, masking_step);
          if (is_jump && masking_step == n_masking_steps - 1) {
            n_block = std::min(n_block, n_block_history);
          }
        }
      }
    }

    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
    if constexpr (IntraWGOverlap) {
      if (lane_predicate) {
        n_block = !Is_arbitrary ? n_block : sValidBlockIds[n_block];
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(
            mainloop_params.tma_load_V.with(
                *pipeline_v.producer_get_barrier(smem_pipe_write_v),
                mcast_mask_kv),
            tVgV(_, n_block),
            tVsV(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
      }
    }
  }

  template <typename Scheduler, typename SharedStorage>
  CUTLASS_DEVICE void load_fp8(
      Params const& mainloop_params,
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_rab,
      MainloopPipeline pipeline_v,
      MainloopPipelineNoTMA pipeline_vt,
      PipelineState& smem_pipe_write,
      PipelineState& smem_pipe_read,
      int n_block_max,
      int n_block_min,
      int n_masking_steps,
      bool is_jump,
      int n_block_history,
      SharedStorage& shared_storage,
      Scheduler& scheduler,
      typename Scheduler::Params const& scheduler_params,
      typename Scheduler::WorkTileInfo& work_tile_info,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      int work_idx,
      const Seqlen_traits& seqlen_traits_q,
      const Seqlen_traits& seqlen_traits_k) {
    using SmemLayoutTransposeV = typename Ktraits::SmemLayoutTransposeV;
    using SmemLayoutTransposeVt = typename Ktraits::SmemLayoutTransposeVt;

    Tensor sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sRab = make_tensor(
        make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sVt =
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});

    Tensor sV_divide = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutTransposeV{}));
    Tensor sVt_divide = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.smem_v_out.data()),
        SmemLayoutTransposeVt{}));

    auto smem_transpose_V = SmemTransposeFp8_64x64();
    auto do_transpose_V = [&](int stage) {
      if constexpr (Quant_mode != 1) {
        pipeline_v.consumer_wait(smem_pipe_read);
        pipeline_vt.producer_acquire(smem_pipe_write);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < shape<2>(SmemLayoutTransposeV{}); ++j) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < shape<1>(SmemLayoutTransposeV{}); ++i) {
            smem_transpose_V(
                flatten(sV_divide(_, i, j, stage)),
                flatten(sVt_divide(_, i, j, stage)));
          }
        }
        cutlass::arch::fence_view_async_shared();
        cutlass::arch::NamedBarrier::sync(
            cutlass::NumThreadsPerWarpGroup,
            static_cast<int>(FwdNamedBarriers::ProducerWG));
        pipeline_vt.producer_commit(smem_pipe_write);
        pipeline_v.consumer_release(smem_pipe_read);
      }
    };

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(
        mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(
        mainloop_params.layout_K.shape());
    Tensor mRab = mainloop_params.tma_load_Rab.get_tma_tensor(
        mainloop_params.layout_Rab.shape());
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(
        mainloop_params.layout_V.shape());

    auto [m_block, bidh, bidb] = block_coord;
    int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);
    int bidh_rab = mainloop_params.qhead_per_rabhead_divmod.divide(bidh);

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {
        block_rank_in_cluster % cluster_shape_x,
        block_rank_in_cluster / cluster_shape_x};
    Tensor gQ = seqlen_traits_q.get_local_tile_tensor(
        mQ, select<0, 2>(TileShape_MNK{}), bidh, bidb)(_, _, m_block);
    Tensor gK = seqlen_traits_k.get_local_tile_tensor(
        mK, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);
    int actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int offset_rab = actual_seqlen_k - actual_seqlen_q;
    Tensor gRab = local_tile(domain_offset(make_coord(offset_rab, _0{}), mRab(_, _, bidh_rab, bidb)),
                             select<0, 1>(TileShape_MNK{}), make_coord(m_block, _));
    Tensor gV = seqlen_traits_k.get_local_tile_tensorT(
        mV, select<1, 2>(TileShape_MNK_PV{}), bidh_kv, bidb);

    Tensor sQ_x =
        make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x =
        make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] = tma_partition(
        mainloop_params.tma_load_Q,
        _0{},
        Layout<_1>{},
        group_modes<0, 2>(sQ_x),
        group_modes<0, 2>(gQ_x));
    auto [tKgK, tKsK] = tma_partition(
        mainloop_params.tma_load_K,
        block_rank_in_cluster,
        Layout<ClusterShape>{},
        group_modes<0, 2>(sK),
        group_modes<0, 2>(gK));
    auto [tRabgRab, tRabsRab] = tma_partition(
        mainloop_params.tma_load_Rab,
        _0{},
        Layout<_1>{},
        group_modes<0, 2>(sRab),
        group_modes<0, 2>(gRab));
    auto [tVgV, tVsV] = tma_partition(
        mainloop_params.tma_load_V,
        block_rank_in_cluster,
        Layout<ClusterShape>{},
        group_modes<0, 2>(sVt),
        group_modes<0, 2>(gV));

    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{};
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_kv |=
            (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
      }
    }

    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // Wait for the MMA warpgroups to say that smem_q is ready
    cutlass::arch::NamedBarrier::sync(
    NumMmaThreads + cutlass::NumThreadsPerWarpGroup,
    static_cast<int>(FwdNamedBarriers::QueryEmpty));

    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
      copy(mainloop_params.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType &>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ);
    }
    shared_storage.barrier_O.wait((work_idx + 1) % 2);

    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});
    int n_block_valid = n_block_max - 1;
    #pragma unroll 2
    for (int masking_step = 0; n_block_valid >= n_block_min; ++masking_step, --n_block_valid) {
      int n_block = Is_arbitrary ? sValidBlockIds[n_block_valid] : n_block_valid;
      if (warp_idx_in_warpgroup == 0 && lane_predicate) {
        pipeline_k.producer_acquire(smem_pipe_write);
        copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
              tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));

        if (Has_rab) {
          pipeline_rab.producer_acquire(smem_pipe_write);
          copy(
              mainloop_params.tma_load_Rab.with(
                  *pipeline_rab.producer_get_barrier(smem_pipe_write), 0),
              tRabgRab(_, n_block),
              tRabsRab(_, smem_pipe_write.index()));
        }

        pipeline_v.producer_acquire(smem_pipe_write);
        copy(
            mainloop_params.tma_load_V.with(
                *pipeline_v.producer_get_barrier(smem_pipe_write),
                mcast_mask_kv),
            tVgV(_, n_block),
            tVsV(_, smem_pipe_write.index()));
      }

      do_transpose_V(smem_pipe_read.index());

      ++smem_pipe_write;
      ++smem_pipe_read;
      if (is_jump && masking_step == n_masking_steps - 1) {
        n_block_valid = std::min(n_block_valid, n_block_history);
      }
    }
    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_rab,
      MainloopPipeline pipeline_v,
      PipelineState& smem_pipe_write_k,
      PipelineState& smem_pipe_write_rab,
      PipelineState& smem_pipe_write_v) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // Issue the epilogue waits
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all Consumer UNLOCKs), or
       * if the stage was never used then would just be acquired since the phase
       * was still inverted from make_producer_start_state
       */
      pipeline_k.producer_tail(smem_pipe_write_k);
      if (Has_rab) {
        pipeline_rab.producer_tail(smem_pipe_write_rab);
      }
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail_one_write(
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_rab,
      MainloopPipeline pipeline_v,
      PipelineState& smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // Issue the epilogue waits
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all Consumer UNLOCKs), or
       * if the stage was never used then would just be acquired since the phase
       * was still inverted from make_producer_start_state
       */
      pipeline_k.producer_tail(smem_pipe_write);
      if (Has_rab) {
        pipeline_rab.producer_tail(smem_pipe_write);
      }
      pipeline_v.producer_tail(smem_pipe_write);
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      cutlass::arch::NamedBarrier::sync(
          NumMmaThreads,
          static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 +
              cutlass::canonical_warp_group_idx());
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_arrive() {
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(
        NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup ||
        NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
    if constexpr (NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup) {
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads,
          static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 +
              (3 - cutlass::canonical_warp_group_idx()));
    } else {
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads,
          static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 +
              (cutlass::canonical_warp_group_idx() <= 2
                   ? cutlass::canonical_warp_group_idx() + 1
                   : cutlass::canonical_warp_group_idx() + 1 - 3));
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads,
          static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 +
              (cutlass::canonical_warp_group_idx() <= 1
                   ? cutlass::canonical_warp_group_idx() + 2
                   : cutlass::canonical_warp_group_idx() + 2 - 3));
    }
  }

  CUTLASS_DEVICE void mma_init() {
    // Tell producer (warp 0) that smem_q is ready
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads + Ktraits::NumProducerThreads,
        static_cast<int>(FwdNamedBarriers::QueryEmpty));
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(
        NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup ||
        NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
    if (cutlass::canonical_warp_group_idx() > 1) {
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads,
          static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 1);
    }
    if constexpr (NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup) {
      if (cutlass::canonical_warp_group_idx() > 2) {
        cutlass::arch::NamedBarrier::arrive(
            NumMmaThreads,
            static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 2);
      }
    }
  }

  template <typename SharedStorage, typename FrgTensorO>
  CUTLASS_DEVICE void mma(
      Params const& mainloop_params,
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_rab,
      MainloopPipeline pipeline_v,
      PipelineState& smem_pipe_read_k,
      PipelineState& smem_pipe_read_rab,
      PipelineState& smem_pipe_read_v,
      FrgTensorO& tOrO,
      int n_block_max,
      int n_block_min,
      int n_masking_steps,
      bool is_jump,
      int n_block_history,
      int thread_idx,
      int work_idx,
      int m_block,
      SharedStorage& shared_storage,
      const Seqlen_traits& seqlen_traits_q,
      const Seqlen_traits& seqlen_traits_k) {
    static_assert(
        is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    Tensor sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sRab = make_tensor(
        make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sVt = make_tensor(
        make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtMma{});

    typename Ktraits::TiledMma0 tiled_mma0;
    typename Ktraits::TiledMma1 tiled_mma1;
    auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
    auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);

    // Allocate "fragments/descriptors" for first matmul.
    Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
    Tensor tSrK = threadMma0.partition_fragment_B(sK);
    // Allocate "fragments/descriptors" for second matmul.
    // Note: S becomes P.
    Tensor tOrV = threadMma1.partition_fragment_B(sVt);

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
    int const actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int const actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int const actual_seqlen_h =
        Is_target ? seqlen_traits_k.actual_seq_len_h : actual_seqlen_k;
    int const actual_seqlen_c =
        Is_context ? seqlen_traits_k.actual_seq_len_c : 0;
    int const actual_seqlen_offset = actual_seqlen_k - actual_seqlen_q;
    int const max_seq_len_q = seqlen_traits_q.max_seq_len;

    // arbitrary func
    Tensor mMaxFunc = make_tensor(make_gmem_ptr(mainloop_params.func_ptr + seqlen_traits_q.offset),
        make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2 + 1>{}, actual_seqlen_q),
        make_stride(/*mainloop_params.func_head_stride*/Int<0>{}, 2 * mainloop_params.func_ids_stride, _1{}));
    Tensor mMinFunc = make_tensor(make_gmem_ptr(mainloop_params.func_ptr + seqlen_traits_q.offset + mainloop_params.func_ids_stride),
        make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2>{}, actual_seqlen_q),
        make_stride(/*mainloop_params.func_head_stride*/Int<0>{}, 2 * mainloop_params.func_ids_stride, _1{}));

    Tensor gMaxFunc = local_tile(mMaxFunc(/*bidh*/Int<0>{}, _, _),
        make_shape(Int<kNFunc/2 + 1>{}, Int<kBlockM>{}),
        make_coord(_0{}, m_block));
    Tensor gMinFunc = local_tile(mMinFunc(/*bidh*/Int<0>{}, _, _),
        make_shape(Int<kNFunc/2>{}, Int<kBlockM>{}),
        make_coord(_0{}, m_block));

    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});

    int n_block = n_block_max - 1;
    n_block = Is_arbitrary ? sValidBlockIds[n_block] : n_block;

    auto col_limit_right = [&](int row) {
      return std::min(
          actual_seqlen_k,
          row + 1 + mainloop_params.window_size_right);
    };

    auto col_limit_left = [&](int row) {
      return std::max(0, row - mainloop_params.window_size_left);
    };

    auto apply_mask = [&](auto& tSrS, int n_block) {
      static constexpr int Row = 0, Col = 1;
      Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
      Tensor tScS = threadMma0.partition_C(cS);
      const int base_row = m_block * kBlockM + actual_seqlen_offset;
      const int base_col = n_block * kBlockN;

      Tensor tSrS_view = make_tensor(tSrS.data(), group<1, 4>(group<0, 2>(select<1, 3, 0, 2, 4>(flatten(tSrS.layout())))));
      Tensor tScS_view = make_tensor(tScS.data(), group<1, 4>(group<0, 2>(select<1, 3, 0, 2, 4>(flatten(tScS.layout())))));

      CUTLASS_PRAGMA_UNROLL
      for (int mma_row = 0; mma_row < size<0>(tSrS_view); mma_row++) {
        const int block_row = int(get<Row>(tScS_view(mma_row, 0)));
        const int row = block_row + base_row;

        [[maybe_unused]] const int target_index = Is_target ? (row - actual_seqlen_h) / mainloop_params.target_group_size : 0;
        [[maybe_unused]] const int target_col_limit_left = Is_target ? actual_seqlen_h + target_index * mainloop_params.target_group_size : 0;

        Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
        Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
        if constexpr (Is_arbitrary) {
          // Below if code introduces BRA. For the forward pass, we don't need to apply a mask in the seq_q direction.
          // The reason for adding the 'if' statement was to guard against gMin and gMax going out of bounds.
          // However, our design ensures that the lengths of the gMin and gMax sequences are both max_seq_q, so there's no need to worry about this issue.
          /*if (row >= actual_seqlen_q) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
            }
            continue;
          }*/
          col_max(0) = gMaxFunc(0, block_row);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(gMinFunc); ++j) {
            col_min(j)   = gMinFunc(j, block_row);
            col_max(j+1) = gMaxFunc(j+1, block_row);
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
          const int block_col = int(get<Col>(tScS_view(mma_row, mma_col)));
          const int col = block_col + base_col;
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
              if (row >= actual_seqlen_h) {
                if (col < target_col_limit_left && col >= actual_seqlen_h) {
                  tSrS_view(mma_row, mma_col) = -INFINITY;
                }
              }
            }
          }
          if constexpr (Is_arbitrary) {
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

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(
        shared_storage.barrier_Q.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.barrier_Q.wait(work_idx % 2);
    }

    // compute Q @ K + rab
    Tensor tSrS =
        partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
    auto smem_tiled_copy_rab =
        make_tiled_copy_C(typename Ktraits::SmemCopyAtomRab{}, tiled_mma0);
    auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(thread_idx);
    Tensor tSsRab = smem_thr_copy_rab.partition_S(sRab);
    Tensor tSrRab = make_tensor<Element>(
        partition_shape_C(tiled_mma0, select<0, 1>(TileShape_MNK{})));
    Tensor tSrRab_view = smem_thr_copy_rab.retile_D(tSrRab);
    static_assert(rank(tSrRab) == rank(tSrS));
    static_assert(size(tSrRab) == size(tSrS));
    static_assert(size(tSrRab) == size(tSrRab_view));

    if constexpr (IntraWGOverlap) {
      if (Has_rab) {
        consumer_wait(pipeline_rab, smem_pipe_read_rab);
        cute::copy(
            smem_tiled_copy_rab,
            tSsRab(_, _, _, smem_pipe_read_rab.index()),
            tSrRab_view(_, _, _));

        cute::copy(
            make_tensor(
                convert_type<ElementAccum>(tSrRab).data(), tSrS.layout()),
            tSrS);
        pipeline_rab.consumer_release(smem_pipe_read_rab);
        ++smem_pipe_read_rab;
      }

      consumer_wait(pipeline_k, smem_pipe_read_k);
      warp_scheduler_barrier_sync();
      flash::gemm</*zero_init=*/!Has_rab, /*wg_wait=*/-1>(
          tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
      warp_scheduler_barrier_arrive();
      warpgroup_wait<0>();
      pipeline_k.consumer_release(smem_pipe_read_k);
      ++smem_pipe_read_k;

      for (int i = 0; i < size(tSrS); ++i) {
        tSrS(i) *= mainloop_params.alpha;
      }
      apply_mask(tSrS, n_block);
      fast_silu(tSrS);
    }

    if (work_idx != 0) {
      int lane_predicate = cute::elect_one_sync();
      if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 &&
          lane_predicate) {
        tma_store_wait<0>();
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
          shared_storage.barrier_O.arrive(cta_id, lane_predicate);
        }
      }
    }

    if constexpr (IntraWGOverlap) {
      Tensor tOrP = make_tensor(
          convert_type<Element>(tSrS).data(),
          flash::convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(
              tSrS.layout()));
      auto fwd_step = [&](int n_block_valid, int masking_step) {
        int n_block_next = !Is_arbitrary ? n_block_valid - 1 : sValidBlockIds[n_block_valid - 1];
        const bool is_masking = masking_step < n_masking_steps - 1 || (n_block_next + 1) * kBlockN > actual_seqlen_h;
        if (Has_rab) {
          consumer_wait(pipeline_rab, smem_pipe_read_rab);
          cute::copy(
              smem_tiled_copy_rab,
              tSsRab(_, _, _, smem_pipe_read_rab.index()),
              tSrRab_view(_, _, _));
          cute::copy(
              make_tensor(
                  convert_type<ElementAccum>(tSrRab).data(), tSrS.layout()),
              tSrS);

          pipeline_rab.consumer_release(smem_pipe_read_rab); // release Rab
          ++smem_pipe_read_rab;
        }
        consumer_wait(pipeline_k, smem_pipe_read_k);
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/!Has_rab, /*wg_wait=*/-1>(
            tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        consumer_wait(pipeline_v, smem_pipe_read_v);

        // compute P @ V
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
            tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        warp_scheduler_barrier_arrive();
        warpgroup_wait<1>();
        pipeline_k.consumer_release(smem_pipe_read_k); // release K
        ++smem_pipe_read_k;

        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) *= mainloop_params.alpha;
        }
        if (Is_local || Is_arbitrary || is_masking) {
          apply_mask(tSrS, n_block_next);
        }
        fast_silu(tSrS);
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_v); // release V
        ++smem_pipe_read_v;
        cute::copy(
            make_tensor(convert_type<Element>(tSrS).data(), tOrP.layout()),
            tOrP);
      };
      n_block = n_block_max - 1;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int masking_step = 0; n_block > n_block_min; ++masking_step, --n_block) {
        if (is_jump && masking_step == n_masking_steps - 1) {
          n_block = std::min(n_block, n_block_history);
        }
        fwd_step(n_block, masking_step);
      }

      // Tell warp 0 that smem_q is ready
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads + cutlass::NumThreadsPerWarp,
          static_cast<int>(FwdNamedBarriers::QueryEmpty));
      consumer_wait(pipeline_v, smem_pipe_read_v);
      flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
          tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
      warpgroup_wait<0>();
      pipeline_v.consumer_release(
          smem_pipe_read_v); // release V, otherwise producers will hang
      ++smem_pipe_read_v;
    } else {
      auto fwd_step = [&](int n_block_valid, int masking_step) {
        int n_block = !Is_arbitrary ? n_block_valid : sValidBlockIds[n_block_valid];
        const bool is_masking = masking_step < n_masking_steps || (n_block + 1) * kBlockN > actual_seqlen_h;
        if (Has_rab) {
          consumer_wait(pipeline_rab, smem_pipe_read_rab);
          cute::copy(
              smem_tiled_copy_rab,
              tSsRab(_, _, _, smem_pipe_read_rab.index()),
              tSrRab_view(_, _, _));
          cute::copy(
              make_tensor(
                  convert_type<ElementAccum>(tSrRab).data(), tSrS.layout()),
              tSrS);
          pipeline_rab.consumer_release(smem_pipe_read_rab); // release Rab
          ++smem_pipe_read_rab;
        }
        consumer_wait(pipeline_k, smem_pipe_read_k);
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/!Has_rab, /*wg_wait=*/-1>(
            tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        warp_scheduler_barrier_arrive();
        warpgroup_wait<0>();
        pipeline_k.consumer_release(smem_pipe_read_k); // release K
        ++smem_pipe_read_k;

        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) *= mainloop_params.alpha;
        }
        if (Is_local || Is_arbitrary || is_masking) {
          apply_mask(tSrS, n_block);
        }
        fast_silu(tSrS);
        Tensor tOrP = make_tensor(
            convert_type<Element>(tSrS).data(),
            flash::convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(
                tSrS.layout()));
        consumer_wait(pipeline_v, smem_pipe_read_v);
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
            tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_v); // release V
        ++smem_pipe_read_v;
      };
      n_block = n_block_max - 1;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int masking_step = 0; n_block >= n_block_min; ++masking_step, --n_block) {
        fwd_step(n_block, masking_step);
        if (is_jump && masking_step == n_masking_steps - 1) {
          n_block = std::min(n_block, n_block_history);
        }
      }
      // Tell warp 0 that smem_q is ready
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads + cutlass::NumThreadsPerWarp,
          static_cast<int>(FwdNamedBarriers::QueryEmpty));
    }
    for (int i = 0; i < size(tOrO); ++i) {
      tOrO(i) /= max_seq_len_q;
    }
    return;
  }

  template <typename SharedStorage, typename FrgTensorO>
  CUTLASS_DEVICE void mma_fp8(
      Params const& mainloop_params,
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_rab,
      MainloopPipelineVt pipeline_vt,
      PipelineState& smem_pipe_read,
      PipelineState& smem_pipe_release,
      FrgTensorO& tOrO,
      int n_block_max,
      int n_block_min,
      int n_masking_steps,
      bool is_jump,
      int n_block_history,
      int thread_idx,
      int work_idx,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      SharedStorage& shared_storage,
      const Seqlen_traits& seqlen_traits_q,
      const Seqlen_traits& seqlen_traits_k) {
    static_assert(
        is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
    using TiledMma0descale = typename Ktraits::TiledMma0descale;
    using TiledMma1descale = typename Ktraits::TiledMma1descale;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    auto [m_block, bidh, bidb] = block_coord;

    int const actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int const actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int const actual_seqlen_h = Is_target ? seqlen_traits_k.actual_seq_len_h : actual_seqlen_k;
    int const actual_seqlen_c = Is_context ? seqlen_traits_k.actual_seq_len_c : 0;
    int const actual_seqlen_offset = actual_seqlen_k - actual_seqlen_q;
    int const max_seq_len_q = seqlen_traits_q.max_seq_len;

    Tensor sQ =
        make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK =
        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sRab = make_tensor(
        make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sVt = make_tensor(
        make_smem_ptr(Quant_mode != 1 ? shared_storage.smem_v_out.data() : shared_storage.smem_v.data()), SmemLayoutVtMma{});

    typename Ktraits::TiledMma0 tiled_mma0;
    typename Ktraits::TiledMma1 tiled_mma1;

    TiledMma0descale tiled_mma0_descale;
    TiledMma1descale tiled_mma1_descale;

    auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
    auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);
    tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
    auto threadMma0descale = tiled_mma0_descale.get_thread_slice(thread_idx);
    auto threadMma1descale = tiled_mma1_descale.get_thread_slice(thread_idx);

    // Allocate "fragments/descriptors" for first matmul.
    Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
    Tensor tSrK = threadMma0.partition_fragment_B(sK);
    // Allocate "fragments/descriptors" for second matmul.
    Tensor tOrV = threadMma1.partition_fragment_B(sVt);

    // define the quantization tensor
    int const num_heads = get<2>(mainloop_params.layout_Q.shape());
    // Quantization Mode == 1
    auto mQ_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr + seqlen_traits_q.cu_seq_len[bidb]),
                                make_shape(num_heads, actual_seqlen_q, _1{}),
                                make_stride(mainloop_params.descale_q_head_stride, _1{}, _0{}));
    auto gQ_descale = local_tile(mQ_descale(bidh, _, _), Shape<Int<kBlockM>, Int<1>>{}, make_coord(m_block, 0));

    auto mK_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr + seqlen_traits_k.cu_seq_len[bidb]),
                                make_shape(num_heads, actual_seqlen_k, _1{}),
                                make_stride(mainloop_params.descale_k_head_stride, _1{}, _0{}));
    auto gK_descale = local_tile(mK_descale(bidh, _, _), Shape<Int<kBlockN>, Int<1>>{}, make_coord(_, 0));

    auto mVt_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_vt_ptr + mainloop_params.cu_seqlens_vt_descale[bidb] * mainloop_params.descale_vt_row_stride),
                                make_shape(num_heads, kHeadDim, mainloop_params.cu_seqlens_vt_descale[bidb + 1] - mainloop_params.cu_seqlens_vt_descale[bidb]),
                                make_stride(mainloop_params.descale_vt_head_stride, _1{}, mainloop_params.descale_vt_row_stride));
    auto gVt_descale = local_tile(mVt_descale(bidh, _, _), Shape<Int<kHeadDim>, Int<1>>{}, make_coord(0, _));

    // Quantization Mode == 2: block scale
    auto mQ_block_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr + mainloop_params.cu_seqlens_q_block_descale[bidb]),
                                make_shape(num_heads, mainloop_params.cu_seqlens_q_block_descale[bidb+1] - mainloop_params.cu_seqlens_q_block_descale[bidb]),
                                make_stride(mainloop_params.q_block_descale_head_stride, _1{}));

    auto gQ_block_descale = local_tile(mQ_block_descale(bidh, _),
                    Shape<Int<1>>{}, make_coord(m_block));

    auto mK_block_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr + mainloop_params.cu_seqlens_kv_block_descale[bidb]),
                                make_shape(num_heads, mainloop_params.cu_seqlens_kv_block_descale[bidb+1] - mainloop_params.cu_seqlens_kv_block_descale[bidb]),
                                make_stride(mainloop_params.kv_block_descale_head_stride, _1{}));

    auto gK_block_descale = local_tile(mK_block_descale(bidh, _), Shape<Int<1>>{}, make_coord(_));

    auto mV_block_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_v_ptr + mainloop_params.cu_seqlens_kv_block_descale[bidb]),
                                make_shape(num_heads, mainloop_params.cu_seqlens_kv_block_descale[bidb+1] - mainloop_params.cu_seqlens_kv_block_descale[bidb]),
                                make_stride(mainloop_params.kv_block_descale_head_stride,  _1{}));

    auto gV_block_descale = local_tile(mV_block_descale(bidh, _), Shape<Int<1>>{}, make_coord(_));

    // Quantization Mode == 3: head scale
    int const num_batch = mainloop_params.num_batch;
    auto mQ_head_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr), make_shape(num_batch, num_heads), make_stride(num_heads, _1{}));
    auto mK_head_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr), make_shape(num_batch, num_heads), make_stride(num_heads, _1{}));
    auto mV_head_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_v_ptr), make_shape(num_batch, num_heads), make_stride(num_heads, _1{}));

    // Quantization Mode == 4: batch scale
    auto mQ_batch_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr), make_shape(num_batch), make_stride(_1{}));
    auto mK_batch_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr), make_shape(num_batch), make_stride(_1{}));
    auto mV_batch_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_v_ptr), make_shape(num_batch), make_stride(_1{}));

    // Quantization Mode == 5: tensor scale
    auto mQ_tensor_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_q_ptr), make_shape(_1{}), make_stride(_0{}));
    auto mK_tensor_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_k_ptr), make_shape(_1{}), make_stride(_0{}));
    auto mV_tensor_descale = make_tensor(make_gmem_ptr(mainloop_params.descale_v_ptr), make_shape(_1{}), make_stride(_0{}));

    Tensor tSrQ_descale = threadMma0descale.partition_fragment_A(gQ_descale);
    Tensor tOrP_descale = make_tensor_like(tSrQ_descale);

    auto gmem_tiled_copy_Q_descale =
      make_tiled_copy_A(typename Ktraits::DescaleCopyAtom{}, tiled_mma0_descale);
    auto gmem_thr_copy_Q_descale = gmem_tiled_copy_Q_descale.get_thread_slice(thread_idx);

    auto tSgQ_descale = gmem_thr_copy_Q_descale.partition_S(gQ_descale);
    Tensor tSrQ_descale_view = gmem_thr_copy_Q_descale.retile_D(tSrQ_descale);
    cute::copy(gmem_tiled_copy_Q_descale, tSgQ_descale, tSrQ_descale_view);

    auto gmem_tiled_copy_K_descale =
      make_tiled_copy_B(typename Ktraits::DescaleCopyAtom{}, tiled_mma0_descale);
    auto gmem_thr_copy_K_descale = gmem_tiled_copy_K_descale.get_thread_slice(thread_idx);

    auto gmem_tiled_copy_V_descale =
      make_tiled_copy_B(typename Ktraits::DescaleCopyAtom{}, tiled_mma1_descale);
    auto gmem_thr_copy_V_descale = gmem_tiled_copy_V_descale.get_thread_slice(thread_idx);

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(
        shared_storage.barrier_Q.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.barrier_Q.wait(work_idx % 2);
    }

    if (work_idx != 0) {
      int lane_predicate = cute::elect_one_sync();
      if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 &&
          lane_predicate) {
        tma_store_wait<0>();
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
          shared_storage.barrier_O.arrive(cta_id, lane_predicate);
        }
      }
    }

    auto smem_tiled_copy_rab =
        make_tiled_copy_C(typename Ktraits::SmemCopyAtomRab{}, tiled_mma0);
    auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(thread_idx);
    Tensor tSsRab = smem_thr_copy_rab.partition_S(sRab);
    Tensor tSrRab = make_tensor<ElementRab>(
        partition_shape_C(tiled_mma0, select<0, 1>(TileShape_MNK{})));
    Tensor tSrRab_copy_view = smem_thr_copy_rab.retile_D(tSrRab);

    // arbitrary func
    Tensor mMaxFunc = make_tensor(make_gmem_ptr(mainloop_params.func_ptr + seqlen_traits_q.offset),
        make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2 + 1>{}, actual_seqlen_q),
        make_stride(/*mainloop_params.func_head_stride*/Int<0>{}, 2 * mainloop_params.func_ids_stride, _1{}));
    Tensor mMinFunc = make_tensor(make_gmem_ptr(mainloop_params.func_ptr + seqlen_traits_q.offset + mainloop_params.func_ids_stride),
        make_shape(/*params.h*/Int<1>{}, Int<kNFunc/2>{}, actual_seqlen_q),
        make_stride(/*mainloop_params.func_head_stride*/Int<0>{}, 2 * mainloop_params.func_ids_stride, _1{}));

    Tensor gMaxFunc = local_tile(mMaxFunc(/*bidh*/Int<0>{}, _, _),
        make_shape(Int<kNFunc/2 + 1>{}, Int<kBlockM>{}),
        make_coord(_0{}, m_block));
    Tensor gMinFunc = local_tile(mMinFunc(/*bidh*/Int<0>{}, _, _),
        make_shape(Int<kNFunc/2>{}, Int<kBlockM>{}),
        make_coord(_0{}, m_block));

    Tensor sValidBlockIds = make_tensor(make_smem_ptr(reinterpret_cast<int*>(shared_storage.smem_valid_block_ids.data())), typename Ktraits::SmemLayoutValidBlockIds{});

    int n_block = n_block_max - 1;

    auto col_limit_right = [&](int row) {
      return std::min(
          actual_seqlen_k, row + 1 + mainloop_params.window_size_right);
    };

    auto col_limit_left = [&](int row) {
      return std::max(0, row - mainloop_params.window_size_left);
    };

    auto apply_mask = [&](auto &tSrS, int n_block) {
      static constexpr int Row = 0, Col = 1;
      Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
      Tensor tScS = threadMma0.partition_C(cS);
      const int base_row = m_block * kBlockM + actual_seqlen_offset;
      const int base_col = n_block * kBlockN;

      Tensor tSrS_view = make_tensor(tSrS.data(), group<1, 4>(group<0, 2>(select<1, 3, 0, 2, 4>(flatten(tSrS.layout())))));
      Tensor tScS_view = make_tensor(tScS.data(), group<1, 4>(group<0, 2>(select<1, 3, 0, 2, 4>(flatten(tScS.layout())))));

      CUTLASS_PRAGMA_UNROLL
      for (int mma_row = 0; mma_row < size<0>(tSrS_view); mma_row++) {
        const int block_row = int(get<Row>(tScS_view(mma_row, 0)));
        const int row = block_row + base_row;

        [[maybe_unused]] const int target_index = Is_target ? (row - actual_seqlen_h) / mainloop_params.target_group_size : 0;
        [[maybe_unused]] const int target_col_limit_left = Is_target ? actual_seqlen_h + target_index * mainloop_params.target_group_size : 0;

        Tensor col_min = make_tensor<int>(make_shape(size<0>(gMinFunc)));
        Tensor col_max = make_tensor<int>(make_shape(size<0>(gMaxFunc)));
        if constexpr (Is_arbitrary) {
          // Below if code introduces BRA. For the forward pass, we don't need to apply a mask in the seq_q direction.
          // The reason for adding the 'if' statement was to guard against gMin and gMax going out of bounds.
          // However, our design ensures that the lengths of the gMin and gMax sequences are both max_seq_q, so there's no need to worry about this issue.
          /*if (row >= actual_seqlen_q) {
            CUTLASS_PRAGMA_UNROLL
            for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
              tSrS_view(mma_row, mma_col) = -INFINITY;
            }
            continue;
          }*/
          col_max(0) = gMaxFunc(0, block_row);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(gMinFunc); ++j) {
            col_min(j)   = gMinFunc(j, block_row);
            col_max(j+1) = gMaxFunc(j+1, block_row);
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int mma_col = 0; mma_col < size<1>(tSrS_view); mma_col++) {
          const int block_col = int(get<Col>(tScS_view(mma_row, mma_col)));
          const int col = block_col + base_col;
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
              if (row >= actual_seqlen_h) {
                if (col < target_col_limit_left && col >= actual_seqlen_h) {
                  tSrS_view(mma_row, mma_col) = -INFINITY;
                }
              }
            }
          }
          if constexpr (Is_arbitrary) {
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

    auto fwd_step = [&](int n_block_valid) {
      int n_block = Is_arbitrary ? sValidBlockIds[n_block_valid] : n_block_valid;
      Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
      float tSrS_max = 1.0;

      consumer_wait(pipeline_k, smem_pipe_read);
      Tensor tSrS_descale = partition_fragment_C(tiled_mma0_descale, select<0, 1>(TileShape_MNK{}));
      auto tSgK_descale = gmem_thr_copy_K_descale.partition_S(gK_descale(_, _, n_block));
      Tensor tSrK_descale = threadMma0descale.partition_fragment_B(gK_descale(_, _, n_block));
      Tensor tSrK_descale_view = gmem_thr_copy_K_descale.retile_D(tSrK_descale);
      if constexpr (Quant_mode == 1) {
        cute::copy(gmem_tiled_copy_K_descale, tSgK_descale, tSrK_descale_view);
        cute::gemm(tiled_mma0_descale, tSrQ_descale, tSrK_descale, tSrS_descale);
      }
      Tensor tSrS_descale_view = make_tensor(tSrS_descale.data(), composition(tSrS_descale.layout(), Layout<Shape<_2, _2, Int<kBlockN / 8>>, Stride<_2, _1, _4>>{}));

      // compute Q @ K + rab
      warp_scheduler_barrier_sync();
      flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);

      warp_scheduler_barrier_arrive();
      pipeline_k.consumer_release(smem_pipe_read);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSrS); ++i) {
        if constexpr (Quant_mode == 1) {
          tSrS(i) *= tSrS_descale_view(i);
        } else if constexpr (Quant_mode == 2) {
          tSrS(i) *= gQ_block_descale(0) * gK_block_descale(n_block);
        } else if constexpr (Quant_mode == 3) {
          tSrS(i) *= mQ_head_descale(bidb, bidh) * mK_head_descale(bidb, bidh);
        } else if constexpr (Quant_mode == 4) {
          tSrS(i) *= mQ_batch_descale(bidb) * mK_batch_descale(bidb);
        } else if constexpr (Quant_mode == 5) {
          tSrS(i) *= mQ_tensor_descale(0) * mK_tensor_descale(0);
        }
      }

      if constexpr (Has_rab) {
        consumer_wait(pipeline_rab, smem_pipe_read);
        cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, smem_pipe_read.index()), tSrRab_copy_view(_, _, _));
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) += static_cast<float>(tSrRab(i));
        }
        cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::AddRabWG1) - 1 + cutlass::canonical_warp_group_idx());
        pipeline_rab.consumer_release(smem_pipe_read);
      }

      for (int i = 0; i < size(tSrS); ++i) {
        tSrS(i) *= mainloop_params.alpha;
      }

      apply_mask(tSrS, n_block);

      fast_silu(tSrS);

      if constexpr (Quant_mode != 0) {
        //compute the P max value
        tSrS_max = EPSILON;
        for (int i = 0; i < size(tSrS); i++) {
          tSrS_max = max(tSrS_max, fabs(tSrS(i)));
        }
        warpReduce(tSrS_max, MaxOp<float>());

        tSrS_max = __shfl_sync(0xffffffff, tSrS_max, 0) / std::numeric_limits<Element>::max();
        for (int i = 0; i < size(tSrS); i ++) {
          tSrS(i) /= tSrS_max;
        }
      }

      Tensor tSrS_converted = make_tensor_like<Element>(tSrS);
      convert_type_safe(tSrS, tSrS_converted);

      Tensor tOrP = make_tensor(
          tSrS_converted.data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
      permute_regs_C_to_A<Quant_mode != 1>(tOrP);

      for (int i = 0; i < size(tOrP_descale); i ++) {
        tOrP_descale(i) = tSrS_max;
      }
      auto tOgV_descale = gmem_thr_copy_V_descale.partition_S(gVt_descale(_, _, n_block / kNBlock_shared));
      Tensor tOrV_descale = threadMma1descale.partition_fragment_B(gVt_descale(_, _, n_block / kNBlock_shared));
      Tensor tOrV_descale_view = gmem_thr_copy_V_descale.retile_D(tOrV_descale);
      Tensor tOrO_descale = partition_fragment_C(tiled_mma1_descale, select<0, 2>(TileShape_MNK{}));
      if constexpr (Quant_mode == 1) {
          cute::copy(gmem_tiled_copy_V_descale, tOgV_descale, tOrV_descale_view);
          cute::gemm(tiled_mma1_descale, tOrP_descale, tOrV_descale, tOrO_descale);
      }

      Tensor tOrO_descale_view = make_tensor(tOrO_descale.data(), composition(tOrO_descale.layout(), Layout<Shape<_2, _2, Int<kHeadDim / 8>>, Stride<_2, _1, _4>>{}));

      consumer_wait(pipeline_vt, smem_pipe_read);

      if constexpr (Quant_mode <= 0) {
        flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
      } else {
        Tensor tOrO_tmp = make_tensor_like(tOrO);
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO_tmp);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tOrO); i ++) {
          if constexpr (Quant_mode == 1) {
            tOrO(i) += tOrO_tmp(i) * tOrO_descale_view(i);
          } else if constexpr (Quant_mode == 2) {
            tOrO(i) += tOrO_tmp(i) * tSrS_max * gV_block_descale(n_block);
          } else if constexpr (Quant_mode == 3) {
            tOrO(i) += tOrO_tmp(i) * tSrS_max * mV_head_descale(bidb, bidh);
          } else if constexpr (Quant_mode == 4) {
            tOrO(i) += tOrO_tmp(i) * tSrS_max * mV_batch_descale(bidb);
          } else if constexpr (Quant_mode == 5) {
            tOrO(i) += tOrO_tmp(i) * tSrS_max * mV_tensor_descale(0);
          }
        }
      }

      pipeline_vt.consumer_release(smem_pipe_read);

      ++smem_pipe_read;
    };

    CUTLASS_PRAGMA_NO_UNROLL
    for (int masking_step = 0; n_block >= n_block_min; ++masking_step, --n_block) {
      fwd_step(n_block);
      if (is_jump && masking_step == n_masking_steps - 1) {
        n_block = std::min(n_block, n_block_history);
      }
    }
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads + cutlass::NumThreadsPerWarpGroup,
        static_cast<int>(FwdNamedBarriers::QueryEmpty));
    for (int i = 0; i < size(tOrO); ++i) {
      tOrO(i) /= max_seq_len_q;
    }
    return;
  }
};

} // namespace flash
