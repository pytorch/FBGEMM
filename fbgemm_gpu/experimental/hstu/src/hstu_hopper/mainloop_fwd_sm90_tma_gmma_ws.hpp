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
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<ldsm_thread_shape>{},
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

  using TiledCopySTSM =
      decltype(make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, Element>{},
                                Layout<stsm_thread_shape>{},
                                Layout<stsm_value_shape, stsm_value_stride>{}));
  TiledCopySTSM tiled_copy_stsm;

  template <class SmemTensor, class SmemTensorOut>
  CUTLASS_DEVICE void operator()(SmemTensor &&s_in, SmemTensorOut &&s_out)
  {
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
      uint32_t *data_32bit = reinterpret_cast<uint32_t *>(&data[n]);
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
  using Element = typename Ktraits::Element;
  using ElementAccum = typename Ktraits::ElementAccum;
  using ElementRab = cute::conditional_t<cutlass::sizeof_bits_v<Element> == 8,
                                          typename Ktraits::OutputType,
                                          typename Ktraits::Element>;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr bool Is_causal = Ktraits::Is_causal;
  static constexpr bool Is_target = Ktraits::Is_target;
  static constexpr bool Is_context = Ktraits::Is_context;
  static constexpr bool Is_delta_q = Ktraits::Is_delta_q;
  static constexpr bool Is_local = Ktraits::Is_local;
  static constexpr bool Has_rab = Ktraits::Has_rab;

  static constexpr int kStages = Ktraits::kStages;
  static constexpr int kHeadDim = Ktraits::kHeadDim;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyRab = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutRab = typename Ktraits::SmemLayoutRab;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const *>(nullptr)),
          repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
          typename Seqlen_traits::StrideT{}),
      SmemLayoutQ{},
      select<0, 2>(TileShape_MNK{}),
      _1{})); // no mcast for Q

  using TMA_Rab = decltype(make_tma_copy(
      GmemTiledCopyRab{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementRab const *>(nullptr)),
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
      take<0, 2>(SmemLayoutK{}),
      select<1, 2>(TileShape_MNK{}),
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  // TMA_V may differ from TMA_K for fp8 kernel (e.g. swizzling mode)
  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const *>(nullptr)),
          repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
          typename Seqlen_traits::StrideT{}),
      take<0, 2>(SmemLayoutV{}),
      select<1, 2>(TileShape_MNK{}),
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using MainloopPipelineNoTMA = typename Ktraits::MainloopPipelineNoTMA;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesRab = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutRab{})) * cutlass::sizeof_bits_v<ElementRab> / 8);
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);

  static constexpr bool UseSchedulerBarrier = true;
  static constexpr bool IntraWGOverlap = cutlass::sizeof_bits_v<Element> == 16 ? kHeadDim < 128 : true;

  // Host side kernel arguments
  struct Arguments {
    Element const *ptr_Q;
    typename Seqlen_traits::LayoutT layout_Q;
    ElementRab const *ptr_Rab;
    typename Seqlen_traits::LayoutRabT layout_Rab;
    Element const *ptr_K;
    typename Seqlen_traits::LayoutT layout_K;
    Element const *ptr_V;
    typename Seqlen_traits::LayoutT layout_V;
    float const *descale_q_ptr;
    float const *descale_k_ptr;
    float const *descale_v_ptr;
    const int window_size_left;
    const int window_size_right;
    const int target_group_size;
    const float target_group_size_inv;
    const float alpha;
  };

  // Device side kernel params
  struct Params {
    typename Seqlen_traits::LayoutT layout_Q;
    typename Seqlen_traits::LayoutRabT layout_Rab;
    typename Seqlen_traits::LayoutT layout_K;
    typename Seqlen_traits::LayoutT layout_V;
    cutlass::FastDivmod qhead_per_khead_divmod;
    cutlass::FastDivmod qhead_per_rabhead_divmod;
    TMA_Q tma_load_Q;
    TMA_Rab tma_load_Rab;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    float const *descale_q_ptr;
    float const *descale_k_ptr;
    float const *descale_v_ptr;
    const int window_size_left;
    const int window_size_right;
    const int target_group_size;
    const float target_group_size_inv;
    const float alpha;
  };

  static Params
  to_underlying_arguments(Arguments const &args) {
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
        SmemLayoutV{}(_, _, _0{}),
        select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
    return {args.layout_Q, args.layout_Rab, args.layout_K, args.layout_V,
            cutlass::FastDivmod(cute::ceil_div(get<2>(args.layout_Q.shape()), get<2>(args.layout_K.shape()))),
            cutlass::FastDivmod(cute::ceil_div(get<2>(args.layout_Q.shape()), get<2>(args.layout_Rab.shape()))),
            tma_load_Q, tma_load_Rab, tma_load_K, tma_load_V,
            args.descale_q_ptr, args.descale_k_ptr, args.descale_v_ptr,
            args.window_size_left, args.window_size_right, args.target_group_size, args.target_group_size_inv, args.alpha};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const &mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_V.get_tma_descriptor());
    if (Has_rab) {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_Rab.get_tma_descriptor());
    }
  }

  CUTLASS_DEVICE
  int get_n_block_min(
      Params const &mainloop_params, int m_block,
      const Seqlen_traits &seqlen_traits_q,
      const Seqlen_traits &seqlen_traits_k) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int const actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int const actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int const actual_seqlen_offset = Is_delta_q ? actual_seqlen_k - actual_seqlen_q : 0;
    int n_block_min = 0;
    if constexpr (Is_local) {
      n_block_min = std::max(
          0,
          (m_block * kBlockM + actual_seqlen_offset - mainloop_params.window_size_left) / kBlockN);
    }
    return n_block_min;
  }

  CUTLASS_DEVICE
  int get_n_block_max(
      Params const &mainloop_params, int m_block,
      const Seqlen_traits &seqlen_traits_q,
      const Seqlen_traits &seqlen_traits_k) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int const actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int const actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int const actual_seqlen_offset = Is_delta_q ? actual_seqlen_k - actual_seqlen_q : 0;
    int n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
      n_block_max = std::min(
          n_block_max,
          cute::ceil_div((m_block + 1) * kBlockM + actual_seqlen_offset + mainloop_params.window_size_right, kBlockN));
    }
    return n_block_max;
  }

  template <typename Scheduler, typename SharedStorage>
  CUTLASS_DEVICE void
  load(Params const &mainloop_params,
        MainloopPipeline pipeline_k,
        MainloopPipeline pipeline_rab,
        MainloopPipeline pipeline_v,
        PipelineState &smem_pipe_write_k,
        PipelineState &smem_pipe_write_rab,
        PipelineState &smem_pipe_write_v,
        int n_block_max,
        int n_block_min,
        int n_masking_steps,
        bool is_jump,
        int n_block_history,
        SharedStorage &shared_storage,
        Scheduler &scheduler,
        typename Scheduler::Params const &scheduler_params,
        typename Scheduler::WorkTileInfo &work_tile_info,
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        int work_idx,
        const Seqlen_traits &seqlen_traits_q,
        const Seqlen_traits &seqlen_traits_k) {

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sRab = make_tensor(make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    auto [m_block, bidh, bidb] = block_coord;
    int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);
    int bidh_rab = mainloop_params.qhead_per_rabhead_divmod.divide(bidh);

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
    Tensor mRab = mainloop_params.tma_load_Rab.get_tma_tensor(mainloop_params.layout_Rab.shape())(_, _, bidh_rab, bidb);
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
    Tensor gQ = seqlen_traits_q.get_local_tile_tensor(mQ, select<0, 2>(TileShape_MNK{}), bidh, bidb)(_, _, m_block);
    Tensor gK = seqlen_traits_k.get_local_tile_tensor(mK, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);
    int actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int offset_rab = actual_seqlen_k - actual_seqlen_q;
    Tensor gRab = local_tile(domain_offset(make_coord(offset_rab, _0{}), mRab), select<0, 1>(TileShape_MNK{}),
                              make_coord(m_block, _));
    Tensor gV = seqlen_traits_k.get_local_tile_tensor(mV, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);

    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] = tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));
    auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K, block_rank_in_cluster, Layout<ClusterShape>{},
                                      group_modes<0, 2>(sK), group_modes<0, 2>(gK));
    auto [tRabgRab, tRabsRab] = tma_partition(mainloop_params.tma_load_Rab, _0{}, Layout<_1>{},
                                              group_modes<0, 2>(sRab), group_modes<0, 2>(gRab));
    auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V, block_rank_in_cluster, Layout<ClusterShape>{},
                                      group_modes<0, 2>(sV), group_modes<0, 2>(gV));

    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>)
    {
      auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m)
      {
        mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
      }
    }
    int n_block = n_block_max - 1;

    int lane_predicate = cute::elect_one_sync();
    if constexpr (IntraWGOverlap) {
      if (lane_predicate) {
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
      }
    }

    // Wait for the MMA warpgroups to say that smem_q is ready
    cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty));
    if (lane_predicate) {
      shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
      copy(mainloop_params.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType &>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ);
    }

    // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
    // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
    // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on O.
    shared_storage.barrier_O.wait((work_idx + 1) % 2);
    int n_block_prev = n_block;
    if (lane_predicate) {
      if constexpr (IntraWGOverlap) {
        #pragma unroll 2
        for (int masking_step = 0; n_block > n_block_min; ++masking_step, --n_block) {
          if (is_jump && masking_step == n_masking_steps - 1) {
            n_block_prev = n_block;
            n_block = std::min(n_block, n_block_history);
          }
          pipeline_k.producer_acquire(smem_pipe_write_k);
          copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
                tKgK(_, n_block - 1), tKsK(_, smem_pipe_write_k.index()));
          ++smem_pipe_write_k;

          if (Has_rab) {
            pipeline_rab.producer_acquire(smem_pipe_write_rab);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write_rab), 0),
                  tRabgRab(_, n_block - 1), tRabsRab(_, smem_pipe_write_rab.index()));
            ++smem_pipe_write_rab;
          }

          pipeline_v.producer_acquire(smem_pipe_write_v);
          if (is_jump && masking_step == n_masking_steps - 1) {
            copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                  tVgV(_, n_block_prev), tVsV(_, smem_pipe_write_v.index()));
          } else {
            copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
                  tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
          }
          ++smem_pipe_write_v;
        }
      } else {
        #pragma unroll 2
        for (int masking_step = 0; n_block >= n_block_min; ++masking_step, --n_block) {
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
          if (is_jump && masking_step == n_masking_steps - 1) {
            n_block = std::min(n_block, n_block_history);
          }
        }
      }
    }

    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
    if constexpr (IntraWGOverlap) {
      if (lane_predicate) {
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv),
              tVgV(_, n_block), tVsV(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
      }
    }
  }

  template <typename Scheduler, typename SharedStorage>
  CUTLASS_DEVICE void
  load_fp8(Params const &mainloop_params,
            MainloopPipeline pipeline_k,
            MainloopPipeline pipeline_rab,
            MainloopPipeline pipeline_v,
            MainloopPipelineNoTMA pipeline_vt,
            PipelineState &smem_pipe_write,
            PipelineState &smem_pipe_read,
            SharedStorage &shared_storage,
            Scheduler &scheduler,
            typename Scheduler::Params const &scheduler_params,
            typename Scheduler::WorkTileInfo &work_tile_info,
            cute::tuple<int32_t, int32_t, int32_t> block_coord,
            int work_idx,
            const Seqlen_traits &seqlen_traits_q,
            const Seqlen_traits &seqlen_traits_k) {
    using SmemLayoutTransposeV = typename Ktraits::SmemLayoutTransposeV;
    using SmemLayoutTransposeVt = typename Ktraits::SmemLayoutTransposeVt;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sRab = make_tensor(make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor sV_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutTransposeV{}));
    Tensor sVt_divide = as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_v_out.data()), SmemLayoutTransposeVt{}));

    auto smem_transpose_V = SmemTransposeFp8_64x64();
    auto do_transpose_V = [&](int stage) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < shape<2>(SmemLayoutTransposeV{}); ++j) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < shape<1>(SmemLayoutTransposeV{}); ++i) {
          smem_transpose_V(flatten(sV_divide(_, i, j, stage)),
                            flatten(sVt_divide(_, i, j, stage)));
        }
      }
      cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::ProducerWG));
    };

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
    Tensor mRab = mainloop_params.tma_load_Rab.get_tma_tensor(mainloop_params.layout_Rab.shape());
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());

    auto [m_block, bidh, bidb] = block_coord;
    int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);
    int bidh_rab = mainloop_params.qhead_per_rabhead_divmod.divide(bidh);

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
    Tensor gQ = seqlen_traits_q.get_local_tile_tensor(mQ, select<0, 2>(TileShape_MNK{}), bidh, bidb)(_, _, m_block);
    Tensor gK = seqlen_traits_k.get_local_tile_tensor(mK, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);
    Tensor gRab = local_tile(mRab(_, _, bidh_rab, bidb), select<0, 1>(TileShape_MNK{}), make_coord(m_block, _));
    Tensor gV = seqlen_traits_k.get_local_tile_tensor(mV, select<1, 2>(TileShape_MNK{}), bidh_kv, bidb);

    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] = tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));
    auto [tKgK, tKsK] = tma_partition(mainloop_params.tma_load_K, block_rank_in_cluster, Layout<ClusterShape>{},
                                      group_modes<0, 2>(sK), group_modes<0, 2>(gK));
    auto [tRabgRab, tRabsRab] = tma_partition(mainloop_params.tma_load_Rab, _0{}, Layout<_1>{},
                                              group_modes<0, 2>(sRab), group_modes<0, 2>(gRab));
    auto [tVgV, tVsV] = tma_partition(mainloop_params.tma_load_V, block_rank_in_cluster, Layout<ClusterShape>{},
                                      group_modes<0, 2>(sV), group_modes<0, 2>(gV));

    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{};
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
      }
    }

    int n_block_max = get_n_block_max(mainloop_params, m_block, seqlen_traits_q, seqlen_traits_k);
    int n_block = n_block_max - 1;

    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      pipeline_k.producer_acquire(smem_pipe_write);
      copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
            tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));

      if (Has_rab) {
        pipeline_rab.producer_acquire(smem_pipe_write);
        copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0),
              tRabgRab(_, n_block), tRabsRab(_, smem_pipe_write.index()));
      }
    }

    // Wait for the MMA warpgroups to say that smem_q is ready
    // for fp8, change from NumThreadsPerWarp to NumThreadsPerWarpGroup
    cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::QueryEmpty));

    if constexpr (Is_causal) {
      if (warp_idx_in_warpgroup == 0 && lane_predicate) {
        shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(mainloop_params.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType &>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ);
        pipeline_v.producer_acquire(smem_pipe_write);
        copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
              tVgV(_, n_block), tVsV(_, smem_pipe_write.index()));
      }

      shared_storage.barrier_O.wait((work_idx + 1) % 2);

      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < kStages && n_block > 0; ++iter, --n_block) {
        pipeline_v.consumer_wait(smem_pipe_read);
        do_transpose_V(smem_pipe_read.index());
        pipeline_vt.producer_commit(smem_pipe_write);
        pipeline_v.consumer_release(smem_pipe_read);

        ++smem_pipe_write;
        ++smem_pipe_read;

        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
          pipeline_k.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                tKgK(_, n_block - 1), tKsK(_, smem_pipe_write.index()));
          if (Has_rab) {
            pipeline_rab.producer_acquire(smem_pipe_write);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0),
                  tRabgRab(_, n_block - 1), tRabsRab(_, smem_pipe_write.index()));
          }

          pipeline_v.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                tVgV(_, n_block - 1), tVsV(_, smem_pipe_write.index()));
        }
      }

      #pragma unroll 2
      for (; n_block > 0; --n_block) {
        pipeline_v.consumer_wait(smem_pipe_read);
        pipeline_vt.producer_acquire(smem_pipe_write);
        do_transpose_V(smem_pipe_read.index());
        pipeline_vt.producer_commit(smem_pipe_write);
        pipeline_v.consumer_release(smem_pipe_read);

        ++smem_pipe_write;
        ++smem_pipe_read;

        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
          pipeline_k.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                tKgK(_, n_block - 1), tKsK(_, smem_pipe_write.index()));

          if (Has_rab) {
            pipeline_rab.producer_acquire(smem_pipe_write);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0),
                  tRabgRab(_, n_block - 1), tRabsRab(_, smem_pipe_write.index()));
          }

          pipeline_v.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                tVgV(_, n_block - 1), tVsV(_, smem_pipe_write.index()));
        }
      }

      scheduler.prefetch_next_work(scheduler_params, work_tile_info);

      pipeline_v.consumer_wait(smem_pipe_read);
      if (n_block_max > kStages)
        pipeline_vt.producer_acquire(smem_pipe_write);
      do_transpose_V(smem_pipe_read.index());
      pipeline_vt.producer_commit(smem_pipe_write);
      pipeline_v.consumer_release(smem_pipe_read);

      ++smem_pipe_write;
      ++smem_pipe_read;
    } else {
      if (warp_idx_in_warpgroup == 0 && lane_predicate) {
        shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(mainloop_params.tma_load_Q.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType &>(shared_storage.barrier_Q), 0 /*mcast_mask*/), tQgQ, tQsQ);
        pipeline_v.producer_acquire(smem_pipe_write);
        copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
              tVgV(_, n_block), tVsV(_, smem_pipe_write.index()));
      }
      // With fp8 kernel, smem_o is in union with smem_v_out, so could use NamedBarrier
      // instead of ClusterBarrier. But this doesn't appear to have any benefit.
      shared_storage.barrier_O.wait((work_idx + 1) % 2);

      pipeline_v.consumer_wait(smem_pipe_read);
      do_transpose_V(smem_pipe_read.index());
      pipeline_vt.producer_commit(smem_pipe_write);
      pipeline_v.consumer_release(smem_pipe_read);

      ++smem_pipe_write;
      ++smem_pipe_read;
      --n_block;

      constexpr int extra_iterations = kStages - 1;
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < extra_iterations && n_block >= 0; ++iter) {
        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
          pipeline_k.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));

          if (Has_rab) {
            pipeline_rab.producer_acquire(smem_pipe_write);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0),
                  tRabgRab(_, n_block), tRabsRab(_, smem_pipe_write.index()));
          }

          pipeline_v.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                tVgV(_, n_block), tVsV(_, smem_pipe_write.index()));
        }

        pipeline_v.consumer_wait(smem_pipe_read);
        do_transpose_V(smem_pipe_read.index());
        pipeline_vt.producer_commit(smem_pipe_write);
        pipeline_v.consumer_release(smem_pipe_read);

        ++smem_pipe_write;
        ++smem_pipe_read;
        --n_block;
      }

      #pragma unroll 2
      for (; n_block >= 0; --n_block) {

        if (warp_idx_in_warpgroup == 0 && lane_predicate) {
          pipeline_k.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                tKgK(_, n_block), tKsK(_, smem_pipe_write.index()));

          if (Has_rab) {
            pipeline_rab.producer_acquire(smem_pipe_write);
            copy(mainloop_params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0),
                  tRabgRab(_, n_block), tRabsRab(_, smem_pipe_write.index()));
          }

          pipeline_v.producer_acquire(smem_pipe_write);
          copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv),
                tVgV(_, n_block), tVsV(_, smem_pipe_write.index()));
        }

        pipeline_v.consumer_wait(smem_pipe_read);
        pipeline_vt.producer_acquire(smem_pipe_write);
        do_transpose_V(smem_pipe_read.index());
        pipeline_vt.producer_commit(smem_pipe_write);
        pipeline_v.consumer_release(smem_pipe_read);

        ++smem_pipe_write;
        ++smem_pipe_read;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_rab, MainloopPipeline pipeline_v,
            PipelineState &smem_pipe_write_k, PipelineState &smem_pipe_write_rab, PipelineState &smem_pipe_write_v) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // Issue the epilogue waits
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
        * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
        * then would just be acquired since the phase was still inverted from make_producer_start_state
        */
      pipeline_k.producer_tail(smem_pipe_write_k);
      if (Has_rab) {
        pipeline_rab.producer_tail(smem_pipe_write_rab);
      }
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail_one_write(MainloopPipeline pipeline_k, MainloopPipeline pipeline_rab, MainloopPipeline pipeline_v,
                      PipelineState &smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // Issue the epilogue waits
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
        * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
        * then would just be acquired since the phase was still inverted from make_producer_start_state
        */
      pipeline_k.producer_tail(smem_pipe_write);
      if (Has_rab) {
        pipeline_rab.producer_tail(smem_pipe_write);
      }
      pipeline_v.producer_tail(smem_pipe_write);
    }
  }

  CUTLASS_DEVICE void
  warp_scheduler_barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + cutlass::canonical_warp_group_idx());
    }
  }

  CUTLASS_DEVICE void
  warp_scheduler_barrier_arrive() {
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup || NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
    if constexpr (NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup) {
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (3 - cutlass::canonical_warp_group_idx()));
    } else {
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (cutlass::canonical_warp_group_idx() <= 2 ? cutlass::canonical_warp_group_idx() + 1 : cutlass::canonical_warp_group_idx() + 1 - 3));
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + (cutlass::canonical_warp_group_idx() <= 1 ? cutlass::canonical_warp_group_idx() + 2 : cutlass::canonical_warp_group_idx() + 2 - 3));
    }
  }

  CUTLASS_DEVICE void
  mma_init() {
    // Tell producer (warp 0) that smem_q is ready
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads + Ktraits::NumProducerThreads, static_cast<int>(FwdNamedBarriers::QueryEmpty));
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup || NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
    if (cutlass::canonical_warp_group_idx() > 1) {
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 1);
    }
    if constexpr (NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup) {
      if (cutlass::canonical_warp_group_idx() > 2) {
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads, static_cast<int>(FwdNamedBarriers::WarpSchedulerWG1) - 1 + 2);
      }
    }
  }

  template <typename SharedStorage, typename FrgTensorO>
  CUTLASS_DEVICE void
  mma(Params const &mainloop_params,
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_rab,
      MainloopPipeline pipeline_v,
      PipelineState &smem_pipe_read_k,
      PipelineState &smem_pipe_read_rab,
      PipelineState &smem_pipe_read_v,
      FrgTensorO &tOrO,
      int n_block_max,
      int n_block_min,
      int n_masking_steps,
      bool is_jump,
      int n_block_history,
      int thread_idx,
      int work_idx,
      int m_block,
      SharedStorage &shared_storage,
      const Seqlen_traits &seqlen_traits_q,
      const Seqlen_traits &seqlen_traits_k) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sRab = make_tensor(make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});

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

    auto consumer_wait = [](auto &pipeline, auto &smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
    int const actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    int const actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int const actual_seqlen_h = Is_target ? seqlen_traits_k.actual_seq_len_h : actual_seqlen_k;
    int const actual_seqlen_c = Is_context ? seqlen_traits_k.actual_seq_len_c : 0;
    int const actual_seqlen_offset = Is_delta_q ? actual_seqlen_k - actual_seqlen_q : 0;
    int const max_seq_len_q = seqlen_traits_q.max_seq_len;

    int n_block = n_block_max - 1;

    auto col_limit_right = [&](int row) {
      return std::min(
          actual_seqlen_k,
          row + 1 + actual_seqlen_offset + mainloop_params.window_size_right);
    };

    auto col_limit_left = [&](int row) {
      return std::max(
          0,
          row + actual_seqlen_offset - mainloop_params.window_size_left);
    };

    auto apply_mask = [&](auto &tensor, int n_block) {
      static constexpr int Row = 0, Col = 1;
      Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
      Tensor tScS = threadMma0.partition_C(cS);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tensor); ++i) {
        int row = int(get<Row>(tScS(i))) + m_block * kBlockM;
        int col = int(get<Col>(tScS(i))) + n_block * kBlockN;
        if constexpr (!Is_causal && !Is_local) {
          if (col >= actual_seqlen_k) {
            tensor(i) = -INFINITY;
          }
        } else {
          if constexpr (Is_context) {
            if (row < actual_seqlen_c && col < actual_seqlen_h) {
              continue;
            }
          }
          // causal mask
          if (col >= col_limit_right(row)) {
            tensor(i) = -INFINITY;
          }
          if constexpr (Is_local) {
            if (col < col_limit_left(row)) {
              tensor(i) = -INFINITY;
            }
          }
          if constexpr (Is_target) {
            const int target_index = (row - actual_seqlen_h) * mainloop_params.target_group_size_inv;
            const int target_col_limit_left = actual_seqlen_h + target_index * mainloop_params.target_group_size;
            if (row >= actual_seqlen_h) {
              if (col < target_col_limit_left && col >= actual_seqlen_h) {
                tensor(i) = -INFINITY;
              }
            }
          }
        }
      }
    };

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_Q.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.barrier_Q.wait(work_idx % 2);
    }

    // compute Q @ K + rab
    Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
    auto smem_tiled_copy_rab = make_tiled_copy_C(typename Ktraits::SmemCopyAtomRab{}, tiled_mma0);
    auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(thread_idx);
    Tensor tSsRab = smem_thr_copy_rab.partition_S(sRab);
    Tensor tSrRab = make_tensor<Element>(partition_shape_C(tiled_mma0, select<0, 1>(TileShape_MNK{})));
    Tensor tSrRab_view = smem_thr_copy_rab.retile_D(tSrRab);
    static_assert(rank(tSrRab) == rank(tSrS));
    static_assert(size(tSrRab) == size(tSrS));
    static_assert(size(tSrRab) == size(tSrRab_view));

    if constexpr (IntraWGOverlap) {
      if (Has_rab) {
        consumer_wait(pipeline_rab, smem_pipe_read_rab);
        cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, smem_pipe_read_rab.index()), tSrRab_view(_, _, _));

        cute::copy(make_tensor(convert_type<ElementAccum>(tSrRab).data(), tSrS.layout()), tSrS);
        pipeline_rab.consumer_release(smem_pipe_read_rab);
        ++smem_pipe_read_rab;
      }

      consumer_wait(pipeline_k, smem_pipe_read_k);
      warp_scheduler_barrier_sync();
      flash::gemm</*zero_init=*/!Has_rab, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
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
      if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
        tma_store_wait<0>();
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
          shared_storage.barrier_O.arrive(cta_id, lane_predicate);
        }
      }
    }

    if constexpr (IntraWGOverlap) {
      Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), flash::convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout()));
      CUTLASS_PRAGMA_NO_UNROLL
      for (int masking_step = 0; n_block > n_block_min; ++masking_step, --n_block) {
        if (is_jump && masking_step == n_masking_steps - 1) {
          n_block = std::min(n_block, n_block_history);
        }
        const bool is_masking = masking_step < n_masking_steps - 1 || (n_block + 1) * kBlockN > actual_seqlen_h;
        if (Has_rab) {
          consumer_wait(pipeline_rab, smem_pipe_read_rab);
          cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, smem_pipe_read_rab.index()), tSrRab_view(_, _, _));
          cute::copy(make_tensor(convert_type<ElementAccum>(tSrRab).data(), tSrS.layout()), tSrS);

          pipeline_rab.consumer_release(smem_pipe_read_rab); // release Rab
          ++smem_pipe_read_rab;
        }
        consumer_wait(pipeline_k, smem_pipe_read_k);
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/!Has_rab, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        consumer_wait(pipeline_v, smem_pipe_read_v);

        // compute P @ V
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        warp_scheduler_barrier_arrive();
        warpgroup_wait<1>();
        pipeline_k.consumer_release(smem_pipe_read_k); // release K
        ++smem_pipe_read_k;

        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) *= mainloop_params.alpha;
        }
        if (Is_local || is_masking) {
          apply_mask(tSrS, n_block - 1);
        }
        fast_silu(tSrS);
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_v); // release V
        ++smem_pipe_read_v;
        cute::copy(make_tensor(convert_type<Element>(tSrS).data(), tOrP.layout()), tOrP);
      }

      // Tell warp 0 that smem_q is ready
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty));
      consumer_wait(pipeline_v, smem_pipe_read_v);
      flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
      warpgroup_wait<0>();
      pipeline_v.consumer_release(smem_pipe_read_v); // release V, otherwise producers will hang
      ++smem_pipe_read_v;
    } else {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int masking_step = 0; n_block >= n_block_min; ++masking_step, --n_block) {
        const bool is_masking = masking_step < n_masking_steps || (n_block + 1) * kBlockN > actual_seqlen_h;
        if (Has_rab) {
          consumer_wait(pipeline_rab, smem_pipe_read_rab);
          cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, smem_pipe_read_rab.index()), tSrRab_view(_, _, _));
          cute::copy(make_tensor(convert_type<ElementAccum>(tSrRab).data(), tSrS.layout()), tSrS);
          pipeline_rab.consumer_release(smem_pipe_read_rab); // release Rab
          ++smem_pipe_read_rab;
        }
        consumer_wait(pipeline_k, smem_pipe_read_k);
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/!Has_rab, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
        warp_scheduler_barrier_arrive();
        warpgroup_wait<0>();
        pipeline_k.consumer_release(smem_pipe_read_k); // release K
        ++smem_pipe_read_k;

        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) *= mainloop_params.alpha;
        }
        if (Is_local || is_masking) {
          apply_mask(tSrS, n_block);
        }
        fast_silu(tSrS);
        Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(), flash::convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout()));
        consumer_wait(pipeline_v, smem_pipe_read_v);
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_v); // release V
        ++smem_pipe_read_v;

        if (is_jump && masking_step == n_masking_steps - 1) {
          n_block = std::min(n_block, n_block_history);
        }
      }
      // Tell warp 0 that smem_q is ready
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(FwdNamedBarriers::QueryEmpty));
    }
    for (int i = 0; i < size(tOrO); ++i) {
      tOrO(i) /= max_seq_len_q;
    }
    return;
  }

  template <bool Delay_V_release = false, typename SharedStorage, typename FrgTensorO>
  CUTLASS_DEVICE void
  mma_fp8(Params const &mainloop_params,
          MainloopPipeline pipeline_k,
          MainloopPipeline pipeline_rab,
          MainloopPipelineNoTMA pipeline_vt,
          PipelineState &smem_pipe_read,
          PipelineState &smem_pipe_release,
          FrgTensorO &tOrO,
          int n_block_count,
          int thread_idx,
          int work_idx,
          int m_block,
          SharedStorage &shared_storage,
          const Seqlen_traits &seqlen_traits_q,
          const Seqlen_traits &seqlen_traits_k) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sRab = make_tensor(make_smem_ptr(shared_storage.smem_rab.data()), SmemLayoutRab{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v_out.data()), SmemLayoutVt{});

    typename Ktraits::TiledMma0 tiled_mma0;
    typename Ktraits::TiledMma1 tiled_mma1;
    auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
    auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);
    float descale_q = *mainloop_params.descale_q_ptr;
    float descale_k = *mainloop_params.descale_k_ptr;
    float descale_v = *mainloop_params.descale_v_ptr;

    // Allocate "fragments/descriptors" for first matmul.
    Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
    Tensor tSrK = threadMma0.partition_fragment_B(sK);
    // Allocate "fragments/descriptors" for second matmul.
    Tensor tOrV = threadMma1.partition_fragment_B(sVt);

    auto consumer_wait = [](auto &pipeline, auto &smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
    int const actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    int const max_seq_len_q = seqlen_traits_q.max_seq_len;
    int n_block = n_block_count - 1;

    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_Q.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.barrier_Q.wait(work_idx % 2);
    }

    Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));

    consumer_wait(pipeline_k, smem_pipe_read);
    warp_scheduler_barrier_sync();

    // compute Q @ K + rab
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
    if (work_idx != 0) {
      int lane_predicate = cute::elect_one_sync();
      if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
        tma_store_wait<0>();
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
          shared_storage.barrier_O.arrive(cta_id, lane_predicate);
        }
      }
    }
    warpgroup_wait<0>();
    warp_scheduler_barrier_arrive();
    pipeline_k.consumer_release(smem_pipe_read);
    for (int i = 0; i < size(tSrS); ++i) {
      tSrS(i) = tSrS(i) * descale_q * descale_k;
    }

    auto smem_tiled_copy_rab = make_tiled_copy_C(typename Ktraits::SmemCopyAtomRab{}, tiled_mma0);
    auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(thread_idx);
    Tensor tSsRab = smem_thr_copy_rab.partition_S(sRab);
    Tensor tSrRab = make_tensor<ElementRab>(partition_shape_C(tiled_mma0, select<0, 1>(TileShape_MNK{})));
    Tensor tSrRab_copy_view = smem_thr_copy_rab.retile_D(tSrRab);
    static_assert(rank(tSrRab) == rank(tSrS));
    static_assert(size(tSrRab) == size(tSrS));
    static_assert(size(tSrRab) == size(tSrRab_copy_view));
    if (Has_rab) {
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

    auto col_limit_right = [&](int row) {
      return std::min(
          actual_seqlen_k,
          row + 1 + mainloop_params.window_size_right);
    };

    auto col_limit_left = [&](int row) {
      return std::max(
          0,
          row - mainloop_params.window_size_left);
    };

    {
      static constexpr int Row = 0, Col = 1;
      Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
      Tensor tScS = threadMma0.partition_C(cS);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSrS); ++i) {
        int row = int(get<Row>(tScS(i))) + m_block * kBlockM;
        int col = int(get<Col>(tScS(i))) + n_block * kBlockN;
        if constexpr (!Is_causal && !Is_local) {
          if (col >= actual_seqlen_k) {
            tSrS(i) = -INFINITY;
          }
        } else {
          // causal mask
          if (col >= col_limit_right(row)) {
            tSrS(i) = -INFINITY;
          }
          if constexpr (Is_local) {
            if (col < col_limit_left(row)) {
              tSrS(i) = -INFINITY;
            }
          }
        }
      }
    }

    fast_silu(tSrS);
    Tensor tSrS_converted = make_tensor_like<Element>(tSrS);
    convert_type_safe(tSrS, tSrS_converted);
    Tensor tOrP = make_tensor(tSrS_converted.data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
    permute_regs_A_to_C(tOrP);

    consumer_wait(pipeline_vt, smem_pipe_read);
    flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
    if constexpr (!Delay_V_release) {
      pipeline_vt.consumer_release(smem_pipe_read);
    }

    ++smem_pipe_read;
    --n_block;
    constexpr int extra_iterations = !Is_causal ? kStages - 1 : cute::ceil_div(kBlockM, kBlockN);

    if constexpr (Is_causal) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < extra_iterations && n_block >= 0; ++iter, --n_block) {
        Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
        consumer_wait(pipeline_k, smem_pipe_read);
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) = tSrS(i) * descale_q * descale_k;
        }
        if (Has_rab) {
          consumer_wait(pipeline_rab, smem_pipe_read);
          cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, smem_pipe_read.index()), tSrRab_copy_view(_, _, _));
          for (int i = 0; i < size(tSrS); ++i) {
            tSrS(i) += static_cast<float>(tSrRab(i));
          }
        }
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) *= mainloop_params.alpha;
        }

        Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
        Tensor tScS = threadMma0.partition_C(cS);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS); ++i) {
          int row = int(get<0>(tScS(i))) + m_block * kBlockM;
          int col = int(get<1>(tScS(i))) + n_block * kBlockN;
          if (col >= col_limit_right(row)) {
            tSrS(i) = -INFINITY;
          }
        }
        fast_silu(tSrS);

        warp_scheduler_barrier_arrive();
        pipeline_k.consumer_release(smem_pipe_read);
        if (Has_rab) {
          cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::AddRabWG1) - 1 + cutlass::canonical_warp_group_idx());
          pipeline_rab.consumer_release(smem_pipe_read);
        }

        if constexpr (Delay_V_release) {
          pipeline_vt.consumer_release(smem_pipe_release);
          ++smem_pipe_release;
        }
        consumer_wait(pipeline_vt, smem_pipe_read);

        Tensor tSrS_converted = make_tensor_like<Element>(tSrS);
        convert_type_safe(tSrS, tSrS_converted);
        Tensor tOrP = make_tensor(tSrS_converted.data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
        permute_regs_A_to_C(tOrP);

        flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
        if constexpr (!Delay_V_release) {
          pipeline_vt.consumer_release(smem_pipe_read);
        }
        ++smem_pipe_read;
      }
    } else if constexpr (!Is_local) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < extra_iterations && n_block >= 0; ++iter, --n_block) {
        Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
        consumer_wait(pipeline_k, smem_pipe_read);
        if constexpr (Delay_V_release) {
          pipeline_vt.consumer_release(smem_pipe_release);
          ++smem_pipe_release;
        }
        warp_scheduler_barrier_sync();
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
        warp_scheduler_barrier_arrive();

        if constexpr (!Delay_V_release) {
          pipeline_k.consumer_release(smem_pipe_read);
        } else {
          consumer_wait(pipeline_vt, smem_pipe_read);
        }
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) = tSrS(i) * descale_q * descale_k;
        }
        if (Has_rab) {
          consumer_wait(pipeline_rab, smem_pipe_read);
          cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, smem_pipe_read.index()), tSrRab_copy_view(_, _, _));
          for (int i = 0; i < size(tSrS); ++i) {
            tSrS(i) += static_cast<float>(tSrRab(i));
          }
          cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::AddRabWG1) - 1 + cutlass::canonical_warp_group_idx());
          if constexpr (!Delay_V_release) {
            pipeline_rab.consumer_release(smem_pipe_read);
          }
        }
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) *= mainloop_params.alpha;
        }
        fast_silu(tSrS);

        Tensor tSrS_converted = make_tensor_like<Element>(tSrS);
        convert_type_safe(tSrS, tSrS_converted);
        Tensor tOrP = make_tensor(tSrS_converted.data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
        permute_regs_A_to_C(tOrP);

        if constexpr (Delay_V_release) {
          pipeline_k.consumer_release(smem_pipe_read);
          pipeline_rab.consumer_release(smem_pipe_read);
        } else {
          consumer_wait(pipeline_vt, smem_pipe_read);
        }
        flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
        if constexpr (!Delay_V_release) {
          pipeline_vt.consumer_release(smem_pipe_read);
        }
        ++smem_pipe_read;
      }
    }

    if constexpr (Delay_V_release) {
      warp_scheduler_barrier_sync();
      CUTLASS_PRAGMA_NO_UNROLL
      for (; n_block >= 0; --n_block) {
        Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
        consumer_wait(pipeline_k, smem_pipe_read);
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
        warp_scheduler_barrier_arrive();
        pipeline_k.consumer_release(smem_pipe_read);
        pipeline_vt.consumer_release(smem_pipe_release);
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) = tSrS(i) * descale_q * descale_k;
        }
        if (Has_rab) {
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

        if constexpr (Is_local) {
          Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
          Tensor tScS = threadMma0.partition_C(cS);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tSrS); ++i) {
            int row = int(get<0>(tScS(i))) + m_block * kBlockM;
            int col = int(get<1>(tScS(i))) + n_block * kBlockN;
            if (col >= col_limit_right(row) || col < col_limit_left(row)) {
              tSrS(i) = -INFINITY;
            }
          }
        }
        fast_silu(tSrS);

        Tensor tSrS_converted = make_tensor_like<Element>(tSrS);
        convert_type_safe(tSrS, tSrS_converted);
        Tensor tOrP = make_tensor(tSrS_converted.data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
        permute_regs_A_to_C(tOrP);

        consumer_wait(pipeline_vt, smem_pipe_read);
        flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
        warp_scheduler_barrier_sync();
        ++smem_pipe_read;
        ++smem_pipe_release;
      }
      warp_scheduler_barrier_arrive();
      pipeline_vt.consumer_release(smem_pipe_release);
      ++smem_pipe_release;
    } else {
      if constexpr (kHeadDim == 128) {
        warp_scheduler_barrier_sync();
      }
      CUTLASS_PRAGMA_NO_UNROLL
      for (; n_block >= 0; --n_block) {
        Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
        consumer_wait(pipeline_k, smem_pipe_read);
        if constexpr (kHeadDim == 256) {
          warp_scheduler_barrier_sync();
        }
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
        warp_scheduler_barrier_arrive();
        pipeline_k.consumer_release(smem_pipe_read);
        for (int i = 0; i < size(tSrS); ++i) {
          tSrS(i) = tSrS(i) * descale_q * descale_k;
        }
        if (Has_rab) {
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

        if constexpr (Is_local) {
          Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
          Tensor tScS = threadMma0.partition_C(cS);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tSrS); ++i) {
            int row = int(get<0>(tScS(i))) + m_block * kBlockM;
            int col = int(get<1>(tScS(i))) + n_block * kBlockN;
            if (col >= col_limit_right(row) || col < col_limit_left(row)) {
              tSrS(i) = -INFINITY;
            }
          }
        }
        fast_silu(tSrS);

        Tensor tSrS_converted = make_tensor_like<Element>(tSrS);
        convert_type_safe(tSrS, tSrS_converted);
        Tensor tOrP = make_tensor(tSrS_converted.data(), convert_layout_acc_Aregs_fp8(tSrS.layout()));
        permute_regs_A_to_C(tOrP);

        consumer_wait(pipeline_vt, smem_pipe_read);
        if constexpr (kHeadDim == 128) {
          warp_scheduler_barrier_sync();
        }
        flash::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma1, tOrP, tOrV(_, _, _, smem_pipe_read.index()), tOrO);
        pipeline_vt.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }
      if constexpr (kHeadDim == 128) {
        warp_scheduler_barrier_arrive();
      }
    }
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarpGroup, static_cast<int>(FwdNamedBarriers::QueryEmpty));
    for (int i = 0; i < size(tOrO); ++i) {
      tOrO(i) *= (descale_v / max_seq_len_q);
    }
    return;
  }
};

} // namespace flash
