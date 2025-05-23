/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao. Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "epilogue_fwd_sm90_tma.hpp"
#include "hstu.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "tile_scheduler.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <typename Ktraits, typename TileScheduler, typename Seqlen_traits>
__global__ void __launch_bounds__(
    Ktraits::kNWarps* cutlass::NumThreadsPerWarp,
    1)
    compute_attn_ws(
        CUTE_GRID_CONSTANT
        typename CollectiveMainloopFwd<Ktraits, Seqlen_traits>::Params const
            mainloop_params,
        CUTE_GRID_CONSTANT
        typename CollectiveEpilogueFwd<Ktraits, Seqlen_traits>::Params const
            epilogue_params,
        CUTE_GRID_CONSTANT
        typename TileScheduler::Params const scheduler_params,
        Seqlen_traits seqlen_traits_q,
        Seqlen_traits seqlen_traits_k) {
  using Element = typename Ktraits::Element;
  using ElementAccum = typename Ktraits::ElementAccum;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
  static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr int kBlockN = Ktraits::kBlockN;

  static constexpr bool Is_causal = Ktraits::Is_causal;
  static constexpr bool Is_target = Ktraits::Is_target;
  static constexpr bool Is_context = Ktraits::Is_context;
  static constexpr bool Is_delta_q = Ktraits::Is_delta_q;
  static constexpr bool Is_local = Ktraits::Is_local;

  using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Seqlen_traits>;
  using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits, Seqlen_traits>;

  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  extern __shared__ char shared_memory[];
  auto& shared_storage =
      *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  // Issue Tma Descriptor Prefetch from a single thread
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
  }

  // Obtain warp index
  int const warp_group_thread_idx =
      threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  PipelineParams pipeline_params;
  pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0
      ? MainloopPipeline::ThreadCategory::Producer
      : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = NumMmaThreads;

  PipelineParams pipeline_params_rab;
  pipeline_params_rab.transaction_bytes =
      CollectiveMainloop::TmaTransactionBytesRab;
  pipeline_params_rab.role = warp_group_idx == 0
      ? MainloopPipeline::ThreadCategory::Producer
      : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params_rab.is_leader = warp_group_thread_idx == 0;
  pipeline_params_rab.num_consumers = NumMmaThreads;

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_Q.init(1 /*numThreads*/);
    shared_storage.barrier_O.init(size(ClusterShape{}) /*numThreads*/);
  }
  // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
  MainloopPipeline pipeline_k(
      shared_storage.pipeline_k, pipeline_params, ClusterShape{});
  MainloopPipeline pipeline_rab(
      shared_storage.pipeline_rab, pipeline_params_rab, Shape<_1, _1, _1>{});
  MainloopPipeline pipeline_v(
      shared_storage.pipeline_v, pipeline_params, ClusterShape{});

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  // We need this to guarantee that the Pipeline init is visible to all
  // producers and consumer blocks in the Cluster
  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }

  auto get_block_info = [&](int m_block, int bidb) {
    seqlen_traits_q.init(bidb);
    seqlen_traits_k.init(bidb);
    if constexpr (Is_target) {
      seqlen_traits_k.init_h(bidb);
    }
    if constexpr (Is_context) {
      seqlen_traits_k.init_c(bidb);
    }
    const int actual_seqlen_q = seqlen_traits_q.actual_seq_len;
    const int actual_seqlen_k = seqlen_traits_k.actual_seq_len;
    const int actual_seqlen_h =
        Is_target ? seqlen_traits_k.actual_seq_len_h : actual_seqlen_k;
    const int actual_seqlen_c =
        Is_context ? seqlen_traits_k.actual_seq_len_c : 0;
    const bool is_jump = Is_target && m_block * kBlockM > actual_seqlen_h;
    const bool is_in_context =
        Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
    const bool is_in_mixed_context = Is_context &&
        (m_block + 1) * kBlockM > actual_seqlen_c &&
        m_block * kBlockM < actual_seqlen_c;

    const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
    const int actual_seqlen_offset =
        Is_delta_q ? actual_seqlen_k - actual_seqlen_q : 0;
    const int target_index = (m_block * kBlockM - actual_seqlen_h) /
        mainloop_params.target_group_size;

    // calculate n_block_min and n_block_max
    int n_block_min = Is_local ? std::max(
                                     0,
                                     (m_block * kBlockM + actual_seqlen_offset -
                                      mainloop_params.window_size_left) /
                                         kBlockN)
                               : 0;
    int n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);
    if constexpr (Is_causal || Is_local) {
      n_block_max = std::min(
          n_block_max,
          cute::ceil_div(
              (m_block + 1) * kBlockM + actual_seqlen_offset +
                  mainloop_params.window_size_right,
              kBlockN));
    }
    if constexpr (Is_context) {
      n_block_min = (is_in_context || is_in_mixed_context) ? 0 : n_block_min;
      n_block_max = (is_in_context || is_in_mixed_context)
          ? std::max(cute::ceil_div(actual_seqlen_h, kBlockN), n_block_max)
          : n_block_max;
    }

    // calculate n_masking_block_max and n_masking_block_min
    int n_masking_block_max = cute::ceil_div(
        std::min(
            actual_seqlen_k, (m_block + 1) * kBlockM + actual_seqlen_offset),
        kBlockN);
    int n_masking_block_min =
        (m_block * kBlockM + actual_seqlen_offset) / kBlockN;
    if constexpr (Is_target) {
      n_masking_block_min = is_jump
          ? (actual_seqlen_h + actual_seqlen_offset +
             target_index * mainloop_params.target_group_size) /
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
    return std::make_tuple(
        n_block_max,
        n_block_min,
        n_masking_steps,
        is_jump,
        n_block_history,
        actual_seqlen_q);
  };

  static_assert(Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
  if (warp_group_idx == 0) {
    // Producer
    // TODO: Tune the reg num
    cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 12 ? 24 : 32>();

    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // Load Q, K, V
    if (warp_idx_in_warpgroup == 0) {
      PipelineState smem_pipe_write_k =
          cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_rab =
          cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_v =
          cutlass::make_producer_start_state<MainloopPipeline>();

      int work_idx = 0;

      TileScheduler scheduler(&shared_storage.tile_count_semaphore);
      for (auto work_tile_info = scheduler.get_initial_work();
           work_tile_info.is_valid(scheduler_params);
           work_tile_info =
               scheduler.template get_next_work</*IsProducer=*/true>(
                   scheduler_params, work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [m_block, bidh, bidb] = block_coord;

        auto
            [n_block_max,
             n_block_min,
             n_masking_steps,
             is_jump,
             n_block_history,
             actual_seqlen_q] = get_block_info(m_block, bidb);
        if (m_block * kBlockM >= actual_seqlen_q) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          continue;
        }

        if ((Is_causal || Is_local) && n_block_max <= n_block_min) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          continue;
        }
        collective_mainloop.load(
            mainloop_params,
            pipeline_k,
            pipeline_rab,
            pipeline_v,
            smem_pipe_write_k,
            smem_pipe_write_rab,
            smem_pipe_write_v,
            n_block_max,
            n_block_min,
            n_masking_steps,
            is_jump,
            n_block_history,
            shared_storage,
            scheduler,
            scheduler_params,
            work_tile_info,
            block_coord,
            work_idx,
            seqlen_traits_q,
            seqlen_traits_k);
        ++work_idx;
      }
      collective_mainloop.load_tail(
          pipeline_k,
          pipeline_rab,
          pipeline_v,
          smem_pipe_write_k,
          smem_pipe_write_rab,
          smem_pipe_write_v);
    }
  } else {
    // Consumer
    // TODO: Tune the reg num
    cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 240 : 160>();

    TileScheduler scheduler(&shared_storage.tile_count_semaphore);
    // Initialize matmul objects.
    typename Ktraits::TiledMma1 tiled_mma1;

    PipelineState smem_pipe_read_k, smem_pipe_read_rab, smem_pipe_read_v;
    // We don't need separate variables smem_pipe_release_k and
    // smem_pipe_release_v (like in Cutlass's gemm) because the read and release
    // pipeline states are always the same.

    collective_mainloop.mma_init();
    scheduler.init_consumer();

    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work();
         work_tile_info.is_valid(scheduler_params);
         work_tile_info =
             scheduler.template get_next_work</*IsProducer=*/false>(
                 scheduler_params, work_tile_info)) {
      // Attention output (GEMM-II) accumulator.
      Tensor tOrO =
          partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
      clear(tOrO);

      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [m_block, bidh, bidb] = block_coord;

      auto
          [n_block_max,
           n_block_min,
           n_masking_steps,
           is_jump,
           n_block_history,
           actual_seqlen_q] = get_block_info(m_block, bidb);
      if (m_block * kBlockM >= actual_seqlen_q) {
        continue;
      }

      if ((Is_causal || Is_local) && n_block_max <= n_block_min) {
        collective_epilogue.store_zero(
            epilogue_params,
            shared_storage,
            threadIdx.x - NumCopyThreads,
            block_coord,
            seqlen_traits_q);
        continue;
      }
      collective_mainloop.mma(
          mainloop_params,
          pipeline_k,
          pipeline_rab,
          pipeline_v,
          smem_pipe_read_k,
          smem_pipe_read_rab,
          smem_pipe_read_v,
          tOrO,
          n_block_max,
          n_block_min,
          n_masking_steps,
          is_jump,
          n_block_history,
          threadIdx.x - NumCopyThreads,
          work_idx,
          m_block,
          shared_storage,
          seqlen_traits_q,
          seqlen_traits_k);
      collective_epilogue.store(
          epilogue_params,
          tOrO,
          shared_storage,
          tiled_mma1,
          threadIdx.x - NumCopyThreads,
          block_coord,
          seqlen_traits_q);

      ++work_idx;
    }
    collective_epilogue.store_tail();
  }
}

template <typename Ktraits, typename TileScheduler, typename Seqlen_traits>
__global__ void __launch_bounds__(
    Ktraits::kNWarps* cutlass::NumThreadsPerWarp,
    1)
    compute_attn_ws_fp8(
        CUTE_GRID_CONSTANT
        typename CollectiveMainloopFwd<Ktraits, Seqlen_traits>::Params const
            mainloop_params,
        CUTE_GRID_CONSTANT
        typename CollectiveEpilogueFwd<Ktraits, Seqlen_traits>::Params const
            epilogue_params,
        CUTE_GRID_CONSTANT
        typename TileScheduler::Params const scheduler_params,
        Seqlen_traits seqlen_traits_q,
        Seqlen_traits seqlen_traits_k) {
  using Element = typename Ktraits::Element;
  static_assert(cutlass::sizeof_bits_v<Element> == 8);
  using ElementAccum = typename Ktraits::ElementAccum;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
  static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr bool Is_causal = Ktraits::Is_causal;
  static constexpr bool Delay_V_release = Is_causal && Ktraits::kHeadDim == 128;

  using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Seqlen_traits>;
  using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits, Seqlen_traits>;

  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using MainloopPipelineVt = typename Ktraits::MainloopPipelineNoTMA;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineParamsVt = typename MainloopPipelineVt::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  extern __shared__ char shared_memory[];
  auto& shared_storage =
      *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  // Issue Tma Descriptor Prefetch from a single thread
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
  }

  // Obtain warp index
  int const warp_group_thread_idx =
      threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  // additional pipeline to synchronize out-of-place smem transpose of V
  PipelineParamsVt pipeline_params_vt;
  pipeline_params_vt.producer_arv_count = NumCopyThreads;
  pipeline_params_vt.consumer_arv_count = NumMmaThreads;
  MainloopPipelineVt pipeline_vt(
      shared_storage.pipeline_vt, pipeline_params_vt);

  PipelineParams pipeline_params;
  pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0
      ? MainloopPipeline::ThreadCategory::Producer
      : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = NumMmaThreads;

  PipelineParams pipeline_params_rab;
  pipeline_params_rab.transaction_bytes =
      CollectiveMainloop::TmaTransactionBytesRab;
  pipeline_params_rab.role = warp_group_idx == 0
      ? MainloopPipeline::ThreadCategory::Producer
      : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params_rab.is_leader = warp_group_thread_idx == 0;
  pipeline_params_rab.num_consumers = NumMmaThreads;

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_Q.init(1 /*numThreads*/);
    shared_storage.barrier_O.init(size(ClusterShape{}) /*numThreads*/);
  }
  // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
  MainloopPipeline pipeline_k(
      shared_storage.pipeline_k, pipeline_params, ClusterShape{});
  MainloopPipeline pipeline_rab(
      shared_storage.pipeline_rab, pipeline_params_rab, Shape<_1, _1, _1>{});
  // pipeline_v has producer warpgroup for its consumer in fp8 kernel
  pipeline_params.num_consumers = NumCopyThreads;
  pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
  MainloopPipeline pipeline_v(
      shared_storage.pipeline_v, pipeline_params, ClusterShape{});

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  float descale_q = *mainloop_params.descale_q_ptr;
  float descale_k = *mainloop_params.descale_k_ptr;
  float descale_v = *mainloop_params.descale_v_ptr;

  // We need this to guarantee that the Pipeline init is visible to all
  // producers and consumer blocks in the Cluster
  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }

  static_assert(Ktraits::kNWarps == 12 || Ktraits::kNWarps == 16);
  if (warp_group_idx == 0) {
    // Producer
    cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 12 ? 40 : 32>();

    PipelineState smem_pipe_write =
        cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_read, smem_pipe_release;

    int work_idx = 0;

    TileScheduler scheduler(&shared_storage.tile_count_semaphore);
    for (auto work_tile_info = scheduler.get_initial_work();
         work_tile_info.is_valid(scheduler_params);
         work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(
             scheduler_params, work_tile_info)) {
      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [m_block, bidh, bidb] = block_coord;

      seqlen_traits_q.init(bidb);
      seqlen_traits_k.init(bidb);
      if (m_block * kBlockM >= seqlen_traits_q.actual_seq_len) {
        continue;
      }
      int n_block_max = collective_mainloop.get_n_block_max(
          mainloop_params, m_block, seqlen_traits_q, seqlen_traits_k);
      if constexpr (Is_causal) {
        if (n_block_max <= 0) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          // need to sync producer warpgroup
          cutlass::arch::NamedBarrier::sync(
              NumCopyThreads, static_cast<int>(FwdNamedBarriers::ProducerWG));
          continue;
        }
      }
      collective_mainloop.load_fp8(
          mainloop_params,
          pipeline_k,
          pipeline_rab,
          pipeline_v,
          pipeline_vt,
          smem_pipe_write,
          smem_pipe_read,
          shared_storage,
          scheduler,
          scheduler_params,
          work_tile_info,
          block_coord,
          work_idx,
          seqlen_traits_q,
          seqlen_traits_k);
      ++work_idx;
    }
    collective_mainloop.load_tail_one_write(
        pipeline_k, pipeline_rab, pipeline_v, smem_pipe_write);
  } else {
    // Consumer
    cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 232 : 160>();
    TileScheduler scheduler(&shared_storage.tile_count_semaphore);
    // Initialize matmul objects.
    typename Ktraits::TiledMma1 tiled_mma1;
    PipelineState smem_pipe_read;
    PipelineState smem_pipe_release;

    collective_mainloop.mma_init();
    scheduler.init_consumer();

    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work();
         work_tile_info.is_valid(scheduler_params);
         work_tile_info =
             scheduler.template get_next_work</*IsProducer=*/false>(
                 scheduler_params, work_tile_info)) {
      // Attention output (GEMM-II) accumulator.
      Tensor tOrO =
          partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));

      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [m_block, bidh, bidb] = block_coord;

      seqlen_traits_q.init(bidb);
      seqlen_traits_k.init(bidb);
      if (m_block * kBlockM >= seqlen_traits_q.actual_seq_len) {
        continue;
      }
      int n_block_max = collective_mainloop.get_n_block_max(
          mainloop_params, m_block, seqlen_traits_q, seqlen_traits_k);
      if constexpr (Is_causal) {
        if (n_block_max <= 0) {
          collective_epilogue.store_zero(
              epilogue_params,
              shared_storage,
              threadIdx.x - NumCopyThreads,
              block_coord,
              seqlen_traits_q);
          continue;
        }
      }

      collective_mainloop.mma_fp8<Delay_V_release>(
          mainloop_params,
          pipeline_k,
          pipeline_rab,
          pipeline_vt,
          smem_pipe_read,
          smem_pipe_release,
          tOrO,
          n_block_max,
          threadIdx.x - NumCopyThreads,
          work_idx,
          m_block,
          shared_storage,
          seqlen_traits_q,
          seqlen_traits_k);

#ifndef NO_FP8_COLUMN_PERMUTE
      collective_epilogue.store_fp8(
          epilogue_params,
          tOrO,
          shared_storage,
          tiled_mma1,
          threadIdx.x - NumCopyThreads,
          block_coord,
          seqlen_traits_q);
#else
      collective_epilogue.store(
          epilogue_params,
          tOrO,
          shared_storage,
          tiled_mma1,
          threadIdx.x - NumCopyThreads,
          block_coord,
          seqlen_traits_q);
#endif
      ++work_idx;
    }
    collective_epilogue.store_tail();
  }
}

} // namespace flash
