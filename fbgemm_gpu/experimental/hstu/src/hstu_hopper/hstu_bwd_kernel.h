
/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "utils.h"
#include "tile_scheduler_bwd.hpp"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_bwd_sm90_tma.hpp"

namespace flash {

using namespace cute;

template <typename Ktraits, typename TileScheduler, typename Seqlen_traits>
__global__ void __launch_bounds__(Ktraits::kNWarps *cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT typename CollectiveMainloopBwd<Ktraits, Seqlen_traits>::Params const mainloop_params,
                    CUTE_GRID_CONSTANT typename CollectiveEpilogueBwd<Ktraits, Seqlen_traits>::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params,
                    Seqlen_traits seqlen_traits_q, Seqlen_traits seqlen_traits_k) {
  static constexpr int NumMmaThreads = Ktraits::NumMmaThreads;
  static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr int kBlockN = Ktraits::kBlockN;
  using ClusterShape = typename Ktraits::ClusterShape;

  static constexpr bool Is_causal = Ktraits::Is_causal;
  static constexpr bool Is_target = Ktraits::Is_target;
  static constexpr bool Is_context = Ktraits::Is_context;
  static constexpr bool Is_delta_q = Ktraits::Is_delta_q;
  static constexpr bool Is_local = Ktraits::Is_local;
  static constexpr bool Has_drab = Ktraits::Has_drab;

  using CollectiveMainloop = CollectiveMainloopBwd<Ktraits, Seqlen_traits>;
  using CollectiveEpilogue = CollectiveEpilogueBwd<Ktraits, Seqlen_traits>;

  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipeline_dO = typename Ktraits::MainloopPipeline_dO;
  using PipelineParams_dO = typename MainloopPipeline_dO::Params;
  using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
  static constexpr bool Q_dO_same_stages = std::is_same_v<MainloopPipeline, MainloopPipeline_dO>;
  using MainloopPipeline_dRab = typename Ktraits::MainloopPipeline_dRab;
  using PipelineParams_dRab = typename MainloopPipeline_dRab::Params;
  using PipelineState_dRab = typename MainloopPipeline_dRab::PipelineState;

  extern __shared__ char shared_memory[];
  auto &shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage *>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  // Issue Tma Descriptor Prefetch from a single thread
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
  }

  // Obtain warp index
  int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  PipelineParams pipeline_params;
  pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0
      ? MainloopPipeline::ThreadCategory::Producer
      : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = NumMmaThreads;

  PipelineParams pipeline_params_rab;
  pipeline_params_rab.transaction_bytes = CollectiveMainloop::TmaTransactionBytesRab;
  pipeline_params_rab.role = warp_group_idx == 0
      ? MainloopPipeline::ThreadCategory::Producer
      : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params_rab.is_leader = warp_group_thread_idx == 0;
  pipeline_params_rab.num_consumers = NumMmaThreads;

  PipelineParams_dRab pipeline_params_drab;
  pipeline_params_drab.producer_arv_count = NumMmaThreads;
  pipeline_params_drab.consumer_arv_count = cutlass::NumThreadsPerWarp;
  pipeline_params_drab.role = warp_group_idx == 0
      ? MainloopPipeline_dRab::ThreadCategory::Consumer  // TMA warp store dRab
      : MainloopPipeline_dRab::ThreadCategory::Producer; // MMA warps produce dRab

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_KV.init(1 /*numThreads*/);
  }
  // We're counting on pipeline_q to call cutlass::arch::fence_barrier_init();
  MainloopPipeline pipeline_q(shared_storage.pipeline_q, pipeline_params, ClusterShape{});
  auto role_dO = warp_group_idx == 0
      ? MainloopPipeline_dO::ThreadCategory::Producer
      : MainloopPipeline_dO::ThreadCategory::Consumer;
  PipelineParams_dO pipeline_params_dO {pipeline_params.transaction_bytes, role_dO, pipeline_params.is_leader, pipeline_params.num_consumers};
  MainloopPipeline_dO pipeline_do(shared_storage.pipeline_do, cute::conditional_return<Q_dO_same_stages>(pipeline_params, pipeline_params_dO), ClusterShape{});
  MainloopPipeline pipeline_rab(shared_storage.pipeline_rab, pipeline_params_rab, ClusterShape{});
  MainloopPipeline_dRab pipeline_drab(shared_storage.pipeline_drab, pipeline_params_drab);

  // Register requirement for Load and Math WGs
  static constexpr uint32_t LoadRegisterRequirement = Ktraits::NumMmaWarpGroups == 2 ? 24 : 32;
  static constexpr uint32_t MmaRegisterRequirement = Ktraits::NumMmaWarpGroups == 2 ? 240 : 160;

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }

  auto get_block_info = [&](int n_block, int bidb) {
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
    const int actual_seqlen_h = Is_target ? seqlen_traits_k.actual_seq_len_h : actual_seqlen_k;
    const int actual_seqlen_c = Is_context ? seqlen_traits_k.actual_seq_len_c : 0;
    const int m_block_context = Is_context ? cute::ceil_div(actual_seqlen_c, kBlockM) : 0;

    const bool is_jump = Is_target && n_block * kBlockN >= actual_seqlen_h;
    const bool is_in_context = Is_context && actual_seqlen_c > 0 && n_block * kBlockN <= actual_seqlen_h;

    const int actual_seqlen_offset = Is_delta_q ? actual_seqlen_k - actual_seqlen_q : 0;
    const int target_index = cute::ceil_div((n_block + 1) * kBlockN - actual_seqlen_h, mainloop_params.target_group_size);

    // calculate m_masking_block_min and m_masking_block_max
    int m_masking_block_max = cute::ceil_div(std::max(actual_seqlen_c, (n_block + 1) * kBlockN - actual_seqlen_offset), kBlockM);
    if constexpr (Is_target) {
      m_masking_block_max = std::max(m_masking_block_max,
          cute::ceil_div(actual_seqlen_h - actual_seqlen_offset + target_index * mainloop_params.target_group_size, kBlockM));
      m_masking_block_max = std::min(m_masking_block_max, cute::ceil_div(actual_seqlen_q, kBlockM));
      const bool is_mixed_target = (n_block + 1) * kBlockN > actual_seqlen_h && n_block * kBlockN < actual_seqlen_h;
      if (is_mixed_target) {
        m_masking_block_max = cute::ceil_div(actual_seqlen_q, kBlockM);
      }
    }
    const int m_masking_block_min = std::max(0, n_block * kBlockN - actual_seqlen_offset) / kBlockM;
    const int m_masking_steps = Is_causal ? m_masking_block_max - m_masking_block_min : 1;

    // calculate m_block_min and m_block_max
    int m_block_min = (!Is_causal && !Is_local) ? 0
            : std::max(0, (n_block * kBlockN - actual_seqlen_offset - mainloop_params.window_size_right) / kBlockM);
    int m_block_max = cute::ceil_div(actual_seqlen_q, kBlockM);
    if constexpr (Is_local) {
      m_block_max = std::min(
          m_block_max,
          cute::ceil_div((n_block + 1) * kBlockN - actual_seqlen_offset + mainloop_params.window_size_left, kBlockM));
    }
    if constexpr (Is_target) {
      m_block_max = is_jump ? m_masking_block_max : m_block_max;
    }
    return std::make_tuple(m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context);
  };

  if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_warpgroup == 0) {  // Load K, V, and do TMA on Q and dO
      PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState_dO smem_pipe_write_do = cutlass::make_producer_start_state<MainloopPipeline_dO>();

      TileScheduler scheduler;
      for (auto work_tile_info = scheduler.get_initial_work();
            work_tile_info.is_valid(scheduler_params);
            work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params, work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [n_block, bidh, bidb] = block_coord;

        auto [m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context] = get_block_info(n_block, bidb);

        if (m_block_min >= m_block_max) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          continue;
        }
        auto scheduler_prefetch = [&scheduler, &scheduler_params, &work_tile_info]() {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
        };
        collective_mainloop.load(mainloop_params, pipeline_q, pipeline_rab, pipeline_do, smem_pipe_write,
                                  smem_pipe_write_do, shared_storage, scheduler_prefetch, block_coord,
                                  m_block_min, m_block_max, is_in_context, m_block_context, seqlen_traits_q, seqlen_traits_k);
      }
      collective_mainloop.load_tail(pipeline_q, pipeline_rab, pipeline_do, smem_pipe_write, smem_pipe_write_do);
    } else if (warp_idx_in_warpgroup == 1) {
      TileScheduler scheduler;
      for (auto work_tile_info = scheduler.get_initial_work();
            work_tile_info.is_valid(scheduler_params);
            work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [n_block, bidh, bidb] = block_coord;

        auto [m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context] = get_block_info(n_block, bidb);

        if (m_block_min >= m_block_max) { continue; }
        collective_mainloop.store_dq(mainloop_params, shared_storage, block_coord,
                                      m_block_min, m_block_max, is_in_context, m_block_context, seqlen_traits_q);
      }
    } else if (Has_drab && warp_idx_in_warpgroup == 2) {
      PipelineState_dRab smem_pipe_read_dRab;
      TileScheduler scheduler;
      for (auto work_tile_info = scheduler.get_initial_work();
            work_tile_info.is_valid(scheduler_params);
            work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [n_block, bidh, bidb] = block_coord;

        auto [m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context] = get_block_info(n_block, bidb);

        if (m_block_min >= m_block_max) { continue; }
        collective_mainloop.store_drab(mainloop_params, pipeline_drab, smem_pipe_read_dRab, shared_storage, block_coord,
                                      m_block_min, m_block_max, is_in_context, m_block_context, seqlen_traits_q, seqlen_traits_k);
      }
    }
  } else {  // Consumer
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

    TileScheduler scheduler;
    // Initialize matmul objects.
    typename Ktraits::TiledMmadKV tiled_mma_dKV;

    PipelineState smem_pipe_read;
    PipelineState_dO smem_pipe_read_do;
    PipelineState_dRab smem_pipe_write_dRab = cutlass::make_producer_start_state<MainloopPipeline_dRab>();

    collective_mainloop.mma_init();
    scheduler.init_consumer();

    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work();
          work_tile_info.is_valid(scheduler_params);
          work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params, work_tile_info)) {
      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [n_block, bidh, bidb] = block_coord;

      auto [m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context] = get_block_info(n_block, bidb);

      if (m_block_min >= m_block_max) {  // We exit early and write 0 to dK and dV
        collective_epilogue.store_zero(epilogue_params, threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_k);
        continue;
      }

      // dK and dV output accumulator.
      static constexpr bool dKV_swapAB = Ktraits::dKV_swapAB;
      using TileShape_MNK = typename Ktraits::TileShape_MNK;
      Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
      Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB? 2 : 1>(TileShape_MNK{}));
      collective_mainloop.mma(mainloop_params, pipeline_q, pipeline_rab, pipeline_do, pipeline_drab,
                              smem_pipe_read, smem_pipe_read_do, smem_pipe_write_dRab,
                              tdKrdK, tdVrdV, m_block_min, m_block_max, m_masking_steps, is_in_context, m_block_context,
                              threadIdx.x - NumCopyThreads, work_idx, block_coord, shared_storage, seqlen_traits_q, seqlen_traits_k);
      collective_epilogue.store(epilogue_params, tdKrdK, tdVrdV, shared_storage, tiled_mma_dKV,
                                threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_k);
      ++work_idx;
    }
  }
}

} // namespace flash
