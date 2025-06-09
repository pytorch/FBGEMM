/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 */

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cluster_launch.hpp"
#include "cutlass/cutlass.h"

#include "hstu.h"
#include "hstu_fwd_kernel.h"
#include "kernel_traits.h"
#include "seq_len.h"
#include "static_switch.h"
#include "tile_scheduler.hpp"
#include "utils.h"

template <int Arch, typename Kernel_traits>
void run_hstu_fwd(Hstu_fwd_params& params, cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using OutputType = typename Kernel_traits::OutputType;
  using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
  using ClusterShape = typename Kernel_traits::ClusterShape_MNK;
  static constexpr bool Is_delta_q = Kernel_traits::Is_delta_q;

  using Seqlen_traits = flash::VarSeqLenTraits;
  using CollectiveMainloop =
      flash::CollectiveMainloopFwd<Kernel_traits, Seqlen_traits>;
  using CollectiveEpilogue =
      flash::CollectiveEpilogueFwd<Kernel_traits, Seqlen_traits>;
  using Scheduler = std::conditional_t<
      cutlass::sizeof_bits_v<Element> == 8,
      flash::SingleTileScheduler<Kernel_traits::Is_balance_fwd>,
      flash::DynamicPersistentTileScheduler<
          Kernel_traits::kNThreads - cutlass::NumThreadsPerWarpGroup,
          Kernel_traits::NumProducerThreads>>;
  Seqlen_traits seqlen_traits_q(
      params.total_q,
      params.seqlen_q,
      params.cu_seqlens_q,
      params.num_targets,
      params.num_contexts);
  Seqlen_traits seqlen_traits_k(
      params.total_k,
      params.seqlen_k,
      params.cu_seqlens_k,
      params.num_targets,
      params.num_contexts);
  typename CollectiveMainloop::Params mainloop_params =
      CollectiveMainloop::to_underlying_arguments(
          {static_cast<Element const*>(params.q_ptr),
           seqlen_traits_q.get_gmem_layout(
               params.seqlen_q,
               params.d,
               params.h,
               params.b,
               params.q_row_stride,
               params.q_head_stride), // layout_Q
           static_cast<OutputType const*>(params.rab_ptr),
           make_layout(
               make_shape(
                   Is_delta_q ? params.seqlen_k : params.seqlen_q,
                   params.seqlen_k,
                   params.h_rab,
                   params.b),
               make_stride(
                   params.rab_row_stride,
                   _1{},
                   params.rab_head_stride,
                   params.rab_batch_stride)), // layout_Rab
           static_cast<Element const*>(params.k_ptr),
           seqlen_traits_k.get_gmem_layout(
               params.seqlen_k,
               params.d,
               params.h_k,
               params.b,
               params.k_row_stride,
               params.k_head_stride), // layout_K
           static_cast<Element const*>(params.v_ptr),
           seqlen_traits_k.get_gmem_layout(
               params.seqlen_k,
               params.d,
               params.h_k,
               params.b,
               params.v_row_stride,
               params.v_head_stride), // layout_V
           params.descale_q_ptr,
           params.descale_k_ptr,
           params.descale_v_ptr,
           params.window_size_left,
           params.window_size_right,
           params.target_group_size,
           1.0f / params.target_group_size,
           params.alpha});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          static_cast<OutputType*>(params.o_ptr),
          seqlen_traits_q.get_gmem_layout(
              params.seqlen_q,
              params.d,
              params.h,
              params.b,
              params.o_row_stride,
              params.o_head_stride) // layout_O
      });

  int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
  num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) *
      size<0>(ClusterShape{});
  typename Scheduler::Arguments scheduler_args = {
      num_blocks_m, params.h, params.b, params.tile_count_semaphore};
  typename Scheduler::Params scheduler_params =
      Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  void* kernel;
  if constexpr (cutlass::sizeof_bits_v<Element> == 8)
    kernel = (void*)
        flash::compute_attn_ws_fp8<Kernel_traits, Scheduler, Seqlen_traits>;
  else
    kernel =
        (void*)flash::compute_attn_ws<Kernel_traits, Scheduler, Seqlen_traits>;
  int smem_size = sizeof(typename Kernel_traits::SharedStorage);
  if (smem_size >= 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, params.num_sm);
  static constexpr int ctaSize =
      Kernel_traits::kNWarps * cutlass::NumThreadsPerWarp;
  dim3 block_dims(ctaSize);
  dim3 cluster_dims(
      size<0>(ClusterShape{}),
      size<1>(ClusterShape{}),
      size<2>(ClusterShape{}));
  cutlass::ClusterLaunchParams launch_params{
      grid_dims, block_dims, cluster_dims, smem_size, stream};
  cutlass::launch_kernel_on_cluster(
      launch_params,
      kernel,
      mainloop_params,
      epilogue_params,
      scheduler_params,
      seqlen_traits_q,
      seqlen_traits_k);
  CHECK_CUDA_KERNEL_LAUNCH();
}

template <
    int Arch,
    typename T,
    int Headdim,
    bool Has_rab,
    bool Is_local,
    bool Is_causal,
    bool Is_context,
    bool Is_target,
    bool Is_delta_q>
void run_hstu_fwd_(Hstu_fwd_params& params, cudaStream_t stream) {
  constexpr static bool Is_fp8 = cutlass::sizeof_bits_v<T> == 8;
  // BOOL_SWITCH(params.is_balance_fwd, Is_balance_fwd, [&] {
  static constexpr bool Is_balance_fwd = false;
  static constexpr auto tile_size =
      flash::get_tile_size_fwd<Headdim, Has_rab, Is_fp8>();
  static constexpr int kBlockM = std::get<0>(tile_size);
  static constexpr int kBlockN = std::get<1>(tile_size);
  static constexpr int kNWarps = std::get<2>(tile_size);
  if constexpr (Is_fp8) {
    run_hstu_fwd<
        90,
        Hstu_fwd_kernel_traits_fp8<
            Headdim,
            kBlockM,
            kBlockN,
            kNWarps,
            2,
            Is_causal,
            false,
            false,
            false,
            Is_local,
            Has_rab,
            1,
            Is_balance_fwd>>(params, stream);
  } else {
    run_hstu_fwd<
        Arch,
        Hstu_fwd_kernel_traits<
            Headdim,
            kBlockM,
            kBlockN,
            kNWarps,
            2,
            Is_causal,
            Is_context,
            Is_target,
            Is_delta_q,
            Is_local,
            Has_rab,
            1,
            Is_balance_fwd,
            T>>(params, stream);
  }
  // });
}
