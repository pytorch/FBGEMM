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

#include "cute/tensor.hpp"

#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"  // For device_kernel

#include "static_switch.h"
#include "hstu.h"
#include "hstu_bwd_postprocess_kernel.h"
#include "tile_scheduler_bwd.hpp"
#include "hstu_bwd_kernel.h"
#include "kernel_traits.h"
#include "seq_len.h"

using namespace cute;

template <int Arch, typename Kernel_traits>
void run_hstu_bwd(Hstu_bwd_params &params, cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = float;
  static constexpr int kBlockM = Kernel_traits::kBlockM;
  static constexpr bool Is_delta_q = Kernel_traits::Is_delta_q;
  int const total_q_padded_rounded = cute::round_up(params.total_q + params.b * kBlockM, kBlockM);

  using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
  using TileShape_MK = typename Kernel_traits::TileShape_MK;
  using ClusterShape = typename Kernel_traits::ClusterShape;
  using Seqlen_traits = flash::VarSeqLenTraits;
  using CollectiveMainloop = flash::CollectiveMainloopBwd<Kernel_traits, Seqlen_traits>;
  using CollectiveEpilogue = flash::CollectiveEpilogueBwd<Kernel_traits, Seqlen_traits>;
  using Scheduler = flash::SingleTileSchedulerBwd<Kernel_traits::Is_balance_bwd>;
  Seqlen_traits seqlen_traits_q(
      params.total_q, params.seqlen_q, params.cu_seqlens_q, params.num_targets, params.num_contexts);
  Seqlen_traits seqlen_traits_k(
      params.total_k, params.seqlen_k, params.cu_seqlens_k, params.num_targets, params.num_contexts);
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments({
      static_cast<Element const*>(params.q_ptr),
      seqlen_traits_q.get_gmem_layout(
          params.seqlen_q, params.d, params.h, params.b,
          params.q_row_stride, params.q_head_stride
      ),  // layout_Q
      static_cast<Element const*>(params.rab_ptr),
      make_layout(make_shape(Is_delta_q ? params.seqlen_k : params.seqlen_q, params.seqlen_k, params.h_rab, params.b),
                  make_stride(params.rab_row_stride, _1{},
                              params.rab_head_stride,
                              params.rab_batch_stride)), // layout_Rab
      static_cast<Element const*>(params.k_ptr),
      seqlen_traits_k.get_gmem_layout(
          params.seqlen_k, params.d, params.h_k, params.b,
          params.k_row_stride, params.k_head_stride
      ),  // layout_K
      static_cast<Element const*>(params.v_ptr),
      seqlen_traits_k.get_gmem_layout(
          params.seqlen_k, params.d, params.h_k, params.b,
          params.v_row_stride, params.v_head_stride
      ),  // layout_V
      static_cast<Element const*>(params.do_ptr),
      make_layout(make_shape(params.total_q, params.d, params.h),
                  make_stride(params.do_row_stride, _1{}, params.do_head_stride)),  // layout_dO
      static_cast<ElementAccum*>(params.dq_accum_ptr),
      make_layout(make_shape(total_q_padded_rounded, params.d, params.h),
                  make_stride((int64_t)params.d, _1{}, (int64_t)params.d * total_q_padded_rounded)),  // layout_dQaccum
      static_cast<Element*>(params.drab_ptr),
      make_layout(make_shape(Is_delta_q ? params.seqlen_k : params.seqlen_q, params.seqlen_k, params.h_rab, params.b),
                  make_stride((int64_t)params.drab_row_stride, _1{}, (int64_t)params.drab_head_stride, (int64_t)params.drab_batch_stride)),  // layout_dRab
      params.b, params.window_size_left, params.window_size_right,
      params.target_group_size, 1.0 / params.target_group_size, params.alpha, params.dq_semaphore
  });

  typename CollectiveEpilogue::Params epilogue_params = CollectiveEpilogue::to_underlying_arguments({
      static_cast<Element*>(params.dk_ptr),
      seqlen_traits_k.get_gmem_layout(
          params.seqlen_k, params.d, params.h_k, params.b,
          params.dk_row_stride, params.dk_head_stride
      ),  // layout_dK
      static_cast<Element*>(params.dv_ptr),
      seqlen_traits_k.get_gmem_layout(
          params.seqlen_k, params.d, params.h_k, params.b,
          params.dv_row_stride, params.dv_head_stride
      )  // layout_dV
  });

  int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
  num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
  typename Scheduler::Arguments scheduler_args = {num_blocks_n, params.h, params.b};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  void *kernel = (void *) flash::compute_attn_ws<Kernel_traits, Scheduler, Seqlen_traits>;
  int smem_size = sizeof(typename Kernel_traits::SharedStorage);
  if (smem_size >= 48 * 1024) {
      CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  int device;
  cudaGetDevice(&device);
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args);
  static constexpr int ctaSize = Kernel_traits::kNWarps * cutlass::NumThreadsPerWarp;
  dim3 block_dims(ctaSize);
  dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
  cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
  cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params, epilogue_params, scheduler_args, seqlen_traits_q, seqlen_traits_k);
  CHECK_CUDA_KERNEL_LAUNCH();

  using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<Kernel_traits, Seqlen_traits>;
  typename PostprocessKernel::Params postprocess_params = PostprocessKernel::to_underlying_arguments({
      static_cast<ElementAccum const*>(params.dq_accum_ptr),
      make_layout(make_shape(total_q_padded_rounded, params.d, params.h),
                  make_stride((int64_t)params.d, _1{}, (int64_t)params.d * total_q_padded_rounded)),  // layout_dQaccum
      static_cast<Element*>(params.dq_ptr),
      seqlen_traits_q.get_gmem_layout(
          params.seqlen_q, params.d, params.h, params.b,
          params.dq_row_stride, params.dq_head_stride
      ),  // layout_dQ
      params.total_q,
      params.seqlen_q,
      params.cu_seqlens_q,
      params.num_targets,
      params.num_contexts
  });
  int num_m_block_postprocess = cute::ceil_div(params.seqlen_q, get<0>(TileShape_MK{}));
  dim3 grid_m_postprocess(num_m_block_postprocess, params.h, params.b);
  // Get the ptr to kernel function.
  auto postprocess_kernel = cutlass::device_kernel<PostprocessKernel>;
  int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
  if (smem_size_postprocess >= 48 * 1024) {
      CHECK_CUDA(cudaFuncSetAttribute(postprocess_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
  }
  postprocess_kernel<<<grid_m_postprocess, PostprocessKernel::MaxThreadsPerBlock, smem_size_postprocess, stream>>>(postprocess_params);
  CHECK_CUDA_KERNEL_LAUNCH();

}

template<int Arch, typename T, int Headdim, bool Has_rab, bool Has_drab, bool Is_local,
         bool Is_causal, bool Is_context, bool Is_target, bool Is_delta_q>
void run_hstu_bwd_(Hstu_bwd_params &params, cudaStream_t stream) {
  static constexpr auto tile_size = flash::get_tile_size_bwd<Headdim, Has_rab>();
  static constexpr int kBlockM = std::get<0>(tile_size);
  static constexpr int kBlockN = std::get<1>(tile_size);
  static constexpr int kNWarpGroups = std::get<2>(tile_size);
  // BOOL_SWITCH(params.deterministic, Deterministic, [&] {
  // BOOL_SWITCH(params.is_balance_bwd, Is_balance_bwd, [&] {
  static constexpr bool Is_balance_bwd = false;
  if constexpr (Headdim == 32) {
    run_hstu_bwd<Arch, Hstu_bwd_kernel_traits<Headdim, kBlockM, kBlockN, Is_causal, Is_context, Is_target, Is_delta_q, Is_local, Has_rab, Has_drab, false, 1, 2, true, false, false, kNWarpGroups, 1, 2, 1, Is_balance_bwd, T>
                  >(params, stream);
  } else if constexpr (Headdim == 64) {
    run_hstu_bwd<Arch, Hstu_bwd_kernel_traits<Headdim, kBlockM, kBlockN, Is_causal, Is_context, Is_target, Is_delta_q, Is_local, Has_rab, Has_drab, false, 1, 2, true, false, true, kNWarpGroups, 1, 2, 2, Is_balance_bwd, T>
                  >(params, stream);
  } else {
    run_hstu_bwd<Arch, Hstu_bwd_kernel_traits<Headdim, kBlockM, kBlockN, Is_causal, Is_context, Is_target, Is_delta_q, Is_local, Has_rab, Has_drab, false, 1, 2, false, true, true, kNWarpGroups, 1, 1, 1, Is_balance_bwd, T>
                  >(params, stream);
  }
  // });
  // });
}
