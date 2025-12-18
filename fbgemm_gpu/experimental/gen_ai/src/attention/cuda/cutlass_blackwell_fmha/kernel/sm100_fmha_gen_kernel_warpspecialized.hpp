// @nolint
/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "cutlass/cutlass.h"
#include "cute/layout.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"

#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "collective/fmha_fusion.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;
using namespace cutlass::fmha::collective;

struct Sm100FmhaGenKernelWarpspecializedSchedule {

  enum class WarpRole {
    Softmax0,
    Softmax1,
    Correction,
    MMA,
    Load,
    Epilogue,
    Empty
  };

  static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    if (warp_idx == 0) return WarpRole::Softmax0;     //   0 -  3
    if (warp_idx == 1) return WarpRole::MMA;         //       12
    if (warp_idx == 2 || warp_idx == 3) return WarpRole::Load;        //       13
    if (warp_idx == 4) return WarpRole::Softmax1;     //   4 -  7
    if (warp_idx == 8) return WarpRole::Correction;     //   8 - 11
    return WarpRole::Empty;                           //       15
  }

  static const int NumWarpsSoftmax = 1;
  static const int NumWarpsCorrection = 1;
  static const int NumWarpsEpilogue = 0;
  static const int NumWarpsLoad = 2;

  static const int NumRegsSoftmax = 192;
  static const int NumRegsCorrection = 104;
  static const int NumRegsOther = 248;
  static const int NumRegsEmpty = 24;

  static const int NumWarps = 12;

};

template<
  class ProblemShapeIn,
  class CollectiveMainloop,
  class CollectiveEpilogue,
  class TileScheduler,
  class KernelSchedule = Sm100FmhaGenKernelWarpspecializedSchedule
>
struct Sm100FmhaGenKernelWarpspecialized {

  using TileShape = typename CollectiveMainloop::TileShape;
  using ProblemShape = decltype(replace<0>(ProblemShapeIn{}, 0));

  using WarpRole = typename KernelSchedule::WarpRole;

  constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    return KernelSchedule::warp_idx_to_WarpRole(warp_idx);
  }

  static const int NumWarpsSoftmax = KernelSchedule::NumWarpsSoftmax;
  static const int NumWarpsCorrection = KernelSchedule::NumWarpsCorrection;
  static const int NumWarpsEpilogue = KernelSchedule::NumWarpsEpilogue;
  static const int NumWarpsLoad = KernelSchedule::NumWarpsLoad;

  static const int NumRegsSoftmax = KernelSchedule::NumRegsSoftmax;
  static const int NumRegsCorrection = KernelSchedule::NumRegsCorrection;
  static const int NumRegsOther = KernelSchedule::NumRegsOther;
  static const int NumRegsEmpty = 24;

  static const int NumWarps = KernelSchedule::NumWarps;

  using ClusterShape = typename CollectiveMainloop::ClusterShape;

  using TmemAllocator = cute::TMEM::Allocator1Sm;

  struct SharedStorage {
    typename CollectiveMainloop::TensorStorage mainloop;
    typename CollectiveEpilogue::TensorStorage epilogue;

    struct PipelineStorage {
      alignas(16) typename CollectiveMainloop::PipelineQ::SharedStorage load_q;
      alignas(16) typename CollectiveMainloop::PipelineKV::SharedStorage load_kv;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage mma_s0;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage mma_s1;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage s0_corr;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage s1_corr;
      alignas(16) typename CollectiveMainloop::PipelineO::SharedStorage mma_corr;
      alignas(16) typename CollectiveMainloop::PipelineE::SharedStorage corr_epi;
      alignas(16) typename CollectiveMainloop::OrderBarrierSoftmax::SharedStorage order_s01;
    } pipelines;

    uint32_t tmem_base_ptr;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  using StrideQOrig = typename CollectiveMainloop::StrideQOrig;
  using StrideOOrig = typename CollectiveMainloop::StrideOOrig;
  using StrideQ = typename CollectiveMainloop::StrideQ;
  using StrideO = typename CollectiveMainloop::StrideO;
  using StrideCacheK = typename CollectiveMainloop::StrideCacheK;
  using StrideCacheV = typename CollectiveMainloop::StrideCacheV;
  using StrideNewK = typename CollectiveMainloop::StrideNewK;
  using StrideNewV = typename CollectiveMainloop::StrideNewV;
  using Element = typename CollectiveMainloop::Element;
  using ElementAcc = typename CollectiveMainloop::ElementAcc;
  using ElementOut = typename CollectiveMainloop::ElementOut;

  struct Arguments {
    // _1, max_seqlen_k, head_dim, ((h_g, h_kv), b)
    ProblemShapeIn problem_shape;
    const int* seqlen_kv;
    const int* cache_batch_idx;
    int splitk_size = 0;
    int window_size = -1;  // If > 0, attend to last window_size tokens only

    const Element* ptr_q;  // 1 x D x (H x B)
    StrideQOrig dQ;
    const Element* ptr_new_k; // 1 x D x (H x B)
    StrideNewK dNewK;
    const Element* ptr_new_v; // 1 x D x (H x B)
    StrideNewV dNewV;
    
    Element* ptr_cache_k;  // seqlen_max x D x (H x B)
    StrideCacheK dCacheK;
    Element* ptr_cache_v;  // seqlen_max x D x (H x B)
    StrideCacheV dCacheV;
    ElementOut* ptr_o;     // 1 x D x (H x B)
    StrideOOrig dO;

    ElementAcc* ptr_LSE; // (B, H_K, H_R)
    cute::Stride<int, int, cute::_1> dLSE; // stride: (H_K*H_R, H_R, 1)

    cutlass::KernelHardwareInfo hw_info;

    ElementAcc scale_softmax = 0.0f;
  };

  struct Params {
    ProblemShape problem_shape;
    const int* seqlen_kv;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params tile_scheduler;
  };

  static const int MinBlocksPerMultiprocessor = 1;
  static const int MaxThreadsPerBlock = NumWarps * cutlass::NumThreadsPerWarp;
  using ArchTag = cutlass::arch::Sm100;

  static size_t get_workspace_size(Arguments const& args) { return 0; }
  static cutlass::Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return cutlass::Status::kSuccess;
  }

  static bool can_implement(Arguments const& args) {
    return true;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.tile_scheduler);
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    ProblemShape problem_shape = replace<0>(args.problem_shape, static_cast<int>(get<0>(args.problem_shape)));
    CUTE_STATIC_ASSERT_V(get<0>(args.problem_shape) == _1{});
    StrideQ dQ = replace<0>(args.dQ, 0);
    StrideO dO = replace<0>(args.dO, 0);
    get<0>(problem_shape) = get<3,0,0>(args.problem_shape);
    get<3,0,0>(problem_shape) = 1;
    get<0>(dQ) = get<2,0,0>(dQ);
    get<0>(dO) = get<2,0,0>(dO);

    typename CollectiveMainloop::Arguments mainloop_args {
      {
        args.cache_batch_idx,
        args.ptr_q, dQ,
        args.ptr_new_k, args.dNewK,
        args.ptr_new_v, args.dNewV,
        args.ptr_cache_k, args.dCacheK,
        args.ptr_cache_v, args.dCacheV,
      },
      args.scale_softmax,
      args.splitk_size,
      1.0f, 1.0f, 1.0f,  // scale_q, scale_k, scale_v
      1.0f,              // inv_scale_o
      args.window_size   // sliding window size
    };

    typename CollectiveEpilogue::Arguments epilogue_args {
      args.ptr_o, dO,
        args.ptr_LSE,
        args.dLSE,
    };

    return Params{
        problem_shape,
        args.seqlen_kv,
        CollectiveMainloop::to_underlying_arguments(problem_shape, mainloop_args, workspace),
        CollectiveEpilogue::to_underlying_arguments(problem_shape, epilogue_args, workspace),
        TileScheduler::to_underlying_arguments(problem_shape, args.hw_info, ClusterShape{}, TileShape{}, args.splitk_size, args.window_size)
    };
  }

  template<class BlkCoord>
  CUTLASS_DEVICE auto apply_batch(const Params &params, ProblemShape const& problem_shape, BlkCoord const& blk_coord) {
    int batch_idx = get<2,1>(blk_coord);
    ProblemShape result = problem_shape;
    
    int seqlen_kv = params.seqlen_kv[batch_idx];
    if (params.mainloop.load.ptr_new_k != nullptr) {
      seqlen_kv += 1;
    }

    // Cap at window_size if specified (sliding window attention)
    int effective_seqlen = seqlen_kv;
    if (params.mainloop.window_size > 0 && params.mainloop.window_size < seqlen_kv) {
      effective_seqlen = params.mainloop.window_size;
    }
    
    // For split-K: compute the split's klen
    int splitk_size = params.mainloop.splitk_size;
    if constexpr (!cute::is_constant<0, decltype(get<1>(blk_coord))>::value) {
      // Split-K mode: blk_coord has split_k_idx as an int
      if (splitk_size > 0) {
        int split_k_idx = get<1>(blk_coord);
        int start_k = split_k_idx * splitk_size;
        int end_k = cute::min(start_k + splitk_size, effective_seqlen);
        // Handle case where start_k >= effective_seqlen (split has no work)
        get<1>(result) = cute::max(0, end_k - start_k);
      } else {
        get<1>(result) = effective_seqlen;
      }
    } else {
      // Non-split-K mode: use effective sequence length
      get<1>(result) = effective_seqlen;
    }
    
    return result;
  }

  // Computes the global K/V offset for sliding window
  // Only called by Load warp
  template<class BlkCoord>
  CUTLASS_DEVICE int get_window_start_offset(
      const Params& params,
      BlkCoord const& blk_coord) {

    // If window disabled, no offset
    if (params.mainloop.window_size <= 0) {
      return 0;
    }

    int batch_idx = get<2, 1>(blk_coord);
    int batch_seqlen_kv = params.seqlen_kv[batch_idx];

    // Add 1 for new K/V if present
    if (params.mainloop.load.ptr_new_k != nullptr) {
      batch_seqlen_kv += 1;
    }

    // If window covers entire sequence, no offset
    if (params.mainloop.window_size >= batch_seqlen_kv) {
      return 0;
    }

    // Offset = batch_seqlen - window_size
    return batch_seqlen_kv - params.mainloop.window_size;
  }

  CUTLASS_DEVICE void operator()(const Params &params, char* smem) {
#if (! defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && \
     ! defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) && ! defined(CUTLASS_ARCH_MMA_SM103A_ENABLED))
    printf("ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.\n");
#else

    TileScheduler tile_scheduler{params.tile_scheduler};

    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto role = warp_idx_to_WarpRole(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();

    if (role == WarpRole::Load && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    if (role == WarpRole::Epilogue && lane_predicate) {
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    typename CollectiveMainloop::PipelineQ::Params pipeline_load_q_params;
    if (role == WarpRole::Load) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Producer;
    }
    if (role == WarpRole::MMA) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Consumer;
    }
    pipeline_load_q_params.producer_arv_count = NumWarpsLoad * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineQ pipeline_load_q(
      shared_storage.pipelines.load_q,
      pipeline_load_q_params,
      ClusterShape{},  cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineKV::Params pipeline_load_kv_params;
    if (role == WarpRole::Load) {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Producer;
    }
    if (role == WarpRole::MMA) {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Consumer;
    }
    pipeline_load_kv_params.producer_arv_count = NumWarpsLoad * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineKV pipeline_load_kv(
      shared_storage.pipelines.load_kv,
      pipeline_load_kv_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s0_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Softmax0) {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s0_params.consumer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineS pipeline_mma_s0(
      shared_storage.pipelines.mma_s0,
      pipeline_mma_s0_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s1_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Softmax1) {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s1_params.consumer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineS pipeline_mma_s1(
      shared_storage.pipelines.mma_s1,
      pipeline_mma_s1_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineC::Params pipeline_s0_corr_params;
    if (role == WarpRole::Softmax0) {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s0_corr_params.producer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    pipeline_s0_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineC pipeline_s0_corr(
      shared_storage.pipelines.s0_corr,
      pipeline_s0_corr_params,
      /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::PipelineC::Params pipeline_s1_corr_params;
    if (role == WarpRole::Softmax1) {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s1_corr_params.producer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    pipeline_s1_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineC pipeline_s1_corr(
      shared_storage.pipelines.s1_corr,
      pipeline_s1_corr_params,
      /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::PipelineO::Params pipeline_mma_corr_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Consumer;
    }
    pipeline_mma_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineO pipeline_mma_corr(
      shared_storage.pipelines.mma_corr,
      pipeline_mma_corr_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineE::Params pipeline_corr_epi_params;
    if (role == WarpRole::Correction) {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Producer;
    }
    if (role == WarpRole::Epilogue) {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Consumer;
    }
    pipeline_corr_epi_params.producer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    pipeline_corr_epi_params.consumer_arv_count = cute::max(1, NumWarpsEpilogue * cutlass::NumThreadsPerWarp);
    typename CollectiveMainloop::PipelineE pipeline_corr_epi(
      shared_storage.pipelines.corr_epi,
      pipeline_corr_epi_params,
      /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::OrderBarrierSoftmax::Params params_order_s01;
    params_order_s01.group_id = role == WarpRole::Softmax1 ? 1 : 0;
    params_order_s01.group_size = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::OrderBarrierSoftmax order_s01(
      shared_storage.pipelines.order_s01, params_order_s01);

    TmemAllocator tmem_allocator;

    __syncthreads();

    pipeline_load_q.init_masks(ClusterShape{});
    pipeline_load_kv.init_masks(ClusterShape{});
    pipeline_mma_s0.init_masks(ClusterShape{});
    pipeline_mma_s1.init_masks(ClusterShape{});
    pipeline_mma_corr.init_masks(ClusterShape{});

    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_consumer_state;
    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineQ>();

    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_consumer_state;
    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineKV>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_consumer_state;
    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineE>();

    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_consumer_state;
    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineO>();

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue(params.epilogue);

    if (role == WarpRole::Softmax0 || role == WarpRole::Softmax1) {
      warpgroup_reg_set<NumRegsSoftmax>();

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, blk_coord);

        // Skip if Q tile is out of bounds
        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        // Skip if there is no K/V sequence to attend to (seqlen_kv = 0 and no new K/V)
        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        bool is_softmax_0 = role == WarpRole::Softmax0;

        mainloop.softmax(
          is_softmax_0 ? 0 : 1, blk_coord,
          params.mainloop, logical_problem_shape,
          is_softmax_0 ? pipeline_mma_s0 : pipeline_mma_s1,
          is_softmax_0 ? pipeline_mma_s0_consumer_state : pipeline_mma_s1_consumer_state,
          is_softmax_0 ? pipeline_s0_corr : pipeline_s1_corr,
          is_softmax_0 ? pipeline_s0_corr_producer_state : pipeline_s1_corr_producer_state,
          order_s01
        );

      }
    }
    else if (role == WarpRole::Correction) {
      cutlass::arch::warpgroup_reg_dealloc<NumRegsCorrection>();

      // Track if any tile was valid (for lazy TMEM deallocation)
      bool has_valid_tile = false;

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, blk_coord);

        // Skip if Q tile is out of bounds
        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        // Handle empty splits (seqlen_kv = 0 for this split)
        // For varlen batches where seqlen_kv < max_seqlen_k, some splits may have no valid K/V data.
        // We must write neutral values (zeros for output, -inf for LSE) so merge produces correct results.
        if (get<1>(logical_problem_shape) == 0) {
          mainloop.write_empty_split(blk_coord, params.mainloop, params.problem_shape, epilogue);
          continue;
        }

        // This block has a tile with valid work
        has_valid_tile = true;

        mainloop.correction(
          blk_coord,
          params.mainloop, logical_problem_shape,
          shared_storage.epilogue,
          pipeline_s0_corr, pipeline_s0_corr_consumer_state,
          pipeline_s1_corr, pipeline_s1_corr_consumer_state,
          pipeline_mma_corr, pipeline_mma_corr_consumer_state,
          pipeline_corr_epi, pipeline_corr_epi_producer_state,
          epilogue
        );


      }

      // Lazy TMEM deallocation: Only free TMEM if it was allocated.
      // For empty splits (klen=0), MMA warp didn't allocate TMEM, so we skip freeing.
      // For non-empty splits, pipeline synchronization with MMA ensures the
      // tmem_base_ptr write is visible before we read it here.
      if constexpr (NumWarpsEpilogue == 0) {
        static_assert(NumWarpsCorrection == 1);

        if (has_valid_tile) {
          uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
          tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
        }
      }

    }
    else if (role == WarpRole::MMA) {
      warpgroup_reg_set<NumRegsOther>();

      // Track if any tile was valid (for lazy TMEM allocation)
      bool has_valid_tile = false;

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, blk_coord);

        // Skip if Q tile is out of bounds
        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        // Skip if there is no K/V sequence to attend to (seqlen_kv = 0 and no new K/V)
        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        // Lazy TMEM allocation: Only allocate on first valid tile
        if (!has_valid_tile) {
          has_valid_tile = true;
          tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
          __syncwarp();
        }

        mainloop.mma(
          blk_coord,
          params.mainloop, logical_problem_shape,
          shared_storage.mainloop,
          pipeline_load_q, pipeline_load_q_consumer_state,
          pipeline_load_kv, pipeline_load_kv_consumer_state,
          pipeline_mma_s0, pipeline_mma_s0_producer_state,
          pipeline_mma_s1, pipeline_mma_s1_producer_state,
          pipeline_mma_corr, pipeline_mma_corr_producer_state
        );


      }
    }
    else if (role == WarpRole::Load) {
      warpgroup_reg_set<NumRegsOther>();

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, blk_coord);

        // Skip if Q tile is out of bounds
        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        // Skip if there is no K/V sequence to attend to (seqlen_kv = 0 and no new K/V)
        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        // Compute window offset for sliding window attention
        int start_k_offset = get_window_start_offset(params, blk_coord);

        mainloop.load(
          blk_coord, logical_problem_shape,
          params.mainloop, params.problem_shape,
          shared_storage.mainloop,
          pipeline_load_q, pipeline_load_q_producer_state,
          pipeline_load_kv, pipeline_load_kv_producer_state,
          start_k_offset
        );

      }
    }
    else if (role == WarpRole::Epilogue) {
      warpgroup_reg_set<NumRegsOther>();

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, blk_coord);

        // Skip if Q tile is out of bounds
        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        // Skip if there is no K/V sequence to attend to (seqlen_kv = 0 and no new K/V)
        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        epilogue.store(
          blk_coord, logical_problem_shape,
          params.epilogue, params.problem_shape,
          shared_storage.epilogue,
          pipeline_corr_epi, pipeline_corr_epi_consumer_state
        );

      }

      static_assert(NumWarpsEpilogue <= 1);
      if constexpr (NumWarpsEpilogue == 1) {
        uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
        tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
      }

    }
    else if (role == WarpRole::Empty) {
      warpgroup_reg_set<NumRegsEmpty>();

      /* no-op, donate regs and exit */
    }
#endif
  }

};

}  // namespace cutlass::fmha::kernel
