/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/algorithm/clear.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor.hpp"
#include "cute/tensor_predicate.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/builders/sm90_gmma_builder.inl"
#include "cutlass/gemm/collective/fp8_accumulation.hpp"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/workspace.h"

namespace cutlass::gemm {
using namespace cute;

struct KernelTmaWarpSpecializedCooperativeFP8BlockScaling {};

template <
    int Stages_,
    class ClusterShape_ = Shape<_1, _1, _1>,
    class KernelSchedule = KernelTmaWarpSpecialized>
struct MainloopSm90TmaGmmaWarpSpecializedFP8BlockScaling
    : MainloopSm90TmaGmmaWarpSpecialized<
          Stages_,
          ClusterShape_,
          KernelSchedule> {
  static_assert(
      cute::is_same_v<
          KernelSchedule,
          KernelTmaWarpSpecializedCooperativeFP8BlockScaling>,
      "KernelSchedule must be one of the warp specialized policies");
};

namespace collective {

template <class EngineAccum, class LayoutAccum>
struct GmmaFP8BlockScalingAccumulation {
  using TensorAccum = cute::Tensor<EngineAccum, LayoutAccum>;

  static_assert(
      is_static<LayoutAccum>::value,
      "Accumulator Layout should be static");
  static_assert(
      is_rmem<TensorAccum>::value,
      "Accumulator tensor must be rmem resident.");

 private:
  TensorAccum& accum_;
  TensorAccum accum_temp_;

  float* scaling_factor_a_ptr_;
  float* scaling_factor_b_ptr_;
  const uint8_t tiles_per_scaling_block_;
  uint8_t tiles_done_in_current_scaling_block_{0};

 public:
  CUTLASS_DEVICE
  GmmaFP8BlockScalingAccumulation(
      TensorAccum& accum,
      float* scaling_factor_a_ptr,
      float* scaling_factor_b_ptr,
      uint8_t tiles_per_scaling_block)
      : accum_(accum),
        scaling_factor_a_ptr_(scaling_factor_a_ptr),
        scaling_factor_b_ptr_(scaling_factor_b_ptr),
        tiles_per_scaling_block_(tiles_per_scaling_block) {
    accum_temp_ = cute::make_fragment_like(accum);
  }

  CUTLASS_DEVICE
  TensorAccum& operator()() {
    return accum_temp_;
  }

  /// promote (add) the results from the MMA accumulators to main accumulator.
  CUTLASS_DEVICE
  void promote() {
    float scaling_factor_a = *scaling_factor_a_ptr_;
    float scaling_factor_b = *scaling_factor_b_ptr_;
    tiles_done_in_current_scaling_block_ += 1;
    uint32_t must_step_to_next_tile = __shfl_sync(
        0xffffffff,
        tiles_done_in_current_scaling_block_ == tiles_per_scaling_block_,
        0);
    if (must_step_to_next_tile) {
      scaling_factor_a_ptr_ += 1;
      scaling_factor_b_ptr_ += 1;
      tiles_done_in_current_scaling_block_ = 0;
    }
    warpgroup_wait<0>();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) = __fmaf_rn(
          accum_temp_(i), scaling_factor_a * scaling_factor_b, accum_(i));
    }
  }
};

template <
    int Stages,
    class ClusterShape,
    class KernelSchedule,
    class TileShape_,
    class ElementA_,
    class StrideA_,
    class ElementB_,
    class StrideB_,
    class TiledMma_,
    class GmemTiledCopyA_,
    class SmemLayoutAtomA_,
    class SmemCopyAtomA_,
    class TransformA_,
    class GmemTiledCopyB_,
    class SmemLayoutAtomB_,
    class SmemCopyAtomB_,
    class TransformB_>
struct CollectiveMma<
    MainloopSm90TmaGmmaWarpSpecializedFP8BlockScaling<
        Stages,
        ClusterShape,
        KernelSchedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedFP8BlockScaling<
      Stages,
      ClusterShape,
      KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;

  static_assert(
      cute::rank(SmemLayoutAtomA{}) == 2,
      "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(
      (size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(
      (size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(
      cute::rank(SmemLayoutAtomB{}) == 2,
      "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert(
      (size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(
      (size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0,
      "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(
          shape<0>(TileShape{}),
          shape<2>(TileShape{}),
          Int<DispatchPolicy::Stages>{}),
      conditional_t<
          ::cutlass::gemm::detail::is_major<0, StrideA>(),
          Step<_2, _1, _3>,
          Step<_1, _2, _3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(
          shape<1>(TileShape{}),
          shape<2>(TileShape{}),
          Int<DispatchPolicy::Stages>{}),
      conditional_t<
          ::cutlass::gemm::detail::is_major<0, StrideB>(),
          Step<_2, _1, _3>,
          Step<_1, _2, _3>>{}));

  static_assert(
      DispatchPolicy::Stages >= 2,
      "Specialization requires Stages set to value 1 or more.");
  static_assert(
      cute::is_base_of<
          cute::GMMA::DescriptorIterator,
          typename TiledMma::FrgTypeA>::value &&
          cute::is_base_of<
              cute::GMMA::DescriptorIterator,
              typename TiledMma::FrgTypeB>::value,
      "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(
      cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> ||
          cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(
      cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> ||
          cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128> {
      cute::array_aligned<
          typename TiledMma::ValTypeA,
          cute::cosize_v<SmemLayoutA>>
          smem_A;
      cute::array_aligned<
          typename TiledMma::ValTypeB,
          cute::cosize_v<SmemLayoutB>>
          smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    float* scaling_factor_a_ptr;
    float* scaling_factor_b_ptr;
    uint8_t tiles_per_scaling_block_m;
    uint8_t tiles_per_scaling_block_n;
    uint8_t tiles_per_scaling_block_k;
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(
            static_cast<ElementA const*>(nullptr),
            repeat_like(StrideA{}, int32_t(0)),
            StrideA{}),
        SmemLayoutA{}(_, _, 0),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{}))); // mcast along N mode for this M load, if any
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(
            static_cast<ElementB const*>(nullptr),
            repeat_like(StrideB{}, int32_t(0)),
            StrideB{}),
        SmemLayoutB{}(_, _, 0),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    float* scaling_factor_a_ptr = nullptr;
    float* scaling_factor_b_ptr = nullptr;
    uint8_t tiles_per_scaling_block_m = 1;
    uint8_t tiles_per_scaling_block_n = 1;
    uint8_t tiles_per_scaling_block_k = 1;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {
    (void)workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is
    // only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A = reinterpret_cast<ElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<ElementB const*>(args.ptr_B);

    Tensor tensor_a =
        make_tensor(ptr_A, make_layout(make_shape(M, K, L), args.dA));
    Tensor tensor_b =
        make_tensor(ptr_B, make_layout(make_shape(N, K, L), args.dB));
    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
    return {
        tma_load_a,
        tma_load_b,
        args.scaling_factor_a_ptr,
        args.scaling_factor_b_ptr,
        args.tiles_per_scaling_block_m,
        args.tiles_per_scaling_block_n,
        args.tiles_per_scaling_block_k};
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A =
        tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable &&
        cutlass::detail::check_alignment<min_tma_aligned_elements_A>(
                        cute::make_shape(M, K, L), StrideA{});
    constexpr int min_tma_aligned_elements_B =
        tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable &&
        cutlass::detail::check_alignment<min_tma_aligned_elements_B>(
                        cute::make_shape(N, K, L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytes =
      (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) *
       static_cast<uint32_t>(sizeof_bits<ElementA>::value)) /
          8 +
      (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) *
       static_cast<uint32_t>(sizeof_bits<ElementB>::value)) /
          8;

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
  /// performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_b.get_tma_descriptor());
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the
  /// contract Returned tuple must contain at least two elements, with the first
  /// two elements being: gA_mkl - The tma tensor, A after a local tile so it
  /// has shape  (BLK_M,BLK_K,m,k,l) gB_nkl - The tma tensor, B after a local
  /// tile so it has shape  (BLK_N,BLK_K,n,k,l)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto load_init(
      ProblemShape_MNKL const& problem_shape_MNKL,
      Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain
    // mapping Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(
        make_shape(M, K, L)); // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(
        make_shape(N, K, L)); // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(
        mA_mkl,
        TileShape{},
        make_coord(_, _, _),
        Step<_1, X, _1>{}); // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(
        mB_nkl,
        TileShape{},
        make_coord(_, _, _),
        Step<X, _1, _1>{}); // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <class TensorA, class TensorB, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter,
      int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(
          make_smem_ptr(shared_tensors.smem_A.data()),
          SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(
          make_smem_ptr(shared_tensors.smem_B.data()),
          SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
      uint2 cluster_local_block_id = {
          block_rank_in_cluster % cluster_shape_x,
          block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_a =
          mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b =
          mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_, _, m_coord, _, l_coord); // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_, _, n_coord, _, l_coord); // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB); // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB); // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout =
            Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) ->
                                                             // block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |=
              (uint16_t(1) << block_layout(
                   cluster_local_block_id.x, n, Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout =
            Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) ->
                                                             // block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |=
              (uint16_t(1) << block_layout(
                   m, cluster_local_block_id.y, Int<0>{}));
        }
      }

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for (; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier =
            pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(
            mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a),
            tAgA(_, _, _, *k_tile_iter),
            tAsA(_, _, _, write_stage));
        copy(
            mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b),
            tBgB(_, _, _, *k_tile_iter),
            tBsB(_, _, _, write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <class FrgTensorC, class BlockCoord>
  CUTLASS_DEVICE void mma(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      BlockCoord const& blk_coord,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    static_assert(
        is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(
        cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(
        cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(
        cute::is_void_v<SmemCopyAtomA>,
        "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(
        cute::is_void_v<SmemCopyAtomB>,
        "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

    Tensor sA = make_tensor(
        make_smem_ptr(shared_tensors.smem_A.data()),
        SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(
        make_smem_ptr(shared_tensors.smem_B.data()),
        SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thread_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum)); // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum)); // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB)); // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB)); // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA)); // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB)); // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert(
        (0 <= K_PIPE_MMAS) && (K_PIPE_MMAS < K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

    GmmaFP8BlockScalingAccumulation accumulation(
        accum,
        mainloop_params.scaling_factor_a_ptr +
            (m_coord / mainloop_params.tiles_per_scaling_block_m) *
                (k_tile_count / mainloop_params.tiles_per_scaling_block_k),
        mainloop_params.scaling_factor_b_ptr +
            (n_coord / mainloop_params.tiles_per_scaling_block_n) *
                (k_tile_count / mainloop_params.tiles_per_scaling_block_k),
        mainloop_params.tiles_per_scaling_block_k);
    warpgroup_fence_operand(accumulation());
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0;
         --k_tile_prologue) {
      // WAIT on smem_pipe_read until its data are available (phase bit flips
      // from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(
            tiled_mma,
            tCrA(_, _, k_block, read_stage),
            tCrB(_, _, k_block, read_stage),
            accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      accumulation.promote();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accumulation());
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      // WAIT on smem_pipe_read until its data are available (phase bit flips
      // from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      warpgroup_fence_operand(accumulation());
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(
            tiled_mma,
            tCrA(_, _, k_block, read_stage),
            tCrB(_, _, k_block, read_stage),
            accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to
      /// ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accumulation());

      accumulation.promote();

      pipeline.consumer_release(smem_pipe_release); // UNLOCK smem_pipe_release,
                                                    // done _computing_ on it

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accumulation());
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void mma_tail(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_release,
      int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release); // UNLOCK smem_pipe_release,
                                                    // done _computing_ on it
      ++smem_pipe_release;
    }
  }
};

template <
    class ElementA,
    class GmemLayoutA,
    int AlignmentA,
    class ElementB,
    class GmemLayoutB,
    int AlignmentB,
    class ElementAccumulator,
    class TileShape_MNK,
    class ClusterShape_MNK,
    class StageCountType,
    class KernelScheduleType>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
        (cute::is_same_v<
            KernelScheduleType,
            KernelTmaWarpSpecializedCooperativeFP8BlockScaling>) &&
        not detail::
            is_use_rmem_A<ElementA, GmemLayoutA, ElementB, GmemLayoutB>()>> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(
      cutlass::detail::dependent_false<ElementA>,
      "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(
      detail::is_aligned<
          ElementA,
          AlignmentA,
          ElementB,
          AlignmentB,
          detail::tma_alignment_bytes>(),
      "Should meet TMA alignment requirement\n");

  // For fp32 types, map to tf32 MMA value type
  using MmaElementA = cute::
      conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using MmaElementB = cute::
      conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr cute::GMMA::Major GmmaMajorA =
      detail::gmma_ss_tag_to_major_A<MmaElementA, GmemLayoutA>();
  static constexpr cute::GMMA::Major GmmaMajorB =
      detail::gmma_ss_tag_to_major_B<MmaElementB, GmemLayoutB>();

  using AtomLayoutMNK = cute::conditional_t<
      cute::is_same_v<
          KernelScheduleType,
          KernelTmaWarpSpecializedCooperativeFP8BlockScaling>,
      Layout<Shape<_2, _1, _1>>,
      Layout<Shape<_1, _1, _1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<
          MmaElementA,
          MmaElementB,
          ElementAccumulator,
          TileShape_MNK,
          GmmaMajorA,
          GmmaMajorB>(),
      AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(
      shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(
      shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
                                   GmmaMajorA,
                                   MmaElementA,
                                   decltype(cute::get<0>(TileShape_MNK{})),
                                   decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
                                   GmmaMajorB,
                                   MmaElementB,
                                   decltype(cute::get<1>(TileShape_MNK{})),
                                   decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<
      detail::sm90_smem_capacity_bytes,
      MmaElementA,
      MmaElementB,
      TileShape_MNK>(StageCountType{});
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedFP8BlockScaling<
      PipelineStages,
      ClusterShape_MNK,
      KernelScheduleType>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutA>,
      ElementB,
      TagToStrideB_t<GmemLayoutB>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity>;
};

} // namespace collective

namespace kernel {

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class GemmUniversal<
    ProblemShape_,
    CollectiveMainloop_,
    CollectiveEpilogue_,
    TileScheduler_,
    cute::enable_if_t<cute::is_same_v<
        KernelTmaWarpSpecializedCooperativeFP8BlockScaling,
        typename CollectiveMainloop_::DispatchPolicy::Schedule>>> {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(
      cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
      "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::
      TileSchedulerSelector<TileScheduler_, ArchTag, TileShape, ClusterShape>::
          Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups =
      CUTE_STATIC_V(size(TiledMma{})) / NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock =
      CUTE_STATIC_V(size(TiledMma{})) +
      (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  /// Register requirement for Load and Math WGs
  static constexpr uint32_t LoadRegisterRequirement = 40;
  static constexpr uint32_t MmaRegisterRequirement = 232;

  // 1 stage ordered sequence between mainloop and epilogue producer load
  // threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1, 2>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16> {
      using MainloopPipelineStorage =
          typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage =
          typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info;
    TileSchedulerParams scheduler;
    void* workspace;
  };

  // IF_SWAP_AB<T>::value will be true only if:
  //   class T has member SwapAB and T::SwapAB is true
  template <typename T, typename = void>
  struct IF_SWAP_AB {
    static constexpr bool value = false;
  };

  template <typename T>
  struct IF_SWAP_AB<T, void_t<decltype(T::SwapAB)>> {
    static constexpr bool value = T::SwapAB;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the
  // aliased type.
  static Params to_underlying_arguments(
      Arguments const& args,
      void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    if constexpr (IF_SWAP_AB<CollectiveMainloop>::value) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST(
          "  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(
          args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST(
        "to_underlying_arguments(): Setting persistent grid SM count to "
        << sm_count);

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* scheduler_workspace = workspace_ptr;
    workspace_offset += TileScheduler::
        template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(
        args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.
    // Therefore it will be used in separate reduction scheme for streamk case,
    // NumEpilogueSubTiles default value is 1, which means subtile will not be
    // used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        problem_shape_MNKL,
        TileShape{},
        ClusterShape{},
        hw_info,
        args.scheduler,
        scheduler_workspace,
        NumEpilogueSubTiles);

    return {
        args.mode,
        problem_shape,
        CollectiveMainloop::to_underlying_arguments(
            args.problem_shape, args.mainloop, mainloop_workspace),
        CollectiveEpilogue::to_underlying_arguments(
            args.problem_shape, args.epilogue, epilogue_workspace),
        hw_info,
        scheduler,
        workspace};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched &&
         cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &=
        CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static size_t get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    workspace_size += TileScheduler::
        template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler,
            args.problem_shape,
            args.hw_info,
            NumMmaWarpGroups,
            NumEpilogueSubTiles);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    workspace_size += CollectiveEpilogue::get_workspace_size(
        args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size, MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles =
        CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    status = TileScheduler::
        template initialize_workspace<ProblemShape, ElementAccumulator>(
            args.scheduler,
            workspace_ptr + workspace_offset,
            stream,
            args.problem_shape,
            args.hw_info,
            NumMmaWarpGroups,
            NumEpilogueSubTiles);
    workspace_offset += TileScheduler::
        template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler,
            args.problem_shape,
            args.hw_info,
            NumMmaWarpGroups,
            NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = CollectiveEpilogue::initialize_workspace(
        args.problem_shape,
        args.epilogue,
        workspace_ptr + workspace_offset,
        stream);
    workspace_offset += CollectiveEpilogue::get_workspace_size(
        args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset, MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread
    // blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order =
        params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN
        ? TileScheduler::RasterOrderOptions::AlongN
        : TileScheduler::RasterOrderOptions::AlongM;
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return TileScheduler::get_tiled_cta_shape_mnl(
        problem_shape_MNKL, TileShape{}, ClusterShape{});
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if !defined(__CUDA_ARCH_FEAT_SM90_ALL)
    printf(
        "ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(
        size(TiledMma{}) == 256,
        "Cooperative kernel must have TiledMMA operating using 256 threads.");
    static_assert(
        size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");

    static_assert(
        cute::rank(StrideA{}) == 3,
        "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(
        cute::rank(StrideB{}) == 3,
        "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(
        cute::rank(StrideC{}) == 3,
        "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(
        cute::rank(StrideD{}) == 3,
        "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the
     * same tile */
    enum class WarpGroupRole { Producer = 0, Consumer0 = 1, Consumer1 = 2 };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      Warp3 = 3
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % size(TiledMma{});
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::Mainloop) {
      mainloop_pipeline_params.role =
          MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 ||
        warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role =
          MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = size(TiledMma{});
    mainloop_pipeline_params.transaction_bytes =
        CollectiveMainloop::TmaTransactionBytes;
    MainloopPipeline mainloop_pipeline(
        shared_storage.pipelines.mainloop,
        mainloop_pipeline_params,
        ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 ||
        warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = size(TiledMma{});
    epi_load_pipeline_params.transaction_bytes =
        CollectiveEpilogue::TmaTransactionBytes;
    EpiLoadPipeline epi_load_pipeline(
        shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id =
        producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(
        shared_storage.pipelines.load_order, params_load_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via
    // scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state =
        cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state =
        cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state =
        cutlass::make_producer_start_state<EpiStorePipeline>();

    auto cluster_wait_fn = []() {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return []() { cute::cluster_wait(); };
      } else {
        __syncthreads();
        return []() {}; // do nothing
      }
    }();

    // Optionally append 1s until problem shape is rank-4 in case it is only
    // rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

    // Get the appropriate blocks for this thread block -- potential for thread
    // block locality
    TiledMma tiled_mma;
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)

    TileScheduler scheduler{params.scheduler};
    auto work_tile_info = scheduler.get_current_work();

    // In a warp specialized kernel, collectives expose data movement and
    // compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(
        params.epilogue, shared_storage.tensors.epilogue);

    // Prepare and partition the input tensors. Expects a tuple of tensors
    // where: get<0>(load_inputs) is the tma tensor A after local tiling so that
    // it has shape (BLK_M,BLK_K,m,k,l) get<1>(load_inputs) is the tma tensor B
    // after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
    auto load_inputs =
        collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(
        cute::tuple_size_v<decltype(load_inputs)> >= 2,
        "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        bool do_load_order_arrive = true;
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            work_tile_info = fetch_next_work(work_tile_info, scheduler);
            continue;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and
          // n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          // Get the number of K tiles to compute for this work as well as the
          // starting K tile offset of the work.
          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(
              work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start =
              TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(
              idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          collective_mainloop.load(
              params.mainloop,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              load_inputs,
              blk_coord,
              k_tile_iter,
              work_k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.mainloop);
          // Update starting pipeline state for the next tile
          mainloop_pipe_producer_state.advance(work_k_tile_count);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          // Get next work tile
          work_tile_info = fetch_next_work(work_tile_info, scheduler);
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(
            mainloop_pipeline, mainloop_pipe_producer_state);
      } // Mainloop Producer Warp End

      // Epilogue Producer Warp
      else if (
          producer_warp_role == ProducerWarpRole::Epilogue &&
          collective_epilogue.is_producer_load_needed()) {
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::requires_separate_reduction(params.scheduler)) {
            load_order_barrier.wait();
          }
          if (TileScheduler::compute_epilogue(
                  work_tile_info, params.scheduler)) {
            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and
            // n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            epi_load_pipe_producer_state = collective_epilogue.load(
                epi_load_pipeline,
                epi_load_pipe_producer_state,
                problem_shape_MNKL,
                blk_shape,
                blk_coord,
                tiled_mma,
                lane_idx,
                shared_storage.tensors.epilogue,
                work_tile_info.reduction_subtile_idx());
          }

          // Get next work tile
          work_tile_info = fetch_next_work(work_tile_info, scheduler);
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(
            epi_load_pipeline, epi_load_pipe_producer_state);
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (
        warp_group_role == WarpGroupRole::Consumer0 ||
        warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      // Do we potentially issue tail arrives for TMA stores, if epilogue load
      // is waiting for it
      bool do_store_tail = false;
      while (work_tile_info.is_valid()) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and
        // n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(
            work_tile_info, problem_shape_MNKL, blk_shape);

        // Allocate the accumulators for the (M,N) blk_shape
        //
        // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
        auto accumulators = partition_fragment_C(
            tiled_mma, take<0, 2>(blk_shape)); // (MMA,MMA_M,MMA_N)
        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
          collective_mainloop.mma(
              mainloop_pipeline,
              mainloop_pipe_consumer_state,
              accumulators,
              blk_coord,
              work_k_tile_count,
              mma_thread_idx,
              shared_storage.tensors.mainloop,
              params.mainloop);

          // Make sure the math instructions are done and free buffers before
          // entering the epilogue
          collective_mainloop.mma_tail(
              mainloop_pipeline,
              mainloop_pipe_consumer_state,
              work_k_tile_count);

          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(work_k_tile_count);
        }
        // Index of warp group within consumer warp groups
        int consumer_warp_group_idx =
            canonical_warp_group_idx() - NumLoadWarpGroups;

        // Perform reduction across splits, if needed
        TileScheduler::fixup(
            params.scheduler,
            work_tile_info,
            accumulators,
            NumMmaWarpGroups,
            consumer_warp_group_idx);

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
          // Epilogue and write to gD
          auto
              [epi_load_pipe_consumer_state_next,
               epi_store_pipe_producer_state_next] =
                  collective_epilogue.store(
                      epi_load_pipeline,
                      epi_load_pipe_consumer_state,
                      epi_store_pipeline,
                      epi_store_pipe_producer_state,
                      problem_shape_MNKL,
                      blk_shape,
                      blk_coord,
                      accumulators,
                      tiled_mma,
                      mma_thread_idx,
                      shared_storage.tensors.epilogue,
                      work_tile_info.reduction_subtile_idx());
          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
          epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
          do_store_tail = true;
        }

        // Get next work tile
        work_tile_info = fetch_next_work(work_tile_info, scheduler);
      } // Scheduler work fetch loop

      if (do_store_tail) {
        collective_epilogue.store_tail(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state);
      }
    } // Consumer Warp Groups End
#endif
  }

 private:
  // Kernel helper function to get next work unit
  CUTLASS_DEVICE
  typename TileScheduler::WorkTileInfo fetch_next_work(
      typename TileScheduler::WorkTileInfo& work_tile_info,
      TileScheduler& scheduler) const {
    // Check whether we should continue on with the current work unit. If this
    // is the case, the work unit will have been updated in
    // continue_current_work to reflect the new tile to be computed.
    if (scheduler.continue_current_work(work_tile_info)) {
      return work_tile_info;
    }

    // Get next work tile
    scheduler.advance_to_next_work();
    return scheduler.get_current_work();
  }
};

} // namespace kernel
} // namespace cutlass::gemm
