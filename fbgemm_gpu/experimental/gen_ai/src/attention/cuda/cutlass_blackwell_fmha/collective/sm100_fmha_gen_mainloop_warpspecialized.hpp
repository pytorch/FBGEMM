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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/arch/simd_sm100.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_load_cpasync_warpspecialized.hpp"
#include "cutlass/detail/dependent_false.hpp"

namespace cutlass::fmha::collective {

using namespace cute;
using namespace constexpr_type_map;
using namespace constexpr_constexpr_map;

template<
  class Element_,
  class ElementQK_,
  class ElementPV_,
  class ElementOut_,
  class TileShape_,
  class StrideQ_,
  class StrideNewK_,
  class StrideNewV_,
  class StrideK_,
  class StrideV_,
  class StrideO_,
  class Mask_ = ResidualMask,
  // shape here is QG K H
  // and referes to the two softmax warps
  // (2, 1, 1) means that they are stacked (best for large Q since it loads the least K/V)
  // (1, 2, 1) means they sit side by side (best for small Q / large K)
  class ThreadShape = Shape<_1, _2, _1>
>
struct Sm100FmhaGenMainloopWarpspecialized {

  using Element = Element_;
  using ElementQK = ElementQK_;
  using ElementPV = ElementPV_;
  using ElementAcc = ElementPV_;
  using ElementOut = ElementOut_;
  using TileShape = TileShape_;
  using StrideQOrig = StrideQ_;
  using StrideQ = decltype(replace<0>(StrideQ_{}, 0));
  using StrideNewK = StrideNewK_;
  using StrideNewV = StrideNewV_;
  using StrideCacheK = StrideK_;
  using StrideCacheV = StrideV_;
  using StrideK = StrideK_;
  using StrideV = StrideV_;
  using StrideOOrig = StrideO_;
  using StrideO = decltype(replace<0>(StrideO_{}, 0));
  using Mask = Mask_;

  using TileM = decltype(get<0>(TileShape{})); // seq Q dim
  static_assert(TileM::value == 64 || TileM::value == 128, "Only expecting TileM to be 64 or 128");
  static constexpr int StageCountQ = get<1>(TileShape{}) == 256 ? 1 : 2;
  // Choose StageCountKV based on:
  // - Tile shape on the M (i.e., Query) dimension
  // - Element size
  using StageCountKVSelector = kValTyMap<
    void,
    kValTyPair<64,
      kValValMap<
        65536 /* default, arbitrarily large to trigger smem OOM error */,
        kValValPair<1, 12>, // fp8
        kValValPair<2, 6>  // bf16/fp16
      >>,
    kValTyPair<128,
      kValValMap<
        65536 /* default, arbitrarily large to trigger smem OOM error */,
        kValValPair<1, 11>, // fp8
        kValValPair<2, 5>   // bf16/fp16
      >>
  >;
  static constexpr int StageCountKV = StageCountQ *
    StageCountKVSelector::
      template query<TileM::value>::
      template query<sizeof(Element)>;
  
  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesKV = cutlass::gemm::collective::StageCount<StageCountKV>;
  
  using ClusterShape = Shape<_1, _1, _1>;

  static const int Alignment = 128 / sizeof_bits_v<Element>;

  using TileShapeQK = decltype(shape_div(TileShape{}, ThreadShape{}));

  using TileShapePV = decltype(select<0,2,1>(TileShapeQK{}));

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Element, StrideQ, Alignment,
      Element, StrideK, Alignment,
      ElementQK,
      TileShapeQK, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // the stride for A does not matter since we do not load from smem at all
      Element, StrideK, Alignment,
      Element, decltype(select<1,0,2>(StrideV{})), Alignment,
      ElementPV,
      TileShapePV, ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using SmemLayoutQ = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StageCountQ>{}));
  using SmemLayoutK = decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutB{}, Int<StageCountKV>{}));
  using SmemLayoutV = decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutB{}, Int<StageCountKV>{}));

  struct TensorStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    };
  };

  // indices for V0 / V1
  enum : int {
    kIdxOldRowMax = 0,
    kIdxNewRowMax = 1,
    kIdxFinalRowSum = 0,
    kIdxFinalRowMax = 1,
    kIdxStatsEnd = 2
  };

  // Each storage reserves kTMEM_V_COLUMNS for row max/sum stats
  // TileM=64 uses 16dp64b --> two threads processing a row
  // TileM=128 uses 32dp32b --> one thread processing a row
  using kTMEM_V_COLUMNS = typename kValTyMap<void,
      kValTyPair<64, Int<kIdxStatsEnd*2>>,
      kValTyPair<128, Int<kIdxStatsEnd>>
      >::template query<TileM::value>;

  // TMEM column allocation, offset will be used to calc the lower 16-bit of tmem addresses.
  // TMEM row/lane dimension is for the Q dim.
  enum class TmemAllocation : uint32_t {
    kSizeS = get<1>(TileShapeQK{}), // i.e., KV dim in a tile
    kSizeO = get<2>(TileShapeQK{}), // i.e., head dim
    // carve kSizeS to two parts: first 1/4 for V0/V1 stats storage; the rest for P0/P1
    // 1/4 is wasting some storage here but there seems to be column-wise address alignment requirements not found in spec.
    // Since there is enough storage left for P0/P1, chose to not debug alignment issues.
    kSizeV = kSizeS / 2,
    // P will be casted to the same type as V
    kSizeP = kSizeS * sizeof(Element) / sizeof(float),
    S0 = 0,
    S1 = S0 + kSizeS,
    V0 = S0,  // stats storage from softmax to correction
    V1 = S1,
    P0 = V0 + kSizeV,
    P1 = V1 + kSizeV,
    O0 = S1 + kSizeS,
    O1 = O0 + kSizeO,
    kEnd = O1 + kSizeO
  };
  static_assert(static_cast<uint32_t>(TmemAllocation::kEnd) <= 512, "Exceeds TMEM 512 columns");
  static_assert(
    static_cast<uint32_t>(TmemAllocation::kSizeV) + static_cast<uint32_t>(TmemAllocation::kSizeP) <=
    static_cast<uint32_t>(TmemAllocation::kSizeS),
    "Not enough storage to carve V and P out of S");
  static_assert(
    static_cast<uint32_t>(kTMEM_V_COLUMNS::value) <= static_cast<uint32_t>(TmemAllocation::kSizeV),
    "Not enough storage reserved for V");

  // from load to mma warp, protects q in smem
  using PipelineQ = cutlass::PipelineUmmaConsumerAsync<
    StageCountQ,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from load to mma warp, protects k/v in smem
  using PipelineKV = cutlass::PipelineUmmaConsumerAsync<
    StageCountKV,
    typename CollectiveMmaQK::AtomThrShapeMNK
  >;

  // from mma to softmax0/1 warp, protects S in tmem
  // (not sure yet about the reverse direction)
  // there is one pipe per softmax warp, and the mma warp alternates between them
  using PipelineS = cutlass::PipelineUmmaAsync<1>;

  // from softmax0/1/ to correction wg
  using PipelineC = cutlass::PipelineAsync<1>;

  // from mma to correction
  using PipelineO = cutlass::PipelineUmmaAsync<2>;

  // from corr to epilogue
  using PipelineE = cutlass::PipelineAsync<2>;

  using OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<
    /*stages*/ 1, /*groups*/ 2>;

  static_assert(cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>) == cutlass::bits_to_bytes(cosize(take<0,3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>), "K and V smem layouts must be of equal size");

  using Load = Sm100FmhaLoadCpAsyncWarpspecialized<
      Element, StrideQ, StrideNewK, StrideNewV, StrideCacheK, StrideCacheV,
      TensorStorage, CollectiveMmaQK, CollectiveMmaPV,
      SmemLayoutQ, SmemLayoutK, SmemLayoutV,
      PipelineQ, PipelineKV, TileShape, Mask
  >;
  
  struct Arguments {
    typename Load::Arguments load;

    // if zero, defaults to 1/sqrt(D)
    float scale_softmax = 0.0f;
    // Split-K size for sequence splitting
    int splitk_size = 0; 
    // scaling factors to dequantize QKV
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;

    // scaling factor to quantize O
    float inv_scale_o = 1.0f;
    
    // Sliding window size: Q only attends to the last window_size tokens of K
    // If <= 0, no windowing (attend to all K tokens)
    int window_size = -1;
  };

  struct Params {
    typename Load::Params load;

    float scale_softmax;
    float scale_softmax_log2;

    float scale_output;
    int splitk_size;  // Split-K size for sequence splitting
    int window_size;  // Sliding window size
  };

  template<class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {

    float scale_softmax = args.scale_softmax;
    if (scale_softmax == 0.0f) {
      scale_softmax = 1.0f / (float) std::sqrt(get<2>(problem_shape));
    }
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));

    return Params{
        Load::to_underlying_arguments(problem_shape, args.load, workspace),
        args.scale_q * args.scale_k * scale_softmax,
        args.scale_q * args.scale_k * log2_e * scale_softmax,
        args.scale_v * args.inv_scale_o,
        args.splitk_size,
        args.window_size
    };
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    Load::prefetch_tma_descriptors(params.load);
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      BlkCoord const& blk_coord, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state,
      int start_k_offset) {

    Load load;
    load.load(blk_coord, problem_shape, params.load, params_problem_shape,
        storage,
        pipeline_q, pipeline_q_producer_state,
        pipeline_kv, pipeline_kv_producer_state,
        params.splitk_size,
        start_k_offset);
  }

  template<class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  mma(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_consumer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_consumer_state,
      PipelineS& pipeline_s0, typename PipelineS::PipelineState& pipeline_s0_producer_state,
      PipelineS& pipeline_s1, typename PipelineS::PipelineState& pipeline_s1_producer_state,
      PipelineO& pipeline_corr, typename PipelineO::PipelineState& pipeline_corr_producer_state) {

    auto pipeline_q_release_state = pipeline_q_consumer_state;
    auto pipeline_kv_release_state = pipeline_kv_consumer_state;

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    typename CollectiveMmaQK::TiledMma mma_qk;
    ThrMMA thr_mma_qk = mma_qk.get_slice(0);

    typename CollectiveMmaPV::TiledMma mma_pv;
    TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
    ThrMMA thr_mma_pv = mma_pv_ts.get_slice(0);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(sK);
    Tensor tOrV = thr_mma_pv.make_fragment_B(sV);

    // tmem layout is
    // S0 S1`O0 O1
    // sequential in memory, where S overlaps with P and V

    Tensor tStS = partition_fragment_C(mma_qk, select<0,1>(TileShapeQK{}));
    Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0,1>(TileShapePV{}));

    Tensor tStS0 = tStS;
    tStS0.data() = tStS.data().get() + uint32_t(TmemAllocation::S0);
    Tensor tStS1 = tStS;
    tStS1.data() = tStS.data().get() + uint32_t(TmemAllocation::S1);

    Tensor tOtO0 = tOtO;
    tOtO0.data() = tOtO.data().get() + uint32_t(TmemAllocation::O0);
    Tensor tOtO1 = tOtO;
    tOtO1.data() = tOtO.data().get() + uint32_t(TmemAllocation::O1);

    Tensor sP = make_tensor(make_smem_ptr((Element*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
    Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});  // slice out staging

    Tensor tOrP0 = tOrP;
    tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAllocation::P0);
    Tensor tOrP1 = tOrP;
    tOrP1.data() = tOrP1.data().get() + uint32_t(TmemAllocation::P1);

    int k_index = 0;
    int v_index = 0;
    int q_index = 0;

    // wait for Q1
    q_index = pipeline_q_consumer_state.index();
    pipeline_q.consumer_wait(pipeline_q_consumer_state);
    ++pipeline_q_consumer_state;

    Tensor tSrQ0 = tSrQ(_,_,_,q_index);


    // wait for K1
    k_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // gemm Q1 * K1 -> S1
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    // release K1
    if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    // wait for Q2
    if constexpr (get<0>(ThreadShape{}) > 1 || get<2>(ThreadShape{}) > 1) {
      q_index = pipeline_q_consumer_state.index();
      pipeline_q.consumer_wait(pipeline_q_consumer_state);
      ++pipeline_q_consumer_state;
    }

    Tensor tSrQ1 = tSrQ(_,_,_,q_index);

    if constexpr (get<1>(ThreadShape{}) > 1) {
      k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    // gemm Q2 * K1 -> S2
    gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // release K1
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    // wait for V1
    v_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;

    // this acquire returns the ownership of all of S0 to the mma warp
    // including the P0 part
    // acquire corr first to take it out of the critical
    // path since softmax takes longer
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);

    // gemm P1 * V1 -> O1
    gemm_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

    if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      // wait for Ki
      k_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm Q1 * Ki -> S1
      gemm_zero_acc(mma_qk, tSrQ0, tSrK(_,_,_,k_index), tStS0);

      pipeline_s0.producer_commit(pipeline_s0_producer_state);
      ++pipeline_s0_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }

      // gemm P2 * V(i-1) -> O2
      if constexpr (get<1>(ThreadShape{}) > 1) {
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      pipeline_corr.producer_acquire(pipeline_corr_producer_state);
      pipeline_s1.producer_acquire(pipeline_s1_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      // release V(i-1)
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        k_index = (pipeline_kv_consumer_state.index());
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      // gemm Q2 * Ki -> S2
      gemm_zero_acc(mma_qk, tSrQ1, tSrK(_,_,_,k_index), tStS1);

      pipeline_s1.producer_commit(pipeline_s1_producer_state);
      ++pipeline_s1_producer_state;

      // release Ki
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;

      // wait for Vi
      v_index = (pipeline_kv_consumer_state.index());
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;

      // gemm P1 * Vi -> O1
      pipeline_corr.producer_acquire(pipeline_corr_producer_state);

      pipeline_s0.producer_acquire(pipeline_s0_producer_state);

      gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_,_,_,v_index), tOtO0);

      pipeline_corr.producer_commit(pipeline_corr_producer_state);
      ++pipeline_corr_producer_state;

      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }
    }

    // release Q1
    pipeline_q.consumer_release(pipeline_q_release_state);
    ++pipeline_q_release_state;

    // release Q2
    if constexpr (get<0>(ThreadShape{}) > 1) {
      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;
    }

    // wait for Vi
    if constexpr (get<1>(ThreadShape{}) > 1) {
      v_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    // gemm P2 * Vi -> O2
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);
    pipeline_s1.producer_acquire(pipeline_s1_producer_state);

    gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_,_,_,v_index), tOtO1);

    pipeline_corr.producer_commit(pipeline_corr_producer_state);
    ++pipeline_corr_producer_state;

    // release Vi
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // T0 S00 B1, T0 S10 B1, T0 S00 B2, T0 S01 B1, T0 S10 B2, T0 S11 B1, T0 S01 B2, T1 S00 B1, T0 S11 B2, ...
    // Q1 * K1  , Q2 * K1  , S11 * V1 , Q1 * K2  , S21 * V1  , Q2 * K2 , S12 * V2 , Q1 * K3  , S22 * K2 , ...
  }

  template<bool need_apply_mask, class Stage, class BlkCoord, class CoordTensor, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax_step(
      float& row_max, float& row_sum,
      Stage stage, bool final_call,
      BlkCoord const& blk_coord, CoordTensor const& cS,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {
    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);
          Tensor tScS =
        typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));
    tStS.data() = uint32_t(stage == _0{} ? TmemAllocation::S0 : TmemAllocation::S1);

    Tensor tStS_v = 
        tStS.compose(make_layout(make_shape(TileM{}, kTMEM_V_COLUMNS{})));
    tStS_v.data() = 
        uint32_t(stage == _0{} ? TmemAllocation::V0 : TmemAllocation::V1);
    Tensor tScS_v = 
        tScS.compose(make_layout(make_shape(TileM{}, kTMEM_V_COLUMNS{})));

    auto tilePlikeFP32 = size<1>(TileShapeQK{}) / Int<sizeof(float)>{} * Int<sizeof(Element)>{};
    Tensor tStS_P = tStS.compose(
      make_layout(make_shape(TileM{}, tilePlikeFP32)));
    tStS_P.data() = warp_uniform(
      uint32_t(stage == _0{} ? TmemAllocation::P0 : TmemAllocation::P1));
    Tensor tScS_P = tScS.compose(
      make_layout(make_shape(TileM{}, tilePlikeFP32)));
  
    using TMEM_LOAD_1xOP = typename kValTyMap<void,
        // Two threads collaborate on a single row
        kValTyPair<64, SM100_TMEM_LOAD_16dp32b1x>,
        // Each thread owns a single row
        kValTyPair<128, SM100_TMEM_LOAD_32dp32b1x>
      >::template query<TileM::value>;
    using TMEM_STORE_1xOP = decltype(TMEM::tmem_load_to_store(TMEM_LOAD_1xOP{}));

    // Summary of per-thread compute logic, each thread process all (TileM=128)/half(TileM=64) of a row:
    // 1. Read all data going to processed by this thread from one row of S=QK.
    // 2. Compute the "row_max" of that row of S.
    // 3. Loop through that row of S via `kTMEMOpsPerRow` number of TMEM_LOAD/STORE
    // instructions, each processing `kTmemLoadNcells` elements:
    //   3.1 P=softmax(S)
    //   3.2 Convert P's dtype from ElementQK -> Element, each conversion step processes `kConversionsPerStep` elements.
    //   3.3 Store the converted P to TMEM, unblocks MMA_PV.
    //
    // Considerations for `kTMEMOpsPerRow`: it determines the intermediate
    // registers required to perform "3.1 softmax" and "3.2 conversion", in
    // addition to the input data (a row of S).

    constexpr int kTMEMOpsPerRow = kValValMap<0 /* default, trigger static_assert error*/,
        kValValPair<64, 1>,  // each thread processes half of a row.
        kValValPair<128, 2>  // each thread processes an entire row, but convert+store half at a time to reduce reg pressure.
      >::template query<TileM::value>;
    static_assert(size<1>(TileShapeQK{}) % kTMEMOpsPerRow == 0);
    constexpr int kTmemLoadNcells = size<1>(TileShapeQK{}) / kTMEMOpsPerRow;
    constexpr int kTmemStoreNcells = kTmemLoadNcells * sizeof_bits_v<Element> / sizeof_bits_v<float>;
    using TMEM_LOAD = decltype(TMEM::op_repeater<TMEM_LOAD_1xOP, kTmemLoadNcells * 32>());
    using TMEM_STORE = decltype(TMEM::op_repeater<TMEM_STORE_1xOP, kTmemStoreNcells * 32>());
    using TMEM_STORE_V = typename kValTyMap<void,
        kValTyPair<64, SM100_TMEM_STORE_16dp32b2x>,
        kValTyPair<128, SM100_TMEM_STORE_32dp32b2x> // 4x32 threads with 2 cols of 32b elem
      >::template query<TileM::value>;

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tStS);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);

    Tensor tTMEM_LOADtS = thr_tmem_load.partition_S(tStS);
    Tensor tTMEM_LOADcS = thr_tmem_load.partition_D(tScS);

    auto tiled_tmem_storev = make_tmem_copy(TMEM_STORE_V{}, tStS_v);
    auto thr_tmem_storev  = tiled_tmem_storev.get_slice(thread_idx);

    Tensor tTMEM_STOREVtS = thr_tmem_storev.partition_D(tStS_v);
    Tensor tTMEM_STOREVcS = thr_tmem_storev.partition_S(tScS_v);

    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tStS_P);
    auto thr_tmem_store  = tiled_tmem_store.get_slice(thread_idx);

    Tensor tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P);
    tTMEM_STOREtS_x4.data() = warp_uniform(tTMEM_STOREtS_x4.data().get());
    Tensor tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P);

    // wait on tensor core pipe
    pipeline_s.consumer_wait(pipeline_s_consumer_state);

    // read all of S from tmem into reg mem
    Tensor tTMEM_LOADrS = make_tensor<ElementQK>(shape(tTMEM_LOADcS));
    copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS);

    if constexpr (need_apply_mask) {
      Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_shape);
    }

    ElementQK old_row_max = row_max;
    {
      // compute rowmax
      float row_max_0 = row_max;
      float row_max_1 = row_max;
      float row_max_2 = row_max;
      float row_max_3 = row_max;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTMEM_LOADrS); i += 4) {
        row_max_0  = ::fmax(row_max_0, tTMEM_LOADrS(i));
        row_max_1 = ::fmax(row_max_1, tTMEM_LOADrS(i+1));
        row_max_2 = ::fmax(row_max_2, tTMEM_LOADrS(i+2));
        row_max_3 = ::fmax(row_max_3, tTMEM_LOADrS(i+3));
      }
      row_max = ::fmax(row_max_0, row_max_1);
      row_max = ::fmax(row_max, row_max_2);
      row_max = ::fmax(row_max, row_max_3);
      if constexpr (TileM{} == 64) {
         ElementQK shuffled_row_max = __shfl_xor_sync(0xffffffff, row_max, 16);
          row_max = ::fmax(row_max, shuffled_row_max);
      }
    }
    ElementQK row_max_safe = row_max == -INFINITY ? 0 : row_max;

    Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
    static_assert(size(tTMEM_STOREVrS) == 2);
    tTMEM_STOREVrS(kIdxOldRowMax) = old_row_max;
    tTMEM_STOREVrS(kIdxNewRowMax) = row_max_safe;
    copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    // notify correction wg that they are ready (might need addtl ordering between S0 and S1 WG's)

    ElementQK scale = params.scale_softmax_log2;
    ElementQK row_max_scale = row_max_safe * scale;

    float2 scale_fp32x2 = make_float2(scale, scale);
    float2 minus_row_max_scale_fp32x2 = make_float2(-row_max_scale, -row_max_scale);

    Tensor tTMEM_STORErS_x4 = make_tensor<uint32_t>(shape(tTMEM_STOREcS));

    constexpr int kConversionsPerStep = 2;

    Tensor tTMEM_STORErS_x4_e = recast<Array<Element, kConversionsPerStep>>(tTMEM_STORErS_x4);

    NumericArrayConverter<Element, ElementQK, kConversionsPerStep> convert;
    const int kReleasePipeCount = 10;  // must be multiple of 2
    
    order_s.wait();
    static_assert(kReleasePipeCount % kConversionsPerStep == 0);
    static_assert(kConversionsPerStep == 2);

    {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tTMEM_LOADrS); i += kConversionsPerStep) {
        float2 in = make_float2(
          tTMEM_LOADrS(i + 0),
          tTMEM_LOADrS(i + 1)
        );

        float2 out;
        cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
        tTMEM_LOADrS(i + 0) = out.x;
        tTMEM_LOADrS(i + 1) = out.y;

        tTMEM_LOADrS(i+0) = ::exp2f(tTMEM_LOADrS(i+0));
        tTMEM_LOADrS(i+1) = ::exp2f(tTMEM_LOADrS(i+1));

        Array<ElementQK, kConversionsPerStep> in_conv;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kConversionsPerStep; j++) {
          in_conv[j] = tTMEM_LOADrS(i + j);
        }
        tTMEM_STORErS_x4_e[i / kConversionsPerStep] = convert(in_conv);


        if (i == size(tTMEM_LOADrS) - kReleasePipeCount) {
          order_s.arrive();
        }

        if constexpr (TileM::value == 128) {
            if constexpr (size<2>(tTMEM_STORErS_x4) == _2{}) {
            //this prevents register spills in fp16
            if (i == size(tTMEM_LOADrS) - 6) {
              copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, 0), tTMEM_STOREtS_x4(_, _, 0));
            }
          }
        }
      }
    } 

    // tmem_store(reg_S8) -> op_P
    CUTE_STATIC_ASSERT_V(size<2>(tTMEM_STORErS_x4) <= _2{});
    CUTE_STATIC_ASSERT_V(size<1>(tTMEM_STORErS_x4) == _1{});
    if constexpr (TileM::value == 128) {
      copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1), tTMEM_STOREtS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1));
    } else {
      copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4);
    }

    cutlass::arch::fence_view_async_tmem_store();

    // notify tensor core warp that P is ready
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    ElementQK acc_scale = (old_row_max == row_max_safe) ? 0.5f : 0.5f * ::exp2f(scale * (old_row_max - row_max_safe));
    row_sum *= acc_scale;
    // row_sum = sum(reg_S)
    float2 local_row_sum_f32x2 = make_float2(row_sum, row_sum);
    float2 local_row_sum_1 = make_float2(0, 0);
    float2 local_row_sum_2 = make_float2(0, 0);
    float2 local_row_sum_3 = make_float2(0, 0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tTMEM_LOADrS); i += 8) {
      // row_sum += tTMEM_LOADrS(i);
      float2 in = make_float2(tTMEM_LOADrS(i), tTMEM_LOADrS(i+1));
      cute::add(local_row_sum_f32x2, local_row_sum_f32x2, in);

      in = make_float2(tTMEM_LOADrS(i+2), tTMEM_LOADrS(i+2+1));
      cute::add(local_row_sum_1, local_row_sum_1, in);

      in = make_float2(tTMEM_LOADrS(i+4), tTMEM_LOADrS(i+4+1));
      cute::add(local_row_sum_2, local_row_sum_2, in);

      in = make_float2(tTMEM_LOADrS(i+6), tTMEM_LOADrS(i+6+1));
      cute::add(local_row_sum_3, local_row_sum_3, in);
    }

    cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_1);
    cute::add(local_row_sum_2, local_row_sum_2, local_row_sum_3);
    cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_2);
    float local_row_sum = local_row_sum_f32x2.x + local_row_sum_f32x2.y;
    
    row_sum = local_row_sum;

    if (final_call) {
      if constexpr (TileM{} == 64) {
          // Sync threads 0 and 16 to get the sum of row_sum between them
          row_sum += __shfl_xor_sync(0xffffffff, row_sum, 16);
      }

      // re-acquire the S part in the final step
      pipeline_s.consumer_wait(pipeline_s_consumer_state);
      

      Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
      tTMEM_STOREVrS(kIdxFinalRowMax) = row_max;
      tTMEM_STOREVrS(kIdxFinalRowSum) = row_sum;
      copy(tiled_tmem_storev, tTMEM_STOREVrS, tTMEM_STOREVtS);
    }
  }

  template<class Stage, class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  softmax(
      Stage stage,
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      PipelineS& pipeline_s, typename PipelineS::PipelineState& pipeline_s_consumer_state,
      PipelineC& pipeline_c, typename PipelineC::PipelineState& pipeline_c_producer_state,
      OrderBarrierSoftmax& order_s) {

    // problem_shape.klen is now the split's length (set by apply_batch)
    int mask_tile_count = Mask{}.get_unmasked_trip_count(blk_coord, TileShape{}, problem_shape);

    ElementQK row_max = -INFINITY;
    ElementQK row_sum = 0;

    // For split-K mode, coordinates should be local (0-based within the split)
    // since problem_shape.klen is already the split's length.
    // We do NOT add start_k offset here - the mask checks against problem_shape.klen
    // which is the split's length, so coordinates must be 0-based within the split.
    
    Tensor cS_base = make_identity_tensor(select<0,1>(TileShapeQK{}));
    auto logical_offset = make_coord(
        get<0>(blk_coord) * get<0>(TileShape{}) + (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{}),
        0 + (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{})
    );
    Tensor cS = domain_offset(logical_offset, cS_base);

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      softmax_step<false /* need_apply_mask */>(
          row_max, row_sum, stage,
          (mask_tile_count == 1) &&
              (Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape) == 0),
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }

    // Masked iterations
    mask_tile_count = Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      softmax_step<true /* need_apply_mask */>(
          row_max, row_sum, stage, mask_tile_count == 1,
          blk_coord, cS, params, problem_shape,
          pipeline_s, pipeline_s_consumer_state,
          pipeline_c, pipeline_c_producer_state,
          order_s
      );

      cS.data() = cS.data() + E<1>{} * get<1>(ThreadShape{}) * get<1>(TileShapeQK{});
    }

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);
    // empty step to sync against pipe s
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;
  }

  template<class Vector, class GTensor, class CTensor, class Shape, class Epilogue, class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE auto
  correction_epilogue(
      float scale_softmax_log2, float scale_out, Vector const& v0, Vector const& v1, 
      GTensor& gO, CTensor const& cO, Shape const& g_shape,
      Epilogue const& epilogue, BlkCoord const& blk_coord, ProblemShape const& problem_shape,int const row_idx) {

    using ElementOut = typename GTensor::value_type;

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    const int kCorrectionTileSize = 32 / sizeof(ElementOut); // 16 or 8
    // TODO: load all values

    // Choose TMEM OP based on TileM shape, then repeat for the appropriate number of columns
    // Use op_repeater pattern like softmax_step to ensure proper layout compatibility
    // For TileM=64 (16dp): 2 threads collaborate per row, each loads half the columns
    // For TileM=128 (32dp): 1 thread per row, loads all columns
    using TMEM_LOAD_1xOP = typename kValTyMap<void,
        kValTyPair<64, SM100_TMEM_LOAD_16dp32b1x>,
        kValTyPair<128, SM100_TMEM_LOAD_32dp32b1x>
      >::template query<TileM::value>;
    static constexpr int kColsPerThread = (TileM::value == 64) ? (kCorrectionTileSize / 2) : kCorrectionTileSize;
    using TMEM_LOAD = decltype(TMEM::op_repeater<TMEM_LOAD_1xOP, kColsPerThread * 32>());

    typename CollectiveMmaPV::TiledMma mma;
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);
    Tensor tOgO = mma.get_slice(0).partition_C(gO);
    
    Tensor tOtO_i = tOtO.compose(make_layout(make_shape(TileM{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = tOcO.compose(make_layout(make_shape(TileM{}, Int<kCorrectionTileSize>{})));
    Tensor tOgO_i = tOgO.compose(make_layout(make_shape(TileM{}, Int<kCorrectionTileSize>{})));

    Tensor tOtO0 = tOtO_i;
    tOtO0.data() = tOtO0.data().get() + uint32_t(TmemAllocation::O0);
    Tensor tOtO1 = tOtO_i;
    tOtO1.data() = tOtO1.data().get() + uint32_t(TmemAllocation::O1);

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);
    
    Tensor tTMEM_LOADtO0 = thr_tmem_load.partition_S(tOtO0);
    Tensor tTMEM_LOADtO1 = thr_tmem_load.partition_S(tOtO1);
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i);
    Tensor tTMEM_LOADgO = thr_tmem_load.partition_D(tOgO_i);

    float row_max = std::max(v0(kIdxFinalRowMax), v1(kIdxFinalRowMax));
    float adj0 = ::exp2f(scale_softmax_log2 * (v0(kIdxFinalRowMax) - row_max));
    float adj1 = ::exp2f(scale_softmax_log2 * (v1(kIdxFinalRowMax) - row_max));
    float row_sum = adj0 * v0(kIdxFinalRowSum) + adj1 * v1(kIdxFinalRowSum);
    float scale0 = scale_out * adj0 / row_sum;
    float scale1 = scale_out * adj1 / row_sum;

    // Compute and store LSE if requested
    if (epilogue.params.ptr_LSE != nullptr) {
      // LSE = log(row_sum) + scale_softmax * row_max
      // scale_softmax_log2 is already in log2 space, convert to natural log
      float lse = cutlass::fast_log(row_sum) + (scale_softmax_log2 / std::log2(std::exp(1.0f))) * row_max;
      int h_r = row_idx;
      int h_k = get<2, 0>(blk_coord);
      int b = get<2, 1>(blk_coord);
      int split_k_idx = get<1>(blk_coord);  // Split-K index

      // After problem_shape transformation in kernel:
      // problem_shape = (H_R, Sk, D, ((1, H_K), B))
      // So: get<0> = H_R, get<3,0,1> = H_K
      int H_R = get<0>(problem_shape);
      int H_K = get<3, 0, 1>(problem_shape);

      // Check bounds
      if (thread_idx < H_R) {
        // LSE tensor layout: [B, num_splits, H] where H = H_K * H_R
        // dLSE strides account for num_splits: (num_splits * H, H_R, 1)
        // Offset pointer by split_k_idx * H, then use stride-based access for (b, h_k, h_r)
        int H = H_K * H_R;
        int linear_idx = split_k_idx * H +
            b * get<0>(epilogue.params.dLSE) +
            h_k * get<1>(epilogue.params.dLSE) +
            h_r * get<2>(epilogue.params.dLSE);
        epilogue.params.ptr_LSE[linear_idx] = lse;
      }
    }

    float2 scale0_f32x2 = make_float2(scale0, scale0);
    float2 scale1_f32x2 = make_float2(scale1, scale1);

    // loop:
    //   TMEM_LOAD, TMEM_LOAD, FMUL2, FFMA2, STG
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < get<2>(TileShape{}) / kCorrectionTileSize; i++) {
      Tensor tTMEM_LOADtO0_i = tTMEM_LOADtO0;
      tTMEM_LOADtO0_i.data() = tTMEM_LOADtO0_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMEM_LOADtO1_i = tTMEM_LOADtO1;
      tTMEM_LOADtO1_i.data() = tTMEM_LOADtO1_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMEM_LOADgO_i = tTMEM_LOADgO;
      tTMEM_LOADgO_i.data() = tTMEM_LOADgO_i.data().get() + i * kCorrectionTileSize * stride<1>(gO);

      Tensor tTMrO0 = make_tensor<ElementPV>(shape(tTMEM_LOADcO));
      Tensor tTMrO1 = make_tensor<ElementPV>(shape(tTMEM_LOADcO));
      
      copy(tiled_tmem_load, tTMEM_LOADtO0_i, tTMrO0);
      copy(tiled_tmem_load, tTMEM_LOADtO1_i, tTMrO1);
      
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO0); j += 2) {
        float2 in0 = make_float2(tTMrO0(j), tTMrO0(j+1));
        float2 in1 = make_float2(tTMrO1(j), tTMrO1(j+1));
        float2 out;
        cute::mul(out, scale0_f32x2, in0);
        cute::fma(out, scale1_f32x2, in1, out);
        tTMrO0(j) = out.x;
        tTMrO0(j+1) = out.y;
      }

      constexpr int N = 4 / sizeof(ElementOut);
      NumericArrayConverter<ElementOut, ElementPV, N> convert;

      Tensor tSMrO = make_tensor_like<ElementOut>(tTMrO0);

      Tensor tCs = recast<decltype(convert)::source_type>(tTMrO0);
      Tensor tCd = recast<decltype(convert)::result_type>(tSMrO);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tCs); j++) {
        tCd(j) = convert.convert(tCs(j));
      }

        Tensor tSMgO_i = recast<uint32_t>(tTMEM_LOADgO_i);
        Tensor tSMrO_i = recast<uint32_t>(tSMrO);

        // could use masking do this right for smaller D
        if (get<0>(tTMEM_LOADcO(_0{})) < get<0>(g_shape)) {
        copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSMrO_i, tSMgO_i);
        }
    }
  }

  // Write neutral values for empty splits (varlen case where start_k >= seqlen_kv)
  // This ensures correct merge behavior: zeros for output, -inf for LSE
  //
  // ============================================================================
  // IMPORTANT: WARP COUNT DISCREPANCY IN DECODE KERNEL
  // ============================================================================
  //
  // This function is called by the Correction warp role. In the decode kernel
  // (sm100_fmha_gen_kernel_warpspecialized.hpp), there is a discrepancy between
  // the warp mapping comments and the actual warp assignments:
  //
  //   Kernel Schedule (Sm100FmhaGenKernelSchedule64x128x128):
  //     warp 0  -> Softmax0
  //     warp 1  -> MMA  
  //     warp 2,3 -> Load
  //     warp 4  -> Softmax1
  //     warp 8  -> Correction   <-- ONLY warp 8, despite comment saying "8-11"
  //     others  -> Empty
  //
  //   NumWarpsCorrection = 1  (only 32 threads)
  //
  // This function is actually only called by 1 warp (32 threads)
  //
  // LONG-TERM FIX NEEDED: T247858409
  //
  // ============================================================================
  template<class BlkCoord, class ProblemShape, class Epilogue>
  CUTLASS_DEVICE void
  write_empty_split(
      BlkCoord const& blk_coord,
      Params const& params,
      ProblemShape const& problem_shape,
      Epilogue const& epilogue) {

    // Use single warp (32 threads) - this matches the actual NumWarpsCorrection = 1
    // in the decode kernel schedule. Each thread will write multiple elements.
    int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarp;

    int split_k_idx = get<1>(blk_coord);
    int h_k = get<2, 0>(blk_coord);
    int b = get<2, 1>(blk_coord);

    // After kernel transformation, problem_shape is:
    // (H_R, klen, D, ((1, H_K), B))
    // where H_R = number of query heads per KV head group (GQA ratio)
    int H_R = get<0>(problem_shape);
    int H_K = get<3, 0, 1>(problem_shape);
    int D = get<2>(TileShape{}); // change to get<?>problem_shape since we have D = 64.

    // Write -inf to LSE for this empty split
    // This ensures the empty split contributes zero weight during merge
    if (epilogue.params.ptr_LSE != nullptr) {
      // LSE layout: [B, num_splits, H] where H = H_K * H_R
      // Each thread writes one LSE value if thread_idx < H_R
      if (thread_idx < H_R) {
        int h_r = thread_idx;
        int H = H_K * H_R;
        int linear_idx = split_k_idx * H +
            b * get<0>(epilogue.params.dLSE) +
            h_k * get<1>(epilogue.params.dLSE) +
            h_r * get<2>(epilogue.params.dLSE);
        
        // Use -inf so exp(-inf) = 0, meaning this split contributes nothing to merge
        epilogue.params.ptr_LSE[linear_idx] = -INFINITY;
      }
    }

    // Write zeros to output for this empty split
    // OUT tensor layout: [B, H, num_splits, D] where H = H_K * H_R
    
    // Compute base pointer for this split's output
    // The split dimension is embedded in the pointer offset: ptr_o + split_k_idx * D
    ElementOut* out_ptr = epilogue.params.ptr_o + split_k_idx * D;

    // Use vectorized 128-bit stores 
    // Each uint4 store writes 16 bytes = 128 bits
    // For bf16/fp16: 16 bytes = 8 elements per store
    // For fp32: 16 bytes = 4 elements per store
    constexpr int kVectorSize = 16 / sizeof(ElementOut);
    uint4 zero_vec = make_uint4(0, 0, 0, 0);
    
    // Each thread writes zeros to its portion of the output
    // For H_R rows (query heads within this KV head group), write D elements each
    for (int h_r = 0; h_r < H_R; h_r++) {
      // Correct stride mapping:
      //   B stride: get<2,1>(dO)
      //   H_K stride: get<2,0,1>(dO)
      //   H_R stride: get<2,0,0>(dO)
      int linear_base =
          b * get<2,1>(epilogue.params.dO) +
          h_k * get<2,0,1>(epilogue.params.dO) +
          h_r * get<2,0,0>(epilogue.params.dO);

      // Vectorized stores: each thread writes non-overlapping 128-bit chunks
      // Each vec_ptr[x] is 16 contiguous bytes
      uint4* vec_ptr = reinterpret_cast<uint4*>(out_ptr + linear_base);
      int num_vectors = D / kVectorSize;
      
      for (int i = thread_idx; i < num_vectors; i += cutlass::NumThreadsPerWarp) {
        vec_ptr[i] = zero_vec;  // 128-bit aligned, contiguous store
      }
    }
  }

  CUTLASS_DEVICE auto
  correction_rescale(
      float scale,
      uint32_t tmem_O) {

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    const int kCorrectionTileSize = 32;

    using TMEM_LOAD = typename kValTyMap<void,
        kValTyPair<64, SM100_TMEM_LOAD_16dp32b16x>,
        kValTyPair<128, SM100_TMEM_LOAD_32dp32b32x> // 4x32 threads with 64 cols of 32b elem
      >::template query<TileM::value>;
    using TMEM_STORE = typename kValTyMap<void,
        kValTyPair<64, SM100_TMEM_STORE_16dp32b16x>,
        kValTyPair<128, SM100_TMEM_STORE_32dp32b32x> // 4x32 threads with 64 cols of 32b elem
      >::template query<TileM::value>;

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0,1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);
    
    Tensor tOtO_i = tOtO.compose(make_layout(make_shape(TileM{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = tOcO.compose(make_layout(make_shape(TileM{}, Int<kCorrectionTileSize>{})));

    tOtO_i.data() = tOtO_i.data().get() + tmem_O;
    
    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i);
    auto thr_tmem_load   = tiled_tmem_load.get_slice(thread_idx);
    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tOtO_i);
    auto thr_tmem_store   = tiled_tmem_store.get_slice(thread_idx);
    
    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i);
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i);
    Tensor tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i);
    Tensor tTMEM_STOREcO = thr_tmem_store.partition_S(tOcO_i);
    static_assert(shape(tTMEM_STOREcO) == shape(tTMEM_LOADcO));

    float2 scale_f32x2 = make_float2(scale, scale);

    Tensor tTMrO = make_tensor<ElementPV>(make_shape(shape(tTMEM_LOADcO), Int<get<2>(TileShape{}) / kCorrectionTileSize>{}));
    
    auto copy_in = [&](int i) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO;
      tTMEM_LOADtO_i.data() = tTMEM_LOADtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i);
    };

    auto copy_out = [&](int i) {
      Tensor tTMEM_STOREtO_i = tTMEM_STOREtO;
      tTMEM_STOREtO_i.data() = tTMEM_STOREtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i);
    };

    // sequence: LLMSLMSLMSS

    // loop:
    //   TMEM_LOAD, FMUL2 scale, TMEM_STORE
    copy_in(0);

    int count = get<2>(TileShape{}) / kCorrectionTileSize;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < count; i++) {
      if (i != count - 1) {
        copy_in(i+1);
      }

      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));

      if (scale != 1.0f) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size(tTMrO_i); j += 2) {
          float2 in = make_float2(tTMrO_i(j), tTMrO_i(j+1));
          float2 out;
          cute::mul(out, scale_f32x2, in);
          tTMrO_i(j) = out.x;
          tTMrO_i(j+1) = out.y;
        }
      }

      copy_out(i);
    }
  }

  template<class BlkCoord, class ProblemShape, class TensorStorageEpi, class Epilogue>
  CUTLASS_DEVICE auto
  correction(
      BlkCoord const& blk_coord,
      Params const& params, ProblemShape const& problem_shape,
      TensorStorageEpi& shared_storage_epi,
      PipelineC& pipeline_s0_c, typename PipelineC::PipelineState& pipeline_s0_c_consumer_state,
      PipelineC& pipeline_s1_c, typename PipelineC::PipelineState& pipeline_s1_c_consumer_state,
      PipelineO& pipeline_o, typename PipelineO::PipelineState& pipeline_o_consumer_state,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      Epilogue const& epilogue) {

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0,1>(TileShapeQK{}));

    Tensor cS = make_identity_tensor(select<0,1>(TileShapeQK{}));
    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

    Tensor tStS_v = tStS.compose(make_layout(make_shape(TileM{}, kTMEM_V_COLUMNS{})));
    Tensor tScS_v = tScS.compose(make_layout(make_shape(TileM{}, kTMEM_V_COLUMNS{})));

    using TMEM_LOAD_V =
        typename kValTyMap<void,
        kValTyPair<64, SM100_TMEM_LOAD_16dp32b2x>,
        kValTyPair<128, SM100_TMEM_LOAD_32dp32b2x> // 4x32 threads with 2 cols of 32b elem
      >::template query<TileM::value>;
    auto tiled_tmem_loadv = make_tmem_copy(TMEM_LOAD_V{}, tStS_v);
    auto thr_tmem_loadv  = tiled_tmem_loadv.get_slice(thread_idx);

    Tensor tTMEM_LOADVtS = thr_tmem_loadv.partition_S(tStS_v);
    Tensor tTMEM_LOADVcS = thr_tmem_loadv.partition_D(tScS_v);

    Tensor tTMEM_LOADVtS0 = tTMEM_LOADVtS;
    tTMEM_LOADVtS0.data() = tTMEM_LOADVtS0.data().get() + uint32_t(TmemAllocation::V0);
    Tensor tTMEM_LOADVtS1 = tTMEM_LOADVtS;
    tTMEM_LOADVtS1.data() = tTMEM_LOADVtS1.data().get() + uint32_t(TmemAllocation::V1);

    // ignore first signal from softmax as no correction is required
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);
    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // handle the last iteration differently (i.e. tmem_load/stsm for epi)
    mask_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

      Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
      static_assert(size(tTMEM_LOADVrS) == 2);

      // read row_wise new global max
      copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

      // e^(scale * (old_max - new_max)
      float scale = (tTMEM_LOADVrS(kIdxOldRowMax) == tTMEM_LOADVrS(kIdxNewRowMax)) ? 1.0f : ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      bool warp_do_correction = __any_sync(0xFFFFFFFF, scale != 1.0f);
      if (warp_do_correction) {
        correction_rescale(scale, uint32_t(TmemAllocation::O0));
      }

      pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
      ++pipeline_s1_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;

      pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

      copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

      scale = (tTMEM_LOADVrS(kIdxOldRowMax) == tTMEM_LOADVrS(kIdxNewRowMax)) ? 1.0f : ::exp2f(params.scale_softmax_log2 * (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      warp_do_correction = __any_sync(0xFFFFFFFF, scale != 1.0f);
      if (warp_do_correction) {
        correction_rescale(scale, uint32_t(TmemAllocation::O1));
      }

      pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
      ++pipeline_s0_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;
    }

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    // do the final correction to O1
    // better to somehow special-case it in the loop above
    // doesn't matter for non-persistent code, but if it were
    // persistent we do not want to release O too early

    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

    // read from V0
    // read row_sum and final row_max here
    Tensor tTMEM_LOADVrS0 = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
    copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS0);

    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    // load from V1
    Tensor tTMEM_LOADVrS1 = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
    copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS1);

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    auto pipeline_o_release_state = pipeline_o_consumer_state;
    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;
    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;
    // store to epi smem

    // loop:
    //    TMEM_LOAD
    //    FMUL2 scale = 1 / global_sum * out_quant_scale
    //    F2FP
    //    store to smem

    Tensor cO = make_identity_tensor(select<0,1>(TileShapePV{}));
    auto g_shape = select<0,2>(problem_shape);
    int split_k_idx = get<1>(blk_coord);  // Split-K index
    int D = get<2>(TileShape{});  // Head dimension
    // OUT tensor layout: [B, H, num_splits, D]
    // Pointer offset: ptr_o + split_k_idx * D
    auto mO = make_tensor(make_gmem_ptr(epilogue.params.ptr_o + split_k_idx * D),
                          append<3>(select<0,1>(TileShapePV{}), get<3>(problem_shape)), 
                          epilogue.params.dO);
    auto gO = mO(_, _, get<2>(blk_coord));
    int row_idx = get<0>(tTMEM_LOADVcS(_0{}));
    correction_epilogue(params.scale_softmax_log2, params.scale_output, tTMEM_LOADVrS0, tTMEM_LOADVrS1, 
      gO, cO, g_shape, epilogue, blk_coord, problem_shape, row_idx);

    cutlass::arch::fence_view_async_tmem_load();

    pipeline_o.consumer_release(pipeline_o_release_state);
    ++pipeline_o_release_state;

    pipeline_o.consumer_release(pipeline_o_release_state);
    ++pipeline_o_release_state;
  }

};

}  // namespace cutlass::fmha::collective
