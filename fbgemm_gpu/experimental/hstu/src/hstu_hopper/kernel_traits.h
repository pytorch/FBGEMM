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

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

template <
    int kStages,
    class Gemm1Type,
    class Gemm2Type,
    class OutputType,
    class SmemLayoutQ,
    class SmemRabStorage,
    class SmemLayoutK,
    class SmemLayoutV,
    class SmemLayoutO>
struct SharedStorageQRabKVO {
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
  SmemRabStorage smem_rab;
  union {
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_rab;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    int tile_count_semaphore;
  };
};

template <
    int kStages,
    class Gemm1Type,
    class Gemm2Type,
    class OutputType,
    class SmemLayoutQ,
    class SmemLayoutRab,
    class SmemLayoutK,
    class SmemLayoutV,
    class SmemLayoutO>
struct SharedStorageQRabKVOVt {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutRab>> smem_rab;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
    union {
      cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
      cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_rab;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
  };
};

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    int kStages_,
    bool Is_causal_,
    bool Is_context_,
    bool Is_target_,
    bool Is_delta_q_,
    bool Is_local_,
    bool Has_rab_,
    int kClusterM_ = 1,
    bool Is_balance_fwd_ = false,
    typename elem_type = cutlass::half_t>
struct Hstu_fwd_kernel_traits {
  using Element = elem_type;
  using ElementAccum = float;
  using OutputType = elem_type;
  using index_t = int64_t;

  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_delta_q = Is_delta_q_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Is_balance_fwd = Is_balance_fwd_;

  // The number of threads.
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;
  static_assert(
      kNWarps_ == 4 || kNWarps_ == 8 || kNWarps_ == 12 || kNWarps_ == 16);

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

  static constexpr int kClusterM = kClusterM_;
  using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

  static constexpr int kStages = kStages_;

  using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
  using TiledMma0 = decltype(cute::make_tiled_mma(
      cute::GMMA::
          ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
      AtomLayoutMNK{}));
  using TiledMma1 = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<
          Element,
          Element,
          ElementAccum,
          decltype(select<0, 2, 1>(TileShape_MNK{})),
          GMMA::Major::K,
          GMMA::Major::MN>(),
      AtomLayoutMNK{}));

  using SmemLayoutAtomQ =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

  using SmemLayoutAtomRab =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutRab = decltype(tile_to_shape(
      SmemLayoutAtomRab{},
      make_shape(
          shape<0>(TileShape_MNK{}),
          shape<1>(TileShape_MNK{}),
          Int<kStages>{})));
  using SmemRabStorage = cute::conditional_t<
      Has_rab,
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutRab>>,
      cute::array_aligned<Element, 128 / sizeof(Element)>>;
  using SmemLayoutAtomK =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(
          shape<1>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));

  using SmemLayoutAtomV =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(
          get<1>(TileShape_MNK{}),
          get<2>(TileShape_MNK{}),
          Int<kStages>{})));

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{},
      make_ordered_layout(
          make_shape(
              get<2>(TileShape_MNK{}),
              get<1>(TileShape_MNK{}),
              Int<kStages>{}),
          Step<_2, _1, _3>{})));

  using SmemLayoutAtomO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               OutputType,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

  using SmemCopyAtomRab = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

  using SharedStorage = SharedStorageQRabKVO<
      kStages,
      Element,
      Element,
      Element,
      SmemLayoutQ,
      SmemRabStorage,
      SmemLayoutK,
      SmemLayoutV,
      SmemLayoutO>;

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
  using PipelineState = typename cutlass::PipelineState<kStages>;
};

// Traits struct for fp8 kernel with in-kernel transpose
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    int kStages_,
    bool Is_causal_,
    bool Is_context_,
    bool Is_target_,
    bool Is_delta_q_,
    bool Is_local_,
    bool Has_rab_,
    int kClusterM_ = 1,
    bool Is_balance_fwd_ = false>
struct Hstu_fwd_kernel_traits_fp8 {
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using OutputType = cutlass::half_t;
  using index_t = int64_t;

  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_delta_q = Is_delta_q_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Is_balance_fwd = Is_balance_fwd_;

  // The number of threads.
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;
  static_assert(kNWarps_ == 12 || kNWarps_ == 16);

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

  static constexpr int kClusterM = kClusterM_;
  using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

  static constexpr int kStages = kStages_;
  static_assert(kStages > 1);

  using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
  using TiledMma0 = decltype(cute::make_tiled_mma(
      cute::GMMA::
          ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
      AtomLayoutMNK{}));

  using TiledMma1 = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<
          Element,
          Element,
          ElementAccum,
          decltype(select<0, 2, 1>(TileShape_MNK{}))>(),
      AtomLayoutMNK{}));

  using SmemLayoutAtomQ =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

  using SmemLayoutAtomK =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(
          shape<1>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));

  using SmemLayoutAtomRab =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               OutputType,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutRab = decltype(tile_to_shape(
      SmemLayoutAtomRab{},
      make_shape(
          shape<0>(TileShape_MNK{}),
          shape<1>(TileShape_MNK{}),
          Int<kStages>{})));

  using TransposeShapeAtomV = Shape<_64, _64>;
  using SmemLayoutAtomV = decltype(tile_to_shape(
      GMMA::Layout_K_SW64_Atom<Element>{},
      TransposeShapeAtomV{}));
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(
          shape<1>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));

  // for fp8 in-kernel transpose -- src layout
  using SmemLayoutDivideV =
      decltype(tiled_divide(SmemLayoutV{}, TransposeShapeAtomV{}));
  using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
  using FactoringShapeV = decltype(make_shape(
      SmemShapeLDSM{},
      shape<1>(SmemLayoutDivideV{}),
      shape<2>(SmemLayoutDivideV{}),
      shape<3>(SmemLayoutDivideV{})));
  using SmemLayoutTransposeV = decltype(composition(
      SmemLayoutDivideV{},
      make_layout(FactoringShapeV{})));

  // For fp8, this is the memory transpose.
  using SmemLayoutAtomVt = decltype(tile_to_shape(
      GMMA::Layout_K_SW64_Atom<Element>{},
      TransposeShapeAtomV{}));
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(
          shape<2>(TileShape_MNK{}),
          shape<1>(TileShape_MNK{}),
          Int<kStages>{})));

  // for fp8 in-kernel transpose -- dst layout
  using SmemLayoutVtTrans = decltype(composition(
      SmemLayoutVt{},
      make_ordered_layout(
          product_each(shape(SmemLayoutV{})),
          Step<_2, _1, _3>{})));
  using SmemLayoutDivideVt =
      decltype(tiled_divide(SmemLayoutVtTrans{}, TransposeShapeAtomV{}));
#ifndef NO_FP8_COLUMN_PERMUTE
  using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_8, _8>>;
#else
  using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
#endif
  using FactoringShapeVt = decltype(make_shape(
      SmemShapeSTSM{},
      shape<1>(SmemLayoutDivideVt{}),
      shape<2>(SmemLayoutDivideVt{}),
      shape<3>(SmemLayoutDivideVt{})));
  using SmemLayoutTransposeVt = decltype(composition(
      SmemLayoutDivideVt{},
      make_layout(FactoringShapeVt{})));

  using SmemLayoutAtomO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               OutputType,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

  // used for rmem -> smem O copy in fp8 kernel to undo column permutation
  using ThreadLayoutrO =
      Layout<Shape<_8, Int<kBlockM / 16>, _4, _1>, Stride<_4, _32, _1, _0>>;
  using ValueLayoutrO = Layout<
      Shape<_1, _2, Shape<_2, _2>, Int<kHeadDim / 16>>,
      Stride<_0, _2, Stride<_4, _1>, _8>>;
  using TiledCopyrO = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<uint16_t>, OutputType>{},
      ThreadLayoutrO{},
      ValueLayoutrO{}));

  using TiledCopyShaperO = Shape<_8, Int<kBlockM / 8>, _16, Int<kHeadDim / 16>>;
  using SmemLayoutrO =
      decltype(composition(SmemLayoutO{}, Layout<TiledCopyShaperO>{}));

  using SmemCopyAtomRab = Copy_Atom<cute::DefaultCopy, OutputType>;

  using SharedStorage = SharedStorageQRabKVOVt<
      kStages,
      Element,
      Element,
      OutputType,
      SmemLayoutQ,
      SmemLayoutRab,
      SmemLayoutK,
      SmemLayoutV,
      SmemLayoutO>;

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
  using PipelineState = typename cutlass::PipelineState<kStages>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    bool Is_causal_,
    bool Is_context_,
    bool Is_target_,
    bool Is_delta_q_,
    bool Is_local_,
    bool Has_rab_,
    bool Has_drab_,
    bool Is_deterministic_,
    int kStages_dO_,
    int kStages_dS_,
    bool SdP_swapAB_,
    bool dKV_swapAB_,
    bool dQ_swapAB_,
    int NumWarpGroups_ = 3,
    int AtomLayoutMSdP_ = 1,
    int AtomLayoutNdKV_ = 2,
    int AtomLayoutMdQ_ = 1,
    bool Is_balance_bwd_ = false,
    typename elem_type = cutlass::half_t>
struct Hstu_bwd_kernel_traits {
  using Element = elem_type;
  using ElementAccum = float;
  using index_t = int64_t;

  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_delta_q = Is_delta_q_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Has_drab = Has_drab_;
  static constexpr bool Is_deterministic = Is_deterministic_;
  static constexpr bool Is_balance_bwd = Is_balance_bwd_;
  // The number of threads.
  static constexpr int NumMmaWarpGroups = NumWarpGroups_ - 1;
  static constexpr int NumMmaThreads =
      NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int kNWarps =
      (NumMmaWarpGroups + 1) * cutlass::NumWarpsPerWarpGroup;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumdQWarpGroups = NumMmaWarpGroups;
  static constexpr int kNThreadsdQ =
      NumdQWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int NumEpilogueThreads = NumMmaThreads;

  static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
  static constexpr int AtomLayoutNdKV = AtomLayoutNdKV_;
  static constexpr int AtomLayoutMdQ = AtomLayoutMdQ_;
  static constexpr int AtomLayoutKdKV = NumMmaWarpGroups / AtomLayoutNdKV;
  static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0);
  static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0);
  static_assert(NumMmaWarpGroups % AtomLayoutMdQ == 0);

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using TileShape_MK = Shape<Int<kBlockM>, Int<kHeadDim>>;

  static constexpr int kClusterN = 1;
  using ClusterShape = Shape<_1, Int<kClusterN>, _1>;

  static constexpr int kStages_dO = kStages_dO_;
  static constexpr int kStages_dS = kStages_dS_;
  static constexpr int kStages = kStages_dS;

  static constexpr bool SdP_swapAB = SdP_swapAB_;
  static constexpr bool dKV_swapAB = dKV_swapAB_;
  static constexpr bool dQ_swapAB = dQ_swapAB_;

  static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 &&
      AtomLayoutNdKV == NumMmaWarpGroups && SdP_swapAB && !dKV_swapAB;
  static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == NumMmaWarpGroups &&
      AtomLayoutMdQ == NumMmaWarpGroups && !SdP_swapAB && !dQ_swapAB;

  using TileShapeAtomSdP = std::conditional_t<
      !SdP_swapAB,
      Shape<
          Int<kBlockM>,
          Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>,
          Int<kHeadDim>>,
      Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>>;
  using AtomLayoutSdP = std::conditional_t<
      !SdP_swapAB,
      Layout<Shape<
          Int<AtomLayoutMSdP>,
          Int<NumMmaWarpGroups / AtomLayoutMSdP>,
          _1>>,
      Layout<Shape<
          Int<NumMmaWarpGroups / AtomLayoutMSdP>,
          Int<AtomLayoutMSdP>,
          _1>>>;
  using TiledMmaSdP = decltype(cute::make_tiled_mma(
      cute::GMMA::
          ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
      AtomLayoutSdP{}));

  static constexpr GMMA::Major PdS_Major = GMMA::Major::K;
  static constexpr GMMA::Major PdSt_Major =
      PdS_Major == GMMA::Major::K ? GMMA::Major::MN : GMMA::Major::K;

  using TileShapeAtomdKV = std::conditional_t<
      !dKV_swapAB,
      Shape<
          Int<kBlockN>,
          Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>,
          Int<kBlockM>>,
      Shape<Int<kHeadDim>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>>;
  using AtomLayoutdKV = std::conditional_t<
      !dKV_swapAB,
      Layout<Shape<
          Int<AtomLayoutNdKV>,
          Int<NumMmaWarpGroups / AtomLayoutNdKV>,
          _1>>,
      Layout<Shape<
          Int<NumMmaWarpGroups / AtomLayoutNdKV>,
          Int<AtomLayoutNdKV>,
          _1>>>;
  using TiledMmadKV = decltype(cute::make_tiled_mma(
      std::conditional_t<
          Mma_dKV_is_RS,
          decltype(cute::GMMA::rs_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShapeAtomdKV,
                   GMMA::Major::K,
                   GMMA::Major::MN>()),
          decltype(cute::GMMA::ss_op_selector < Element, Element, ElementAccum, TileShapeAtomdKV, !dKV_swapAB ? PdSt_Major : GMMA::Major::MN, !dKV_swapAB ? GMMA::Major::MN : PdSt_Major > ())>{},
      AtomLayoutdKV{}));

  using TileShapeAtomdQ = std::conditional_t<
      !dQ_swapAB,
      Shape<
          Int<kBlockM>,
          Int<kHeadDim / (NumdQWarpGroups / AtomLayoutMdQ)>,
          Int<kBlockN>>,
      Shape<Int<kHeadDim>, Int<kBlockM / AtomLayoutMdQ>, Int<kBlockN>>>;
  using AtomLayoutdQ = std::conditional_t<
      !dQ_swapAB,
      Layout<
          Shape<Int<AtomLayoutMdQ>, Int<NumdQWarpGroups / AtomLayoutMdQ>, _1>>,
      Layout<
          Shape<Int<NumdQWarpGroups / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>>;
  static constexpr GMMA::Major MmadQMajorA =
      !dQ_swapAB ? GMMA::Major::K : GMMA::Major::MN;
  static constexpr GMMA::Major MmadQMajorB =
      !dQ_swapAB ? GMMA::Major::MN : GMMA::Major::K;
  using TiledMmadQ = decltype(cute::make_tiled_mma(
      std::conditional_t<
          Mma_dQ_is_RS,
          decltype(cute::GMMA::rs_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShapeAtomdQ,
                   GMMA::Major::K,
                   GMMA::Major::MN>()),
          decltype(cute::GMMA::ss_op_selector < Element, Element, ElementAccum, TileShapeAtomdQ, !dQ_swapAB ? PdS_Major : GMMA::Major::MN, !dQ_swapAB ? GMMA::Major::MN : PdS_Major > ())>{},
      AtomLayoutdQ{}));

  using SmemLayoutAtomQdO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               Int<kBlockM>,
               Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>>());

  using GmemTiledCopyQdO =
      decltype(cutlass::gemm::collective::detail::
                   sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
  using GmemTiledCopyRab = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
  using GmemTiledCopydQaccum = cute::SM90_TMA_REDUCE_ADD;
  using GmemTiledCopydRab = cute::SM90_TMA_REDUCE_ADD;

  using SmemLayoutAtomQ =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQdO{},
      make_shape(
          shape<0>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));
  using SmemLayoutdO = decltype(tile_to_shape(
      SmemLayoutAtomQdO{},
      make_shape(
          shape<0>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages_dO>{})));

  using SmemLayoutAtomRab =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutRab = decltype(tile_to_shape(
      SmemLayoutAtomRab{},
      make_shape(
          shape<0>(TileShape_MNK{}),
          shape<1>(TileShape_MNK{}),
          Int<kStages>{})));

  using SmemLayoutRabt = decltype(cute::composition(
      SmemLayoutRab{},
      make_layout(
          make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

  using SmemLayoutAtomK =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               Int<kBlockN>,
               Int<kHeadDim / (NumdQWarpGroups / AtomLayoutMdQ)>>());
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

  using SmemLayoutAtomV =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutV =
      decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

  using SmemLayoutAtomPdS =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               PdS_Major,
               Element,
               Int<kBlockM / AtomLayoutMSdP>,
               Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>>());
  using SmemLayoutPdS = decltype(tile_to_shape(
      SmemLayoutAtomPdS{},
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages_dS>{}),
      std::conditional_t<
          PdS_Major == GMMA::Major::K,
          cute::Step<_1, _2, _3>,
          cute::Step<_2, _1, _3>>{}));

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutQt = decltype(cute::composition(
      SmemLayoutQ{},
      make_layout(
          make_shape(
              get<2>(TileShape_MNK{}),
              get<0>(TileShape_MNK{}),
              Int<kStages>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
  using SmemLayoutdOt = decltype(cute::composition(
      SmemLayoutdO{},
      make_layout(
          make_shape(
              get<2>(TileShape_MNK{}),
              get<0>(TileShape_MNK{}),
              Int<kStages_dO>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
  using SmemLayoutKt = decltype(cute::composition(
      SmemLayoutK{},
      make_layout(
          make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
          make_stride(Int<kBlockN>{}, _1{}))));
  using SmemLayoutPdSt = decltype(cute::composition(
      SmemLayoutPdS{},
      make_layout(
          make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages_dS>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

  // Thread layout, 256 or 384 threads per row
  using R2SLayoutAtomdQaccum = Layout<Shape<Int<kNThreadsdQ>>, Stride<_1>>;
  using R2STiledCopydQaccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      R2SLayoutAtomdQaccum{},
      Layout<Shape<_4>>{})); // Val layout, 4 vals per store
  using SmemLayoutdQaccum = Layout<Shape<Int<kBlockM * kHeadDim>>, Stride<_1>>;
  using SmemLayoutAtomdQaccumTMA = decltype(composition(
      Swizzle<0, 4, 3>{}, // We don't want any swizzle
      Layout<Shape<Int<8>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>{}));
  using SmemLayoutdQaccumTMA = decltype(tile_to_shape(
      SmemLayoutAtomdQaccumTMA{},
      select<0, 2>(TileShape_MNK{})));
  using SmemLayoutdQaccumTMANoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutdQaccumTMA{}));

  static constexpr int kBlockKSmem =
      kHeadDim % 64 == 0 ? 64 : (kHeadDim % 32 == 0 ? 32 : 16);
  static constexpr int kSwizzle =
      kBlockKSmem == 64 ? 3 : (kBlockKSmem == 32 ? 2 : 1);
  using SmemLayoutAtomdQ = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<Int<kNThreadsdQ / 32>, Int<32>>, Stride<Int<32>, _1>>{}));
  using SmemLayoutdQ = decltype(tile_to_shape(
      SmemLayoutAtomdQ{},
      make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
  using SmemLayoutdQt = decltype(cute::composition(
      SmemLayoutdQ{},
      make_layout(
          make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
          make_stride(Int<kBlockM>{}, _1{}))));

  static constexpr int kNumPdSStore = kBlockM * kBlockN / NumMmaThreads;
  using SmemCopyAtomPdS = Copy_Atom<
      std::conditional_t<
          (!SdP_swapAB) ^ (PdS_Major == GMMA::Major::MN),
          std::conditional_t<
              kNumPdSStore % 8 == 0,
              cute::SM90_U32x4_STSM_N,
              cute::SM90_U32x2_STSM_N>,
          std::conditional_t<
              kNumPdSStore % 8 == 0,
              cute::SM90_U16x8_STSM_T,
              cute::SM90_U16x4_STSM_T>>,
      Element>;
  using SmemCopyAtomRab = Copy_Atom<
      std::conditional_t<
          !SdP_swapAB,
          cute::SM75_U32x4_LDSM_N,
          cute::SM75_U16x8_LDSM_T>,
      Element>;

  using SmemLayoutAtomdKV = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutdKV = decltype(tile_to_shape(
      SmemLayoutAtomdKV{},
      select<1, 2>(TileShape_MNK{})));

  static constexpr bool dQacc_use_TMA = kHeadDim < 256;
  // These are for the case where we don't use TMA to do atomicAdd on dQaccum,
  // but just use direct atomicAdd.
  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(ElementAccum);
  static_assert(
      kHeadDim % kGmemElemsPerLoad == 0,
      "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kGmemThreadsPerRow =
      cutlass::gcd(kHeadDim / kGmemElemsPerLoad, int(kNThreadsdQ));
  using GmemLayoutAtomdQaccum = Layout<
      Shape<Int<kNThreadsdQ / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopydQaccumAtomic = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtomdQaccum{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{})); // Val layout, 4 vals per
                                                     // store

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
  using MainloopPipeline_dRab = typename cutlass::PipelineAsync<kStages_dS>;

  static constexpr size_t SmemAlignmentP =
      cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  static constexpr size_t SmemAlignmentdS =
      cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  // Without this SmemAlignment, with hdim 256 we get "misaligned address" error
  // in TMA
  static constexpr size_t SmemAlignmentQKVdO = kHeadDim % 256 == 0 ? 256 : 128;
  using SmemRabStorage = cute::conditional_t<
      Has_rab,
      cute::array_aligned<
          Element,
          cute::cosize_v<SmemLayoutRab>,
          SmemAlignmentQKVdO>,
      cute::array_aligned<Element, 128 / sizeof(Element)>>;
  static_assert(
      SmemAlignmentP >= 128 && SmemAlignmentdS >= 128,
      "Require at least 128B alignment");

  static constexpr size_t SmemAlignmentdKV =
      cutlass::detail::alignment_for_swizzle(SmemLayoutdKV{});
  static_assert(SmemAlignmentdKV >= 128, "Require at least 128B alignment");

  // Kernel level shared memory storage
  struct SharedStorage {
    union {
      struct {
        cute::array_aligned<
            Element,
            cute::cosize_v<SmemLayoutK>,
            SmemAlignmentQKVdO>
            smem_k;
        cute::array_aligned<
            Element,
            cute::cosize_v<SmemLayoutV>,
            SmemAlignmentQKVdO>
            smem_v;
      };
      struct {
        cute::array_aligned<
            Element,
            cute::cosize_v<SmemLayoutdKV>,
            SmemAlignmentdKV>
            smem_dk;
        cute::array_aligned<
            Element,
            cute::cosize_v<SmemLayoutdKV>,
            SmemAlignmentdKV>
            smem_dv;
      };
    };
    cute::array_aligned<
        ElementAccum,
        dQacc_use_TMA ? cute::cosize_v<SmemLayoutdQaccum> : 0>
        smem_dqacc;
    cute::
        array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO>
            smem_q;
    cute::
        array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO>
            smem_do;
    cute::array_aligned<
        Element,
        Mma_dKV_is_RS ? 0 : cute::cosize_v<SmemLayoutPdS>,
        SmemAlignmentP>
        smem_p;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS>
        smem_ds;
    SmemRabStorage smem_rab;

    struct {
      alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_KV;
      alignas(16) cutlass::arch::ClusterBarrier barrier_dKV;
      alignas(16) typename MainloopPipeline::SharedStorage pipeline_q;
      alignas(16) typename MainloopPipeline::SharedStorage pipeline_rab;
      alignas(16) typename MainloopPipeline_dO::SharedStorage pipeline_do;
      alignas(16) typename MainloopPipeline_dRab::SharedStorage pipeline_drab;
    };
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
