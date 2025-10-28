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
    class SmemLayoutVt,
    class SmemLayoutO,
    class SmemLayoutValidBlockIds,
    class SmemLayoutMaxFunc,
    class SmemLayoutMinFunc>
struct SharedStorageQRabKVO {
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
  SmemRabStorage smem_rab;
  union {
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutVt>> smem_v;
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
  struct {
    cute::array_aligned<int, cute::cosize_v<SmemLayoutMaxFunc>> smem_max_func;
    cute::array_aligned<int, cute::cosize_v<SmemLayoutMinFunc>> smem_min_func;
    cute::array_aligned<int, cute::cosize_v<SmemLayoutValidBlockIds>> smem_valid_block_ids;
    int sn_valid_block_max;
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
    class SmemLayoutVt,
    class SmemLayoutVtMma,
    class SmemLayoutO,
    class SmemLayoutValidBlockIds,
    class SmemLayoutMaxFunc,
    class SmemLayoutMinFunc>
struct SharedStorageQRabKVOVt {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutRab>> smem_rab;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutVt>> smem_v;
    union {
      cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutVtMma>> smem_v_out;
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
  struct {
    cute::array_aligned<int, cute::cosize_v<SmemLayoutMaxFunc>> smem_max_func;
    cute::array_aligned<int, cute::cosize_v<SmemLayoutMinFunc>> smem_min_func;
    cute::array_aligned<int, cute::cosize_v<SmemLayoutValidBlockIds>> smem_valid_block_ids;
    int sn_valid_block_max;
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
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
    bool Has_rab_,
    int kClusterM_ = 1,
    typename elem_type = cutlass::half_t>
struct Hstu_fwd_kernel_traits {
  using Element = elem_type;
  using ElementAccum = float;
  using OutputType = elem_type;
  using index_t = int64_t;

  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr int Quant_mode = -1;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int kNFunc = Is_arbitrary ? kNFunc_ : 0;

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
  using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;

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

  static constexpr int NumMmaThreads = size(TiledMma0{});
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

  using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::MN, Element,
                                    decltype(cute::get<1>(TileShape_MNK_PV{})), decltype(cute::get<2>(TileShape_MNK_PV{}))>());
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_ordered_layout(
          make_shape(
              get<1>(TileShape_MNK_PV{}),
              get<2>(TileShape_MNK_PV{}),
              Int<kStages>{}),
          Step<_2, _1, _3>{})));
  using SmemLayoutVtMma = SmemLayoutVt;

  using SmemLayoutAtomO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               OutputType,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

  using SmemCopyAtomRab = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

  static constexpr int MaxSeqLenK = (kHeadDim == 256 && Has_rab) ? 32 * 1024 : 64 * 1024; // 48K and 64K
  static constexpr int MaxValidBlock = Is_arbitrary ? MaxSeqLenK / kBlockN : 1; // 4KB

  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>; // 0 refer to number of valid blocks
  using SmemLayoutMaxFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;
  using SmemLayoutMinFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;

  using SharedStorage = SharedStorageQRabKVO<
      kStages,
      Element,
      Element,
      Element,
      SmemLayoutQ,
      SmemRabStorage,
      SmemLayoutK,
      SmemLayoutVt,
      SmemLayoutO,
      SmemLayoutValidBlockIds,
      SmemLayoutMaxFunc,
      SmemLayoutMinFunc>;

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
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
    bool Has_rab_,
    int kClusterM_ = 1,
    int Quant_mode_ = 0>
struct Hstu_fwd_kernel_traits_fp8 {
  using Element = cutlass::float_e4m3_t;
  using ElementAccum = float;
  using OutputType = cutlass::half_t;
  using index_t = int64_t;

  // Quantization mode: 0: cast to fp8, 1: 1xDIM&128x1 quantization,
  // 2: per-block quantization, 3: per-head quantization, 4: per-batch quantization, 5: per-tensor quantization.
  static constexpr int Quant_mode = Quant_mode_;
  static constexpr GMMA::Major MajorV = Quant_mode != 1 ? GMMA::Major::MN : GMMA::Major::K;

  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int  kNFunc = Is_arbitrary ? kNFunc_ : 0;

  // The number of threads.
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;
  static_assert(kNWarps_ == 12 || kNWarps_ == 16);

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;

  static constexpr int kBlockSeqQ_scale = Quant_mode == 1 ? 128 : kBlockM;
  static constexpr int kBlockSeqK_scale = Quant_mode == 1 ? 128 : kBlockN;
  static_assert(kBlockSeqQ_scale % kBlockM == 0);
  static_assert(kBlockSeqK_scale % kBlockN == 0);
  static constexpr int kMBlock_shared = kBlockSeqQ_scale / kBlockM;
  static constexpr int kNBlock_shared = kBlockSeqK_scale / kBlockN;

  static_assert(kHeadDim % 32 == 0);
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;

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

  static constexpr int NumMmaThreads = size(TiledMma0{});
  using TiledMma0descale = decltype(make_tiled_mma(UniversalFMA<ElementAccum, ElementAccum, ElementAccum>{},
    Layout<Shape<Int<NumMmaThreads / 4>, _4, _1>, Stride<_4, _1, _0>>{},
    Tile<Layout<Shape<_8, Int<kBlockM / 16>, _2>, Stride<_1, _16, _8>>, Layout<Shape<_4, _2, Int<kBlockN / 8>>, Stride<_2, _1, _8>>, _1>{}));

  using TiledMma1descale = decltype(make_tiled_mma(UniversalFMA<ElementAccum, ElementAccum, ElementAccum>{},
    Layout<Shape<Int<NumMmaThreads / 4>, _4, _1>, Stride<_4, _1, _0>>{},
    Tile<Layout<Shape<_8, Int<kBlockM / 16>, _2>, Stride<_1, _16, _8>>, Layout<Shape<_4, _2, Int<kHeadDim / 8>>, Stride<_2, _1, _8>>, _1>{}));

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

  using SmemLayoutAtomVt = std::conditional_t<Quant_mode == 1,
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<MajorV, Element,
                                          decltype(cute::get<1>(TileShape_MNK_PV{})), decltype(cute::get<2>(TileShape_MNK_PV{}))>()),
    decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Element>{}, TransposeShapeAtomV{}))
  >;
  using SmemLayoutVt =
      decltype(tile_to_shape(SmemLayoutAtomVt{},
                make_shape(cute::get<1>(TileShape_MNK_PV{}), shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
                std::conditional_t<MajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

  using SmemLayoutAtomVtMma = std::conditional_t<Quant_mode == 1,
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                          decltype(cute::get<1>(TileShape_MNK_PV{})), decltype(cute::get<2>(TileShape_MNK_PV{}))>()),
    decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtomV{}))
  >;

  using SmemLayoutVtMma = decltype(tile_to_shape(
        SmemLayoutAtomVtMma{},
        make_shape(cute::get<1>(TileShape_MNK_PV{}), shape<2>(TileShape_MNK_PV{}), Int<kStages>{})));

  // for fp8 in-kernel transpose -- dst layout
  using SmemLayoutVtTrans = decltype(composition(
      SmemLayoutVtMma{},
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
  using DescaleCopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, ElementAccum>;

  static constexpr int MaxSeqLenK = (kHeadDim == 256 && Has_rab) ? 32 * 1024 : 64 * 1024; // 48K and 64K
  static constexpr int MaxValidBlock = Is_arbitrary ? MaxSeqLenK / kBlockN : 1; // 4KB

  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>; // 0 refer to number of valid blocks
  using SmemLayoutMaxFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;
  using SmemLayoutMinFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;

  using SharedStorage = SharedStorageQRabKVOVt<
      kStages,
      Element,
      Element,
      OutputType,
      SmemLayoutQ,
      SmemLayoutRab,
      SmemLayoutK,
      SmemLayoutVt,
      SmemLayoutVtMma,
      SmemLayoutO,
      SmemLayoutValidBlockIds,
      SmemLayoutMaxFunc,
      SmemLayoutMinFunc>;

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
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
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
    typename elem_type = cutlass::half_t>
struct Hstu_bwd_kernel_traits {
  using Element = elem_type;
  using ElementAccum = float;
  using ElementRab = Element;
  using ElementOut = Element;
  using index_t = int64_t;

  static constexpr bool Is_fp8 = false;
  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int  kNFunc = Is_arbitrary ? kNFunc_ : 0;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Has_drab = Has_drab_;
  static constexpr bool Is_deterministic = Is_deterministic_;
  static constexpr int Quant_mode = -1;
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
          decltype(cute::GMMA::ss_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShapeAtomdKV,
                   !dKV_swapAB ? PdSt_Major : GMMA::Major::MN,
                   !dKV_swapAB ? GMMA::Major::MN : PdSt_Major>())>{},
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
          decltype(cute::GMMA::ss_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShapeAtomdQ,
                   !dQ_swapAB ? PdS_Major : GMMA::Major::MN,
                   !dQ_swapAB ? GMMA::Major::MN : PdS_Major>())>{},
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
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutdQ = decltype(tile_to_shape(SmemLayoutAtomdQ{}, TileShape_MK{}));
  using SmemLayoutdQt = decltype(cute::composition(
      SmemLayoutdQ{},
      make_layout(
          make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
          make_stride(Int<kBlockM>{}, _1{}))));
  using SmemCopyAtomdQ = Copy_Atom<
      std::conditional_t<!dQ_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>,
      Element>;

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

  static constexpr int MaxSeqLenQ = 64 * 1024; // 64K
  static constexpr int MaxValidBlock = Is_arbitrary ? MaxSeqLenQ / kBlockM : 1; // 4KB
  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>; // 0 refer to number of valid blocks

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
  using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
  using MainloopPipelinedO = typename cutlass::PipelineTmaAsync<kStages_dO>;
  using MainloopPipelineNoTMAdO = typename cutlass::PipelineAsync<kStages_dO>;
  using MainloopPipelinedRab = typename cutlass::PipelineAsync<kStages_dS>;

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
        cute::array_aligned<ElementAccum, dQacc_use_TMA ? cute::cosize_v<SmemLayoutdQaccum> : 0> smem_dqacc;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        cute::array_aligned<Element, Mma_dKV_is_RS ? 0 : cute::cosize_v<SmemLayoutPdS>, SmemAlignmentP> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
        SmemRabStorage smem_rab;
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

    // arbitrary mask
    struct {
      cute::array_aligned<int, cute::cosize_v<SmemLayoutValidBlockIds>> smem_valid_block_ids;
      int sm_valid_block_max;
    };

    struct {
      alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_KV;
      alignas(16) cutlass::arch::ClusterBarrier barrier_dKV;
      alignas(16) typename MainloopPipeline::SharedStorage pipeline_q;
      alignas(16) typename MainloopPipeline::SharedStorage pipeline_rab;
      alignas(16) typename MainloopPipelinedO::SharedStorage pipeline_do;
      alignas(16) typename MainloopPipelinedRab::SharedStorage pipeline_drab;
    };
  };

};

template<int kHeadDim_, int kBlockM_, int kBlockN_,
         bool Is_causal_, bool Is_context_, bool Is_target_, bool Is_local_, bool Is_arbitrary_,
         int kNFunc_, bool Has_rab_, bool Has_drab_, bool Is_deterministic_,
         int kStages_dO_, int kStages_dS_, int NumWarpGroups_=3,
         int Quant_mode_ = 0, typename Element_=cutlass::float_e5m2_t>
struct Hstu_bwd_kernel_traits_fp8 {
  using Element = Element_;
  using ElementAccum = float;
  using ElementRab = cutlass::half_t;
  using ElementOut = cutlass::half_t;
  using index_t = int64_t;

  static constexpr bool Is_fp8 = true;
  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int  kNFunc = Is_arbitrary ? kNFunc_ : 0;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Has_drab = Has_drab_;
  static constexpr bool Is_deterministic = Is_deterministic_;
  // Quantization mode: 0: cast to fp8, 1: 1xDIM&128x1 quantization,
  // 2: per-block quantization, 3: per-head quantization, 4: per-batch quantization, 5: per-tensor quantization.
  static constexpr int Quant_mode = Quant_mode_;
  // The number of threads.
  static constexpr int NumMmaWarpGroups = NumWarpGroups_ - 1;
  static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int NumMmaWarps = NumMmaWarpGroups * cutlass::NumWarpsPerWarpGroup;
  static constexpr int kNWarps = (NumMmaWarpGroups + 1) * cutlass::NumWarpsPerWarpGroup;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumdQWarpGroups = NumMmaWarpGroups;
  static constexpr int kNThreadsdQ = NumdQWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int NumEpilogueThreads = NumMmaThreads;

  static constexpr int AtomLayoutMSdP = 1;
  static constexpr int AtomLayoutNdKV = 2;
  static constexpr int AtomLayoutMdQ = 1;
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

  static constexpr int kBlockSeqQ_scale = Quant_mode == 1 ? 128 : kBlockM;
  static constexpr int kBlockSeqK_scale = Quant_mode == 1 ? 128 : kBlockN;
  static_assert(kBlockSeqQ_scale % kBlockM == 0);
  static_assert(kBlockSeqK_scale % kBlockN == 0);
  static constexpr int kMBlock_shared = kBlockSeqQ_scale / kBlockM;
  static constexpr int kNBlock_shared = kBlockSeqK_scale / kBlockN;

  static constexpr int kClusterN = 1;
  using ClusterShape = Shape<_1, Int<kClusterN>, _1>;

  static constexpr int kStages_dO = kStages_dO_;
  static constexpr int kStages_dS = kStages_dS_;
  static constexpr int kStages = kStages_dS;

  static constexpr bool SdP_swapAB = true;
  static constexpr bool dKV_swapAB = false;
  static constexpr bool dQ_swapAB = false;
  static_assert(SdP_swapAB == true, "SdP_swapAB=false is not supported for hstu fp8 backward.");

  static constexpr bool Mma_dKV_is_RS = true;
  static constexpr bool Mma_dQ_is_RS = false;

  using TileShapeAtomSdP = Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>;
  using AtomLayoutSdP = Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>;
  using TiledMmaSdP = decltype(make_tiled_mma(
      cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
      AtomLayoutSdP{}));

  static constexpr GMMA::Major PdS_Major = GMMA::Major::K;

  using TileShapeAtomdKV = Shape<Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>, Int<kBlockM>>;
  using AtomLayoutdKV = Layout<Shape<Int<AtomLayoutNdKV>, Int<NumMmaWarpGroups / AtomLayoutNdKV>, _1>>;
  using TiledMmadKV = decltype(make_tiled_mma(
      cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::K>(),
      AtomLayoutdKV{}));

  using TileShapeAtomdQ = Shape<Int<kBlockM>, Int<kHeadDim / (NumdQWarpGroups / AtomLayoutMdQ)>, Int<kBlockN>>;
  using AtomLayoutdQ = Layout<Shape<Int<AtomLayoutMdQ>, Int<NumdQWarpGroups / AtomLayoutMdQ>, _1>>;
  using TiledMmadQ = decltype(make_tiled_mma(
      cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::K>(),
      AtomLayoutdQ{}));

  using TiledMmaQKdescale = decltype(make_tiled_mma(
      UniversalFMA<ElementAccum, ElementAccum, ElementAccum>(),
      Layout<Shape<Int<NumMmaThreads/4>, _4, _1>, Stride<_4, _1, _0>>{},
      Tile<Layout<Shape<_8, Int<kBlockN/16>, _2>, Stride<_1, _16, _8>>, Layout<Shape<_4, _2, Int<kBlockM/8>>, Stride<_2, _1, _8>>, _1>{})
  );

  using TiledMmadKVdescale = decltype(make_tiled_mma(
      UniversalFMA<ElementAccum, ElementAccum, ElementAccum>(),
      Layout<Shape<Int<NumMmaThreads/4>, _4, _1>, Stride<_4, _1, _0>>{},
      Tile<Layout<Shape<_8, Int<kBlockN/16>, _2>, Stride<_1, _16, _8>>, Layout<Shape<_4, _2, Int<kHeadDim/8>>, Stride<_2, _1, _8>>, _1>{})
  );

  using TiledMmadQdescale = decltype(make_tiled_mma(
      UniversalFMA<ElementAccum, ElementAccum, ElementAccum>(),
      Layout<Shape<Int<NumMmaThreads/8>, Shape<_4, _2>, _1>, Stride<_4, Stride<_1, _128>, _0>>{},
      Tile<Layout<Shape<_8, Int<kBlockM/16>, _2>, Stride<_1, _16, _8>>, Layout<Shape<_4, _2, _2, Int<kHeadDim/16>>, Stride<_2, Int<kHeadDim/2>, _1, _8>>, _1>{})
  );

  using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                    Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>>());

  using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
  using GmemTiledCopyRab = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
  using GmemTiledCopydQaccum = cute::SM90_TMA_REDUCE_ADD;
  using GmemTiledCopydRab = cute::SM90_TMA_REDUCE_ADD;

  using SmemLayoutAtomRab = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementRab,
                                  decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutRab =
      decltype(tile_to_shape(SmemLayoutAtomRab{},
                make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

  using SmemLayoutRabt =
      decltype(cute::composition(SmemLayoutRab{},
                                  make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages>{}),
                                              make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

  using SmemLayoutAtomPdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<PdS_Major, ElementRab,
      Int<kBlockM / AtomLayoutMSdP>,
      Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>>());
  using SmemLayoutPdS = decltype(tile_to_shape(
      SmemLayoutAtomPdS{},
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages_dS>{}),
      std::conditional_t<PdS_Major == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

  using SmemLayoutAtomdSfp8 = decltype(cutlass::gemm::collective::detail::ss_smem_selector<PdS_Major, Element,
      Int<kBlockM / AtomLayoutMSdP>,
      Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>>());
  using SmemLayoutdSfp8 = decltype(tile_to_shape(
      SmemLayoutAtomdSfp8{},
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages_dS>{}),
      std::conditional_t<PdS_Major == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

  // for fp8 in-kernel transpose
  using TransposeShapeAtom = Shape<_64, _64>;
  using SmemLayoutAtom = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtom{}));
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtom{},
                make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
  using SmemLayoutdO =
      decltype(tile_to_shape(SmemLayoutAtom{},
                make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages_dO>{})));
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtom{},
                make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}))));

  using SmemLayoutAtomQt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                  decltype(cute::get<2>(TileShape_MNK{})), decltype(cute::get<0>(TileShape_MNK{}))>()); // dV = Pt * dO
  using SmemLayoutQt = decltype(tile_to_shape(SmemLayoutAtomQt{},
                make_shape(shape<2>(TileShape_MNK{}), shape<0>(TileShape_MNK{}), Int<kStages>{}), cute::Step<_1, _2, _3>{}));

  using SmemLayoutAtomdOt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                  decltype(cute::get<2>(TileShape_MNK{})), decltype(cute::get<0>(TileShape_MNK{}))>()); // dQ = dS * K (DIM, kBlockN)
  using SmemLayoutdOt = decltype(tile_to_shape(SmemLayoutAtomdOt{},
                make_shape(shape<2>(TileShape_MNK{}), shape<0>(TileShape_MNK{}), Int<kStages_dO>{}), cute::Step<_1, _2, _3>{}));

  using SmemLayoutAtomKt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                  decltype(cute::get<2>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>()); // dQ = dS * K (DIM, kBlockN)
  using SmemLayoutKt = decltype(tile_to_shape(SmemLayoutAtomKt{},
                make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}))));

  // for fp8 in-kernel transpose -- src layout
  using SmemLayoutDivide = decltype(tiled_divide(SmemLayoutQ{}, TransposeShapeAtom{}));
  using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
  using FactoringShape = decltype(make_shape(SmemShapeLDSM{},
      shape<1>(SmemLayoutDivide{}), shape<2>(SmemLayoutDivide{}), shape<3>(SmemLayoutDivide{})));
  using SmemLayoutTranspose = decltype(composition(SmemLayoutDivide{}, make_layout(FactoringShape{})));

  using SmemLayoutDivideK = decltype(tiled_divide(SmemLayoutK{}, TransposeShapeAtom{}));
  using SmemShapeKLDSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
  using FactoringShapeK = decltype(make_shape(SmemShapeKLDSM{},
      shape<1>(SmemLayoutDivideK{}), shape<2>(SmemLayoutDivideK{})));
  using SmemLayoutTransposeK = decltype(composition(SmemLayoutDivideK{}, make_layout(FactoringShapeK{})));

  // For fp8, this is the memory transpose.
  using SmemLayoutAtomT = decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtom{}));
  using SmemLayoutT =
      decltype(tile_to_shape(SmemLayoutAtomT{},
                make_shape(shape<2>(TileShape_MNK{}), shape<0>(TileShape_MNK{}), Int<kStages>{})));

  // for fp8 in-kernel transpose -- dst layout
  using SmemLayoutTrans =
      decltype(composition(SmemLayoutT{},
                            make_ordered_layout(product_each(shape(SmemLayoutQ{})), Step<_2, _1, _3>{})));
  using SmemLayoutDivideT = decltype(tiled_divide(SmemLayoutTrans{}, TransposeShapeAtom{}));
#ifndef NO_FP8_COLUMN_PERMUTE
  using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_8, _8>>;
#else
  using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
#endif
  using FactoringShapeT = decltype(make_shape(SmemShapeSTSM{},
      shape<1>(SmemLayoutDivideT{}), shape<2>(SmemLayoutDivideT{}), shape<3>(SmemLayoutDivideT{})));
  using SmemLayoutTransposeT = decltype(composition(SmemLayoutDivideT{}, make_layout(FactoringShapeT{})));

  using SmemLayoutTransK =
      decltype(composition(SmemLayoutKt{},
                          make_ordered_layout(product_each(shape(SmemLayoutK{})), Step<_2, _1>{})));
  using SmemLayoutDivideKt = decltype(tiled_divide(SmemLayoutTransK{}, TransposeShapeAtom{}));
#ifndef NO_FP8_COLUMN_PERMUTE
  using SmemShapeKSTSM = Shape<Shape<_16, _4>, Shape<_8, _8>>;
#else
  using SmemShapeKSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
#endif
  using FactoringShapeKT = decltype(make_shape(SmemShapeKSTSM{},
      shape<1>(SmemLayoutDivideKt{}), shape<2>(SmemLayoutDivideKt{})));
  using SmemLayoutTransposeKt = decltype(composition(SmemLayoutDivideKt{}, make_layout(FactoringShapeKT{})));

  using SmemLayoutPdSt =
      decltype(cute::composition(SmemLayoutPdS{},
                                  make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages_dS>{}),
                                              make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));
  using SmemLayoutdStfp8 =
      decltype(cute::composition(SmemLayoutdSfp8{},
                                  make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages_dS>{}),
                                              make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

  // Thread layout, 256 or 384 threads per row
  using R2SLayoutAtomdQaccum = Layout<Shape<Int<kNThreadsdQ>>, Stride<_1>>;
  using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                        Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
  using SmemLayoutdQaccum = Layout<Shape<Int<kBlockM * kHeadDim>>, Stride<_1>>;
  using SmemLayoutAtomdQaccumTMA =
      decltype(composition(Swizzle<0, 4, 3>{},  // We don't want any swizzle
                            Layout<Shape<Int<8>, Int<kHeadDim>>,
                            Stride<Int<kHeadDim>, _1>>{}));
  using SmemLayoutdQaccumTMA =
      decltype(tile_to_shape(SmemLayoutAtomdQaccumTMA{}, select<0, 2>(TileShape_MNK{})));
  using SmemLayoutdQaccumTMANoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutdQaccumTMA{}));

  static constexpr int kNumPdSStoreLoad = kBlockM * kBlockN / NumMmaThreads;
  using SmemCopyAtomdSStore = Copy_Atom<
      std::conditional_t<kNumPdSStoreLoad % 8 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>,
      ElementRab
  >;
  using SmemCopyAtomdSfp8Store = Copy_Atom<cute::DefaultCopy, Element>;

  using SmemCopyAtomRab_LDSM = Copy_Atom<cute::SM75_U16x8_LDSM_T, ElementRab>;
  using SmemCopyAtomRab = Copy_Atom<cute::DefaultCopy, Element>;

  static constexpr bool dQacc_use_TMA = kHeadDim < 256;
  // These are for the case where we don't use TMA to do atomicAdd on dQaccum, but just use direct atomicAdd.
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementAccum);
  static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, int(kNThreadsdQ));
  using GmemLayoutAtomdQaccum = Layout<Shape <Int<kNThreadsdQ / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                        Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopydQaccumAtomic = decltype(
      make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                      GmemLayoutAtomdQaccum{},
                      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 4 vals per store

  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : (kHeadDim % 32 == 0 ? 32 : 16);
  static constexpr int kSwizzle = kBlockKSmem == 64 ? 3 : (kBlockKSmem == 32 ? 2 : 1);
  using SmemLayoutAtomdQ =
        decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                 Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                 Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutdQ = decltype(tile_to_shape(SmemLayoutAtomdQ{}, TileShape_MK{}));
  using SmemLayoutdQt =
      decltype(cute::composition(SmemLayoutdQ{},
                                  make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{})),
                                              make_stride(Int<kBlockM>{}, _1{}))));
  using SmemCopyAtomdQ = Copy_Atom<cute::SM90_U32x4_STSM_N, ElementOut>;

  using SmemLayoutAtomdKV =
      decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                            Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                            Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, select<1, 2>(TileShape_MNK{})));
  using DescaleCopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, ElementAccum>;

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
  using MainloopPipelinedO = typename cutlass::PipelineTmaAsync<kStages_dO>;
  using MainloopPipelineNoTMAdO = typename cutlass::PipelineAsync<kStages_dO>;
  using MainloopPipelinedRab = typename cutlass::PipelineAsync<kStages_dS>;

  static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  static constexpr size_t SmemAlignmentdS = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  static constexpr size_t SmemAlignmentdSfp8 = cutlass::detail::alignment_for_swizzle(SmemLayoutdSfp8{});
  // Without this SmemAlignment, with hdim 256 we get "misaligned address" error in TMA
  static constexpr size_t SmemAlignmentQKVdO = kHeadDim % 256 == 0 ? 256 : 128;
  using SmemRabStorage = cute::conditional_t<
    Has_rab,
    cute::array_aligned<ElementRab, cute::cosize_v<SmemLayoutRab>, SmemAlignmentQKVdO>,
    cute::array_aligned<ElementRab, 128/sizeof(ElementRab)>>;
  static_assert(SmemAlignmentP >= 128 && SmemAlignmentdS >= 128, "Require at least 128B alignment");

  static constexpr size_t SmemAlignmentdKV = cutlass::detail::alignment_for_swizzle(SmemLayoutdKV{});
  static_assert(SmemAlignmentdKV >= 128, "Require at least 128B alignment");

  static constexpr int MaxSeqLenQ = 16 * 1024; // 16K
  static constexpr int MaxValidBlock = Is_arbitrary ? MaxSeqLenQ / kBlockM : 1; // 4KB
  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>; // 0 refer to number of valid blocks

  // Kernel level shared memory storage
  struct SharedStorage {
    union {
      struct {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutKt>, SmemAlignmentQKVdO> smem_kt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentQKVdO> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQt>, SmemAlignmentQKVdO> smem_qt;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdOt>, SmemAlignmentQKVdO> smem_dot;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdSfp8>, SmemAlignmentdSfp8> smem_ds_fp8;
        cute::array_aligned<ElementRab, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
        SmemRabStorage smem_rab;
        cute::array_aligned<ElementAccum, dQacc_use_TMA ? cute::cosize_v<SmemLayoutdQaccum> : 0> smem_dqacc;
        cute::array_aligned<ElementAccum, NumMmaWarps, SmemAlignmentQKVdO> smem_reduce_max;
      };
      struct {
        cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dk;
        cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dv;
      };
    };

    // arbitrary mask
    struct {
      cute::array_aligned<int, cute::cosize_v<SmemLayoutValidBlockIds>> smem_valid_block_ids;
      int sm_valid_block_max;
    };

    struct {
      alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_KV;
      alignas(16) cutlass::arch::ClusterBarrier barrier_dKV;
      alignas(16) typename MainloopPipeline::SharedStorage pipeline_q;
      alignas(16) typename MainloopPipeline::SharedStorage pipeline_rab;
      alignas(16) typename MainloopPipelinedO::SharedStorage pipeline_do;
      alignas(16) typename MainloopPipelinedRab::SharedStorage pipeline_drab;
      // for fp8 trans
      alignas(16) typename MainloopPipelinedO::SharedStorage pipeline_dot;
      alignas(16) typename MainloopPipeline::SharedStorage pipeline_qt;
    };
  };

  // 4 warps
  struct SmemTransposeFp8_64x64 {
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
    CUTLASS_DEVICE void operator()(const int &tid, SmemTensor &&s_in, SmemTensorOut &&s_out) {
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

  struct SmemTransposeKFp8_64x64 {
    using ldsm_thread_shape = Shape<_4, _1, _8, _4>;
    using ldsm_value_shape = Shape<_4, _4, _2, _1>;
    using ldsm_value_stride = Stride<_2, _8, _1, _0>;
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
    CUTLASS_DEVICE void operator()(const int &tid, SmemTensor &&s_in, SmemTensorOut &&s_out) {
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
};

////////////////////////////////////////////////////////////////////////////////////////////////////
