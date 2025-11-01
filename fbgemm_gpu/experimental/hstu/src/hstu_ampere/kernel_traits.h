/*
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"

using namespace cute;

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    typename elem_type = cutlass::half_t>
struct Flash_kernel_traits {
  using Element = elem_type;
  using ElementAccum = float;
  using index_t = int64_t;

  using MMA_Atom_Arch = std::conditional_t<
      std::is_same_v<elem_type, cutlass::half_t>,
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    bool Is_causal_,
    bool Is_target_,
    bool Is_context_,
    bool Is_local_,
    bool Is_arbitrary_,
    int  kNFunc_,
    bool Has_rab_,
    bool Paged_KV_,
    bool Is_Q_in_regs_ = false,
    bool Share_Q_K_smem_ = false,
    typename elem_type = cutlass::half_t,
    typename Base =
        Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Hstu_fwd_kernel_traits : public Base {
  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int  kNFunc = Is_arbitrary_ ? kNFunc_ : 0;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Paged_KV = Paged_KV_;

  using Element = typename Base::Element;
  using ElementAccum = typename Base::ElementAccum;
  using index_t = typename Base::index_t;
  using SmemCopyAtom = typename Base::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

  static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
  static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

  // The number of threads.
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKSmemRab = kBlockN % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKGmem =
      kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
  static constexpr int kSwizzleRab = kBlockKSmemRab == 32 ? 2 : 3;
  static constexpr int kStages = 1;

  using TiledMma = TiledMMA<
      typename Base::MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _16>>;
  static_assert(16 * kNWarps <= kBlockM);

  using SmemLayoutAtomQ = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

  using SmemLayoutAtomRab = decltype(composition(
      Swizzle<kSwizzleRab, 3, 3>{},
      Layout<
          Shape<_8, Int<kBlockKSmemRab>>,
          Stride<Int<kBlockKSmemRab>, _1>>{}));

  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutRab = decltype(tile_to_shape(
      SmemLayoutAtomRab{},
      Shape<Int<kBlockM>, Int<kBlockN>, Int<kStages>>{}));

  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

  // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
  using SmemLayoutVtransposed = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{},
          Stride<Int<kBlockN>, _1, Int<kHeadDim * kBlockN>>{})));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  using SmemLayoutAtomO = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopy, Element>;

  static constexpr int MaxSeqLenK = 64 * 1024; // 64K
  static constexpr int MaxValidBlock = MaxSeqLenK / kBlockN; // 4KB

  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>;
  using SmemLayoutMaxFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;
  using SmemLayoutMinFunc = Layout<Shape<Int<kNFunc/2 + 1>>, Stride<_1>>;

  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemRabSize =
      Has_rab ? size(SmemLayoutRab{}) * sizeof(Element) : 0;
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int kSmemSizeQKV = Share_Q_K_smem
      ? std::max(kSmemQSize, kSmemKVSize)
      : kSmemQSize + kSmemKVSize;
  static constexpr int kSmemSizeQKVRab = kSmemSizeQKV + kSmemRabSize;
  static constexpr int kSmemSizeQKVRabValidBlockIds = kSmemSizeQKVRab + (Is_arbitrary ? (size(SmemLayoutValidBlockIds{})) * sizeof(int) : 0);
  static constexpr int kSmemSize = kSmemSizeQKVRabValidBlockIds + (Is_arbitrary ? (size(SmemLayoutMaxFunc{}) + size(SmemLayoutMinFunc{}) + 1) * sizeof(int) : 0);

  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);

  static_assert(
      kHeadDim % kGmemElemsPerLoad == 0,
      "kHeadDim must be a multiple of kGmemElemsPerLoad");
  // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because
  // of bank conflicts. For example, for d=128, smem is split into 2 "pages",
  // each page takes care of columns 0-63 and 64-127. If we have 16 threads per
  // row for gmem read, when we write to smem, thread 0 - 7 will write to the
  // first page and thread 8 - 15 will write to the second page, to the same
  // banks.
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
  static constexpr int kGmemThreadsPerRowRab =
      kBlockKSmemRab / kGmemElemsPerLoad;
  static_assert(
      kNThreads % kGmemThreadsPerRow == 0,
      "kNThreads must be a multiple of kGmemThreadsPerRow");
  static_assert(
      kNThreads % kGmemThreadsPerRowRab == 0,
      "kNThreads must be a multiple of kGmemThreadsPerRow_Rab");

  using GmemLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>; // (128/4, 4) or (128/8, 8)
  using GmemLayoutAtomRab = Layout<
      Shape<
          Int<kNThreads / kGmemThreadsPerRowRab>,
          Int<kGmemThreadsPerRowRab>>,
      Stride<Int<kGmemThreadsPerRowRab>, _1>>; // (128/4, 4) or (128/8, 8)

  // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we
  // won't be reading from the same address by the same threadblock. This is
  // slightly faster.
  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{},
      GmemLayoutAtom{}, // 16*8
      Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per read
                                 // kGmemElemsPerLoad

  static constexpr int rab_row_size = Has_rab ? kBlockN / 16 : 1;
  using GmemTiledCopyRab = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{},
      GmemLayoutAtomRab{}, // 16*8
      Layout<Shape<Int<rab_row_size>, _8>, Stride<_8, _1>>{}));

  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per store
                                 // kGmemElemsPerLoad
};

template <
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNWarps_,
    bool Is_causal_,
    bool Is_target_,
    bool Is_context_,
    bool Is_local_,
    bool Is_arbitrary_,
    int kNFunc_,
    bool Is_deterministic_,
    bool Has_rab_,
    bool Has_drab_,
    bool Rab_one_head_,
    int AtomLayoutMSdP_ = 2,
    int AtomLayoutNdKV_ = 4,
    int AtomLayoutMdQ_ = 4,
    bool Is_V_in_regs_ = false,
    typename elem_type = cutlass::half_t,
    typename Base =
        Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type>>
struct Hstu_bwd_kernel_traits : public Base {
  using Element = typename Base::Element;
  using ElementAccum = typename Base::ElementAccum;
  using index_t = typename Base::index_t;
  using SmemCopyAtom = typename Base::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

  static constexpr bool Is_causal = Is_causal_;
  static constexpr bool Is_target = Is_target_;
  static constexpr bool Is_context = Is_context_;
  static constexpr bool Is_local = Is_local_;
  static constexpr bool Is_arbitrary = Is_arbitrary_;
  static constexpr int  kNFunc = Is_arbitrary_ ? kNFunc_ : 0;
  static constexpr bool Is_deterministic = Is_deterministic_;
  static constexpr bool Has_rab = Has_rab_;
  static constexpr bool Paged_KV = false;
  static constexpr bool Has_drab = Has_drab_;
  static constexpr bool Rab_one_head = Rab_one_head_;
  static constexpr bool Is_V_in_regs = Is_V_in_regs_;

  // The number of threads.
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kStages = kHeadDim <= 64 ? 2 : 1;
  static_assert(kHeadDim % 32 == 0);
  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKSmemRab = kBlockN % 64 == 0 ? 64 : 32;
  static constexpr int kBlockKGmem =
      kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

  static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
  static constexpr int AtomLayoutNdKV = AtomLayoutNdKV_;
  static constexpr int AtomLayoutMdQ = AtomLayoutMdQ_;
  static_assert(kNWarps % AtomLayoutMSdP == 0);
  static_assert(kNWarps % AtomLayoutNdKV == 0);
  static_assert(kNWarps % AtomLayoutMdQ == 0);

  using TiledMmaSdP = TiledMMA<
      typename Base::MMA_Atom_Arch,
      Layout<Shape<Int<AtomLayoutMSdP>, Int<kNWarps / AtomLayoutMSdP>, _1>>,
      Tile<Int<16 * AtomLayoutMSdP>, Int<16 * kNWarps / AtomLayoutMSdP>, _16>>;

  using TiledMmadKV = TiledMMA<
      typename Base::MMA_Atom_Arch,
      Layout<Shape<Int<AtomLayoutNdKV>, Int<kNWarps / AtomLayoutNdKV>, _1>>,
      Tile<Int<16 * AtomLayoutNdKV>, Int<16 * kNWarps / AtomLayoutNdKV>, _16>>;

  using TiledMmadQ = TiledMMA<
      typename Base::MMA_Atom_Arch,
      Layout<Shape<
          Int<AtomLayoutMdQ>,
          Int<kNWarps / AtomLayoutMdQ>,
          _1>>, // 2x4x1 or 4x2x1 thread group
      Tile<Int<16 * AtomLayoutMdQ>, Int<16 * kNWarps / AtomLayoutMdQ>, _16>>;

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  using SmemLayoutAtomQdO = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutdO = decltype(tile_to_shape(
      SmemLayoutAtomQdO{},
      make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));
  using SmemLayoutdOtransposed = decltype(composition(
      SmemLayoutdO{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockM>>{}, GenRowMajor{})));
  using SmemLayoutdOtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutdOtransposed{}));

  using SmemLayoutQ = decltype(
      tile_to_shape(SmemLayoutAtomQdO{},
      make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages>{})));
  using SmemLayoutQtransposed = decltype(composition(
      SmemLayoutQ{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockM>, Int<kStages>>{},
                  Stride<Int<kBlockM>, _1, Int<kHeadDim * kBlockM>>{})));
  using SmemLayoutQtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutQtransposed{}));

  using SmemLayoutAtomKV = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<
          Shape<Int<kBlockM / kNWarps>, Int<kBlockKSmem>>,
          Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtomKV{},
      make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));
  using SmemLayoutKtransposed = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
  using SmemLayoutKtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutKtransposed{}));

  static_assert(kBlockN >= 32);
  static constexpr int kPBlockN = kBlockN >= 64 ? 64 : 32;
  static constexpr int kSwizzlePdS = 3;

  using SmemLayoutAtomPdS = decltype(composition(
      Swizzle<kSwizzlePdS, 3, 3>{},
      Layout<Shape<Int<kBlockM>, Int<kPBlockN>>, Stride<Int<kPBlockN>, _1>>{}));
  using SmemLayoutPdS = decltype(tile_to_shape(
      SmemLayoutAtomPdS{},
      make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
  using SmemLayoutPdStransposed = decltype(composition(
      SmemLayoutPdS{},
      make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));
  using SmemLayoutPdStransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutPdStransposed{}));

  using SmemLayoutRab = decltype(tile_to_shape(
      SmemLayoutAtomPdS{},
      Shape<Int<kBlockM>, Int<kBlockN>>{}));

  using SmemLayoutAtomdKV = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutdKV = decltype(tile_to_shape(
      SmemLayoutAtomdKV{},
      make_shape(Int<kBlockN>{}, Int<kHeadDim>{})));

  using SmemLayoutAtomdQ = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutdQ = decltype(tile_to_shape(
      SmemLayoutAtomdQ{},
      make_shape(Int<kBlockM>{}, Int<kHeadDim>{})));

  static constexpr int MaxSeqLenQ = 64 * 1024; // 64K
  static constexpr int MaxValidBlock = MaxSeqLenQ / kBlockM; // 4KB
  using SmemLayoutValidBlockIds = Layout<Shape<Int<MaxValidBlock>>, Stride<_1>>; // 0 refer to number of valid blocks

  // Double buffer for sQ
  static constexpr int kSmemdOSize = size(SmemLayoutdO{}) * sizeof(Element);
  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int kSmemdSSize = size(SmemLayoutPdS{}) * sizeof(Element);
  static constexpr int kSmemPSize = size(SmemLayoutPdS{}) * sizeof(Element);
  static constexpr int kSmemRabSize =
      Has_rab ? size(SmemLayoutRab{}) * sizeof(Element) : 0;
  static constexpr int kSmemdQSize = size(SmemLayoutdQ{}) * sizeof(Element);
  static constexpr int kSmemSize1colblock = kSmemdOSize + kSmemQSize + kSmemRabSize +
      (!Is_V_in_regs
           ? kSmemKVSize + kSmemdSSize + kSmemPSize
           : std::max(kSmemKVSize, kSmemKVSize / 2 + kSmemdSSize + kSmemPSize));
  static constexpr int kSmemSize1colblock_validblockids = kSmemSize1colblock + (Is_arbitrary ? (size(SmemLayoutValidBlockIds{})) * sizeof(int) : 0);
  static constexpr int kSmemSize = kSmemSize1colblock_validblockids + (Is_arbitrary ? 128 : 0);
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  using SmemCopyAtomPdS = Copy_Atom<AutoVectorizingCopy, elem_type>;
  using SmemCopyAtomdKV = Copy_Atom<AutoVectorizingCopy, elem_type>;
  using SmemCopyAtomdQ = Copy_Atom<AutoVectorizingCopy, elem_type>;

  ////////////////////////////////////////////////////////////////////////////////////////////////////

  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(
      kHeadDim % kGmemElemsPerLoad == 0,
      "kHeadDim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
  static_assert(
      kNThreads % kGmemThreadsPerRow == 0,
      "kNThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;

  // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we
  // won't be reading from the same address by the same threadblock. This is
  // slightly faster.
  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, elem_type>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per read
  using GmemTiledCopydO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, elem_type>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per store
  using GmemTiledCopydKV = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, elem_type>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per store
  using GmemTiledCopydQ = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, elem_type>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per store
  using GmemLayoutAtomdQaccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, ElementAccum>{},
      Layout<
          Shape<Int<kNWarps>, _32>, // Thread layout, 8 threads per row
          Stride<_32, _1>>{},
      Layout<Shape<_1, _1>>{})); // Val layout, 1 val per store

  using GmemTiledCopydRab = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, elem_type>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, _8>>{})); // Val layout, 8 vals per store

  static constexpr int kGmemElemsPerdRab =
      2; // The biggest bit_width supported for atomicAdd on SM8X GPUs is
         // 32-bit.
  static constexpr int kGmemThreadsPerRowdRab = kBlockN / kGmemElemsPerdRab;
  using ElementV2 = std::conditional_t<
      std::is_same_v<elem_type, cutlass::half_t>,
      __half2,
      __nv_bfloat162>;
  using GmemLayoutAtomdRab = Layout<
      Shape<
          Int<kNThreads / kGmemThreadsPerRowdRab>,
          Int<kGmemThreadsPerRowdRab>>,
      Stride<Int<kGmemThreadsPerRowdRab>, _1>>;
  using GmemTiledAtomdRab = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopy, Element>{},
      GmemLayoutAtomdRab{},
      Layout<Shape<_1, Int<kGmemElemsPerdRab>>>{})); // Val layout, 2 vals per
                                                     // store
};
////////////////////////////////////////////////////////////////////////////////////////////////////
