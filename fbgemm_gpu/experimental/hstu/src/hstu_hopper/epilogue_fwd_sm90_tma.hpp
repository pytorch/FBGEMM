/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

using namespace cute;

template <typename Ktraits, typename Seqlen_traits>
struct CollectiveEpilogueFwd {
  using Element = typename Ktraits::OutputType;
  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr int kBlockN = Ktraits::kBlockN;
  static constexpr int kHeadDim = Ktraits::kHeadDim;
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

  static constexpr int kNWarps = Ktraits::kNWarps;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int NumMmaThreads = kNThreads - NumCopyThreads;

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

  using SmemCopyAtomO = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;
  using SharedStorage = cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>>;

  // These are for storing the output tensor without TMA (e.g., for setting output to zero and var-seq-len)
  static constexpr int kNumVecElem = ceil_div(128, sizeof_bits_v<Element>);
  static_assert(kHeadDim % kNumVecElem == 0);
  static constexpr int kNumThreadsPerRow = kHeadDim / kNumVecElem;
  static_assert(NumMmaThreads % kNumThreadsPerRow == 0);
  static constexpr int kNumRows = NumMmaThreads / kNumThreadsPerRow;
  using TiledCopyOAtom = cute::Copy_Atom<cute::UniversalCopy<cutlass::uint128_t>, Element>;
  using TiledCopyOThrLayout = decltype(cute::make_layout(
      cute::make_shape(Int<kNumRows>{}, Int<kNumThreadsPerRow>{}),
      LayoutRight{}));
  using TiledCopyOValLayout = decltype(cute::make_layout(
      cute::make_shape(_1{}, Int<kNumVecElem>{}),
      LayoutRight{}));
  using TiledCopyO = decltype(make_tiled_copy(
      TiledCopyOAtom{},
      TiledCopyOThrLayout{}, // Thr layout
      TiledCopyOValLayout{} // Val layout
  ));

  // used for rmem -> smem O copy in fp8 kernel to undo column permutation
  using ThreadLayoutrO = Layout<Shape<_8, Int<kBlockM/16>, _4, _1>,
                                Stride<_4, _32, _1, _0>>;
  using ValueLayoutrO = Layout<Shape<_1, _2, Shape<_2, _2>, Int<kHeadDim/16>>,
                              Stride<_0, _2, Stride<_4, _1>, _8>>;
  using TiledCopyrO = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint16_t>, Element>{},
                    ThreadLayoutrO{}, ValueLayoutrO{}));
  using TiledCopyShaperO = Shape<_8, Int<kBlockM/8>, _16, Int<kHeadDim/16>>;
  using SmemLayoutrO = decltype(composition(SmemLayoutO{}, Layout<TiledCopyShaperO>{}));

  // Host side kernel arguments
  struct Arguments {
      Element* ptr_O;
      typename Seqlen_traits::LayoutT const layout_O;
  };

  // Device side kernel params
  struct Params {
      Element* ptr_O;
      typename Seqlen_traits::LayoutT const layout_O;
  };

  static Params
  to_underlying_arguments(Arguments const& args) {
    Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.layout_O);
    return {args.ptr_O, args.layout_O};
  }

  template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
  CUTLASS_DEVICE void
  store(Params const& epilogue_params,
        FrgTensorO const& tOrO,
        SharedStorage& shared_storage,
        TiledMma tiled_mma,
        int thread_idx,
        cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
        const Seqlen_traits& seqlen_traits_q) {

    auto [m_block, bidh, bidb] = block_coord;
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

    Tensor tOrO_out = make_tensor_like<Element>(tOrO);
    flash::convert_type_safe(tOrO, tOrO_out);
    Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    // Make sure all WGs have finished reading V
    cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::ValueEmpty));
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA

    Tensor caccO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor taccOcO = thread_mma.partition_C(caccO);
    static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
    static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
    Tensor taccOcO_row = taccOcO(make_coord(_0{}, _, _0{}), _, _0{});

    cutlass::arch::NamedBarrier::sync(NumMmaThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    TiledCopyO gmem_tiled_copy_O;
    flash::write_tiled<NumCopyThreads>(
        epilogue_params.ptr_O, gmem_tiled_copy_O,
        epilogue_params.layout_O, select<0, 2>(TileShape_MNK{}), sO,
        m_block, bidh, bidb, seqlen_traits_q
    );
  }

  template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
  CUTLASS_DEVICE void
  store_fp8(Params const& epilogue_params,
        FrgTensorO const& tOrO,
        SharedStorage& shared_storage,
        TiledMma tiled_mma,
        int thread_idx,
        cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
        const Seqlen_traits& seqlen_traits_q) {
    auto [m_block, bidh, bidb] = block_coord;

    TiledCopyrO rmem_tiled_copy_O;
    Tensor sOacc = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutrO{});
    auto rmem_thr_copy_O = rmem_tiled_copy_O.get_thread_slice(thread_idx);

    Tensor taccOsO = rmem_thr_copy_O.partition_D(sOacc);
    Tensor tOrO_out = make_tensor_like<Element>(tOrO);
    flash::convert_type_safe(tOrO, tOrO_out);
    Tensor taccOrO = make_tensor(tOrO_out.data(), shape(taccOsO));

    // Make sure all WGs have finished reading V
    cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(FwdNamedBarriers::ValueEmpty));
    cute::copy(rmem_tiled_copy_O, taccOrO, taccOsO);
    cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
    Tensor caccO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor taccOcO = thread_mma.partition_C(caccO);
    static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
    static_assert(decltype(size<0, 1>(taccOcO))::value == 2);
    Tensor taccOcO_row = taccOcO(make_coord(_0{}, _, _0{}), _, _0{});

    cutlass::arch::NamedBarrier::sync(NumMmaThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    TiledCopyO gmem_tiled_copy_O;
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    flash::write_tiled<NumCopyThreads>(
        epilogue_params.ptr_O, gmem_tiled_copy_O,
        epilogue_params.layout_O, select<0, 2>(TileShape_MNK{}), sO,
        m_block, bidh, bidb, seqlen_traits_q
    );
  }

  CUTLASS_DEVICE void
  store_tail() {
      tma_store_wait<0>();
  }

  // Write 0 to output
  template<typename SharedStorage>
  CUTLASS_DEVICE void
  store_zero(
        Params const& epilogue_params,
        SharedStorage& shared_storage,
        int thread_idx,
        cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
        const Seqlen_traits& seqlen_traits_q) {
    auto [m_block, bidh, bidb] = block_coord;
    Tensor mO = make_tensor(make_gmem_ptr(epilogue_params.ptr_O), epilogue_params.layout_O);
    Tensor gO = seqlen_traits_q.get_local_tile_tensor(
        mO, select<0, 2>(TileShape_MNK{}), bidh, bidb)(_, _, m_block);

    TiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_fragment_like(tOgO);
    clear(tOrO);
    // Construct identity layout for sO
    Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    #pragma unroll
    for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(epilogue_params.layout_O.shape()); }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, seqlen_traits_q.actual_seq_len - m_block * kBlockM
    );
    static_assert(kBlockM <= NumMmaThreads);
  }

};

} // namespace flash
