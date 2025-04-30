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
struct CollectiveEpilogueBwd {

    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using Element = typename Ktraits::Element;
    static constexpr int NumEpilogueThreads = Ktraits::NumEpilogueThreads;
    static constexpr bool dKV_swapAB = Ktraits::dKV_swapAB;
    static constexpr int AtomLayoutKdKV = Ktraits::AtomLayoutKdKV;

    using GmemTiledCopydKVTMA = cute::SM90_TMA_STORE;

    // These are for storing the output tensor without TMA (e.g., for setting output to zero)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(get<2>(TileShape_MNK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, NumEpilogueThreads);
    static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using SmemLayoutAtomdKVTMA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), Int<CUTE_STATIC_V(cute::get<2>(TileShape_MNK{})) / AtomLayoutKdKV>>());
    using SmemLayoutdKVTMA = decltype(tile_to_shape(SmemLayoutAtomdKVTMA{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdKVtTMA =
        decltype(cute::composition(SmemLayoutdKVTMA{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

    // If we don't use TMA
    static constexpr int kBlockKSmem = Ktraits::kBlockKSmem;
    static constexpr int kSwizzle = Ktraits::kSwizzle;
    using SmemLayoutAtomdKV =
        decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                             Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                             Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, select<1, 2>(TileShape_MNK{})));
    using SmemLayoutdKVt =
        decltype(cute::composition(SmemLayoutdKV{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(decltype(get<1>(TileShape_MNK{})){}, _1{}))));

    using SmemCopyAtomdKV = Copy_Atom<
        std::conditional_t<!dKV_swapAB, cute::SM90_U32x4_STSM_N, cute::SM90_U16x8_STSM_T>, Element>;

    using TMA_dKV = decltype(make_tma_copy(
        GmemTiledCopydKVTMA{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
                    repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
                    typename Seqlen_traits::StrideT{}),
        SmemLayoutdKVTMA{},
        select<1, 2>(TileShape_MNK{}),
        _1{}));  // no mcast for dKV

    // Host side kernel arguments
    struct Arguments {
        Element* ptr_dK;
        typename Seqlen_traits::LayoutT layout_dK;
        Element* ptr_dV;
        typename Seqlen_traits::LayoutT layout_dV;
    };

    // Device side kernel params
    struct Params {
        Element* ptr_dK;
        typename Seqlen_traits::LayoutT layout_dK;
        Element* ptr_dV;
        typename Seqlen_traits::LayoutT layout_dV;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        return {args.ptr_dK, args.layout_dK, args.ptr_dV, args.layout_dV};
    }

    template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& epilogue_params,
          FrgTensorO const& tdKrdK,
          FrgTensorO const& tdVrdV,
          SharedStorage& shared_storage,
          TiledMma tiled_mma,
          int thread_idx,
          cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
          const Seqlen_traits& seqlen_traits_k) {

        auto [n_block, bidh, bidb] = block_coord;
        Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_dk.data()), SmemLayoutdKV{}));
        Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_dv.data()), SmemLayoutdKV{}));
        Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_dk.data()), SmemLayoutdKVt{}));
        Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.smem_dv.data()), SmemLayoutdKVt{}));
        auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma);
        auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(thread_idx);

        Tensor tdVrdV_out = make_tensor_like<Element>(tdVrdV);
        flash::convert_type_safe(tdVrdV, tdVrdV_out);
        Tensor tdKrdK_out = make_tensor_like<Element>(tdKrdK);
        flash::convert_type_safe(tdKrdK, tdKrdK_out);
        Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV_out);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt));     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // Make sure all WGs have finished reading K and V
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
        cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        int const offset = seqlen_traits_k.cu_seq_len[bidb];
        int const seqlen = seqlen_traits_k.actual_seq_len;

        Tensor mdK = make_tensor(make_gmem_ptr(epilogue_params.ptr_dK), epilogue_params.layout_dK)(_, _, bidh);
        Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
        Tensor mdV = make_tensor(make_gmem_ptr(epilogue_params.ptr_dV), epilogue_params.layout_dV)(_, _, bidh);
        Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)

        GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
        Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        Tensor tdKVsdV = gmem_thr_copy_dKV.partition_S(sdV); // (TMA, TMA_M, TMA_K)
        Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdKVsdK = gmem_thr_copy_dKV.partition_S(sdK); // (TMA, TMA_M, TMA_K)
        Tensor tdKVrdV = make_fragment_like(tdKVgdV);
        Tensor tdKVrdK = make_fragment_like(tdKVgdK);
        cute::copy(gmem_tiled_copy_dKV, tdKVsdV, tdKVrdV);
        cute::copy(gmem_tiled_copy_dKV, tdKVsdK, tdKVrdK);
        // Construct identity layout for gdKV
        Tensor cdKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdV)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(epilogue_params.layout_dK.shape()); }
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdV, tdKVgdV, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
        );
        flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKVrdK, tdKVgdK, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
        );
    }

    // Write 0 to dK and dV
    CUTLASS_DEVICE void
    store_zero(
         Params const& epilogue_params,
         int thread_idx,
         cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
         const Seqlen_traits& seqlen_traits_k) {
      static constexpr int kBlockN = get<1>(TileShape_MNK{});
      auto [n_block, bidh, bidb] = block_coord;
      int const offset = seqlen_traits_k.cu_seq_len[bidb];
      int const seqlen = seqlen_traits_k.actual_seq_len;

      Tensor mdK = make_tensor(make_gmem_ptr(epilogue_params.ptr_dK), epilogue_params.layout_dK)(_, _, bidh);
      Tensor gdK = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)
      Tensor mdV = make_tensor(make_gmem_ptr(epilogue_params.ptr_dV), epilogue_params.layout_dV)(_, _, bidh);
      Tensor gdV = local_tile(cute::domain_offset(make_coord(offset, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (M, K)

      GmemTiledCopydKV gmem_tiled_copy_dKV;
      auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);
      Tensor tdKVgdK = gmem_thr_copy_dKV.partition_D(gdK);
      Tensor tdKVgdV = gmem_thr_copy_dKV.partition_D(gdV);
      Tensor tdKVrdKV = make_fragment_like(tdKVgdK);
      clear(tdKVrdKV);
      // Construct identity layout for gdKV
      Tensor cdKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
      // Repeat the partitioning with identity layouts
      Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
      Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKVgdK)));
      #pragma unroll
      for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(_0{}, _0{}, k)) < get<1>(epilogue_params.layout_dK.shape()); }
      // Clear_OOB_K must be false since we don't want to write zeros to gmem
      flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdK, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
      );
      flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dKV, tdKVrdKV, tdKVgdV, tdKVcdKV, tdKVpdKV, seqlen - n_block * kBlockN
      );
    }

};

} // namespace flash
