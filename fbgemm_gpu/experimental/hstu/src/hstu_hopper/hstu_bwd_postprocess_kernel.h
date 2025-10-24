/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao. Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/arch/barrier.h"

#include "seq_len.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <typename Ktraits, typename Seqlen_traits>
class FlashAttnBwdPostprocessConvertdQ {
 public:
  // Type Aliases
  using Element = typename Ktraits::ElementOut;
  using ElementAccum = typename Ktraits::ElementAccum;
  using TileShape_MK = typename Ktraits::TileShape_MK;
  using SmemLayoutdQaccumTMA = typename Ktraits::SmemLayoutdQaccumTMA;
  using TiledMma = typename Ktraits::TiledMmadQ;
  static constexpr bool dQ_swapAB = Ktraits::dQ_swapAB;
  static constexpr int kNThreads = Ktraits::kNThreadsdQ;

  static constexpr uint32_t MaxThreadsPerBlock = kNThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 2;

  static constexpr int kHeadDim = get<1>(TileShape_MK{});
  using R2SLayoutAtomdQaccum = Layout<Shape<Int<kNThreads>>, Stride<_1>>;
  using R2STiledCopydQaccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      R2SLayoutAtomdQaccum{},
      Layout<Shape<_4>>{})); // Val layout, 4 vals per read
  static constexpr int SmemdQaccumSize = size(TileShape_MK{});
  static_assert(
      size(TileShape_MK{}) == size(SmemLayoutdQaccumTMA{}),
      "TileShape_MK and SmemLayoutdQaccumTMA must have the same size");
  using SmemLayoutdQaccum = Layout<Shape<Int<SmemdQaccumSize>>, Stride<_1>>;

  using SmemLayoutdQ = typename Ktraits::SmemLayoutdQ;
  using SmemLayoutdQt = typename Ktraits::SmemLayoutdQt;
  using SmemCopyAtomdQ = typename Ktraits::SmemCopyAtomdQ;

  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(
      kHeadDim % kGmemElemsPerLoad == 0,
      "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kGmemThreadsPerRow =
      cutlass::gcd(kHeadDim / kGmemElemsPerLoad, int(MaxThreadsPerBlock));
  static_assert(
      MaxThreadsPerBlock % kGmemThreadsPerRow == 0,
      "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<
      Shape<
          Int<MaxThreadsPerBlock / kGmemThreadsPerRow>,
          Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{})); // Val layout, 8 or 16 vals
                                                     // per load

  using GmemTiledCopydQaccum = cute::SM90_TMA_LOAD;

  struct SharedStorage : cute::aligned_struct<128> {
    cute::
        array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccumTMA>, 1024>
            smem_dqacc;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdQ>> smem_dq;
    alignas(16) cutlass::arch::ClusterTransactionBarrier barrier_dQaccum;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  using TMA_dQaccum = decltype(make_tma_copy(
      GmemTiledCopydQaccum{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementAccum*>(nullptr)),
          repeat_like(typename Seqlen_traits::StrideT{}, int32_t(0)),
          typename Seqlen_traits::StrideT{}),
      SmemLayoutdQaccumTMA{},
      TileShape_MK{},
      _1{})); // no mcast for dQ

  // Device side arguments
  struct Arguments {
    ElementAccum const* ptr_dQaccum;
    typename Seqlen_traits::LayoutT layout_dQaccum;
    Element* ptr_dQ;
    typename Seqlen_traits::LayoutT layout_dQ;
    const int total_q;
    const int seqlen_q;
    const float alpha;
    const int* cu_seqlens_q;
    const int* num_targets;
    const int* num_contexts;
  };

  // Kernel entry point API
  struct Params {
    TMA_dQaccum tma_load_dQaccum;
    typename Seqlen_traits::LayoutT layout_dQaccum;
    Element* ptr_dQ;
    typename Seqlen_traits::LayoutT layout_dQ;
    const int total_q;
    const int seqlen_q;
    const float alpha;
    const int* cu_seqlens_q;
    const int* num_targets;
    const int* num_contexts;
  };

  // Convert to underlying arguments. In this case, a simple copy for the
  // aliased type.
  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mdQaccum =
        make_tensor(make_gmem_ptr(args.ptr_dQaccum), args.layout_dQaccum);
    TMA_dQaccum tma_load_dQaccum = make_tma_copy(
        GmemTiledCopydQaccum{},
        mdQaccum,
        SmemLayoutdQaccumTMA{},
        TileShape_MK{},
        _1{}); // no mcast for dQaccum
    return {
        tma_load_dQaccum,
        args.layout_dQaccum,
        args.ptr_dQ,
        args.layout_dQ,
        args.total_q,
        args.seqlen_q,
        args.alpha,
        args.cu_seqlens_q,
        args.num_targets,
        args.num_contexts};
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    static constexpr int kBlockM = get<0>(TileShape_MK{});
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    Tensor sdQaccumTMA = make_tensor(
        make_smem_ptr(shared_storage.smem_dqacc.data()),
        SmemLayoutdQaccumTMA{});
    Tensor sdQaccum = make_tensor(
        make_smem_ptr(shared_storage.smem_dqacc.data()), SmemLayoutdQaccum{});
    Tensor sdQ = make_tensor(
        make_smem_ptr(shared_storage.smem_dq.data()), SmemLayoutdQ{});
    Tensor sdQt = make_tensor(
        make_smem_ptr(shared_storage.smem_dq.data()), SmemLayoutdQt{});

    int const thread_idx = threadIdx.x;
    int const m_block = blockIdx.x;
    int const bidh = blockIdx.y;
    int const bidb = blockIdx.z;

    Seqlen_traits seqlen_traits_q(
        params.total_q,
        params.seqlen_q,
        params.cu_seqlens_q,
        params.num_targets,
        params.num_contexts);
    seqlen_traits_q.init(bidb);
    int const max_seq_len_q = seqlen_traits_q.max_seq_len;
    int const seqlen = seqlen_traits_q.actual_seq_len;
    if (m_block * kBlockM >= seqlen) {
      return;
    }

    int lane_predicate = cute::elect_one_sync();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    // Issue Tma Descriptor Prefetch from a single thread
    if (warp_idx == 0 && lane_predicate) {
      cute::prefetch_tma_descriptor(
          params.tma_load_dQaccum.get_tma_descriptor());
      shared_storage.barrier_dQaccum.init(1 /*numThreads*/);
    }
    __syncthreads();

    // Step 1: TMA to load dQaccum from gmem to smem
    int const offset_padded =
        (seqlen_traits_q.cu_seq_len[bidb] + bidb * kBlockM) / kBlockM * kBlockM;
    Tensor mdQaccum = params.tma_load_dQaccum.get_tma_tensor(
        params.layout_dQaccum.shape())(_, _, bidh);
    Tensor gdQaccum = local_tile(
        domain_offset(make_coord(offset_padded, _0{}), mdQaccum),
        TileShape_MK{},
        make_coord(m_block, _0{})); // (M, K)
    auto block_tma_dQ = params.tma_load_dQaccum.get_slice(_0{});
    Tensor tdQgdQaccumTMA =
        block_tma_dQ.partition_D(gdQaccum); // (TMA, TMA_M, TMA_K)
    Tensor tdQsdQaccumTMA =
        block_tma_dQ.partition_S(sdQaccumTMA); // (TMA, TMA_M, TMA_K)
    static constexpr uint32_t TmaTransactionBytesdQaccum =
        static_cast<uint32_t>(
            size(SmemLayoutdQaccumTMA{}) * cute::sizeof_bits_v<ElementAccum> /
            8);
    if (warp_idx == 0 && lane_predicate) {
      shared_storage.barrier_dQaccum.arrive_and_expect_tx(
          TmaTransactionBytesdQaccum);
      copy(
          params.tma_load_dQaccum.with(
              reinterpret_cast<
                  cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                  shared_storage.barrier_dQaccum),
              0 /*mcast_mask*/),
          tdQgdQaccumTMA,
          tdQsdQaccumTMA);
    }
    shared_storage.barrier_dQaccum.wait(0);

    // Step 2: Load dQaccum from smem to register, then convert fp32 ->
    // fp16/bf16
    R2STiledCopydQaccum s2r_tiled_copy_dQaccum;
    auto s2r_thr_copy_dQaccum =
        s2r_tiled_copy_dQaccum.get_thread_slice(thread_idx);
    Tensor tdQsdQaccum = s2r_thr_copy_dQaccum.partition_S(sdQaccum);
    TiledMma tiled_mma_dQ;
    Tensor taccdQrdQaccum = partition_fragment_C(
        tiled_mma_dQ,
        select < !dQ_swapAB ? 0 : 1,
        !dQ_swapAB ? 1 : 0 > (TileShape_MK{}));
    CUTE_STATIC_ASSERT_V(size(taccdQrdQaccum) == size(tdQsdQaccum));
    Tensor tdQrdQaccum = s2r_thr_copy_dQaccum.retile_D(taccdQrdQaccum);
    cute::copy(s2r_tiled_copy_dQaccum, tdQsdQaccum, tdQrdQaccum);
    if constexpr (!Ktraits::Has_drab) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tdQrdQaccum); ++i) {
            tdQrdQaccum(i) /= max_seq_len_q;
            tdQrdQaccum(i) *= params.alpha;
        }
    }
    // Convert tdQrdQ from fp32 to fp16
    Tensor rdQ = make_tensor_like<Element>(taccdQrdQaccum);
    flash::convert_type_safe(taccdQrdQaccum, rdQ);

    // Step 3: Copy dQ from register to smem
    auto smem_tiled_copy_dQ = make_tiled_copy_C(SmemCopyAtomdQ{}, tiled_mma_dQ);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(thread_idx);
    Tensor taccdQrdQ =
        smem_thr_copy_dQ.retile_S(rdQ); // ((Atom,AtomNum), MMA_N, MMA_N)
    if constexpr (!dQ_swapAB) {
      Tensor taccdQsdQ =
          smem_thr_copy_dQ.partition_D(sdQ); // ((Atom,AtomNum),PIPE_M,PIPE_N)
      cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    } else {
      Tensor taccdQsdQt =
          smem_thr_copy_dQ.partition_D(sdQt); // ((Atom,AtomNum),PIPE_M,PIPE_N)
      cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQt);
    }
    __syncthreads();

    // Step 4: Copy dQ from smem to register to prepare for coalesced write to
    // gmem
    int const offset = seqlen_traits_q.cu_seq_len[bidb];
    Tensor mdQ =
        make_tensor(make_gmem_ptr(params.ptr_dQ), params.layout_dQ)(_, _, bidh);
    Tensor gdQ = local_tile(
        domain_offset(make_coord(offset, _0{}), mdQ),
        TileShape_MK{},
        make_coord(m_block, _0{})); // (M, K)
    GmemTiledCopy gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(thread_idx);
    Tensor tdQsdQ =
        gmem_thr_copy_dQ.partition_S(sdQ); // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);

    Tensor tdQrdQ = make_fragment_like(tdQsdQ);
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);

    // Step 5: Copy dQ from register to gmem
    // Construct identity layout for gdQ
    Tensor cdQ = cute::make_identity_tensor(
        TileShape_MK{}); // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tdQpdQ); ++k) {
      tdQpdQ(k) =
          get<1>(tdQcdQ(_0{}, _0{}, k)) < get<1>(params.layout_dQ.shape());
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<
        /*Is_even_MN=*/false,
        /*Is_even_K=*/false,
        /*Clear_OOB_MN=*/false,
        /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ,
        tdQrdQ,
        tdQgdQ,
        tdQcdQ,
        tdQpdQ,
        seqlen - m_block * kBlockM);
  }
};

} // namespace flash
