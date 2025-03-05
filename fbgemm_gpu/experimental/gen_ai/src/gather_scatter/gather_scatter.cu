/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef USE_ROCM

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor.hpp"
#include "cute/tensor_predicate.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"

namespace fbgemm_gpu {

namespace {

constexpr int kTmaGmemAlignment = 16;
constexpr int kL = 256;

template <int kBlkNOrM, class DataType, class IndexType, class SmemLayout>
struct SharedStorage {
  static constexpr int kPipeMax = cute::size<0>(SmemLayout{});
  static constexpr int kTmaAlignment = 128;
  static constexpr int kMbarAlignemnt = 8;

  cute::array_aligned<IndexType, kBlkNOrM> index;
  cute::array_aligned<DataType, cute::cosize_v<SmemLayout>, kTmaAlignment> data;

  CUTE_ALIGNAS(kMbarAlignemnt) uint64_t tma_load_barrier[kPipeMax];
};

template <
    bool IsGather,
    class ProblemShape,
    class TileShape,
    class DataType,
    class IndexType,
    class SmemLayout,
    class TmaLoad,
    class TmaStore>
__global__ static void gather_or_scatter_along_first_dim_kernel(
    ProblemShape problem_shape,
    TileShape tile_shape,
    CUTLASS_GRID_CONSTANT TmaLoad const tma_load_input,
    const IndexType* index,
    CUTLASS_GRID_CONSTANT TmaStore const tma_store_output) {
  // Input shape: A [M, K]
  // Output shape: B [N, K]
  int M = cute::get<0>(problem_shape);
  int N = cute::get<1>(problem_shape);
  int K = cute::get<2>(problem_shape);
  int L = cute::get<3>(problem_shape);

  static_assert(cute::is_static<TileShape>::value);
  constexpr int kBlkNOrM = cute::size<0>(tile_shape);
  constexpr int kBlkK = cute::size<1>(tile_shape);
  constexpr int kBlkL = cute::size<2>(tile_shape);

  using SmemT = SharedStorage<kBlkNOrM, DataType, IndexType, SmemLayout>;
  constexpr int kPipeMax = SmemT::kPipeMax;

  extern __shared__ char smem_raw[];
  SmemT& smem = *reinterpret_cast<SmemT*>(smem_raw);

  int indexing_dim = IsGather ? N : M;
  const int n_or_m_offset = blockIdx.x * kBlkNOrM;
  if (n_or_m_offset >= indexing_dim) {
    return;
  }

  // Straight-forward direct global read of indices.
  if (threadIdx.x < kBlkNOrM && n_or_m_offset + threadIdx.x < indexing_dim) {
    smem.index[threadIdx.x] = index[n_or_m_offset + threadIdx.x];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // Tensors on HBM.
    cute::Tensor gA = tma_load_input.get_tma_tensor(cute::make_shape(M, K, L));
    cute::Tensor gB =
        tma_store_output.get_tma_tensor(cute::make_shape(N, K, L));
    // Tensors on SMEM.
    cute::Tensor sA = cute::make_tensor(
        cute::make_smem_ptr(smem.data.data()), cute::group<0, 2>(SmemLayout{}));

    constexpr int kTmaTransactionBytes = kBlkK * kBlkL * sizeof(DataType);
    const int kNumKTiles = ((K + kBlkK - 1) / kBlkK);
    const int kNumNOrMKTiles =
        std::min(kBlkNOrM, indexing_dim - n_or_m_offset) * kNumKTiles;
    const int kNumIterations = kNumNOrMKTiles + kPipeMax - 1;

    for (int iteration = 0; iteration < kNumIterations; ++iteration) {
      // Load.
      if (iteration < kNumNOrMKTiles) {
        int load_pipe = iteration % kPipeMax;

        int m, n, k;
        if constexpr (IsGather) {
          n = iteration / kNumKTiles;
          k = iteration % kNumKTiles;
          m = smem.index[n];
        } else {
          m = iteration / kNumKTiles + n_or_m_offset;
          k = iteration % kNumKTiles;
          // n is not needed here
        }

        cute::tma_store_wait<kPipeMax - 1>();

        cute::Tensor tAgA = cute::local_tile(
            gA,
            cute::Tile<cute::_1, cute::Int<kBlkK>, cute::Int<kBlkL>>{},
            cute::make_coord(m, k, 0));
        cute::Tensor tAsA = cute::local_tile(
            sA,
            cute::Tile<cute::_1, cute::Int<kBlkK>, cute::Int<kBlkL>>{},
            cute::make_coord(load_pipe, 0, 0));

        auto& tma_load_mbar = smem.tma_load_barrier[load_pipe];
        cute::initialize_barrier(smem.tma_load_barrier[load_pipe], 1);
        cute::set_barrier_transaction_bytes(
            tma_load_mbar, kTmaTransactionBytes);

        auto tma_load_per_cta = tma_load_input.get_slice(0);
        cute::copy(
            tma_load_input.with(tma_load_mbar),
            tma_load_per_cta.partition_S(tAgA),
            tma_load_per_cta.partition_D(tAsA));
      }

      // Store
      if (iteration >= kPipeMax - 1) {
        int processing_index = iteration - kPipeMax + 1;
        int store_pipe = processing_index % kPipeMax;

        int m, n, k;
        if constexpr (IsGather) {
          n = processing_index / kNumKTiles + n_or_m_offset;
          k = processing_index % kNumKTiles;
          // m is not needed here
        } else {
          m = processing_index / kNumKTiles;
          k = processing_index % kNumKTiles;
          n = smem.index[m];
        }

        cute::wait_barrier(smem.tma_load_barrier[store_pipe], 0);

        cute::Tensor tAgB = cute::local_tile(
            gB,
            cute::Tile<cute::_1, cute::Int<kBlkK>, cute::Int<kBlkL>>{},
            cute::make_coord(n, k, 0));
        cute::Tensor tAsA = cute::local_tile(
            sA,
            cute::Tile<cute::_1, cute::Int<kBlkK>, cute::Int<kBlkL>>{},
            cute::make_coord(store_pipe, 0, 0));

        auto tma_store_per_cta = tma_store_output.get_slice(0);
        cute::copy(
            tma_store_output,
            tma_store_per_cta.partition_S(tAsA),
            tma_store_per_cta.partition_D(tAgB));
        cute::tma_store_arrive();
      }
    }
  }
  cute::tma_store_wait<0>();
}

template <class T>
struct TorchDTypeTrait {};

template <>
struct TorchDTypeTrait<cutlass::bfloat16_t> {
  static auto dtype() {
    return at::kBFloat16;
  };
};

template <>
struct TorchDTypeTrait<int32_t> {
  static auto dtype() {
    return at::kInt;
  };
};

template <>
struct TorchDTypeTrait<int64_t> {
  static auto dtype() {
    return at::kLong;
  };
};

template <
    bool IsGather,
    class DataType,
    class IndexType,
    class TMAStoreInst = cute::SM90_TMA_STORE>
void gather_or_scatter_along_first_dim(
    at::Tensor src,
    at::Tensor index,
    at::Tensor dst) {
  assert(src.dtype() == TorchDTypeTrait<DataType>::dtype());
  assert(dst.dtype() == TorchDTypeTrait<DataType>::dtype());
  assert(index.dtype() == TorchDTypeTrait<IndexType>::dtype());

  constexpr int L = kL;
  const int M = src.size(0);
  const int N = dst.size(0);
  const int K = src.size(1) / L;

  auto src_gmem_layout = cute::make_layout(
      cute::make_shape(M, K, L), cute::make_stride(K * L, L, 1));
  auto src_gmem_tensor = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<DataType*>(src.data_ptr())),
      src_gmem_layout);

  auto dst_gmem_layout = cute::make_layout(
      cute::make_shape(N, K, L), cute::make_stride(K * L, L, 1));
  auto dst_gmem_tensor = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<DataType*>(dst.data_ptr())),
      dst_gmem_layout);

  constexpr int kBlkNOrM = 1;
  constexpr int kBlkK = 4;
  constexpr int kBlkL = L;
  constexpr int kPipeMax = 3;

  auto smem_layout = cute::make_layout(
      cute::make_shape(
          cute::Int<kPipeMax>{},
          cute::_1{},
          cute::Int<kBlkK>{},
          cute::Int<kBlkL>{}),
      cute::make_stride(
          cute::Int<kBlkK * kBlkL>{},
          cute::Int<kBlkK * kBlkL>{},
          cute::Int<kBlkL>{},
          cute::_1{}));
  auto tma_load = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{},
      src_gmem_tensor,
      smem_layout(0, cute::_, cute::_, cute::_));
  auto tma_store = cute::make_tma_copy(
      TMAStoreInst{},
      dst_gmem_tensor,
      smem_layout(0, cute::_, cute::_, cute::_));

  auto problem_shape = cute::make_shape(M, N, K, L);
  auto tile_shape = cute::make_shape(
      cute::Int<kBlkNOrM>{}, cute::Int<kBlkK>{}, cute::Int<kBlkL>{});

  using SmemT =
      SharedStorage<kBlkNOrM, DataType, IndexType, decltype(smem_layout)>;

  int num_ctas = ((IsGather ? N : M) + kBlkNOrM - 1) / kBlkNOrM;
  dim3 grid_dims(num_ctas, 1, 1);
  dim3 block_dims(32, 1, 1);
  dim3 cluster_dims(1, 1, 1);
  int smem_size = sizeof(SmemT);
  auto stream = c10::cuda::getCurrentCUDAStream();

  cutlass::ClusterLaunchParams launch_params{
      grid_dims, block_dims, cluster_dims, smem_size, stream};
  void* kernel = (void*)gather_or_scatter_along_first_dim_kernel<
      IsGather,
      decltype(problem_shape),
      decltype(tile_shape),
      DataType,
      IndexType,
      decltype(smem_layout),
      decltype(tma_load),
      decltype(tma_store)>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      launch_params,
      kernel,
      problem_shape,
      tile_shape,
      tma_load,
      reinterpret_cast<IndexType*>(index.data_ptr()),
      tma_store);

  if (status != cutlass::Status::kSuccess) {
    cudaError_t error = cudaGetLastError();
    CUTE_ERROR_EXIT(error);
  }
}

} // namespace

at::Tensor gather_along_first_dim(at::Tensor data, at::Tensor index) {
  if (data.is_contiguous() && data.dim() == 2 && index.is_contiguous() &&
      index.dim() == 1) {
    using T = cutlass::bfloat16_t;

    const int M = data.size(0);
    const int K = data.size(1);
    const int N = index.size(0);
    // TODO(shikaili): Make it supports more configurations.
    if (data.dtype() == at::kBFloat16 &&
        (K * sizeof(T) % kTmaGmemAlignment == 0) && (K % kL == 0)) {
      at::Tensor output = at::empty(
          {N, K},
          at::TensorOptions().dtype(at::kBFloat16).device(data.device()));
      if (index.dtype() == at::kInt) {
        gather_or_scatter_along_first_dim<
            true,
            T,
            int32_t,
            cute::SM90_TMA_STORE>(data, index, output);
        return output;
      } else if (index.dtype() == at::kLong) {
        gather_or_scatter_along_first_dim<
            true,
            T,
            int64_t,
            cute::SM90_TMA_STORE>(data, index, output);
        return output;
      }
    }
  }
  return at::index_select(data, 0, index);
}

void scatter_add_along_first_dim(
    at::Tensor dst,
    at::Tensor src,
    at::Tensor index) {
  if (dst.is_contiguous() && dst.dim() == 2 && src.is_contiguous() &&
      src.dim() == 2 && index.is_contiguous() && index.dim() == 1) {
    using T = cutlass::bfloat16_t;

    const int M = src.size(0);
    const int K = src.size(1);
    const int N = index.size(0);
    assert(dst.size(1) == K);
    // TODO(shikaili): Make it supports more configurations.
    if (dst.dtype() == at::kBFloat16 && src.dtype() == at::kBFloat16 &&
        (K * sizeof(T) % kTmaGmemAlignment == 0) && (K % kL == 0)) {
      if (index.dtype() == at::kInt) {
        gather_or_scatter_along_first_dim<
            false,
            T,
            int32_t,
            cute::SM90_TMA_REDUCE_ADD>(src, index, dst);
        return;
      } else if (index.dtype() == at::kLong) {
        gather_or_scatter_along_first_dim<
            false,
            T,
            int64_t,
            cute::SM90_TMA_REDUCE_ADD>(src, index, dst);
        return;
      }
    }
  }

  const int K = src.size(1);
  dst.scatter_add_(0, index.to(at::kLong).unsqueeze(1).expand({-1, K}), src);
}

} // namespace fbgemm_gpu

#endif
