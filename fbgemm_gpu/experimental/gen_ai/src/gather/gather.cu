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

#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"

namespace fbgemm_gpu {

namespace {

template <int kBlkN, class DataType, class SmemLayout>
struct SharedStorage {
  static constexpr int kPipeMax = cute::size<0>(SmemLayout{});
  static constexpr int kTmaAlignment = 128;
  static constexpr int kMbarAlignemnt = 8;

  cute::array_aligned<int32_t, kBlkN> index;
  cute::array_aligned<DataType, cute::cosize_v<SmemLayout>, kTmaAlignment> data;

  CUTE_ALIGNAS(kMbarAlignemnt) uint64_t tma_load_barrier[kPipeMax];
};

template <
    class ProblemShape,
    class TileShape,
    class DataType,
    class SmemLayout,
    class TmaLoad,
    class TmaStore>
__global__ static void gather_along_first_dim_kernel(
    ProblemShape problem_shape,
    TileShape tile_shape,
    CUTLASS_GRID_CONSTANT TmaLoad const tma_load_input,
    const int32_t* index,
    CUTLASS_GRID_CONSTANT TmaStore const tma_store_output) {
  // Input shape: A [M, K]
  // Output shape: B [N, K]
  int M = cute::get<0>(problem_shape);
  int N = cute::get<1>(problem_shape);
  int K = cute::get<2>(problem_shape);

  static_assert(cute::is_static<TileShape>::value);
  constexpr int kBlkN = cute::size<0>(tile_shape);
  constexpr int kBlkK = cute::size<1>(tile_shape);

  using SmemT = SharedStorage<kBlkN, DataType, SmemLayout>;
  constexpr int kPipeMax = SmemT::kPipeMax;

  extern __shared__ char smem_raw[];
  SmemT& smem = *reinterpret_cast<SmemT*>(smem_raw);

  const int n_offset = blockIdx.x * kBlkN;
  if (n_offset >= N) {
    return;
  }

  // Straight-forward direct global read of indices.
  if (threadIdx.x < kBlkN && n_offset + threadIdx.x < N) {
    smem.index[threadIdx.x] = index[n_offset + threadIdx.x];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // Tensors on HBM.
    cute::Tensor gA = tma_load_input.get_tma_tensor(cute::make_shape(M, K));
    cute::Tensor gB = tma_store_output.get_tma_tensor(cute::make_shape(N, K));
    // Tensors on SMEM.
    cute::Tensor sA = cute::make_tensor(
        cute::make_smem_ptr(smem.data.data()), cute::group<0, 2>(SmemLayout{}));

    constexpr int kTmaTransactionBytes = kBlkK * sizeof(DataType);
    const int kNumKTiles = ((K + kBlkK - 1) / kBlkK);
    const int kNumNKTiles = kBlkN * kNumKTiles;
    const int kNumIterations = kNumNKTiles + kPipeMax - 1;

    for (int iteration = 0; iteration < kNumIterations; ++iteration) {
      // Load.
      if (iteration < kNumNKTiles) {
        int load_pipe = iteration % kPipeMax;

        int n = iteration / kNumKTiles;
        int k = iteration % kNumKTiles;
        int m = smem.index[n];

        cute::tma_store_wait<kPipeMax - 1>();

        cute::Tensor tAgA = cute::local_tile(
            gA,
            cute::Tile<cute::_1, cute::Int<kBlkK>>{},
            cute::make_coord(m, k));
        cute::Tensor tAsA = cute::local_tile(
            sA,
            cute::Tile<cute::_1, cute::Int<kBlkK>>{},
            cute::make_coord(load_pipe, 0));

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

        int n = processing_index / kNumKTiles;
        int k = processing_index % kNumKTiles;

        cute::wait_barrier(smem.tma_load_barrier[store_pipe], 0);

        cute::Tensor tAgB = cute::local_tile(
            gB,
            cute::Tile<cute::_1, cute::Int<kBlkK>>{},
            cute::make_coord(n + n_offset, k));
        cute::Tensor tAsA = cute::local_tile(
            sA,
            cute::Tile<cute::_1, cute::Int<kBlkK>>{},
            cute::make_coord(store_pipe, 0));

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

} // namespace

// TODO(shikaili): Templatize it and make it supports more configurations.
at::Tensor gather_along_first_dim(at::Tensor data, at::Tensor index) {
  using DataType = cutlass::bfloat16_t;
  constexpr auto kDataTypeEnum = at::kBFloat16;
  using IndexType = int32_t;
  constexpr auto kIndexTypeEnum = at::kInt;
  constexpr int kTmaGmemAlignment = 16;

  bool compatible = (data.dtype() == kDataTypeEnum && data.is_contiguous() &&
                     data.dim() == 2) &&
      (index.dtype() == kIndexTypeEnum && index.is_contiguous() &&
       index.dim() == 1) &&
      (data.size(1) * sizeof(DataType) % kTmaGmemAlignment == 0);

  if (!compatible) {
    return at::index_select(data, 0, index);
  }

  const int M = data.size(0);
  const int K = data.size(1);
  const int N = index.size(0);

  auto src_gmem_layout =
      cute::make_layout(cute::make_shape(M, K), cute::make_stride(K, 1));
  auto src_gmem_tensor = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<DataType*>(data.data_ptr())),
      src_gmem_layout);

  at::Tensor output = at::empty(
      {N, K}, at::TensorOptions().dtype(at::kBFloat16).device(data.device()));
  auto dst_gmem_layout =
      cute::make_layout(cute::make_shape(N, K), cute::make_stride(K, 1));
  auto dst_gmem_tensor = cute::make_tensor(
      cute::make_gmem_ptr(reinterpret_cast<DataType*>(output.data_ptr())),
      dst_gmem_layout);

  constexpr int kBlkN = 1;
  constexpr int kBlkK = 256;
  constexpr int kPipeMax = 4;

  auto smem_layout = cute::make_layout(
      cute::make_shape(cute::Int<kPipeMax>{}, cute::_1{}, cute::Int<kBlkK>{}),
      cute::make_stride(cute::Int<kBlkK>{}, cute::Int<kBlkK>{}, cute::_1{}));
  auto tma_load = cute::make_tma_copy(
      cute::SM90_TMA_LOAD{}, src_gmem_tensor, smem_layout(0, cute::_, cute::_));
  auto tma_store = cute::make_tma_copy(
      cute::SM90_TMA_STORE{},
      dst_gmem_tensor,
      smem_layout(0, cute::_, cute::_));

  auto problem_shape = cute::make_shape(M, N, K);
  auto tile_shape = cute::make_shape(cute::Int<kBlkN>{}, cute::Int<kBlkK>{});

  using SmemT = SharedStorage<kBlkN, DataType, decltype(smem_layout)>;

  int num_ctas = (N + kBlkN - 1) / kBlkN;
  dim3 grid_dims(num_ctas, 1, 1);
  dim3 block_dims(32, 1, 1);
  dim3 cluster_dims(1, 1, 1);
  int smem_size = sizeof(SmemT);
  auto stream = c10::cuda::getCurrentCUDAStream();

  cutlass::ClusterLaunchParams launch_params{
      grid_dims, block_dims, cluster_dims, smem_size, stream};
  void* kernel = (void*)gather_along_first_dim_kernel<
      decltype(problem_shape),
      decltype(tile_shape),
      DataType,
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

  return output;
}

} // namespace fbgemm_gpu

#endif
