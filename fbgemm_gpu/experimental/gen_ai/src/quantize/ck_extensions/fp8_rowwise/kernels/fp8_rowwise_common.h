/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>

#ifdef HIPIFY_V2
#define getCurrentHIPStream getCurrentCUDAStream
#endif

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using D0DataType = float;
using D1DataType = float;
using DsDataType = ck::Tuple<D0DataType, D1DataType>;
using AccDataType = float;
using CShuffleDataType = float;

using ALayout = Row;
using BLayout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;

using ComputeType = ck::f8_t;

struct RowwiseScale {
  template <typename E, typename C, typename D0, typename D1>
  __host__ __device__ constexpr void
  operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

  template <>
  __host__ __device__ constexpr void
  operator()<ck::bhalf_t, float, float, float>(
      ck::bhalf_t& e,
      const float& c,
      const float& d0,
      const float& d1) const {
    const float x0_f = c * d0 * d1;

    e = ck::type_convert<ck::bhalf_t>(x0_f);
  }

  template <>
  __host__ __device__ constexpr void
  operator()<ck::half_t, float, float, float>(
      ck::half_t& e,
      const float& c,
      const float& d0,
      const float& d1) const {
    const float x0_f = c * d0 * d1;

    e = ck::type_convert<ck::half_t>(x0_f);
  }
};

using CDEElementOp = RowwiseScale;

template <
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int WAVE_TILE_M,
    int WAVE_TILE_N,
    int WAVE_MAP_M,
    int WAVE_MAP_N,
    typename ABLOCK_TRANSFER,
    typename BBLOCK_TRANSFER,
    typename CBLOCK_TRANSFER,
    typename CBLOCK_SPV,
    int CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
    int CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    ck::index_t AReadVecLength = 16,
    ck::index_t BReadVecLength = 16,
    ck::index_t ADstVecLength = 16,
    ck::index_t BDstVecLength = 16,
    int AK1 = 16,
    int BK1 = 16,
    ck::tensor_operation::device::GemmSpecialization GEMM_SPEC =
        ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    typename EDataType = ck::bhalf_t>
at::Tensor f8f8_rowwise_impl(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y,
    int KBatch = 1) {
  // Create GEMM definition.
  using DeviceGemmInstance =
      ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
          ALayout,
          BLayout,
          DsLayout,
          ELayout,
          ADataType,
          BDataType,
          DsDataType,
          EDataType,
          AccDataType,
          CShuffleDataType,
          AElementOp,
          BElementOp,
          CDEElementOp,
          GEMM_SPEC,
          BLOCK_SIZE, // Block Size
          MBLOCK, // M per Block
          NBLOCK, // N per Block
          KBLOCK, // K per Block
          AK1,
          BK1,
          WAVE_TILE_M, // M per Xdl
          WAVE_TILE_N, // N per Xdl
          WAVE_MAP_M, // Mxdl per Wave
          WAVE_MAP_N, // Nxdl per Wave
          ABLOCK_TRANSFER,
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          AReadVecLength,
          ADstVecLength,
          0,
          BBLOCK_TRANSFER,
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          BReadVecLength,
          BDstVecLength,
          0,
          CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
          CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
          CBLOCK_TRANSFER,
          CBLOCK_SPV,
          LOOP_SCHED,
          PIPELINE_VERSION,
          ComputeType>;

  // Get input information.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);

  int StrideA = K;
  int StrideB = K;
  int StrideE = N;

  // Create gemm launcher and arguments.
  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto cde_element_op = CDEElementOp{};

  constexpr ck::index_t NumDTensor = ck::Number<2>{};

  auto argument = gemm.MakeArgument(
      reinterpret_cast<ADataType*>(XQ.data_ptr()),
      reinterpret_cast<BDataType*>(WQ.data_ptr()),
      std::array<const void*, NumDTensor>{
          reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
          reinterpret_cast<D1DataType*>(x_scale.data_ptr())},
      reinterpret_cast<EDataType*>(Y.data_ptr()),
      M,
      N,
      K,
      StrideA,
      StrideB,
      std::array<ck::index_t, NumDTensor>{0, 0},
      StrideE,
      KBatch,
      a_element_op,
      b_element_op,
      cde_element_op);

  if (!gemm.IsSupportedArgument(argument)) {
    std::cerr << "Error: " << gemm.GetTypeString()
              << " does not support this problem {" << M << ", " << N << ", "
              << K << "}" << std::endl;
  }

  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(argument, StreamConfig{stream, false});

  return Y;
}

template <
    typename OutDType,
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int WAVE_TILE_M,
    int WAVE_TILE_N,
    int WAVE_MAP_M,
    int WAVE_MAP_N,
    typename ABLOCK_TRANSFER,
    typename BBLOCK_TRANSFER,
    typename CBLOCK_TRANSFER,
    typename CBLOCK_SPV,
    int CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
    int CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    ck::index_t AReadVecLength = 16,
    ck::index_t BReadVecLength = 16,
    ck::index_t ADstVecLength = 16,
    ck::index_t BDstVecLength = 16,
    int AK1 = 16,
    int BK1 = 16>
at::Tensor f8f8_rowwise_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y,
    int KBatch = 1) {
  // Check if this kernel needs K padding.
  int64_t K = WQ.size(1);
  bool k_padding = K % KBLOCK != 0;

  // Create proper dispatch around various kernel configurations.
  if (k_padding) {
    // kernel without preshuffle + padding
    return f8f8_rowwise_impl<
        BLOCK_SIZE,
        MBLOCK,
        NBLOCK,
        KBLOCK,
        WAVE_TILE_M,
        WAVE_TILE_N,
        WAVE_MAP_M,
        WAVE_MAP_N,
        ABLOCK_TRANSFER,
        BBLOCK_TRANSFER,
        CBLOCK_TRANSFER,
        CBLOCK_SPV,
        CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
        CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
        LOOP_SCHED,
        PIPELINE_VERSION,
        AReadVecLength,
        BReadVecLength,
        ADstVecLength,
        BDstVecLength,
        AK1,
        BK1,
        ck::tensor_operation::device::GemmSpecialization::KPadding,
        OutDType>(XQ, WQ, x_scale, w_scale, Y, KBatch);

  } else {
    // kernel without preshuffle + no padding
    return f8f8_rowwise_impl<
        BLOCK_SIZE,
        MBLOCK,
        NBLOCK,
        KBLOCK,
        WAVE_TILE_M,
        WAVE_TILE_N,
        WAVE_MAP_M,
        WAVE_MAP_N,
        ABLOCK_TRANSFER,
        BBLOCK_TRANSFER,
        CBLOCK_TRANSFER,
        CBLOCK_SPV,
        CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
        CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
        LOOP_SCHED,
        PIPELINE_VERSION,
        AReadVecLength,
        BReadVecLength,
        ADstVecLength,
        BDstVecLength,
        AK1,
        BK1,
        ck::tensor_operation::device::GemmSpecialization::Default,
        OutDType>(XQ, WQ, x_scale, w_scale, Y, KBatch);
  }
}

template <
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int WAVE_TILE_M,
    int WAVE_TILE_N,
    int WAVE_MAP_M,
    int WAVE_MAP_N,
    typename ABLOCK_TRANSFER,
    typename BBLOCK_TRANSFER,
    typename CBLOCK_TRANSFER,
    typename CBLOCK_SPV,
    int CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
    int CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    ck::index_t AReadVecLength = 16,
    ck::index_t BReadVecLength = 16,
    ck::index_t ADstVecLength = 16,
    ck::index_t BDstVecLength = 16,
    int AK1 = 16,
    int BK1 = 16>
at::Tensor f8f8f16_rowwise_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y,
    int KBatch = 1) {
  return f8f8_rowwise_wrapper<
      ck::half_t,
      BLOCK_SIZE,
      MBLOCK,
      NBLOCK,
      KBLOCK,
      WAVE_TILE_M,
      WAVE_TILE_N,
      WAVE_MAP_M,
      WAVE_MAP_N,
      ABLOCK_TRANSFER,
      BBLOCK_TRANSFER,
      CBLOCK_TRANSFER,
      CBLOCK_SPV,
      CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
      CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
      LOOP_SCHED,
      PIPELINE_VERSION,
      AReadVecLength,
      BReadVecLength,
      ADstVecLength,
      BDstVecLength,
      AK1,
      BK1>(XQ, WQ, x_scale, w_scale, Y, KBatch);
}

template <
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int WAVE_TILE_M,
    int WAVE_TILE_N,
    int WAVE_MAP_M,
    int WAVE_MAP_N,
    typename ABLOCK_TRANSFER,
    typename BBLOCK_TRANSFER,
    typename CBLOCK_TRANSFER,
    typename CBLOCK_SPV,
    int CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
    int CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    ck::index_t AReadVecLength = 16,
    ck::index_t BReadVecLength = 16,
    ck::index_t ADstVecLength = 16,
    ck::index_t BDstVecLength = 16,
    int AK1 = 16,
    int BK1 = 16>
at::Tensor f8f8bf16_rowwise_wrapper(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y,
    int KBatch = 1) {
  return f8f8_rowwise_wrapper<
      ck::bhalf_t,
      BLOCK_SIZE,
      MBLOCK,
      NBLOCK,
      KBLOCK,
      WAVE_TILE_M,
      WAVE_TILE_N,
      WAVE_MAP_M,
      WAVE_MAP_N,
      ABLOCK_TRANSFER,
      BBLOCK_TRANSFER,
      CBLOCK_TRANSFER,
      CBLOCK_SPV,
      CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
      CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
      LOOP_SCHED,
      PIPELINE_VERSION,
      AReadVecLength,
      BReadVecLength,
      ADstVecLength,
      BDstVecLength,
      AK1,
      BK1>(XQ, WQ, x_scale, w_scale, Y, KBatch);
}
