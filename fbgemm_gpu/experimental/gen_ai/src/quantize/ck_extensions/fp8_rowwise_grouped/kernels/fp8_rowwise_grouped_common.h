/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#ifdef USE_ROCM
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp"

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;

using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using D0DataType = float;
using D1DataType = float;
using DsDataType = ck::Tuple<D0DataType, D1DataType>;
using EDataType = ck::bhalf_t;
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
using CDEElementOp = MultiplyMultiply;

using ComputeType = ck::f8_t;

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
    ck::tensor_operation::device::GemmSpecialization GEMM_SPEC =
        ck::tensor_operation::device::GemmSpecialization::MNPadding>
using DeviceGemmHelper =
    ck::tensor_operation::device::DeviceGroupedGemmMultipleDXdlCShuffleTileLoop<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementOp,
        BElementOp,
        CDEElementOp,
        GEMM_SPEC,
        1, // NumGemmK
        BLOCK_SIZE, // Block Size
        MBLOCK, // M per Block
        NBLOCK, // N per Block
        KBLOCK, // K per Block
        16, // AK1
        16, // BK1
        WAVE_TILE_M, // M per Xdl
        WAVE_TILE_N, // N per Xdl
        WAVE_MAP_M, // Mxdl per Wave
        WAVE_MAP_N, // Nxdl per Wave
        ABLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        16,
        16,
        0,
        BBLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        16,
        16,
        0,
        CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
        CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
        CBLOCK_TRANSFER,
        CBLOCK_SPV,
        LOOP_SCHED,
        PIPELINE_VERSION,
        ComputeType>;

// Templated kernel launch to accommodate different input and output types.
template <typename DeviceGemmInstance, typename InputType, typename OutputType>
OutputType f8f8bf16_rowwise_grouped_impl(
    InputType XQ,
    InputType WQ,
    InputType x_scale,
    InputType w_scale,
    at::Tensor kernel_args,
    OutputType Y) {
  // Get input information.
  int group_count;
  if constexpr (std::is_same_v<InputType, at::Tensor>) {
    group_count = WQ.size(0);
  } else {
    group_count = XQ.size();
  }
  using KernelArguments =
      ck::tensor_operation::device::GroupedGemmKernelArgument<2>;
  using GemmDesc = ck::tensor_operation::device::GemmDesc;
  // Create gemm shape containers.
  std::vector<GemmDesc> gemm_descs;
  // Create container for input arguments.
  std::vector<const void*> A_args;
  std::vector<const void*> B_args;
  std::vector<void*> C_args;
  std::vector<std::array<const void*, 2>> D_args = {};
  // Reserve space in argument arrays.
  gemm_descs.reserve(group_count);
  A_args.reserve(group_count);
  B_args.reserve(group_count);
  C_args.reserve(group_count);
  D_args.reserve(group_count);
  int M;
  int K;
  int N;
  // Declare pointers to input and output buffers.
  ADataType* a_ptr;
  BDataType* b_ptr;
  EDataType* c_ptr;
  D0DataType* d0_ptr;
  D1DataType* d1_ptr;
  // Populate arguments.
  for (int i = 0; i < group_count; i++) {
    // Compute appropriate data pointers.
    // Set the shape arguments for this gemm.
    if constexpr (std::is_same_v<InputType, at::Tensor>) {
      M = XQ.size(XQ.dim() - 2);
      N = WQ.size(1);
      K = WQ.size(2);
      // These pointers dont seem to actually be used since the kernel arguments
      // contains the correct version. For simplicity, we just point to the
      // start of the tensor.
      a_ptr = reinterpret_cast<ADataType*>(XQ.data_ptr());
      b_ptr = reinterpret_cast<BDataType*>(WQ.data_ptr());
      d0_ptr = reinterpret_cast<D0DataType*>(w_scale.data_ptr());
      d1_ptr = reinterpret_cast<D1DataType*>(x_scale.data_ptr());
    } else {
      M = XQ[i].size(0);
      N = WQ[i].size(0);
      K = XQ[i].size(1);
      a_ptr = reinterpret_cast<ADataType*>(XQ[i].data_ptr());
      b_ptr = reinterpret_cast<BDataType*>(WQ[i].data_ptr());
      d0_ptr = reinterpret_cast<D0DataType*>(w_scale[i].data_ptr());
      d1_ptr = reinterpret_cast<D1DataType*>(x_scale[i].data_ptr());
    }
    if constexpr (std::is_same_v<OutputType, at::Tensor>) {
      c_ptr = reinterpret_cast<EDataType*>(Y.data_ptr()) + (i * M * N);
    } else {
      c_ptr = reinterpret_cast<EDataType*>(Y[i].data_ptr());
    }

    GemmDesc gemm_desc = {M, N, K, K, K, N, {0, 0}};
    gemm_descs.push_back(gemm_desc);

    // Set pointers to inputs and outputs.
    A_args.push_back(a_ptr);
    B_args.push_back(b_ptr);
    C_args.push_back(c_ptr);
    D_args.emplace_back(std::array<const void*, 2>{d0_ptr, d1_ptr});
  }

  // Create gemm launcher and arguments.
  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto cde_element_op = CDEElementOp{};
  // Setup Gemm arguments.
  auto argument = gemm.MakeArgument(
      A_args,
      B_args,
      D_args,
      C_args,
      gemm_descs,
      a_element_op,
      b_element_op,
      cde_element_op);

  // Set gemm kernel arguments.
  gemm.SetDeviceKernelArgs(argument, kernel_args.data_ptr());

  // Get hip graph stream if it exists.
  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(argument, StreamConfig{stream, false});

  return Y;
}
