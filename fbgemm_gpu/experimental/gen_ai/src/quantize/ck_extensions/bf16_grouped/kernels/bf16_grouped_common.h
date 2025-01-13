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
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp"

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType = ck::bhalf_t;
using BDataType = ck::bhalf_t;
using DsDataType = ck::Tuple<>;
using CDataType = ck::bhalf_t;
using AccDataType = float;
using CShuffleDataType = float;

using ALayout = Row;
using BLayout = Col;
using DsLayout = ck::Tuple<>;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CDEElementOp = PassThrough;

using ComputeType = ck::bhalf_t;

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
        CLayout,
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        CDataType,
        AElementOp,
        BElementOp,
        CDEElementOp,
        GEMM_SPEC,
        1, // NumGemmK
        BLOCK_SIZE, // Block Size
        MBLOCK, // M per Block
        NBLOCK, // N per Block
        KBLOCK, // K per Block
        8, // AK1
        8, // BK1
        WAVE_TILE_M, // M per Xdl
        WAVE_TILE_N, // N per Xdl
        WAVE_MAP_M, // Mxdl per Wave
        WAVE_MAP_N, // Nxdl per Wave
        ABLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        0,
        BBLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        0,
        CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
        CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
        CBLOCK_TRANSFER,
        CBLOCK_SPV,
        LOOP_SCHED,
        PIPELINE_VERSION,
        ComputeType>;

template <typename DeviceGemmInstance>
std::vector<at::Tensor> bf16_grouped_impl(
    at::TensorList A,
    at::TensorList B,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y) {
  // Get input information.
  int group_count = A.size();
  using KernelArguments =
      ck::tensor_operation::device::GroupedGemmKernelArgument<0>;
  using GemmDesc = ck::tensor_operation::device::GemmDesc;
  // Create gemm shape containers.
  std::vector<GemmDesc> gemm_descs;
  // Create container for input arguments.
  std::vector<const void*> A_args;
  std::vector<const void*> B_args;
  std::vector<void*> C_args;
  std::vector<std::array<const void*, 0>> D_args = {};
  // Reserve space in argument arrays.
  gemm_descs.reserve(group_count);
  A_args.reserve(group_count);
  B_args.reserve(group_count);
  C_args.reserve(group_count);
  // Populate arguments.
  for (int i = 0; i < group_count; i++) {
    // Set the shape arguments for this gemm.
    int M = A[i].size(0);
    int K = A[i].size(1);
    int N = B[i].size(0);
    GemmDesc gemm_desc = {M, N, K, K, K, N, {}};
    gemm_descs.push_back(gemm_desc);
    // Set pointers to inputs and outputs.
    A_args.push_back(reinterpret_cast<ADataType*>(A[i].data_ptr()));
    B_args.push_back(reinterpret_cast<BDataType*>(B[i].data_ptr()));
    C_args.push_back(reinterpret_cast<CDataType*>(Y[i].data_ptr()));
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
