/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

struct Scale {
  Scale(float scale) : scale_(scale){};

  __host__ __device__ void operator()(ck::bhalf_t& e, const float& c) const {
    e = ck::type_convert<ck::bhalf_t>(scale_ * c);
  }

  float scale_;
};

namespace fbgemm_gpu {

template <
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int MPER_WAVE,
    int NPER_WAVE,
    bool PADDING = false>
at::Tensor
f8f8bf16_tensorwise_impl(at::Tensor XQ, at::Tensor WQ, double scale) {
  // Get input information.
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);

  int StrideA = K;
  int StrideB = K;
  int StrideC = N;

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));

  using ADataType = ck::f8_t;
  using BDataType = ck::f8_t;
  using CDataType = ck::bhalf_t;
  using AccDataType = float;
  using CShuffleDataType = float;

  using ALayout = Row;
  using BLayout = Col;
  using CLayout = Row;

  using AElementOp = PassThrough;
  using BElementOp = PassThrough;
  using CElementOp = Scale;

  static constexpr auto GemmDefault =
      ck::tensor_operation::device::GemmSpecialization::Default;
  static constexpr auto GemmMNKPadding =
      ck::tensor_operation::device::GemmSpecialization::MNKPadding;
  static constexpr auto GemmSpec = PADDING ? GemmMNKPadding : GemmDefault;
  static constexpr auto LoopSched = ck::make_default_loop_scheduler();
  static constexpr auto PipelineVer = ck::PipelineVersion::v1;
  using ComputeType = ck::f8_t;

  // Define derivative constants based on template parameters.
  static constexpr int BLOCK_CLUSTER = BLOCK_SIZE / 4;
  static constexpr int CBLOCK_N = NBLOCK / 16;
  static constexpr int CBLOCK_M = BLOCK_SIZE / CBLOCK_N;
  // Create the GEMM Operator.
  using DeviceGemmInstance =
      ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<
          ALayout,
          BLayout,
          ck::Tuple<>, // D Layouts
          CLayout, // Output Layouts
          ADataType,
          BDataType,
          AccDataType,
          CShuffleDataType,
          ck::Tuple<>, // D Datatypes
          CDataType, // Output datatype
          AElementOp,
          BElementOp,
          CElementOp,
          GemmSpec, // Kernel Schedule.
          1, // Prefetch stage
          BLOCK_SIZE, // Block size
          MBLOCK, // M per block
          NBLOCK, // N per block
          KBLOCK, // K per block
          16, // AK1
          16, // BK1
          32, // M Per Xdl
          32, // N Per Xdl
          MPER_WAVE, // Mxdl per wave
          NPER_WAVE, // Nxdl per wave
          S<4, BLOCK_CLUSTER, 1>, // ABlockTransfer Threadcluster K0_M_K1
          S<1, 0, 2>, // ABlockTransfer ThreadCluster ArrangeOrder
          S<1, 0, 2>, // ABlockTransfer SrcAccessOrder
          2, // ABlockTransfer SrcVectorDim
          16, // ABlockTransfer SrcScalar PerVector
          16, // ABlockTransfer DstScalar PerVector_K1
          1, // ABlockLds AddExtraM
          S<4, BLOCK_CLUSTER, 1>, // BBlockTransfer ThreadCluster K0_N_K1
          S<1, 0, 2>, // BBlockTransfer ThreadCluster ArrangeOrder
          S<1, 0, 2>, // BBlockTransfer SrcAccess Order
          2, // BBlockTransfer SrcVectorDim
          8, // BBlockTransfer SrcScalarPerVector
          8, // BBlockTransfer DstScalar PerVector_K1
          1, // BBlockLds AddExtraN
          1, // CShuffle MXdlPerWave PerShuffle
          1, // CShuffle NXdlPerWave PerShuffle
          S<1, CBLOCK_M, 1, CBLOCK_N>, // CBlockTransferClusterLengths
          8, // CBlockTransfer ScalarPerVector
          LoopSched, // Loop Scheduler
          PipelineVer, // Pipeline version
          ComputeType>; // Compute datatype

  // Create gemm launcher and arguments.
  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto c_element_op = CElementOp{float(scale)};

  auto argument = gemm.MakeArgument(
      reinterpret_cast<ADataType*>(XQ.data_ptr()),
      reinterpret_cast<BDataType*>(WQ.data_ptr()),
      std::array<const void*, 0>{},
      reinterpret_cast<CDataType*>(Y.data_ptr()),
      M,
      N,
      K,
      StrideA,
      StrideB,
      std::array<ck::index_t, 0>{},
      StrideC,
      a_element_op,
      b_element_op,
      c_element_op);

  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(argument, StreamConfig{stream, false});

  return Y;
}

enum class KernelMode { Small, Large, Default };

std::tuple<KernelMode, bool> get_kernel_mode(at::Tensor XQ, at::Tensor WQ) {
  auto M = XQ.size(0);
  auto K = XQ.size(1);
  auto N = WQ.size(0);
  // Use small kernel when input matrices are small.
  bool use_small_kernel = (M <= 512 && N <= 512) || (M <= 128) || (N <= 128);
  // For larger workloads, specialize to large gemm.
  bool use_large_kernel =
      ((M >= 4096 && N >= 4096) || (M >= 8192 && N >= 2048) ||
       (N >= 8192 && M >= 2048) || (K >= 8192 && M >= 2048 && N >= 2048));
  // Must use specialized padding kernel for shapes that cant be tiled.
  bool use_pad = ((M % 256 != 0) || (N % 256 != 0) || (K % 256 != 0));
  if (use_small_kernel) {
    return {KernelMode::Small, use_pad};
  } else if (use_large_kernel) {
    return {KernelMode::Large, use_pad};
  } else {
    return {KernelMode::Default, use_pad};
  }
}

at::Tensor f8f8bf16_tensorwise(
    at::Tensor XQ,
    at::Tensor WQ,
    double scale,
    bool use_fast_accum) {
  // Check that input types are compatible with AMD FP8.
  TORCH_CHECK(
      (XQ.dtype() == at::kFloat8_e4m3fnuz) &&
          (WQ.dtype() == at::kFloat8_e4m3fnuz),
      "Inputs must be type float8_e4m3fnuz.");
  TORCH_CHECK(use_fast_accum, "AMD does not support disabling use_fast_accum");
  auto [kernel, pad] = get_kernel_mode(XQ, WQ);
  if (pad) {
    if (kernel == KernelMode::Small) {
      return f8f8bf16_tensorwise_impl<64, 32, 64, 64, 1, 2, true>(
          XQ, WQ, scale);
    } else if (kernel == KernelMode::Large) {
      return f8f8bf16_tensorwise_impl<256, 256, 128, 64, 4, 2, true>(
          XQ, WQ, scale);
    } else {
      return f8f8bf16_tensorwise_impl<256, 128, 128, 64, 2, 2, true>(
          XQ, WQ, scale);
    }
  } else {
    if (kernel == KernelMode::Small) {
      return f8f8bf16_tensorwise_impl<64, 32, 64, 64, 1, 2>(XQ, WQ, scale);
    } else if (kernel == KernelMode::Large) {
      return f8f8bf16_tensorwise_impl<256, 256, 128, 64, 4, 2>(XQ, WQ, scale);
    } else {
      return f8f8bf16_tensorwise_impl<256, 128, 128, 64, 2, 2>(XQ, WQ, scale);
    }
  }
}

} // namespace fbgemm_gpu
