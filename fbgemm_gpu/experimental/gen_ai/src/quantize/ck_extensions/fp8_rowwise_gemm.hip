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
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#if defined(USE_ROCM)

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

#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

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
};

namespace fbgemm_gpu {

template <
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int MPER_WAVE,
    int NPER_WAVE,
    bool PADDING = false,
    bool TINY = false>
at::Tensor f8f8bf16_rowwise_impl(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> output = c10::nullopt) {
  // Get input information.
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);

  int StrideA = K;
  int StrideB = K;
  int StrideE = N;

  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  at::Tensor Y;
  if (output.has_value()) {
    Y = output.value();
    // Make sure the provided output has the proper shape and dtype.
    TORCH_CHECK(Y.size(0) == M && Y.size(1) == N);
    TORCH_CHECK(Y.dtype() == at::kBFloat16);
  } else {
    Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  }

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
  using CDEElementOp = RowwiseScale;

  static constexpr auto GemmDefault =
      ck::tensor_operation::device::GemmSpecialization::Default;
  static constexpr auto GemmMNKPadding =
      ck::tensor_operation::device::GemmSpecialization::MNKPadding;
  static constexpr auto GemmSpec = PADDING ? GemmMNKPadding : GemmDefault;
  using ComputeType = ck::f8_t;

  // Define derivative constants based on template parameters.
  static constexpr int BLOCK_CLUSTER = BLOCK_SIZE / 4;
  static constexpr int CBLOCK_N = TINY ? 4 : NBLOCK / 16;
  static constexpr int CBLOCK_M = TINY ? 16 : BLOCK_SIZE / CBLOCK_N;

  // A few modes change for tiny kernels.
  static constexpr int MPER_XDL = TINY ? 16 : 32;
  static constexpr int NPER_XDL = TINY ? 16 : 32;
  static constexpr auto LOOP_SCHED = TINY
      ? ck::BlockGemmPipelineScheduler::Intrawave
      : ck::BlockGemmPipelineScheduler::Interwave;
  using ABLOCK_TRANSFER =
      std::conditional_t<TINY, S<BLOCK_CLUSTER, 4, 1>, S<4, BLOCK_CLUSTER, 1>>;
  using BBLOCK_TRANSFER =
      std::conditional_t<TINY, S<BLOCK_CLUSTER, 4, 1>, S<4, BLOCK_CLUSTER, 1>>;
  using CBLOCK_TRANSFER = std::conditional_t<TINY, S<4, 4, 1>, S<8, 8, 1>>;

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
          GemmSpec,
          BLOCK_SIZE, // Block Size
          MBLOCK, // M per Block
          NBLOCK, // N per Block
          KBLOCK, // K per Block
          16, // AK1
          16, // BK1
          MPER_XDL, // M per Xdl
          NPER_XDL, // N per Xdl
          MPER_WAVE, // Mxdl per Wave
          NPER_WAVE, // Nxdl per Wave
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
          1,
          1,
          S<1, CBLOCK_M, 1, CBLOCK_N>,
          CBLOCK_TRANSFER,
          LOOP_SCHED,
          ck::BlockGemmPipelineVersion::v1,
          ComputeType>;

  // Create gemm launcher and arguments.
  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto cde_element_op = CDEElementOp{};

  constexpr ck::index_t NumDTensor = DsDataType::Size();
  constexpr auto I0 =
      ck::Number<0>{}; // Used to indicate 0 stride for row and col broadcast.

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
      std::array<ck::index_t, NumDTensor>{I0, I0},
      StrideE,
      a_element_op,
      b_element_op,
      cde_element_op);

  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(argument, StreamConfig{stream, false});

  return Y;
}

enum class RowKernelMode { Tiny, Small, Large, Default };

std::tuple<RowKernelMode, bool> get_row_kernel_mode(
    at::Tensor XQ,
    at::Tensor WQ) {
  auto M = XQ.size(0);
  auto K = XQ.size(1);
  auto N = WQ.size(0);
  // Tiny kernels should be used when M and N are so small we need to load
  // mostly from K. We also find this kernel is good for many shapes with small
  // M up until a certain N.
  bool use_tiny_kernel = (M <= 32 || N <= 32) || (M <= 128 && N <= 8192);
  // For other cases where M is small but N is large, we have a specialized
  // kernel.
  bool use_small_kernel = (M <= 128);
  // Larger workloads can load big chunks.
  bool use_large_kernel =
      ((M >= 4096 && N >= 4096) || (M >= 8192 && N >= 2048) ||
       (N >= 8192 && M >= 2048) || (K >= 8192 && M >= 2048 && N >= 2048));
  // Set padding based on the selected mode.
  bool use_pad;
  if (use_tiny_kernel) {
    // Tiny kernels use chunks of 16 in M and N and 256 in K.
    // If any dimension cant be divided into proper chunks, we pad.
    use_pad = (M % 16 != 0) || (N % 16 != 0) || (K % 256 != 0);
    return {RowKernelMode::Tiny, use_pad};
  } else if (use_small_kernel) {
    // Small kernels load chunks of 32 in M, 128 in N and 128 in K.
    use_pad = (M % 32 != 0) || (N % 128 != 0) || (K % 128 != 0);
    return {RowKernelMode::Small, use_pad};
  } else if (use_large_kernel) {
    // Large kernels load chunks of 256 in M, 128 in K and 64 in K.
    use_pad = (M % 256 != 0) || (N % 128 != 0) || (K % 64 != 0);
    return {RowKernelMode::Large, use_pad};
  } else {
    // Default kernel loads chunks of 128 in M and N and 64 in K.
    use_pad = (M % 128 != 0) || (N % 128 != 0) || (K % 64 != 0);
    return {RowKernelMode::Default, use_pad};
  }
}

at::Tensor f8f8bf16_rowwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    std::optional<at::Tensor> bias,
    bool use_fast_accum,
    std::optional<at::Tensor> output = c10::nullopt) {
  // Check that input datatypes are valid.
  TORCH_CHECK(
      (XQ.dtype() == at::kFloat8_e4m3fnuz) &&
          (WQ.dtype() == at::kFloat8_e4m3fnuz),
      "Inputs must be type float8_e4m3fnuz.");
  TORCH_CHECK(
      (x_scale.dtype() == at::kFloat) && (w_scale.dtype() == at::kFloat),
      "Scales must be float32.");
  TORCH_CHECK(use_fast_accum, "AMD does not support disabling use_fast_accum.");
  TORCH_CHECK(!(bias.has_value()), "AMD does not yet support bias.");
  auto [kernel, pad] = get_row_kernel_mode(XQ, WQ);
  if (pad) {
    if (kernel == RowKernelMode::Tiny) {
      return f8f8bf16_rowwise_impl<64, 16, 16, 256, 1, 1, true, true>(
          XQ, WQ, x_scale, w_scale, output);
    } else if (kernel == RowKernelMode::Small) {
      return f8f8bf16_rowwise_impl<128, 32, 128, 128, 1, 2, true, false>(
          XQ, WQ, x_scale, w_scale, output);
    } else if (kernel == RowKernelMode::Large) {
      return f8f8bf16_rowwise_impl<256, 256, 128, 64, 4, 2, true, false>(
          XQ, WQ, x_scale, w_scale, output);
    } else {
      return f8f8bf16_rowwise_impl<256, 128, 128, 64, 2, 2, true, false>(
          XQ, WQ, x_scale, w_scale, output);
    }
  } else {
    if (kernel == RowKernelMode::Tiny) {
      return f8f8bf16_rowwise_impl<64, 16, 16, 256, 1, 1, false, true>(
          XQ, WQ, x_scale, w_scale, output);
    } else if (kernel == RowKernelMode::Small) {
      return f8f8bf16_rowwise_impl<128, 32, 128, 128, 1, 2, true, false>(
          XQ, WQ, x_scale, w_scale, output);
    } else if (kernel == RowKernelMode::Large) {
      return f8f8bf16_rowwise_impl<256, 256, 128, 64, 4, 2, false, false>(
          XQ, WQ, x_scale, w_scale, output);
    } else {
      return f8f8bf16_rowwise_impl<256, 128, 128, 64, 2, 2, false, false>(
          XQ, WQ, x_scale, w_scale, output);
    }
  }
}

} // namespace fbgemm_gpu

#endif // defined(USE_ROCM)
