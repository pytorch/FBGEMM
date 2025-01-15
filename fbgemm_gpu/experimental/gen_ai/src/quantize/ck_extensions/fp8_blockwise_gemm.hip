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

#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

namespace fbgemm_gpu {

template <
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int MPER_WAVE,
    int NPER_WAVE,
    bool PADDING = false>
at::Tensor f8f8bf16_blockwise_impl(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale) {
  // Check that inputs are valid.
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());

  // Get input information.
  int M = XQ.size(0);
  int N = WQ.size(0);
  int K = XQ.size(1);

  // Create output tensor.
  auto Y = at::empty({M, N}, XQ.options().dtype(at::kBFloat16));
  // If inputs are empty return an empty tensor.
  if (M == 0 || N == 0 || K == 0) {
    return Y;
  }

  int StrideA = K;
  int StrideB = K;
  int StrideE = N;

  // Only block_size = 128 is supported.
  static constexpr ck::index_t Scale_Block_M = 128;
  static constexpr ck::index_t Scale_Block_N = 128;
  static constexpr ck::index_t Scale_Block_K = 128;

  using ADataType = ck::f8_t;
  using AScaleDataType = float;
  using BDataType = ck::f8_t;
  using BScaleDataType = float;
  using AccDataType = float;
  using CShuffleDataType = float;
  using DsDataType = ck::Tuple<>;
  using EDataType = ck::bhalf_t;

  using ALayout = Row;
  using AScaleLayout = Row;
  using BLayout = Col;
  using BScaleLayout = Col;
  using DsLayout = ck::Tuple<>;
  using ELayout = Row;

  using AElementOp = PassThrough;
  using BElementOp = PassThrough;
  using CDEElementOp = PassThrough;

  static constexpr auto GemmDefault =
      ck::tensor_operation::device::GemmSpecialization::Default;
  static constexpr auto GemmMNKPadding =
      ck::tensor_operation::device::GemmSpecialization::MNKPadding;
  static constexpr auto GemmSpec = PADDING ? GemmMNKPadding : GemmDefault;

  // Define derivative constants based on template parameters.
  static constexpr int BLOCK_CLUSTER = BLOCK_SIZE / 4;
  static constexpr int CBLOCK_N = NBLOCK / 16;
  static constexpr int CBLOCK_M = BLOCK_SIZE / CBLOCK_N;

  using DeviceGemmInstance =
      ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3<
          ALayout,
          BLayout,
          DsLayout,
          ELayout,
          ADataType,
          AScaleDataType,
          BDataType,
          BScaleDataType,
          DsDataType,
          EDataType,
          AccDataType,
          CShuffleDataType,
          AElementOp,
          BElementOp,
          CDEElementOp,
          GemmSpec,
          BLOCK_SIZE,
          Scale_Block_M,
          Scale_Block_N,
          Scale_Block_K,
          MBLOCK,
          NBLOCK,
          KBLOCK,
          16,
          16,
          32,
          32,
          MPER_WAVE,
          NPER_WAVE,
          S<4, BLOCK_CLUSTER, 1>,
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          16,
          16,
          0,
          S<4, BLOCK_CLUSTER, 1>,
          S<1, 0, 2>,
          S<1, 0, 2>,
          2,
          16,
          16,
          0,
          1,
          1,
          S<1, CBLOCK_M, 1, CBLOCK_N>,
          S<8, 8, 1>,
          ck::BlockGemmPipelineScheduler::Intrawave,
          ck::BlockGemmPipelineVersion::v1,
          ck::f8_t>;

  // Create gemm launcher and arguments.
  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto cde_element_op = CDEElementOp{};

  auto argument = gemm.MakeArgument(
      reinterpret_cast<ADataType*>(XQ.data_ptr()),
      reinterpret_cast<BDataType*>(WQ.data_ptr()),
      std::array<const void*, 0>{},
      reinterpret_cast<EDataType*>(Y.data_ptr()),
      M,
      N,
      K,
      StrideA,
      StrideB,
      std::array<ck::index_t, 0>{},
      StrideE,
      reinterpret_cast<AScaleDataType*>(x_scale.data_ptr()),
      reinterpret_cast<BScaleDataType*>(w_scale.data_ptr()),
      a_element_op,
      b_element_op,
      cde_element_op);

  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(argument, StreamConfig{stream, false});

  return Y;
}

enum class BlockKernelMode { Small, Large, Default };

std::tuple<BlockKernelMode, bool> get_block_kernel_mode(
    at::Tensor XQ,
    at::Tensor WQ) {
  auto M = XQ.size(0);
  auto K = XQ.size(1);
  auto N = WQ.size(0);
  // For other cases where M is small but N is large, we have a specialized
  // kernel.
  bool use_small_kernel = (M <= 128);
  // Larger workloads can load big chunks.
  bool use_large_kernel =
      ((M >= 4096 && N >= 4096) || (M >= 8192 && N >= 2048) ||
       (N >= 8192 && M >= 2048) || (K >= 8192 && M >= 2048 && N >= 2048));
  // Set padding based on the selected mode.
  bool use_pad;
  if (use_small_kernel) {
    // Small kernels load chunks of 32 in M, 128 in N and 128 in K.
    use_pad = (M % 32 != 0) || (N % 128 != 0) || (K % 128 != 0);
    return {BlockKernelMode::Small, use_pad};
  } else if (use_large_kernel) {
    // Large kernels load chunks of 256 in M, 128 in K and 64 in K.
    use_pad = (M % 256 != 0) || (N % 128 != 0) || (K % 64 != 0);
    return {BlockKernelMode::Large, use_pad};
  } else {
    // Default kernel loads chunks of 128 in M and N and 64 in K.
    use_pad = (M % 128 != 0) || (N % 128 != 0) || (K % 64 != 0);
    return {BlockKernelMode::Default, use_pad};
  }
}

at::Tensor f8f8bf16_blockwise(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    int64_t block_m = 128,
    int64_t block_n = 128,
    int64_t block_k = 128) {
  // Check that input datatypes are valid.
  TORCH_CHECK(
      (XQ.dtype() == at::kFloat8_e4m3fnuz) &&
          (WQ.dtype() == at::kFloat8_e4m3fnuz),
      "Inputs must be type float8_e4m3fnuz.");
  TORCH_CHECK(
      (x_scale.dtype() == at::kFloat) && (w_scale.dtype() == at::kFloat),
      "Scales must be float32.");
  TORCH_CHECK(
      (block_m == 128 && block_n == 128 && block_k == 128),
      "Only block_size = 128 is supported.");
  auto [kernel, pad] = get_block_kernel_mode(XQ, WQ);
  if (pad) {
    if (kernel == BlockKernelMode::Small) {
      return f8f8bf16_blockwise_impl<128, 32, 128, 128, 1, 2, true>(
          XQ, WQ, x_scale, w_scale);
    } else if (kernel == BlockKernelMode::Large) {
      return f8f8bf16_blockwise_impl<256, 256, 128, 64, 4, 2, true>(
          XQ, WQ, x_scale, w_scale);
    } else {
      return f8f8bf16_blockwise_impl<256, 128, 128, 64, 2, 2, true>(
          XQ, WQ, x_scale, w_scale);
    }
  } else {
    if (kernel == BlockKernelMode::Small) {
      return f8f8bf16_blockwise_impl<128, 32, 128, 128, 1, 2, true>(
          XQ, WQ, x_scale, w_scale);
    } else if (kernel == BlockKernelMode::Large) {
      return f8f8bf16_blockwise_impl<256, 256, 128, 64, 4, 2, false>(
          XQ, WQ, x_scale, w_scale);
    } else {
      return f8f8bf16_blockwise_impl<256, 128, 128, 64, 2, 2, false>(
          XQ, WQ, x_scale, w_scale);
    }
  }
}

} // namespace fbgemm_gpu
