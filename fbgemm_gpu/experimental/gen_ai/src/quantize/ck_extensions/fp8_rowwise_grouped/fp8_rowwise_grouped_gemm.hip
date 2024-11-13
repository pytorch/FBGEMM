/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp"
#include "kernels/fp8_rowwise_grouped_kernel_manifest.h"

namespace fbgemm_gpu {

// Define useful types that are needed for various kernels.
using KernelArguments =
    ck::tensor_operation::device::GroupedGemmTileLoopKernelArguments<2>;
using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using D0DataType = float;
using D1DataType = float;
using DsDataType = ck::Tuple<D0DataType, D1DataType>;
using EDataType = ck::bhalf_t;

RowwiseGroupedKernel rowwise_grouped_heuristic_dispatch(int M, int N, int K) {
  // We use shape heuristics to find the best kernel.
  // To do this, we divide by the size of M and find the best
  // option within that grouping.
  if (M <= 16) {
    if (N < 8192 && K <= 8192) {
      return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1;
    }
    if (K <= 8192) {
      return fp8_rowwise_grouped_128x16x64x128_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2;
    }
    return fp8_rowwise_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2;
  }
  if (M <= 32) {
    if (N < 8192 && K <= 8192) {
      return fp8_rowwise_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2;
    }
    if (K <= 8192) {
      return fp8_rowwise_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2;
    }
    return fp8_rowwise_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2;
  }
  if (M <= 64) {
    return fp8_rowwise_grouped_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  if (M <= 128) {
    if (N < 8192 && K <= 8192) {
      return fp8_rowwise_grouped_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
    }
    return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  if (M <= 256) {
    return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  if (M <= 512) {
    if (K <= 8192) {
      return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1;
    }
    return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  // Default kernel for all other shapes.
  return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1;
}

__global__ void set_kernel_args_kernel(
    KernelArguments* kernel_args,
    ADataType* XQ,
    BDataType* WQ,
    D0DataType* w_scale,
    D1DataType* x_scale,
    EDataType* output,
    int M,
    int N,
    int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each kernel annoyingly can only set the kernel args for one group.
  // This could only be avoided with complicated memory management.
  if (idx == 0) {
    // Write kernel arguments directly to memory.
    KernelArguments kernel_group_args = {
        XQ, WQ, {w_scale, x_scale}, output, M, N, K, K, K, {0, 0}, N};
    kernel_args[0] = kernel_group_args;
  }
}

void set_grouped_kernel_args(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor kernel_args,
    std::vector<at::Tensor> output) {
  TORCH_CHECK(
      XQ.size() == WQ.size() && XQ.size() == x_scale.size() &&
          XQ.size() == w_scale.size(),
      "All inputs must have the same number of groups.");
  int group_count = XQ.size();
  // We use the smallest reasonable block size since we effectively need only 1 thread.
  int blockSize = 32;
  int numBlocks = 1;
  auto stream = at::cuda::getCurrentHIPStream().stream();

  // Launch a kernel for each group to set kernel memory on device.
  for (int i = 0; i < group_count; i++) {
    int M = XQ[i].size(0);
    int K = XQ[i].size(1);
    int N = WQ[i].size(0);
    // Launch kernel to set kernel arguments.
    set_kernel_args_kernel<<<numBlocks, blockSize, 0, stream>>>(
        reinterpret_cast<KernelArguments*>(
            reinterpret_cast<char*>(kernel_args.data_ptr()) +
            (i * sizeof(KernelArguments))),
        reinterpret_cast<ADataType*>(XQ[i].data_ptr()),
        reinterpret_cast<BDataType*>(WQ[i].data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale[i].data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale[i].data_ptr()),
        reinterpret_cast<EDataType*>(output[i].data_ptr()),
        M,
        N,
        K);
  }
}

std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<std::vector<at::Tensor>> output = std::nullopt,
    std::optional<std::string> kernel_name = std::nullopt) {
  // Check that input datatypes are valid.
  // First confirm that there are the same number of groups in all inputs.
  TORCH_CHECK(
      XQ.size() == WQ.size() && XQ.size() == x_scale.size() &&
          XQ.size() == w_scale.size(),
      "All inputs must have the same number of groups.");
  int group_count = XQ.size();
  // Iterate over inputs and check they are valid.
  for (at::Tensor x : XQ) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(
        x.dtype() == at::kFloat8_e4m3fnuz,
        "Inputs must be type float8_e4m3fnuz.");
  }
  for (at::Tensor w : WQ) {
    TORCH_CHECK(w.is_cuda() && w.is_contiguous());
    TORCH_CHECK(w.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(
        w.dtype() == at::kFloat8_e4m3fnuz,
        "Inputs must be type float8_e4m3fnuz.");
  }
  for (at::Tensor xs : x_scale) {
    TORCH_CHECK(xs.dtype() == at::kFloat, "Scales must be float32.");
  }
  for (at::Tensor ws : x_scale) {
    TORCH_CHECK(ws.dtype() == at::kFloat, "Scales must be float32.");
  }

  std::vector<at::Tensor> Y;
  if (output.has_value()) {
    Y = output.value();
    TORCH_CHECK(
        Y.size() == group_count,
        "Output and input must have same number of groups.");
    // Check that output shapes are correct.
    for (int i = 0; i < group_count; i++) {
      int M = XQ[i].size(0);
      int N = WQ[i].size(0);
      int out_M = Y[i].size(0);
      int out_N = Y[i].size(1);
      TORCH_CHECK(
          M == out_M && N == out_N,
          "Output tensors do not have the expected shape.");
      TORCH_CHECK(
          Y[i].dtype() == at::kBFloat16, "Output dtype must be bfloat16.");
    }
  } else {
    for (int i = 0; i < group_count; i++) {
      int M = XQ[i].size(0);
      int N = WQ[i].size(0);
      Y.push_back(at::empty({M, N}, XQ[i].options().dtype(at::kBFloat16)));
    }
  }

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args = at::empty({1000}, XQ[0].options().dtype(at::kByte));
  set_grouped_kernel_args(XQ, WQ, x_scale, w_scale, kernel_args, Y);

  // If provided a specific kernel implementation, dispatch to it.
  if (kernel_name.has_value()) {
    auto it = kernel_name_map.find(kernel_name.value());
    // If not found, raise an error.
    TORCH_CHECK(
        it != kernel_name_map.end(),
        "Could not find kernel " + kernel_name.value());
    // If found, always use requested kernel.
    return it->second(XQ, WQ, x_scale, w_scale, kernel_args, Y);
  }
  // Otherwise, use heuristics to find the best kernel options.
  // We use the largest of each shape for heuristics.
  int MaxM = 0;
  int MaxN = 0;
  int MaxK = 0;
  for (int i = 0; i < group_count; i++) {
    MaxM = max(MaxM, XQ[i].size(0));
    MaxN = max(MaxN, WQ[i].size(0));
    MaxK = max(MaxK, XQ[i].size(1));
  }
  RowwiseGroupedKernel selected_kernel =
      rowwise_grouped_heuristic_dispatch(MaxM, MaxN, MaxK);
  return selected_kernel(XQ, WQ, x_scale, w_scale, kernel_args, Y);
}

std::vector<std::string> get_f8f8bf16_rowwise_grouped_kernels() {
  /* Helper function to get the names of avaialable grouped gemm kernels.*/
  std::vector<std::string> kernel_names;
  for (const auto& pair : kernel_name_map) {
    kernel_names.push_back(pair.first);
  }
  return kernel_names;
}

} // namespace fbgemm_gpu
