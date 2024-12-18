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

__global__ void set_kernel_args_fixed_nk_kernel(
    KernelArguments* kernel_args,
    ADataType* XQ,
    BDataType* WQ,
    D0DataType* w_scale,
    D1DataType* x_scale,
    EDataType* output,
    int32_t* prepad_M,
    int M,
    int N,
    int K,
    int group_count) {
  int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread is responsible for setting up the arguments for one group.
  if (group_idx < group_count) {
    // Compute offsets for this group.
    int group_M = prepad_M[group_idx];
    KernelArguments kernel_group_args = {
        XQ + (group_idx * M * K),
        WQ + (group_idx * N * K),
        {w_scale + (group_idx * N), x_scale + (group_idx * M)},
        output + (group_idx * M * N),
        group_M,
        N,
        K,
        K,
        K,
        {0, 0},
        N};
    // Write kernel args to memory.
    kernel_args[group_idx] = kernel_group_args;
  }
}

at::Tensor get_grouped_kernel_args(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::Tensor> zero_start_index_M,
    std::vector<at::Tensor> output) {
  int group_count = XQ.size();
  auto stream = at::cuda::getCurrentHIPStream().stream();

  // Get space on device for the kernel argument tensor.
  at::Tensor kernel_args = at::empty(
      {static_cast<long>(group_count * sizeof(KernelArguments))},
      XQ[0].options().dtype(at::kByte));

  // There are two different modes for this kernel.
  // When zero_start_index_M is provided, we assume that data is sequential and
  // that N and K are constants. This allows a more efficient kernel
  // launch and is best suited to MOE use cases where M is truly dynamic.
  // When zero_start_index_M is not provided, we assume M, N, and K can all vary
  // and set them for each group. It is important to note that this does not
  // work well with cuda graphs and runtime dynamism so if possible we recommend
  // using zero_start_index_M.

  if (zero_start_index_M.has_value()) {
    // Make sure zero_start_index_M is configured properly.
    at::Tensor prepad_M = zero_start_index_M.value();
    // Confirm M is on the proper device.
    TORCH_CHECK(
        XQ[0].device() == prepad_M.device(),
        "zero_start_index_M and inputs must be on the same device.");
    TORCH_CHECK(
        prepad_M.size(0) == group_count,
        "zero_start_index_M must have an entry for each group.");

    // We assume that M, N, and K are fixed across groups.
    // The actual m values are sstored in the passed M tensor.
    int M = XQ[0].size(0);
    int K = XQ[0].size(1);
    int N = WQ[0].size(0);

    // Make sure that inputs are allocated in sequential memory as required by
    // this mode.
    for (int i = 1; i < group_count; i++) {
      // Check that all inputs are allocated directly following preceding input.
      TORCH_CHECK(
          XQ[i].data_ptr() ==
              (reinterpret_cast<ADataType*>(XQ[i - 1].data_ptr()) + (M * K)),
          "Inputs must be sequential in memory to support dynamic M, but XQ is not.");
      TORCH_CHECK(
          WQ[i].data_ptr() ==
              (reinterpret_cast<BDataType*>(WQ[i - 1].data_ptr()) + (N * K)),
          "Inputs must be sequential in memory to support dynamic M, but WQ is not.");
      TORCH_CHECK(
          x_scale[i].data_ptr() ==
              (reinterpret_cast<D0DataType*>(x_scale[i - 1].data_ptr()) + (M)),
          "Inputs must be sequential in memory to support dynamic M, but x_scale is not.");
      TORCH_CHECK(
          w_scale[i].data_ptr() ==
              (reinterpret_cast<D1DataType*>(w_scale[i - 1].data_ptr()) + (N)),
          "Inputs must be sequential in memory to support dynamic M, but w_scale is not.");
      TORCH_CHECK(
          output[i].data_ptr() ==
              (reinterpret_cast<EDataType*>(output[i - 1].data_ptr()) +
               (M * N)),
          "Inputs must be sequential in memory to support dynamic M, but output is not.");
    }

    // Launch a kernel that sets kernel argument memory.
    int const blockSize = std::min(1024, group_count);
    int const numBlocks = (group_count + blockSize - 1) / blockSize;
    set_kernel_args_fixed_nk_kernel<<<numBlocks, blockSize, 0, stream>>>(
        reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
        reinterpret_cast<ADataType*>(XQ[0].data_ptr()),
        reinterpret_cast<BDataType*>(WQ[0].data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale[0].data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale[0].data_ptr()),
        reinterpret_cast<EDataType*>(output[0].data_ptr()),
        reinterpret_cast<int32_t*>(prepad_M.data_ptr()),
        M,
        N,
        K,
        group_count);
    return kernel_args;
  } else {
    // When running in eager mode, we assume we can directly interact with host
    // values.
    // Note that this version is not supported with cuda graphs.
    TORCH_CHECK(
        stream == 0,
        "f8f8bf16_rowwise_grouped eager mode is not supported with cuda graphs.");

    std::vector<KernelArguments> ggemm_kargs;
    ggemm_kargs.reserve(group_count);

    // Iterate over inputs and get group information.
    for (int i = 0; i < group_count; i++) {
      int M = XQ[i].size(0);
      int K = XQ[i].size(1);
      int N = WQ[i].size(0);
      KernelArguments group_args = {
          reinterpret_cast<ADataType*>(XQ[i].data_ptr()),
          reinterpret_cast<BDataType*>(WQ[i].data_ptr()),
          {reinterpret_cast<D0DataType*>(w_scale[i].data_ptr()),
           reinterpret_cast<D1DataType*>(x_scale[i].data_ptr())},
          reinterpret_cast<EDataType*>(output[i].data_ptr()),
          M,
          N,
          K,
          K,
          K,
          {0, 0},
          N};
      ggemm_kargs.push_back(group_args);
    }
    // Copy data onto device.
    hipMemcpy(
      kernel_args.data_ptr(), // Destination
      ggemm_kargs.data(), // Source
      sizeof(KernelArguments) * group_count, // Number of bytes
      hipMemcpyHostToDevice); // Copy Type
  }

  return kernel_args;
}

std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt,
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
    TORCH_CHECK(
        w.size(0) >= 512 && w.size(1) >= 512,
        "N and K must be at least 512 for grouped gemm. For smaller inputs, consider unrolling.");
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
    // Two modes for allocating output. When m_values is provided, we need
    // the output tensor to be contiguous and can assume M, N, and K are the
    // same across groups. Otherwise, we can allocate each output separately.
    if (zero_start_index_M.has_value()) {
      int M = XQ[0].size(0);
      int N = WQ[0].size(0);
      // Fill output with zeros to simplify integration. This prevents nans from
      // showing up in the tensor.
      at::Tensor Y_full =
          at::zeros({group_count, M, N}, XQ[0].options().dtype(at::kBFloat16));
      // Split the output into groups.
      Y = at::unbind(Y_full, 0);
    } else {
      for (int i = 0; i < group_count; i++) {
        int M = XQ[i].size(0);
        int N = WQ[i].size(0);
        Y.push_back(at::empty({M, N}, XQ[i].options().dtype(at::kBFloat16)));
      }
    }
  }

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args =
      get_grouped_kernel_args(XQ, WQ, x_scale, w_scale, zero_start_index_M, Y);

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
