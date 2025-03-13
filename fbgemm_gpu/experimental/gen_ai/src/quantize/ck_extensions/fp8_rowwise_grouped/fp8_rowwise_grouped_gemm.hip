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

template <typename InputType, typename OutputType>
using RowwiseGroupedKernel = std::function<OutputType(
    InputType,
    InputType,
    InputType,
    InputType,
    at::Tensor,
    OutputType)>;

// Define useful types that are needed for various kernels.
using KernelArguments =
    ck::tensor_operation::device::GroupedGemmKernelArgument<2>;
using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using D0DataType = float;
using D1DataType = float;
using DsDataType = ck::Tuple<D0DataType, D1DataType>;
using EDataType = ck::bhalf_t;

template <typename InputType, typename OutputType>
RowwiseGroupedKernel<InputType, OutputType>
rowwise_grouped_heuristic_dispatch(int M, int N, int K) {
  // We use shape heuristics to find the best kernel.
  // To do this, we divide by the size of M and find the best
  // option within that grouping.
  if (M <= 16) {
    if (N < 8192 && K <= 8192) {
      return fp8_rowwise_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1<
          InputType,
          OutputType>;
    }
    if (K <= 8192) {
      return fp8_rowwise_grouped_128x16x64x128_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2<
          InputType,
          OutputType>;
    }
    return fp8_rowwise_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2<
        InputType,
        OutputType>;
  }
  if (M <= 32) {
    if (N < 8192 && K <= 8192) {
      return fp8_rowwise_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2<
          InputType,
          OutputType>;
    }
    if (K <= 8192) {
      return fp8_rowwise_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2<
          InputType,
          OutputType>;
    }
    return fp8_rowwise_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2<
        InputType,
        OutputType>;
  }
  if (M <= 64) {
    return fp8_rowwise_grouped_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
        InputType,
        OutputType>;
  }
  if (M <= 128) {
    if (N < 8192 && K <= 8192) {
      return fp8_rowwise_grouped_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
          InputType,
          OutputType>;
    }
    return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
        InputType,
        OutputType>;
  }
  if (M <= 256) {
    return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
        InputType,
        OutputType>;
  }
  if (M <= 512) {
    if (K <= 8192) {
      return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
          InputType,
          OutputType>;
    }
    return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<
        InputType,
        OutputType>;
  }
  // Default kernel for all other shapes.
  return fp8_rowwise_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<
      InputType,
      OutputType>;
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

template <typename OutputType>
void set_static_kernel_args(
    at::Tensor kernel_args,
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    OutputType output) {
  // Get current cuda stream.
  auto stream = at::cuda::getCurrentHIPStream().stream();
  int group_count = XQ.size();
  // Declare variables for loop.
  EDataType* output_ptr;
  int output_offset = 0;
  // When group count is large, we can more efficiently initialize
  // by doing host setup and a memcpy. This is only viable if cuda
  // graphs arent being used.
  // Iterate over inputs and get group information.
  for (int i = 0; i < group_count; i++) {
    int M = XQ[i].size(0);
    int K = XQ[i].size(1);
    int N = WQ[i].size(0);

    // Compute proper output pointer.
    if constexpr (std::is_same_v<OutputType, std::vector<at::Tensor>>) {
      // Output is a list of tensors and we can access each individually.
      output_ptr = reinterpret_cast<EDataType*>(output[i].data_ptr());
    } else {
      // Output is a single contiguous tensor and must be accessed via offset.
      output_ptr =
          reinterpret_cast<EDataType*>(output.data_ptr()) + output_offset;
      output_offset += M * N;
    }

    // We use the smallest reasonable block size since we effectively need only
    // 1 thread.
    const int blockSize = 32;
    const int numBlocks = 1;
    // Launch a kernel for each group to set kernel memory on device.
    // Using multiple kernels this way allows us to support arbitrary M,N,K.
    // For some reason, this approach is faster than using hipmemcpy.
    // Launch kernel to set kernel arguments.
    set_kernel_args_kernel<<<numBlocks, blockSize, 0, stream>>>(
        reinterpret_cast<KernelArguments*>(
            reinterpret_cast<char*>(kernel_args.data_ptr()) +
            (i * sizeof(KernelArguments))),
        reinterpret_cast<ADataType*>(XQ[i].data_ptr()),
        reinterpret_cast<BDataType*>(WQ[i].data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale[i].data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale[i].data_ptr()),
        output_ptr,
        M,
        N,
        K);
  }
}

__global__ void set_kernel_args_m_sizes_kernel(
    KernelArguments* kernel_args,
    ADataType* XQ,
    BDataType* WQ,
    D0DataType* w_scale,
    D1DataType* x_scale,
    EDataType* output,
    int64_t* M_sizes,
    int M,
    int N,
    int K,
    int group_count) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread is responsible for setting up the arguments for one group.
  if (thread_idx < group_count) {
    // Get M information for this group.
    int kernel_M = M_sizes[thread_idx];
    int offset_M = 0;
    // Offset is computed by finding the sum of previous group Ms.
    for (int i = 0; i < thread_idx; i++) {
      offset_M += M_sizes[i];
    }
    KernelArguments kernel_group_args = {
        XQ + (offset_M * K),
        WQ + (thread_idx * N * K),
        {w_scale + (thread_idx * N), x_scale + offset_M},
        output + (offset_M * N),
        kernel_M,
        N,
        K,
        K,
        K,
        {0, 0},
        N};
    // Write kernel args to memory.
    kernel_args[thread_idx] = kernel_group_args;
  }
}

__global__ void set_kernel_args_fixed_nk_kernel(
    KernelArguments* kernel_args,
    ADataType* XQ,
    BDataType* WQ,
    D0DataType* w_scale,
    D1DataType* x_scale,
    EDataType* output,
    int64_t* prepad_M,
    int M,
    int N,
    int K,
    int group_count) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread is responsible for setting up the arguments for one group.
  if (thread_idx < group_count) {
    // Compute offsets for this group.
    int group_M = prepad_M[thread_idx];
    KernelArguments kernel_group_args = {
        XQ + (thread_idx * M * K),
        WQ + (thread_idx * N * K),
        {w_scale + (thread_idx * N), x_scale + (thread_idx * M)},
        output + (thread_idx * M * N),
        group_M,
        N,
        K,
        K,
        K,
        {0, 0},
        N};
    // Write kernel args to memory.
    kernel_args[thread_idx] = kernel_group_args;
  }
}

__global__ void set_kernel_args_fixed_nk_kernel_zeroing(
    KernelArguments* kernel_args,
    ADataType* XQ,
    BDataType* WQ,
    D0DataType* w_scale,
    D1DataType* x_scale,
    EDataType* output,
    int64_t* prepad_M,
    int M,
    int N,
    int K,
    int group_count) {
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread is responsible for setting up the arguments for one group.
  if (thread_idx < group_count) {
    // Compute offsets for this group.
    int group_M = prepad_M[thread_idx];
    KernelArguments kernel_group_args = {
        XQ + (thread_idx * M * K),
        WQ + (thread_idx * N * K),
        {w_scale + (thread_idx * N), x_scale + (thread_idx * M)},
        output + (thread_idx * M * N),
        group_M,
        N,
        K,
        K,
        K,
        {0, 0},
        N};
    // Write kernel args to memory.
    kernel_args[thread_idx] = kernel_group_args;
  }

  // Figure out where in memory we are.
  // Each thread sets one float 4 which corresponds to 8 bf16 values.
  int output_offset = (thread_idx * 8);
  int current_group = output_offset / (M * N);
  // Skip if outside of valid groups.
  if (current_group < group_count) {
    int nonzeros = prepad_M[current_group];
    int current_M = (output_offset % (M * N)) / N;
    // Only write zeros if we're currently in a sparse row.
    if (current_M >= nonzeros) {
      // Write out a block of 8 output values via vectorized float4.
      float4* output_block = reinterpret_cast<float4*>(output + output_offset);
      *output_block = {0, 0, 0, 0};
    }
  }
}

void set_dynamic_kernel_args(
    at::Tensor kernel_args,
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> zero_start_index_M,
    std::optional<at::Tensor> M_sizes,
    bool zeroing_output_tensor) {
  // Get current cuda stream.
  auto stream = at::cuda::getCurrentHIPStream().stream();
  int group_count;
  // Check provided tensors are valid.
  TORCH_CHECK(
      zero_start_index_M.has_value() != M_sizes.has_value(),
      "One of zero_start_index_M or M_sizes must be provided.");
  if (zero_start_index_M.has_value()) {
    group_count = zero_start_index_M.value().size(0);
    TORCH_CHECK(
        XQ.device() == zero_start_index_M.value().device(),
        "zero_start_index_M and inputs must be on the same device.");
    TORCH_CHECK(
        zero_start_index_M.value().dtype() == at::kLong,
        "zero_start_index_M must be int64.");
  }
  if (M_sizes.has_value()) {
    group_count = M_sizes.value().size(0);
    TORCH_CHECK(
        XQ.device() == M_sizes.value().device(),
        "M_sizes and inputs must be on the same device.");
    TORCH_CHECK(M_sizes.value().dtype() == at::kLong, "M_sizes must be int64.");
  }

  // When m_sizes is used XQ is shape [tota_M, K]. When zero_start_index_M is
  // used it is shape [G, M, K].
  int M = XQ.size(XQ.dim() - 2);
  int K = WQ.size(2);
  int N = WQ.size(1);

  // Depending on the mode, use appropriate setup kernel.
  if (M_sizes.has_value()) {
    set_kernel_args_m_sizes_kernel<<<1, group_count, 0, stream>>>(
        reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
        reinterpret_cast<ADataType*>(XQ.data_ptr()),
        reinterpret_cast<BDataType*>(WQ.data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale.data_ptr()),
        reinterpret_cast<EDataType*>(output.data_ptr()),
        reinterpret_cast<int64_t*>(M_sizes.value().data_ptr()),
        M,
        N,
        K,
        group_count);
  } else if (zeroing_output_tensor) {
    // Launch a kernel that sets kernel argument memory and zeros the output.
    // Each thread sets one float4 which corresponds to 8 bf16 values.
    const int BLOCK_SIZE = 8;
    TORCH_CHECK(
        N % BLOCK_SIZE == 0, "N must be divisible 8 for dynamic grouped gemm.");
    int block_factor =
        std::max(group_count, (group_count * M * N) / BLOCK_SIZE);
    int blockSize = std::min(512, block_factor);
    int numBlocks = (block_factor + blockSize - 1) / blockSize;
    set_kernel_args_fixed_nk_kernel_zeroing<<<
        numBlocks,
        blockSize,
        0,
        stream>>>(
        reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
        reinterpret_cast<ADataType*>(XQ.data_ptr()),
        reinterpret_cast<BDataType*>(WQ.data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale.data_ptr()),
        reinterpret_cast<EDataType*>(output.data_ptr()),
        reinterpret_cast<int64_t*>(zero_start_index_M.value().data_ptr()),
        M,
        N,
        K,
        group_count);
  } else {
    set_kernel_args_fixed_nk_kernel<<<1, group_count, 0, stream>>>(
        reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
        reinterpret_cast<ADataType*>(XQ.data_ptr()),
        reinterpret_cast<BDataType*>(WQ.data_ptr()),
        reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
        reinterpret_cast<D1DataType*>(x_scale.data_ptr()),
        reinterpret_cast<EDataType*>(output.data_ptr()),
        reinterpret_cast<int64_t*>(zero_start_index_M.value().data_ptr()),
        M,
        N,
        K,
        group_count);
  }
}

std::vector<at::Tensor> f8f8bf16_rowwise_grouped(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<std::vector<at::Tensor>> output = std::nullopt) {
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
  // Need to handle different output modes separately.
  // First handle tensor list output.
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
  at::Tensor kernel_args = at::empty(
      {static_cast<long>(group_count * sizeof(KernelArguments))},
      XQ[0].options().dtype(at::kByte));
  set_static_kernel_args<std::vector<at::Tensor>>(
      kernel_args, XQ, WQ, x_scale, w_scale, Y);

  // We use the largest of each shape for heuristics.
  int MaxM = 0;
  int MaxN = 0;
  int MaxK = 0;
  for (int i = 0; i < group_count; i++) {
    MaxM = max(MaxM, XQ[i].size(0));
    MaxN = max(MaxN, WQ[i].size(0));
    MaxK = max(MaxK, XQ[i].size(1));
  }
  RowwiseGroupedKernel<at::TensorList, std::vector<at::Tensor>>
      selected_kernel = rowwise_grouped_heuristic_dispatch<
          at::TensorList,
          std::vector<at::Tensor>>(MaxM, MaxN, MaxK);
  return selected_kernel(XQ, WQ, x_scale, w_scale, kernel_args, Y);
}

// Wrapper function for list input single tensor output.
at::Tensor f8f8bf16_rowwise_grouped_stacked(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor M_sizes,
    std::optional<at::Tensor> output = std::nullopt) {
  // Check that input datatypes are valid.
  // First confirm that there are the same number of groups in all inputs.
  int group_count = M_sizes.size(0);
  // XQ is expected to be shape [total_M, K].
  int total_M = XQ.size(0);
  // WQ is expected to be shape [G, N, K].
  int N = WQ.size(1);
  int K = XQ.size(1);
  TORCH_CHECK(
      WQ.size(0) == group_count && x_scale.numel() == total_M &&
          w_scale.numel() / group_count == N,
      "All inputs must have the same number of groups.");
  // Iterate over inputs and check they are valid.
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(XQ.dim() == 2, "Input XQ must be 2D (total_M,K).");
  TORCH_CHECK(
      XQ.dtype() == at::kFloat8_e4m3fnuz,
      "Input XQ must be type float8_e4m3fnuz.");

  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(WQ.dim() == 3, "Input WQ must be 3D (G,N,K).");
  TORCH_CHECK(
      WQ.dtype() == at::kFloat8_e4m3fnuz,
      "Input WQ must be type float8_e4m3fnuz.");
  TORCH_CHECK(
      WQ.size(1) >= 512 && WQ.size(2) >= 512,
      "N and K must be at least 512 for grouped gemm. For smaller inputs, consider unrolling.");

  TORCH_CHECK(x_scale.dtype() == at::kFloat, "Scales must be float32.");
  TORCH_CHECK(w_scale.dtype() == at::kFloat, "Scales must be float32.");

  // Allocate an empty output array. We will set its values to zero as part
  // of kernel setup.
  at::Tensor Y = at::empty({total_M, N}, XQ.options().dtype(at::kBFloat16));

  // Early exit for empty input.
  if (total_M == 0) {
    return Y;
  }

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args = at::empty(
      {static_cast<long>(group_count * sizeof(KernelArguments))},
      XQ.options().dtype(at::kByte));
  set_dynamic_kernel_args(
      kernel_args, XQ, WQ, x_scale, w_scale, Y, std::nullopt, M_sizes, false);

  RowwiseGroupedKernel<at::Tensor, at::Tensor> selected_kernel =
      rowwise_grouped_heuristic_dispatch<at::Tensor, at::Tensor>(
          total_M / group_count, N, K);
  return selected_kernel(XQ, WQ, x_scale, w_scale, kernel_args, Y);
}

at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor zero_start_index_M,
    bool zeroing_output_tensor = true) {
  // Check that input datatypes are valid.
  // First confirm that there are the same number of groups in all inputs.
  int group_count = XQ.size(0);
  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = WQ.size(2);
  TORCH_CHECK(
      WQ.size(0) == group_count && x_scale.numel() / group_count == M &&
          w_scale.numel() / group_count == N,
      "All inputs must have the same number of groups.");
  // Iterate over inputs and check they are valid.
  TORCH_CHECK(XQ.is_cuda() && XQ.is_contiguous());
  TORCH_CHECK(XQ.dim() == 3, "Input XQ must be 3D (G,M,K).");
  TORCH_CHECK(
      XQ.dtype() == at::kFloat8_e4m3fnuz,
      "Input XQ must be type float8_e4m3fnuz.");

  TORCH_CHECK(WQ.is_cuda() && WQ.is_contiguous());
  TORCH_CHECK(WQ.dim() == 3, "Input WQ must be 3D (G,N,K).");
  TORCH_CHECK(
      WQ.dtype() == at::kFloat8_e4m3fnuz,
      "Input WQ must be type float8_e4m3fnuz.");
  TORCH_CHECK(
      WQ.size(1) >= 512 && WQ.size(2) >= 512,
      "N and K must be at least 512 for grouped gemm. For smaller inputs, consider unrolling.");

  TORCH_CHECK(x_scale.dtype() == at::kFloat, "Scales must be float32.");
  TORCH_CHECK(w_scale.dtype() == at::kFloat, "Scales must be float32.");

  // Allocate an empty output array. We will set its values to zero as part
  // of kernel setup.
  at::Tensor Y =
      at::empty({group_count, M, N}, XQ.options().dtype(at::kBFloat16));

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args = at::empty(
      {static_cast<long>(group_count * sizeof(KernelArguments))},
      XQ.options().dtype(at::kByte));
  set_dynamic_kernel_args(
      kernel_args,
      XQ,
      WQ,
      x_scale,
      w_scale,
      Y,
      zero_start_index_M,
      std::nullopt,
      zeroing_output_tensor);

  RowwiseGroupedKernel<at::Tensor, at::Tensor> selected_kernel =
      rowwise_grouped_heuristic_dispatch<at::Tensor, at::Tensor>(M, N, K);
  return selected_kernel(XQ, WQ, x_scale, w_scale, kernel_args, Y);
}

} // namespace fbgemm_gpu
