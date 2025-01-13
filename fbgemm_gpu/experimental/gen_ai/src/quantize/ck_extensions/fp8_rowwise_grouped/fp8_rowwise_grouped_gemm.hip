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
#include <hip_bf16.h>
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp"
#include "kernels/fp8_rowwise_grouped_kernel_manifest.h"

namespace fbgemm_gpu {

// Define useful types that are needed for various kernels.
using KernelArguments =
    ck::tensor_operation::device::GroupedGemmKernelArgument<2>;
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

void set_static_kernel_args(
    at::Tensor kernel_args,
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::vector<at::Tensor> output) {
  // Get current cuda stream.
  auto stream = at::cuda::getCurrentHIPStream().stream();
  int group_count = XQ.size();
  // When group count is large, we can more efficiently initialize
  // by doing host setup and a memcpy. This is only viable if cuda
  // graphs arent being used.
  if (group_count >= 16 && stream == 0) {
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
  } else {
    // We use the smallest reasonable block size since we effectively need only
    // 1 thread.
    int blockSize = 32;
    int numBlocks = 1;
    // Launch a kernel for each group to set kernel memory on device.
    // Using multiple kernels this way allows us to support arbitrary M,N,K.
    // For some reason, this approach is faster than using hipmemcpy.
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
    int group_count,
    const int BLOCK_SIZE) {
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

  // We also fuse in initialization of the output tensor.
  // We write in chunks of 2 bfloats at a time for efficiency.
  for (int i = 0; i < BLOCK_SIZE / 2; i++) {
    // Figure out where in memory we are.
    int output_offset = (thread_idx * BLOCK_SIZE) + (i * 2);
    int current_group = output_offset / (M * N);
    // Skip if outside of valid groups.
    if (current_group < group_count) {
      int nonzeros = prepad_M[current_group];
      int current_M = output_offset / N;
      // Only write if this block needs initialization.
      // Avoid writing to final element if number of elements is odd.
      if (current_M >= nonzeros && output_offset < (M * N * group_count) - 1) {
        __hip_bfloat162* output_block =
            reinterpret_cast<__hip_bfloat162*>(output + output_offset);
        *output_block = __hip_bfloat162(0, 0);
      }
    }
  }
  // Handle case where there are an odd number of total elements.
  if (((M * N * group_count) % 2) != 0 &&
      ((M * N * group_count) - (thread_idx * BLOCK_SIZE) < BLOCK_SIZE)) {
    // Write out the final element.
    __hip_bfloat16* output_block =
        reinterpret_cast<__hip_bfloat16*>(output + (M * N * group_count) - 1);
    *output_block = __hip_bfloat16(0);
  }
}

void set_dynamic_kernel_args(
    at::Tensor kernel_args,
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::vector<at::Tensor> output,
    at::Tensor zero_start_index_M) {
  // Get current cuda stream.
  auto stream = at::cuda::getCurrentHIPStream().stream();
  int group_count = XQ.size();
  // Confirm M is on the proper device.
  TORCH_CHECK(
      XQ[0].device() == zero_start_index_M.device(),
      "zero_start_index_M and inputs must be on the same device.");
  TORCH_CHECK(
      zero_start_index_M.size(0) == group_count,
      "zero_start_index_M must have an entry for each group.");
  TORCH_CHECK(
      zero_start_index_M.dtype() == at::kLong,
      "zero_start_index_M must be int64.");

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
            (reinterpret_cast<EDataType*>(output[i - 1].data_ptr()) + (M * N)),
        "Inputs must be sequential in memory to support dynamic M, but output is not.");
  }

  // Launch a kernel that sets kernel argument memory.
  const int BLOCK_SIZE = 8;
  int block_factor = std::max(group_count, (group_count * M * N) / BLOCK_SIZE);
  int blockSize = std::min(1024, block_factor);
  int numBlocks = (block_factor + blockSize - 1) / blockSize;
  set_kernel_args_fixed_nk_kernel<<<numBlocks, blockSize, 0, stream>>>(
      reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
      reinterpret_cast<ADataType*>(XQ[0].data_ptr()),
      reinterpret_cast<BDataType*>(WQ[0].data_ptr()),
      reinterpret_cast<D0DataType*>(w_scale[0].data_ptr()),
      reinterpret_cast<D1DataType*>(x_scale[0].data_ptr()),
      reinterpret_cast<EDataType*>(output[0].data_ptr()),
      reinterpret_cast<int64_t*>(zero_start_index_M.data_ptr()),
      M,
      N,
      K,
      group_count,
      BLOCK_SIZE);
}

at::Tensor get_grouped_kernel_args(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    std::optional<at::Tensor> zero_start_index_M,
    std::vector<at::Tensor> output) {
  int group_count = XQ.size();
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
    set_dynamic_kernel_args(
        kernel_args,
        XQ,
        WQ,
        x_scale,
        w_scale,
        output,
        zero_start_index_M.value());
  } else {
    set_static_kernel_args(kernel_args, XQ, WQ, x_scale, w_scale, output);
  }
  return kernel_args;
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
    for (int i = 0; i < group_count; i++) {
      int M = XQ[i].size(0);
      int N = WQ[i].size(0);
      Y.push_back(at::empty({M, N}, XQ[i].options().dtype(at::kBFloat16)));
    }
  }

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args =
      get_grouped_kernel_args(XQ, WQ, x_scale, w_scale, std::nullopt, Y);

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

at::Tensor f8f8bf16_rowwise_grouped_dynamic(
    at::TensorList XQ,
    at::TensorList WQ,
    at::TensorList x_scale,
    at::TensorList w_scale,
    at::Tensor zero_start_index_M,
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

  // Create a single chunk of tensor but view it as a list for compatibility.
  int M = XQ[0].size(0);
  int N = WQ[0].size(0);
  // Allocate an empty output array. We will set its values to zero as part
  // of kernel setup.
  at::Tensor Y_full =
      at::empty({group_count, M, N}, XQ[0].options().dtype(at::kBFloat16));
  // Split the output into groups.
  std::vector<at::Tensor> Y = at::unbind(Y_full, 0);

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
    it->second(XQ, WQ, x_scale, w_scale, kernel_args, Y);
    return Y_full;
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
  // Run kernel to populate Y.
  selected_kernel(XQ, WQ, x_scale, w_scale, kernel_args, Y);
  // Return unified view of Y_full.
  return Y_full;
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
