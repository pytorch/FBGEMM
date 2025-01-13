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
#include "kernels/bf16_grouped_kernel_manifest.h"

namespace fbgemm_gpu {

// Define useful types that are needed for various kernels.
using KernelArguments =
    ck::tensor_operation::device::GroupedGemmKernelArgument<0>;
using ADataType = ck::bhalf_t;
using BDataType = ck::bhalf_t;
using CDataType = ck::bhalf_t;

GroupedKernel grouped_heuristic_dispatch(int M, int N, int K) {
  // We use shape heuristics to find the best kernel.
  // To do this, we divide by the size of M and find the best
  // option within that grouping.
  if (M <= 16) {
    if (N < 8192 && K <= 8192) {
      return bf16_grouped_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1;
    }
    if (K <= 8192) {
      return bf16_grouped_128x16x64x128_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2;
    }
    return bf16_grouped_128x16x32x256_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2;
  }
  if (M <= 32) {
    if (N < 8192 && K <= 8192) {
      return bf16_grouped_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2;
    }
    if (K <= 8192) {
      return bf16_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2;
    }
    return bf16_grouped_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2;
  }
  if (M <= 64) {
    return bf16_grouped_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  if (M <= 128) {
    if (N < 8192 && K <= 8192) {
      return bf16_grouped_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
    }
    return bf16_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  if (M <= 256) {
    return bf16_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  if (M <= 512) {
    if (K <= 8192) {
      return bf16_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1;
    }
    return bf16_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3;
  }
  // Default kernel for all other shapes.
  return bf16_grouped_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1;
}

__global__ void set_kernel_args_kernel(
    KernelArguments* kernel_args,
    ADataType* A,
    BDataType* B,
    CDataType* output,
    int M,
    int N,
    int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Each kernel annoyingly can only set the kernel args for one group.
  // This could only be avoided with complicated memory management.
  if (idx == 0) {
    // Write kernel arguments directly to memory.
    KernelArguments kernel_group_args = {
        A, B, {}, output, M, N, K, K, K, {}, N};
    kernel_args[0] = kernel_group_args;
  }
}

void set_static_kernel_args(
    at::Tensor kernel_args,
    at::TensorList A,
    at::TensorList B,
    std::vector<at::Tensor> output) {
  // Get current cuda stream.
  auto stream = at::cuda::getCurrentHIPStream().stream();
  int group_count = A.size();
  // When group count is large, we can more efficiently initialize
  // by doing host setup and a memcpy. This is only viable if cuda
  // graphs arent being used.
  if (group_count >= 16 && stream == 0) {
    std::vector<KernelArguments> ggemm_kargs;
    ggemm_kargs.reserve(group_count);

    // Iterate over inputs and get group information.
    for (int i = 0; i < group_count; i++) {
      int M = A[i].size(0);
      int K = A[i].size(1);
      int N = B[i].size(0);
      KernelArguments group_args = {
          reinterpret_cast<ADataType*>(A[i].data_ptr()),
          reinterpret_cast<BDataType*>(B[i].data_ptr()),
          {},
          reinterpret_cast<CDataType*>(output[i].data_ptr()),
          M,
          N,
          K,
          K,
          K,
          {},
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
      int M = A[i].size(0);
      int K = A[i].size(1);
      int N = B[i].size(0);
      // Launch kernel to set kernel arguments.
      set_kernel_args_kernel<<<numBlocks, blockSize, 0, stream>>>(
          reinterpret_cast<KernelArguments*>(
              reinterpret_cast<char*>(kernel_args.data_ptr()) +
              (i * sizeof(KernelArguments))),
          reinterpret_cast<ADataType*>(A[i].data_ptr()),
          reinterpret_cast<BDataType*>(B[i].data_ptr()),
          reinterpret_cast<CDataType*>(output[i].data_ptr()),
          M,
          N,
          K);
    }
  }
}

__global__ void set_kernel_args_fixed_nk_kernel(
    KernelArguments* kernel_args,
    ADataType* A,
    BDataType* B,
    CDataType* output,
    int64_t* prepad_M,
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
        A + (group_idx * M * K),
        B + (group_idx * N * K),
        {},
        output + (group_idx * M * N),
        group_M,
        N,
        K,
        K,
        K,
        {},
        N};
    // Write kernel args to memory.
    kernel_args[group_idx] = kernel_group_args;
  }
}

void set_dynamic_kernel_args(
    at::Tensor kernel_args,
    at::TensorList A,
    at::TensorList B,
    std::vector<at::Tensor> output,
    at::Tensor zero_start_index_M) {
  // Get current cuda stream.
  auto stream = at::cuda::getCurrentHIPStream().stream();
  int group_count = A.size();
  // Confirm M is on the proper device.
  TORCH_CHECK(
      A[0].device() == zero_start_index_M.device(),
      "zero_start_index_M and inputs must be on the same device.");
  TORCH_CHECK(
      zero_start_index_M.size(0) == group_count,
      "zero_start_index_M must have an entry for each group.");
  TORCH_CHECK(
      zero_start_index_M.dtype() == at::kLong,
      "zero_start_index_M must be int64.");

  // We assume that M, N, and K are fixed across groups.
  // The actual m values are sstored in the passed M tensor.
  int M = A[0].size(0);
  int K = A[0].size(1);
  int N = B[0].size(0);

  // Make sure that inputs are allocated in sequential memory as required by
  // this mode.
  for (int i = 1; i < group_count; i++) {
    // Check that all inputs are allocated directly following preceding input.
    TORCH_CHECK(
        A[i].data_ptr() ==
            (reinterpret_cast<ADataType*>(A[i - 1].data_ptr()) + (M * K)),
        "Inputs must be sequential in memory to support dynamic M, but XQ is not.");
    TORCH_CHECK(
        B[i].data_ptr() ==
            (reinterpret_cast<BDataType*>(B[i - 1].data_ptr()) + (N * K)),
        "Inputs must be sequential in memory to support dynamic M, but WQ is not.");
    TORCH_CHECK(
        output[i].data_ptr() ==
            (reinterpret_cast<CDataType*>(output[i - 1].data_ptr()) + (M * N)),
        "Inputs must be sequential in memory to support dynamic M, but output is not.");
  }

  // Launch a kernel that sets kernel argument memory.
  int const blockSize = std::min(1024, group_count);
  int const numBlocks = (group_count + blockSize - 1) / blockSize;
  set_kernel_args_fixed_nk_kernel<<<numBlocks, blockSize, 0, stream>>>(
      reinterpret_cast<KernelArguments*>(kernel_args.data_ptr()),
      reinterpret_cast<ADataType*>(A[0].data_ptr()),
      reinterpret_cast<BDataType*>(B[0].data_ptr()),
      reinterpret_cast<CDataType*>(output[0].data_ptr()),
      reinterpret_cast<int64_t*>(zero_start_index_M.data_ptr()),
      M,
      N,
      K,
      group_count);
}

at::Tensor get_grouped_kernel_args(
    at::TensorList A,
    at::TensorList B,
    std::optional<at::Tensor> zero_start_index_M,
    std::vector<at::Tensor> output) {
  int group_count = A.size();
  // Get space on device for the kernel argument tensor.
  at::Tensor kernel_args = at::empty(
      {static_cast<long>(group_count * sizeof(KernelArguments))},
      A[0].options().dtype(at::kByte));

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
        kernel_args, A, B, output, zero_start_index_M.value());
  } else {
    set_static_kernel_args(kernel_args, A, B, output);
  }
  return kernel_args;
}

std::vector<at::Tensor> bf16bf16bf16_grouped(
    at::TensorList A,
    at::TensorList B,
    std::optional<std::vector<at::Tensor>> output = std::nullopt) {
  // Check that input datatypes are valid.
  // First confirm that there are the same number of groups in all inputs.
  TORCH_CHECK(
      A.size() == B.size(), "A and B must have the same number of groups.");
  int group_count = A.size();
  // Iterate over inputs and check they are valid.
  for (at::Tensor a : A) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous());
    TORCH_CHECK(a.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(a.dtype() == at::kBFloat16, "Inputs must be type bfloat16.");
  }
  for (at::Tensor b : B) {
    TORCH_CHECK(b.is_cuda() && b.is_contiguous());
    TORCH_CHECK(b.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(b.dtype() == at::kBFloat16, "Inputs must be type bfloat16.");
  }

  std::vector<at::Tensor> Y;
  if (output.has_value()) {
    Y = output.value();
    TORCH_CHECK(
        Y.size() == group_count,
        "Output and input must have same number of groups.");
    // Check that output shapes are correct.
    for (int i = 0; i < group_count; i++) {
      int M = A[i].size(0);
      int N = B[i].size(0);
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
      int M = A[i].size(0);
      int N = B[i].size(0);
      Y.push_back(at::empty({M, N}, A[i].options().dtype(at::kBFloat16)));
    }
  }

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args = get_grouped_kernel_args(A, B, std::nullopt, Y);

  // Perform shape lookup to find best kernel.
  // We use the largest of each shape for heuristics.
  int MaxM = 0;
  int MaxN = 0;
  int MaxK = 0;
  for (int i = 0; i < group_count; i++) {
    MaxM = max(MaxM, A[i].size(0));
    MaxN = max(MaxN, B[i].size(0));
    MaxK = max(MaxK, A[i].size(1));
  }
  GroupedKernel selected_kernel = grouped_heuristic_dispatch(MaxM, MaxN, MaxK);
  return selected_kernel(A, B, kernel_args, Y);
}

at::Tensor bf16bf16bf16_grouped_dynamic(
    at::TensorList A,
    at::TensorList B,
    std::optional<at::Tensor> zero_start_index_M = std::nullopt) {
  // Check that input datatypes are valid.
  // First confirm that there are the same number of groups in all inputs.
  TORCH_CHECK(
      A.size() == B.size(), "A and B must have the same number of groups.");
  int group_count = A.size();
  // Iterate over inputs and check they are valid.
  for (at::Tensor a : A) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous());
    TORCH_CHECK(a.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(a.dtype() == at::kBFloat16, "Inputs must be type bfloat16.");
  }
  for (at::Tensor b : B) {
    TORCH_CHECK(b.is_cuda() && b.is_contiguous());
    TORCH_CHECK(b.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(b.dtype() == at::kBFloat16, "Inputs must be type bfloat16.");
  }

  std::vector<at::Tensor> Y;
  at::Tensor Y_full;
  int N = B[0].size(0);
  int K = A[0].size(1);

  if (zero_start_index_M.has_value()) {
    int M = A[0].size(0);
    // Fill output with zeros to simplify integration. This prevents nans from
    // showing up in the tensor.
    Y_full =
        at::zeros({group_count, M, N}, A[0].options().dtype(at::kBFloat16));
    // Split the output into groups.
    Y = at::unbind(Y_full, 0);
  } else {
    // If not provided, we try to allocate a single blob that can store each
    // group.
    int total_M = 0;
    std::vector<int> group_sizes = {};
    for (int i = 0; i < group_count; i++) {
      TORCH_CHECK(
          A[i].size(1) == K && B[i].size(0) == N,
          "Dynamic grouped gemm requires fixed N and K.");
      int group_M = A[i].size(0);
      total_M += group_M;
      group_sizes.push_back(group_M);
    }
    // Allocate a contiguous array for all groups.
    Y_full = at::empty({total_M, N}, A[0].options().dtype(at::kBFloat16));
    // Split the full array into appropriate groups.
    // We do this with narrow to make sure there are no extra copies.
    int offset = 0;
    for (int size : group_sizes) {
      Y.push_back(Y_full.narrow(0, offset, size));
      offset += size;
    }
  }

  // Prepare kernel arguments by copying them to the proper device location.
  at::Tensor kernel_args = get_grouped_kernel_args(A, B, zero_start_index_M, Y);

  // Perform shape lookup to find best kernel.
  // We use the largest of each shape for heuristics.
  int MaxM = 0;
  int MaxN = 0;
  int MaxK = 0;
  for (int i = 0; i < group_count; i++) {
    MaxM = max(MaxM, A[i].size(0));
    MaxN = max(MaxN, B[i].size(0));
    MaxK = max(MaxK, A[i].size(1));
  }
  GroupedKernel selected_kernel = grouped_heuristic_dispatch(MaxM, MaxN, MaxK);
  // Run kernel to populate output.
  selected_kernel(A, B, kernel_args, Y);
  // Return coalesced view of output tensor.
  return Y_full;
}

} // namespace fbgemm_gpu
