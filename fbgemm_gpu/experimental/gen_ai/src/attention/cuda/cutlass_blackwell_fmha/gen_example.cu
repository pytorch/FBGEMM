// @nolint
/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "blackwell_gen_interface.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <random>
#include <vector>

#include "cutlass/util/reference/device/tensor_fill.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

struct TensorCreationResult {
  at::Tensor q, k, v, seqlen_kv, batch_idx;
};

template <typename Element>
TensorCreationResult init_tensors(int B, int H, int Hk, int D, int Sk) {
  std::cout << "Creating tensor options..." << std::endl;

  at::TensorOptions options;
  try {
    options = at::TensorOptions().dtype(to_torch_type<Element>()).device(at::kCUDA);
    std::cout << "Successfully created tensor options" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create tensor options: " << e.what() << std::endl;
    throw std::runtime_error("Failed to create tensor options");
  }

  std::cout << "Creating empty input tensors..." << std::endl;

  // Create empty tensors first
  at::Tensor q = at::empty({B, 1, H, D}, options).to(at::kCUDA);
  at::Tensor k = at::empty({B, Sk, Hk, D}, options).to(at::kCUDA);
  at::Tensor v = at::empty_like(k);

  // Initialize tensors using CUTLASS reference functions
  std::cout << "Initializing tensors with CUTLASS reference functions..." << std::endl;

  // Initialize Q tensor
  cutlass::reference::device::BlockFillRandomGaussian(
      static_cast<Element*>(q.data_ptr()),
      q.numel(),
      2023,
      Element(0),
      Element(1));

  // Initialize K tensor
  cutlass::reference::device::BlockFillRandomGaussian(
      static_cast<Element*>(k.data_ptr()),
      k.numel(),
      2024,
      Element(0),
      Element(1));

  // Initialize V tensor
  cutlass::reference::device::BlockFillRandomGaussian(
      static_cast<Element*>(v.data_ptr()),
      v.numel(),
      2025,
      Element(0),
      Element(1));

  std::cout << "Input tensors initialized" << std::endl;

  // Create integer tensors using host-based initialization
  std::cout << "Creating integer tensors with host initialization..." << std::endl;

  // Create host vectors first
  std::vector<int> host_seqlen_kv(B, Sk);
  std::vector<int> host_batch_idx(B);
  for (int i = 0; i < B; ++i) {
    host_seqlen_kv[i] = Sk;
    host_batch_idx[i] = i;
  }

  // Create GPU tensors and copy host data to them
  at::Tensor seqlen_kv =
      at::from_blob(
          host_seqlen_kv.data(), {B}, at::TensorOptions().dtype(at::kInt))
          .to(at::kCUDA);

  at::Tensor batch_idx =
      at::from_blob(
          host_batch_idx.data(), {B}, at::TensorOptions().dtype(at::kInt))
          .to(at::kCUDA);

  return {q, k, v, seqlen_kv, batch_idx};
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const** args) {
  std::cout << "Running Blackwell Fused Multi-Head Attention Kernel" << std::endl;

  // Initialize CUDA context to prevent errors
  std::cout << "Initializing CUDA context..." << std::endl;
  cudaFree(0); // This initializes the CUDA context

  // Check CUDA device availability
  int device_count = 0;
  cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
  if (cuda_status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << std::endl;
    return -1;
  }
  std::cout << "Found " << device_count << " CUDA devices" << std::endl;

  if (device_count == 0) {
    std::cerr << "No CUDA devices available" << std::endl;
    return -1;
  }

// Check if SM100 support is available
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cout << "CUTLASS_ARCH_MMA_SM100_SUPPORTED is defined" << std::endl;
#else
  std::cerr << "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined" << std::endl;
  return -1;
#endif

  int B = 2, H = 5, Hk = 1, D = 128, Sk = 1024;
  auto input_dtype = cutlass::float_e4m3_t{};
  //auto input_dtype = cutlass::bfloat16_t{};
  // Test with different kernel types
  const auto kernel_type = static_cast<int>(KernelType::UMMA_I); // Can be "UMMA_I" or "UMMA_P"

  // Use helper method to create and initialize tensors
  auto tensor_result = init_tensors<decltype(input_dtype)>(B, H, Hk, D, Sk);
  at::Tensor q = tensor_result.q;
  at::Tensor k = tensor_result.k;
  at::Tensor v = tensor_result.v;
  at::Tensor seqlen_kv = tensor_result.seqlen_kv;
  at::Tensor batch_idx = tensor_result.batch_idx;

  std::cout << "Input tensor shapes:" << std::endl;
  std::cout << "  Q: " << q.sizes() << ", dtype: " << q.dtype() << std::endl;
  std::cout << "  K: " << k.sizes() << ", dtype: " << k.dtype() << std::endl;
  std::cout << "  V: " << v.sizes() << ", dtype: " << v.dtype() << std::endl;
  std::cout << "  seqlen_kv: " << seqlen_kv.sizes()
            << ", dtype: " << seqlen_kv.dtype() << std::endl;
  std::cout << "  batch_idx: " << batch_idx.sizes()
            << ", dtype: " << batch_idx.dtype() << std::endl;

  std::cout << "Calling dispatch_fmha_gen_fwd..." << std::endl;

  auto o = dispatch_fmha_gen_fwd(q, k, v, seqlen_kv, batch_idx, kernel_type);

  std::cout << "dispatch_fmha_gen_fwd completed" << std::endl;

  // Check if output tensor is valid
  if (o.defined()) {
    std::cout << "Output tensor is valid" << std::endl;
    std::cout << "Output dtype: " << o.dtype() << std::endl;
    std::cout << "Output shape: " << o.sizes() << std::endl;
  } else {
    std::cout << "Output tensor is NOT defined/valid" << std::endl;
  }

  return 0;
}
