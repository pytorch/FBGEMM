/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h> // @manual

#include "cutlass/gemm/device/gemm_grouped.h" // @manual
#include "fbgemm_gpu/group_gemm_ops.h"
#include "fbgemm_gpu/group_gemm_ops_gpu.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm_gpu {

namespace {
using Tensor = at::Tensor;

bool is_transposed_2d_(const Tensor& ten) {
  return ten.dim() == 2 && ten.stride(0) == 1 && ten.stride(1) > 1 &&
      ten.stride(1) == ten.size(0);
}

} // namespace

std::vector<Tensor> gemm_grouped_gpu(
    const std::vector<Tensor>& a_group,
    const std::vector<Tensor>& b_group,
    const c10::optional<std::vector<Tensor>>& c_group) {
  const int problem_count = a_group.size();
  TORCH_CHECK(problem_count == b_group.size())
  if (c_group.has_value()) {
    TORCH_CHECK(problem_count == c_group.value().size())
  }

  if (problem_count == 0) {
    return std::vector<Tensor>();
  }

  auto device = a_group[0].get_device();

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(device);

  cudaDeviceProp* deviceProp = at::cuda::getDeviceProperties(device);

  bool is_b_transposed = is_transposed_2d_(b_group[0]);
  for (int i = 0; i < problem_count; ++i) {
    if (is_b_transposed) {
      TORCH_CHECK(
          is_transposed_2d_(b_group[i]),
          "second tensor group transpose is not consistent");
    }
    const int n = b_group[i].size(1);
    const int k = a_group[i].size(1);

    // CUTLASS gemm_grouped fails with the misalign address error if n or k
    // dimension is not multiple of 8
    // TODO: Fix this (T114637949)
    TORCH_CHECK(n % 8 == 0 && k % 8 == 0)
  }

#define LAUNCH_GEMM_GROUPED(LAYOUT, ARCH) \
  output_group =                          \
      gemm_grouped_cuda<scalar_t, LAYOUT, ARCH>(a_group, b_group, c_group);

  std::vector<Tensor> output_group;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      a_group[0].scalar_type(), "gemm_grouped", [&] {
        if (deviceProp->major >= 8) {
          if (is_b_transposed) {
            LAUNCH_GEMM_GROUPED(
                cutlass::layout::ColumnMajor, cutlass::arch::Sm80)
          } else {
            LAUNCH_GEMM_GROUPED(cutlass::layout::RowMajor, cutlass::arch::Sm80)
          }
        } else if (deviceProp->major >= 7) {
          if (is_b_transposed) {
            LAUNCH_GEMM_GROUPED(
                cutlass::layout::ColumnMajor, cutlass::arch::Sm70)
          } else {
            LAUNCH_GEMM_GROUPED(cutlass::layout::RowMajor, cutlass::arch::Sm70)
          }
        } else {
          // TODO: support older GPUs
          TORCH_CHECK(false, "Your GPU is too old.");
        }
      });

#undef LAUNCH_GEMM_GROUPED

  return output_group;
}

std::vector<Tensor> group_linear_forward_gpu(
    const std::vector<Tensor>& input_group,
    const std::vector<Tensor>& weight_group,
    const c10::optional<std::vector<Tensor>>& bias_group) {
  auto weight_group_transposed =
      group_linear_forward_helper(input_group, weight_group, bias_group);
  return gemm_grouped_gpu(input_group, weight_group_transposed, bias_group);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA("gmm", fbgemm_gpu::gemm_grouped_gpu);
  DISPATCH_TO_CUDA(
      "group_linear_forward", fbgemm_gpu::group_linear_forward_gpu);
}
