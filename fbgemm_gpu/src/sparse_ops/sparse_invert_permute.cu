/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void invert_permute_kernel(
    int32_t permute_size,
    const index_t* __restrict__ permute,
    index_t* __restrict__ inversed_permute) {
  CUDA_KERNEL_LOOP(i, permute_size) {
    inversed_permute[permute[i]] = i;
  }
}

DLL_PUBLIC Tensor invert_permute_cuda(const Tensor& permute) {
  TENSOR_ON_CUDA_GPU(permute);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(permute.get_device());
  const auto permute_contig = permute.contiguous();
  const auto permute_size = permute.numel();
  Tensor inversed_permute = at::empty_like(permute);

  constexpr int32_t threads_1 = kMaxThreads;
  const auto blocks_1 = cuda_calc_xblock_count(permute_size, threads_1);
  AT_DISPATCH_INDEX_TYPES(permute.scalar_type(), "invert_permute_kernel", [&] {
    invert_permute_kernel<index_t>
        <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
            permute_size,
            permute_contig.data_ptr<index_t>(),
            inversed_permute.data_ptr<index_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return inversed_permute;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(CUDA, "invert_permute", fbgemm_gpu::invert_permute_cuda);
