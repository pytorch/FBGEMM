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
__global__
__launch_bounds__(kMaxThreads) void compute_frequency_sequence_kernel(
    index_t* input,
    int64_t* output,
    index_t start_input,
    const int input_size) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= input_size) {
    return;
  }
  // Atomic could become a bottleneck if frequencies are very skew
  atomicAdd(&output[input[i] - start_input], 1);
}

DLL_PUBLIC void compute_frequency_sequence(
    const Tensor& input,
    Tensor& output,
    const int start_input,
    const int output_size) {
  output = at::zeros({output_size}, input.options().dtype(at::kLong));

  AT_DISPATCH_INDEX_TYPES(
      input.scalar_type(), "compute_frequency_sequence_kernel_1", [&] {
        compute_frequency_sequence_kernel<index_t>
            <<<cuda_calc_xblock_count(input.numel(), kWarpSize),
               kWarpSize,
               0,
               at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<index_t>(),
                output.data_ptr<int64_t>(),
                start_input,
                input.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace fbgemm_gpu
