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

template <typename Length_T, typename Data_T>
__global__ void unpack_segments_cuda_kernel(
    const Data_T* const data_ptr,
    const Length_T* const lengths_ptr,
    const Length_T* const lengths_cum_sum,
    const Length_T max_length,
    const int64_t num_seq,
    const int64_t cell_size,
    Data_T* const out_ptr) {
  CUDA_KERNEL_LOOP(i, num_seq * max_length * cell_size) {
    const auto seq = (i / cell_size) / max_length;
    const auto cell = (i / cell_size) % max_length;
    const auto offset = i % cell_size;
    if (cell < lengths_ptr[seq]) {
      const auto idx = (lengths_cum_sum[seq] + cell) * cell_size + offset;
      out_ptr[idx] = data_ptr[i];
    }
  }
}

/// Map N+1 dim tensor to N dim based on lengths tensor
/// Sequences that are shorter than the longest sequence are padded with
/// zeros.
/// @param data         N+1 dim Tensor.
/// @param lengths      1D int/long tensor contains the length in each of the
/// input.
/// @param total_length Sum of elements in the 1D tensor legnths
/// @param max_length   The pre-defined max_length for the packed segments.
/// @return unpacked_tensor N-dimensional tensor
DLL_PUBLIC Tensor pack_segments_backward_cuda(
    const Tensor& data,
    const Tensor& lengths,
    int64_t total_length,
    int64_t max_length) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(data, lengths);
  TENSOR_NDIM_IS_GE(data, 2);
  TENSOR_NDIM_EQUALS(lengths, 1);
  TORCH_CHECK_EQ(data.size(0), lengths.size(0));
  TORCH_CHECK(
      data.dtype() == at::ScalarType::Float ||
          data.dtype() == at::ScalarType::Double ||
          data.dtype() == at::ScalarType::Half ||
          data.dtype() == at::ScalarType::BFloat16,
      "data must be of type float or double or half or bfloat16");
  TORCH_CHECK(
      max_length == data.size(1),
      "max_length should be equal to the second dimension of the packed segments");

  CUDA_DEVICE_GUARD(data);

  Tensor unpacked_tensor; // The output tensor

  AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "unpack_segments_cuda", [&] {
    const auto* const lengths_data = lengths.data_ptr<index_t>();

    // Create output tensor of appropriate dimensions
    auto shape = data.sizes().vec();
    shape.erase(shape.begin());
    shape[0] = total_length;
    unpacked_tensor = at::empty(shape, data.options());

    if (!(data.size(0) && data.size(1))) { // TODO: What does this mean?
      return;
    }

    auto lengths_prefix_sum =
        fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths);
    auto lps_data = lengths_prefix_sum.data_ptr<index_t>();

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        data.scalar_type(),
        "unpack_segments_cuda-unpacking",
        [&] {
          const auto num_seq = lengths.size(0);
          const auto cell_size = data.numel() / (data.size(0) * data.size(1));
          const auto* const data_ptr = data.data_ptr<scalar_t>();
          auto* const out_data = unpacked_tensor.data_ptr<scalar_t>();

          unpack_segments_cuda_kernel<index_t, scalar_t>
              <<<cuda_calc_xblock_count(num_seq * max_length * cell_size, 128),
                 128,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  data_ptr,
                  lengths_data,
                  lps_data,
                  max_length,
                  num_seq,
                  cell_size,
                  out_data);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  });

  return unpacked_tensor;
}

} // namespace fbgemm_gpu
