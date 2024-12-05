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
__global__ void pack_segments_cuda_kernel(
    const Data_T* const data_ptr,
    const int64_t data_size_0,
    const Length_T* const lengths_ptr,
    const Length_T* const lengths_cum_sum,
    const Length_T max_length,
    const int64_t num_seq,
    const int64_t cell_size,
    const Data_T padding,
    Data_T* const out_ptr,
    TORCH_DSA_KERNEL_ARGS) {
  // PackSegments requires that the sum of the lengths is equal to the first
  //  dimension of data
  CUDA_KERNEL_ASSERT2(
      data_size_0 == lengths_cum_sum[num_seq - 1] + lengths_ptr[num_seq - 1]);

  CUDA_KERNEL_LOOP(i, num_seq * max_length * cell_size) {
    const auto seq = (i / cell_size) / max_length;
    const auto cell = (i / cell_size) % max_length;
    const auto offset = i % cell_size;
    if (cell >= lengths_ptr[seq]) {
      out_ptr[i] = padding;
    } else {
      const auto idx = (lengths_cum_sum[seq] + cell) * cell_size + offset;
      out_ptr[i] = data_ptr[idx];
    }
  }
}

template <typename Length_T, typename Data_T>
__global__ void pack_segments_cuda_v2_kernel(
    const Data_T* const data_ptr,
    const int64_t data_size_0,
    const Length_T* const lengths_ptr,
    const Length_T* const lengths_cum_sum,
    const Length_T max_length,
    const int64_t num_seq,
    const int64_t cell_size,
    const Data_T padding,
    bool* const presence_ptr,
    Data_T* const out_ptr,
    TORCH_DSA_KERNEL_ARGS) {
  // PackSegments requires that the sum of the lengths is equal to the first
  //  dimension of data
  CUDA_KERNEL_ASSERT2(
      data_size_0 == lengths_cum_sum[num_seq - 1] + lengths_ptr[num_seq - 1]);

  CUDA_KERNEL_LOOP_TYPE(i, num_seq * max_length * cell_size, int64_t) {
    const auto seq = (i / cell_size) / max_length;
    const auto cell = (i / cell_size) % max_length;
    const auto offset = i % cell_size;
    if (presence_ptr && offset == 0) {
      presence_ptr[i / cell_size] = cell < lengths_ptr[seq];
    }
    if (cell >= lengths_ptr[seq]) {
      out_ptr[i] = padding;
    } else {
      const auto idx = (lengths_cum_sum[seq] + cell) * cell_size + offset;
      out_ptr[i] = data_ptr[idx];
    }
  }
}

/// Map N dim tensor to N+1 dim based on lengths tensor.
/// Sequences that are shorter than the longest sequence are padded with
/// zeros.
/// @param t_in         N dim Tensor.
/// @param lengths      1D int/long tensor contains the length in each of the
/// output.
/// @param max_length   The pre-defined max_length for the packed segments.
/// @return packed_tensor
///         packed_tensor  N + 1 dim Tensor where dim(1) is the max length,
///                        dim(0) is the batch size.
DLL_PUBLIC Tensor pack_segments_forward_cuda(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(t_in, lengths);
  TENSOR_NDIM_IS_GE(t_in, 1);
  TENSOR_NDIM_EQUALS(lengths, 1);
  TORCH_CHECK(
      t_in.dtype() == at::ScalarType::Float ||
          t_in.dtype() == at::ScalarType::Double ||
          t_in.dtype() == at::ScalarType::Half ||
          t_in.dtype() == at::ScalarType::BFloat16,
      "t_in must be of type float or double or half or bfloat16");
  TORCH_CHECK_GT(max_length, 0);

  CUDA_DEVICE_GUARD(t_in);

  const auto t_in_c = t_in.contiguous();

  Tensor packed_tensor;

  AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "pack_segments_cuda", [&] {
    const auto* const lengths_data = lengths.data_ptr<index_t>();

    // Shape of output is batch_size x max_len x ...
    auto shape = t_in_c.sizes().vec(); // Get copy of current shape
    shape[0] = max_length; // Set first element to max_len
    shape.insert(
        shape.begin(), lengths.numel()); // Insert batch size at beginning
    packed_tensor = at::zeros(shape, t_in_c.options());

    if (t_in_c.size(0) == 0 || lengths.size(0) == 0) {
      return; // Return empty output (with the proper shape)
    }

    auto lengths_prefix_sum =
        fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths);
    auto lps_data = lengths_prefix_sum.data_ptr<index_t>();

    FBGEMM_DISPATCH_ALL_TYPES(
        t_in_c.scalar_type(), "pack_segments_cuda-packing", [&] {
          const auto* const data_ptr = t_in_c.data_ptr<scalar_t>();
          auto* const out_data = packed_tensor.data_ptr<scalar_t>();
          const auto num_seq = lengths.size(0);
          const auto cell_size = t_in_c.numel() / t_in_c.size(0);
          TORCH_DSA_KERNEL_LAUNCH(
              (pack_segments_cuda_kernel<index_t, scalar_t>),
              cuda_calc_xblock_count(num_seq * max_length * cell_size, 128),
              128,
              0,
              at::cuda::getCurrentCUDAStream(),
              data_ptr,
              t_in_c.size(0),
              lengths_data,
              lps_data,
              max_length,
              num_seq,
              cell_size,
              static_cast<scalar_t>(0),
              out_data);
        });
  });

  return packed_tensor;
}

/// Map N dim tensor to N+1 dim based on lengths tensor.
/// Sequences that are shorter than the longest sequence are padded with
/// zeros.
/// @param t_in         N dim Tensor.
/// @param lengths      1D int/long tensor contains the length in each of the
/// output.
/// @param max_length   The pre-defined max_length for the packed segments.
/// @return packed_tensor
///         packed_tensor  N + 1 dim Tensor where dim(1) is the max length,
///                        dim(0) is the batch size.
DLL_PUBLIC std::tuple<Tensor, std::optional<Tensor>>
pack_segments_forward_cuda_v2(
    const Tensor& t_in,
    const Tensor& lengths,
    const int64_t max_length,
    const bool pad_minf,
    const bool return_presence_mask) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(t_in, lengths);
  TENSOR_NDIM_IS_GE(t_in, 1);
  TENSOR_NDIM_EQUALS(lengths, 1);
  TORCH_CHECK(
      t_in.dtype() == at::ScalarType::Float ||
          t_in.dtype() == at::ScalarType::Half ||
          t_in.dtype() == at::ScalarType::BFloat16 ||
          t_in.dtype() == at::ScalarType::Int ||
          t_in.dtype() == at::ScalarType::Long,
      "t_in must be of type float, half, bfloat16, int or long");
  TORCH_CHECK_GT(max_length, 0);

  CUDA_DEVICE_GUARD(t_in);

  const auto t_in_c = t_in.contiguous();

  Tensor packed_tensor;
  std::optional<Tensor> presence_mask;

  AT_DISPATCH_INDEX_TYPES(lengths.scalar_type(), "pack_segments_cuda", [&] {
    const auto* const lengths_data = lengths.data_ptr<index_t>();

    // Shape of output is batch_size x max_len x ...
    auto shape = t_in_c.sizes().vec(); // Get copy of current shape
    shape[0] = max_length; // Set first element to max_len
    shape.insert(
        shape.begin(), lengths.numel()); // Insert batch size at beginning
    packed_tensor = at::zeros(shape, t_in_c.options());

    if (pad_minf) {
      // Downcasting double infinity to float should still give infinity
      packed_tensor = at::full(
          shape, -std::numeric_limits<double>::infinity(), t_in_c.options());
    } else {
      packed_tensor = at::zeros(shape, t_in_c.options());
    }

    bool* presence_mask_data = nullptr;
    if (return_presence_mask) {
      // Shape of presence is batch_size x max_len
      presence_mask = at::zeros(
          {lengths.numel(), max_length}, t_in_c.options().dtype(at::kBool));
      presence_mask_data = presence_mask->data_ptr<bool>();
    }

    if (t_in_c.size(0) == 0 || lengths.size(0) == 0) {
      return; // Return empty output (with the proper shape)
    }

    auto lengths_prefix_sum =
        fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths);
    auto lps_data = lengths_prefix_sum.data_ptr<index_t>();

    FBGEMM_DISPATCH_ALL_TYPES(
        t_in_c.scalar_type(), "pack_segments_cuda-packing", [&] {
          const auto* const data_ptr = t_in_c.data_ptr<scalar_t>();
          auto* const out_data = packed_tensor.data_ptr<scalar_t>();
          const auto num_seq = lengths.size(0);
          const auto cell_size = t_in_c.numel() / t_in_c.size(0);
          TORCH_DSA_KERNEL_LAUNCH(
              (pack_segments_cuda_v2_kernel<index_t, scalar_t>),
              cuda_calc_xblock_count(num_seq * max_length * cell_size, 128),
              128,
              0,
              at::cuda::getCurrentCUDAStream(),
              data_ptr,
              t_in_c.size(0),
              lengths_data,
              lps_data,
              max_length,
              num_seq,
              cell_size,
              pad_minf ? -std::numeric_limits<scalar_t>::infinity()
                       : static_cast<scalar_t>(0),
              presence_mask_data,
              out_data);
        });
  });

  return {packed_tensor, presence_mask};
}

} // namespace fbgemm_gpu
