/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>

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
  CUDA_KERNEL_ASSERT(
      data_size_0 == lengths_cum_sum[num_seq - 1] + lengths_ptr[num_seq - 1] &&
      "data first dimension must equal the sum of segment lengths");

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

template <int32_t kThreadsPerBlock, typename Length_T, typename Data_T>
__global__ void pack_segments_small_n_cuda_kernel(
    const Data_T* const data_ptr,
    const int64_t data_size_0,
    const Length_T* const lengths_ptr,
    const Length_T max_length,
    const int64_t num_seq,
    const int64_t cell_size,
    const Data_T padding,
    Data_T* const out_ptr,
    TORCH_DSA_KERNEL_ARGS) {
  using BlockReduce =
      FBGEMM_GPU_CUB_NS_PREFIX cub::BlockReduce<Length_T, kThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ Length_T input_start;

  const int64_t seq = blockIdx.y;
  const Length_T length =
      threadIdx.x < seq ? lengths_ptr[threadIdx.x] : static_cast<Length_T>(0);
  const Length_T prefix = BlockReduce(temp_storage).Sum(length);

  if (threadIdx.x == 0) {
    input_start = prefix;
  }

  __syncthreads();
  if (seq == num_seq - 1) {
    CUDA_KERNEL_ASSERT(
        data_size_0 == input_start + lengths_ptr[seq] &&
        "data first dimension must equal the sum of segment lengths");
  }

  const int64_t elements_per_seq = static_cast<int64_t>(max_length) * cell_size;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements_per_seq;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t cell = i / cell_size;
    const int64_t offset = i % cell_size;
    const int64_t output_idx = seq * elements_per_seq + i;

    if (cell >= lengths_ptr[seq]) {
      out_ptr[output_idx] = padding;
    } else {
      const int64_t input_idx =
          (static_cast<int64_t>(input_start) + cell) * cell_size + offset;
      out_ptr[output_idx] = data_ptr[input_idx];
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
  CUDA_KERNEL_ASSERT(
      data_size_0 == lengths_cum_sum[num_seq - 1] + lengths_ptr[num_seq - 1] &&
      "data first dimension must equal the sum of segment lengths");

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
          t_in.dtype() == at::ScalarType::BFloat16 ||
          t_in.dtype() == at::ScalarType::Int,
      "t_in must be of type float or double or half, bfloat16 or int");
  TORCH_CHECK_GT(max_length, 0);

  CUDA_DEVICE_GUARD(t_in);

  const auto t_in_c = t_in.contiguous();
  const auto lengths_c = lengths.contiguous();

  Tensor packed_tensor;

  AT_DISPATCH_INDEX_TYPES(lengths_c.scalar_type(), "pack_segments_cuda", [&] {
    const auto* const lengths_data = lengths_c.const_data_ptr<index_t>();

    // Shape of output is batch_size x max_len x ...
    auto shape = t_in_c.sizes().vec(); // Get copy of current shape
    shape[0] = max_length; // Set first element to max_len
    shape.insert(
        shape.begin(), lengths_c.numel()); // Insert batch size at beginning

    if (lengths_c.size(0) == 0) {
      packed_tensor = at::zeros(shape, t_in_c.options());
      return; // Return empty output (with the proper shape)
    }

    constexpr int64_t kSmallNumSeqThreshold = 128;
    constexpr int32_t kThreadsPerBlock = 128;
    constexpr int64_t kTargetElementsPerBlock = 1600;
    constexpr int32_t kMaxBlocksPerSeq = 256;

    const auto num_seq = lengths_c.size(0);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto* device_properties =
        at::cuda::getDeviceProperties(stream.device_index());
#ifdef USE_ROCM
    static const bool enable_small_n_kernel = [] {
      const char* value = std::getenv("FBGEMM_ENABLE_PACK_SEGMENTS_SMALL_N");
      return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    const bool use_small_n_kernel = enable_small_n_kernel &&
        num_seq <= kSmallNumSeqThreshold &&
        std::strncmp(device_properties->gcnArchName, "gfx950", 6) == 0;
#else
    const bool use_small_n_kernel = false;
#endif

    if (t_in_c.size(0) == 0) {
      packed_tensor = at::zeros(shape, t_in_c.options());
      return; // Return empty output (with the proper shape)
    }
    packed_tensor = at::empty(shape, t_in_c.options());
    const auto cell_size = t_in_c.numel() / t_in_c.size(0);

    Tensor lengths_prefix_sum;
    if (!use_small_n_kernel) {
      lengths_prefix_sum =
          fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths_c);
    }

    FBGEMM_DISPATCH_ALL_TYPES(
        t_in_c.scalar_type(), "pack_segments_cuda-packing", [&] {
          const auto* const data_ptr = t_in_c.const_data_ptr<scalar_t>();
          auto* const out_data = packed_tensor.mutable_data_ptr<scalar_t>();

          if (use_small_n_kernel) {
            const int64_t elements_per_seq = max_length * cell_size;
            // Bound redundant prefix work while keeping enough packing blocks.
            const int64_t blocks_for_work = elements_per_seq == 0
                ? 1
                : (elements_per_seq - 1) / kTargetElementsPerBlock + 1;
            const int64_t blocks_for_occupancy =
                (device_properties->multiProcessorCount - 1) / num_seq + 1;
            const int64_t desired_blocks =
                std::max(blocks_for_work, blocks_for_occupancy);
            const auto blocks_x = std::min<uint32_t>(
                static_cast<uint32_t>(
                    std::min<int64_t>(desired_blocks, kMaxBlocksPerSeq)),
                utils::cuda::cap_grid_dim_x_from_workload(
                    elements_per_seq, kThreadsPerBlock, stream));
            const dim3 blocks(blocks_x, static_cast<uint32_t>(num_seq));

            FBGEMM_LAUNCH_DSA_KERNEL(
                (pack_segments_small_n_cuda_kernel<
                    kThreadsPerBlock,
                    index_t,
                    scalar_t>),
                blocks,
                kThreadsPerBlock,
                0,
                stream,
                data_ptr,
                t_in_c.size(0),
                lengths_data,
                max_length,
                num_seq,
                cell_size,
                static_cast<scalar_t>(0),
                out_data);
          } else {
            auto* const lps_data =
                lengths_prefix_sum.mutable_data_ptr<index_t>();

            // HIP enforces a hard limit of 2^32 total threads per launch
            // (unlike CUDA, which silently wraps). pack_segments_cuda_kernel
            // uses CUDA_KERNEL_LOOP, which already grid-strides, so capping is
            // correctness-preserving.
            // See: https://github.com/ROCm/hip/issues/2253
            const auto blocks = utils::cuda::cap_grid_dim_x_from_workload(
                num_seq * max_length * cell_size, kThreadsPerBlock, stream);

            FBGEMM_LAUNCH_DSA_KERNEL(
                (pack_segments_cuda_kernel<index_t, scalar_t>),
                blocks,
                kThreadsPerBlock,
                0,
                stream,
                data_ptr,
                t_in_c.size(0),
                lengths_data,
                lps_data,
                max_length,
                num_seq,
                cell_size,
                static_cast<scalar_t>(0),
                out_data);
          }
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
  const auto lengths_c = lengths.contiguous();

  Tensor packed_tensor;
  std::optional<Tensor> presence_mask;

  AT_DISPATCH_INDEX_TYPES(lengths_c.scalar_type(), "pack_segments_cuda", [&] {
    const auto* const lengths_data = lengths_c.const_data_ptr<index_t>();

    // Shape of output is batch_size x max_len x ...
    auto shape = t_in_c.sizes().vec(); // Get copy of current shape
    shape[0] = max_length; // Set first element to max_len
    shape.insert(
        shape.begin(), lengths_c.numel()); // Insert batch size at beginning
    packed_tensor = at::zeros(shape, t_in_c.options());

    if (pad_minf) {
      packed_tensor = at::full(
          shape, -std::numeric_limits<double>::infinity(), t_in_c.options());
    } else {
      packed_tensor = at::zeros(shape, t_in_c.options());
    }

    bool* presence_mask_data = nullptr;
    if (return_presence_mask) {
      presence_mask = at::zeros(
          {lengths_c.numel(), max_length}, t_in_c.options().dtype(at::kBool));
      presence_mask_data = presence_mask->mutable_data_ptr<bool>();
    }

    if (t_in_c.size(0) == 0 || lengths_c.size(0) == 0) {
      return; // Return empty output (with the proper shape)
    }

    auto lengths_prefix_sum =
        fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths_c);
    auto lps_data = lengths_prefix_sum.data_ptr<index_t>();

    FBGEMM_DISPATCH_ALL_TYPES(
        t_in_c.scalar_type(), "pack_segments_cuda-packing", [&] {
          const auto* const data_ptr = t_in_c.const_data_ptr<scalar_t>();
          auto* const out_data = packed_tensor.mutable_data_ptr<scalar_t>();
          const auto num_seq = lengths_c.size(0);
          const auto cell_size = t_in_c.numel() / t_in_c.size(0);

          // HIP enforces a hard limit of 2^32 total threads per launch
          // (unlike CUDA, which silently wraps). pack_segments_cuda_v2_kernel
          // uses CUDA_KERNEL_LOOP_TYPE, which already grid-strides, so capping
          // is correctness-preserving.
          // See: https://github.com/ROCm/hip/issues/2253
          const auto blocks = utils::cuda::cap_grid_dim_x_from_workload(
              num_seq * max_length * cell_size,
              128,
              at::cuda::getCurrentCUDAStream());

          FBGEMM_LAUNCH_DSA_KERNEL(
              (pack_segments_cuda_v2_kernel<index_t, scalar_t>),
              blocks,
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
