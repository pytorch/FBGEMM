/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {
namespace {

constexpr int32_t kLegacyThreadsPerBlock = 128;

// Each X block recomputes lengths[0:seq], so prefix work grows quadratically.
// 7168 is the largest tested all-shape win across H100 and MI350.
constexpr int32_t kFusedPrefixMaxNumSeq = 7168;
constexpr int32_t kFusedPrefixMinWarpsPerBlock = 2;
// TPB512 was more consistent; TPB1024 regressed 7/15 H100 cases.
constexpr int32_t kFusedPrefixMaxThreadsPerBlock = 512;
// Eight loads/thread tracked the best fixed TPBs in the H100 sweep.
constexpr int32_t kFusedPrefixMinLoadsPerThread = 8;

int32_t fused_prefix_threads_per_block(
    const int64_t loads_per_seq,
    const int64_t num_seq,
    const int64_t blocks_per_seq,
    const int32_t multiprocessor_count,
    const int32_t max_threads_per_multiprocessor,
    const int32_t warp_size) {
  const int32_t min_threads_per_block =
      kFusedPrefixMinWarpsPerBlock * warp_size;
  const int64_t total_blocks = num_seq * blocks_per_seq;
  const int64_t target_resident_threads =
      static_cast<int64_t>(multiprocessor_count) *
      max_threads_per_multiprocessor / 2;
  const int64_t target_threads_per_block =
      (target_resident_threads - 1) / total_blocks + 1;

  int32_t threads_per_block = min_threads_per_block;
  while (threads_per_block < kFusedPrefixMaxThreadsPerBlock &&
         threads_per_block < target_threads_per_block) {
    threads_per_block *= 2;
  }
  while (threads_per_block > min_threads_per_block &&
         loads_per_seq < blocks_per_seq * threads_per_block *
                 kFusedPrefixMinLoadsPerThread) {
    threads_per_block /= 2;
  }
  return threads_per_block;
}

} // namespace

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

  CUDA_KERNEL_LOOP_TYPE(i, num_seq * max_length * cell_size, int64_t) {
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

// Fuses the exclusive prefix sum of lengths into the packing kernel. Unlike the
// legacy path, which materializes the prefix sum with a separate kernel, each
// block reduces lengths[0:seq] before directly copying or padding its sequence.
// This avoids an extra launch and temporary tensor at the cost of redundant
// prefix work across blocks, making it most useful when num_seq is small.
template <
    int32_t kThreadsPerBlock,
    typename Length_T,
    typename Data_T,
    typename Vec_T>
__global__ void pack_segments_fused_prefix_cuda_kernel(
    const Data_T* const data_ptr,
    const int64_t data_size_0,
    const Length_T* const lengths_ptr,
    const int64_t max_length,
    const int64_t num_seq,
    const int64_t cell_size,
    const Vec_T padding,
    Data_T* const out_ptr,
    TORCH_DSA_KERNEL_ARGS) {
  static_assert(
      sizeof(Vec_T) % sizeof(Data_T) == 0 ||
      sizeof(Data_T) % sizeof(Vec_T) == 0);

  using BlockReduce =
      FBGEMM_GPU_CUB_NS_PREFIX cub::BlockReduce<Length_T, kThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ Length_T input_start;

  const int64_t seq = blockIdx.y;
  Length_T length = 0;
  for (int64_t i = threadIdx.x; i < seq; i += kThreadsPerBlock) {
    length += lengths_ptr[i];
  }
  const Length_T prefix = BlockReduce(temp_storage).Sum(length);

  if (threadIdx.x == 0) {
    input_start = prefix;
  }

  __syncthreads();
  const Length_T seq_length = lengths_ptr[seq];
  if (seq == num_seq - 1) {
    CUDA_KERNEL_ASSERT(
        data_size_0 == input_start + seq_length &&
        "data first dimension must equal the sum of segment lengths");
  }

  const int64_t bytes_per_seq = max_length * cell_size * sizeof(Data_T);
  const int64_t valid_bytes = seq_length * cell_size * sizeof(Data_T);
  const int64_t input_byte_offset = input_start * cell_size * sizeof(Data_T);
  const int64_t output_byte_offset = seq * bytes_per_seq;
  const int64_t loads_per_seq = bytes_per_seq / sizeof(Vec_T);
  const int64_t valid_loads = valid_bytes / sizeof(Vec_T);
  const auto* const input_vec = reinterpret_cast<const Vec_T*>(
      reinterpret_cast<const char*>(data_ptr) + input_byte_offset);
  auto* const output_vec = reinterpret_cast<Vec_T*>(
      reinterpret_cast<char*>(out_ptr) + output_byte_offset);

  // Promote the grid indices before multiplication to avoid 32-bit overflow.
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < loads_per_seq;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    output_vec[i] = i >= valid_loads ? padding : input_vec[i];
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
/// TODO: Add an API that accepts precomputed offsets so callers can reuse them
/// across operators instead of recomputing a prefix sum.
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

    if (lengths_c.size(0) == 0 || t_in_c.size(0) == 0) {
      packed_tensor = at::zeros(shape, t_in_c.options());
      return;
    }

    const auto num_seq = lengths_c.size(0);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto* device_properties =
        at::cuda::getDeviceProperties(stream.device_index());
    const bool use_fused_prefix_kernel = num_seq <= kFusedPrefixMaxNumSeq;
    const int64_t blocks_per_seq_for_device_coverage =
        (device_properties->multiProcessorCount - 1) / num_seq + 1;

    packed_tensor = at::empty(shape, t_in_c.options());
    const auto cell_size = t_in_c.numel() / t_in_c.size(0);

    Tensor lengths_prefix_sum;
    if (!use_fused_prefix_kernel) {
      lengths_prefix_sum =
          fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths_c);
    }

    FBGEMM_DISPATCH_ALL_TYPES(
        t_in_c.scalar_type(), "pack_segments_cuda-packing", [&] {
          const auto* const data_ptr = t_in_c.const_data_ptr<scalar_t>();
          auto* const out_data = packed_tensor.mutable_data_ptr<scalar_t>();

          if (use_fused_prefix_kernel) {
            const int64_t elements_per_seq = max_length * cell_size;
            const auto vector_layout_compatible =
                [&](const int64_t vector_bytes) {
                  if (vector_bytes <= sizeof(scalar_t) ||
                      vector_bytes % sizeof(scalar_t) != 0) {
                    return false;
                  }
                  const int64_t elements_per_load =
                      vector_bytes / sizeof(scalar_t);
                  return reinterpret_cast<uintptr_t>(data_ptr) % vector_bytes ==
                      0 &&
                      reinterpret_cast<uintptr_t>(out_data) % vector_bytes ==
                      0 &&
                      cell_size % elements_per_load == 0;
                };
            const int64_t preferred_vector_bytes =
                vector_layout_compatible(sizeof(uint4))      ? sizeof(uint4)
                : vector_layout_compatible(sizeof(uint2))    ? sizeof(uint2)
                : vector_layout_compatible(sizeof(uint32_t)) ? sizeof(uint32_t)
                                                             : sizeof(scalar_t);
            const int64_t preferred_elements_per_load =
                preferred_vector_bytes / sizeof(scalar_t);
            const int64_t preferred_loads_per_seq =
                elements_per_seq / preferred_elements_per_load;
            const int32_t threads_per_block = fused_prefix_threads_per_block(
                preferred_loads_per_seq,
                num_seq,
                blocks_per_seq_for_device_coverage,
                device_properties->multiProcessorCount,
                device_properties->maxThreadsPerMultiProcessor,
                device_properties->warpSize);
            const auto launch_fused_prefix_kernel = [&](auto threads_constant) {
              constexpr int32_t kThreadsPerBlock =
                  decltype(threads_constant)::value;
              const int64_t target_work_items =
                  blocks_per_seq_for_device_coverage * kThreadsPerBlock;

              const auto can_vectorize = [&](const int64_t vector_bytes) {
                if (!vector_layout_compatible(vector_bytes)) {
                  return false;
                }
                const int64_t elements_per_load =
                    vector_bytes / sizeof(scalar_t);
                return elements_per_seq / elements_per_load >=
                    target_work_items;
              };

              const int64_t vector_bytes = can_vectorize(sizeof(uint4))
                  ? sizeof(uint4)
                  : can_vectorize(sizeof(uint2))    ? sizeof(uint2)
                  : can_vectorize(sizeof(uint32_t)) ? sizeof(uint32_t)
                                                    : 0;
              const int64_t elements_per_load =
                  vector_bytes == 0 ? 1 : vector_bytes / sizeof(scalar_t);
              const int64_t loads_per_seq =
                  elements_per_seq / elements_per_load;
              const auto blocks_x = std::min<uint32_t>(
                  static_cast<uint32_t>(blocks_per_seq_for_device_coverage),
                  utils::cuda::cap_grid_dim_x_from_workload(
                      loads_per_seq, kThreadsPerBlock, stream));
              const dim3 blocks(blocks_x, static_cast<uint32_t>(num_seq));

              const auto launch_vector_type = [&](auto zero) {
                using Vec_T = decltype(zero);
                FBGEMM_LAUNCH_DSA_KERNEL(
                    (pack_segments_fused_prefix_cuda_kernel<
                        kThreadsPerBlock,
                        index_t,
                        scalar_t,
                        Vec_T>),
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
                    zero,
                    out_data);
              };

              if (vector_bytes == sizeof(uint4)) {
                launch_vector_type(uint4{});
              } else if (vector_bytes == sizeof(uint2)) {
                launch_vector_type(uint2{});
              } else if constexpr (sizeof(scalar_t) < sizeof(uint32_t)) {
                if (vector_bytes == sizeof(uint32_t)) {
                  launch_vector_type(uint32_t{});
                } else {
                  launch_vector_type(scalar_t{});
                }
              } else {
                launch_vector_type(scalar_t{});
              }
            };

            switch (threads_per_block) {
              case 64:
                launch_fused_prefix_kernel(
                    std::integral_constant<int32_t, 64>{});
                break;
              case 128:
                launch_fused_prefix_kernel(
                    std::integral_constant<int32_t, 128>{});
                break;
              case 256:
                launch_fused_prefix_kernel(
                    std::integral_constant<int32_t, 256>{});
                break;
              case 512:
                launch_fused_prefix_kernel(
                    std::integral_constant<int32_t, 512>{});
                break;
              default:
                TORCH_CHECK(
                    false,
                    "Unsupported fused-prefix threads per block: ",
                    threads_per_block);
            }
          } else {
            auto* const lps_data =
                lengths_prefix_sum.mutable_data_ptr<index_t>();

            // HIP enforces a hard limit of 2^32 total threads per launch
            // (unlike CUDA, which silently wraps). pack_segments_cuda_kernel
            // uses CUDA_KERNEL_LOOP, which already grid-strides, so capping is
            // correctness-preserving.
            // See: https://github.com/ROCm/hip/issues/2253
            const auto blocks = utils::cuda::cap_grid_dim_x_from_workload(
                num_seq * max_length * cell_size,
                kLegacyThreadsPerBlock,
                stream);

            FBGEMM_LAUNCH_DSA_KERNEL(
                (pack_segments_cuda_kernel<index_t, scalar_t>),
                blocks,
                kLegacyThreadsPerBlock,
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
