/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm_gpu/sparse_ops.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/library.h>

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include "cub/device/device_scan.cuh"
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

std::tuple<uint32_t, uint32_t, uint32_t> calc_offsets_range_thread_block(
    const int64_t output_size,
    const int64_t num_seq) {
  uint32_t threads_per_block;
  uint32_t vector_size;
  if (output_size / num_seq < 2) {
    threads_per_block = 512;
    vector_size = 2;
  } else if (output_size / num_seq < 4) {
    threads_per_block = 512;
    vector_size = 4;
  } else if (output_size / num_seq < 64) {
    threads_per_block = 512;
    vector_size = 8;
  } else if (output_size / num_seq < 128) {
    threads_per_block = 512;
    vector_size = 16;
  } else {
    threads_per_block = 512;
    vector_size = 32;
  }
  uint32_t rows_per_block = threads_per_block / vector_size;
  const auto num_blocks = cuda_calc_xblock_count(num_seq, rows_per_block);

  return std::make_tuple(num_blocks, rows_per_block, vector_size);
}

// Kernel for calculating the offsets ranges
template <typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void _offsets_range_cuda_kernel(
    int64_t N,
    int64_t range_size,
    const scalar_t* __restrict__ offsets_data,
    scalar_t* __restrict__ range_data) {
  int start_row_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int row_idx = start_row_idx; row_idx < N; row_idx += stride) {
    scalar_t row_start = offsets_data[row_idx];
    scalar_t row_end =
        (row_idx < N - 1 ? offsets_data[row_idx + 1] : range_size);
    if (blockDim.x == 32) {
      scalar_t i = row_start - (row_start & 31) + threadIdx.x;
      // unaligned part
      if (i >= row_start && i < row_end) {
        range_data[i] = i - row_start;
      }
      // aligned part
      for (i += 32; i < row_end; i += 32) {
        range_data[i] = i - row_start;
      }
    } else {
      for (scalar_t i = row_start + threadIdx.x; i < row_end; i += blockDim.x) {
        range_data[i] = i - row_start;
      }
    }
  }
}

Tensor offsets_range_cuda(const Tensor& offsets, int64_t range_size) {
  TENSOR_ON_CUDA_GPU(offsets);
  TENSOR_NDIM_EQUALS(offsets, 1);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(offsets.get_device());

  auto offsets_arg = at::TensorArg(offsets, "offsets", 1);
  checkScalarTypes("_offsets_range_cuda", offsets_arg, {at::kLong, at::kInt});
  auto range = at::empty(range_size, offsets.options());
  if (range_size == 0) {
    return range;
  }
  auto offsets_contig = offsets.contiguous();
  int64_t N = offsets_contig.numel();

  uint32_t vector_size;
  uint32_t rows_per_block;
  uint32_t num_blocks;
  std::tie(num_blocks, rows_per_block, vector_size) =
      calc_offsets_range_thread_block(range_size, N);
  dim3 threads(vector_size, rows_per_block);
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig.scalar_type(), "offsets_range_kernel", [&] {
        _offsets_range_cuda_kernel<index_t>
            <<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                range_size,
                offsets_contig.data_ptr<index_t>(),
                range.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return range;
}

// Kernel for calculating the segmented sum for sparse matrix with CSR format.
// See https://moderngpu.github.io/segreduce.html
template <typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void _segment_sum_csr_cuda_kernel(
    int num_segments,
    int batch_size,
    const int* csr_seg_data,
    const scalar_t* values_data,
    scalar_t* output_data) {
  typedef FBGEMM_GPU_CUB_NS_PREFIX cub::BlockReduce<scalar_t, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int seg_start = csr_seg_data[blockIdx.x] * batch_size;
  int seg_end = csr_seg_data[blockIdx.x + 1] * batch_size;
  scalar_t sum = 0;
  for (int i = seg_start; i < seg_end; i += blockDim.x) {
    scalar_t thread_data;
    if (threadIdx.x < seg_end - i) {
      thread_data = values_data[i + threadIdx.x];
    }
    scalar_t aggregate =
        BlockReduce(temp_storage).Sum(thread_data, seg_end - i);
    __syncthreads();
    if (threadIdx.x == 0) {
      sum += aggregate;
    }
  }
  if (threadIdx.x == 0) {
    output_data[blockIdx.x] = sum;
  }
}

Tensor segment_sum_csr_cuda(
    const int64_t batch_size,
    const Tensor& csr_seg,
    const Tensor& values) {
  TENSOR_ON_CUDA_GPU(csr_seg);
  TENSOR_ON_CUDA_GPU(values);

  TENSORS_ON_SAME_DEVICE(csr_seg, values);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  auto output = at::empty(csr_seg.numel() - 1, values.options());
  constexpr uint32_t threads_per_block = 256;
  const uint32_t num_blocks = csr_seg.numel() - 1;
  AT_DISPATCH_ALL_TYPES(values.type(), "_segment_sum_csr_cuda", [&] {
    _segment_sum_csr_cuda_kernel<scalar_t>
        <<<num_blocks,
           threads_per_block,
           0,
           at::cuda::getCurrentCUDAStream()>>>(
            csr_seg.numel() - 1,
            batch_size,
            csr_seg.data_ptr<int>(),
            values.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return output;
}

Tensor asynchronous_inclusive_cumsum_gpu(const Tensor& t_in) {
  TENSOR_ON_CUDA_GPU(t_in);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  auto t_out = at::empty_like(t_in);
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      t_in.options().dtype(at::kByte));
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper2", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  return t_out;
}

Tensor asynchronous_exclusive_cumsum_gpu(const Tensor& t_in) {
  TENSOR_ON_CUDA_GPU(t_in);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  auto t_out = at::empty_like(t_in);
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_exclusive_sum_wrapper1", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::ExclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      t_in.options().dtype(at::kByte));
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_exclusive_sum_wrapper2", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::ExclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>(),
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  return t_out;
}

Tensor asynchronous_complete_cumsum_gpu(const Tensor& t_in) {
  TENSOR_ON_CUDA_GPU(t_in);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(t_in.dim() == 1);
  auto t_out = at::empty({t_in.numel() + 1}, t_in.options());
  t_out[0].zero_();
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            nullptr,
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  auto temp_storage = at::empty(
      {static_cast<int64_t>(temp_storage_bytes)},
      t_in.options().dtype(at::kByte));
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper2", [&] {
        AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            t_in.data_ptr<index_t>(),
            t_out.data_ptr<index_t>() + 1,
            t_in.numel(),
            at::cuda::getCurrentCUDAStream()));
      });
  return t_out;
}

// Kernel for permuting the indices and weights. Used for permutation of sparse
// data
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
__global__ __launch_bounds__(kMaxThreads) void permute_2D_data_kernel(
    int32_t len,
    int32_t T,
    int32_t B,
    const indices_t* __restrict__ indices,
    const weights_t* __restrict__ weights,
    const int32_t* __restrict__ permute,
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    indices_t* __restrict__ permuted_indices,
    weights_t* __restrict__ permuted_weights) {
  int32_t b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    offsets_t output_start = output_offsets[b_t];
    offsets_t segment_length;
    if (b_t == B * T - 1) {
      segment_length = len - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    offsets_t input_start = input_offsets[permute[t] * B + b];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        permuted_weights[output_start + i] = weights[input_start + i];
      }
    }
  }
}

// Kernel for permuting the lengths. Used for permutation of sparse features.
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void permute_2D_lengths_kernel(
    int32_t T,
    int32_t B,
    const index_t* __restrict__ lengths,
    const int32_t* __restrict__ permute,
    index_t* __restrict__ permuted_lengths) {
  CUDA_KERNEL_LOOP(b_t, B * T) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    permuted_lengths[b_t] = lengths[permute[t] * B + b];
  }
}

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_2D_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum) {
  TENSOR_ON_CUDA_GPU(permute);
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(weights);
  TORCH_CHECK(lengths.dim() == 2);

  TENSORS_ON_SAME_DEVICE(permute, lengths);
  TENSORS_ON_SAME_DEVICE(permute, indices);
  TENSORS_ON_SAME_DEVICE(permute, weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the data to permute over can be less or more with or without
  // repetitions
  const auto T = permute.numel();
  const auto B = lengths.size(1);

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;

  permuted_lengths = at::empty({T, B}, lengths.options());

  constexpr int32_t threads_1 = 256;
  const auto blocks_1 = cuda_calc_xblock_count(B * T, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_2D_lengths_kernel", [&] {
        permute_2D_lengths_kernel<index_t>
            <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
                T,
                B,
                lengths_contig.data_ptr<index_t>(),
                permute.data_ptr<int32_t>(),
                permuted_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // convert lengths to offsets
  const auto input_offsets = asynchronous_exclusive_cumsum_gpu(lengths_contig);
  const auto output_offsets =
      asynchronous_exclusive_cumsum_gpu(permuted_lengths);
  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size = permuted_lengths.sum().item<int64_t>();
  }

  constexpr int32_t BT_blocks = 32;
  dim3 threads_2(32, BT_blocks);
  const auto blocks_2 = cuda_calc_xblock_count(B * T, BT_blocks);
  permuted_indices = at::empty(permuted_indices_size, indices.options());

  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_2D_data_kernel_1", [&] {
        using offsets_t = index_t;
        AT_DISPATCH_ALL_TYPES_AND(
            at::ScalarType::Half,
            indices.scalar_type(),
            "permute_2D_data_kernel_2",
            [&] {
              using indices_t = scalar_t;
              if (weights.has_value()) {
                const Tensor weights_value = weights.value();
                const auto weights_value_contig = weights_value.contiguous();
                permuted_weights =
                    at::empty(permuted_indices_size, weights_value.options());
                AT_DISPATCH_ALL_TYPES_AND(
                    at::ScalarType::Half,
                    weights_value.scalar_type(),
                    "permute_2D_data_kernel_3",
                    [&] {
                      using weights_t = scalar_t;
                      permute_2D_data_kernel<
                          true,
                          offsets_t,
                          indices_t,
                          weights_t>
                          <<<blocks_2,
                             threads_2,
                             0,
                             at::cuda::getCurrentCUDAStream()>>>(
                              permuted_indices_size,
                              T,
                              B,
                              indices_contig.data_ptr<indices_t>(),
                              weights_value_contig.data_ptr<weights_t>(),
                              permute_contig.data_ptr<int32_t>(),
                              input_offsets.data_ptr<offsets_t>(),
                              output_offsets.data_ptr<offsets_t>(),
                              permuted_indices.data_ptr<indices_t>(),
                              permuted_weights.data_ptr<weights_t>());
                      C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }); // for each weights_t
              } else {
                permute_2D_data_kernel<
                    false,
                    offsets_t,
                    indices_t,
                    std::nullptr_t>
                    <<<blocks_2,
                       threads_2,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        permuted_indices_size,
                        T,
                        B,
                        indices_contig.data_ptr<indices_t>(),
                        nullptr,
                        permute_contig.data_ptr<int32_t>(),
                        input_offsets.data_ptr<offsets_t>(),
                        output_offsets.data_ptr<offsets_t>(),
                        permuted_indices.data_ptr<indices_t>(),
                        nullptr);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
            }); // for each indices_t
      }); // for each offsets_t
  return {permuted_lengths, permuted_indices, permuted_weights};
}

// Kernel for permuting 1D lengths. Used for permutation of sparse features.
template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void permute_1D_lengths_kernel(
    const index_t* __restrict__ lengths,
    int32_t permuted_lengths_size,
    const int32_t* __restrict__ permute,
    index_t* __restrict__ permuted_lengths) {
  CUDA_KERNEL_LOOP(i, permuted_lengths_size) {
    permuted_lengths[i] = lengths[permute[i]];
  }
}

// Kernel for permuting the indices and weights. Used for permutation of sparse
// data
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
__global__ __launch_bounds__(kMaxThreads) void permute_1D_data_kernel(
    int32_t permuted_indices_size,
    int32_t permuted_lengths_size,
    const indices_t* __restrict__ indices,
    const weights_t* __restrict__ weights,
    const int32_t* __restrict__ permute,
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    indices_t* __restrict__ permuted_indices,
    weights_t* __restrict__ permuted_weights) {
  int32_t b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < permuted_lengths_size; b_t += stride) {
    offsets_t output_start = output_offsets[b_t];
    offsets_t segment_length;
    if (b_t == permuted_lengths_size - 1) {
      segment_length = permuted_indices_size - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    offsets_t input_start = input_offsets[permute[b_t]];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        permuted_weights[output_start + i] = weights[input_start + i];
      }
    }
  }
}

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_1D_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum) {
  TENSOR_ON_CUDA_GPU(permute);
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(weights);

  TENSORS_ON_SAME_DEVICE(permute, lengths);
  TENSORS_ON_SAME_DEVICE(permute, indices);
  TENSORS_ON_SAME_DEVICE(permute, weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the data to permute over can be less or more with or without
  // repetitions

  const auto lengths_size = lengths.numel();

  const auto permuted_lengths_size = permute.numel();
  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;
  permuted_lengths = at::empty({permuted_lengths_size}, lengths.options());

  constexpr int32_t threads_1 = kMaxThreads;
  const auto blocks_1 =
      cuda_calc_xblock_count(permuted_lengths_size, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_1D_lengths_kernel", [&] {
        permute_1D_lengths_kernel<index_t>
            <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
                lengths_contig.data_ptr<index_t>(),
                permuted_lengths_size,
                permute.data_ptr<int32_t>(),
                permuted_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // convert lengths to offsets
  const auto input_offsets = asynchronous_exclusive_cumsum_gpu(lengths_contig);
  const auto output_offsets =
      asynchronous_exclusive_cumsum_gpu(permuted_lengths);
  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size = permuted_lengths.sum().item<int64_t>();
  }

  constexpr int32_t BT_blocks = 32;
  dim3 threads_2(32, BT_blocks);
  const auto blocks_2 =
      cuda_calc_xblock_count(permuted_lengths_size, BT_blocks);
  permuted_indices = at::empty(permuted_indices_size, indices.options());

  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_1D_data_kernel_1", [&] {
        using offsets_t = index_t;
        AT_DISPATCH_ALL_TYPES_AND(
            at::ScalarType::Half,
            indices.scalar_type(),
            "permute_1D_data_kernel_2",
            [&] {
              using indices_t = scalar_t;
              if (weights.has_value()) {
                const Tensor weights_value = weights.value();
                const auto weights_value_contig = weights_value.contiguous();
                permuted_weights =
                    at::empty(permuted_indices_size, weights_value.options());
                AT_DISPATCH_ALL_TYPES_AND(
                    at::ScalarType::Half,
                    weights_value.scalar_type(),
                    "permute_1D_data_kernel_3",
                    [&] {
                      using weights_t = scalar_t;
                      permute_1D_data_kernel<
                          true,
                          offsets_t,
                          indices_t,
                          weights_t>
                          <<<blocks_2,
                             threads_2,
                             0,
                             at::cuda::getCurrentCUDAStream()>>>(
                              permuted_indices_size,
                              permuted_lengths_size,
                              indices_contig.data_ptr<indices_t>(),
                              weights_value_contig.data_ptr<weights_t>(),
                              permute_contig.data_ptr<int32_t>(),
                              input_offsets.data_ptr<offsets_t>(),
                              output_offsets.data_ptr<offsets_t>(),
                              permuted_indices.data_ptr<indices_t>(),
                              permuted_weights.data_ptr<weights_t>());
                      C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }); // for each weights_t
              } else {
                permute_1D_data_kernel<
                    false,
                    offsets_t,
                    indices_t,
                    std::nullptr_t>
                    <<<blocks_2,
                       threads_2,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        permuted_indices_size,
                        permuted_lengths_size,
                        indices_contig.data_ptr<indices_t>(),
                        nullptr,
                        permute_contig.data_ptr<int32_t>(),
                        input_offsets.data_ptr<offsets_t>(),
                        output_offsets.data_ptr<offsets_t>(),
                        permuted_indices.data_ptr<indices_t>(),
                        nullptr);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
            }); // for each indices_t
      }); // for each offsets_t

  return {permuted_lengths, permuted_indices, permuted_weights};
}

// Kernel for generate 1D data permute from dimension permute index.
// Used for permutation of sparse features.
template <typename index_t, typename offsets_t>
__global__
__launch_bounds__(kMaxThreads) void expand_into_jagged_permute_kernel(
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    int32_t input_size,
    const index_t* __restrict__ permute,
    index_t* __restrict__ output_permute) {
  const int32_t t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int t = t_start; t < input_size; t += stride) {
    const offsets_t output_start = output_offsets[t];
    const offsets_t segment_length = output_offsets[t + 1] - output_offsets[t];
    const offsets_t input_start = input_offsets[permute[t]];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      output_permute[output_start + i] = input_start + i;
    }
  }
}

Tensor expand_into_jagged_permute_cuda(
    const Tensor& permute,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    int64_t output_size) {
  TENSOR_ON_CUDA_GPU(permute);
  TENSOR_ON_CUDA_GPU(input_offsets);
  TENSOR_ON_CUDA_GPU(output_offsets);

  TENSORS_ON_SAME_DEVICE(permute, input_offsets);
  TENSORS_ON_SAME_DEVICE(permute, output_offsets);
  TORCH_CHECK(permute.numel() > 0);
  TORCH_CHECK(permute.numel() == input_offsets.numel() - 1);
  TORCH_CHECK(permute.numel() == output_offsets.numel() - 1);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(permute.get_device());

  const auto permute_contig = permute.contiguous();
  const auto permute_size = permute.numel();

  Tensor output_permute = at::empty({output_size}, permute.options());

  // number of table per block
  constexpr int32_t T_blocks = kMaxThreads / kWarpSize;
  dim3 threads(kWarpSize, T_blocks);
  const auto blocks = cuda_calc_xblock_count(permute_size, T_blocks);
  AT_DISPATCH_INDEX_TYPES(
      permute.scalar_type(), "expand_into_jagged_permute_kernel", [&] {
        using offsets_t = index_t;
        expand_into_jagged_permute_kernel<index_t, offsets_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_offsets.data_ptr<offsets_t>(),
                output_offsets.data_ptr<offsets_t>(),
                permute_size,
                permute.data_ptr<index_t>(),
                output_permute.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output_permute;
}

// Kernel for bucketize lengths, with the Block distribution (vs. cyclic,
// block-cyclic distribution). Used for bucketize sparse feature, especially for
// checkpointing with row-wise partition (sparse_feature is partitioned
// continuously along the sparse dimension into my_size blocks)
template <typename offset_t, typename index_t>
__global__
__launch_bounds__(kMaxThreads) void _block_bucketize_sparse_features_cuda_kernel1(
    int32_t lengths_size,
    int32_t B,
    const index_t* __restrict__ block_sizes_data,
    int my_size,
    const offset_t* __restrict__ offsets_data,
    const index_t* __restrict__ indices_data,
    offset_t* __restrict__ new_lengths_data) {
  using uindex_t = std::make_unsigned_t<index_t>;
  CUDA_KERNEL_LOOP(b_t, lengths_size) {
    int32_t t = b_t / B;
    index_t blk_size = block_sizes_data[t];
    offset_t rowstart = (b_t == 0 ? 0 : offsets_data[b_t - 1]);
    offset_t rowend = offsets_data[b_t];
    for (index_t i = rowstart; i < rowend; ++i) {
      // We have use cases using none-hashed raw indices that can be either
      // negative or larger than embedding table hash_size (blk_size *
      // my_size). In cases of none-hashed indices we need to ensure
      // bucketization can distribute them into different ranks and within
      // range of blk_size, we expect the later embedding module to take care
      // of hashing indices calculation.
      uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      uindex_t p = idx < blk_size * my_size ? idx / blk_size : idx % my_size;
      new_lengths_data[p * lengths_size + b_t]++;
    }
  }
}

// Kernel for bucketize offsets, indices, and positional weights, with the Block
// distribution (vs. cyclic, block-cyclic distribution). Used for bucketize
// sparse feature, especially for checkpointing with row-wise partition
// (sparse_feature is partitioned continuously along the sparse dimension into
// my_size blocks)
template <
    bool sequence,
    bool has_weight,
    bool bucketize_pos,
    typename offset_t,
    typename index_t,
    typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void _block_bucketize_sparse_features_cuda_kernel2(
    int lengths_size,
    int32_t B,
    const index_t* __restrict__ block_sizes_data,
    int my_size,
    const offset_t* __restrict__ offsets_data,
    const index_t* __restrict__ indices_data,
    const scalar_t* __restrict__ weights_data,
    offset_t* __restrict__ new_offsets_data,
    index_t* __restrict__ new_indices_data,
    scalar_t* __restrict__ new_weights_data,
    index_t* __restrict__ new_pos_data,
    index_t* __restrict__ unbucketize_permute_data) {
  using uindex_t = std::make_unsigned_t<index_t>;
  using uoffset_t = std::make_unsigned_t<offset_t>;
  CUDA_KERNEL_LOOP(b_t, lengths_size) {
    int32_t t = b_t / B;
    index_t blk_size = block_sizes_data[t];
    offset_t rowstart = (b_t == 0 ? 0 : offsets_data[b_t - 1]);
    offset_t rowend = offsets_data[b_t];
    for (index_t i = rowstart; i < rowend; ++i) {
      // We have use cases using none-hashed raw indices that can be either
      // negative or larger than embedding table hash_size (blk_size *
      // my_size). In cases of none-hashed indices we need to ensure
      // bucketization can distribute them into different ranks and within
      // range of blk_size, we expect the later embedding module to take care
      // of hashing indices calculation.
      uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      uindex_t p = idx < blk_size * my_size ? idx / blk_size : idx % my_size;
      uindex_t new_idx =
          idx < blk_size * my_size ? idx % blk_size : idx / my_size;
      uoffset_t pos = new_offsets_data[p * lengths_size + b_t];
      new_indices_data[pos] = new_idx;
      new_offsets_data[p * lengths_size + b_t]++;
      if (sequence) {
        unbucketize_permute_data[i] = pos;
      }
      if (has_weight) {
        new_weights_data[pos] = weights_data[i];
      }
      if (bucketize_pos) {
        new_pos_data[pos] = i - rowstart;
      }
    }
  }
}

// This function partitions sparse features
// continuously along the sparse dimension into my_size blocks
std::tuple<
    Tensor,
    Tensor,
    c10::optional<Tensor>,
    c10::optional<Tensor>,
    c10::optional<Tensor>>
block_bucketize_sparse_features_cuda(
    Tensor lengths,
    Tensor indices,
    bool bucketize_pos,
    bool sequence,
    Tensor block_sizes,
    int64_t my_size,
    c10::optional<Tensor> weights) {
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(indices);
  TENSORS_ON_SAME_DEVICE(lengths, indices);
  TENSOR_ON_CUDA_GPU(weights);
  TENSORS_ON_SAME_DEVICE(lengths, weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lengths.get_device());
  // allocate tensors and buffers
  const int lengths_size = lengths.numel();
  const int T = block_sizes.numel();
  const int B = lengths_size / T;
  const int new_lengths_size = lengths_size * my_size;
  auto offsets = at::empty({lengths_size}, lengths.options());
  auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
  auto new_offsets = at::empty({new_lengths_size}, lengths.options());
  auto new_indices = at::empty_like(indices);
  auto lengths_contig = lengths.contiguous();
  auto indices_contig = indices.contiguous();
  auto offsets_contig = offsets.contiguous();
  Tensor new_weights;
  Tensor new_pos;
  Tensor unbucketize_permute;
  // count nonzeros
  offsets_contig = asynchronous_inclusive_cumsum_gpu(lengths);
  int threads_per_block = 256;
  int num_blocks = (lengths_size + threads_per_block - 1) / threads_per_block;
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig.scalar_type(),
      "_block_bucketize_sparse_features_cuda_kernel1",
      [&] {
        using offset_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            indices_contig.scalar_type(),
            "_block_bucketize_sparse_features_cuda_kernel2",
            [&] {
              _block_bucketize_sparse_features_cuda_kernel1<<<
                  num_blocks,
                  threads_per_block,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  lengths_size,
                  B,
                  block_sizes.data_ptr<index_t>(),
                  my_size,
                  offsets_contig.data_ptr<offset_t>(),
                  indices_contig.data_ptr<index_t>(),
                  new_lengths.data_ptr<offset_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });

  // bucketize nonzeros
  new_offsets = asynchronous_exclusive_cumsum_gpu(new_lengths);
  if (sequence) {
    const auto lengths_sum = indices.numel();
    unbucketize_permute = at::empty({lengths_sum}, indices.options());
    if (weights.has_value() & bucketize_pos) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                [&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      [&] {
                        _block_bucketize_sparse_features_cuda_kernel2<
                            true,
                            true,
                            true,
                            offset_t,
                            index_t,
                            scalar_t>
                            <<<num_blocks,
                               threads_per_block,
                               0,
                               at::cuda::getCurrentCUDAStream()>>>(
                                lengths_size,
                                B,
                                block_sizes.data_ptr<index_t>(),
                                my_size,
                                offsets_contig.data_ptr<offset_t>(),
                                indices_contig.data_ptr<index_t>(),
                                weights_value_contig.data_ptr<scalar_t>(),
                                new_offsets.data_ptr<offset_t>(),
                                new_indices.data_ptr<index_t>(),
                                new_weights.data_ptr<scalar_t>(),
                                new_pos.data_ptr<index_t>(),
                                unbucketize_permute.data_ptr<index_t>());
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      });
                });
          });
    } else if (weights.has_value()) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                [&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      [&] {
                        _block_bucketize_sparse_features_cuda_kernel2<
                            true,
                            true,
                            false,
                            offset_t,
                            index_t,
                            scalar_t>
                            <<<num_blocks,
                               threads_per_block,
                               0,
                               at::cuda::getCurrentCUDAStream()>>>(
                                lengths_size,
                                B,
                                block_sizes.data_ptr<index_t>(),
                                my_size,
                                offsets_contig.data_ptr<offset_t>(),
                                indices_contig.data_ptr<index_t>(),
                                weights_value_contig.data_ptr<scalar_t>(),
                                new_offsets.data_ptr<offset_t>(),
                                new_indices.data_ptr<index_t>(),
                                new_weights.data_ptr<scalar_t>(),
                                nullptr,
                                unbucketize_permute.data_ptr<index_t>());
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      });
                });
          });
    } else if (bucketize_pos) {
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                [&] {
                  _block_bucketize_sparse_features_cuda_kernel2<
                      true,
                      false,
                      true,
                      offset_t,
                      index_t,
                      std::nullptr_t>
                      <<<num_blocks,
                         threads_per_block,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          lengths_size,
                          B,
                          block_sizes.data_ptr<index_t>(),
                          my_size,
                          offsets_contig.data_ptr<offset_t>(),
                          indices_contig.data_ptr<index_t>(),
                          nullptr,
                          new_offsets.data_ptr<offset_t>(),
                          new_indices.data_ptr<index_t>(),
                          nullptr,
                          new_pos.data_ptr<index_t>(),
                          unbucketize_permute.data_ptr<index_t>());
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    } else {
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                [&] {
                  _block_bucketize_sparse_features_cuda_kernel2<
                      true,
                      false,
                      false,
                      offset_t,
                      index_t,
                      std::nullptr_t>
                      <<<num_blocks,
                         threads_per_block,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          lengths_size,
                          B,
                          block_sizes.data_ptr<index_t>(),
                          my_size,
                          offsets_contig.data_ptr<offset_t>(),
                          indices_contig.data_ptr<index_t>(),
                          nullptr,
                          new_offsets.data_ptr<offset_t>(),
                          new_indices.data_ptr<index_t>(),
                          nullptr,
                          nullptr,
                          unbucketize_permute.data_ptr<index_t>());
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    }
  } else {
    if (weights.has_value() & bucketize_pos) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                [&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      [&] {
                        _block_bucketize_sparse_features_cuda_kernel2<
                            false,
                            true,
                            true,
                            offset_t,
                            index_t,
                            scalar_t>
                            <<<num_blocks,
                               threads_per_block,
                               0,
                               at::cuda::getCurrentCUDAStream()>>>(
                                lengths_size,
                                B,
                                block_sizes.data_ptr<index_t>(),
                                my_size,
                                offsets_contig.data_ptr<offset_t>(),
                                indices_contig.data_ptr<index_t>(),
                                weights_value_contig.data_ptr<scalar_t>(),
                                new_offsets.data_ptr<offset_t>(),
                                new_indices.data_ptr<index_t>(),
                                new_weights.data_ptr<scalar_t>(),
                                new_pos.data_ptr<index_t>(),
                                nullptr);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      });
                });
          });
    } else if (weights.has_value()) {
      Tensor weights_value = weights.value();
      auto weights_value_contig = weights_value.contiguous();
      new_weights = at::empty_like(weights_value);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_bucketize_sparse_features_weight_cuda_kernel2_2",
                [&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "_block_bucketize_sparse_features_cuda_weight_kernel2_3",
                      [&] {
                        _block_bucketize_sparse_features_cuda_kernel2<
                            false,
                            true,
                            false,
                            offset_t,
                            index_t,
                            scalar_t>
                            <<<num_blocks,
                               threads_per_block,
                               0,
                               at::cuda::getCurrentCUDAStream()>>>(
                                lengths_size,
                                B,
                                block_sizes.data_ptr<index_t>(),
                                my_size,
                                offsets_contig.data_ptr<offset_t>(),
                                indices_contig.data_ptr<index_t>(),
                                weights_value_contig.data_ptr<scalar_t>(),
                                new_offsets.data_ptr<offset_t>(),
                                new_indices.data_ptr<index_t>(),
                                new_weights.data_ptr<scalar_t>(),
                                nullptr,
                                nullptr);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      });
                });
          });
    } else if (bucketize_pos) {
      new_pos = at::empty_like(indices);
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                [&] {
                  _block_bucketize_sparse_features_cuda_kernel2<
                      false,
                      false,
                      true,
                      offset_t,
                      index_t,
                      std::nullptr_t>
                      <<<num_blocks,
                         threads_per_block,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          lengths_size,
                          B,
                          block_sizes.data_ptr<index_t>(),
                          my_size,
                          offsets_contig.data_ptr<offset_t>(),
                          indices_contig.data_ptr<index_t>(),
                          nullptr,
                          new_offsets.data_ptr<offset_t>(),
                          new_indices.data_ptr<index_t>(),
                          nullptr,
                          new_pos.data_ptr<index_t>(),
                          nullptr);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    } else {
      AT_DISPATCH_INDEX_TYPES(
          offsets_contig.scalar_type(),
          "_bucketize_sparse_features_weight_cuda_kernel2_1",
          [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(),
                "_block_bucketize_sparse_features_cuda_kernel2_2",
                [&] {
                  _block_bucketize_sparse_features_cuda_kernel2<
                      false,
                      false,
                      false,
                      offset_t,
                      index_t,
                      std::nullptr_t>
                      <<<num_blocks,
                         threads_per_block,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          lengths_size,
                          B,
                          block_sizes.data_ptr<index_t>(),
                          my_size,
                          offsets_contig.data_ptr<offset_t>(),
                          indices_contig.data_ptr<index_t>(),
                          nullptr,
                          new_offsets.data_ptr<offset_t>(),
                          new_indices.data_ptr<index_t>(),
                          nullptr,
                          nullptr,
                          nullptr);
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    }
  }

  return {new_lengths, new_indices, new_weights, new_pos, unbucketize_permute};
}

// Kernel for bucketize lengths, with the Cyclic distribution (vs. block,
// block-cyclic distribution). Used for bucketize sparse feature with row-wise
// partition (sparse_feature is partitioned cyclically along the sparse
// dimension into my_size blocks)
template <typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void _bucketize_sparse_features_cuda_kernel1(
    int lengths_size,
    int my_size,
    const scalar_t* __restrict__ offsets_data,
    const scalar_t* __restrict__ indices_data,
    scalar_t* __restrict__ new_lengths_data) {
  using uscalar_t = std::make_unsigned_t<scalar_t>;
  CUDA_KERNEL_LOOP(r, lengths_size) {
    scalar_t rowstart = (r == 0 ? 0 : offsets_data[r - 1]);
    scalar_t rowend = offsets_data[r];
    for (scalar_t i = rowstart; i < rowend; ++i) {
      // Need to handle negative indices if we use raw indices instead of hashed
      // indices, convert to unsigned
      uscalar_t idx = static_cast<uscalar_t>(indices_data[i]);
      uscalar_t p = idx % my_size;
      new_lengths_data[p * lengths_size + r]++;
    }
  }
}

// Kernel for bucketize offsets, indices, and positional weights, with the
// Cyclic distribution (vs. block, block-cyclic distribution). Used for
// bucketize sparse feature with row-wise partition (sparse_feature is
// partitioned cyclically along the sparse dimension into my_size blocks)
template <
    bool has_weight,
    bool bucketize_pos,
    typename index_t,
    typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void _bucketize_sparse_features_cuda_kernel2(
    int lengths_size,
    int my_size,
    const index_t* __restrict__ offsets_data,
    const index_t* __restrict__ indices_data,
    const scalar_t* __restrict__ weights_data,
    index_t* __restrict__ new_offsets_data,
    index_t* __restrict__ new_indices_data,
    scalar_t* __restrict__ new_weights_data,
    index_t* __restrict__ new_pos_data) {
  using uindex_t = std::make_unsigned_t<index_t>;
  CUDA_KERNEL_LOOP(r, lengths_size) {
    index_t rowstart = r == 0 ? 0 : offsets_data[r - 1];
    index_t rowend = offsets_data[r];
    for (index_t i = rowstart; i < rowend; ++i) {
      // Need to handle negative indices if we use raw indices instead of hashed
      // indices, convert to unsigned
      uindex_t idx = static_cast<uindex_t>(indices_data[i]);
      uindex_t p = idx % my_size;
      uindex_t new_idx = idx / my_size;
      uindex_t pos = new_offsets_data[p * lengths_size + r];
      new_indices_data[pos] = new_idx;
      new_offsets_data[p * lengths_size + r]++;
      if (has_weight) {
        new_weights_data[pos] = weights_data[i];
      }
      if (bucketize_pos) {
        new_pos_data[pos] = i - rowstart;
      }
    }
  }
}

// This function partitions sparse features
// cyclically along the sparse dimension into my_size blocks
std::tuple<Tensor, Tensor, c10::optional<Tensor>, c10::optional<Tensor>>
bucketize_sparse_features_cuda(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const int64_t my_size,
    const c10::optional<Tensor>& weights) {
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(indices);
  TENSORS_ON_SAME_DEVICE(lengths, indices);
  TENSOR_ON_CUDA_GPU(weights);
  TENSORS_ON_SAME_DEVICE(lengths, weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(lengths.get_device());
  // allocate tensors and buffers
  const int lengths_size = lengths.numel();
  const int new_lengths_size = lengths_size * my_size;
  auto offsets = at::empty({lengths_size}, lengths.options());
  auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
  auto new_offsets = at::empty({new_lengths_size}, lengths.options());
  auto new_indices = at::empty_like(indices);
  auto lengths_contig = lengths.contiguous();
  auto indices_contig = indices.contiguous();
  auto offsets_contig = offsets.contiguous();
  Tensor new_weights;
  Tensor new_pos;
  // count nonzeros
  offsets_contig = fbgemm_gpu::asynchronous_inclusive_cumsum_gpu(lengths);
  int threads_per_block = 256;
  const auto num_blocks =
      cuda_calc_xblock_count(lengths_size, threads_per_block);
  AT_DISPATCH_INDEX_TYPES(
      indices_contig.scalar_type(),
      "_bucketize_sparse_features_cuda_kernel1",
      ([&] {
        _bucketize_sparse_features_cuda_kernel1<<<
            num_blocks,
            threads_per_block,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            lengths_size,
            my_size,
            offsets_contig.data_ptr<index_t>(),
            indices_contig.data_ptr<index_t>(),
            new_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
  // bucketize nonzeros
  new_offsets = fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(new_lengths);
  if (weights.has_value() & bucketize_pos) {
    Tensor weights_value = weights.value();
    auto weights_value_contig = weights_value.contiguous();
    new_weights = at::empty_like(weights_value);
    new_pos = at::empty_like(indices);
    AT_DISPATCH_INDEX_TYPES(
        indices_contig.scalar_type(),
        "_bucketize_sparse_features_weight_cuda_kernel2_1",
        ([&] {
          AT_DISPATCH_FLOATING_TYPES(
              weights_value.scalar_type(),
              "_bucketize_sparse_features_cuda_weight_kernel2_2",
              ([&] {
                _bucketize_sparse_features_cuda_kernel2<
                    true,
                    true,
                    index_t,
                    scalar_t>
                    <<<num_blocks,
                       threads_per_block,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        lengths_size,
                        my_size,
                        offsets_contig.data_ptr<index_t>(),
                        indices_contig.data_ptr<index_t>(),
                        weights_value_contig.data_ptr<scalar_t>(),
                        new_offsets.data_ptr<index_t>(),
                        new_indices.data_ptr<index_t>(),
                        new_weights.data_ptr<scalar_t>(),
                        new_pos.data_ptr<index_t>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }));
        }));
  } else if (weights.has_value()) {
    Tensor weights_value = weights.value();
    auto weights_value_contig = weights_value.contiguous();
    new_weights = at::empty_like(weights_value);
    AT_DISPATCH_INDEX_TYPES(
        indices_contig.scalar_type(),
        "_bucketize_sparse_features_weight_cuda_kernel2_1",
        ([&] {
          AT_DISPATCH_FLOATING_TYPES(
              weights_value.scalar_type(),
              "_bucketize_sparse_features_cuda_weight_kernel2_2",
              ([&] {
                _bucketize_sparse_features_cuda_kernel2<
                    true,
                    false,
                    index_t,
                    scalar_t>
                    <<<num_blocks,
                       threads_per_block,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        lengths_size,
                        my_size,
                        offsets_contig.data_ptr<index_t>(),
                        indices_contig.data_ptr<index_t>(),
                        weights_value_contig.data_ptr<scalar_t>(),
                        new_offsets.data_ptr<index_t>(),
                        new_indices.data_ptr<index_t>(),
                        new_weights.data_ptr<scalar_t>(),
                        nullptr);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }));
        }));
  } else if (bucketize_pos) {
    new_pos = at::empty_like(indices);
    AT_DISPATCH_INDEX_TYPES(
        indices_contig.scalar_type(),
        "_bucketize_sparse_features_cuda_kernel2",
        ([&] {
          _bucketize_sparse_features_cuda_kernel2<
              false,
              true,
              index_t,
              std::nullptr_t>
              <<<num_blocks,
                 threads_per_block,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  lengths_size,
                  my_size,
                  offsets_contig.data_ptr<index_t>(),
                  indices_contig.data_ptr<index_t>(),
                  nullptr,
                  new_offsets.data_ptr<index_t>(),
                  new_indices.data_ptr<index_t>(),
                  nullptr,
                  new_pos.data_ptr<index_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
  } else {
    AT_DISPATCH_INDEX_TYPES(
        indices_contig.scalar_type(),
        "_bucketize_sparse_features_cuda_kernel2",
        ([&] {
          _bucketize_sparse_features_cuda_kernel2<
              false,
              false,
              index_t,
              std::nullptr_t>
              <<<num_blocks,
                 threads_per_block,
                 0,
                 at::cuda::getCurrentCUDAStream()>>>(
                  lengths_size,
                  my_size,
                  offsets_contig.data_ptr<index_t>(),
                  indices_contig.data_ptr<index_t>(),
                  nullptr,
                  new_offsets.data_ptr<index_t>(),
                  new_indices.data_ptr<index_t>(),
                  nullptr,
                  nullptr);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
  }

  return {new_lengths, new_indices, new_weights, new_pos};
}

template <typename Dtype>
__global__
__launch_bounds__(kMaxThreads) void reorder_batched_ad_lengths_kernel(
    // reorder lengths from (ragged) [B  x T x #num_ads_b)] to
    // [T][B][#num_ads_b], i.e. [T][sum(#num_ads_b)].
    const at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        cat_ad_lengths,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        batch_offsets,
    at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        reordered_cat_ad_lengths,
    int32_t T) {
  const int32_t B = batch_offsets.size(0) - 1;

  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const int32_t num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const int32_t input_segment_start = T * batch_offsets[b] + t * num_ads_b;
  const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets[b];

  for (int32_t i = threadIdx.x; i < num_ads_b; i += blockDim.x) {
    reordered_cat_ad_lengths[output_segment_start + i] =
        cat_ad_lengths[input_segment_start + i];
  }
}

Tensor reorder_batched_ad_lengths_gpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch) {
  TENSOR_ON_CUDA_GPU(cat_ad_lengths);
  TENSOR_ON_CUDA_GPU(batch_offsets);
  TENSORS_ON_SAME_DEVICE(cat_ad_lengths, batch_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cat_ad_lengths.get_device());

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = cat_ad_lengths.numel() / num_ads_in_batch;

  Tensor reordered_cat_ad_lengths = at::empty_like(cat_ad_lengths);

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES(
      cat_ad_lengths.type(), "reorder_batched_ad_lengths_gpu_kernel", [&] {
        reorder_batched_ad_lengths_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                batch_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                reordered_cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                T);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return reordered_cat_ad_lengths;
}

template <typename Dtype, typename index_t = int32_t>
__global__
__launch_bounds__(kMaxThreads) void reorder_batched_ad_indices_kernel(
    // reorder indices from (ragged) [B  x T x #num_ads_b x length_{b, t, a})]
    // to [T][B][#num_ads_b][length_{b, t, a}], i.e. [sum(length_{b, t, a})],
    // laid out as [T][B][A][L] (if all lengths were equal).
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        cat_ad_offsets,
    const at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        cat_ad_indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        reordered_cat_ad_offsets,
    at::PackedTensorAccessor32<Dtype, 1, at::RestrictPtrTraits>
        reordered_cat_ad_indices,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        batch_offsets,
    int32_t T) {
  const int32_t B = batch_offsets.size(0) - 1;
  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }
  // for each ad,
  const int32_t num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const int32_t b_t_start = T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_start =
      T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_end =
      T * batch_offsets[b] + t * num_ads_b + num_ads_b;

  // Idea: we want to copy the entire segment of size sum_a(length_{b, t, a})
  // from starting point (given by cat_ad_offsets[b, t])
  // to end point (given by reordered_cat_ad_indices[t][b])
  const int32_t input_segment_start =
      cat_ad_offsets[input_segment_offset_start];
  const int32_t input_segment_end = cat_ad_offsets[input_segment_offset_end];

  const int32_t output_segment_offset_start =
      t * num_ads_in_batch + batch_offsets[b];
  const int32_t output_segment_start =
      reordered_cat_ad_offsets[output_segment_offset_start];

  for (int32_t i = threadIdx.x; i < input_segment_end - input_segment_start;
       i += blockDim.x) {
    reordered_cat_ad_indices[output_segment_start + i] =
        cat_ad_indices[input_segment_start + i];
  }
}

Tensor reorder_batched_ad_indices_gpu(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch) {
  TENSOR_ON_CUDA_GPU(cat_ad_offsets);
  TENSOR_ON_CUDA_GPU(cat_ad_indices);
  TENSOR_ON_CUDA_GPU(reordered_cat_ad_offsets);
  TENSOR_ON_CUDA_GPU(batch_offsets);
  TENSORS_ON_SAME_DEVICE(cat_ad_offsets, cat_ad_indices);
  TENSORS_ON_SAME_DEVICE(cat_ad_offsets, reordered_cat_ad_offsets);
  TENSORS_ON_SAME_DEVICE(cat_ad_offsets, batch_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cat_ad_offsets.get_device());

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = (cat_ad_offsets.numel() - 1) / num_ads_in_batch;
  Tensor reordered_cat_ad_indices = at::empty_like(cat_ad_indices);

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES(
      cat_ad_indices.type(), "reorder_batched_ad_indices_gpu_kernel_1", [&] {
        AT_DISPATCH_INDEX_TYPES(
            cat_ad_offsets.scalar_type(),
            "reorder_batched_ad_indices_gpu_kernel_2",
            [&] {
              reorder_batched_ad_indices_kernel<scalar_t, index_t><<<
                  blocks,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  batch_offsets
                      .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                  T);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return reordered_cat_ad_indices;
}

// Forward kernel for batched unary embedding op
template <typename scalar_t, typename index_t>
__global__
__launch_bounds__(kMaxThreads) void batched_unary_embeddings_forward_kernel(
    const int32_t N,
    const int32_t B,
    const int32_t T,
    const scalar_t* __restrict__ weight, // N * sum(E) * 1 (embedding dimension
                                         // is 1)
    const index_t* __restrict__ table_offsets,
    const index_t* __restrict__ offsets,
    const index_t* __restrict__ indices,
    scalar_t* __restrict__ output // N * B * T
) {
  index_t sum_E = table_offsets[T];
  int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) {
    return;
  }
  int32_t t = blockIdx.y;
  int32_t n = blockIdx.z;
  index_t table_offset = table_offsets[t];
  index_t indices_start = offsets[t * B + b];
  index_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;
  at::acc_type<scalar_t, true> sum = 0.0;
  for (int32_t l = 0; l < L; ++l) {
    auto idx = __ldg(&indices[indices_start + l]);
    sum += weight[n * sum_E + table_offset + idx + 0];
  }
  output[(n * B + b) * T + t] = sum;
}

Tensor batched_unary_embeddings_forward_cuda(
    const Tensor& weight,
    const Tensor& table_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(table_offsets);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(weight);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(offsets);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(indices);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weight.get_device());
  // N: number of tasks, T: number of tables, B: batch size
  const int32_t N = weight.size(0);
  const int32_t T = table_offsets.numel() - 1;
  const int32_t B = (offsets.numel() - 1) / T;
  TORCH_CHECK(N > 0);
  TORCH_CHECK(B > 0);
  TORCH_CHECK(T > 0);
  TORCH_CHECK(T <= 65535);
  TORCH_CHECK(N <= 65535);
  int32_t threads = std::min<int32_t>(B, 512);
  dim3 blocks(cuda_calc_xblock_count(B, threads), T, N);
  auto output = at::empty({N, B, T}, weight.options());
  AT_DISPATCH_INDEX_TYPES(
      indices.type(), "batched_unary_embeddings_forward_kernel", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            weight.type(), "batched_unary_embeddings_forward_kernel", [&] {
              batched_unary_embeddings_forward_kernel<scalar_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      N,
                      B,
                      T,
                      weight.data_ptr<scalar_t>(),
                      table_offsets.data_ptr<index_t>(),
                      offsets.data_ptr<index_t>(),
                      indices.data_ptr<index_t>(),
                      output.data_ptr<scalar_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return output;
}

// Backward kernel for batched unary embedding op
// We sort input indices so we don't have race conditions, an approach similar
// to the usual split table batched embedding backward.
// We can think of the following alternatives but each with challenges:
// 1) Assign output elements to different threads. Each thread scan all indices
//    corresponding to the table it owns but only accumulate gradients when an
//    index value matches with the output element it owns.
//    A challenge is each thread need to binary search to map from [0 .. sum_E]
//    to table id.
// 2) Densify indices and offsets to create [B, sum_E] matrix. Then, do batched
//    GEMM where ith GEMM multiplies [N, B] submatrix of grad_output with
//    [B, E_i] submatrix where E_i is the num of embeddings of ith table.
//    Concatenating the GEMM outputs will result in [N, B, T]
//    A challenge is there's no available batched GEMM routine with varying K
//    dimension.
template <typename scalar_t, typename index_t>
__global__
__launch_bounds__(kMaxThreads) void batched_unary_embeddings_backward_kernel(
    const int32_t N,
    const int32_t B,
    const int32_t T,
    const scalar_t* __restrict__ grad_output, // [N * B * T]
    const index_t* __restrict__ table_offsets,
    scalar_t* __restrict__ grad_weight, // [N * sum_E * 1] (embedding
                                        // dimension is 1)
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        sorted_linear_indices_run,
    const int32_t* __restrict__ sorted_linear_indices_cumulative_run_lengths,
    const int32_t* __restrict__ sorted_infos,
    const int32_t* __restrict__ sorted_linear_indices_num_runs,
    FixedDivisor fd) {
  int32_t run_id = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t n = blockIdx.y;
  if (n >= N) {
    return;
  }
  if (run_id >= sorted_linear_indices_run.size(0)) {
    return;
  }
  if (run_id >= sorted_linear_indices_num_runs[0]) {
    return;
  }
  int64_t linear_index = sorted_linear_indices_run[run_id];
  int32_t segment_start = sorted_linear_indices_cumulative_run_lengths[run_id];
  int32_t segment_end =
      sorted_linear_indices_cumulative_run_lengths[run_id + 1];
  int32_t SL = segment_end - segment_start;

  if (SL == 0) {
    return;
  }

  // now, each segment corresponds to exactly one table `t` and row in
  // that table (`idx`). Thus, we can hoist out some of the book-keeping.
  auto info = sorted_infos[segment_start];
  int t = fd.Div(info);

  at::acc_type<scalar_t, true> grad_sum = 0.0;
  for (int32_t sl = 0; sl < SL; ++sl) {
    int32_t b = fd.Mod(sorted_infos[segment_start + sl]);
    grad_sum += grad_output[(n * B + b) * T + t];
  }

  index_t table_offset = table_offsets[t];
  index_t sum_E = table_offsets[T];
  int64_t idx = linear_index - table_offset;
  grad_weight[n * sum_E + table_offset + idx] = grad_sum;
}

Tensor batched_unary_embeddings_backward_cuda(
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& table_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  TENSOR_ON_CUDA_GPU(grad_output);
  TENSOR_ON_CUDA_GPU(weight);
  TENSOR_ON_CUDA_GPU(table_offsets);
  TENSOR_ON_CUDA_GPU(offsets);
  TENSOR_ON_CUDA_GPU(indices);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  // N: number of tasks, T: number of tables, B: batch size
  const int32_t N = grad_output.size(0);
  const int32_t B = grad_output.size(1);
  const int32_t T = grad_output.size(2);
  TORCH_CHECK(N > 0);
  TORCH_CHECK(B > 0);
  TORCH_CHECK(T > 0);

  // weight: [N, sum_E]
  // total_hash_size_bits = log2(sum_E)
  int64_t total_hash_size_bits = log2(weight.numel() / N) + 1;

  Tensor linear_indices, linear_indices_sorted;
  Tensor infos_sorted;
  Tensor sorted_linear_indices_run, sorted_linear_indices_run_lengths,
      sorted_linear_indices_num_runs,
      sorted_linear_indices_cumulative_run_lengths;
  std::tie(
      linear_indices,
      linear_indices_sorted,
      infos_sorted,
      sorted_linear_indices_run,
      sorted_linear_indices_run_lengths,
      sorted_linear_indices_num_runs,
      sorted_linear_indices_cumulative_run_lengths) =
      transpose_embedding_input(
          table_offsets, total_hash_size_bits, indices, offsets);

  int threads = std::min<int32_t>(sorted_linear_indices_run.numel(), 512);
  dim3 blocks(
      cuda_calc_xblock_count(sorted_linear_indices_run.numel(), threads), N);
  auto grad_weight = at::zeros_like(weight);
  AT_DISPATCH_INDEX_TYPES(
      indices.type(), "batched_unary_embeddings_backward_kernel", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_output.type(),
            "batched_unary_embeddings_backward_kernel",
            [&] {
              batched_unary_embeddings_backward_kernel<scalar_t>
                  <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                      N,
                      B,
                      T,
                      grad_output.data_ptr<scalar_t>(),
                      table_offsets.data_ptr<index_t>(),
                      grad_weight.data_ptr<scalar_t>(),
                      sorted_linear_indices_run.packed_accessor32<
                          index_t,
                          1,
                          at::RestrictPtrTraits>(),
                      sorted_linear_indices_cumulative_run_lengths
                          .data_ptr<int32_t>(),
                      infos_sorted.data_ptr<int32_t>(),
                      sorted_linear_indices_num_runs.data_ptr<int32_t>(),
                      FixedDivisor(B));
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return grad_weight;
}

Tensor lengths_range_cuda(
    const Tensor& t_in,
    const c10::optional<std::vector<int64_t>>& shape) {
  TENSOR_ON_CUDA_GPU(t_in);
  TENSOR_NDIM_EQUALS(t_in, 1);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());

  const auto t_in_contig = t_in.contiguous();
  const auto num_seq = t_in_contig.numel();

  Tensor offsets;
  int64_t output_size = 1;

  if (shape.has_value()) {
    offsets = fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(t_in_contig);
    output_size = c10::multiply_integers(shape.value());
  } else {
    // if we don't provide the the shape info, this is a slow path
    // we need to transfer the size of the output from GPU to CPU
    offsets = fbgemm_gpu::asynchronous_complete_cumsum_gpu(t_in_contig);
    AT_DISPATCH_INDEX_TYPES(
        t_in_contig.scalar_type(), "lengths_range_output_size", [&] {
          output_size = *(offsets[num_seq].cpu().data_ptr<index_t>());
        });
  }

  auto output = at::empty({output_size}, t_in.options());

  uint32_t vector_size;
  uint32_t rows_per_block;
  uint32_t num_blocks;
  std::tie(num_blocks, rows_per_block, vector_size) =
      calc_offsets_range_thread_block(output_size, num_seq);
  dim3 threads(vector_size, rows_per_block);
  AT_DISPATCH_INDEX_TYPES(
      t_in_contig.scalar_type(), "lengths_range_compute", [&] {
        fbgemm_gpu::_offsets_range_cuda_kernel<index_t>
            <<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_seq,
                output_size,
                offsets.data_ptr<index_t>(),
                output.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output;
}

// Kernel for permuting the indices and weights. Used for permutation of
// sparse features
template <bool has_weight, typename index_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void permute_indices_weights_kernel(
    int32_t len,
    int32_t T,
    int32_t B,
    const index_t* __restrict__ indices,
    const scalar_t* __restrict__ weights,
    const int32_t* __restrict__ permute,
    const index_t* __restrict__ input_offsets,
    const index_t* __restrict__ output_offsets,
    index_t* __restrict__ permuted_indices,
    scalar_t* __restrict__ permuted_weights) {
  int32_t b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    index_t output_start = output_offsets[b_t];
    index_t segment_length;
    if (b_t == B * T - 1) {
      segment_length = len - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    index_t input_start = input_offsets[permute[t] * B + b];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        permuted_weights[output_start + i] = weights[input_start + i];
      }
    }
  }
}

std::tuple<Tensor, Tensor, c10::optional<Tensor>> permute_sparse_features_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights) {
  TENSOR_ON_CUDA_GPU(permute);
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(weights);

  TENSORS_ON_SAME_DEVICE(permute, lengths);
  TENSORS_ON_SAME_DEVICE(permute, indices);
  TENSORS_ON_SAME_DEVICE(permute, weights);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(indices.get_device());

  // the following implementation requires lengths and indices has the same
  // dtype if usecase comes up that requires different dtype (e.g. int32 for
  // lengths and int64 for indices, this will give a better error msg for
  // debugging
  TENSORS_HAVE_SAME_TYPE(lengths, indices);

  TORCH_CHECK(
      lengths.dim() == 2,
      "The dimension of lengths tensor should be equal to 2 to correctly infer number of features and batch size.")

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the features to permute over can be less or more with or without
  // repetitions
  const auto num_output_features = permute.numel();
  const auto num_features = lengths.size(0);
  const auto B = lengths.size(1);

  Tensor permuted_lengths;
  Tensor permuted_indices;
  Tensor permuted_weights;

  permuted_lengths = at::empty({num_output_features, B}, lengths.options());

  constexpr int32_t threads_1 = 256;
  const auto blocks_1 =
      cuda_calc_xblock_count(B * num_output_features, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_2D_lengths_kernel", [&] {
        fbgemm_gpu::permute_2D_lengths_kernel<index_t>
            <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_output_features,
                B,
                lengths_contig.data_ptr<index_t>(),
                permute.data_ptr<int32_t>(),
                permuted_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // convert lengths to offsets
  const auto input_offsets =
      fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths_contig);
  const auto output_offsets =
      fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(permuted_lengths);
  int64_t permuted_lengths_sum = indices.numel();

  /* TODO: Remove the condition protecting the slow path because even when the
   * condition below is true permuted_lengths.sum() could still be needed. For
   * instance if there are three features with indices `[0, 1, 2]`, `permute`
   * can be `[0, 1, 1]` for which permuted lengths sum would be needed to
   * create permuted_{indices, weights} and `permuted_lengths_sum =
   * indices.numel() or weights.numdel() would be incorrect.
   */
  if (num_features != num_output_features) {
    permuted_lengths_sum = permuted_lengths.sum().item<int64_t>();
  }

  constexpr int32_t BT_blocks = 32;
  dim3 threads_2(32, BT_blocks);
  const auto blocks_2 =
      cuda_calc_xblock_count(B * num_output_features, BT_blocks);
  permuted_indices = at::empty(permuted_lengths_sum, indices.options());
  if (weights.has_value()) {
    const Tensor weights_value = weights.value();
    const auto weights_value_contig = weights_value.contiguous();
    permuted_weights = at::empty(permuted_lengths_sum, weights_value.options());
    AT_DISPATCH_INDEX_TYPES(
        input_offsets.scalar_type(), "permute_indices_weights_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND(
              at::ScalarType::Int,
              weights_value.scalar_type(),
              "permute_indices_weights_kernel_2",
              [&] {
                permute_indices_weights_kernel<true, index_t, scalar_t>
                    <<<blocks_2,
                       threads_2,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        permuted_lengths_sum,
                        num_output_features,
                        B,
                        indices_contig.data_ptr<index_t>(),
                        weights_value_contig.data_ptr<scalar_t>(),
                        permute_contig.data_ptr<int32_t>(),
                        input_offsets.data_ptr<index_t>(),
                        output_offsets.data_ptr<index_t>(),
                        permuted_indices.data_ptr<index_t>(),
                        permuted_weights.data_ptr<scalar_t>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  } else {
    AT_DISPATCH_INDEX_TYPES(
        indices.scalar_type(), "permute_indices_kernel", [&] {
          permute_indices_weights_kernel<false, index_t, std::nullptr_t>
              <<<blocks_2, threads_2, 0, at::cuda::getCurrentCUDAStream()>>>(
                  permuted_lengths_sum,
                  num_output_features,
                  B,
                  indices_contig.data_ptr<index_t>(),
                  nullptr,
                  permute_contig.data_ptr<int32_t>(),
                  input_offsets.data_ptr<index_t>(),
                  output_offsets.data_ptr<index_t>(),
                  permuted_indices.data_ptr<index_t>(),
                  nullptr);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  return {permuted_lengths, permuted_indices, permuted_weights};
}

// A: m, batch_size, k
// B: batch_size, k, n
// C: m, batch_size, n
// bias: batch_size, n
Tensor permute102_baddbmm_permute102_cuda(
    const Tensor& bias,
    const Tensor& A,
    const Tensor& B) {
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(A);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(B);
  TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(bias);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(A.get_device());

  TENSORS_ON_SAME_DEVICE(A, B);
  TENSORS_ON_SAME_DEVICE(A, bias);
  TENSOR_NDIM_EQUALS(A, 3);
  TENSOR_NDIM_EQUALS(B, 3);

  const auto m = A.size(0);
  const auto batch_size = A.size(1);
  const auto k = A.size(2);
  const auto n = B.size(2);
  TORCH_CHECK(B.size(0) == batch_size);
  TORCH_CHECK(B.size(1) == k);
  TORCH_CHECK(bias.size(0) == batch_size);
  TORCH_CHECK(bias.size(1) == n);

  // auto C = at::empty({m, batch_size, n}, A.options());
  // auto C = at::broadcast_to(bias, {m, batch_size, n}).contiguous();
  auto C = bias.unsqueeze(0).broadcast_to({m, batch_size, n}).contiguous();

  auto handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, c10::cuda::getCurrentCUDAStream());

  // C (m, b, n) = A (m, b, k) * B (b, k, n) ---> row major
  // C (m, b, n) = (B^T (b, k, n) * A^T (m, b, k))^T ---> column major

  float alpha = 1.0f;
  float beta = 1.0f;

  auto Btype = CUDA_R_16F;
  auto ldb = n;
  auto strideB = n * k;

  auto Atype = CUDA_R_16F;
  auto lda = k * batch_size;
  auto strideA = k;

  auto Ctype = CUDA_R_16F;
  auto ldc = n * batch_size;
  auto strideC = n;

  auto computeType = CUBLAS_COMPUTE_32F;

  auto result = cublasGemmStridedBatchedEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      B.data_ptr<at::Half>(),
      Btype,
      ldb,
      strideB,
      A.data_ptr<at::Half>(),
      Atype,
      lda,
      strideA,
      &beta,
      C.data_ptr<at::Half>(),
      Ctype,
      ldc,
      strideC,
      batch_size,
      computeType,
      CUBLAS_GEMM_DEFAULT);
  TORCH_CHECK(result == CUBLAS_STATUS_SUCCESS);
  return C;
}

// Kernel for permuting the indices and weights. Used for permutation of
// table-wise partitioned sequence embeddings

template <typename index_t, typename scalar_t>
__global__ void permute_embeddings_kernel(
    int32_t len,
    int32_t T,
    int32_t B,
    const scalar_t* __restrict__ embeddings,
    // bag level permute
    const int32_t* __restrict__ permute,
    const index_t* __restrict__ input_offsets,
    const index_t* __restrict__ output_offsets,
    scalar_t* __restrict__ permuted_embeddings) {
  int32_t b_t_start = blockIdx.x * blockDim.y + threadIdx.y;
  const int stride = gridDim.x * blockDim.y;
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    index_t output_start = output_offsets[b_t];
    index_t segment_length;
    if (b_t == B * T - 1) {
      segment_length = len - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    index_t input_start = input_offsets[permute[t] * B + b];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_embeddings[output_start + i] = embeddings[input_start + i];
    }
  }
}

std::tuple<Tensor, Tensor> permute_sequence_embeddings_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& embeddings) {
  TENSOR_ON_CUDA_GPU(permute);
  TENSOR_ON_CUDA_GPU(lengths);
  TENSOR_ON_CUDA_GPU(embeddings);

  TENSORS_ON_SAME_DEVICE(permute, lengths);
  TENSORS_ON_SAME_DEVICE(permute, embeddings);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(embeddings.get_device());

  TORCH_CHECK(
      lengths.dim() == 2,
      "The dimension of lengths tensor should be equal to 2"
      "to correctly infer number of features and batch size.")

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto embeddings_contig = embeddings.contiguous();
  // the features to permute over can be less or more with or without
  // repetitions
  const auto num_output_embeddings = permute.numel();
  const auto num_embeddings = lengths.size(0);
  const auto B = lengths.size(1);

  Tensor permuted_lengths;
  Tensor permuted_embeddings;

  permuted_lengths = at::empty({num_output_embeddings, B}, lengths.options());

  constexpr int32_t threads_1 = 256;
  const auto blocks_1 =
      cuda_calc_xblock_count(B * num_output_embeddings, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_2D_lengths_kernel", [&] {
        fbgemm_gpu::permute_2D_lengths_kernel<index_t>
            <<<blocks_1, threads_1, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_output_embeddings,
                B,
                lengths_contig.data_ptr<index_t>(),
                permute_contig.data_ptr<int32_t>(),
                permuted_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // convert lengths to offsets
  const auto input_offsets =
      fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(lengths_contig);
  const auto output_offsets =
      fbgemm_gpu::asynchronous_exclusive_cumsum_gpu(permuted_lengths);
  int64_t permuted_lengths_sum = embeddings.numel();

  /* TODO: Remove the condition protecting the slow path because even when the
   * condition below is true permuted_lengths.sum() could still be needed. For
   * instance if there are three features with indices `[0, 1, 2]`, `permute`
   * can be `[0, 1, 1]` for which permuted lengths sum would be needed to
   * create permuted_embeddings and `permuted_lengths_sum = embeddings.numel()
   * would be incorrect.
   */
  if (num_embeddings != num_output_embeddings) {
    permuted_lengths_sum = permuted_lengths.sum().item<int64_t>();
  }

  constexpr int32_t BT_blocks = 32;
  dim3 threads_2(32, BT_blocks);
  const auto blocks_2 =
      cuda_calc_xblock_count(B * num_output_embeddings, BT_blocks);
  permuted_embeddings = at::empty(permuted_lengths_sum, embeddings.options());
  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_embeddings_kernel_1", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND(
            at::ScalarType::Int,
            embeddings.scalar_type(),
            "permute_embeddings_kernel_2",
            [&] {
              permute_embeddings_kernel<index_t, scalar_t>
                  <<<blocks_2,
                     threads_2,
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      permuted_lengths_sum,
                      num_output_embeddings,
                      B,
                      embeddings_contig.data_ptr<scalar_t>(),
                      permute_contig.data_ptr<int32_t>(),
                      input_offsets.data_ptr<index_t>(),
                      output_offsets.data_ptr<index_t>(),
                      permuted_embeddings.data_ptr<scalar_t>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });

  return {permuted_lengths, permuted_embeddings};
}

} // namespace fbgemm_gpu
