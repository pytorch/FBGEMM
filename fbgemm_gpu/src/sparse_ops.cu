/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/sparse_ops.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDADeviceAssertion.h>
#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

/*
 * We annotate the public fbgemm functions and hide the rest. Those
 * public symbols can be called via fbgemm_gpu::func() or pytorch
 * operator dispatcher. We'll hide other symbols, especially cub APIs,
 * because different .so may include the same cub CUDA kernels, which
 * results in confusion and libA may end up calling libB's cub kernel,
 * causing failures when we static link libcudart_static.a
 */
#define DLL_PUBLIC __attribute__((visibility("default")))

#ifdef __HIP_PLATFORM_HCC__
#include <hipblas.h>
#endif

#ifdef __HIP_PLATFORM_HCC__
#define LDG(ptr) (*(ptr))
#else
#define LDG(ptr) (__ldg(ptr))
#endif

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

DLL_PUBLIC Tensor
offsets_range_cuda(const Tensor& offsets, int64_t range_size) {
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

DLL_PUBLIC Tensor segment_sum_csr_cuda(
    const int64_t batch_size,
    const Tensor& csr_seg,
    const Tensor& values) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(csr_seg, values);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  auto output = at::empty(csr_seg.numel() - 1, values.options());
  constexpr uint32_t threads_per_block = 256;
  const uint32_t num_blocks = csr_seg.numel() - 1;
  AT_DISPATCH_ALL_TYPES(values.scalar_type(), "_segment_sum_csr_cuda", [&] {
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

DLL_PUBLIC Tensor asynchronous_inclusive_cumsum_gpu(const Tensor& t_in) {
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

DLL_PUBLIC Tensor asynchronous_exclusive_cumsum_gpu(const Tensor& t_in) {
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

DLL_PUBLIC Tensor asynchronous_complete_cumsum_gpu(const Tensor& t_in) {
  TENSOR_ON_CUDA_GPU(t_in);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  TORCH_CHECK(t_in.dim() == 1 || t_in.dim() == 2);
  if (t_in.dim() == 1) {
    // CUB only handles up to INT_MAX elements.
    TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
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
  } else {
    // Workaround for the unstable custom op
    // TODO: Re-enable the custom op
    const auto num_vecs = t_in.size(0);
    const auto num_entries = t_in.size(1);
    TORCH_CHECK(num_entries < std::numeric_limits<int32_t>::max());
    auto t_out = at::zeros({num_vecs, num_entries + 1}, t_in.options());

    AT_DISPATCH_INDEX_TYPES(
        t_in.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              nullptr,
              temp_storage_bytes,
              t_in.data_ptr<index_t>(),
              t_out.data_ptr<index_t>() + 1,
              num_entries,
              at::cuda::getCurrentCUDAStream()));
        });
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        t_in.options().dtype(at::kByte));
    for (auto v = 0; v < num_vecs; v++) {
      AT_DISPATCH_INDEX_TYPES(
          t_in.scalar_type(), "cub_inclusive_sum_wrapper2", [&] {
            AT_CUDA_CHECK(
                FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
                    temp_storage.data_ptr(),
                    temp_storage_bytes,
                    t_in.data_ptr<index_t>() + v * num_entries,
                    t_out.data_ptr<index_t>() + v * (num_entries + 1) + 1,
                    num_entries,
                    at::cuda::getCurrentCUDAStream()));
          });
    }

    return t_out;
  }
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

DLL_PUBLIC std::tuple<Tensor, Tensor, c10::optional<Tensor>>
permute_2D_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, indices, weights);
  TORCH_CHECK(lengths.dim() == 2);

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
      asynchronous_complete_cumsum_gpu(permuted_lengths.flatten());
  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size = output_offsets[-1].item<int64_t>();
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

DLL_PUBLIC std::tuple<Tensor, Tensor, c10::optional<Tensor>>
permute_1D_sparse_data_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, indices, weights);

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
                permute_contig.data_ptr<int32_t>(),
                permuted_lengths.data_ptr<index_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  // convert lengths to offsets
  const auto input_offsets = asynchronous_exclusive_cumsum_gpu(lengths_contig);
  const auto output_offsets =
      asynchronous_complete_cumsum_gpu(permuted_lengths.flatten());
  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size = output_offsets[-1].item<int64_t>();
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

DLL_PUBLIC Tensor expand_into_jagged_permute_cuda(
    const Tensor& permute,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    int64_t output_size) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      permute, input_offsets, output_offsets);

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
DLL_PUBLIC std::tuple<
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
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(lengths, indices);

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
DLL_PUBLIC
std::tuple<Tensor, Tensor, c10::optional<Tensor>, c10::optional<Tensor>>
bucketize_sparse_features_cuda(
    const Tensor& lengths,
    const Tensor& indices,
    const bool bucketize_pos,
    const int64_t my_size,
    const c10::optional<Tensor>& weights) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(lengths, indices);

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
    const int32_t T,
    const bool broadcast_lengths) {
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
  const int32_t input_segment_start =
      broadcast_lengths ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets[b];

  for (int32_t i = threadIdx.x; i < num_ads_b; i += blockDim.x) {
    reordered_cat_ad_lengths[output_segment_start + i] = broadcast_lengths
        ? cat_ad_lengths[input_segment_start]
        : cat_ad_lengths[input_segment_start + i];
  }
}

DLL_PUBLIC Tensor reorder_batched_ad_lengths_gpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(cat_ad_lengths, batch_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cat_ad_lengths.get_device());

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = broadcast_lengths
      ? cat_ad_lengths.numel() / B
      : cat_ad_lengths.numel() / num_ads_in_batch;

  Tensor reordered_cat_ad_lengths = broadcast_lengths
      ? at::empty({T * num_ads_in_batch}, cat_ad_lengths.options())
      : at::empty_like(cat_ad_lengths);

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES(
      cat_ad_lengths.scalar_type(),
      "reorder_batched_ad_lengths_gpu_kernel",
      [&] {
        reorder_batched_ad_lengths_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                batch_offsets
                    .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                reordered_cat_ad_lengths
                    .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                T,
                broadcast_lengths);
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

    // if broadcast_indices is enabled, all the indices will be copies of the
    // first batch of the cat_ad_indices, this is useful for request-only
    // broadcast
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
    const int32_t T,
    const bool broadcast_indices) {
  const int32_t B = batch_offsets.size(0) - 1;
  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  const int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const auto num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const auto output_segment_offset_start =
      t * num_ads_in_batch + batch_offsets[b];
  const auto output_segment_start =
      reordered_cat_ad_offsets[output_segment_offset_start];
  const int32_t input_segment_offset_start =
      broadcast_indices ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_end = broadcast_indices
      ? input_segment_offset_start + 1
      : input_segment_offset_start + num_ads_b;
  const auto input_segment_start = cat_ad_offsets[input_segment_offset_start];
  const auto input_segment_end = cat_ad_offsets[input_segment_offset_end];
  const auto num_elements = input_segment_end - input_segment_start;

  if (broadcast_indices) {
    for (int32_t i = threadIdx.x; i < num_ads_b * num_elements;
         i += blockDim.x) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i % num_elements];
    }
  } else {
    // Idea: we want to copy the entire segment of size sum_a(length_{b, t, a})
    // from starting point (given by cat_ad_offsets[b, t])
    // to end point (given by reordered_cat_ad_indices[t][b])
    for (int32_t i = threadIdx.x; i < input_segment_end - input_segment_start;
         i += blockDim.x) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i];
    }
  }
}

DLL_PUBLIC Tensor reorder_batched_ad_indices_gpu(
    const Tensor& cat_ad_offsets,
    const Tensor& cat_ad_indices,
    const Tensor& reordered_cat_ad_offsets,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    const int64_t num_indices_after_broadcast) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets, batch_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(cat_ad_offsets.get_device());

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = (reordered_cat_ad_offsets.numel() - 1) / num_ads_in_batch;
  Tensor reordered_cat_ad_indices;
  if (broadcast_indices) {
    TORCH_CHECK_GE(num_indices_after_broadcast, 0);
    reordered_cat_ad_indices =
        at::empty({num_indices_after_broadcast}, cat_ad_indices.options());
  } else {
    reordered_cat_ad_indices = at::empty_like(cat_ad_indices);
  }

  const dim3 threads(32, 32);
  const dim3 blocks((B * T + 32 - 1) / 32);

  AT_DISPATCH_ALL_TYPES(
      cat_ad_indices.scalar_type(),
      "reorder_batched_ad_indices_gpu_kernel_1",
      [&] {
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
                  T,
                  broadcast_indices);
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
    const scalar_t* __restrict__ weight, // N * sum(E) * 1 (embedding
                                         // dimension is 1)
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
    auto idx = LDG(&indices[indices_start + l]);
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
      indices.scalar_type(), "batched_unary_embeddings_forward_kernel", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            weight.scalar_type(),
            "batched_unary_embeddings_forward_kernel",
            [&] {
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
// 1) Assign output elements to different threads. Each thread scan all
// indices
//    corresponding to the table it owns but only accumulate gradients when an
//    index value matches with the output element it owns.
//    A challenge is each thread need to binary search to map from [0 ..
//    sum_E] to table id.
// 2) Densify indices and offsets to create [B, sum_E] matrix. Then, do
// batched
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
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask) {
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
  const auto info =
      reinterpret_cast<const uint32_t*>(sorted_infos)[segment_start];
  int t = info >> info_B_num_bits;

  at::acc_type<scalar_t, true> grad_sum = 0.0;
  for (int32_t sl = 0; sl < SL; ++sl) {
    const auto b =
        reinterpret_cast<const uint32_t*>(sorted_infos)[segment_start + sl] &
        info_B_mask;
    grad_sum += grad_output[(n * B + b) * T + t];
  }

  index_t table_offset = table_offsets[t];
  index_t sum_E = table_offsets[T];
  int64_t idx = linear_index - table_offset;
  grad_weight[n * sum_E + table_offset + idx] = grad_sum;
}

DLL_PUBLIC Tensor batched_unary_embeddings_backward_cuda(
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& table_offsets,
    const Tensor& offsets,
    const Tensor& indices) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      grad_output, weight, table_offsets, offsets, indices);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  // N: number of tasks, T: number of tables, B: batch size
  const int32_t N = grad_output.size(0);
  const int32_t B = grad_output.size(1);
  const int32_t T = grad_output.size(2);
  TORCH_CHECK(N > 0);
  TORCH_CHECK(B > 0);
  TORCH_CHECK(T > 0);

  int32_t info_B_num_bits;
  uint32_t info_B_mask;
  std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(B, T);

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
          table_offsets,
          total_hash_size_bits,
          indices,
          offsets,
          false, // nobag
          c10::optional<Tensor>(),
          info_B_num_bits,
          info_B_mask);

  int threads = std::min<int32_t>(sorted_linear_indices_run.numel(), 512);
  dim3 blocks(
      cuda_calc_xblock_count(sorted_linear_indices_run.numel(), threads), N);
  auto grad_weight = at::zeros_like(weight);
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "batched_unary_embeddings_backward_kernel", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            grad_output.scalar_type(),
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
                      info_B_num_bits,
                      info_B_mask);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return grad_weight;
}

DLL_PUBLIC Tensor lengths_range_cuda(
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
    index_t segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    index_t input_start = input_offsets[permute[t] * B + b];
    for (int32_t i = threadIdx.x; i < segment_length; i += blockDim.x) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        permuted_weights[output_start + i] = weights[input_start + i];
      }
    }
  }
}

DLL_PUBLIC std::tuple<Tensor, Tensor, c10::optional<Tensor>>
permute_sparse_features_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& indices,
    const c10::optional<Tensor>& weights) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, indices, weights);

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
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(permuted_lengths.flatten());
  int64_t permuted_lengths_sum = indices.numel();

  /* TODO: Remove the condition protecting the slow path because even when the
   * condition below is true permuted_lengths.sum() could still be needed. For
   * instance if there are three features with indices `[0, 1, 2]`, `permute`
   * can be `[0, 1, 1]` for which permuted lengths sum would be needed to
   * create permuted_{indices, weights} and `permuted_lengths_sum =
   * indices.numel() or weights.numdel() would be incorrect.
   */
  if (num_features != num_output_features) {
    permuted_lengths_sum = output_offsets[-1].item<int64_t>();
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
DLL_PUBLIC Tensor permute102_baddbmm_permute102_cuda(
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

#ifdef __HIP_PLATFORM_HCC__
  float alpha = 1.0f;
  float beta = 1.0f;

  auto Btype = HIPBLAS_R_16F;
  auto ldb = n;
  auto strideB = n * k;

  auto Atype = HIPBLAS_R_16F;
  auto lda = k * batch_size;
  auto strideA = k;

  auto Ctype = HIPBLAS_R_16F;
  auto ldc = n * batch_size;
  auto strideC = n;

  auto computeType = HIPBLAS_R_32F;

  auto result = hipblasGemmStridedBatchedEx(
      handle,
      HIPBLAS_OP_N,
      HIPBLAS_OP_N,
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
      HIPBLAS_GEMM_DEFAULT);
  TORCH_CHECK(result == CUBLAS_STATUS_SUCCESS);
  return C;
}
#else
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
#endif

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

DLL_PUBLIC std::tuple<Tensor, Tensor> permute_sequence_embeddings_cuda(
    const Tensor& permute,
    const Tensor& lengths,
    const Tensor& embeddings) {
  // wrapper for permute_2D_sparse_data_cuda, kept for BC
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(permute, lengths, embeddings);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(embeddings.get_device());

  TORCH_CHECK(
      lengths.dim() == 2,
      "The dimension of lengths tensor should be equal to 2"
      "to correctly infer number of features and batch size.")

  Tensor permuted_lengths;
  Tensor permuted_embeddings;
  c10::optional<Tensor> weights_dummy;
  c10::optional<int64_t> permuted_lengths_sum_dummy;

  const auto T = permute.numel();
  const auto B = lengths.size(1);

  permuted_lengths = at::empty({T, B}, lengths.options());

  // ignore the third element in the tuple
  std::tie(permuted_lengths, permuted_embeddings, std::ignore) =
      fbgemm_gpu::permute_2D_sparse_data_cuda(
          permute,
          lengths,
          embeddings,
          weights_dummy,
          permuted_lengths_sum_dummy);

  return {permuted_lengths, permuted_embeddings};
}

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

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());

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

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        t_in_c.scalar_type(),
        "pack_segments_cuda-packing",
        [&] {
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

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(data.get_device());

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

constexpr int MAX_ELEMENTS_PER_THREAD = 4;

template <
    typename index_t,
    typename scalar_t,
    int UNROLL_FACTOR,
    bool indices_sorted>
__global__ __launch_bounds__(kMaxThreads) void index_select_2d_kernel(
    const at::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> input,
    const at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits>
        orig_indices,
    at::PackedTensorAccessor64<scalar_t, 2> output,
    TORCH_DSA_KERNEL_ARGS) {
  const int N = indices.size(0);
  const int input_size = input.size(0);
  const int D = input.size(1);
  CUDA_KERNEL_ASSERT2(output.size(0) == N);

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const index_t src_idx = indices[row];
    const int64_t dst_idx = indices_sorted ? orig_indices[row] : row;
    CUDA_KERNEL_ASSERT2(src_idx < input_size);
    int col;
    for (col = threadIdx.x * UNROLL_FACTOR;
         col < D / UNROLL_FACTOR * UNROLL_FACTOR;
         col += blockDim.x * UNROLL_FACTOR) {
#pragma unroll
      for (int i = 0; i < UNROLL_FACTOR; i++) {
        output[dst_idx][col + i] = LDG(&input[src_idx][col + i]);
      }
    }
    for (; col < D; ++col) {
      output[dst_idx][col] = LDG(&input[src_idx][col]);
    }
  }
}

template <typename index_t, typename scalar_t, int UNROLL_FACTOR>
__global__
__launch_bounds__(kMaxThreads) void index_add_2d_with_unique_indices_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        out_grad,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        unique_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        orig_indices,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor64<scalar_t, 2> in_deduped_grad,
    const int stride_D,
    const int rounded_D,
    const int remaining_D,
    const bool consecutive_indices,
    const int consecutive_range_start) {
  const int start_offset = blockIdx.x == 0 ? 0 : offsets[blockIdx.x - 1];
  const int end_offset = offsets[blockIdx.x];
  index_t dst_idx = consecutive_indices ? blockIdx.x + consecutive_range_start
                                        : unique_indices[blockIdx.x];
  const bool has_remainder = blockIdx.y == blockDim.y - 1 && remaining_D > 0 &&
      threadIdx.x < remaining_D;

  // Buffer for storing temporary results
  scalar_t sum[MAX_ELEMENTS_PER_THREAD];
  for (int i = 0; i < MAX_ELEMENTS_PER_THREAD; i++) {
    sum[i] = 0;
  }

  scalar_t sum_remainder = 0;

  // Each thread block processes max of stride_D elements
  int start_D = (blockIdx.y * stride_D) + (threadIdx.x * UNROLL_FACTOR);

  // For each row
  for (int row = start_offset; row < end_offset; row++) {
    int64_t src_idx = orig_indices[row];
    int col, i;
    for (col = start_D, i = 0; col < start_D + stride_D && col < rounded_D;
         col += blockDim.x * UNROLL_FACTOR, i += UNROLL_FACTOR) {
#pragma unroll
      for (int j = 0; j < UNROLL_FACTOR; j++) {
        sum[i + j] += LDG(&out_grad[src_idx][col + j]);
      }
    }
    if (has_remainder) {
      sum_remainder += LDG(&out_grad[src_idx][rounded_D + threadIdx.x]);
    }
  } // for each row

  // Write results to global memory
  int col, i;
  for (col = start_D, i = 0; col < start_D + stride_D && col < rounded_D;
       col += blockDim.x * UNROLL_FACTOR, i += UNROLL_FACTOR) {
#pragma unroll
    for (int j = 0; j < UNROLL_FACTOR; j++) {
      in_deduped_grad[dst_idx][col + j] = sum[i + j];
    }
  }
  if (has_remainder) {
    in_deduped_grad[dst_idx][rounded_D + threadIdx.x] += sum_remainder;
  }
}

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

template <
    typename scalar_t,
    int ndim,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
at::PackedTensorAccessor32<scalar_t, ndim, PtrTraits>
dummy_packed_accessor32() {
  std::array<int64_t, ndim> zeros{};
  return {nullptr, zeros.data(), zeros.data()};
}

template <
    typename scalar_t,
    int ndim,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
at::PackedTensorAccessor64<scalar_t, ndim, PtrTraits>
dummy_packed_accessor64() {
  std::array<int64_t, ndim> zeros{};
  return {nullptr, zeros.data(), zeros.data()};
}

DLL_PUBLIC Tensor index_select_cuda(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& orig_indices,
    const bool indices_sorted) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int N = indices.size(0);
  auto output_shape = input.sizes().vec();
  output_shape[0] = N;

  if (input.numel() == 0 || N == 0) {
    return at::empty(output_shape, input.options());
  }

  Tensor input_reshaped = input.reshape({input.size(0), -1});
  const int D = input_reshaped.size(1);

  Tensor output = at::empty({N, D}, input_reshaped.options());

  const int UNROLL_FACTOR = 2;

#define LAUNCH_INDEX_SELECT(INDICES_SORTED)                                   \
  TORCH_DSA_KERNEL_LAUNCH(                                                    \
      (index_select_2d_kernel<                                                \
          index_t,                                                            \
          scalar_t,                                                           \
          UNROLL_FACTOR,                                                      \
          INDICES_SORTED>),                                                   \
      cuda_calc_xblock_count(N, 1),                                           \
      std::min(div_round_up(D, UNROLL_FACTOR), kMaxThreads),                  \
      0,                                                                      \
      at::cuda::getCurrentCUDAStream(),                                       \
      input_reshaped.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(), \
      indices.packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),         \
      INDICES_SORTED                                                          \
          ? orig_indices                                                      \
                .packed_accessor64<int64_t, 1, at::RestrictPtrTraits>()       \
          : dummy_packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),     \
      output.packed_accessor64<scalar_t, 2>());

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "index_add_2d_kernel_1", [&] {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input_reshaped.scalar_type(), "index_add_2d_kernel_2", [&] {
          if (indices_sorted) {
            LAUNCH_INDEX_SELECT(true)
          } else {
            LAUNCH_INDEX_SELECT(false)
          }
        });
  });

#undef LAUNCH_INDEX_SELECT

  return output.reshape(output_shape);
}

DLL_PUBLIC Tensor index_add_with_unique_indices_cuda(
    const Tensor& grad_output,
    const Tensor& sorted_indices,
    const Tensor& orig_indices,
    std::vector<int64_t>& input_shape,
    const int consecutive_range_start,
    const int consecutive_range_length) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const int N = grad_output.size(0);

  if (grad_output.numel() == 0) {
    return at::zeros(input_shape, grad_output.options());
  }

  const Tensor grad_output_reshaped = grad_output.reshape({N, -1});
  const int D = grad_output_reshaped.size(1);

  TORCH_CHECK(sorted_indices.size(0) == N);

  Tensor input_grad = at::zeros({input_shape[0], D}, grad_output.options());
  bool consecutive_indices =
      consecutive_range_start >= 0 && consecutive_range_length > 0;

  AT_DISPATCH_INDEX_TYPES(
      sorted_indices.scalar_type(), "index_add_2d_kernel_1", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            grad_output.scalar_type(), "index_add_2d_kernel_2", [&] {
              // UNROLL_FACTOR is determined based on the empirical study
              const int UNROLL_FACTOR = std::is_same<scalar_t, float>() ? 4 : 2;
              const int rounded_D = D / UNROLL_FACTOR * UNROLL_FACTOR;
              const int remaining_D = D - rounded_D;
              int block_size =
                  std::min(div_round_up(D, UNROLL_FACTOR), kMaxThreads);
              block_size = std::max(remaining_D, block_size);
              // Number of elements per block
              const int stride_D = MAX_ELEMENTS_PER_THREAD * block_size;

              int num_unique_indices;
              Tensor unique_indices, offsets;
              if (consecutive_indices) {
                TORCH_CHECK(
                    consecutive_range_start < input_shape[0] &&
                    consecutive_range_start + consecutive_range_length - 1 <
                        input_shape[0]);

                // Since indices are selected from consecutive range, we can
                // infer the number of unique indices from
                // consecutive_range_length
                num_unique_indices = consecutive_range_length;
                compute_frequency_sequence(
                    sorted_indices,
                    offsets,
                    consecutive_range_start,
                    num_unique_indices);
                offsets = offsets.cumsum(0);
              } else {
                Tensor unique_count;
                // Unique consecutive does D->H transfer internally
                // (enforcing synchronization between host and device)
                std::tie(unique_indices, std::ignore, unique_count) =
                    at::unique_consecutive(sorted_indices, false, true, 0);

                // This does D->H transfer
                num_unique_indices = unique_indices.numel();
                offsets = unique_count.cumsum(0);
              }

              const dim3 grid_size(
                  cuda_calc_xblock_count(num_unique_indices, 1),
                  (D + stride_D - 1) / stride_D,
                  1);

              index_add_2d_with_unique_indices_kernel<
                  index_t,
                  scalar_t,
                  UNROLL_FACTOR><<<
                  grid_size,
                  block_size,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  grad_output_reshaped
                      .packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                  consecutive_indices ? dummy_packed_accessor32<
                                            index_t,
                                            1,
                                            at::RestrictPtrTraits>()
                                      : unique_indices.packed_accessor32<
                                            index_t,
                                            1,
                                            at::RestrictPtrTraits>(),
                  orig_indices
                      .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                  offsets
                      .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                  input_grad.packed_accessor64<scalar_t, 2>(),
                  stride_D, // Pass constants as kernel args
                  rounded_D,
                  remaining_D,
                  consecutive_indices,
                  consecutive_range_start);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
      });
  return input_grad.reshape(input_shape);
}

// TODO: Update UNROLL_FACTOR
constexpr int GROUP_INDEX_SELECT_UNROLL_FACTOR = 1;
constexpr int GROUP_INDEX_SELECT_COLS_PER_WARP =
    GROUP_INDEX_SELECT_UNROLL_FACTOR * kWarpSize;
// GROUP_INDEX_SELECT_COLS_PER_WARP must be power of two
constexpr int GROUP_INDEX_SELECT_LOG_COLS_PER_WARP =
    log2_calc<GROUP_INDEX_SELECT_COLS_PER_WARP>::value;

int get_group_index_select_cols_per_warp() {
  return GROUP_INDEX_SELECT_COLS_PER_WARP;
}

template <
    typename index_t,
    typename scalar_t,
    bool USE_INDEX_SELECT,
    bool USE_VAR_COLS,
    int UNROLL_FACTOR,
    int COLS_PER_WARP,
    int LOG_COLS_PER_WARP>
__global__
__launch_bounds__(kMaxThreads) void group_index_select_or_add_2d_kernel(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const int64_t max_indices,
    const int64_t num_work_rows, // number of rows to work on per member
    const int64_t group_size) {
  const auto total_num_warps = warp_offsets_group[group_size];
  for (int64_t warp_id = threadIdx.y * gridDim.x + blockIdx.x;
       warp_id < total_num_warps;
       warp_id += gridDim.x * blockDim.y) {
    int32_t member_id, member_warp_id, num_cols, warps_per_row;
    if (USE_VAR_COLS) {
      __shared__ int member_ids[kMaxThreads / kWarpSize];
      if (threadIdx.x == 0) {
        binary_search_range(
            &member_ids[threadIdx.y],
            warp_offsets_group + 1,
            warp_id,
            group_size);
      }
      syncwarp();
      member_id = member_ids[threadIdx.y];
      num_cols = num_cols_group[member_id];
      warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
      member_warp_id = warp_id - warp_offsets_group[member_id];
    } else {
      // All columns are the same
      num_cols = num_cols_group[0];
      warps_per_row = (num_cols + COLS_PER_WARP - 1) >> LOG_COLS_PER_WARP;
      member_id = warp_id / (warps_per_row * num_work_rows);
      member_warp_id = warp_id - (member_id * warps_per_row * num_work_rows);
    }
    const auto row = member_warp_id / warps_per_row;
    const auto col_offset =
        ((member_warp_id % warps_per_row) << LOG_COLS_PER_WARP) +
        (threadIdx.x * UNROLL_FACTOR);
    scalar_t* input =
        reinterpret_cast<scalar_t*>(input_ptrs[member_id]) + col_offset;
    scalar_t* output =
        reinterpret_cast<scalar_t*>(output_ptrs[member_id]) + col_offset;
    index_t* indices = reinterpret_cast<index_t*>(indices_ptrs[member_id]);
    const index_t idx = indices[row];
    CUDA_KERNEL_ASSERT(idx < max_indices)
#pragma unroll
    for (int i = 0; i < UNROLL_FACTOR && col_offset + i < num_cols; i++) {
      // Compile time conditional
      if (USE_INDEX_SELECT) {
        output[row * num_cols + i] = LDG(&input[idx * num_cols + i]);
      } else {
        gpuAtomicAddNoReturn(
            &output[idx * num_cols + i], input[row * num_cols + i]);
      }
    }
  }
}

DLL_PUBLIC void group_index_select_or_add_cuda(
    const int64_t* input_ptrs,
    const int64_t* output_ptrs,
    const int64_t* indices_ptrs,
    const int64_t* warp_offsets_group,
    const int32_t* num_cols_group,
    const c10::ScalarType& input_scalar_type,
    const c10::ScalarType& indices_scalar_type,
    const c10::DeviceIndex& device,
    const int max_indices,
    const int num_work_rows,
    const int64_t total_num_warps,
    const int group_size,
    const bool use_index_select,
    const bool use_var_cols) {
  if (group_size == 0) {
    return;
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(device);

  // Partition work based on num_work_rows
  uint32_t num_warps_per_threadblock = kMaxThreads / kWarpSize;
  uint32_t max_grid_size =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount * 8;
  uint32_t grid_size = std::min(
      cuda_calc_xblock_count(total_num_warps, num_warps_per_threadblock),
      max_grid_size);
  dim3 block_size(kWarpSize, num_warps_per_threadblock, 1);

#define INVOKE_GROUP_INDEX_SELECT_OR_ADD(USE_INDEX_SELECT, USE_VAR_COLS) \
  group_index_select_or_add_2d_kernel<                                   \
      index_t,                                                           \
      scalar_t,                                                          \
      USE_INDEX_SELECT,                                                  \
      USE_VAR_COLS,                                                      \
      GROUP_INDEX_SELECT_UNROLL_FACTOR,                                  \
      GROUP_INDEX_SELECT_COLS_PER_WARP,                                  \
      GROUP_INDEX_SELECT_LOG_COLS_PER_WARP>                              \
      <<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(  \
          input_ptrs,                                                    \
          output_ptrs,                                                   \
          indices_ptrs,                                                  \
          warp_offsets_group,                                            \
          num_cols_group,                                                \
          max_indices,                                                   \
          num_work_rows,                                                 \
          group_size)

  AT_DISPATCH_INDEX_TYPES(
      indices_scalar_type, "group_index_select_2d_wrapper_1", [&] {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input_scalar_type, "group_index_select_2d_wrapper_2", [&] {
              if (use_index_select) {
                if (use_var_cols) {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(true, true);
                } else {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(true, false);
                }
              } else {
                if (use_var_cols) {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(false, true);
                } else {
                  INVOKE_GROUP_INDEX_SELECT_OR_ADD(false, false);
                }
              }
            });
      });

#undef INVOKE_GROUP_INDEX_SELECT_OR_ADD
}

// Copied from cupy/random/_kernels.py v11
// (commit id 420e41fd41157d4cf526b0e94eb86a3f8eb5a231)

typedef struct {
  unsigned int xor128[4];
  double gauss;
  int has_gauss; // !=0: gauss contains a gaussian deviate

#ifdef CUPY_USE_BINOMIAL
  int has_binomial; // !=0: following parameters initialized for binomial
  /* The rk_state structure has been extended to store the following
   * information for the binomial generator. If the input values of n or p
   * are different than nsave and psave, then the other parameters will be
   * recomputed. RTK 2005-09-02 */
  int nsave, m;
  double psave, r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
#endif
} rk_state;

__device__ void rk_seed(unsigned long long s, rk_state* state) {
  for (int i = 1; i <= 4; i++) {
    s = 1812433253U * (s ^ (s >> 30)) + i;
    state->xor128[i - 1] = s;
  }
  state->has_gauss = 0;
#ifdef CUPY_USE_BINOMIAL
  state->has_binomial = 0;
#endif
}

__device__ unsigned long rk_random(rk_state* state) {
  unsigned int* xor128 = state->xor128;
  unsigned int t = xor128[0] ^ (xor128[0] << 11);
  xor128[0] = xor128[1];
  xor128[1] = xor128[2];
  xor128[2] = xor128[3];
  return xor128[3] ^= (xor128[3] >> 19) ^ t ^ (t >> 8);
}

__device__ double rk_double(rk_state* state) {
  /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
  int a = rk_random(state) >> 5, b = rk_random(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

__device__ long rk_zipf(rk_state* state, double a) {
  double am1, b;

  am1 = a - 1.0;
  b = pow(2.0, am1);
  while (1) {
    double T, U, V, X;

    U = 1.0 - rk_double(state);
    V = rk_double(state);
    X = floor(pow(U, -1.0 / am1));

    if (X < 1.0) {
      continue;
    }

    T = pow(1.0 + 1.0 / X, am1);
    if (V * X * (T - 1.0) / (b - 1.0) <= T / b) {
      return (long)X;
    }
  }
}

__global__ void zipf_kernel(
    const double a,
    const int64_t seed,
    at::PackedTensorAccessor64<long, 1, at::RestrictPtrTraits> y) {
  rk_state internal_state;
  auto N = y.size(0);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    rk_seed(seed + i, &internal_state);
    y[i] = rk_zipf(&internal_state, a);
  }
}

DLL_PUBLIC Tensor
zipf_cuda(const double a, const int64_t n, const int64_t seed) {
  Tensor y = at::empty(
      {n},
      at::TensorOptions().dtype(at::kLong).device(
          at::kCUDA, at::cuda::current_device()));
  zipf_kernel<<<
      cuda_calc_xblock_count(n, kMaxThreads),
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      a, seed, y.packed_accessor64<long, 1, at::RestrictPtrTraits>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return y;
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("zipf_cuda(float a, int n, int seed) -> Tensor");
  DISPATCH_TO_ALL("zipf_cuda", fbgemm_gpu::zipf_cuda);
}
