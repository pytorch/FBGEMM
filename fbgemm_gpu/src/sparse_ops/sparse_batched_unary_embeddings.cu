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

} // namespace fbgemm_gpu
