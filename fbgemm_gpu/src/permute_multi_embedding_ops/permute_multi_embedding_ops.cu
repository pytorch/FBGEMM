/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ostream>

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/permute_multi_embedding_function.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Kernerl for permute pooled embedding op.
// This kernel is moving D elements per warp.
template <typename scalar_t>
__global__ void permute_multi_embs_kernel(
    const scalar_t** __restrict__ inputs,
    const scalar_t** __restrict__ outputs,
    const int64_t* __restrict__ permutes,
    const int64_t* __restrict__ input_lengths,
    const int64_t* __restrict__ output_lengths,
    const int64_t batch_size,
    const int64_t permute_size,
    const bool reverse_permute) {
  // workers in a warp handle a feature
  const int32_t worker_id = threadIdx.x % warpSize;
  const int32_t worker_size = warpSize;
  const int32_t permute_id =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  const int32_t batch_id = blockIdx.y + gridDim.y * blockIdx.z;
  if (batch_id >= batch_size) {
    return;
  }
  if (permute_id >= permute_size) {
    return;
  }

  // parse permutes
  const int64_t params = 6;
  int64_t in_tensor, out_tensor, in_start, out_start, length, jump;
  if (reverse_permute) {
    out_tensor = permutes[params * permute_id];
    in_tensor = permutes[params * permute_id + 1];
    out_start = permutes[params * permute_id + 2];
    in_start = permutes[params * permute_id + 3];
  } else {
    in_tensor = permutes[params * permute_id];
    out_tensor = permutes[params * permute_id + 1];
    in_start = permutes[params * permute_id + 2];
    out_start = permutes[params * permute_id + 3];
  }
  length = permutes[params * permute_id + 4];
  jump = permutes[params * permute_id + 5];

  if (worker_id >= length) {
    return;
  }
  if (reverse_permute && jump < 0) {
    return;
  }

  // locate the batch_id
  int64_t in_length = input_lengths[in_tensor];
  scalar_t* input_ptr = (scalar_t*)inputs[in_tensor];
  input_ptr += batch_id * in_length;

  int64_t out_length = output_lengths[out_tensor];
  scalar_t* output_ptr = (scalar_t*)outputs[out_tensor];
  output_ptr += batch_id * out_length;

  // printf( // debug print
  //     "input_tensors[%ld][%ld][%d] = %f\n",
  //     in_tensor,
  //     batch_id,
  //     in_start + worker_id,
  //     input_ptr[in_start + worker_id]);
  if (fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(
          &output_ptr[out_start]) &&
      fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(
          &input_ptr[in_start])) {
    const int32_t vec_size = 4;
    const int32_t loop_end = length / (vec_size) * (vec_size);
    for (int32_t i = worker_id * vec_size; i < loop_end;
         i += worker_size * vec_size) {
      fbgemm_gpu::Vec4T<scalar_t>::copy(
          &input_ptr[in_start + i], &output_ptr[out_start + i]);
    }
    // Use elementwise access for the last incomplete vector.
    for (int32_t i = loop_end + worker_id; i < length; i += worker_size) {
      output_ptr[out_start + i] = input_ptr[in_start + i];
    }
  } else { // Fallback if not aligned.
    for (int32_t i = worker_id; i < length; i += worker_size) {
      output_ptr[out_start + i] = input_ptr[in_start + i];
    }
  }

  // for reverse_permute (backward) with jump
  while (reverse_permute && jump > 0 && jump < permute_size) {
    in_tensor = permutes[params * jump + 1];
    in_start = permutes[params * jump + 3];
    length = permutes[params * jump + 4];
    jump = -permutes[params * jump + 5];

    int64_t in_length = input_lengths[in_tensor];
    scalar_t* input_ptr = (scalar_t*)inputs[in_tensor];
    input_ptr += batch_id * in_length;

    for (int32_t i = worker_id; i < length; i += worker_size) {
      output_ptr[out_start + i] += input_ptr[in_start + i];
    }
  }
}

template <typename scalar_t>
const scalar_t** tensors_ptr(const at::TensorList& tensors) {
  auto size = tensors.size();
  std::vector<const scalar_t*> tp_vec;
  tp_vec.reserve(size);

  const scalar_t** ptrVec;
  cudaMalloc(&ptrVec, size * sizeof(scalar_t*));

  for (int32_t i = 0; i < size; i++) {
    tp_vec.push_back(tensors[i].data_ptr<scalar_t>());
  }
  cudaMemcpy(
      ptrVec, tp_vec.data(), size * sizeof(scalar_t*), cudaMemcpyHostToDevice);
  return ptrVec;
}

std::vector<Tensor> permute_multi_embedding_gpu(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const std::vector<int64_t>& lengths,
    const Tensor& input_lengths,
    const Tensor& output_lengths,
    const bool& reverse_permute = false) {
  int64_t num_of_input_tensors = pooled_embs.size();
  int64_t num_of_output_tensors = lengths.size();
  int64_t batch_size = pooled_embs[0].size(0);
  int64_t permute_size = permutes.size(0);
  std::vector<Tensor> inputs;
  inputs.reserve(num_of_input_tensors);
  for (int32_t i = 0; i < num_of_input_tensors; i++) {
    TENSORS_ON_SAME_DEVICE(pooled_embs[i], permutes);
    // how does this contiguous impact performance in forward and backward?
    inputs.push_back(pooled_embs[i].contiguous());
    TENSORS_ON_SAME_DEVICE(inputs[i], permutes);
  }
  TENSORS_ON_SAME_DEVICE(input_lengths, permutes);
  TENSORS_ON_SAME_DEVICE(input_lengths, output_lengths);

  // initiate output tensors
  std::vector<Tensor> outputs;
  outputs.reserve(num_of_output_tensors);
  for (int32_t i = 0; i < num_of_output_tensors; i++) {
    Tensor output =
        at::empty({batch_size, lengths[i]}, pooled_embs[0].options());
    outputs.push_back(output);
  }

  // This kernel is moving D elements per warp.
  // We are launching ( div_round_up(T, warp_per_block), B ) blocks.
  // The grid z dimension is also used by batch_size in case it's greater than
  // 65535.
  const int32_t warp_per_block =
      fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize;
  const int32_t max_grid_dim_y =
      32768; // The CUDA maximum is 65535, not a power of 2.
  const dim3 threads(fbgemm_gpu::kMaxThreads);
  const dim3 blocks(
      fbgemm_gpu::div_round_up(permute_size, warp_per_block),
      std::min(static_cast<int32_t>(batch_size), max_grid_dim_y),
      (batch_size + max_grid_dim_y - 1) / max_grid_dim_y);

  FBGEMM_DISPATCH_FLOATING_TYPES(
      pooled_embs[0].scalar_type(), "permute_multi_embedding", [&] {
        const scalar_t** input_ptr = tensors_ptr<scalar_t>(inputs);
        const scalar_t** output_ptr = tensors_ptr<scalar_t>(outputs);
        permute_multi_embs_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                (const scalar_t**)input_ptr,
                (const scalar_t**)output_ptr,
                (const int64_t*)permutes.data_ptr<int64_t>(),
                (const int64_t*)input_lengths.data_ptr<int64_t>(),
                (const int64_t*)output_lengths.data_ptr<int64_t>(),
                batch_size,
                permute_size,
                reverse_permute);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); // why this failed?
        cudaFree(input_ptr);
        cudaFree(output_ptr);
      });
  return outputs;
}

} // namespace fbgemm_gpu
