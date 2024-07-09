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
#include "fbgemm_gpu/fbgemm_tensor_accessor.h"
#include "fbgemm_gpu/permute_multi_embedding_function.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Kernerl for permute pooled embedding op.
// This kernel is moving D elements per warp.
template <typename scalar_t, bool reverse_permute>
__global__ void permute_multi_embs_kernel(
    const scalar_t** __restrict__ inputs,
    const scalar_t** __restrict__ outputs,
    const pta::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits>
        permutes,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        in_lengths,
    const pta::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        out_lengths,
    const int32_t batch_size,
    const int32_t permute_size) {
  // workers in a warp handle exact one permute (of a feature/key)
  const int32_t worker_id = threadIdx.x;
  const int32_t permute_id = threadIdx.y + blockIdx.x * blockDim.x;
  const int32_t batch_id = blockIdx.y + gridDim.y * blockIdx.z;
  if (batch_id >= batch_size) {
    return;
  }
  if (permute_id >= permute_size) {
    return;
  }

  // parse permutes
  int32_t in_tensor, out_tensor, in_start, out_start, length, next;
  if (reverse_permute) {
    out_tensor = permutes[permute_id][0];
    in_tensor = permutes[permute_id][1];
    out_start = permutes[permute_id][2];
    in_start = permutes[permute_id][3];
  } else {
    in_tensor = permutes[permute_id][0];
    out_tensor = permutes[permute_id][1];
    in_start = permutes[permute_id][2];
    out_start = permutes[permute_id][3];
  }
  length = permutes[permute_id][4];
  next = permutes[permute_id][5];

  if (worker_id >= length) {
    return;
  }
  if (reverse_permute && next < 0) {
    return;
  }

  // locate the batch_id
  int32_t in_length = in_lengths[in_tensor];
  scalar_t* input_ptr = (scalar_t*)inputs[in_tensor];
  input_ptr += batch_id * in_length;

  int32_t out_length = out_lengths[out_tensor];
  scalar_t* output_ptr = (scalar_t*)outputs[out_tensor];
  output_ptr += batch_id * out_length;

  if (fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(
          &output_ptr[out_start]) &&
      fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(
          &input_ptr[in_start])) {
    constexpr int32_t vec_size = 4;
    const int32_t loop_end = round_down(length, vec_size);
    for (int32_t i = worker_id * vec_size; i < loop_end;
         i += blockDim.x * vec_size) {
      fbgemm_gpu::Vec4T<scalar_t>::copy(
          &input_ptr[in_start + i], &output_ptr[out_start + i]);
    }
    // Use elementwise access for the last incomplete vector.
    for (int32_t i = loop_end + worker_id; i < length; i += blockDim.x) {
      output_ptr[out_start + i] = input_ptr[in_start + i];
    }
  } else { // Fallback if not aligned.
    for (int32_t i = worker_id; i < length; i += blockDim.x) {
      output_ptr[out_start + i] = input_ptr[in_start + i];
    }
  }

  // for reverse_permute (backward) with next
  while (reverse_permute && next > 0 && next < permute_size) {
    in_tensor = permutes[next][1];
    in_start = permutes[next][3];
    length = permutes[next][4];
    next = -permutes[next][5];

    int32_t in_length = in_lengths[in_tensor];
    scalar_t* input_ptr = (scalar_t*)inputs[in_tensor];
    input_ptr += batch_id * in_length;

    for (int32_t i = worker_id; i < length; i += blockDim.x) {
      output_ptr[out_start + i] += input_ptr[in_start + i];
    }
  }
}

template <typename index_t>
Tensor from_vec(const std::vector<index_t> input) {
  const auto int_opts =
      torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
  Tensor output = at::empty({static_cast<index_t>(input.size())}, int_opts);
  // Ensure that output is contiguous
  TORCH_CHECK(output.is_contiguous());
  std::memcpy(
      output.data_ptr<index_t>(), input.data(), input.size() * sizeof(index_t));
  return output;
}

template <typename scalar_t>
Tensor tensors_ptr(const at::TensorList& tensors) {
  auto size = tensors.size();
  Tensor ptr_tensor = at::empty(
      {static_cast<long>(size * sizeof(scalar_t*))},
      at::TensorOptions().dtype(tensors[0].scalar_type()).pinned_memory(true));

  // Ensure that ptr_tensor is contiguous
  TORCH_CHECK(ptr_tensor.is_contiguous());
  auto tp = reinterpret_cast<scalar_t**>(ptr_tensor.data_ptr());
  for (int32_t i = 0; i < tensors.size(); i++) {
    tp[i] = tensors[i].data_ptr<scalar_t>();
  }
  // Ensure that ptr_tensor is contiguous
  TORCH_CHECK(ptr_tensor.is_contiguous());
  return ptr_tensor;
}

std::vector<Tensor> permute_multi_embedding_gpu(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const std::vector<int64_t>& out_lengths,
    const bool& reverse_permute) {
  int32_t num_of_input_tensors = in_shapes.size(0);
  int32_t num_of_output_tensors = out_lengths.size();
  int32_t batch_size = pooled_embs[0].size(0);
  int32_t permute_size = permutes.size(0);

  // check input tensors
  std::vector<Tensor> inputs;
  inputs.reserve(pooled_embs.size());
  for (int32_t i = 0; i < num_of_input_tensors; i++) {
    Tensor cont_tensor = pooled_embs[i].contiguous();
    inputs.push_back(cont_tensor);
    TENSORS_ON_SAME_DEVICE(cont_tensor, pooled_embs[i]);
    TENSORS_ON_SAME_DEVICE(pooled_embs[i], pooled_embs[0]);
    CUDA_DEVICE_GUARD(cont_tensor);
  }
  TORCH_CHECK(in_shapes.is_contiguous());
  TORCH_CHECK(out_shapes.is_contiguous());

  // initiate output tensors
  std::vector<Tensor> outputs;
  outputs.reserve(num_of_output_tensors);
  for (int32_t i = 0; i < num_of_output_tensors; i++) {
    Tensor output =
        at::empty({batch_size, out_lengths[i]}, pooled_embs[0].options());
    outputs.push_back(output);
  }
  auto device = pooled_embs[0].device();

  // This kernel is moving one feature/key per warp.
  // We are launching ( permute_size//warp_per_block, batch_size, ?)
  // blocks. The grid z dimension is also used by batch_size in case it's
  // greater than 65535.
  const int32_t warp_per_block =
      fbgemm_gpu::kMaxThreads / fbgemm_gpu::kWarpSize;
  const int32_t max_grid_dim = 32768; // The CUDA maximum is 65535, not 1<<N.
  const dim3 block_dim(fbgemm_gpu::kWarpSize, warp_per_block);
  const dim3 grid_dim(
      fbgemm_gpu::div_round_up(permute_size, warp_per_block),
      std::min(static_cast<int32_t>(batch_size), max_grid_dim),
      (batch_size + max_grid_dim - 1) / max_grid_dim);

  FBGEMM_DISPATCH_FLOATING_TYPES(
      pooled_embs[0].scalar_type(), "permute_multi_embedding", [&] {
        Tensor in_ptr = tensors_ptr<scalar_t>(inputs);
        Tensor out_ptr = tensors_ptr<scalar_t>(outputs);
        in_ptr = in_ptr.to(device, /*non_blocking=*/true);
        out_ptr = out_ptr.to(device, /*non_blocking=*/true);
        const auto permute_kernel = reverse_permute
            ? permute_multi_embs_kernel<scalar_t, true>
            : permute_multi_embs_kernel<scalar_t, false>;
        const auto stream = at::cuda::getCurrentCUDAStream();
#ifdef FBGEMM_GPU_MEMCHECK
        const char* func_name = "permute_multi_embs_kernel";
#endif
        permute_kernel<<<grid_dim, block_dim, 0, stream>>>(
            reinterpret_cast<const scalar_t**>(in_ptr.data_ptr()),
            reinterpret_cast<const scalar_t**>(out_ptr.data_ptr()),
            MAKE_PTA_WITH_NAME(func_name, permutes, int32_t, 2, 32),
            MAKE_PTA_WITH_NAME(func_name, in_shapes, int32_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, out_shapes, int32_t, 1, 32),
            batch_size,
            permute_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return outputs;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "permute_multi_embedding_function",
    fbgemm_gpu::permute_multi_embedding_gpu);
