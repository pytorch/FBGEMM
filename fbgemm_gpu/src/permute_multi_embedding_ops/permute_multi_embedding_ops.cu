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

#include "fbgemm_gpu/permute_multi_embedding_function.h"
#include "fbgemm_gpu/utils/cuda_prelude.cuh"
#include "fbgemm_gpu/utils/tensor_accessor.h"
#include "fbgemm_gpu/utils/vec4.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

// Kernel for permute pooled embedding op.
// This kernel is moving D elements per warp.
template <typename scalar_t, bool reverse_permute>
__global__ void permute_multi_embs_kernel(
    const scalar_t** __restrict__ inputs,
    scalar_t** __restrict__ outputs,
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
  const int32_t permute_id = threadIdx.y + blockIdx.x * blockDim.y;
  const int32_t batch_id = blockIdx.y + gridDim.y * blockIdx.z;
  if (batch_id >= batch_size) {
    return;
  }
  if (permute_id >= permute_size) {
    return;
  }

  // parse permutes
  int32_t in_tensor, out_tensor, in_offset, out_offset, length, next;
  int32_t* __restrict__ pp = permutes[permute_id].data();
  if (reverse_permute) {
    out_tensor = pp[PermuteParam::in_tensor];
    in_tensor = pp[PermuteParam::out_tensor];
    out_offset = pp[PermuteParam::in_offset];
    in_offset = pp[PermuteParam::out_offset];
  } else {
    in_tensor = pp[PermuteParam::in_tensor];
    out_tensor = pp[PermuteParam::out_tensor];
    in_offset = pp[PermuteParam::in_offset];
    out_offset = pp[PermuteParam::out_offset];
  }
  length = pp[PermuteParam::length];
  next = pp[PermuteParam::next];

  if (worker_id >= length) {
    return;
  }
  if (reverse_permute && next < 0) {
    return;
  }

  // locate the batch_id
  const auto in_length = in_lengths[in_tensor];
  // Sometimes batch_id * length can go beyond 32-bit int (e.g., 3900 * 654060)
  // so cast them to int64_t.
  const scalar_t* __restrict__ input_ptr =
      reinterpret_cast<const scalar_t*>(inputs[in_tensor]) +
      static_cast<int64_t>(batch_id) * static_cast<int64_t>(in_length) +
      in_offset;

  const auto out_length = out_lengths[out_tensor];
  scalar_t* __restrict__ output_ptr =
      reinterpret_cast<scalar_t*>(outputs[out_tensor]) +
      static_cast<int64_t>(batch_id) * static_cast<int64_t>(out_length) +
      out_offset;

  if (fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(output_ptr) &&
      fbgemm_gpu::is_aligned<fbgemm_gpu::Vec4T<scalar_t>>(input_ptr)) {
    constexpr int32_t vec_size = 4;
    const int32_t loop_end = round_down(length, vec_size);
    for (int32_t i = worker_id * vec_size; i < loop_end;
         i += blockDim.x * vec_size) {
      fbgemm_gpu::Vec4T<scalar_t>::copy(&input_ptr[i], &output_ptr[i]);
    }
    // Use elementwise access for the last incomplete vector.
    for (int32_t i = loop_end + worker_id; i < length; i += blockDim.x) {
      output_ptr[i] = input_ptr[i];
    }
  } else { // Fallback if not aligned.
    for (int32_t i = worker_id; i < length; i += blockDim.x) {
      output_ptr[i] = input_ptr[i];
    }
  }

  // for reverse_permute (backward) with next
  while (reverse_permute && next > 0 && next < permute_size) {
    int32_t* __restrict__ pp = permutes[next].data();
    in_tensor = pp[PermuteParam::out_tensor];
    in_offset = pp[PermuteParam::out_offset];
    length = pp[PermuteParam::length];
    next = -pp[PermuteParam::next];

    const auto in_length = in_lengths[in_tensor];
    const scalar_t* input_ptr =
        reinterpret_cast<const scalar_t*>(inputs[in_tensor]) +
        static_cast<int64_t>(batch_id) * static_cast<int64_t>(in_length) +
        in_offset;

    for (int32_t i = worker_id; i < length; i += blockDim.x) {
      output_ptr[i] += input_ptr[i];
    }
  }
}

template <typename index_t>
Tensor from_vec(const std::vector<index_t>& input) {
  Tensor output = at::empty(
      {static_cast<index_t>(input.size())},
      torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
  // Ensure that output is contiguous
  TORCH_CHECK(output.is_contiguous());
  std::memcpy(
      output.data_ptr<index_t>(), input.data(), input.size() * sizeof(index_t));
  return output;
}

/// @ingroup permute pooled embedding function group
///
/// @brief generate the permutes arguments for permute_multi_embedding
/// operator
///
/// This is a helper function for the permute_multi_embedding operator. It
/// generates the required arguments for permute_multi_embedding operator.
/// including permutes, in_shapes, out_shapes, and out_lengths. It also move
/// the arguments (tensors) to the corresponding CUDA device.
///
/// **Example:**
/// ```python
/// # input arguments
/// keys = [["F1", "F2"], ["F3", "F4"]]
/// lengths = [[128, 128], [64, 32]]
/// batch_size = 1024
/// values = [torch.randn(batch_size, 256), torch.randn(batch_size, 96)]
///
/// # target output KTs
/// groups = [["F1", "F3"], ["F2", "F4"]]
///
/// # generate permutes
/// permutes, in_shapes, out_shapes, out_lengths = kt_regroup_arguments(keys,
/// lengths, groups)
///
/// # permute and regroup
/// permuted_values = permute_multi_embedding(values, permutes, in_shapes,
/// out_shapes, lengths)
/// ```
///
///
/// @param emb one of the tensors from KTs' values
/// @param keys List[List[str]], each string represents a feature/key in a KT
/// a list of keys represents a KT
/// @param lengths List[List[int64_t]], each int represents the length of a
/// feature/key in a KT, and a list of lengths represents a KT
/// @param groups List[List[str]], each string represents a feature/key in an
/// output KT a list of strings represents one output KT
/// @return tuple of permutes, in_shapes, out_shapes and output_lengths. See the
/// inputs of permute_multi_embedding for more details. The output tensors
/// should be contiguous, and on the same device as the input tensor.
///
/// @note This operator doesn't need autograd since it's purely about index.
///
/// @warning this gpu/cuda version will move the output tensors to corresponding
/// CUDA device.
///
std::tuple<Tensor, Tensor, Tensor, std::vector<int64_t>>
kt_regroup_arguments_gpu(
    const Tensor& emb,
    const std::vector<std::vector<std::string>>& keys,
    const std::vector<std::vector<int64_t>>& lengths,
    const std::vector<std::vector<std::string>>& groups) {
  auto [permutes, in_lengths, out_lengths] =
      kt_regroup_arguments_impl(keys, lengths, groups);
  CUDA_DEVICE_GUARD(emb);
  auto device = emb.device();
  auto pt = from_vec<int32_t>(permutes)
                .view({-1, PermuteParam::size})
                .to(device, true);
  auto in_shapes = from_vec<int32_t>(in_lengths).to(device, true);
  auto out_shapes = from_vec<int32_t>(out_lengths).to(device, true);
  std::vector<int64_t> out(out_lengths.begin(), out_lengths.end());
  return {pt, in_shapes, out_shapes, out};
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

std::vector<Tensor> permute_multi_embedding_function_gpu(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const c10::IntArrayRef out_lengths,
    const bool& reverse_permute) {
  // we assume that there's at least one input tensor in the list
  // it should be enforced from the caller side who has the knowledge.
  TORCH_CHECK(pooled_embs.size() > 0);
  CUDA_DEVICE_GUARD(pooled_embs[0]);
  TENSORS_ON_SAME_DEVICE(permutes, pooled_embs[0]);
  TENSORS_ON_SAME_DEVICE(permutes, in_shapes);
  TENSORS_ON_SAME_DEVICE(permutes, out_shapes);
  TORCH_CHECK(in_shapes.is_contiguous());
  TORCH_CHECK(out_shapes.is_contiguous());

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
    TORCH_CHECK(cont_tensor.is_contiguous());
  }

  // initiate output tensors
  std::vector<Tensor> outputs;
  outputs.reserve(num_of_output_tensors);
  const auto lengths = reinterpret_cast<const int64_t*>(out_lengths.data());
  for (int32_t i = 0; i < num_of_output_tensors; i++) {
    Tensor output =
        at::empty({batch_size, lengths[i]}, pooled_embs[0].options());
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
            reinterpret_cast<scalar_t**>(out_ptr.data_ptr()),
            MAKE_PTA_WITH_NAME(func_name, permutes, int32_t, 2, 32),
            MAKE_PTA_WITH_NAME(func_name, in_shapes, int32_t, 1, 32),
            MAKE_PTA_WITH_NAME(func_name, out_shapes, int32_t, 1, 32),
            batch_size,
            permute_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return outputs;
}
std::vector<Tensor> permute_multi_embedding_gpu(
    const at::TensorList& pooled_embs,
    const Tensor& permutes,
    const Tensor& in_shapes,
    const Tensor& out_shapes,
    const c10::IntArrayRef out_lengths) {
  return permute_multi_embedding_function_gpu(
      pooled_embs, permutes, in_shapes, out_shapes, out_lengths, false);
}

std::vector<Tensor> regroup_keyed_tensor_gpu(
    const at::TensorList& pooled_embs,
    const std::vector<std::vector<std::string>>& keys,
    const std::vector<std::vector<int64_t>>& lengths,
    const std::vector<std::vector<std::string>>& groups) {
  auto [permutes, in_shapes, out_shapes, out_lengths] =
      kt_regroup_arguments_gpu(pooled_embs[0], keys, lengths, groups);
  return permute_multi_embedding_function_gpu(
      pooled_embs, permutes, in_shapes, out_shapes, out_lengths, false);
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "permute_multi_embedding_function",
    fbgemm_gpu::permute_multi_embedding_function_gpu);

FBGEMM_OP_DISPATCH(
    CUDA,
    "permute_multi_embedding",
    fbgemm_gpu::permute_multi_embedding_gpu);

FBGEMM_OP_DISPATCH(
    CUDA,
    "kt_regroup_arguments",
    fbgemm_gpu::kt_regroup_arguments_gpu);

FBGEMM_OP_DISPATCH(
    CUDA,
    "regroup_keyed_tensor",
    fbgemm_gpu::regroup_keyed_tensor_gpu);
