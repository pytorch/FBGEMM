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

DLL_PUBLIC Tensor asynchronous_inclusive_cumsum_gpu(const Tensor& t_in) {
  TENSOR_ON_CUDA_GPU(t_in);
  CUDA_DEVICE_GUARD(t_in);

  if (t_in.numel() == 0) {
    return at::empty_like(t_in);
  }

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
  CUDA_DEVICE_GUARD(t_in);

  if (t_in.numel() == 0) {
    return at::empty_like(t_in);
  }

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
  CUDA_DEVICE_GUARD(t_in);

  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  TORCH_CHECK(t_in.dim() == 1 || t_in.dim() == 2);
  if (t_in.dim() == 1) {
    // CUB only handles up to INT_MAX elements.
    TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
    auto t_out = at::empty({t_in.numel() + 1}, t_in.options());
    t_out[0].zero_();

    if (t_in.numel() == 0) {
      return t_out;
    }

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

    if (t_in.numel() == 0) {
      return t_out;
    }

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

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "asynchronous_exclusive_cumsum",
    fbgemm_gpu::asynchronous_exclusive_cumsum_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "asynchronous_complete_cumsum",
    fbgemm_gpu::asynchronous_complete_cumsum_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "asynchronous_inclusive_cumsum",
    fbgemm_gpu::asynchronous_inclusive_cumsum_gpu);
