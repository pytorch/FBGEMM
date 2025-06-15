/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/block/block_scan.hpp>
#else
#include <cub/block/block_scan.cuh>
#endif
#include "common.cuh"

static constexpr uint32_t kMaxThreads = 1024;

#ifdef __HIP_PLATFORM_AMD__
namespace cub = hipcub;
#endif

namespace fbgemm_gpu {

C10_ALWAYS_INLINE uint32_t next_power_of_2(uint32_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

template <
    typename val_t,
    typename = std::enable_if_t<std::is_integral<val_t>::value>>
struct BlockPrefixCallbackOp {
  val_t running_total;

  __device__ BlockPrefixCallbackOp(val_t running_total)
      : running_total(running_total) {}

  __device__ val_t operator()(val_t block_aggregate) {
    val_t old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <
    typename val_t,
    uint32_t nthreads_per_block,
    typename = std::enable_if_t<std::is_integral<val_t>::value>>
__global__ __launch_bounds__(kMaxThreads) void _batched_complete_cumsum_kernel(
    const at::PackedTensorAccessor64<val_t, 2, at::RestrictPtrTraits> values,
    const uint32_t len,
    const uint32_t items_per_thread,
    at::PackedTensorAccessor64<val_t, 2, at::RestrictPtrTraits> out) {
  using BlockScan = cub::BlockScan<val_t, nthreads_per_block>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  BlockPrefixCallbackOp<val_t> prefix_op(0);
  if (threadIdx.x == 0) {
    out[blockIdx.x][0] = 0;
  }

  for (uint32_t offset = 0; offset < items_per_thread; offset++) {
    uint32_t i = offset * nthreads_per_block + threadIdx.x;
    val_t data = 0;
    if (i < len) {
      data = (val_t)values[blockIdx.x][i];
    }
    BlockScan(temp_storage).InclusiveSum(data, data, prefix_op);
    cub::CTA_SYNC();
    if (i < len) {
      out[blockIdx.x][i + 1] = data;
    }
  }
}

at::Tensor asynchronous_batched_complete_cumsum_gpu(const at::Tensor& values) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  TORCH_CHECK(values.dim() == 2, "values of batched_complete_cumsum must be 2")
  TORCH_CHECK(
      values.size(0) <= UINT32_MAX,
      "values.size(0) must be no higher than UINT32_MAX")
  TORCH_CHECK(
      values.size(1) <= UINT32_MAX,
      "values.size(1) must be no higher than UINT32_MAX")

  const uint32_t B = values.size(0);
  const uint32_t len = values.size(1);
  const uint32_t nthreads_per_block =
      min(max(next_power_of_2(len), 64), kMaxThreads);
  const uint32_t items_per_thread = div_round_up(len, nthreads_per_block);

  auto cumsum = at::empty({B, len + 1}, values.options());

  AT_DISPATCH_INTEGRAL_TYPES(
      values.scalar_type(), "batched_complete_cumsum_cuda_input1", [&] {
        using val_t = scalar_t;
        if (nthreads_per_block == 64) {
          _batched_complete_cumsum_kernel<val_t, 64>
              <<<B, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
                  values.packed_accessor64<val_t, 2, at::RestrictPtrTraits>(),
                  len,
                  items_per_thread,
                  cumsum.packed_accessor64<val_t, 2, at::RestrictPtrTraits>());
        } else if (nthreads_per_block == 128) {
          _batched_complete_cumsum_kernel<val_t, 128>
              <<<B, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                  values.packed_accessor64<val_t, 2, at::RestrictPtrTraits>(),
                  len,
                  items_per_thread,
                  cumsum.packed_accessor64<val_t, 2, at::RestrictPtrTraits>());
        } else if (nthreads_per_block == 256) {
          _batched_complete_cumsum_kernel<val_t, 256>
              <<<B, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                  values.packed_accessor64<val_t, 2, at::RestrictPtrTraits>(),
                  len,
                  items_per_thread,
                  cumsum.packed_accessor64<val_t, 2, at::RestrictPtrTraits>());
        } else if (nthreads_per_block == 512) {
          _batched_complete_cumsum_kernel<val_t, 512>
              <<<B, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                  values.packed_accessor64<val_t, 2, at::RestrictPtrTraits>(),
                  len,
                  items_per_thread,
                  cumsum.packed_accessor64<val_t, 2, at::RestrictPtrTraits>());
        } else {
          _batched_complete_cumsum_kernel<val_t, 1024>
              <<<B, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                  values.packed_accessor64<val_t, 2, at::RestrictPtrTraits>(),
                  len,
                  items_per_thread,
                  cumsum.packed_accessor64<val_t, 2, at::RestrictPtrTraits>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return cumsum;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "asynchronous_batched_complete_cumsum",
    fbgemm_gpu::asynchronous_batched_complete_cumsum_gpu);
