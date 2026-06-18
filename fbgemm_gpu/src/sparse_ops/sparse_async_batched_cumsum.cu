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
#include <algorithm>
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
    const pta::PackedTensorAccessor64<val_t, 2, at::RestrictPtrTraits> values,
    const uint32_t B,
    const uint32_t len,
    const uint32_t items_per_thread,
    pta::PackedTensorAccessor64<val_t, 2, at::RestrictPtrTraits> out) {
  using BlockScan = cub::BlockScan<val_t, nthreads_per_block>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Grid-stride over rows so a capped grid (used on ROCm to avoid the 2^32
  // launch-side limit) still covers all B rows.
  for (uint32_t row = blockIdx.x; row < B; row += gridDim.x) {
    // RESET prefix scan state at the top of each iteration: running total
    // resets to 0 for each row.
    BlockPrefixCallbackOp<val_t> prefix_op(0);
    if (threadIdx.x == 0) {
      out[row][0] = 0;
    }

    for (uint32_t offset = 0; offset < items_per_thread; offset++) {
      uint32_t i = offset * nthreads_per_block + threadIdx.x;
      val_t data = 0;
      if (i < len) {
        data = (val_t)values[row][i];
      }
      BlockScan(temp_storage).InclusiveSum(data, data, prefix_op);

#if CUDA_VERSION >= 13000
      __syncthreads();
#else
      cub::CTA_SYNC();
#endif

      if (i < len) {
        out[row][i + 1] = data;
      }
    }

    // Ensure all threads finished using temp_storage before the next
    // outer-iteration's BlockScan overwrites it.
#if CUDA_VERSION >= 13000
    __syncthreads();
#else
    cub::CTA_SYNC();
#endif
  }
}

#define BATCHED_COMPLETE_CUMSUM_KERNEL(NTHREADS_PER_BLOCK)          \
  FBGEMM_LAUNCH_KERNEL(                                             \
      (_batched_complete_cumsum_kernel<val_t, NTHREADS_PER_BLOCK>), \
      num_blocks,                                                   \
      NTHREADS_PER_BLOCK,                                           \
      0,                                                            \
      at::cuda::getCurrentCUDAStream(),                             \
      PTA_B(values, val_t, 2, 64),                                  \
      B,                                                            \
      len,                                                          \
      items_per_thread,                                             \
      PTA_B(cumsum, val_t, 2, 64));

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
  const uint32_t nthreads_per_block = std::min<uint32_t>(
      std::max<uint32_t>(next_power_of_2(len), 64), kMaxThreads);
  const uint32_t items_per_thread = div_round_up(len, nthreads_per_block);

  auto cumsum = at::empty({B, len + 1}, values.options());

  // HIP enforces a hard limit of 2^32 total threads per launch (unlike CUDA,
  // which silently wraps). _batched_complete_cumsum_kernel grid-strides over
  // rows, so capping is correctness-preserving.
  // See: https://github.com/ROCm/hip/issues/2253
  const uint32_t num_blocks = utils::cuda::cap_grid_dim_x(
      B, nthreads_per_block, at::cuda::getCurrentCUDAStream());

  AT_DISPATCH_INTEGRAL_TYPES(
      values.scalar_type(), "batched_complete_cumsum_cuda_input1", [&] {
        using val_t = scalar_t;

        if (nthreads_per_block == 64) {
          BATCHED_COMPLETE_CUMSUM_KERNEL(64);

        } else if (nthreads_per_block == 128) {
          BATCHED_COMPLETE_CUMSUM_KERNEL(128);

        } else if (nthreads_per_block == 256) {
          BATCHED_COMPLETE_CUMSUM_KERNEL(256);

        } else if (nthreads_per_block == 512) {
          BATCHED_COMPLETE_CUMSUM_KERNEL(512);

        } else {
          BATCHED_COMPLETE_CUMSUM_KERNEL(1024);
        }
      });

  return cumsum;
}

#undef BATCHED_COMPLETE_CUMSUM_KERNEL

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "asynchronous_batched_complete_cumsum",
    fbgemm_gpu::asynchronous_batched_complete_cumsum_gpu);
