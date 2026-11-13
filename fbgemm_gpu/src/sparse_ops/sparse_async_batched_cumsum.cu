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

[[maybe_unused]] static constexpr uint32_t kMaxThreads = 1024;

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

#if defined(USE_ROCM) || CUDA_VERSION >= 13000
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
#if defined(USE_ROCM) || CUDA_VERSION >= 13000
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

#ifndef USE_ROCM
// Kernel parameters are limited to 4 KiB.
constexpr size_t kMaxKernelParamBytes = 4096;

template <typename val_t>
using Accessor1D = pta::PackedTensorAccessor64<val_t, 1, at::RestrictPtrTraits>;

// The parameter space left after the kernel's other arguments holds the output
// offsets of the segments, so launching needs no host-to-device copy and CUDA
// graph capture records them.
constexpr int64_t kMaxSegmentsPerLaunch =
    (kMaxKernelParamBytes - 2 * sizeof(Accessor1D<int64_t>) -
     2 * sizeof(int64_t)) /
        sizeof(int64_t) -
    1;

struct SegmentLayout {
  int64_t first_segment;
  int64_t num_segments;
  // Output offset of each segment in a launch, then the end of the last one.
  // Every segment's output starts with its 0, so these strictly increase.
  int64_t out_offsets[kMaxSegmentsPerLaunch + 1];
};
static_assert(
    2 * sizeof(Accessor1D<int64_t>) + sizeof(SegmentLayout) <=
    kMaxKernelParamBytes);

// The segment whose output holds position o, given each segment's output
// offset (strictly increasing, the first at or before o).
__device__ __forceinline__ int64_t find_segment(
    const int64_t* out_offsets,
    const int64_t num_segments,
    const int64_t o) {
  int64_t next = -1;
  binary_search_range(&next, out_offsets, o, num_segments);
  return next == -1 ? num_segments - 1 : next - 1;
}

// Turns the inclusive scan of the whole input into each segment's complete
// cumsum: a leading 0, then each scan value minus the one just before the
// segment. Unsigned arithmetic wraps exactly like a per-segment int32 sum.
template <typename val_t>
__global__ __launch_bounds__(kMaxThreads) void _rebase_segmented_cumsum_kernel(
    const Accessor1D<val_t> inclusive,
    Accessor1D<val_t> out,
    const SegmentLayout layout) {
  using uval_t = std::make_unsigned_t<val_t>;
  // Every thread searches the offsets, so stage them in shared memory.
  __shared__ int64_t out_offsets[kMaxSegmentsPerLaunch + 1];
  for (int64_t k = threadIdx.x; k <= layout.num_segments; k += blockDim.x) {
    out_offsets[k] = layout.out_offsets[k];
  }
  __syncthreads();

  const int64_t out_end = out_offsets[layout.num_segments];
  for (int64_t o = out_offsets[0] +
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       o < out_end;
       o += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t k = find_segment(out_offsets, layout.num_segments, o);
    const int64_t i = o - out_offsets[k];
    if (i == 0) {
      out[o] = 0;
      continue;
    }
    // The input is shorter by the leading 0s of this and earlier segments.
    const int64_t begin = out_offsets[k] - layout.first_segment - k;
    const uval_t base =
        begin > 0 ? static_cast<uval_t>(inclusive[begin - 1]) : uval_t{0};
    out[o] = static_cast<val_t>(
        static_cast<uval_t>(inclusive[begin + i - 1]) - base);
  }
}

template <typename val_t>
void rebase_segmented_cumsum(
    const at::Tensor& inclusive,
    const at::Tensor& output,
    at::IntArrayRef segment_sizes) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto total_segments = static_cast<int64_t>(segment_sizes.size());
  int64_t out_offset = 0;
  for (int64_t first_segment = 0; first_segment < total_segments;
       first_segment += kMaxSegmentsPerLaunch) {
    SegmentLayout layout;
    layout.first_segment = first_segment;
    layout.num_segments =
        std::min(kMaxSegmentsPerLaunch, total_segments - first_segment);
    for (const auto k : c10::irange(layout.num_segments)) {
      layout.out_offsets[k] = out_offset;
      out_offset += segment_sizes[first_segment + k] + 1;
    }
    layout.out_offsets[layout.num_segments] = out_offset;
    const int64_t num_outputs = out_offset - layout.out_offsets[0];
    FBGEMM_LAUNCH_KERNEL(
        (_rebase_segmented_cumsum_kernel<val_t>),
        utils::cuda::cap_grid_dim_x(
            div_round_up(num_outputs, kMaxThreads),
            kMaxThreads,
            stream,
            utils::cuda::BlockCapPolicy::Always),
        kMaxThreads,
        0,
        stream,
        PTA_B(inclusive, val_t, 1, 64),
        PTA_B(output, val_t, 1, 64),
        layout);
  }
}

#endif // USE_ROCM

at::Tensor asynchronous_segmented_complete_cumsum_gpu(
    const at::Tensor& t_in,
    at::IntArrayRef segment_sizes) {
  TENSOR_ON_CUDA_GPU(t_in);
  CUDA_DEVICE_GUARD(t_in);
  const auto total_size = check_segmented_cumsum_inputs(t_in, segment_sizes);

  const auto t_in_contig = t_in.expect_contiguous();
  if (segment_sizes.size() == 1) {
    return asynchronous_complete_cumsum_gpu(*t_in_contig);
  }
  auto output = at::empty(
      {total_size + static_cast<int64_t>(segment_sizes.size())},
      t_in.options());
  if (segment_sizes.empty()) {
    return output;
  }

#ifndef USE_ROCM
  // asynchronous_inclusive_cumsum_gpu scans fewer than INT32_MAX elements.
  if (total_size < std::numeric_limits<int32_t>::max()) {
    const auto inclusive = asynchronous_inclusive_cumsum_gpu(*t_in_contig);
    AT_DISPATCH_INDEX_TYPES(
        t_in.scalar_type(), "asynchronous_segmented_complete_cumsum_gpu", [&] {
          rebase_segmented_cumsum<index_t>(inclusive, output, segment_sizes);
        });
    return output;
  }
#endif // USE_ROCM

  // One asynchronous_complete_cumsum per segment: on ROCm, where the scan and
  // rebase above has only been validated on NVIDIA GPUs, and for inputs too
  // long for one scan.
  std::vector<at::Tensor> segment_cumsums;
  segment_cumsums.reserve(segment_sizes.size());
  for (const auto& segment : t_in_contig->split_with_sizes(segment_sizes)) {
    segment_cumsums.push_back(asynchronous_complete_cumsum_gpu(segment));
  }
  at::cat_out(output, segment_cumsums);
  return output;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "asynchronous_batched_complete_cumsum",
    fbgemm_gpu::asynchronous_batched_complete_cumsum_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "asynchronous_segmented_complete_cumsum",
    fbgemm_gpu::asynchronous_segmented_complete_cumsum_gpu);
