/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <type_traits>
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh" // @manual
#include "fbgemm_gpu/split_embeddings_utils.cuh" // @manual
#include "fbgemm_gpu/utils/kernel_launcher.cuh" // @manual
#include "fbgemm_gpu/utils/ops_utils.h" // @manual
#include "fbgemm_gpu/utils/tensor_accessor_builder.h" // @manual
#ifdef USE_ROCM
#include <rocprim/device/device_radix_sort.hpp>
#endif
// clang-format off
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh" // @manual
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh" // @manual
// clang-format on

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

inline at::Tensor asynchronous_complete_cumsum(at::Tensor t_in) {
  CUDA_DEVICE_GUARD(t_in);

  size_t temp_storage_bytes = 0;
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // CUB only handles up to INT_MAX elements.
  TORCH_CHECK_LT(t_in.numel(), std::numeric_limits<int32_t>::max());
  TORCH_CHECK_EQ(t_in.dim(), 1);
  auto t_out = at::empty({t_in.numel() + 1}, t_in.options());
  t_out[0].zero_();
  AT_DISPATCH_INDEX_TYPES(
      t_in.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
        AT_CUDA_CHECK(
            FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
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
        AT_CUDA_CHECK(
            FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
                temp_storage.data_ptr(),
                temp_storage_bytes,
                t_in.const_data_ptr<index_t>(),
                t_out.data_ptr<index_t>() + 1,
                t_in.numel(),
                at::cuda::getCurrentCUDAStream()));
      });
  return t_out;
}

// TBE_ROCM_LINEARIZE_GROUP=8|16|32 (ROCm only): lanes per bag in
// linearize_index_kernel's bag loop. 0, the value for unset, invalid values and
// every CUDA build, selects the stock whole-wave loop. Read once per process.
static int32_t linearize_lane_group() {
#ifdef USE_ROCM
  static const int32_t g = [] {
    const char* v = std::getenv("TBE_ROCM_LINEARIZE_GROUP");
    if (v == nullptr) {
      return 0;
    }
    // the whole string must be the number: "8x" or "8 " is invalid
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (end == v || *end != '\0') {
      return 0;
    }
    return (x == 8 || x == 16 || x == 32) ? static_cast<int32_t>(x) : 0;
  }();
  return g;
#else
  return 0;
#endif
}

template <typename index_t, typename info_acc_t, bool nobag, bool vbe>
__global__ __launch_bounds__(kMaxThreads) void linearize_index_kernel(
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    pta::PackedTensorAccessor32<info_acc_t, 1, at::RestrictPtrTraits> infos,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    [[maybe_unused]] const uint32_t max_T,
    [[maybe_unused]] const uint32_t max_B,
    // Use a raw pointer to avoid creating dummy PackedTensorAccessor
    const uint32_t* const __restrict__ vbe_b_t_map,
    FixedDivisor fd,
    const int32_t lane_group) {
  const auto T = hash_size_cumsum.size(0) - 1;
  const auto total_B = offsets.size(0) - 1;
  const auto lane_id = threadIdx.x % fbgemm_gpu::kWarpSize;
  int32_t b;
  int32_t t;
  // Grid-stride over b_t so the kernel covers the whole workload even when the
  // launch caps grid.x -- ROCm does that to stay within the HIP 2^32
  // threads-per-launch limit. On CUDA the grid already spans total_B, so the
  // body executes exactly once per thread.
  //
  // The bound is warp-aligned (b_t - lane_id) so whole warps iterate together
  // and the shfl_sync collectives below always have every lane present; the
  // per-lane `valid` masking keeps out-of-range lanes inert. The comparison is
  // done in 64 bits so a large total_B cannot truncate.
  for (auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
       static_cast<int64_t>(b_t - lane_id) < total_B;
       b_t += gridDim.x * blockDim.x) {
    const bool valid = static_cast<int64_t>(b_t) < total_B;
    // info must be uint32_t (using auto will assign int32_t to info)
    uint32_t info = 0;

    if (vbe && valid) {
      info = vbe_b_t_map[b_t];
      reinterpret_cast<uint32_t*>(&t)[0] = info >> info_B_num_bits;
      reinterpret_cast<uint32_t*>(&b)[0] = info & info_B_mask;
    } else {
      fd.DivMod(b_t, &t, &b);
    }

    const index_t hash_offset = valid ? hash_size_cumsum[t] : -1;
    const auto indices_start = valid ? offsets[b_t] : -1;
    const auto L = valid ? offsets[b_t + 1] - indices_start : 0;

    // Compile-time conditional
    if (nobag) {
      for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
        const auto indices_start_warp = fbgemm_gpu::shfl_sync(indices_start, j);
        const auto t_warp = fbgemm_gpu::shfl_sync(t, j);
        const auto L_warp = fbgemm_gpu::shfl_sync(L, j);
        const index_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
        for (auto i = lane_id; static_cast<int64_t>(i) < L_warp;
             i += fbgemm_gpu::kWarpSize) {
          const index_t idx = __ldg(&indices[indices_start_warp + i]);
          const auto l_t = (indices_start_warp + i) * T + t_warp;
          infos[indices_start_warp + i] = l_t;
          linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
        }
      }
    } else {
      // Store t in upper (32 - DEFAULT_INFO_B_NUM_BITS).
      // Store b in lower (DEFAULT_INFO_B_NUM_BITS).
      if (!vbe && valid) {
        info = (reinterpret_cast<uint32_t*>(&t)[0] << info_B_num_bits) |
            reinterpret_cast<uint32_t*>(&b)[0];
      }
      if (lane_group <= 0 || lane_group >= fbgemm_gpu::kWarpSize) {
        for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
          const auto indices_start_warp =
              fbgemm_gpu::shfl_sync(indices_start, j);
          const auto info_warp = fbgemm_gpu::shfl_sync(info, j);
          const auto L_warp = fbgemm_gpu::shfl_sync(L, j);
          const index_t hash_offset_warp =
              fbgemm_gpu::shfl_sync(hash_offset, j);
          for (auto i = lane_id; static_cast<int64_t>(i) < L_warp;
               i += fbgemm_gpu::kWarpSize) {
            const index_t idx = __ldg(&indices[indices_start_warp + i]);
            reinterpret_cast<uint32_t*>(&infos[0])[indices_start_warp + i] =
                info_warp;
            linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
          }
        }
      } else {
        // lane_group lanes per bag, kWarpSize / lane_group bags at a time
        const int32_t sub = lane_id / lane_group;
        const int32_t sub_lane = lane_id % lane_group;
        const int32_t n_sub = fbgemm_gpu::kWarpSize / lane_group;
        for (int32_t j0 = 0; j0 < fbgemm_gpu::kWarpSize; j0 += n_sub) {
          const int32_t j = j0 + sub;
          const auto indices_start_warp =
              fbgemm_gpu::shfl_sync(indices_start, j);
          const auto info_warp = fbgemm_gpu::shfl_sync(info, j);
          const auto L_warp = fbgemm_gpu::shfl_sync(L, j);
          const index_t hash_offset_warp =
              fbgemm_gpu::shfl_sync(hash_offset, j);
          for (int32_t i = sub_lane; i < L_warp; i += lane_group) {
            const index_t idx = __ldg(&indices[indices_start_warp + i]);
            reinterpret_cast<uint32_t*>(&infos[0])[indices_start_warp + i] =
                info_warp;
            linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
          }
        }
      }
    }
  } // for b_t (warp-aligned grid-stride loop)
}

template <typename index_t, typename info_acc_t>
__global__
__launch_bounds__(kMaxThreads) void linearize_index_index_select_kernel(
    const pta::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        indices,
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        total_L_offsets,
    pta::PackedTensorAccessor32<info_acc_t, 1, at::RestrictPtrTraits> infos,
    pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices,
    FixedDivisor fd,
    int32_t fixed_L_per_warp) {
  const auto T = hash_size_cumsum.size(0) - 1;
  // Flattened (B * T) extent this kernel walks. Computed in 64 bits so it
  // cannot truncate.
  const auto total_b_t = static_cast<int64_t>(fd.D()) * T;
  const auto lane_id = threadIdx.x % fbgemm_gpu::kWarpSize;
  int32_t b;
  int32_t t;
  // Grid-stride over b_t so the kernel covers the whole workload even when the
  // launch caps grid.x -- ROCm does that to stay within the HIP 2^32
  // threads-per-launch limit. On CUDA the grid already spans the workload, so
  // the body executes exactly once per thread.
  //
  // The bound is warp-aligned (b_t - lane_id) so whole warps iterate together
  // and the shfl_sync collectives below always have every lane present; the
  // per-lane `valid` / `t < T` masking keeps out-of-range lanes inert.
  for (auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
       static_cast<int64_t>(b_t - lane_id) < total_b_t;
       b_t += gridDim.x * blockDim.x) {
    // Tail lanes of the final warp can hold b_t >= total_b_t, because the loop
    // bound is warp-aligned. Mask them explicitly rather than relying on
    // `t < T`: DivMod takes an int32_t, so a b_t past INT32_MAX converts to a
    // negative t, which would satisfy `t < T` and index total_L_offsets /
    // hash_size_cumsum out of bounds. These lanes must stay in the loop (no
    // `continue`) so every lane is still present for the shfl_sync below.
    const bool valid = static_cast<int64_t>(b_t) < total_b_t;

    fd.DivMod(b_t, &t, &b);

    index_t hash_offset = -1;
    index_t indices_start = -1;
    int32_t L = 0;
    int32_t L_start = 0;
    if (valid && t < T) {
      const auto total_L_start = total_L_offsets[t];
      const auto total_L = total_L_offsets[t + 1] - total_L_start;
      L_start = b * fixed_L_per_warp;
      if (L_start < total_L) {
        hash_offset = hash_size_cumsum[t];
        indices_start = total_L_start + L_start;
        L = (total_L - L_start >= fixed_L_per_warp) ? fixed_L_per_warp
                                                    : (total_L - L_start);
      }
    }

    // Compile-time conditional
    for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
      const index_t indices_start_warp =
          fbgemm_gpu::shfl_sync(indices_start, j);
      const auto t_warp = fbgemm_gpu::shfl_sync(t, j);
      const auto L_warp = fbgemm_gpu::shfl_sync(L, j);
      const auto L_start_warp = fbgemm_gpu::shfl_sync(L_start, j);
      const index_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
      for (auto i = lane_id; static_cast<int64_t>(i) < L_warp;
           i += fbgemm_gpu::kWarpSize) {
        const index_t idx = __ldg(&indices[indices_start_warp + i]);
        // l is the relative l in the feature (i.e., the first l in the feature
        // is 0)
        const int64_t l_t = (L_start_warp + i) * T + t_warp;
        infos[indices_start_warp + i] = l_t;
        linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
      }
    }
  } // for b_t (warp-aligned grid-stride loop)
}

DLL_PUBLIC std::tuple<
    Tensor /*linear_indices*/,
    Tensor /*linear_indices_sorted*/,
    Tensor /*infos_sorted*/,
    Tensor /*sorted_linear_indices_run*/,
    Tensor /*sorted_linear_indices_run_lengths*/,
    Tensor /*sorted_linear_indices_num_runs*/,
    Tensor /*sorted_linear_indices_cumulative_run_lengths*/>
transpose_embedding_input(
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    bool nobag,
    const std::optional<Tensor>& vbe_b_t_map,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask,
    const int64_t total_unique_indices,
    const bool is_index_select,
    const std::optional<Tensor>& total_L_offsets,
    const int64_t fixed_L_per_warp,
    const int64_t num_warps_per_feature) {
  const bool vbe = vbe_b_t_map.has_value();
  TORCH_CHECK(nobag || !vbe || info_B_num_bits > 0);
  TORCH_CHECK(!vbe || info_B_mask > 0);
  TORCH_CHECK(
      !is_index_select || (fixed_L_per_warp > 0 && num_warps_per_feature > 0));

  const auto T = hash_size_cumsum.size(0) - 1;
  const auto total_B =
      !is_index_select ? (offsets.size(0) - 1) : (num_warps_per_feature * T);

  TORCH_CHECK(
      !is_index_select ||
      (total_L_offsets.has_value() &&
       total_L_offsets.value().numel() == T + 1));

  // The linearize_index kernels flatten (B, T) into a 32-bit thread index b_t
  // and hand it to FixedDivisor::DivMod, which takes an int32_t. Assert the
  // bound here so an oversized workload fails loudly instead of silently
  // wrapping the grid-stride index inside the kernel.
  TORCH_CHECK_LE(total_B, std::numeric_limits<int32_t>::max());

  auto infos = at::empty_like(
      indices,
      indices.options().dtype(
          (nobag || is_index_select) ? at::kLong : at::kInt));
  auto infos_sorted = at::empty_like(infos);
  auto linear_indices = at::empty_like(indices);
  auto linear_indices_sorted = at::empty_like(indices);

  Tensor sorted_linear_indices_run;
  Tensor sorted_linear_indices_run_lengths;
  Tensor sorted_linear_indices_num_runs;

  using at::RestrictPtrTraits;

#define INVOKE_LINEARIZE_INDEX_KERNEL(INFO_ACC_T, NOBAG)                   \
  {                                                                        \
    const auto linearize_index_kernel_ =                                   \
        (vbe ? linearize_index_kernel<index_t, INFO_ACC_T, NOBAG, true>    \
             : linearize_index_kernel<index_t, INFO_ACC_T, NOBAG, false>); \
    FBGEMM_LAUNCH_KERNEL(                                                  \
        linearize_index_kernel_,                                           \
        utils::cuda::cap_grid_dim_x_from_workload(                         \
            total_B, kMaxThreads, at::cuda::getCurrentCUDAStream()),       \
        kMaxThreads,                                                       \
        0,                                                                 \
        at::cuda::getCurrentCUDAStream(),                                  \
        PTA_B(hash_size_cumsum, int64_t, 1, 32),                           \
        PTA_B(indices, index_t, 1, 32),                                    \
        PTA_B(offsets, index_t, 1, 32),                                    \
        PTA_B(infos, INFO_ACC_T, 1, 32),                                   \
        PTA_B(linear_indices, index_t, 1, 32),                             \
        info_B_num_bits,                                                   \
        info_B_mask,                                                       \
        (1u << (DEFAULT_INFO_NUM_BITS - info_B_num_bits)) - 1,             \
        (1u << info_B_num_bits) - 1,                                       \
        vbe ? reinterpret_cast<uint32_t*>(vbe_b_t_map.value().data_ptr())  \
            : nullptr,                                                     \
        FixedDivisor(total_B / T),                                         \
        linearize_lane_group());                                           \
  }

#define FBGEMM_DISPATCH_INFO_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                               \
      TYPE,                                         \
      NAME,                                         \
      AT_PRIVATE_CASE_TYPE_USING_HINT(              \
          at::ScalarType::Int, info_t, __VA_ARGS__) \
          AT_PRIVATE_CASE_TYPE_USING_HINT(          \
              at::ScalarType::Long, info_t, __VA_ARGS__))

  FBGEMM_DISPATCH_INFO_TYPES(
      infos.scalar_type(), "transpose_embedding_input_1", [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "transpose_embedding_input_2", [&] {
              if (!is_index_select) {
                if (!nobag) {
                  INVOKE_LINEARIZE_INDEX_KERNEL(int32_t, false);
                } else {
                  INVOKE_LINEARIZE_INDEX_KERNEL(int64_t, true);
                }
              } else {
                // index_select is a special case of TBE (dense, nobag, with
                // fixed_L_per_warp)
                FBGEMM_LAUNCH_KERNEL(
                    (linearize_index_index_select_kernel<index_t, int64_t>),
                    utils::cuda::cap_grid_dim_x_from_workload(
                        total_B, kMaxThreads, at::cuda::getCurrentCUDAStream()),
                    kMaxThreads,
                    0,
                    at::cuda::getCurrentCUDAStream(),
                    PTA_B(hash_size_cumsum, int64_t, 1, 32),
                    PTA_B(indices, index_t, 1, 32),
                    PTA_B(total_L_offsets.value(), index_t, 1, 32),
                    PTA_B(infos, int64_t, 1, 32),
                    PTA_B(linear_indices, index_t, 1, 32),
                    FixedDivisor(total_B / T),
                    fixed_L_per_warp);
              }
              {
                size_t temp_storage_bytes = 0;
#ifndef USE_ROCM
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
                        nullptr,
                        temp_storage_bytes,
                        linear_indices.const_data_ptr<index_t>(),
                        linear_indices_sorted.data_ptr<index_t>(),
                        infos.data_ptr<info_t>(),
                        infos_sorted.data_ptr<info_t>(),
                        linear_indices.numel(),
                        0,
                        total_hash_size_bits,
                        at::cuda::getCurrentCUDAStream()));
                auto temp_storage = at::empty(
                    {static_cast<int64_t>(temp_storage_bytes)},
                    indices.options().dtype(at::kByte));
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
                        temp_storage.data_ptr(),
                        temp_storage_bytes,
                        linear_indices.data_ptr<index_t>(),
                        linear_indices_sorted.data_ptr<index_t>(),
                        infos.data_ptr<info_t>(),
                        infos_sorted.data_ptr<info_t>(),
                        linear_indices.numel(),
                        0,
                        total_hash_size_bits,
                        at::cuda::getCurrentCUDAStream()));
#else
	        using config = rocprim::radix_sort_config<
                rocprim::default_config,
                rocprim::default_config,
                rocprim::default_config,
                400000>;
                using config7 = rocprim::radix_sort_config<
                rocprim::default_config,
                rocprim::default_config,
                rocprim::radix_sort_onesweep_config<
                    rocprim::kernel_config<1024, 32>,
                    rocprim::kernel_config<1024, 12>,
                    7,
                    rocprim::block_radix_rank_algorithm::match>,
                400000>;
                static const bool sort_radix7_env = [] {
                  const char* v = std::getenv("TBE_ROCM_SORT_RADIX_BITS");
                  return v != nullptr && v[0] == '7' && v[1] == '\0';
                }();
                const bool sort_radix7 = sort_radix7_env &&
                    (total_hash_size_bits + 6) / 7 == (total_hash_size_bits + 7) / 8;
                auto sort_pairs = [&](auto config_tag) {
                using C = decltype(config_tag);
	        AT_CUDA_CHECK(rocprim::radix_sort_pairs<C>(
                    nullptr,
                    temp_storage_bytes,
                    linear_indices.data_ptr<index_t>(),
                    linear_indices_sorted.data_ptr<index_t>(),
                    infos.data_ptr<info_t>(),
                    infos_sorted.data_ptr<info_t>(),
                    linear_indices.numel(),
                    0,
                    total_hash_size_bits,
                    at::cuda::getCurrentCUDAStream(),
                    false));
                auto temp_storage = at::empty(
                    {static_cast<int64_t>(temp_storage_bytes)},
                    indices.options().dtype(at::kByte));
                AT_CUDA_CHECK(rocprim::radix_sort_pairs<C>(
                    temp_storage.data_ptr(),
                    temp_storage_bytes,
                    linear_indices.data_ptr<index_t>(),
                    linear_indices_sorted.data_ptr<index_t>(),
                    infos.data_ptr<info_t>(),
                    infos_sorted.data_ptr<info_t>(),
                    linear_indices.numel(),
                    0,
                    total_hash_size_bits,
                    at::cuda::getCurrentCUDAStream(),
                    false));
                };
                // the 7-bit config is sized for 4-byte keys and values; int64 instantiations keep the stock config
                // (selected by type, since this code is not a template and `if constexpr` would not discard it)
                using config7_or_stock = std::conditional_t<
                    sizeof(index_t) == 4 && sizeof(info_t) == 4,
                    config7,
                    config>;
                if (sort_radix7) {
                  sort_pairs(config7_or_stock{});
                } else {
                  sort_pairs(config{});
                }
#endif
              }
              if (total_unique_indices != -1) {
                TORCH_CHECK(total_unique_indices >= 0);
                sorted_linear_indices_run =
                    at::empty({total_unique_indices}, indices.options());
                sorted_linear_indices_run_lengths = at::zeros(
                    {total_unique_indices}, indices.options().dtype(at::kInt));
              } else {
                sorted_linear_indices_run = at::empty_like(indices);
                sorted_linear_indices_run_lengths =
                    at::zeros_like(indices, indices.options().dtype(at::kInt));
              }
              sorted_linear_indices_num_runs =
                  at::zeros({1}, indices.options().dtype(at::kInt));

              {
                size_t temp_storage_bytes = 0;
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                        nullptr,
                        temp_storage_bytes,
                        linear_indices_sorted.data_ptr<index_t>(),
                        sorted_linear_indices_run.data_ptr<index_t>(),
                        sorted_linear_indices_run_lengths.data_ptr<int32_t>(),
                        sorted_linear_indices_num_runs.data_ptr<int32_t>(),
                        linear_indices_sorted.numel(),
                        at::cuda::getCurrentCUDAStream()));
                // Allocate temporary storage
                auto temp_storage = at::empty(
                    {static_cast<int64_t>(temp_storage_bytes)},
                    indices.options().dtype(at::kByte));
                // Run encoding
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRunLengthEncode::Encode(
                        temp_storage.data_ptr(),
                        temp_storage_bytes,
                        linear_indices_sorted.data_ptr<index_t>(),
                        sorted_linear_indices_run.data_ptr<index_t>(),
                        sorted_linear_indices_run_lengths.data_ptr<int32_t>(),
                        sorted_linear_indices_num_runs.data_ptr<int32_t>(),
                        linear_indices_sorted.numel(),
                        at::cuda::getCurrentCUDAStream()));
              }
            });
      });

  auto sorted_linear_indices_cumulative_run_lengths =
      asynchronous_complete_cumsum(sorted_linear_indices_run_lengths);

#undef INVOKE_LINEARIZE_INDEX_KERNEL
#undef FBGEMM_DISPATCH_INFO_TYPES

  return {
      linear_indices,
      linear_indices_sorted,
      infos_sorted,
      sorted_linear_indices_run,
      sorted_linear_indices_run_lengths,
      sorted_linear_indices_num_runs,
      sorted_linear_indices_cumulative_run_lengths};
}
