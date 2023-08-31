/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/split_embeddings_utils.cuh"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/ops_utils.h"

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#ifdef __HIP_PLATFORM_HCC__
#include <rocm_version.h>
#endif

inline at::Tensor asynchronous_complete_cumsum(at::Tensor t_in) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(t_in.get_device());
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
}

using Tensor = at::Tensor;

using namespace fbgemm_gpu;

template <typename index_t, typename info_acc_t, bool nobag, bool vbe>
__global__ __launch_bounds__(kMaxThreads) void linearize_index_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<info_acc_t, 1, at::RestrictPtrTraits> infos,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices,
    const int32_t info_B_num_bits,
    const uint32_t info_B_mask,
    const uint32_t max_T,
    const uint32_t max_B,
    // Use a raw pointer to avoid creating dummy PackedTensorAccessor
    const uint32_t* const __restrict__ vbe_b_t_map,
    FixedDivisor fd) {
  const int32_t T = hash_size_cumsum.size(0) - 1;
  auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b;
  int32_t t;
  const auto total_B = offsets.size(0) - 1;
  bool valid = b_t < total_B;
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
  const index_t indices_start = valid ? offsets[b_t] : -1;
  const int32_t L = valid ? offsets[b_t + 1] - indices_start : 0;
  const int32_t lane_id = threadIdx.x % fbgemm_gpu::kWarpSize;

  // Compile-time conditional
  if (nobag) {
    for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
      const index_t indices_start_warp =
          fbgemm_gpu::shfl_sync(indices_start, j);
      const int32_t t_warp = fbgemm_gpu::shfl_sync(t, j);
      const int32_t L_warp = fbgemm_gpu::shfl_sync(L, j);
      const index_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
      for (int32_t i = lane_id; i < L_warp; i += fbgemm_gpu::kWarpSize) {
        const index_t idx = __ldg(&indices[indices_start_warp + i]);
        const int64_t l_t = (indices_start_warp + i) * T + t_warp;
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
    for (int32_t j = 0; j < fbgemm_gpu::kWarpSize; ++j) {
      const index_t indices_start_warp =
          fbgemm_gpu::shfl_sync(indices_start, j);
      const uint32_t info_warp = fbgemm_gpu::shfl_sync(info, j);
      const int32_t L_warp = fbgemm_gpu::shfl_sync(L, j);
      const index_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
      for (int32_t i = lane_id; i < L_warp; i += fbgemm_gpu::kWarpSize) {
        const index_t idx = __ldg(&indices[indices_start_warp + i]);
        reinterpret_cast<uint32_t*>(&infos[0])[indices_start_warp + i] =
            info_warp;
        linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
      }
    }
  }
}

template <typename index_t, typename info_acc_t>
__global__
__launch_bounds__(kMaxThreads) void linearize_index_index_select_kernel(
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        hash_size_cumsum,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> indices,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        total_L_offsets,
    at::PackedTensorAccessor32<info_acc_t, 1, at::RestrictPtrTraits> infos,
    at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        linear_indices,
    FixedDivisor fd,
    int32_t fixed_L_per_warp) {
  const int32_t T = hash_size_cumsum.size(0) - 1;
  auto b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b;
  int32_t t;

  fd.DivMod(b_t, &t, &b);

  const int32_t lane_id = threadIdx.x % fbgemm_gpu::kWarpSize;

  index_t hash_offset = -1;
  index_t indices_start = -1;
  int32_t L = 0;
  int32_t L_start = 0;
  if (t < T) {
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
    const index_t indices_start_warp = fbgemm_gpu::shfl_sync(indices_start, j);
    const auto t_warp = fbgemm_gpu::shfl_sync(t, j);
    const auto L_warp = fbgemm_gpu::shfl_sync(L, j);
    const auto L_start_warp = fbgemm_gpu::shfl_sync(L_start, j);
    const index_t hash_offset_warp = fbgemm_gpu::shfl_sync(hash_offset, j);
    for (int32_t i = lane_id; i < L_warp; i += fbgemm_gpu::kWarpSize) {
      const index_t idx = __ldg(&indices[indices_start_warp + i]);
      // l is the relative l in the feature (i.e., the first l in the feature
      // is 0)
      const int64_t l_t = (L_start_warp + i) * T + t_warp;
      infos[indices_start_warp + i] = l_t;
      linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
    }
  }
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
    const c10::optional<Tensor>& vbe_b_t_map,
    const int64_t info_B_num_bits,
    const int64_t info_B_mask,
    const int64_t total_unique_indices,
    const bool is_index_select,
    const c10::optional<Tensor>& total_L_offsets,
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
  const auto linearize_index_kernel_ =                                     \
      (vbe ? linearize_index_kernel<index_t, INFO_ACC_T, NOBAG, true>      \
           : linearize_index_kernel<index_t, INFO_ACC_T, NOBAG, false>);   \
  linearize_index_kernel_<<<                                               \
      div_round_up(total_B, kMaxThreads),                                  \
      kMaxThreads,                                                         \
      0,                                                                   \
      at::cuda::getCurrentCUDAStream()>>>(                                 \
      hash_size_cumsum.packed_accessor32<index_t, 1, RestrictPtrTraits>(), \
      indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),          \
      offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),          \
      infos.packed_accessor32<INFO_ACC_T, 1, RestrictPtrTraits>(),         \
      linear_indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),   \
      info_B_num_bits,                                                     \
      info_B_mask,                                                         \
      (1u << (DEFAULT_INFO_NUM_BITS - info_B_num_bits)) - 1,               \
      (1u << info_B_num_bits) - 1,                                         \
      vbe ? reinterpret_cast<uint32_t*>(vbe_b_t_map.value().data_ptr())    \
          : nullptr,                                                       \
      FixedDivisor(total_B / T));                                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK()

  AT_DISPATCH_INDEX_TYPES(
      infos.scalar_type(), "transpose_embedding_input1", [&] {
        using info_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "transpose_embedding_input2", [&] {
              if (!is_index_select) {
                if (!nobag) {
                  INVOKE_LINEARIZE_INDEX_KERNEL(int32_t, false);
                } else {
                  INVOKE_LINEARIZE_INDEX_KERNEL(int64_t, true);
                }
              } else {
                // index_select is a special case of TBE (dense, nobag, with
                // fixed_L_per_warp)
                linearize_index_index_select_kernel<<<
                    div_round_up(total_B, kMaxThreads),
                    kMaxThreads,
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    hash_size_cumsum
                        .packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    total_L_offsets.value()
                        .packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    infos.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                    linear_indices
                        .packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                    FixedDivisor(total_B / T),
                    fixed_L_per_warp);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
              {
                size_t temp_storage_bytes = 0;
                AT_CUDA_CHECK(
                    FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs(
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
                        at::cuda::getCurrentCUDAStream(),
                        false));
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

  return {
      linear_indices,
      linear_indices_sorted,
      infos_sorted,
      sorted_linear_indices_run,
      sorted_linear_indices_run_lengths,
      sorted_linear_indices_num_runs,
      sorted_linear_indices_cumulative_run_lengths};
}

std::tuple<int64_t, int64_t>
get_infos_metadata(Tensor unused, int64_t B, int64_t T) {
  return adjust_info_B_num_bits(B, T);
}

DLL_PUBLIC std::tuple<int32_t, uint32_t> adjust_info_B_num_bits(
    int32_t B,
    int32_t T) {
  int32_t info_B_num_bits = DEFAULT_INFO_B_NUM_BITS;
  uint32_t info_B_mask = DEFAULT_INFO_B_MASK;
  uint32_t max_T = MAX_T;
  uint32_t max_B = MAX_B;
  bool invalid_T = T > max_T;
  bool invalid_B = B > max_B;

  TORCH_CHECK(
      !(invalid_T && invalid_B),
      "Not enough infos bits to accommodate T and B. Default num bits = ",
      DEFAULT_INFO_NUM_BITS);

  if (invalid_T) {
    // Reduce info_B_num_bits
    while (invalid_T && !invalid_B && info_B_num_bits > 0) {
      info_B_num_bits--;
      max_T = ((max_T + 1) << 1) - 1;
      max_B = ((max_B + 1) >> 1) - 1;
      invalid_T = T > max_T;
      invalid_B = B > max_B;
    }
  } else if (invalid_B) {
    // Increase info_B_num_bits
    while (!invalid_T && invalid_B && info_B_num_bits < DEFAULT_INFO_NUM_BITS) {
      info_B_num_bits++;
      max_T = ((max_T + 1) >> 1) - 1;
      max_B = ((max_B + 1) << 1) - 1;
      invalid_T = T > max_T;
      invalid_B = B > max_B;
    }
  }

  TORCH_CHECK(
      !invalid_T && !invalid_B,
      "Not enough infos bits to accommodate T and B. Default num bits = ",
      DEFAULT_INFO_NUM_BITS);

  // Recompute info_B_mask using new info_B_num_bits
  info_B_mask = (1u << info_B_num_bits) - 1;

  return {info_B_num_bits, info_B_mask};
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
#define DEF_RADIX_SORT_PAIRS_FN(KeyT, ValueT)                        \
  DLL_PUBLIC cudaError_t radix_sort_pairs(                           \
      void* d_temp_storage,                                          \
      size_t& temp_storage_bytes,                                    \
      const KeyT* d_keys_in,                                         \
      KeyT* d_keys_out,                                              \
      const ValueT* d_values_in,                                     \
      ValueT* d_values_out,                                          \
      const int num_items,                                           \
      const int begin_bit,                                           \
      const int end_bit,                                             \
      cudaStream_t stream) {                                         \
    return FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs( \
        d_temp_storage,                                              \
        temp_storage_bytes,                                          \
        d_keys_in,                                                   \
        d_keys_out,                                                  \
        d_values_in,                                                 \
        d_values_out,                                                \
        num_items,                                                   \
        begin_bit,                                                   \
        end_bit,                                                     \
        stream);                                                     \
  }
#else
#define DEF_RADIX_SORT_PAIRS_FN(KeyT, ValueT)                        \
  DLL_PUBLIC cudaError_t radix_sort_pairs(                           \
      void* d_temp_storage,                                          \
      size_t& temp_storage_bytes,                                    \
      const KeyT* d_keys_in,                                         \
      KeyT* d_keys_out,                                              \
      const ValueT* d_values_in,                                     \
      ValueT* d_values_out,                                          \
      const int num_items,                                           \
      const int begin_bit,                                           \
      const int end_bit,                                             \
      cudaStream_t stream) {                                         \
    return FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceRadixSort::SortPairs( \
        d_temp_storage,                                              \
        temp_storage_bytes,                                          \
        d_keys_in,                                                   \
        d_keys_out,                                                  \
        d_values_in,                                                 \
        d_values_out,                                                \
        num_items,                                                   \
        begin_bit,                                                   \
        end_bit,                                                     \
        stream,                                                      \
        false);                                                      \
  }
#endif

DEF_RADIX_SORT_PAIRS_FN(int64_t, float);
DEF_RADIX_SORT_PAIRS_FN(int64_t, double);
DEF_RADIX_SORT_PAIRS_FN(int64_t, int64_t);
DEF_RADIX_SORT_PAIRS_FN(int64_t, int32_t);

__global__
__launch_bounds__(kMaxThreads) void generate_vbe_metadata_foreach_sample_kernel(
    at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        row_output_offsets,
    at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits> b_t_map,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        B_offsets,
    const at::PackedTensorAccessor32<int32_t, 2, at::RestrictPtrTraits>
        B_offsets_rank_per_feature,
    const at::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits>
        output_offsets_feature_rank,
    const at::PackedTensorAccessor32<int32_t, 1, at::RestrictPtrTraits>
        D_offsets,
    const int32_t D,
    const bool nobag,
    FixedDivisor fd_max_B,
    FixedDivisor fd_max_B_T,
    const int32_t info_B_num_bits) {
  const auto r_b_t = blockIdx.x * blockDim.x + threadIdx.x;
  const auto T = B_offsets.size(0) - 1; // Num tables
  const auto R = B_offsets_rank_per_feature.size(1) - 1; // Num ranks

  int32_t b_t;
  int32_t r; // Rank ID
  int32_t t; // Table ID
  int32_t b; // Relative sample ID in the rank-table matrix

  fd_max_B_T.DivMod(r_b_t, &r, &b_t);
  if (r >= R) {
    return;
  }

  fd_max_B.DivMod(b_t, &t, &b);
  if (t >= T) {
    return;
  }

  const auto B_start_r_t = B_offsets_rank_per_feature[t][r];
  const auto B_r_t = B_offsets_rank_per_feature[t][r + 1] - B_start_r_t;
  if (b >= B_r_t) {
    return;
  }

  const auto B_start_t = B_offsets[t];
  // Update b_t
  b_t = B_start_t + B_start_r_t + b;
  const auto D_ = nobag ? D : D_offsets[t + 1] - D_offsets[t];
  row_output_offsets[b_t] = output_offsets_feature_rank[r * T + t] + b * D_;

  // Relative sample ID in the table
  const auto b_ = B_start_r_t + b;
  // b_t is always positive.
  *reinterpret_cast<uint32_t*>(&b_t_map[b_t]) =
      (reinterpret_cast<uint32_t*>(&t)[0] << info_B_num_bits) |
      reinterpret_cast<const uint32_t*>(&b_)[0];
}

/// Generate VBE metadata namely output_offsets and b_t_map
///
/// row_output_offsets A 1D tensor that contains the output offset of each b
///                    (sample) and t (feature/table) pair. The output
///                    serializes O_r_t where O_r_t is the local output of rank
///                    r and feature/table t (t is the fastest moving index).
/// b_t_map            A 1D tensor that contains the b and t information of the
///                    linearized b and t (b is the fastest moving index).
///
/// @param B_offsets                   Batch size offsets for all features.
/// @param B_offsets_rank_per_feature  Batch size offsets for all ranks (GPUs)
///                                    for each feature.
/// @param output_offsets_feature_rank Output offsets for all features and ranks
///                                    and features.
/// @param D_offsets                   Embedding dimension offsets. Required if
///                                    nobag is false.
/// @param D                           The embedding dimension. Required if
///                                    nobag is true.
/// @param nobag                       A boolean to indicate if TBE is pooled
///                                    (false) or sequence (true).
/// @param info_B_num_bits             The number of bits used to encode a
///                                    sample ID. (Used for populating b_t_map).
/// @param total_B                     The total number of samples (i.e., the
///                                    total number of b and t pairs).
DLL_PUBLIC std::tuple<Tensor /*row_output_offsets*/, Tensor /*b_t_map*/>
generate_vbe_metadata(
    const Tensor& B_offsets,
    const Tensor& B_offsets_rank_per_feature,
    const Tensor& output_offsets_feature_rank,
    const Tensor& D_offsets,
    const int64_t D,
    const bool nobag,
    const int64_t max_B_feature_rank,
    const int64_t info_B_num_bits,
    const int64_t total_B) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(
      B_offsets, B_offsets_rank_per_feature, output_offsets_feature_rank);

  TENSOR_NDIM_EQUALS(B_offsets, 1);
  TENSOR_NDIM_EQUALS(B_offsets_rank_per_feature, 2);
  TENSOR_NDIM_EQUALS(output_offsets_feature_rank, 1);

  const int32_t T = B_offsets.numel() - 1;
  if (!nobag) {
    TENSOR_ON_CUDA_GPU(D_offsets);
    TENSORS_ON_SAME_DEVICE(B_offsets, D_offsets);
    TORCH_CHECK(D_offsets.numel() == T + 1)
  }

  const auto num_ranks = B_offsets_rank_per_feature.size(1) - 1;
  TORCH_CHECK(B_offsets_rank_per_feature.size(0) == T);
  TORCH_CHECK(output_offsets_feature_rank.numel() == num_ranks * T + 1);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(B_offsets.get_device());

  Tensor row_output_offsets =
      at::empty({total_B}, output_offsets_feature_rank.options());
  Tensor b_t_map = at::empty({total_B}, B_offsets.options());

  // Over allocate total number of threads to avoid using binary search
  generate_vbe_metadata_foreach_sample_kernel<<<
      div_round_up(max_B_feature_rank * T * num_ranks, kMaxThreads),
      kMaxThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      row_output_offsets.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      b_t_map.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      B_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      B_offsets_rank_per_feature
          .packed_accessor32<int32_t, 2, at::RestrictPtrTraits>(),
      output_offsets_feature_rank
          .packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
      D_offsets.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
      D,
      nobag,
      FixedDivisor(max_B_feature_rank),
      FixedDivisor(max_B_feature_rank * T),
      info_B_num_bits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {row_output_offsets, b_t_map};
}
