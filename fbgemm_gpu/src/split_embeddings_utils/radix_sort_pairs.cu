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

using Tensor = at::Tensor;
using namespace fbgemm_gpu;

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
