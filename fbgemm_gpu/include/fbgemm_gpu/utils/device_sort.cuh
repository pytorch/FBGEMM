/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cub/cub.cuh>

#ifdef USE_ROCM
#include <thrust/functional.h>
#else
#include <cuda/functional>
#endif

// clang-format off
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh"
// clang-format on

namespace fbgemm_gpu {

#ifdef USE_ROCM
template <typename T>
using Max = thrust::maximum<T>;
#else
#if CUDA_VERSION >= 13000
template <typename T>
using Max = cuda::maximum<T>;
#else
template <typename T>
using Max = cub::Max;
#endif
#endif

#ifdef USE_ROCM
template <typename T>
using Min = thrust::minimum<T>;
#else
#if CUDA_VERSION >= 13000
template <typename T>
using Min = cuda::minimum<T>;
#else
template <typename T>
using Min = cub::Min;
#endif
#endif

} // namespace fbgemm_gpu
