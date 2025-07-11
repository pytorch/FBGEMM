/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/sparse_ops.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/utils/cuda_block_count.h"
#include "fbgemm_gpu/utils/cuda_utilities.cuh"
#include "fbgemm_gpu/utils/ops_utils.h"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDADeviceAssertion.h>
#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// clang-format off
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh"
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"
#include "fbgemm_gpu/utils/binary_search_range.cuh"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/kernel_launcher.cuh"
#include "fbgemm_gpu/utils/log2.h"
#include "fbgemm_gpu/utils/tensor_accessor_builder.h"

#ifdef USE_ROCM
#include <hipblas/hipblas.h>
#endif

#ifdef USE_ROCM
#define LDG(ptr) (*(ptr))
#else
#define LDG(ptr) (__ldg(ptr))
#endif

using Tensor = at::Tensor;

namespace fbgemm_gpu {

constexpr int MAX_ELEMENTS_PER_THREAD = 4;

template <
    typename scalar_t,
    int ndim,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
at::PackedTensorAccessor32<scalar_t, ndim, PtrTraits>
dummy_packed_accessor32() {
  std::array<int64_t, ndim> zeros{};
  return {nullptr, zeros.data(), zeros.data()};
}

template <
    typename scalar_t,
    int ndim,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
pta::PackedTensorAccessor64<scalar_t, ndim, PtrTraits>
dummy_packed_accessor64() {
  std::array<int64_t, ndim> zeros{};
  return {nullptr, zeros.data(), zeros.data()};
}

} // namespace fbgemm_gpu
