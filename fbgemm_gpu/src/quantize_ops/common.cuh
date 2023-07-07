/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/TensorIterator.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#ifndef __HIP_PLATFORM_HCC__
#include <math_constants.h>
#endif

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/ops_utils.h"
#include "fbgemm_gpu/quantize_ops.cuh"
#include "fbgemm_gpu/quantize_ops_utils.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

#define QUANTIZE_OPS_MAX(a, b) ((a) > (b) ? (a) : (b))
#define QUANTIZE_OPS_MIN(a, b) ((a) < (b) ? (a) : (b))

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {

template <typename T>
__device__ inline __attribute__((always_inline)) T
quantize_ops_shfl_xor(const T val, int laneMask, int width) {
#if defined(__HIP_PLATFORM_HCC__) || CUDA_VERSION < 9000
  return __shfl_xor(val, laneMask, width);
#else
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#endif
}

} // namespace

} // namespace fbgemm_gpu
