/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/sparse_ops.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

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
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/embedding_backward_template_helpers.cuh"
#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/split_embeddings_utils.cuh"

/*
 * We annotate the public fbgemm functions and hide the rest. Those
 * public symbols can be called via fbgemm_gpu::func() or pytorch
 * operator dispatcher. We'll hide other symbols, especially cub APIs,
 * because different .so may include the same cub CUDA kernels, which
 * results in confusion and libA may end up calling libB's cub kernel,
 * causing failures when we static link libcudart_static.a
 */
#define DLL_PUBLIC __attribute__((visibility("default")))

#ifdef __HIP_PLATFORM_HCC__
#include <hipblas.h>
#endif

#ifdef __HIP_PLATFORM_HCC__
#define LDG(ptr) (*(ptr))
#else
#define LDG(ptr) (__ldg(ptr))
#endif

using Tensor = at::Tensor;
