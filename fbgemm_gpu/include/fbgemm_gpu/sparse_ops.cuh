/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __HIP_PLATFORM_HCC__
#define HIPCUB_ARCH 1
#endif

#include <ATen/ATen.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <cuda.h>

// clang-format off
#include "./cub_namespace_prefix.cuh"
#include <cub/block/block_reduce.cuh>
#include "./cub_namespace_postfix.cuh"
// clang-format on
