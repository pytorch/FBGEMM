/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright 2004-present Facebook. All Rights Reserved.
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA(
      "FloatToFused8BitRowwiseQuantized",
      fbgemm_gpu::_float_to_fused8bitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "HalfToFused8BitRowwiseQuantized",
      fbgemm_gpu::_half_to_fused8bitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "Fused8BitRowwiseQuantizedToFloat",
      fbgemm_gpu::_fused8bitrowwise_to_float_gpu);
  DISPATCH_TO_CUDA(
      "Fused8BitRowwiseQuantizedToFloatMixedDim",
      fbgemm_gpu::_fused8bitrowwise_to_float_mixed_dim_gpu);
  DISPATCH_TO_CUDA(
      "Fused8BitRowwiseQuantizedToHalf",
      fbgemm_gpu::_fused8bitrowwise_to_half_gpu);
  DISPATCH_TO_CUDA(
      "FloatToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::_float_to_fusednbitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "FusedNBitRowwiseQuantizedSBHalfToFloat",
      fbgemm_gpu::_fusednbitrowwise_to_float_gpu);
  DISPATCH_TO_CUDA(
      "FusedNBitRowwiseQuantizedSBHalfToHalf",
      fbgemm_gpu::_fusednbitrowwise_to_half_gpu);
  DISPATCH_TO_CUDA(
      "HalfToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::_half_to_fusednbitrowwise_gpu);
}
