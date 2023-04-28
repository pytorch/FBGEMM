/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
      "FloatToFP8RowwiseQuantized", fbgemm_gpu::_float_to_FP8rowwise_gpu);
  DISPATCH_TO_CUDA(
      "FloatToPaddedFP8RowwiseQuantized",
      fbgemm_gpu::_float_to_paddedFP8rowwise_gpu);
  DISPATCH_TO_CUDA(
      "HalfToFused8BitRowwiseQuantized",
      fbgemm_gpu::_half_to_fused8bitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "FloatOrHalfToFused8BitRowwiseQuantized",
      fbgemm_gpu::_float_or_half_to_fused8bitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "Fused8BitRowwiseQuantizedToFloat",
      fbgemm_gpu::_fused8bitrowwise_to_float_gpu);
  DISPATCH_TO_CUDA(
      "FP8RowwiseQuantizedToFloat", fbgemm_gpu::_FP8rowwise_to_float_gpu);
  DISPATCH_TO_CUDA(
      "Fused8BitRowwiseQuantizedToHalf",
      fbgemm_gpu::_fused8bitrowwise_to_half_gpu);
  DISPATCH_TO_CUDA(
      "Fused8BitRowwiseQuantizedToFloatOrHalf",
      fbgemm_gpu::_fused8bitrowwise_to_float_or_half_gpu);
  DISPATCH_TO_CUDA(
      "Fused8BitRowwiseQuantizedToFloatMixedDim",
      fbgemm_gpu::_fused8bitrowwise_to_float_mixed_dim_gpu);
  DISPATCH_TO_CUDA(
      "FloatToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::_float_to_fusednbitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "HalfToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::_half_to_fusednbitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf",
      fbgemm_gpu::_float_or_half_to_fusednbitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "FusedNBitRowwiseQuantizedSBHalfToFloat",
      fbgemm_gpu::_fusednbitrowwise_to_float_gpu);
  DISPATCH_TO_CUDA(
      "FusedNBitRowwiseQuantizedSBHalfToHalf",
      fbgemm_gpu::_fusednbitrowwise_to_half_gpu);
  DISPATCH_TO_CUDA(
      "FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf",
      fbgemm_gpu::_fusednbitrowwise_to_float_or_half_gpu);
  DISPATCH_TO_CUDA("FloatToHFP8Quantized", fbgemm_gpu::_float_to_hfp8_gpu);
  DISPATCH_TO_CUDA("HFP8QuantizedToFloat", fbgemm_gpu::_hfp8_to_float_gpu);
  DISPATCH_TO_CUDA("FloatToMSFPQuantized", fbgemm_gpu::_float_to_msfp_gpu);
  DISPATCH_TO_CUDA("MSFPQuantizedToFloat", fbgemm_gpu::_msfp_to_float_gpu);
  DISPATCH_TO_CUDA(
      "PaddedFP8RowwiseQuantizedToFloat",
      fbgemm_gpu::_paddedFP8rowwise_to_float_gpu);
}
