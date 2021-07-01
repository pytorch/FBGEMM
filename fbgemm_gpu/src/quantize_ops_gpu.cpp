/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright 2004-present Facebook. All Rights Reserved.
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using namespace at;
TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA(
      "FloatToFused8BitRowwiseQuantized", _float_to_fused8bitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "Fused8BitRowwiseQuantizedToFloat", _fused8bitrowwise_to_float_gpu);
  DISPATCH_TO_CUDA(
      "FloatToFusedNBitRowwiseQuantizedSBHalf", _float_to_fusednbitrowwise_gpu);
  DISPATCH_TO_CUDA(
      "FusedNBitRowwiseQuantizedSBHalfToFloat", _fusednbitrowwise_to_float_gpu);
}
