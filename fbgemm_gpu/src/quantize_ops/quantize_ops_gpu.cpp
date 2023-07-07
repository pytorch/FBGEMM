/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm_gpu/ops_utils.h"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

/*
  NOTE: Some operators are dispatched here in a .cpp file because the PyTorch
  macros for registering operators fail compilation when they are declared in a
  .cu file AND the operator signature declaration contains default arguments.
*/

FBGEMM_OP_DISPATCH(
    CUDA,
    "FloatToFP8RowwiseQuantized",
    fbgemm_gpu::_float_to_FP8rowwise_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "FP8RowwiseQuantizedToFloat",
    fbgemm_gpu::_FP8rowwise_to_float_gpu);

FBGEMM_OP_DISPATCH(
    CUDA,
    "FloatToPaddedFP8RowwiseQuantized",
    fbgemm_gpu::_float_to_paddedFP8rowwise_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "PaddedFP8RowwiseQuantizedToFloat",
    fbgemm_gpu::_paddedFP8rowwise_to_float_gpu);
