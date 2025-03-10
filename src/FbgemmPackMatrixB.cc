/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fbgemm/FbgemmFP16.h"
#define FBGEMM_EXPORTS

namespace fbgemm {

// takes smat input mamtrix in row-major format;
// packs it into gemm-friendly blocked format;
// allocate space and sets up all the internal variables;
// also premultiplies by alpha during packing.
// brow_ contains tile size along k dimension
// and also is # of fmas updates into int16 container
// before flushing into fp32.
// the smaller the brow_, the higher overhead
// of flushing is.
// kernel_ncol_blocks is the number of column blocks (in the size of 8 fp16,
// or 128 bit, or 1 xmm register size) in the kernel. Because the batch size
// can be dynamic and we need to prepack the weight matrix B, the internal
// packing layout of the weight matrix and kernel_ncol_blocks have to be
// fixed. We can choose kernel_ncol_blocks = 1 (with kernels of 1x1~14x1
// register layouts), 2 (with kernels of 1x2~6x2 register layout), or 3 (with
// kernels of 1x3~4x3 register layout).

#ifndef _M_X64

template <>
FBGEMM_API
PackedGemmMatrixB<float16, TypeConverter<float16>>::PackedGemmMatrixB(
    const matrix_op_t trans,
    const int nrow,
    const int ncol,
    const float alpha,
    const float* smat,
    const int brow)
    : nrow_(nrow), ncol_(ncol), brow_(brow), kernel_ncol_blocks_(2) {
#if defined(FBGEMM_ENABLE_KLEIDIAI)
  kernel_ncol_blocks_ = 1;
#endif
  initializeParam();
  initializeMemory();
  // copy source matrix into packed matrix
  this->PackedGemmMatrixB<float16, TypeConverter<float16>>::packFromSrc(
      trans, alpha, smat);
}

template <>
FBGEMM_API
PackedGemmMatrixB<float16, TypeConverter<float16>>::PackedGemmMatrixB(
    const int nrow,
    const int ncol,
    const int brow,
    const int last_brow,
    const int bcol,
    const int nbrow,
    const int nbcol,
    const uint64_t size)
    : nrow_(nrow),
      ncol_(ncol),
      brow_(brow),
      last_brow_(last_brow),
      bcol_(bcol),
      nbrow_(nbrow),
      nbcol_(nbcol),
      size_(size),
      kernel_ncol_blocks_(2) {
#if defined(FBGEMM_ENABLE_KLEIDIAI)
  kernel_ncol_blocks_ = 1;
#endif
  initializeMemory();
}

#endif

} // namespace fbgemm
