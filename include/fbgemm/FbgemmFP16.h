/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

// WARNING: this is a legacy fp16 fbgemm implementation and will soon be
// upgraded to match with new fbgemm interface.

#include <cassert>
#include <cstdlib>
#include <memory>
#include <vector>

#include "Types.h"
#include "Utils.h"

namespace fbgemm2 {

/// class that performs packing of matrix in
/// row-major format into
/// internal packed blocked-row major format
class PackedGemmMatrixFP16 {
 public:
  // takes smat input mamtrix in row-major format;
  // and packs it into gemm-friendly blocked format;
  // allocate space and sets up all the internal variables;
  // also premultiplies by alpha during packing
  // brow_ contains tile size along k dimension
  // and also is # of fmas updates into int16 container
  // before flushing into fp32
  // the smaller the brow_, the higher overhead
  // of flushing is
  PackedGemmMatrixFP16(
      const matrix_op_t trans,
      const int nrow,
      const int ncol,
      const float alpha,
      const float* smat,
      const int brow = 512)
      : nrow_(nrow), ncol_(ncol), brow_(brow) {
    bcol_ = 8 * 1; // hardwired

    // set up internal packing parameters
    nbrow_ = ((numRows() % blockRowSize()) == 0)
        ? (numRows() / blockRowSize())
        : ((numRows() + blockRowSize()) / blockRowSize());
    last_brow_ = ((nrow % blockRowSize()) == 0) ? blockRowSize()
                                                : (nrow % blockRowSize());
    nbcol_ = ((numCols() % blockColSize()) == 0)
        ? (numCols() / blockColSize())
        : ((numCols() + blockColSize()) / blockColSize());

    if (numCols() != blockColSize() * nbcol_) {
#ifdef VLOG
      VLOG(0) << "Packer warning: ncol(" << numCols()
              << ") is not a multiple of internal block size ("
              << blockColSize() << ")";
      VLOG(0)
          << "lefover is currently done via MKL: hence overhead will inccur";
#endif
    }

    // allocate and initialize packed memory
    const int padding = 1024; // required by sw pipelined kernels
    size_ = (blockRowSize() * nbrow_) * (blockColSize() * nbcol_);
    // pmat_ = (float16 *)aligned_alloc(64, matSize() * sizeof(float16) +
    // padding);
    posix_memalign((void**)&pmat_, 64, matSize() * sizeof(float16) + padding);
    for (auto i = 0; i < matSize(); i++) {
      pmat_[i] = tconv(0.f, pmat_[i]);
    }

    // copy source matrix into packed matrix
    this->packFromSrc(trans, alpha, smat);
  }

  ~PackedGemmMatrixFP16() {
    free(pmat_);
  }

  // protected:
  // blocked row-major format address arithmetic
  uint64_t addr(const int r_, const int c_) const {
    uint64_t r = (uint64_t)r_;
    uint64_t c = (uint64_t)c_;

    uint64_t block_row_id = r / blockRowSize(),
             brow_offset =
                 (block_row_id * nbcol_) * (blockRowSize() * blockColSize());
    uint64_t block_col_id = c / blockColSize(),
             bcol_offset = block_col_id *
        ((block_row_id != nbrow_ - 1) ? (blockRowSize() * blockColSize())
                                      : (last_brow_ * blockColSize()));
    uint64_t block_offset = brow_offset + bcol_offset;
    uint64_t inblock_offset =
        r % blockRowSize() * blockColSize() + c % blockColSize();

    uint64_t index = block_offset + inblock_offset;
    assert(index < matSize());
    return index;
  }

  void
  packFromSrc(const matrix_op_t trans, const float alpha, const float* smat) {
    bool tr = (trans == matrix_op_t::Transpose);
    // pack
    for (int i = 0; i < numRows(); i++) {
      for (int j = 0; j < numCols(); j++) {
        pmat_[addr(i, j)] = tconv(
            alpha *
                ((tr == false) ? smat[i * numCols() + j]
                               : smat[i + numRows() * j]),
            pmat_[addr(i, j)]);
      }
    }
  }

  const float16& operator()(const int r, const int c) const {
    uint64_t a = addr(r, c);
    assert(r < numRows());
    assert(c < numCols());
    assert(a < this->matSize());
    return pmat_[a];
  }

  int matSize() const {
    return size_;
  }
  int numRows() const {
    return nrow_;
  }
  int numCols() const {
    return ncol_;
  }
  inline int blockRowSize() const {
    return brow_;
  }
  inline int blockColSize() const {
    return bcol_;
  }

  int nrow_, ncol_;
  int brow_, last_brow_, bcol_;
  int nbrow_, nbcol_;
  uint64_t size_;
  float16* pmat_;

  friend void cblas_gemm_compute(
      const matrix_op_t transa,
      const int m,
      const float* A,
      const PackedGemmMatrixFP16& Bp,
      const float beta,
      float* C);
  friend void cblas_gemm_compute(
      const matrix_op_t transa,
      const int m,
      const float* A,
      const PackedGemmMatrixFP16& Bp,
      const float beta,
      float* C);
};

/**
 * restrictions: transa == CblasNoTrans
 */
extern void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C);
extern void cblas_gemm_compute(
    const matrix_op_t transa,
    const int m,
    const float* A,
    const PackedGemmMatrixFP16& Bp,
    const float beta,
    float* C);

}; // namespace fbgemm2
