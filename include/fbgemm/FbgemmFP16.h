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
#include <stdexcept>
#include <vector>

#include "Types.h"
#include "Utils.h"

namespace fbgemm {

/// class that performs packing of matrix in
/// row-major format into
/// internal packed blocked-row major format
class PackedGemmMatrixFP16 {
 public:
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
  PackedGemmMatrixFP16(
      const matrix_op_t trans,
      const int nrow,
      const int ncol,
      const float alpha,
      const float* smat,
      const int brow = 512,
      const int kernel_ncol_blocks = 2)
      : nrow_(nrow),
        ncol_(ncol),
        brow_(brow),
        kernel_ncol_blocks_(kernel_ncol_blocks) {
    initializeParam();
    initializeMemory();
    // copy source matrix into packed matrix
    this->packFromSrc(trans, alpha, smat);
  }

  PackedGemmMatrixFP16(
      const int nrow,
      const int ncol,
      const int brow,
      const int last_brow,
      const int bcol,
      const int nbrow,
      const int nbcol,
      const uint64_t size,
      const int kernel_ncol_blocks = 2)
      : nrow_(nrow),
        ncol_(ncol),
        brow_(brow),
        last_brow_(last_brow),
        bcol_(bcol),
        nbrow_(nbrow),
        nbcol_(nbcol),
        size_(size),
        kernel_ncol_blocks_(kernel_ncol_blocks) {
    initializeMemory();
  }

  void initializeParam() {
    bcol_ = 8 * kernelNumColBlocks();

    // set up internal packing parameters
    nbrow_ = ((numRows() % blockRowSize()) == 0)
        ? (numRows() / blockRowSize())
        : ((numRows() + blockRowSize()) / blockRowSize());
    last_brow_ = ((nrow_ % blockRowSize()) == 0) ? blockRowSize()
                                                 : (nrow_ % blockRowSize());
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
  }

  void setPacked(bool p) {
    packed_ = p;
  }

  bool packed() const {
    return packed_;
  }

  void initializeMemory() {
    // allocate and initialize packed memory
    const int padding = 1024; // required by sw pipelined kernels
    size_ = (blockRowSize() * nbrow_) * (blockColSize() * nbcol_);
#ifdef _MSC_VER
    pmat_ = (float16 *)_aligned_malloc(matSize() * sizeof(float16) +
      padding, 64);
#else
    int result = posix_memalign((void**)&pmat_, 64, matSize() * sizeof(float16) + padding);
    assert(result == 0);
#endif
    for (auto i = 0; i < matSize(); i++) {
      pmat_[i] = tconv(0.f, pmat_[i]);
    }
  }

  ~PackedGemmMatrixFP16() {
#ifdef _MSC_VER
    _aligned_free(pmat_);
#else
    free(pmat_);
#endif
  }

  void unpackFromSrc(const matrix_op_t trans, float16* src_mat) {
    bool tr = (trans == matrix_op_t::Transpose);
    for (int i = 0; i < numRows(); i++) {
      for (int j = 0; j < numCols(); j++) {
        pmat_[tr ? i + numRows() * j : i * numCols() + j] = src_mat[addr(i, j)];
      }
    }
    packed_ = false;
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
    packed_ = true;
  }

  // This function takes in an unpacked float16 matrix of the same size and
  // packs it. There is no floating type conversion.
  void packFromSrc(const matrix_op_t trans, const float16* smat) {
    bool tr = (trans == matrix_op_t::Transpose);
    for (int i = 0; i < numRows(); ++i) {
      for (int j = 0; j < numCols(); ++j) {
        pmat_[addr(i, j)] = smat[tr ? i + numRows() * j : i * numCols() + j];
      }
    }
    packed_ = true;
  }

  const float16& operator()(const int r, const int c) const {
    uint64_t a = addr(r, c);
    assert(r < numRows());
    assert(c < numCols());
    assert(a < this->matSize());
    return pmat_[a];
  }

  int matSize() const {
    return (int)size_;
  }
  int numRows() const {
    return nrow_;
  }
  int numCols() const {
    return ncol_;
  }
  int lastBrow() const {
    return last_brow_;
  }
  int numBrow() const {
    return nbrow_;
  }
  int numBcol() const {
    return nbcol_;
  }
  float16* pmat() const {
    return pmat_;
  }
  inline int blockRowSize() const {
    return brow_;
  }
  inline int blockColSize() const {
    return bcol_;
  }
  inline int kernelNumColBlocks() const {
    return kernel_ncol_blocks_;
  }

  int nrow_, ncol_;
  int brow_, last_brow_, bcol_;
  int nbrow_, nbcol_;
  uint64_t size_;
  int kernel_ncol_blocks_;
  float16* pmat_;
  bool packed_{false};

  friend void cblas_gemm_compute(
      const matrix_op_t transa,
      const int m,
      const float* A,
      const PackedGemmMatrixFP16& Bp,
      const float beta,
      float* C,
      int thread_id,
      int num_threads);
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
    float* C,
    int thread_id = 0,
    int num_threads = 1);
}; // namespace fbgemm
