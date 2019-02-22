/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cpuinfo.h>
#include <cassert>
#include <iomanip>
#include <iostream>
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T, typename accT>
PackBMatrix<T, accT>::PackBMatrix(
    matrix_op_t trans,
    int32_t nRow,
    int32_t nCol,
    const T* smat,
    int32_t ld,
    inpType* pmat,
    int groups)
    : PackMatrix<PackBMatrix<T, accT>, T, accT>(nRow, nCol, pmat, groups),
      trans_(trans),
      smat_(smat),
      ld_(ld) {
  if (cpuinfo_has_x86_avx512f()) {
    BaseType::brow_ = PackingTraits<T, accT, inst_set_t::avx512>::KCB;
    BaseType::bcol_ = PackingTraits<T, accT, inst_set_t::avx512>::NCB;
    row_interleave_ =
        PackingTraits<T, accT, inst_set_t::avx512>::ROW_INTERLEAVE;
  } else if (cpuinfo_has_x86_avx2()) {
    BaseType::brow_ = PackingTraits<T, accT, inst_set_t::avx2>::KCB;
    BaseType::bcol_ = PackingTraits<T, accT, inst_set_t::avx2>::NCB;
    row_interleave_ = PackingTraits<T, accT, inst_set_t::avx2>::ROW_INTERLEAVE;
  } else {
    // Error
    assert(0 && "unknown architecure");
  }
  if (BaseType::numRows() % groups != 0) {
    throw std::runtime_error(
        "groups = " + std::to_string(groups) +
        " does not divide numRows = " + std::to_string(BaseType::numRows()));
  }

  // blocking for one group
  block_type_t block{
      0, BaseType::numRows() / BaseType::numGroups(), 0, BaseType::numCols()};
  BaseType::packedBlock(block);
  if (!pmat) {
    BaseType::bufAllocatedHere_ = true;
    BaseType::buf_ = (T*)fbgemmAlignedAlloc(
        64,
        BaseType::numGroups() * BaseType::blockRows() * BaseType::brow_ *
            BaseType::blockCols() * BaseType::bcol_ * sizeof(T));
  }
  pack(block);
}

template <typename T, typename accT>
void PackBMatrix<T, accT>::pack(const block_type_t& block) {
  assert((BaseType::blockRowSize() % row_interleave_) == 0);
  assert((block.row_start % BaseType::blockRowSize()) == 0);
  assert((block.col_start % BaseType::blockColSize()) == 0);

  BaseType::packedBlock(block);
  bool tr = (trans_ == matrix_op_t::Transpose);
  for (int g = 0; g < this->numGroups(); ++g) {
    T* out = BaseType::getBuf() +
        g * this->packedBufferSize(block.row_size, block.col_size);
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      int r_offset = ((i / BaseType::blockRowSize()) * BaseType::blockCols()) *
              (BaseType::blockRowSize() * BaseType::blockColSize()) +
          (i % BaseType::blockRowSize() / row_interleave_) *
              BaseType::blockColSize() * row_interleave_ +
          i % row_interleave_;

      int c_start_offset = (block.col_start / BaseType::blockColSize()) *
              BaseType::blockRowSize() * BaseType::blockColSize() +
          (block.col_start % BaseType::blockColSize()) * row_interleave_;

      int c_idx_offset = 0;
      int c_blk_offset = 0;
      for (int j = block.col_start; j < block.col_start + block.col_size; ++j) {
        // int c_offset = (j / BaseType::blockColSize()) *
        //         BaseType::blockRowSize() * BaseType::blockColSize() +
        //     (j % BaseType::blockColSize()) * row_interleave_;
        // 1. Loop invariant hoisting (move block offset calculation out of
        // inner loop); 2. Strength reduction (change modulus in inner loop to
        // an increment + rollover).
        int c_offset = c_start_offset +
            c_blk_offset * BaseType::blockRowSize() * BaseType::blockColSize() +
            c_idx_offset * row_interleave_;

        int out_idx = r_offset + c_offset;
        T val = tr ? smat_[i + (g * block.col_size + j) * ld_]
                   : smat_[(g * block.row_size + i) * ld_ + j];
        out[out_idx] = val;

        c_idx_offset++;
        if (c_idx_offset == BaseType::blockColSize()) {
          c_idx_offset = 0;
          c_blk_offset++;
        }
      }
    }
    // fill the remaining with zero.
    // Please see the comment in PackAMatrix.cc on zero vs zero_pt fill.
    for (int i = block.row_start + block.row_size;
         i < (block.row_start + block.row_size + row_interleave_ - 1) /
             row_interleave_ * row_interleave_;
         ++i) {
      int r_offset = ((i / BaseType::blockRowSize()) * BaseType::blockCols()) *
              (BaseType::blockRowSize() * BaseType::blockColSize()) +
          (i % BaseType::blockRowSize() / row_interleave_) *
              BaseType::blockColSize() * row_interleave_ +
          i % row_interleave_;
      for (int j = block.col_start; j < block.col_start + block.col_size; j++) {
        int c_offset = (j / BaseType::blockColSize()) *
                BaseType::blockRowSize() * BaseType::blockColSize() +
            (j % BaseType::blockColSize()) * row_interleave_;

        int out_idx = r_offset + c_offset;
        out[out_idx] = 0;
      }
    }
  } // for each group
}

template <typename T, typename accT>
int32_t PackBMatrix<T, accT>::addr(int32_t r, int32_t c) const {
  int32_t block_row_id = r / BaseType::blockRowSize();
  int32_t brow_offset = (block_row_id * BaseType::blockCols()) *
      (BaseType::blockRowSize() * BaseType::blockColSize());

  int32_t block_col_id = c / BaseType::blockColSize();
  int32_t bcol_offset =
      block_col_id * BaseType::blockRowSize() * BaseType::blockColSize();
  int32_t block_offset = brow_offset + bcol_offset;
  int32_t inblock_offset = (r % BaseType::blockRowSize() / row_interleave_) *
          BaseType::blockColSize() * row_interleave_ +
      (c % BaseType::blockColSize()) * row_interleave_ + r % row_interleave_;

  int32_t index = block_offset + inblock_offset;

  return index;
}

template <typename T, typename accT>
void PackBMatrix<T, accT>::printPackedMatrix(std::string name) {
  std::cout << name << ":"
            << "[" << BaseType::numPackedRows() << ", "
            << BaseType::numPackedCols() << "]" << std::endl;
  std::cout << "block size:"
            << "[" << BaseType::blockRowSize() << ", "
            << BaseType::blockColSize() << "]" << std::endl;

  T* out = BaseType::getBuf();

  for (auto nr = 0; nr < BaseType::blockRows(); ++nr) {
    auto rows = (nr == BaseType::blockRows() - 1) ? BaseType::lastBrow()
                                                  : BaseType::blockRowSize();
    for (auto nc = 0; nc < BaseType::blockCols(); ++nc) {
      std::cout << "block:" << nr << ", " << nc << std::endl;
      auto cols = (nc == BaseType::blockCols() - 1) ? BaseType::lastBcol()
                                                    : BaseType::blockColSize();
      for (auto r = 0; r < (rows + row_interleave_ - 1) / row_interleave_;
           ++r) {
        for (auto c = 0; c < cols * row_interleave_; ++c) {
          T val =
              out[nr * BaseType::blockCols() * BaseType::blockRowSize() *
                      BaseType::blockColSize() +
                  nc * BaseType::blockRowSize() * BaseType::blockColSize() +
                  r * BaseType::blockColSize() * row_interleave_ + c];
          if (std::is_integral<T>::value) {
            // cast to int64 because cout doesn't print int8_t type directly
            std::cout << std::setw(5) << static_cast<int64_t>(val) << " ";
          } else {
            std::cout << std::setw(5) << val << " ";
          }
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
}

template <typename T, typename accT>
bool PackBMatrix<T, accT>::metaEquals(const PackBMatrix<T, accT>& that) const {
  if (BaseType::numRows() != that.numRows() ||
      BaseType::numCols() != that.numCols() ||
      BaseType::blockRowSize() != that.blockRowSize() ||
      BaseType::blockColSize() != that.blockColSize() ||
      BaseType::blockRows() != that.blockRows() ||
      BaseType::blockCols() != that.blockCols() ||
      BaseType::numPackedRows() != that.numPackedRows() ||
      BaseType::numPackedCols() != that.numPackedCols() ||
      trans_ != that.trans_ || BaseType::numGroups() != that.numGroups() ||
      row_interleave_ != that.row_interleave_) {
    return false;
  }

  return true;
}

template <typename T, typename accT>
bool PackBMatrix<T, accT>::equals(const PackBMatrix<T, accT>& that) const {
  if (!metaEquals(that)) {
    return false;
  }

  for (int i = 0; i < this->numRows(); ++i) {
    for (int j = 0; j < this->numCols(); ++j) {
      if (this->buf_[addr(i, j)] != that.buf_[that.addr(i, j)]) {
        return false;
      }
    }
  }

  return true;
}

template class PackBMatrix<int8_t, int32_t>;
template class PackBMatrix<int8_t, int16_t>;
} // namespace fbgemm
