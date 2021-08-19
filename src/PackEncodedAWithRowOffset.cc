/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include <cpuinfo.h>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "./OptimizedKernelsAvx2.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T, typename accT>
PackEncodedAWithRowOffset<T, accT>::PackEncodedAWithRowOffset(
    matrix_op_t trans,
    uint32_t nRow,
    uint32_t nCol,
    const T* smat,
    uint32_t ld,
    inpType* pmat,
    int groups,
    int32_t* row_offset,
    const BlockingFactors* params)
    : PackMatrix<PackEncodedAWithRowOffset<T, accT>, T, accT>(
          nRow + 1,
          nCol,
          pmat,
          groups,
          params),
      trans_(trans),
      smat_(smat),
      ld_(ld),
      row_offset_(row_offset) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if ((!fbgemmHasAvx512VnniSupport() && !fbgemmHasAvx512Support() &&
       !fbgemmHasAvx2Support())) {
    assert(0 && "unknown architecure");
  }

  if (params) {
    BaseType::brow_ = params->MCB;
    BaseType::bcol_ = params->KCB;
    row_interleave_B_ = params->ROW_INTERLEAVE;
  } else {
    if (fbgemmHasAvx512VnniSupport()) {
      BaseType::brow_ = PackingTraits<T, accT, inst_set_t::avx512_vnni>::MCB;
      BaseType::bcol_ = PackingTraits<T, accT, inst_set_t::avx512_vnni>::KCB;
      row_interleave_B_ =
          PackingTraits<T, accT, inst_set_t::avx512_vnni>::ROW_INTERLEAVE;
    } else if (fbgemmHasAvx512Support()) {
      BaseType::brow_ = PackingTraits<T, accT, inst_set_t::avx512>::MCB;
      BaseType::bcol_ = PackingTraits<T, accT, inst_set_t::avx512>::KCB;
      row_interleave_B_ =
          PackingTraits<T, accT, inst_set_t::avx512>::ROW_INTERLEAVE;
    } else {
      // AVX2
      BaseType::brow_ = PackingTraits<T, accT, inst_set_t::avx2>::MCB;
      BaseType::bcol_ = PackingTraits<T, accT, inst_set_t::avx2>::KCB;
      row_interleave_B_ =
          PackingTraits<T, accT, inst_set_t::avx2>::ROW_INTERLEAVE;
    }
  }

  rowOffsetAllocatedHere = false;

  if (BaseType::numCols() % groups != 0) {
    throw std::runtime_error(
        "groups = " + std::to_string(groups) +
        " does not divide numCols = " + std::to_string(BaseType::numCols()));
  }
  if (pmat) {
    BaseType::buf_ = pmat;
  } else {
    BaseType::bufAllocatedHere_ = true;
    BaseType::buf_ = static_cast<T*>(
        fbgemmAlignedAlloc(64, BaseType::brow_ * BaseType::bcol_ * sizeof(T)));
  }
  if (!row_offset_) {
    rowOffsetAllocatedHere = true;
    row_offset_ = static_cast<int32_t*>(
        fbgemmAlignedAlloc(64, BaseType::brow_ * sizeof(int32_t)));
  }
  // allocate for column sum of A
  col_sum_int32_ = new int32_t[BaseType::numCols()]();
  // actual number of rows of A is 1 less than its encoded version
  actual_n_rows_ = BaseType::numRows() - 1;
}

template <typename T, typename accT>
void PackEncodedAWithRowOffset<T, accT>::pack(const block_type_t& block) {
  // assert(block.row_start % BaseType::blockRowSize() == 0);
  assert(block.row_size <= BaseType::blockRowSize());
  assert(block.col_size <= BaseType::blockColSize());

  block_type_t block_p = {block.row_start,
                          block.row_size,
                          block.col_start,
                          (block.col_size + row_interleave_B_ - 1) /
                              row_interleave_B_ * row_interleave_B_};
  assert(block_p.col_size <= BaseType::blockColSize());
  BaseType::packedBlock(block_p);
  T* out = BaseType::getBuf();
  bool tr = (trans_ == matrix_op_t::Transpose);
  // accumulate into row offset?
  bool row_offset_acc =
      (block.col_start % (this->numCols() / this->numGroups())) != 0;
  int32_t* row_offset_buf = getRowOffsetBuffer();
  if (block.row_start + block.row_size <
      BaseType::numRows()) { // not the last row of blocks
    if (tr) {
      for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
        int buf_idx = i - block.row_start;
        int32_t row_sum = row_offset_acc ? row_offset_buf[buf_idx] : 0;
        for (int j = block.col_start; j < block.col_start + block.col_size;
             ++j) {
          T val = smat_[i + j * ld_];
          row_sum += val;
          // accumulate column checksum
          col_sum_int32_[j] += val;
          out[buf_idx * BaseType::blockColSize() + (j - block.col_start)] = val;
        }
        row_offset_buf[buf_idx] = row_sum;
        // zero fill
        // Please see the comment in PackAMatrix.cc on zero vs zero_pt fill.
        for (int j = block.col_size; j < block_p.col_size; ++j) {
          out[buf_idx * BaseType::blockColSize() + j] = 0;
        }
      }
    } else {
      // reduceAvx2 only written for T == uint8_t
      static_assert(
          std::is_same<T, uint8_t>::value,
          "PackEncodedAWithRowOffset<T, accT>::pack only works for T == uint8_t");
      for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
        int buf_idx = i - block.row_start;
        memcpy(
            out + buf_idx * BaseType::blockColSize(),
            smat_ + i * ld_ + block.col_start,
            block.col_size * sizeof(T));
        // zero fill
        for (int j = block.col_size; j < block_p.col_size; ++j) {
          out[buf_idx * BaseType::blockColSize() + j] = 0;
        }
        int32_t row_sum = row_offset_acc ? row_offset_buf[buf_idx] : 0;
        // accumulate row_sum for offset and column sum for encoding A
        row_sum += reduceWithColSumAvx2(
            smat_ + i * ld_ + block.col_start,
            block.col_size,
            col_sum_int32_ + block.col_start);
        row_offset_buf[buf_idx] = row_sum;
      }
    }
  } else { // last row of blocks
    if (tr) {
      // exclude the last row of encoded A, handle it separately
      for (int i = block.row_start; i < block.row_start + block.row_size - 1;
           ++i) {
        int buf_idx = i - block.row_start;
        int32_t row_sum = row_offset_acc ? row_offset_buf[buf_idx] : 0;
        for (int j = block.col_start; j < block.col_start + block.col_size;
             ++j) {
          T val = smat_[i + j * ld_];
          row_sum += val;
          // accumulate column sum
          col_sum_int32_[j] += val;
          out[buf_idx * BaseType::blockColSize() + (j - block.col_start)] = val;
        }
        row_offset_buf[buf_idx] = row_sum;
        // zero fill
        // Please see the comment in PackAMatrix.cc on zero vs zero_pt fill.
        for (int j = block.col_size; j < block_p.col_size; ++j) {
          out[buf_idx * BaseType::blockColSize() + j] = 0;
        }
      }

      // separate handling of the last row of encoded A
      // pack the last row which is the encoded checksum row
      int i = block.row_start + block.row_size - 1;
      int buf_idx = i - block.row_start;
      for (int j = 0; j < block.col_size; j++) {
        out[buf_idx * BaseType::blockColSize() + j] =
            static_cast<T>(col_sum_int32_[block.col_start + j] % MOD_Uint8);
      }
      for (int j = block.col_size; j < block_p.col_size; j++) {
        out[buf_idx * BaseType::blockColSize() + j] = 0;
      }

    } else {
      // reduceAvx2 only written for T == uint8_t
      static_assert(
          std::is_same<T, uint8_t>::value,
          "PackEncodedAWithRowOffset<T, accT>::pack only works for T == uint8_t");

      // exclude the last row of encoded A, handle it separately
      for (int i = block.row_start; i < block.row_start + block.row_size - 1;
           ++i) {
        int buf_idx = i - block.row_start;
        memcpy(
            out + buf_idx * BaseType::blockColSize(),
            smat_ + i * ld_ + block.col_start,
            block.col_size * sizeof(T));
        // zero fill
        for (int j = block.col_size; j < block_p.col_size; ++j) {
          out[buf_idx * BaseType::blockColSize() + j] = 0;
        }
        int32_t row_sum = row_offset_acc ? row_offset_buf[buf_idx] : 0;
        row_sum += reduceWithColSumAvx2(
            smat_ + i * ld_ + block.col_start,
            block.col_size,
            col_sum_int32_ + block.col_start);
        row_offset_buf[buf_idx] = row_sum;
      }

      // pack the last row which is the checksum row
      int i = block.row_start + block.row_size - 1;
      int buf_idx = i - block.row_start;
      for (int j = 0; j < block.col_size; j++) {
        out[buf_idx * BaseType::blockColSize() + j] =
            static_cast<T>(col_sum_int32_[block.col_start + j] % MOD_Uint8);
      }
      for (int j = block.col_size; j < block_p.col_size; j++) {
        out[buf_idx * BaseType::blockColSize() + j] = 0;
      }
    }
  } // end separate handling of last row of blocks
}

template <typename T, typename accT>
int32_t PackEncodedAWithRowOffset<T, accT>::addr(int32_t r, int32_t c) const {
  int32_t block_row_id = r / BaseType::blockRowSize();
  int32_t brow_offset = (block_row_id * BaseType::blockCols()) *
      (BaseType::blockRowSize() * BaseType::blockColSize());

  int32_t block_col_id = c / BaseType::blockColSize();
  int32_t bcol_offset =
      block_col_id * BaseType::blockRowSize() * BaseType::blockColSize();
  int32_t block_offset = brow_offset + bcol_offset;
  int32_t inblock_offset =
      (r % BaseType::blockRowSize()) * BaseType::blockColSize() +
      (c % BaseType::blockColSize());

  int32_t index = block_offset + inblock_offset;

  return index;
}

template <typename T, typename accT>
void PackEncodedAWithRowOffset<T, accT>::printPackedMatrix(std::string name) {
  std::cout << name << ":"
            << "[" << BaseType::numPackedRows() << ", "
            << BaseType::numPackedCols() << "]" << std::endl;

  T* out = BaseType::getBuf();
  for (auto r = 0; r < BaseType::numPackedRows(); ++r) {
    for (auto c = 0; c < BaseType::numPackedCols(); ++c) {
      T val = out[addr(r, c)];
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

template <typename T, typename accT>
void PackEncodedAWithRowOffset<T, accT>::encodeA(T* ColCksm, int32_t mod) {
  std::vector<int32_t> tmpSum(BaseType::numCols());
  for (auto i = 0; i < BaseType::numRows(); i++) {
    for (auto j = 0; j < BaseType::numCols(); j++) {
      tmpSum[j] += smat_[i * BaseType::numCols() + j];
    }
  }
  for (auto j = 0; j < BaseType::numCols(); j++) {
    ColCksm[j] = static_cast<T>(tmpSum[j] % mod);
  }
}

template <typename T, typename accT>
int PackEncodedAWithRowOffset<T, accT>::rowOffsetBufferSize(
    const BlockingFactors* params) {
  if (cpuinfo_initialize()) {
    if (params) {
      return params->MCB;
    } else {
      if (fbgemmHasAvx512VnniSupport()) {
        return PackingTraits<T, accT, inst_set_t::avx512_vnni>::MCB;
      } else if (fbgemmHasAvx512Support()) {
        return PackingTraits<T, accT, inst_set_t::avx512>::MCB;
      } else if (fbgemmHasAvx2Support()) {
        return PackingTraits<T, accT, inst_set_t::avx2>::MCB;
      } else {
        // TODO: Have default slower path
        assert(0 && "unsupported architecture");
        return -1;
      }
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

template class PackEncodedAWithRowOffset<uint8_t, int32_t>;
template class PackEncodedAWithRowOffset<uint8_t, int16_t>;

} // namespace fbgemm
