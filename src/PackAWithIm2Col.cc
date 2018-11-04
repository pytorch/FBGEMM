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

#include <algorithm>

namespace fbgemm2 {

template <typename T, typename accT>
PackAWithIm2Col<T, accT>::PackAWithIm2Col(
    const conv_param_t& conv_p,
    const T* sdata,
    inpType* pmat,
    int32_t zero_pt,
    int32_t* row_offset)
    : PackMatrix<PackAWithIm2Col<T, accT>, T, accT>(
          conv_p.MB * conv_p.OH * conv_p.OW,
          conv_p.KH * conv_p.KW * conv_p.IC,
          pmat,
          zero_pt),
      conv_p_(conv_p),
      sdata_(sdata) {
  assert(conv_p.G == 1 && "Groups != 1 not supported yet");

  if (cpuinfo_has_x86_avx512f()) {
    BaseType::brow_ = PackingTraits<T, accT, inst_set_t::avx512>::MCB;
    BaseType::bcol_ = PackingTraits<T, accT, inst_set_t::avx512>::KCB;
    row_interleave_B_ =
        PackingTraits<T, accT, inst_set_t::avx512>::ROW_INTERLEAVE;
  } else if (cpuinfo_has_x86_avx2()) {
    BaseType::brow_ = PackingTraits<T, accT, inst_set_t::avx2>::MCB;
    BaseType::bcol_ = PackingTraits<T, accT, inst_set_t::avx2>::KCB;
    row_interleave_B_ =
        PackingTraits<T, accT, inst_set_t::avx2>::ROW_INTERLEAVE;
  } else {
    // TODO: Have default slower path
    assert(0 && "unsupported architecure");
  }
  if (pmat) {
    BaseType::buf_ = pmat;
  } else {
    BaseType::bufAllocatedHere_ = true;
    BaseType::buf_ = static_cast<T*>(
        fbgemmAlignedAlloc(64, BaseType::brow_ * BaseType::bcol_ * sizeof(T)));
        //aligned_alloc(64, BaseType::brow_ * BaseType::bcol_ * sizeof(T)));
  }
  if (row_offset) {
    rowOffsetAllocatedHere = false;
    row_offset_ = row_offset;
  } else {
    rowOffsetAllocatedHere = true;
    row_offset_ = static_cast<int32_t*>(
        fbgemmAlignedAlloc(64, BaseType::brow_ * sizeof(int32_t)));
  }
}

template <typename T, typename accT>
void PackAWithIm2Col<T, accT>::pack(const block_type_t& block) {
  block_type_t block_p = {block.row_start,
                          block.row_size,
                          block.col_start,
                          (block.col_size + row_interleave_B_ - 1) /
                              row_interleave_B_ * row_interleave_B_};
  BaseType::packedBlock(block_p);
  T* out = BaseType::getBuf();

  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    int n = i / (conv_p_.OH * conv_p_.OW);
    int hw = i % (conv_p_.OH * conv_p_.OW);
    int w = hw % conv_p_.OW;
    int h = hw / conv_p_.OW;
    for (int j = block.col_start;
         j < block.col_start + block.col_size + conv_p_.IC - 1;
         j += conv_p_.IC) {
      int j_blk_id = j / conv_p_.IC;
      // max( j_blk_id * IC, START)  -> min( END, (j_blk_id + 1) * IC )
      int j_blk_start = std::max(j_blk_id * conv_p_.IC, block.col_start);
      int j_blk_end = std::min(
          (j_blk_id + 1) * conv_p_.IC, block.col_start + block.col_size);
      if (j_blk_start >= j_blk_end) {
        break;
      }

      int rs = j / conv_p_.IC;
      int s = rs % conv_p_.KW;
      int r = rs / conv_p_.KW;

      int w_in = -conv_p_.pad_w + w * conv_p_.stride_w + s;
      int h_in = -conv_p_.pad_h + h * conv_p_.stride_h + r;

      if (h_in < 0 || h_in >= conv_p_.IH || w_in < 0 || w_in >= conv_p_.IW) {
        // Please note that padding for convolution should be filled with
        // zero_pt
        std::memset(
            &out
                [(i - block.row_start) * BaseType::blockColSize() +
                 (j_blk_start - block.col_start)],
            BaseType::zeroPoint(),
            sizeof(T) * (j_blk_end - j_blk_start));
      } else {
        std::memcpy(
            &out
                [(i - block.row_start) * BaseType::blockColSize() +
                 j_blk_start - block.col_start],
            &sdata_
                [((n * conv_p_.IH + h_in) * conv_p_.IW + w_in) * conv_p_.IC +
                 (j_blk_start % conv_p_.IC)],
            sizeof(T) * (j_blk_end - j_blk_start));
      }
    }
    // zero fill
    // Please see the comment in PackAMatrix.cc for zero vs zero_pt fill.
    if ((block_p.col_start + block_p.col_size) -
            (block.col_start + block.col_size) >
        0) {
      std::memset(
          &out
              [(i - block.row_start) * BaseType::blockColSize() +
               (block.col_size)],
          0,
          sizeof(T) *
              ((block_p.col_start + block_p.col_size) -
               (block.col_start + block.col_size)));
    }
  }
}

template <typename T, typename accT>
void PackAWithIm2Col<T, accT>::printPackedMatrix(std::string name) {
  std::cout << name << ":"
            << "[" << BaseType::numPackedRows() << ", "
            << BaseType::numPackedCols() << "]" << std::endl;

  T* out = BaseType::getBuf();
  for (auto r = 0; r < BaseType::numPackedRows(); ++r) {
    for (auto c = 0; c < BaseType::numPackedCols(); ++c) {
      T val = out[r * BaseType::blockColSize() + c];
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
int PackAWithIm2Col<T, accT>::rowOffsetBufferSize() {
  if (cpuinfo_initialize()) {
    if (cpuinfo_has_x86_avx512f()) {
      return PackingTraits<T, accT, inst_set_t::avx512>::MCB;
    } else if (cpuinfo_has_x86_avx2()) {
      return PackingTraits<T, accT, inst_set_t::avx2>::MCB;
    } else {
      // TODO: Have default slower path
      assert(0 && "unsupported architecture");
      return -1;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

template class PackAWithIm2Col<uint8_t, int32_t>;
template class PackAWithIm2Col<uint8_t, int16_t>;
} // namespace fbgemm2
