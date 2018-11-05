/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <cpuinfo.h>

#include "fbgemm/Fbgemm.h"

namespace fbgemm2 {

template <typename T, typename accT, int SPATIAL_DIM>
PackAWithIm2Col<T, accT, SPATIAL_DIM>::PackAWithIm2Col(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const T* sdata,
    inpType* pmat,
    int32_t zero_pt,
    int32_t* row_offset)
    : PackMatrix<PackAWithIm2Col<T, accT, SPATIAL_DIM>, T, accT>(
          conv_p.MB *
              std::accumulate(
                  conv_p.OUT_DIM.begin(),
                  conv_p.OUT_DIM.end(),
                  1,
                  std::multiplies<int>()),
          std::accumulate(
              conv_p.K.begin(),
              conv_p.K.end(),
              1,
              std::multiplies<int>()) *
              conv_p.IC,
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

template <typename T, typename accT, int SPATIAL_DIM>
void PackAWithIm2Col<T, accT, SPATIAL_DIM>::pack(const block_type_t& block) {
  block_type_t block_p = {block.row_start,
                          block.row_size,
                          block.col_start,
                          (block.col_size + row_interleave_B_ - 1) /
                              row_interleave_B_ * row_interleave_B_};
  BaseType::packedBlock(block_p);
  T* out = BaseType::getBuf();

  if (SPATIAL_DIM == 3) { // static if
    for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
      int n =
          i / (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1] * conv_p_.OUT_DIM[2]);
      int thw =
          i % (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1] * conv_p_.OUT_DIM[2]);
      int w = thw % conv_p_.OUT_DIM[2];
      int h = thw / conv_p_.OUT_DIM[2] % conv_p_.OUT_DIM[1];
      int t = thw / conv_p_.OUT_DIM[2] / conv_p_.OUT_DIM[1];
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

        int qrs = j / conv_p_.IC;
        int s = qrs % conv_p_.K[2];
        int r = qrs / conv_p_.K[2] % conv_p_.K[1];
        int q = qrs / conv_p_.K[2] / conv_p_.K[1];

        int t_in = -conv_p_.pad[0] + t * conv_p_.stride[0] + q;
        int h_in = -conv_p_.pad[1] + h * conv_p_.stride[1] + r;
        int w_in = -conv_p_.pad[2] + w * conv_p_.stride[2] + s;

        if (t_in < 0 || t_in >= conv_p_.IN_DIM[0] || h_in < 0 ||
            h_in >= conv_p_.IN_DIM[1] || w_in < 0 ||
            w_in >= conv_p_.IN_DIM[2]) {
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
                  [(((n * conv_p_.IN_DIM[0] + t_in) * conv_p_.IN_DIM[1] +
                     h_in) *
                        conv_p_.IN_DIM[2] +
                    w_in) *
                       conv_p_.IC +
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
    return;
  }

  assert(SPATIAL_DIM == 2 && "unsupported conv dimension");

  for (int i = block.row_start; i < block.row_start + block.row_size; ++i) {
    int n = i / (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1]);
    int hw = i % (conv_p_.OUT_DIM[0] * conv_p_.OUT_DIM[1]);
    int w = hw % conv_p_.OUT_DIM[1];
    int h = hw / conv_p_.OUT_DIM[1];
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
      int s = rs % conv_p_.K[1];
      int r = rs / conv_p_.K[1];

      int h_in = -conv_p_.pad[0] + h * conv_p_.stride[0] + r;
      int w_in = -conv_p_.pad[1] + w * conv_p_.stride[1] + s;

      if (h_in < 0 || h_in >= conv_p_.IN_DIM[0] || w_in < 0 ||
          w_in >= conv_p_.IN_DIM[1]) {
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
                [((n * conv_p_.IN_DIM[0] + h_in) * conv_p_.IN_DIM[1] + w_in) *
                     conv_p_.IC +
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

template <typename T, typename accT, int SPATIAL_DIM>
void PackAWithIm2Col<T, accT, SPATIAL_DIM>::printPackedMatrix(
    std::string name) {
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

template <typename T, typename accT, int SPATIAL_DIM>
int PackAWithIm2Col<T, accT, SPATIAL_DIM>::rowOffsetBufferSize() {
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
template class PackAWithIm2Col<uint8_t, int32_t, 3>;
template class PackAWithIm2Col<uint8_t, int16_t, 3>;

} // namespace fbgemm2
