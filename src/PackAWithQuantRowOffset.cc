/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cpuinfo.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename T, typename accT>
PackAWithQuantRowOffset<T, accT>::PackAWithQuantRowOffset(
    matrix_op_t trans,
    int32_t nRow,
    int32_t nCol,
    const float* smat,
    int32_t ld,
    inpType* pmat,
    float scale,
    int32_t zero_pt,
    int groups,
    int32_t* row_offset)
    : PackMatrix<PackAWithQuantRowOffset<T, accT>, T, accT>(
          nRow,
          nCol,
          pmat,
          groups),
      trans_(trans),
      smat_(smat),
      ld_(ld),
      scale_(scale),
      zero_pt_(zero_pt),
      row_offset_(row_offset) {
  rowOffsetAllocatedHere = false;

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
    assert(0 && "unknown architecure");
  }
  if (BaseType::numCols() % groups != 0) {
    throw std::runtime_error(
        "groups = " + std::to_string(groups) +
        " does not divide numCols = " + std::to_string(BaseType::numCols()));
  }
  if (pmat) {
    BaseType::buf_ = pmat;
  } else {
    BaseType::bufAllocatedHere_ = true;
    BaseType::buf_ = (T*)fbgemmAlignedAlloc(
        64, BaseType::brow_ * BaseType::bcol_ * sizeof(T));
  }
  if (!row_offset_) {
    rowOffsetAllocatedHere = true;
    row_offset_ = reinterpret_cast<int32_t*>(
        fbgemmAlignedAlloc(64, BaseType::brow_ * sizeof(accT)));
  }
}

template <typename T, typename accT>
void PackAWithQuantRowOffset<T, accT>::pack(const block_type_t& block) {
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

  float smat_transposed[block.row_size * block.col_size];
  if (tr) {
    transpose_simd(
        block.col_size,
        block.row_size,
        smat_ + block.col_start * ld_ + block.row_start,
        ld_,
        smat_transposed,
        block.col_size);
  }
  const float* smat_temp =
      tr ? smat_transposed : smat_ + block.row_start * ld_ + block.col_start;
  int32_t ld_temp = tr ? block.col_size : ld_;

#if defined(__AVX2__) && defined(__FMA__)
  constexpr int VLEN = 8;
  __m256 inverse_scale_v = _mm256_set1_ps(1.0f / scale_);
  __m256i shuffle_mask_v = _mm256_set_epi8(
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0xff,
      0x0c,
      0x08,
      0x04,
      0x00);
  __m256i permute_mask_v =
      _mm256_set_epi32(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00);
#endif

  for (int i = 0; i < block.row_size; ++i) {
    int32_t row_sum = row_offset_acc ? row_offset_buf[i] : 0;
    int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
    static_assert(
        std::is_same<T, uint8_t>::value,
        "PackAWithQuantRowOffset<T, accT>::pack only works for T == uint8_t");
    for (; j < block.col_size / VLEN * VLEN; j += VLEN) {
      __m256 val_v = _mm256_loadu_ps(smat_temp + i * ld_temp + j);
      __m256 transformed_v = _mm256_fmadd_ps(
          val_v, inverse_scale_v, _mm256_set1_ps(zero_pt_));
      __m256 clipped_v = _mm256_max_ps(
          _mm256_set1_ps(std::numeric_limits<uint8_t>::min()),
          _mm256_min_ps(
              transformed_v,
              _mm256_set1_ps(std::numeric_limits<uint8_t>::max())));
      __m256i res_v = _mm256_cvtps_epi32(clipped_v);

      // An instruction sequence to save 8 32-bit integers as 8 8-bit integers
      res_v = _mm256_shuffle_epi8(res_v, shuffle_mask_v);
      res_v = _mm256_permutevar8x32_epi32(res_v, permute_mask_v);
      _mm_storel_epi64(
          reinterpret_cast<__m128i*>(out + i * BaseType::blockColSize() + j),
          _mm256_castsi256_si128(res_v));

      for (int j2 = j; j2 < j + VLEN; ++j2) {
        row_sum += out[i * BaseType::blockColSize() + j2];
      }
    }
#endif
    for (; j < block.col_size; ++j) {
      float val = smat_temp[i * ld_temp + j];
      float transformed = val / scale_ + zero_pt_;
      float clipped = std::min<float>(
          std::max<float>(transformed, std::numeric_limits<uint8_t>::min()),
          std::numeric_limits<uint8_t>::max());
      T res = nearbyint(clipped);
      row_sum += res;
      out[i * BaseType::blockColSize() + j] = res;
    }
    // zero fill
    // Please see the comment in PackAMatrix.cc on zero vs zero_pt fill.
    for (; j < block_p.col_size; ++j) {
      out[i * BaseType::blockColSize() + j] = 0;
    }
    row_offset_buf[i] = row_sum;
  }
}

template <typename T, typename accT>
int32_t PackAWithQuantRowOffset<T, accT>::addr(int32_t r, int32_t c) const {
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
void PackAWithQuantRowOffset<T, accT>::printPackedMatrix(std::string name) {
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
int PackAWithQuantRowOffset<T, accT>::rowOffsetBufferSize() {
  if (cpuinfo_initialize()) {
    if (cpuinfo_has_x86_avx512f()) {
      return PackingTraits<T, accT, inst_set_t::avx512>::MCB;
    } else if (cpuinfo_has_x86_avx2()) {
      return PackingTraits<T, accT, inst_set_t::avx2>::MCB;
    } else {
      assert(0 && "unsupported architecture");
      return -1;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

template class PackAWithQuantRowOffset<uint8_t, int32_t>;

} // namespace fbgemm
