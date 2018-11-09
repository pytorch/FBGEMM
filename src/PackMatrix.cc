/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cpuinfo.h>
#include <iomanip>
#include <stdexcept>
#include <type_traits>
#include "fbgemm/ConvUtils.h"
#include "fbgemm/Fbgemm.h"

namespace fbgemm {

template <typename PT, typename inpType, typename accType>
PackMatrix<PT, inpType, accType>::PackMatrix(
    int32_t rows,
    int32_t cols,
    inpType* buf,
    int32_t zero_pt)
    : buf_(buf), nrows_(rows), ncols_(cols), zero_pt_(zero_pt) {
  bufAllocatedHere_ = false;
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
}

template <typename PT, typename inpType, typename accType>
int PackMatrix<PT, inpType, accType>::packedBufferSize(int rows, int cols) {
  if (cpuinfo_has_x86_avx512f()) {
    if (isA()) {
      return PackingTraits<inpType, accType, inst_set_t::avx512>::MCB *
          PackingTraits<inpType, accType, inst_set_t::avx512>::KCB;
    } else {
      int rowBlock = PackingTraits<inpType, accType, inst_set_t::avx512>::KCB;
      int colBlock = PackingTraits<inpType, accType, inst_set_t::avx512>::NCB;
      return (((rows + rowBlock - 1) / rowBlock) * rowBlock) *
          (((cols + colBlock - 1) / colBlock) * colBlock);
    }
  } else if (cpuinfo_has_x86_avx2()) {
    if (isA()) {
      return PackingTraits<inpType, accType, inst_set_t::avx2>::MCB *
          PackingTraits<inpType, accType, inst_set_t::avx2>::KCB;
    } else {
      int rowBlock = PackingTraits<inpType, accType, inst_set_t::avx2>::KCB;
      int colBlock = PackingTraits<inpType, accType, inst_set_t::avx2>::NCB;
      return (((rows + rowBlock - 1) / rowBlock) * rowBlock) *
          (((cols + colBlock - 1) / colBlock) * colBlock);
    }
  } else {
    // TODO: Have default slower path
    assert(0 && "unsupported architecure");
  }
  return -1;
}

// int32 accumulation
template class PackMatrix<PackAMatrix<uint8_t, int32_t>, uint8_t, int32_t>;

template class PackMatrix<
    PackAWithRowOffset<uint8_t, int32_t>,
    uint8_t,
    int32_t>;

template class PackMatrix<PackAWithIm2Col<uint8_t, int32_t>, uint8_t, int32_t>;
template class PackMatrix<
    PackAWithIm2Col<uint8_t, int32_t, 3>,
    uint8_t,
    int32_t>;

template class PackMatrix<
    PackAWithQuantRowOffset<uint8_t, int32_t>,
    uint8_t,
    int32_t>;

template class PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>;

// int16 accumulation
template class PackMatrix<PackAWithIm2Col<uint8_t, int16_t>, uint8_t, int16_t>;
template class PackMatrix<
    PackAWithIm2Col<uint8_t, int16_t, 3>,
    uint8_t,
    int16_t>;

template class PackMatrix<
    PackAWithRowOffset<uint8_t, int16_t>,
    uint8_t,
    int16_t>;

template class PackMatrix<PackAMatrix<uint8_t, int16_t>, uint8_t, int16_t>;

template class PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>;
} // namespace fbgemm
