/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "ExecuteKernelU8S8.h"
#include <cpuinfo.h>
#include <chrono>


#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double kernel_time = 0.0;
double postprocessing_time = 0.0;
#endif

namespace fbgemm2 {

template <typename packingAMatrix, typename cT, typename processOutputType>
ExecuteKernel<
    packingAMatrix,
    PackBMatrix<int8_t, typename packingAMatrix::accType>,
    cT,
    processOutputType>::
    ExecuteKernel(
        PackMatrix<packingAMatrix, uint8_t, typename packingAMatrix::accType>&
            packA,
        PackMatrix<
            PackBMatrix<int8_t, typename packingAMatrix::accType>,
            int8_t,
            typename packingAMatrix::accType>& packB,
        int32_t kBlock,
        cT* matC,
        int32_t* C_buffer,
        int32_t ldc,
        const processOutputType& outputProcess)
    : packedA_(packA),
      packedB_(packB),
      kBlock_(kBlock),
      matC_(matC),
      C_buffer_(C_buffer),
      ldc_(ldc),
      outputProcess_(outputProcess) {
  if (cpuinfo_has_x86_avx512f()) {
    mbSize_ = PackingTraits<
        int8_t,
        typename packingAMatrix::accType,
        inst_set_t::avx512>::MCB;
    nbSize_ = PackingTraits<
        int8_t,
        typename packingAMatrix::accType,
        inst_set_t::avx512>::NCB;
  } else if (cpuinfo_has_x86_avx2()) {
    mbSize_ = PackingTraits<
        int8_t,
        typename packingAMatrix::accType,
        inst_set_t::avx2>::MCB;
    nbSize_ = PackingTraits<
        int8_t,
        typename packingAMatrix::accType,
        inst_set_t::avx2>::NCB;
  } else {
    assert(0 && "unsupported architecure");
  }
  C_tile_ = new int32_t[mbSize_ * nbSize_];
}

template <typename packingAMatrix, typename cT, typename processOutputType>
void ExecuteKernel<
    packingAMatrix,
    PackBMatrix<int8_t, typename packingAMatrix::accType>,
    cT,
    processOutputType>::execute(int kBlock) {
  // packedA_.printPackedMatrix("packedA from kernel");
  // packedB_.printPackedMatrix("packedB from kernel");

  int32_t bColBlocks = packedB_.blockCols();

  int8_t* bBuf;
  int8_t* bBuf_pf;

  uint8_t* aBuf = packedA_.getBuf(0);

  int32_t packed_rows_A = packedA_.numPackedRows();
  int32_t row_start_A = packedA_.packedRowStart();

  bool lastKBlock = packedB_.isThisLastKBlock(kBlock);
  bool accum = kBlock > 0;

  typename BaseType::jit_micro_kernel_fp fn;

  if (cpuinfo_initialize()) {
    if (cpuinfo_has_x86_avx512f()) {
      fn = BaseType::template getOrCreate<inst_set_t::avx512>(
          accum,
          packed_rows_A,
          packedB_.blockColSize(),
          packedA_.numPackedCols(),
          nbSize_);
    } else if (cpuinfo_has_x86_avx2()) {
      fn = BaseType::template getOrCreate<inst_set_t::avx2>(
          accum,
          packed_rows_A,
          packedB_.blockColSize(),
          packedA_.numPackedCols(),
          nbSize_);
    } else {
      // TODO: Have default slower path
      assert(0 && "unsupported architecture");
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
  double dt;
  t_start = std::chrono::high_resolution_clock::now();
#endif

  for (int jb = 0; jb < bColBlocks; ++jb) {

    bBuf = packedB_.getBuf(jb, kBlock);
    // prefetch addr of the next packed block of B matrix
    bBuf_pf = packedB_.getBuf(jb == bColBlocks - 1 ? jb : jb + 1, kBlock);

    // Reuse the first rowblock of C_buffer_ unless when C_buffer_ is same as
    // matC_ (inplace output processing)
    int32_t* C_buffer_row_start = C_buffer_ +
        ((C_buffer_ == reinterpret_cast<int32_t*>(matC_)) ? row_start_A * ldc_
                                                          : 0);
    int32_t* C_buffer_start = C_buffer_row_start + jb * nbSize_;
    int32_t leadingDim = ldc_;
    if (packedB_.isThereColRemainder() && (jb == bColBlocks - 1)) {
      // In case we will access memory past C_buffer_, we use C_tile_ instead.
      C_buffer_start = C_tile_;
      leadingDim = nbSize_;
    }

    fn(aBuf,
       bBuf,
       bBuf_pf,
       C_buffer_start,
       packedA_.numPackedCols(),
       leadingDim);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      t_end = std::chrono::high_resolution_clock::now();
      dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
               .count();
      kernel_time += (dt);
      t_start = std::chrono::high_resolution_clock::now();
#endif

    // Output processing is done only once per rowblock
    if (lastKBlock && jb == bColBlocks - 1) {
      // When C_tile_ is used for the last column block, we need a separate
      // handling for the last column block.
      int32_t nSize =
          C_buffer_start == C_tile_ ? jb * nbSize_ : packedB_.numCols();
      if (nSize) {
        if (cpuinfo_has_x86_avx512f()) {
          // TODO: avx512 path
          // Currently use avx2 code
          outputProcess_.template f<inst_set_t::avx2>(
              matC_,
              C_buffer_row_start,
              {row_start_A, packed_rows_A, 0, nSize},
              ldc_,
              ldc_);
        } else if (cpuinfo_has_x86_avx2()) {
          outputProcess_.template f<inst_set_t::avx2>(
              matC_,
              C_buffer_row_start,
              {row_start_A, packed_rows_A, 0, nSize},
              ldc_,
              ldc_);
        } else {
          // TODO: Have default slower path
          assert(0 && "unsupported architecure");
        }
      }

      if (C_buffer_start == C_tile_) {
        if (cpuinfo_has_x86_avx512f()) {
          // TODO: avx512 path
          // Currently use avx2 code
          outputProcess_.template f<inst_set_t::avx2>(
              matC_,
              C_tile_,
              {row_start_A, packed_rows_A, jb * nbSize_, packedB_.lastBcol()},
              ldc_,
              leadingDim);
        } else if (cpuinfo_has_x86_avx2()) {
          outputProcess_.template f<inst_set_t::avx2>(
              matC_,
              C_tile_,
              {row_start_A, packed_rows_A, jb * nbSize_, packedB_.lastBcol()},
              ldc_,
              leadingDim);
        } else {
          // TODO: Have default slower path
          assert(0 && "unsupported architecure");
        }
      }
    } // output processing

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      t_end = std::chrono::high_resolution_clock::now();
      dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
               .count();
      postprocessing_time += (dt);
      t_start = std::chrono::high_resolution_clock::now();
#endif

  } // for each j block
}
template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    uint8_t,
    ReQuantizeOutput<false /* FUSE_RELU*/>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    uint8_t,
    ReQuantizeOutput<true>>;

template class ExecuteKernel<
    PackAWithQuantRowOffset<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    float,
    ReQuantizeForFloat<false>>;

template class ExecuteKernel<
    PackAWithQuantRowOffset<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    float,
    ReQuantizeForFloat<true>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    float,
    ReQuantizeForFloat<false>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    float,
    ReQuantizeForFloat<true>>;

template class ExecuteKernel<
    PackAMatrix<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAMatrix<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    uint8_t,
    ReQuantizeOutput<false>>;

template class ExecuteKernel<
    PackAMatrix<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    uint8_t,
    DoSpmdmOnInpBuffer<
        ReQuantizeOutput<false>::outType,
        int32_t,
        ReQuantizeOutput<false>>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    uint8_t,
    DoSpmdmOnInpBuffer<
        ReQuantizeOutput<true>::outType,
        int32_t,
        ReQuantizeOutput<true>>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    float,
    DoSpmdmOnInpBuffer<
        ReQuantizeForFloat<false>::outType,
        int32_t,
        ReQuantizeForFloat<false>>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    uint8_t,
    ReQuantizeOutput<false>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    uint8_t,
    ReQuantizeOutput<true>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAWithIm2Col<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAWithIm2Col<uint8_t, int16_t, 3>,
    PackBMatrix<int8_t, int16_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAWithIm2Col<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAWithIm2Col<uint8_t, int32_t, 3>,
    PackBMatrix<int8_t, int32_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAWithQuantRowOffset<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    float,
    ReQuantizeForFloat<false>>;

template class ExecuteKernel<
    PackAMatrix<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    int32_t,
    DoNothing<int32_t, int32_t>>;

} // namespace fbgemm2
