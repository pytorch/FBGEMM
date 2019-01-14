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

namespace fbgemm {

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
        const processOutputType& outputProcess,
        int thread_id,
        int num_threads)
    : packedA_(packA),
      packedB_(packB),
      kBlock_(kBlock),
      matC_(matC),
      C_buffer_(C_buffer),
      ldc_(ldc),
      outputProcess_(outputProcess),
      thread_id_(thread_id),
      num_threads_(num_threads) {
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

  int group = kBlock / packedB_.blockRows();
  int NDim = packedB_.numCols();
  bool lastKBlock = packedB_.isThisLastKBlock(kBlock % packedB_.blockRows());
  bool accum = (kBlock % packedB_.blockRows()) > 0;

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

    // If the accumulation buffer C_buffer_ is the same as matC_ (inplace output
    // processing), then each thread use the different parts of output buffer
    // matC_;
    // Otherwise, each thread uses different portions of the accumulation
    // buffer C_buffer_. If m is large enough (m >= nthreads * MC), then we only
    // need to use (nthreads * MC) x n portion of C_buffer_, each thread access
    // the C_buffer_row_start as tid * MC * ldc_; else when m is very small, we
    // juse use the whole m x n C_buffer_: each thread use the different
    // portion.
    int32_t* C_buffer_row_start = C_buffer_ +
        ((C_buffer_ == reinterpret_cast<int32_t*>(matC_) ||
          num_threads_ * mbSize_ > packedA_.numRows())
             ? row_start_A * ldc_ + NDim * group
             : thread_id_ * mbSize_ * ldc_ + NDim * group);

    int32_t* C_buffer_start = C_buffer_row_start + jb * nbSize_;
    int32_t leadingDim = ldc_;
    if (packedB_.isThereColRemainder() && (jb == bColBlocks - 1)) {
      // In case we will access memory past C_buffer_, we use C_tile_ scratchpad
      // instead.
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

    // Output processing is done only once per rowblock to amortize overhead
    // and for better spatial locality.
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
              {row_start_A, packed_rows_A, NDim * group, nSize},
              ldc_,
              ldc_);
        } else if (cpuinfo_has_x86_avx2()) {
          outputProcess_.template f<inst_set_t::avx2>(
              matC_,
              C_buffer_row_start,
              {row_start_A, packed_rows_A, NDim * group, nSize},
              ldc_,
              ldc_);
        } else {
          // TODO: Have default slower path
          assert(0 && "unsupported architecure");
        }
      }

      if (C_buffer_start == C_tile_) {
        // When C_tile_ scratchpad was used to avoid accessing memory past
        // C_buffer_ .
        if (cpuinfo_has_x86_avx512f()) {
          // TODO: avx512 path
          // Currently use avx2 code
          outputProcess_.template f<inst_set_t::avx2>(
              matC_,
              C_tile_,
              {row_start_A,
               packed_rows_A,
               NDim * group + jb * nbSize_,
               packedB_.lastBcol()},
              ldc_,
              leadingDim);
        } else if (cpuinfo_has_x86_avx2()) {
          outputProcess_.template f<inst_set_t::avx2>(
              matC_,
              C_tile_,
              {row_start_A,
               packed_rows_A,
               NDim * group + jb * nbSize_,
               packedB_.lastBcol()},
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

////////////////////////////////////////////////////////////////////////////////
// ReQuantizeOutput
#define INSTANTIATE_BASE(PACK_A, ACC_T, RELU, Q_GRAN) \
  template class ExecuteKernel<                       \
      PACK_A<uint8_t, ACC_T>,                         \
      PackBMatrix<int8_t, ACC_T>,                     \
      uint8_t,                                        \
      ReQuantizeOutput<RELU, Q_GRAN>>;

#define INSTANTIATE_Q_GRANS(PACK_A, ACC_T, RELU)                          \
  INSTANTIATE_BASE(PACK_A, ACC_T, RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BASE(PACK_A, ACC_T, RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BASE(PACK_A, ACC_T, RELU, QuantizationGranularity::OUT_CHANNEL);

#define INSTANTIATE_RELU(PACK_A, ACC_T)      \
  INSTANTIATE_Q_GRANS(PACK_A, ACC_T, false); \
  INSTANTIATE_Q_GRANS(PACK_A, ACC_T, true);

#define INSTANTIATE_ACC_T(PACK_A)    \
  INSTANTIATE_RELU(PACK_A, int32_t); \
  INSTANTIATE_RELU(PACK_A, int16_t);

INSTANTIATE_ACC_T(PackAMatrix);
INSTANTIATE_ACC_T(PackAWithRowOffset);

#undef INSTANTIATE_ACC_T
#undef INSTANTIATE_RELU
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN) \
  template class ExecuteKernel<                            \
      PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,        \
      PackBMatrix<int8_t, ACC_T>,                          \
      uint8_t,                                             \
      ReQuantizeOutput<RELU, Q_GRAN>>;

#define INSTANTIATE_Q_GRANS(ACC_T, RELU, SPATIAL_DIM)                          \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BASE(                                                            \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::OUT_CHANNEL);

#define INSTANTIATE_SPATIAL_DIM(ACC_T, RELU) \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 2);       \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 3);

#define INSTANTIATE_RELU(ACC_T)          \
  INSTANTIATE_SPATIAL_DIM(ACC_T, false); \
  INSTANTIATE_SPATIAL_DIM(ACC_T, true);

INSTANTIATE_RELU(int32_t);
INSTANTIATE_RELU(int16_t);

#undef INSTANTIATE_RELU
#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

////////////////////////////////////////////////////////////////////////////////
// ReQuantizeForFloat
#define INSTANTIATE_BASE(PACK_A, RELU, Q_GRAN) \
  template class ExecuteKernel<                \
      PACK_A<uint8_t, int32_t>,                \
      PackBMatrix<int8_t, int32_t>,            \
      float,                                   \
      ReQuantizeForFloat<RELU, Q_GRAN>>;

#define INSTANTIATE_Q_GRANS(PACK_A, RELU)                          \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::OUT_CHANNEL);

#define INSTANTIATE_RELU(PACK_A)      \
  INSTANTIATE_Q_GRANS(PACK_A, false); \
  INSTANTIATE_Q_GRANS(PACK_A, true);

INSTANTIATE_RELU(PackAWithRowOffset);
INSTANTIATE_RELU(PackAWithQuantRowOffset);

#undef INSTANTIATE_RELU
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN) \
  template class ExecuteKernel<                            \
      PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,        \
      PackBMatrix<int8_t, ACC_T>,                          \
      float,                                               \
      ReQuantizeForFloat<RELU, Q_GRAN>>;

#define INSTANTIATE_Q_GRANS(ACC_T, RELU, SPATIAL_DIM)                          \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BASE(                                                            \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::OUT_CHANNEL);

#define INSTANTIATE_SPATIAL_DIM(ACC_T, RELU) \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 2);       \
  INSTANTIATE_Q_GRANS(ACC_T, RELU, 3);

#define INSTANTIATE_RELU(ACC_T)          \
  INSTANTIATE_SPATIAL_DIM(ACC_T, false); \
  INSTANTIATE_SPATIAL_DIM(ACC_T, true);

INSTANTIATE_RELU(int32_t);
INSTANTIATE_RELU(int16_t);

#undef INSTANTIATE_RELU
#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    float,
    ReQuantizeForFloat<false /* FUSE_RELU*/>>;

////////////////////////////////////////////////////////////////////////////////
// DoSpmdmOnInpBuffer
#define INSTANTIATE_BASE(RELU, Q_GRAN)      \
  template class ExecuteKernel<             \
      PackAWithRowOffset<uint8_t, int16_t>, \
      PackBMatrix<int8_t, int16_t>,         \
      uint8_t,                              \
      DoSpmdmOnInpBuffer<uint8_t, int32_t, ReQuantizeOutput<RELU, Q_GRAN>>>;

#define INSTANTIATE_Q_GRANS(RELU)                          \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::OUT_CHANNEL);

INSTANTIATE_Q_GRANS(false);
INSTANTIATE_Q_GRANS(true);

#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(RELU, Q_GRAN)   \
  template class ExecuteKernel<          \
      PackAWithIm2Col<uint8_t, int16_t>, \
      PackBMatrix<int8_t, int16_t>,      \
      uint8_t,                           \
      DoSConvOnInpBuffer<uint8_t, int32_t, ReQuantizeOutput<RELU, Q_GRAN>>>;

#define INSTANTIATE_Q_GRANS(RELU)                          \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::OUT_CHANNEL);

INSTANTIATE_Q_GRANS(false);
INSTANTIATE_Q_GRANS(true);

#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    float,
    DoSpmdmOnInpBuffer<float, int32_t, ReQuantizeForFloat<false>>>;

////////////////////////////////////////////////////////////////////////////////
// memCopy
#define INSTANTIATE_BASE(PACK_A, ACC_T) \
  template class ExecuteKernel<         \
      PACK_A<uint8_t, ACC_T>,           \
      PackBMatrix<int8_t, ACC_T>,       \
      int32_t,                          \
      memCopy<>>;

#define INSTANTIATE_ACC_T(PACK_A)   \
  INSTANTIATE_BASE(PACK_A, int32_t) \
  INSTANTIATE_BASE(PACK_A, int16_t)

INSTANTIATE_ACC_T(PackAMatrix);
INSTANTIATE_ACC_T(PackAWithRowOffset);

#undef INSTANTIATE_ACC_T
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(ACC_T, SPATIAL_DIM)        \
  template class ExecuteKernel<                     \
      PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>, \
      PackBMatrix<int8_t, ACC_T>,                   \
      int32_t,                                      \
      memCopy<>>;

#define INSTANTIATE_SPATIAL_DIM(ACC_T) \
  INSTANTIATE_BASE(ACC_T, 2);          \
  INSTANTIATE_BASE(ACC_T, 3);

INSTANTIATE_SPATIAL_DIM(int32_t);
INSTANTIATE_SPATIAL_DIM(int16_t);

#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_BASE

template class ExecuteKernel<
    PackAWithQuantRowOffset<uint8_t, int32_t>,
    PackBMatrix<int8_t, int32_t>,
    int32_t,
    memCopy<>>;

template class ExecuteKernel<
    PackAMatrix<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    int32_t,
    DoNothing<int32_t, int32_t>>;

} // namespace fbgemm
