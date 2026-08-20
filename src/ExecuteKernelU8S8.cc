/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "./ExecuteKernelU8S8.h" // @manual
#include <cpuinfo.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double kernel_time = 0.0;
double postprocessing_time = 0.0;
#endif

namespace fbgemm {

#ifdef __aarch64__
namespace {

// Saturating narrow to int16, i.e. what every x86 "s"-suffixed pack/add does.
std::int16_t satInt16(std::int32_t v) {
  return static_cast<std::int16_t>(std::clamp(v, -32768, 32767));
}

// Portable reference implementation of the u8s8 GEMM micro-kernel for aarch64.
//
// On x86 the compute kernel is JIT-generated assembly (see
// GenerateKernelU8S8*.cc); no such kernel exists for arm, so we compute the
// mc x nc x kc block directly from the packed A/B buffers. Both the packed
// layout and the integer arithmetic mirror the avx2 kernel exactly:
//   - A is packed row-major with row stride kBlock (== KCB); element A[i][k]
//     lives at aBuf[i * kBlock + k] and is unsigned 8-bit.
//   - B is packed with row_interleave consecutive K values adjacent per column;
//     element B[k][n] lives at
//       bBuf[k * nBlock + n * row_interleave + (k % row_interleave)]
//     (k here is already a multiple of row_interleave) and is signed 8-bit.
//   - acc16 (accT == int16_t, row_interleave == 2): vpmaddubsw multiplies the
//     unsigned A byte by the signed B byte and horizontally adds the pair into
//     an int16 that saturates; the running accumulator is a saturating int16
//     add (vpaddsw). The int16 result is sign-extended to int32 on store.
//   - acc32 (accT == int32_t, row_interleave == 4): each pair is reduced with
//     the saturating int16 vpmaddubsw, then widened and summed to int32
//     (vpmaddwd) and accumulated without saturation (vpaddd).
// `accum` selects overwrite vs. accumulate into the int32 C buffer, matching
// the JIT store path (used across successive K-blocks).
//
// Loop order mirrors the register blocking of the JIT kernel: one row of
// accumulators is held across the whole K sweep, so the inner loop walks the
// packed B block and the accumulators contiguously. Reversing it (accumulating
// a single C element at a time) strides B by row_interleave * NCB per step and
// re-reads the whole block for every column, which costs well over an order of
// magnitude.
template <typename accT>
void computeBlockRef(
    const uint8_t* aBuf,
    const int8_t* bBuf,
    int32_t* cBuf,
    int mc,
    int nc,
    int kc,
    int kBlock,
    int nBlock,
    bool accum,
    int ldc) {
  constexpr int row_interleave = std::is_same_v<accT, std::int16_t> ? 2 : 4;
  static thread_local std::vector<accT> acc;
  acc.resize(nc);

  for (int i = 0; i < mc; ++i) {
    std::fill(acc.begin(), acc.end(), accT{0});
    for (int k = 0; k < kc; k += row_interleave) {
      const uint8_t* aPtr = aBuf + i * kBlock + k;
      const int8_t* bPtr = bBuf + k * nBlock;
      if constexpr (std::is_same_v<accT, std::int16_t>) {
        const std::int32_t a0 = aPtr[0];
        const std::int32_t a1 = aPtr[1];
        for (int n = 0; n < nc; ++n) {
          // vpmaddubsw: (u8 * s8) + (u8 * s8) saturated to int16.
          const std::int16_t prod =
              satInt16(a0 * bPtr[2 * n] + a1 * bPtr[2 * n + 1]);
          // vpaddsw: saturating int16 accumulate.
          acc[n] = satInt16(acc[n] + prod);
        }
      } else {
        const std::int32_t a0 = aPtr[0];
        const std::int32_t a1 = aPtr[1];
        const std::int32_t a2 = aPtr[2];
        const std::int32_t a3 = aPtr[3];
        for (int n = 0; n < nc; ++n) {
          const std::int16_t p01 =
              satInt16(a0 * bPtr[4 * n] + a1 * bPtr[4 * n + 1]);
          const std::int16_t p23 =
              satInt16(a2 * bPtr[4 * n + 2] + a3 * bPtr[4 * n + 3]);
          // vpmaddwd (widen+add) followed by vpaddd (non-saturating int32).
          acc[n] += p01 + p23;
        }
      }
    }
    // Sign-extend the accumulator to int32 to match vpmovsxwd (acc16) / the
    // native int32 register (acc32).
    int32_t* cRow = cBuf + i * ldc;
    if (accum) {
      for (int n = 0; n < nc; ++n) {
        cRow[n] += acc[n];
      }
    } else {
      for (int n = 0; n < nc; ++n) {
        cRow[n] = acc[n];
      }
    }
  }
}
} // namespace
#endif // __aarch64__

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
        cT* matC,
        int32_t* C_buffer,
        int32_t ldc,
        const processOutputType& outputProcess,
        thread_type_t th_info,
        const BlockingFactors* params)
    : CodeGenBase<uint8_t, int8_t, int32_t, typename packingAMatrix::accType>(
          params),
      packedA_(packA),
      packedB_(packB),
      matC_(matC),
      C_buffer_(C_buffer),
      ldc_(ldc),
      outputProcess_(outputProcess),
      th_info_(th_info) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if (params) {
#ifdef __aarch64__
    // aarch64 reference compute path consumes the avx2/user-provided blocking.
    mbSize_ = params->MCB;
    nbSize_ = params->NCB;
    nrMinSize_ = params->NR_MIN;
    nrSize_ = params->NR;
#else
    if (fbgemmHasAvx2Support()) {
      mbSize_ = params->MCB;
      nbSize_ = params->NCB;
      nrMinSize_ = params->NR_MIN;
      nrSize_ = params->NR;
    } else {
      // TODO: Have default slower path
      assert(0 && "unsupported architecure");
      throw std::runtime_error("unsupported architecure");
    }
#endif
  } else {
    const inst_set_t isa = fbgemmInstructionSet();
    switch (isa) {
      case inst_set_t::avx512_vnni:
        std::tie(mbSize_, nbSize_, nrMinSize_, nrSize_) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx512_vnni>::getKernelParams();
        break;

      case inst_set_t::avx512_vnni_ymm:
        std::tie(mbSize_, nbSize_, nrMinSize_, nrSize_) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx512_vnni_ymm>::getKernelParams();
        break;

      case inst_set_t::avx512:
        std::tie(mbSize_, nbSize_, nrMinSize_, nrSize_) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx512>::getKernelParams();
        break;

      case inst_set_t::avx512_ymm:
        std::tie(mbSize_, nbSize_, nrMinSize_, nrSize_) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx512_ymm>::getKernelParams();
        break;

#ifdef __aarch64__
      // aarch64 reference compute path reuses the avx2 kernel blocking params.
      case inst_set_t::sve:
      case inst_set_t::anyarch:
#endif
      case inst_set_t::avx2:
        std::tie(mbSize_, nbSize_, nrMinSize_, nrSize_) = PackingTraits<
            typename packingAMatrix::inpType,
            typename packingAMatrix::accType,
            inst_set_t::avx2>::getKernelParams();
        break;

      default:
        assert(0 && "unknown architecure");
        throw std::runtime_error("unknown architecure");
    }
  }
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

  int8_t* bBuf = nullptr;
  int8_t* bBuf_pf = nullptr;

  uint8_t* aBuf = packedA_.getBuf(0);

  int32_t packed_rows_A = packedA_.numPackedRows();
  int32_t row_start_A = packedA_.packedRowStart();

  int group = kBlock / packedB_.blockRows();
  int NDim = packedB_.numCols();
  bool lastKBlock = packedB_.isThisLastKBlock(kBlock % packedB_.blockRows());
  bool accum = (kBlock % packedB_.blockRows()) > 0;

  int64_t jb_begin = 0, jb_end = 0;
  fbgemmPartition1D(
      th_info_.n_thread_id,
      th_info_.n_num_threads,
      bColBlocks,
      jb_begin,
      jb_end);
  if (jb_end == jb_begin) {
    return;
  }

#ifdef __aarch64__
  // OutputProcessing-inl.h forces its generic implementation on aarch64, so the
  // avx2 tag below resolves to portable code and needs no x86 SIMD support.
  constexpr bool hasOutputProcessingKernel = true;
#else
  const bool hasOutputProcessingKernel = fbgemmHasAvx2Support();

  typename BaseType::jit_micro_kernel_fp fn;

  const inst_set_t isa = fbgemmInstructionSet();
  switch (isa) {
    case inst_set_t::avx512_vnni:
      if constexpr (std::is_same_v<
                        typename packingAMatrix::accType,
                        std::int16_t>) {
        // For AVX512VNNI, we redirect int16_t to int32_t accumulation.
        CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
        fn = codeObj.getOrCreate<inst_set_t::avx512_vnni>(
            accum,
            packed_rows_A,
            packedB_.blockColSize(),
            packedA_.numPackedCols());
      } else {
        fn = BaseType::template getOrCreate<inst_set_t::avx512_vnni>(
            accum,
            packed_rows_A,
            packedB_.blockColSize(),
            packedA_.numPackedCols());
      }
      break;

    case inst_set_t::avx512_vnni_ymm:
      if constexpr (std::is_same_v<
                        typename packingAMatrix::accType,
                        std::int16_t>) {
        // For AVX512VNNI, we redirect int16_t to int32_t accumulation.
        CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
        fn = codeObj.getOrCreate<inst_set_t::avx512_vnni_ymm>(
            accum,
            packed_rows_A,
            packedB_.blockColSize(),
            packedA_.numPackedCols());
      } else {
        fn = BaseType::template getOrCreate<inst_set_t::avx512_vnni_ymm>(
            accum,
            packed_rows_A,
            packedB_.blockColSize(),
            packedA_.numPackedCols());
      }
      break;

    case inst_set_t::avx512:
      fn = BaseType::template getOrCreate<inst_set_t::avx512>(
          accum,
          packed_rows_A,
          packedB_.blockColSize(),
          packedA_.numPackedCols());
      break;

    case inst_set_t::avx512_ymm:
      fn = BaseType::template getOrCreate<inst_set_t::avx512_ymm>(
          accum,
          packed_rows_A,
          packedB_.blockColSize(),
          packedA_.numPackedCols());
      break;

    case inst_set_t::avx2:
      fn = BaseType::template getOrCreate<inst_set_t::avx2>(
          accum,
          packed_rows_A,
          packedB_.blockColSize(),
          packedA_.numPackedCols());
      break;

    default:
      // TODO: Have default slower path
      assert(0 && "unsupported architecture");
      throw std::runtime_error("unsupported architecure");
  }
#endif // __aarch64__

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_end;
  double dt;
  auto t_start = std::chrono::high_resolution_clock::now();
#endif

  for (int jb = jb_begin; jb < jb_end; ++jb) {
    // Columns actually computed for this block: the last one is rounded up to
    // NR_MIN. The JIT bakes this into the generated code; the aarch64 reference
    // kernel takes it as an argument, so it is computed once here for both.
    const int nc = jb == bColBlocks - 1
        ? ((packedB_.lastBcol() - 1) / nrMinSize_ + 1) * nrMinSize_
        : nbSize_;

#ifndef __aarch64__
    if (nc != nbSize_) {
      switch (isa) {
        case inst_set_t::avx512_vnni:
          if constexpr (std::is_same_v<
                            typename packingAMatrix::accType,
                            std::int16_t>) {
            // For AVX512VNNI, we redirect int16_t to int32_t accumulation.
            CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
            fn = codeObj.getOrCreate<inst_set_t::avx512_vnni>(
                accum, packed_rows_A, nc, packedA_.numPackedCols());
          } else {
            fn = BaseType::template getOrCreate<inst_set_t::avx512_vnni>(
                accum, packed_rows_A, nc, packedA_.numPackedCols());
          }
          break;

        case inst_set_t::avx512_vnni_ymm:
          if constexpr (std::is_same_v<
                            typename packingAMatrix::accType,
                            std::int16_t>) {
            // For AVX512VNNI, we redirect int16_t to int32_t accumulation.
            CodeGenBase<uint8_t, int8_t, int32_t, int32_t> codeObj;
            fn = codeObj.getOrCreate<inst_set_t::avx512_vnni_ymm>(
                accum, packed_rows_A, nc, packedA_.numPackedCols());
          } else {
            fn = BaseType::template getOrCreate<inst_set_t::avx512_vnni_ymm>(
                accum, packed_rows_A, nc, packedA_.numPackedCols());
          }
          break;

        case inst_set_t::avx512:
          fn = BaseType::template getOrCreate<inst_set_t::avx512>(
              accum, packed_rows_A, nc, packedA_.numPackedCols());
          break;

        case inst_set_t::avx512_ymm:
          fn = BaseType::template getOrCreate<inst_set_t::avx512_ymm>(
              accum, packed_rows_A, nc, packedA_.numPackedCols());
          break;

        case inst_set_t::avx2:
          fn = BaseType::template getOrCreate<inst_set_t::avx2>(
              accum, packed_rows_A, nc, packedA_.numPackedCols());
          break;

        default:
          // TODO: Have default slower path
          assert(0 && "unsupported architecture");
          throw std::runtime_error("unsupported architecure");
      }
    }
#endif // __aarch64__

    bBuf = packedB_.getBuf(jb, kBlock);
    // prefetch addr of the next packed block of B matrix
    bBuf_pf = packedB_.getBuf(jb == bColBlocks - 1 ? jb : jb + 1, kBlock);

    // If the accumulation buffer C_buffer_ is the same as matC_ (inplace output
    // processing), then each thread use the different parts of output buffer
    // matC_;
    // Otherwise, each thread uses different portions of the accumulation
    // buffer C_buffer_. If m is large enough (m >= m_nthreads * MC), then we
    // only need to use (m_nthreads * MC) x n portion of C_buffer_, each thread
    // access the C_buffer_row_start as tid * MC * ldc_; else when m is very
    // small, we juse use the whole m x n C_buffer_: each thread use the
    // different portion.
    int32_t* C_buffer_row_start = C_buffer_ +
        ((C_buffer_ == reinterpret_cast<int32_t*>(matC_) ||
          th_info_.m_num_threads * mbSize_ > packedA_.numRows())
             ? row_start_A * ldc_ + NDim * group
             : th_info_.m_thread_id * mbSize_ * ldc_ + NDim * group);

    int32_t* C_buffer_start = C_buffer_row_start + jb * nbSize_;
    int32_t leadingDim = ldc_;
    static thread_local std::vector<int32_t> C_tile_;
    if (packedB_.isThereColRemainder() && (jb == bColBlocks - 1)) {
      // In case we will access memory past C_buffer_, we use C_tile_ scratchpad
      // instead.
      C_tile_.resize(mbSize_ * nbSize_);
      C_buffer_start = C_tile_.data();
      leadingDim = nbSize_;
    }

#ifdef __aarch64__
    // No JIT micro-kernel exists for arm; compute this block with the portable
    // reference kernel. `fn` bakes mc/nc/kc into the JIT'd code, so the
    // reference kernel takes them as arguments instead.
    (void)bBuf_pf; // prefetch hint is unused on the reference path
    computeBlockRef<typename packingAMatrix::accType>(
        aBuf,
        bBuf,
        C_buffer_start,
        packed_rows_A, // mc
        nc,
        packedA_.numPackedCols(), // kc
        packedA_.blockColSize(), // kBlock == A row stride (KCB)
        packedB_.blockColSize(), // nBlock == NCB
        accum,
        leadingDim);
#else
    fn(aBuf,
       bBuf,
       bBuf_pf,
       C_buffer_start,
       packedA_.numPackedCols(),
       leadingDim);
#endif

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
    t_end = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
             .count();
    kernel_time += (dt);
    t_start = std::chrono::high_resolution_clock::now();
#endif

    // Output processing is done only once per rowblock to amortize overhead
    // and for better spatial locality.
    if (lastKBlock && jb == jb_end - 1) {
      // When C_tile_ is used for the last column block, we need a separate
      // handling for the last column block.
      int32_t nSize =
          (C_buffer_start == C_tile_.data() ? (jb - jb_begin) * nbSize_
                                            : (jb_end - jb_begin) * nbSize_);
      if (nSize) {
        if (hasOutputProcessingKernel) {
          // TODO: avx512 path
          // Currently use avx2 code
          outputProcess_.template f<inst_set_t::avx2>(
              matC_,
              C_buffer_row_start + jb_begin * nbSize_,
              {row_start_A,
               packed_rows_A,
               static_cast<int>(NDim * group + jb_begin * nbSize_),
               nSize},
              ldc_,
              ldc_);
        } else {
          // TODO: Have default slower path
          assert(0 && "unsupported architecure");
          throw std::runtime_error("unsupported architecure");
        }
      }

      if (C_buffer_start == C_tile_.data()) {
        // When C_tile_ scratchpad was used to avoid accessing memory past
        // C_buffer_ .
        if (hasOutputProcessingKernel) {
          // TODO: avx512 path
          // Currently use avx2 code
          outputProcess_.template f<inst_set_t::avx2>(
              matC_,
              C_tile_.data(),
              {row_start_A,
               packed_rows_A,
               NDim * group + jb * nbSize_,
               packedB_.lastBcol()},
              ldc_,
              leadingDim);
        } else {
          // TODO: Have default slower path
          assert(0 && "unsupported architecure");
          throw std::runtime_error("unsupported architecure");
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
#define INSTANTIATE_REQUANT_BASE(PACK_A, ACC_T, RELU, Q_GRAN, BIAS_TYPE) \
  template class ExecuteKernel<                                          \
      PACK_A<uint8_t, ACC_T>,                                            \
      PackBMatrix<int8_t, ACC_T>,                                        \
      uint8_t,                                                           \
      ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>>;

#define INSTANTIATE_REQUANT_BIAS_T(PACK_A, ACC_T, RELU, Q_GRAN) \
  INSTANTIATE_REQUANT_BASE(PACK_A, ACC_T, RELU, Q_GRAN, float)  \
  INSTANTIATE_REQUANT_BASE(PACK_A, ACC_T, RELU, Q_GRAN, int32_t)

#define INSTANTIATE_REQUANT_Q_GRANS(PACK_A, ACC_T, RELU)    \
  INSTANTIATE_REQUANT_BIAS_T(                               \
      PACK_A, ACC_T, RELU, QuantizationGranularity::TENSOR) \
  INSTANTIATE_REQUANT_BIAS_T(                               \
      PACK_A, ACC_T, RELU, QuantizationGranularity::GROUP)  \
  INSTANTIATE_REQUANT_BIAS_T(                               \
      PACK_A, ACC_T, RELU, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_REQUANT_RELU(PACK_A, ACC_T)     \
  INSTANTIATE_REQUANT_Q_GRANS(PACK_A, ACC_T, false) \
  INSTANTIATE_REQUANT_Q_GRANS(PACK_A, ACC_T, true)

#define INSTANTIATE_REQUANT_ACC_T(PACK_A)   \
  INSTANTIATE_REQUANT_RELU(PACK_A, int32_t) \
  INSTANTIATE_REQUANT_RELU(PACK_A, int16_t)

INSTANTIATE_REQUANT_ACC_T(PackAMatrix)
INSTANTIATE_REQUANT_ACC_T(PackAWithRowOffset)

#undef INSTANTIATE_REQUANT_ACC_T
#undef INSTANTIATE_REQUANT_RELU
#undef INSTANTIATE_REQUANT_Q_GRANS
#undef INSTANTIATE_REQUANT_BIAS_T
#undef INSTANTIATE_REQUANT_BASE

#define INSTANTIATE_IM2COL_REQUANT_BASE(            \
    ACC_T, RELU, SPATIAL_DIM, Q_GRAN, BIAS_TYPE)    \
  template class ExecuteKernel<                     \
      PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>, \
      PackBMatrix<int8_t, ACC_T>,                   \
      uint8_t,                                      \
      ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>>;

#define INSTANTIATE_IM2COL_REQUANT_BIAS_T(ACC_T, RELU, SPATIAL_DIM, Q_GRAN) \
  INSTANTIATE_IM2COL_REQUANT_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN, float)  \
  INSTANTIATE_IM2COL_REQUANT_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN, int32_t)

#define INSTANTIATE_IM2COL_REQUANT_Q_GRANS(ACC_T, RELU, SPATIAL_DIM) \
  INSTANTIATE_IM2COL_REQUANT_BIAS_T(                                 \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::TENSOR)     \
  INSTANTIATE_IM2COL_REQUANT_BIAS_T(                                 \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::GROUP)      \
  INSTANTIATE_IM2COL_REQUANT_BIAS_T(                                 \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_IM2COL_REQUANT_SPATIAL_DIM(ACC_T, RELU) \
  INSTANTIATE_IM2COL_REQUANT_Q_GRANS(ACC_T, RELU, 1)        \
  INSTANTIATE_IM2COL_REQUANT_Q_GRANS(ACC_T, RELU, 2)        \
  INSTANTIATE_IM2COL_REQUANT_Q_GRANS(ACC_T, RELU, 3)

#define INSTANTIATE_IM2COL_REQUANT_RELU(ACC_T)         \
  INSTANTIATE_IM2COL_REQUANT_SPATIAL_DIM(ACC_T, false) \
  INSTANTIATE_IM2COL_REQUANT_SPATIAL_DIM(ACC_T, true)

INSTANTIATE_IM2COL_REQUANT_RELU(int32_t)
INSTANTIATE_IM2COL_REQUANT_RELU(int16_t)

#undef INSTANTIATE_IM2COL_REQUANT_RELU
#undef INSTANTIATE_IM2COL_REQUANT_SPATIAL_DIM
#undef INSTANTIATE_IM2COL_REQUANT_Q_GRANS
#undef INSTANTIATE_IM2COL_REQUANT_BIAS_T
#undef INSTANTIATE_IM2COL_REQUANT_BASE

////////////////////////////////////////////////////////////////////////////////
// ReQuantizeForFloat
#define INSTANTIATE_REQUANT_FLOAT_BASE(PACK_A, RELU, Q_GRAN) \
  template class ExecuteKernel<                              \
      PACK_A<uint8_t, int32_t>,                              \
      PackBMatrix<int8_t, int32_t>,                          \
      float,                                                 \
      ReQuantizeForFloat<RELU, Q_GRAN>>;

#define INSTANTIATE_REQUANT_FLOAT_Q_GRANS(PACK_A, RELU)                        \
  INSTANTIATE_REQUANT_FLOAT_BASE(                                              \
      PACK_A, RELU, QuantizationGranularity::TENSOR)                           \
  INSTANTIATE_REQUANT_FLOAT_BASE(PACK_A, RELU, QuantizationGranularity::GROUP) \
  INSTANTIATE_REQUANT_FLOAT_BASE(                                              \
      PACK_A, RELU, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_REQUANT_FLOAT_RELU(PACK_A)     \
  INSTANTIATE_REQUANT_FLOAT_Q_GRANS(PACK_A, false) \
  INSTANTIATE_REQUANT_FLOAT_Q_GRANS(PACK_A, true)

INSTANTIATE_REQUANT_FLOAT_RELU(PackAWithRowOffset)
INSTANTIATE_REQUANT_FLOAT_RELU(PackAWithQuantRowOffset)

#undef INSTANTIATE_REQUANT_FLOAT_RELU
#undef INSTANTIATE_REQUANT_FLOAT_Q_GRANS
#undef INSTANTIATE_REQUANT_FLOAT_BASE

#define INSTANTIATE_REQUANT_FLOAT_IM2COL_BASE(      \
    ACC_T, RELU, SPATIAL_DIM, Q_GRAN)               \
  template class ExecuteKernel<                     \
      PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>, \
      PackBMatrix<int8_t, ACC_T>,                   \
      float,                                        \
      ReQuantizeForFloat<RELU, Q_GRAN>>;

#define INSTANTIATE_REQUANT_FLOAT_IM2COL_Q_GRANS(ACC_T, RELU, SPATIAL_DIM) \
  INSTANTIATE_REQUANT_FLOAT_IM2COL_BASE(                                   \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::TENSOR)           \
  INSTANTIATE_REQUANT_FLOAT_IM2COL_BASE(                                   \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::GROUP)            \
  INSTANTIATE_REQUANT_FLOAT_IM2COL_BASE(                                   \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_REQUANT_FLOAT_IM2COL_SPATIAL_DIM(ACC_T, RELU) \
  INSTANTIATE_REQUANT_FLOAT_IM2COL_Q_GRANS(ACC_T, RELU, 1)        \
  INSTANTIATE_REQUANT_FLOAT_IM2COL_Q_GRANS(ACC_T, RELU, 2)        \
  INSTANTIATE_REQUANT_FLOAT_IM2COL_Q_GRANS(ACC_T, RELU, 3)

#define INSTANTIATE_REQUANT_FLOAT_IM2COL_RELU(ACC_T)         \
  INSTANTIATE_REQUANT_FLOAT_IM2COL_SPATIAL_DIM(ACC_T, false) \
  INSTANTIATE_REQUANT_FLOAT_IM2COL_SPATIAL_DIM(ACC_T, true)

INSTANTIATE_REQUANT_FLOAT_IM2COL_RELU(int32_t)
INSTANTIATE_REQUANT_FLOAT_IM2COL_RELU(int16_t)

#undef INSTANTIATE_REQUANT_FLOAT_IM2COL_RELU
#undef INSTANTIATE_REQUANT_FLOAT_IM2COL_SPATIAL_DIM
#undef INSTANTIATE_REQUANT_FLOAT_IM2COL_Q_GRANS
#undef INSTANTIATE_REQUANT_FLOAT_IM2COL_BASE

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    float,
    ReQuantizeForFloat<false /* FUSE_RELU*/>>;

////////////////////////////////////////////////////////////////////////////////
// DoSpmdmOnInpBuffer
#define INSTANTIATE_SPMDM_BASE(PACK_A, RELU, Q_GRAN) \
  template class ExecuteKernel<                      \
      PACK_A<uint8_t, int16_t>,                      \
      PackBMatrix<int8_t, int16_t>,                  \
      uint8_t,                                       \
      DoSpmdmOnInpBuffer<uint8_t, int32_t, ReQuantizeOutput<RELU, Q_GRAN>>>;

#define INSTANTIATE_SPMDM_Q_GRANS(PACK_A, RELU)                         \
  INSTANTIATE_SPMDM_BASE(PACK_A, RELU, QuantizationGranularity::TENSOR) \
  INSTANTIATE_SPMDM_BASE(PACK_A, RELU, QuantizationGranularity::GROUP)  \
  INSTANTIATE_SPMDM_BASE(PACK_A, RELU, QuantizationGranularity::OUT_CHANNEL)

#define INSTANTIATE_SPMDM_RELU(PACK_A)     \
  INSTANTIATE_SPMDM_Q_GRANS(PACK_A, false) \
  INSTANTIATE_SPMDM_Q_GRANS(PACK_A, true)

INSTANTIATE_SPMDM_RELU(PackAMatrix)
INSTANTIATE_SPMDM_RELU(PackAWithRowOffset)

#undef INSTANTIATE_SPMDM_RELU
#undef INSTANTIATE_SPMDM_Q_GRANS
#undef INSTANTIATE_SPMDM_BASE

#define INSTANTIATE_SCONV_BASE(RELU, Q_GRAN) \
  template class ExecuteKernel<              \
      PackAWithIm2Col<uint8_t, int16_t>,     \
      PackBMatrix<int8_t, int16_t>,          \
      uint8_t,                               \
      DoSConvOnInpBuffer<uint8_t, int32_t, ReQuantizeOutput<RELU, Q_GRAN>>>;

#define INSTANTIATE_SCONV_Q_GRANS(RELU)                         \
  INSTANTIATE_SCONV_BASE(RELU, QuantizationGranularity::TENSOR) \
  INSTANTIATE_SCONV_BASE(RELU, QuantizationGranularity::GROUP)  \
  INSTANTIATE_SCONV_BASE(RELU, QuantizationGranularity::OUT_CHANNEL)

INSTANTIATE_SCONV_Q_GRANS(false)
INSTANTIATE_SCONV_Q_GRANS(true)

#undef INSTANTIATE_SCONV_Q_GRANS
#undef INSTANTIATE_SCONV_BASE

template class ExecuteKernel<
    PackAWithRowOffset<uint8_t, int16_t>,
    PackBMatrix<int8_t, int16_t>,
    float,
    DoSpmdmOnInpBuffer<float, int32_t, ReQuantizeForFloat<false>>>;

////////////////////////////////////////////////////////////////////////////////
// memCopy
#define INSTANTIATE_MEMCPY_BASE(PACK_A, ACC_T) \
  template class ExecuteKernel<                \
      PACK_A<uint8_t, ACC_T>,                  \
      PackBMatrix<int8_t, ACC_T>,              \
      int32_t,                                 \
      memCopy<>>;

#define INSTANTIATE_MEMCPY_ACC_T(PACK_A)   \
  INSTANTIATE_MEMCPY_BASE(PACK_A, int32_t) \
  INSTANTIATE_MEMCPY_BASE(PACK_A, int16_t)

INSTANTIATE_MEMCPY_ACC_T(PackAMatrix)
INSTANTIATE_MEMCPY_ACC_T(PackAWithRowOffset)

#undef INSTANTIATE_MEMCPY_ACC_T
#undef INSTANTIATE_MEMCPY_BASE

#define INSTANTIATE_MEMCPY_IM2COL_BASE(ACC_T, SPATIAL_DIM) \
  template class ExecuteKernel<                            \
      PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,        \
      PackBMatrix<int8_t, ACC_T>,                          \
      int32_t,                                             \
      memCopy<>>;

#define INSTANTIATE_MEMCPY_IM2COL_SPATIAL_DIM(ACC_T) \
  INSTANTIATE_MEMCPY_IM2COL_BASE(ACC_T, 1)           \
  INSTANTIATE_MEMCPY_IM2COL_BASE(ACC_T, 2)           \
  INSTANTIATE_MEMCPY_IM2COL_BASE(ACC_T, 3)

INSTANTIATE_MEMCPY_IM2COL_SPATIAL_DIM(int32_t)
INSTANTIATE_MEMCPY_IM2COL_SPATIAL_DIM(int16_t)

#undef INSTANTIATE_MEMCPY_IM2COL_SPATIAL_DIM
#undef INSTANTIATE_MEMCPY_IM2COL_BASE

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
