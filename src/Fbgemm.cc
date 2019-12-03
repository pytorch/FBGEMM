/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "fbgemm/Fbgemm.h"
#include <cpuinfo.h>
#include <stdexcept>
#include "ExecuteKernel.h"

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double packing_time = 0.0;
double computing_time = 0.0;
double run_time = 0.0;
#endif

namespace fbgemm {

template <
    typename packingAMatrix,
    typename packingBMatrix,
    typename cT,
    typename processOutputType>
void fbgemmPacked(
    PackMatrix<
        packingAMatrix,
        typename packingAMatrix::inpType,
        typename packingAMatrix::accType>& packA,
    PackMatrix<
        packingBMatrix,
        typename packingBMatrix::inpType,
        typename packingBMatrix::accType>& packB,
    cT* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const processOutputType& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params) {
  static_assert(
      std::is_same<
          typename packingAMatrix::accType,
          typename packingBMatrix::accType>::value,
      "Accumulation type of both matrices should be the same");

  // Run time CPU detection
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if ((!fbgemmHasAvx512VnniSupport() && !fbgemmHasAvx512Support() &&
       !fbgemmHasAvx2Support())) {
    assert(0 && "unknown architecure");
  }

  int MCB;
  int KCB;
  int MR;

  if (blocking_params) {
    MCB = blocking_params->MCB;
    KCB = blocking_params->KCB;
    MR = blocking_params->MR;

  } else {
    if (fbgemmHasAvx512VnniSupport()) {
      MCB = PackingTraits<
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType,
          inst_set_t::avx512_vnni>::MCB;
      KCB = PackingTraits<
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType,
          inst_set_t::avx512_vnni>::KCB;
      MR = PackingTraits<
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType,
          inst_set_t::avx512_vnni>::MR;
    } else if (fbgemmHasAvx512Support()) {
      MCB = PackingTraits<
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType,
          inst_set_t::avx512>::MCB;
      KCB = PackingTraits<
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType,
          inst_set_t::avx512>::KCB;
      MR = PackingTraits<
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType,
          inst_set_t::avx512>::MR;
    } else {
      MCB = PackingTraits<
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType,
          inst_set_t::avx2>::MCB;
      KCB = PackingTraits<
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType,
          inst_set_t::avx2>::KCB;
      MR = PackingTraits<
          typename packingAMatrix::inpType,
          typename packingAMatrix::accType,
          inst_set_t::avx2>::MR;
    }
  }

  if (!packB.isPrePacked()) {
    throw std::runtime_error("B matrix must be prepacked");
  }
  int G = packA.numGroups();
  if (G != packB.numGroups()) {
    throw std::runtime_error(
        "A.groups = " + std::to_string(G) + " and B.groups = " +
        std::to_string(packB.numGroups()) + " are not the same");
  }

  int MDim = packA.numRows();
  int KDimPerGroup = packB.numRows() / G;

  int kBlocks = (KDimPerGroup + KCB - 1) / KCB;

  // remainders
  int _kc = KDimPerGroup % KCB;

  int kc, mc;

  block_type_t blockA{0, 0, 0, 0};

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_very_start,
      t_start, t_end;
  double dt;
  t_start = std::chrono::high_resolution_clock::now();
  t_very_start = std::chrono::high_resolution_clock::now();
#endif

  int g_begin, g_end, i_begin, i_end;
  if (G >= num_threads) {
    // When G >= nthreads, just parallelize over G
    // TODO: when G == nthreads + 1, we'll have a big load imbalance because
    // only one thread will get 2 groups.
    fbgemmGetRange(num_threads, thread_id, G, 1, g_begin, g_end);
    i_begin = 0;
    i_end = MDim;
  } else {
    // Otherwise, each group is parallelized by multiple threads.
    // nthreads_per_group is floor(nthreads / G).
    // If we use ceil, some groups won't be handled by any thread.
    int nthreads_per_group = num_threads / G;
    g_begin = std::max(std::min(thread_id / nthreads_per_group, G - 1), 0);
    g_end = std::min(g_begin + 1, G);

    int tid_of_g_begin = std::min(g_begin * nthreads_per_group, num_threads);
    int tid_of_g_end = std::min(
        (g_end == G) ? num_threads : (tid_of_g_begin + nthreads_per_group),
        num_threads);
    int nthreads_within_group = tid_of_g_end - tid_of_g_begin;
    int tid_within_group = thread_id - tid_of_g_begin;
    fbgemmGetRange(
        nthreads_within_group, tid_within_group, MDim, MR, i_begin, i_end);
  }

  for (int g = g_begin; g < g_end; ++g) {
    ExecuteKernel<packingAMatrix, packingBMatrix, cT, processOutputType>
        exeKernelObj(
            packA,
            packB,
            C,
            C_buffer,
            ldc,
            outProcess,
            thread_id,
            num_threads,
            blocking_params);
    for (int i = i_begin; i < i_end; i += MCB) { // i is the element index
      mc = std::min(i_end - i, MCB);
      for (int kb = 0; kb < kBlocks; ++kb) { // kb is the block index
        kc = (kb != kBlocks - 1 || _kc == 0) ? KCB : _kc;
        // pack A matrix
        blockA = {i, mc, g * KDimPerGroup + kb * KCB, kc};
        packA.pack(blockA);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        t_end = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 t_end - t_start)
                 .count();
        packing_time += (dt);
        t_start = std::chrono::high_resolution_clock::now();
#endif

        exeKernelObj.execute(g * kBlocks + kb);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
        t_end = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 t_end - t_start)
                 .count();
        computing_time += (dt);
        t_start = std::chrono::high_resolution_clock::now();
#endif
      }
    }
  } // for each group

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_very_start)
          .count();
  run_time += (dt);
  t_start = std::chrono::high_resolution_clock::now();
#endif
}

template <int SPATIAL_DIM>
FBGEMM_API bool fbgemmOptimizedGConv(const conv_param_t<SPATIAL_DIM>& conv_p) {
  int C_per_G = conv_p.IC / conv_p.G;
  int K_per_G = conv_p.OC / conv_p.G;

  return (SPATIAL_DIM == 2) && (C_per_G == K_per_G) &&
      (C_per_G == 4 || C_per_G == 8 || C_per_G == 16) && (conv_p.G % 8 == 0) &&
      (conv_p.K[0] == conv_p.K[1]) && (conv_p.K[0] == 3) &&
      (conv_p.pad[0] == 1) && (conv_p.pad[1] == 1) &&
      (conv_p.pad[0] == conv_p.pad[2]) && (conv_p.pad[1] == conv_p.pad[3]) &&
      (conv_p.dilation[0] == 1) && (conv_p.dilation[0] == conv_p.dilation[1]) &&
      (conv_p.stride[0] == 1) && (conv_p.stride[0] == conv_p.stride[1]);
}

template FBGEMM_API bool fbgemmOptimizedGConv(const conv_param_t<2>& conv_p);
template FBGEMM_API bool fbgemmOptimizedGConv(const conv_param_t<3>& conv_p);

bool fbgemmSupportedCPU() {
  return (cpuinfo_initialize() && fbgemmHasAvx2Support());
}

////////////////////////////////////////////////////////////////////////////////
// ReQuantizeOutput
#define INSTANTIATE_BASE(PACK_A, ACC_T, RELU, Q_GRAN, BIAS_TYPE)    \
  template void fbgemmPacked(                                       \
      PackMatrix<PACK_A<uint8_t, ACC_T>, uint8_t, ACC_T>& packA,    \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB, \
      uint8_t* C,                                                   \
      int32_t* C_buffer,                                            \
      uint32_t ldc,                                                 \
      const ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>& outProcess,  \
      int thread_id,                                                \
      int num_threads,                                              \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_BIAS_T(PACK_A, ACC_T, RELU, Q_GRAN) \
  INSTANTIATE_BASE(PACK_A, ACC_T, RELU, Q_GRAN, float); \
  INSTANTIATE_BASE(PACK_A, ACC_T, RELU, Q_GRAN, int32_t);

#define INSTANTIATE_Q_GRANS(PACK_A, ACC_T, RELU)                            \
  INSTANTIATE_BIAS_T(PACK_A, ACC_T, RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BIAS_T(PACK_A, ACC_T, RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BIAS_T(PACK_A, ACC_T, RELU, QuantizationGranularity::OUT_CHANNEL);

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
#undef INSTANTIATE_BIAS_T
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN, BIAS_TYPE) \
  template void fbgemmPacked(                                         \
      PackMatrix<                                                     \
          PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,               \
          uint8_t,                                                    \
          ACC_T>& packA,                                              \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB,   \
      uint8_t* C,                                                     \
      int32_t* C_buffer,                                              \
      uint32_t ldc,                                                   \
      const ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>& outProcess,    \
      int thread_id,                                                  \
      int num_threads,                                                \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_BIAS_T(ACC_T, RELU, SPATIAL_DIM, Q_GRAN) \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN, float); \
  INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN, int32_t);

#define INSTANTIATE_Q_GRANS(ACC_T, RELU, SPATIAL_DIM)             \
  INSTANTIATE_BIAS_T(                                             \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BIAS_T(                                             \
      ACC_T, RELU, SPATIAL_DIM, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BIAS_T(                                             \
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
#undef INSTANTIATE_BIAS_T
#undef INSTANTIATE_BASE

////////////////////////////////////////////////////////////////////////////////
// ReQuantizeForFloat
#define INSTANTIATE_BASE(PACK_A, RELU, Q_GRAN)                          \
  template void fbgemmPacked(                                           \
      PackMatrix<PACK_A<uint8_t, int32_t>, uint8_t, int32_t>& packA,    \
      PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB, \
      float* C,                                                         \
      int32_t* C_buffer,                                                \
      uint32_t ldc,                                                     \
      const ReQuantizeForFloat<RELU, Q_GRAN>& outProcess,               \
      int thread_id,                                                    \
      int num_threads,                                                  \
      const BlockingFactors* blocking_params);

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

#define INSTANTIATE_BASE(ACC_T, RELU, SPATIAL_DIM, Q_GRAN)          \
  template void fbgemmPacked(                                       \
      PackMatrix<                                                   \
          PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,             \
          uint8_t,                                                  \
          ACC_T>& packA,                                            \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB, \
      float* C,                                                     \
      int32_t* C_buffer,                                            \
      uint32_t ldc,                                                 \
      const ReQuantizeForFloat<RELU, Q_GRAN>& outProcess,           \
      int thread_id,                                                \
      int num_threads,                                              \
      const BlockingFactors* blocking_params);

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

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeForFloat<false>& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params);

////////////////////////////////////////////////////////////////////////////////
// DoSpmdmOnInpBuffer
#define INSTANTIATE_BASE(PACK_A, RELU, Q_GRAN)                          \
  template void fbgemmPacked(                                           \
      PackMatrix<PACK_A<uint8_t, int16_t>, uint8_t, int16_t>& packA,    \
      PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB, \
      uint8_t* C,                                                       \
      int32_t* C_buffer,                                                \
      uint32_t ldc,                                                     \
      const DoSpmdmOnInpBuffer<                                         \
          uint8_t,                                                      \
          int32_t,                                                      \
          ReQuantizeOutput<RELU, Q_GRAN>>& outProcess,                  \
      int thread_id,                                                    \
      int num_threads,                                                  \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_Q_GRANS(PACK_A, RELU)                          \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BASE(PACK_A, RELU, QuantizationGranularity::OUT_CHANNEL);

#define INSTANTIATE_RELU(PACK_A)      \
  INSTANTIATE_Q_GRANS(PACK_A, false); \
  INSTANTIATE_Q_GRANS(PACK_A, true);

INSTANTIATE_RELU(PackAMatrix);
INSTANTIATE_RELU(PackAWithRowOffset);

#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE
#undef INSTANTIATE_RELU

#define INSTANTIATE_BASE(RELU, Q_GRAN)                                        \
  template void fbgemmPacked(                                                 \
      PackMatrix<PackAWithIm2Col<uint8_t, int16_t>, uint8_t, int16_t>& packA, \
      PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,       \
      uint8_t* C,                                                             \
      int32_t* C_buffer,                                                      \
      uint32_t ldc,                                                           \
      const DoSConvOnInpBuffer<                                               \
          uint8_t,                                                            \
          int32_t,                                                            \
          ReQuantizeOutput<RELU, Q_GRAN>>& outProcess,                        \
      int thread_id,                                                          \
      int num_threads,                                                        \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_Q_GRANS(RELU)                          \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BASE(RELU, QuantizationGranularity::OUT_CHANNEL);

INSTANTIATE_Q_GRANS(false);
INSTANTIATE_Q_GRANS(true);

#undef INSTANTIATE_Q_GRANS
#undef INSTANTIATE_BASE

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const DoSpmdmOnInpBuffer<float, int32_t, ReQuantizeForFloat<false>>&
        outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params);

////////////////////////////////////////////////////////////////////////////////
// memCopy
#define INSTANTIATE_BASE(PACK_A, ACC_T)                             \
  template void fbgemmPacked(                                       \
      PackMatrix<PACK_A<uint8_t, ACC_T>, uint8_t, ACC_T>& packA,    \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB, \
      int32_t* C,                                                   \
      int32_t* C_buffer,                                            \
      uint32_t ldc,                                                 \
      const memCopy<>& outProcess,                                  \
      int thread_id,                                                \
      int num_threads,                                              \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_ACC_T(PACK_A)   \
  INSTANTIATE_BASE(PACK_A, int32_t) \
  INSTANTIATE_BASE(PACK_A, int16_t)

INSTANTIATE_ACC_T(PackAMatrix);
INSTANTIATE_ACC_T(PackAWithRowOffset);

#undef INSTANTIATE_ACC_T
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(ACC_T, SPATIAL_DIM)                        \
  template void fbgemmPacked(                                       \
      PackMatrix<                                                   \
          PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,             \
          uint8_t,                                                  \
          ACC_T>& packA,                                            \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB, \
      int32_t* C,                                                   \
      int32_t* C_buffer,                                            \
      uint32_t ldc,                                                 \
      const memCopy<>& outProcess,                                  \
      int thread_id,                                                \
      int num_threads,                                              \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_SPATIAL_DIM(ACC_T) \
  INSTANTIATE_BASE(ACC_T, 2);          \
  INSTANTIATE_BASE(ACC_T, 3);

INSTANTIATE_SPATIAL_DIM(int32_t);
INSTANTIATE_SPATIAL_DIM(int16_t);

#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_BASE

template void fbgemmPacked(
    PackMatrix<PackAWithQuantRowOffset<uint8_t, int32_t>, uint8_t, int32_t>&
        packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params);

template void fbgemmPacked(
    PackMatrix<PackAMatrix<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const DoNothing<int32_t, int32_t>& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params);

} // namespace fbgemm
