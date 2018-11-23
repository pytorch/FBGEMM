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

using namespace fbgemm;

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
    int num_threads) {
  static_assert(
      std::is_same<
          typename packingAMatrix::accType,
          typename packingBMatrix::accType>::value,
      "Accumulation type of both matrices should be the same");

  int MCB, KCB;
  int MR;

  // Run time CPU detection
  if (cpuinfo_initialize()) {
    if (cpuinfo_has_x86_avx512f()) {
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
    } else if (cpuinfo_has_x86_avx2()) {
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
    } else {
      // TODO: Have default slower path
      assert(0 && "unsupported architecture");
      return;
    }
  } else {
    throw std::runtime_error("Failed to initialize cpuinfo!");
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
            0,
            C,
            C_buffer,
            ldc,
            outProcess,
            thread_id,
            num_threads);
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

bool fbgemmSupportedCPU() {
  return (cpuinfo_initialize() && cpuinfo_has_x86_avx2());
}

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int32_t>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeOutput<false>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int32_t>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeOutput<true>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithQuantRowOffset<uint8_t, int32_t>, uint8_t, int32_t>&
        packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeForFloat<false>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithQuantRowOffset<uint8_t, int32_t>, uint8_t, int32_t>&
        packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeForFloat<true>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAMatrix<uint8_t, int32_t>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int32_t>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeForFloat<false>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int32_t>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeForFloat<true>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int32_t>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithIm2Col<uint8_t, int32_t>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithIm2Col<uint8_t, int32_t, 3>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithIm2Col<uint8_t, int32_t>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeOutput<false>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithIm2Col<uint8_t, int32_t, 3>, uint8_t, int32_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeOutput<false>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithQuantRowOffset<uint8_t, int32_t>, uint8_t, int32_t>&
        packA,
    PackMatrix<PackBMatrix<int8_t, int32_t>, int8_t, int32_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads);

// 16 bit accumulation functions
template void fbgemmPacked(
    PackMatrix<PackAMatrix<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAMatrix<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeOutput<false>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const DoSpmdmOnInpBuffer<uint8_t, int32_t, ReQuantizeOutput<false>>&
        outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const DoSpmdmOnInpBuffer<uint8_t, int32_t, ReQuantizeOutput<true>>&
        outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const DoSpmdmOnInpBuffer<float, int32_t, ReQuantizeForFloat<false>>&
        outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeOutput<false>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeOutput<true>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithIm2Col<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithIm2Col<uint8_t, int16_t, 3>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const memCopy<>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithIm2Col<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeOutput<false>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithIm2Col<uint8_t, int16_t, 3>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    uint8_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeOutput<false>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAMatrix<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const DoNothing<int32_t, int32_t>& outProcess,
    int thread_id,
    int num_threads);

template void fbgemmPacked(
    PackMatrix<PackAWithRowOffset<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    float* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const ReQuantizeForFloat<false>& outProcess,
    int thread_id,
    int num_threads);

} // namespace fbgemm
