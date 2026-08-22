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
#include <cstring>
#include <type_traits>
#include <vector>
#include "fbgemm/Utils.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

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

#if FEAT_I8MM

// Loads the 8 K values one USMMLA operand row consumes from a packed A row.
// `full` is false for a trailing k-quad, where only 4 K values are left in
// the block; zeroing A's upper half neutralises B's correspondingly unread
// half.
uint8x8_t loadPackedAOctet(const uint8_t* aRow, bool full) {
  uint32x2_t v =
      vld1_lane_u32(reinterpret_cast<const uint32_t*>(aRow), vdup_n_u32(0), 0);
  if (full) {
    v = vld1_lane_u32(reinterpret_cast<const uint32_t*>(aRow + 4), v, 1);
  }
  return vreinterpret_u8_u32(v);
}

// Extracts one output row from the two USMMLA 2x2 tiles that cover four
// consecutive columns. Each tile is [row0col0, row0col1, row1col0, row1col1].
int32x4_t usmmlaTileRow(int32x4_t tile01, int32x4_t tile23, bool secondRow) {
  const int64x2_t lo = vreinterpretq_s64_s32(tile01);
  const int64x2_t hi = vreinterpretq_s64_s32(tile23);
  return vreinterpretq_s32_s64(
      secondRow ? vuzp2q_s64(lo, hi) : vuzp1q_s64(lo, hi));
}

void storeAcc32(int32_t* cRow, int32x4_t lo, int32x4_t hi, bool accum) {
  if (accum) {
    lo = vaddq_s32(lo, vld1q_s32(cRow));
    hi = vaddq_s32(hi, vld1q_s32(cRow + 4));
  }
  vst1q_s32(cRow, lo);
  vst1q_s32(cRow + 4, hi);
}

// Sign-extends the two int16 accumulators covering eight columns into the
// int32 C buffer, the way the JIT kernel's acc16 store path does.
void storeAcc16(int32_t* cRow, int16x4_t lo, int16x4_t hi, bool accum) {
  storeAcc32(cRow, vmovl_s16(lo), vmovl_s16(hi), accum);
}

// Reads the K values of one packed A pair into the low half of a word, in the
// little-endian byte order the USDOT operand slots expect.
uint32_t loadPackedAPair(const uint8_t* aRow) {
  std::uint16_t pair = 0;
  std::memcpy(&pair, aRow, sizeof(pair));
  return pair;
}

// Reads two consecutive packed A pairs, the low one in the low half.
uint32_t loadPackedAQuad(const uint8_t* aRow) {
  std::uint32_t quad = 0;
  std::memcpy(&quad, aRow, sizeof(quad));
  return quad;
}

// Splats a K pair into the two byte slots of every USDOT four-byte group that
// belong to the group's even column.
uint8x16_t aPairEven(uint32_t pair) {
  return vreinterpretq_u8_u32(vdupq_n_u32(pair & 0xFFFFU));
}

// ... and the two that belong to its odd column.
uint8x16_t aPairOdd(uint32_t pair) {
  return vreinterpretq_u8_u32(vdupq_n_u32(pair << 16));
}

// Accumulates one K pair into the GROUPS eight-column groups of a strip.
// Zeroing the USDOT destination is what keeps each pair's reduction its own:
// letting it accumulate would fuse pairs and skip a clip_16bit.
template <int GROUPS>
void accumulateKPairAcc16(
    int16x8_t (&acc)[GROUPS],
    uint32_t pair,
    const int8_t* bPtr) {
  const uint8x16_t aEven = aPairEven(pair);
  const uint8x16_t aOdd = aPairOdd(pair);
  for (int g = 0; g < GROUPS; ++g) {
    const int8x16_t b = vld1q_s8(bPtr + 16 * g);
    const int32x4_t even = vusdotq_s32(vdupq_n_s32(0), aEven, b);
    const int32x4_t odd = vusdotq_s32(vdupq_n_s32(0), aOdd, b);
    acc[g] = vqaddq_s16(acc[g], vqmovn_high_s32(vqmovn_s32(even), odd));
  }
}

// Computes one strip of 8 * GROUPS columns of a single A row, two K pairs at a
// time. The pairs stay sequential on each accumulator because saturating
// int16 addition does not associate.
template <int GROUPS>
void computeStripAcc16(
    const uint8_t* aRow,
    const int8_t* bBuf,
    int32_t* cRow,
    int n,
    int kc,
    int nBlock,
    bool accum) {
  int16x8_t acc[GROUPS];
  for (int g = 0; g < GROUPS; ++g) {
    acc[g] = vdupq_n_s16(0);
  }

  int k = 0;
  for (; k + 4 <= kc; k += 4) {
    const uint32_t quad = loadPackedAQuad(aRow + k);
    const int8_t* bPtr = bBuf + k * nBlock + 2 * n;
    accumulateKPairAcc16(acc, quad, bPtr);
    accumulateKPairAcc16(acc, quad >> 16, bPtr + 2 * nBlock);
  }
  // kc is a multiple of row_interleave but need not be a multiple of the four
  // K values a step consumes, and A holds nothing past kc.
  if (k < kc) {
    accumulateKPairAcc16(
        acc, loadPackedAPair(aRow + k), bBuf + k * nBlock + 2 * n);
  }

  for (int g = 0; g < GROUPS; ++g) {
    const int16x4_t even = vget_low_s16(acc[g]);
    const int16x4_t odd = vget_high_s16(acc[g]);
    storeAcc16(
        cRow + n + 8 * g, vzip1_s16(even, odd), vzip2_s16(even, odd), accum);
  }
}

// FEAT_I8MM implementation of the u8s8 GEMM micro-kernel, used when the
// compilation target guarantees the Arm Int8 Matrix Multiplication extension.
//
// On x86 the compute kernel is JIT-generated assembly (see
// GenerateKernelU8S8*.cc); no such kernel exists for arm, so we compute the
// mc x nc x kc block directly from the packed A/B buffers, consuming the
// packed layout and reproducing the integer arithmetic of the avx2 kernel:
//   - A is packed row-major with row stride kBlock (== KCB); element A[i][k]
//     lives at aBuf[i * kBlock + k] and is unsigned 8-bit.
//   - B is packed with row_interleave consecutive K values adjacent per
//     column; element B[k][n] lives at
//       bBuf[k * nBlock + n * row_interleave + (k % row_interleave)]
//     (k here is already a multiple of row_interleave) and is signed 8-bit.
//   - acc16 (accT == int16_t, row_interleave == 2): vpmaddubsw multiplies the
//     unsigned A byte by the signed B byte and horizontally adds the pair
//     into an int16 that saturates; the running accumulator is a saturating
//     int16 add (vpaddsw). The int16 result is sign-extended to int32 on
//     store.
//   - acc32 (accT == int32_t, row_interleave == 4): each pair is reduced with
//     the saturating int16 vpmaddubsw, then widened and summed to int32
//     (vpmaddwd) and accumulated without saturation (vpaddd).
// `accum` selects overwrite vs. accumulate into the int32 C buffer, matching
// the JIT store path (used across successive K-blocks).
//
// acc16 uses USDOT (FEAT_I8MM), exact by construction. Interleaving by two
// makes each of USDOT's four-byte groups hold one K pair for each of two
// adjacent columns, so zeroing half of the A operand's byte slots selects one
// column of the pair: the A word [a[k], a[k+1], 0, 0] reduces a 16-byte B
// load to the four even columns' pair sums and [0, 0, a[k], a[k+1]] to the
// four odd ones. Those sums are exact in int32 (|.| <= 65280), so vpmaddubsw
// is USDOT followed by the saturating narrow vqmovn_s32 and vpaddsw is
// vqadd_s16. Even and odd columns stay in the two halves of one accumulator
// for the whole K loop and are interleaved back once at the store.
//
// acc32 uses USMMLA (FEAT_I8MM), which multiplies a 2x8 uint8 A tile by a 2x8
// int8 B tile and accumulates the 2x2 int32 product, so it consumes 8 K
// values and produces two rows x two columns per instruction. That skips the
// intermediate int16 saturation vpmaddubsw applies to each K pair: the result
// therefore matches matmul_u8i8acc32_ref, which accumulates in int32
// throughout, and differs from the avx2 kernel only for a K pair whose
// product sum leaves int16 range.
//
// Columns beyond the last multiple of 8 take the scalar tail.
template <typename accT>
void computeBlockI8mm(
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

  // Each USMMLA covers two A rows, so the acc32 path steps the M loop by two.
  constexpr int row_step = std::is_same_v<accT, std::int16_t> ? 1 : 2;

  for (int i = 0; i < mc; i += row_step) {
    const uint8_t* aRow = aBuf + i * kBlock;
    int32_t* cRow = cBuf + i * ldc;
    int n = 0;

    if constexpr (std::is_same_v<accT, std::int16_t>) {
      for (; n + 32 <= nc; n += 32) {
        computeStripAcc16<4>(aRow, bBuf, cRow, n, kc, nBlock, accum);
      }
      for (; n + 16 <= nc; n += 16) {
        computeStripAcc16<2>(aRow, bBuf, cRow, n, kc, nBlock, accum);
      }
      for (; n + 8 <= nc; n += 8) {
        computeStripAcc16<1>(aRow, bBuf, cRow, n, kc, nBlock, accum);
      }
      for (; n < nc; ++n) {
        std::int16_t acc = 0;
        for (int k = 0; k < kc; k += 2) {
          const int8_t* bPtr = bBuf + k * nBlock + 2 * n;
          const std::int16_t prod =
              satInt16(aRow[k] * bPtr[0] + aRow[k + 1] * bPtr[1]);
          acc = satInt16(acc + prod);
        }
        cRow[n] = accum ? cRow[n] + acc : acc;
      }
    } else {
      static_assert(row_interleave == 4);
      // When mc is odd the second USMMLA row aliases the first; its half of
      // every tile is computed and then dropped.
      const bool hasRow1 = i + 1 < mc;
      const uint8_t* aRow1 = hasRow1 ? aRow + kBlock : aRow;

      for (; n + 8 <= nc; n += 8) {
        // One 2x2 tile per column pair, laid out as
        // [row0col0, row0col1, row1col0, row1col1].
        int32x4_t acc01 = vdupq_n_s32(0);
        int32x4_t acc23 = vdupq_n_s32(0);
        int32x4_t acc45 = vdupq_n_s32(0);
        int32x4_t acc67 = vdupq_n_s32(0);

        // `full` is false only for a trailing k-quad: kc is a multiple of
        // row_interleave but need not be a multiple of the 8 K values one
        // USMMLA consumes, and neither packed buffer holds anything past kc.
        const auto accumulateKOctet = [&](int k, bool full) {
          const uint8x16_t a = vcombine_u8(
              loadPackedAOctet(aRow + k, full),
              loadPackedAOctet(aRow1 + k, full));
          // Each 16-byte load is 4 columns x 4 K values; zipping the k and
          // k+4 quads word-wise builds the 2x8 B tile for a column pair.
          const int8_t* bPtr = bBuf + k * nBlock + 4 * n;
          const uint32x4_t bLo0 = vreinterpretq_u32_s8(vld1q_s8(bPtr));
          const uint32x4_t bLo1 = vreinterpretq_u32_s8(vld1q_s8(bPtr + 16));
          const uint32x4_t bHi0 = full
              ? vreinterpretq_u32_s8(vld1q_s8(bPtr + 4 * nBlock))
              : vdupq_n_u32(0);
          const uint32x4_t bHi1 = full
              ? vreinterpretq_u32_s8(vld1q_s8(bPtr + 4 * nBlock + 16))
              : vdupq_n_u32(0);
          acc01 = vusmmlaq_s32(
              acc01, a, vreinterpretq_s8_u32(vzip1q_u32(bLo0, bHi0)));
          acc23 = vusmmlaq_s32(
              acc23, a, vreinterpretq_s8_u32(vzip2q_u32(bLo0, bHi0)));
          acc45 = vusmmlaq_s32(
              acc45, a, vreinterpretq_s8_u32(vzip1q_u32(bLo1, bHi1)));
          acc67 = vusmmlaq_s32(
              acc67, a, vreinterpretq_s8_u32(vzip2q_u32(bLo1, bHi1)));
        };

        int k = 0;
        for (; k + 8 <= kc; k += 8) {
          accumulateKOctet(k, true);
        }
        if (k < kc) {
          accumulateKOctet(k, false);
        }

        storeAcc32(
            cRow + n,
            usmmlaTileRow(acc01, acc23, false),
            usmmlaTileRow(acc45, acc67, false),
            accum);
        if (hasRow1) {
          storeAcc32(
              cRow + ldc + n,
              usmmlaTileRow(acc01, acc23, true),
              usmmlaTileRow(acc45, acc67, true),
              accum);
        }
      }
      for (; n < nc; ++n) {
        for (int r = 0; r < (hasRow1 ? 2 : 1); ++r) {
          const uint8_t* a = aRow + r * kBlock;
          int32_t* c = cRow + r * ldc;
          std::int32_t acc = 0;
          for (int k = 0; k < kc; k += 4) {
            const int8_t* bPtr = bBuf + k * nBlock + 4 * n;
            acc += a[k] * bPtr[0] + a[k + 1] * bPtr[1] + a[k + 2] * bPtr[2] +
                a[k + 3] * bPtr[3];
          }
          c[n] = accum ? c[n] + acc : acc;
        }
      }
    }
  }
}
#else // !FEAT_I8MM

// NEON implementation of the u8s8 GEMM micro-kernel for aarch64.
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
// NEON mapping, exact by construction: the interleaved layout makes vld2/vld4
// de-interleave a K-pair/quad for 8 columns per load; vpmaddubsw is
// vmull/vmlal into int32 lanes followed by the saturating narrow vqmovn_s32;
// vpaddsw is vqaddq_s16; vpmaddwd+vpaddd is vaddl_s16 into a non-saturating
// vaddq_s32. Columns beyond the last multiple of 8 take the scalar tail.
template <typename accT>
void computeBlockNeon(
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

  for (int i = 0; i < mc; ++i) {
    const uint8_t* aRow = aBuf + i * kBlock;
    int32_t* cRow = cBuf + i * ldc;
    int n = 0;

    if constexpr (std::is_same_v<accT, std::int16_t>) {
      for (; n + 8 <= nc; n += 8) {
        int16x8_t acc = vdupq_n_s16(0);
        for (int k = 0; k < kc; k += 2) {
          const std::int16_t a0 = aRow[k];
          const std::int16_t a1 = aRow[k + 1];
          const int8x8x2_t b = vld2_s8(bBuf + k * nBlock + 2 * n);
          const int16x8_t b0 = vmovl_s8(b.val[0]);
          const int16x8_t b1 = vmovl_s8(b.val[1]);
          int32x4_t lo = vmull_n_s16(vget_low_s16(b0), a0);
          lo = vmlal_n_s16(lo, vget_low_s16(b1), a1);
          int32x4_t hi = vmull_n_s16(vget_high_s16(b0), a0);
          hi = vmlal_n_s16(hi, vget_high_s16(b1), a1);
          const int16x8_t prod = vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
          acc = vqaddq_s16(acc, prod);
        }
        int32x4_t outLo = vmovl_s16(vget_low_s16(acc));
        int32x4_t outHi = vmovl_s16(vget_high_s16(acc));
        if (accum) {
          outLo = vaddq_s32(outLo, vld1q_s32(cRow + n));
          outHi = vaddq_s32(outHi, vld1q_s32(cRow + n + 4));
        }
        vst1q_s32(cRow + n, outLo);
        vst1q_s32(cRow + n + 4, outHi);
      }
      for (; n < nc; ++n) {
        std::int16_t acc = 0;
        for (int k = 0; k < kc; k += 2) {
          const int8_t* bPtr = bBuf + k * nBlock + 2 * n;
          const std::int16_t prod =
              satInt16(aRow[k] * bPtr[0] + aRow[k + 1] * bPtr[1]);
          acc = satInt16(acc + prod);
        }
        cRow[n] = accum ? cRow[n] + acc : acc;
      }
    } else {
      static_assert(row_interleave == 4);
      for (; n + 8 <= nc; n += 8) {
        int32x4_t accLo = vdupq_n_s32(0);
        int32x4_t accHi = vdupq_n_s32(0);
        for (int k = 0; k < kc; k += 4) {
          const std::int16_t a0 = aRow[k];
          const std::int16_t a1 = aRow[k + 1];
          const std::int16_t a2 = aRow[k + 2];
          const std::int16_t a3 = aRow[k + 3];
          const int8x8x4_t b = vld4_s8(bBuf + k * nBlock + 4 * n);
          const int16x8_t b0 = vmovl_s8(b.val[0]);
          const int16x8_t b1 = vmovl_s8(b.val[1]);
          const int16x8_t b2 = vmovl_s8(b.val[2]);
          const int16x8_t b3 = vmovl_s8(b.val[3]);
          int32x4_t lo01 = vmull_n_s16(vget_low_s16(b0), a0);
          lo01 = vmlal_n_s16(lo01, vget_low_s16(b1), a1);
          int32x4_t hi01 = vmull_n_s16(vget_high_s16(b0), a0);
          hi01 = vmlal_n_s16(hi01, vget_high_s16(b1), a1);
          const int16x8_t p01 =
              vcombine_s16(vqmovn_s32(lo01), vqmovn_s32(hi01));
          int32x4_t lo23 = vmull_n_s16(vget_low_s16(b2), a2);
          lo23 = vmlal_n_s16(lo23, vget_low_s16(b3), a3);
          int32x4_t hi23 = vmull_n_s16(vget_high_s16(b2), a2);
          hi23 = vmlal_n_s16(hi23, vget_high_s16(b3), a3);
          const int16x8_t p23 =
              vcombine_s16(vqmovn_s32(lo23), vqmovn_s32(hi23));
          accLo =
              vaddq_s32(accLo, vaddl_s16(vget_low_s16(p01), vget_low_s16(p23)));
          accHi = vaddq_s32(
              accHi, vaddl_s16(vget_high_s16(p01), vget_high_s16(p23)));
        }
        if (accum) {
          accLo = vaddq_s32(accLo, vld1q_s32(cRow + n));
          accHi = vaddq_s32(accHi, vld1q_s32(cRow + n + 4));
        }
        vst1q_s32(cRow + n, accLo);
        vst1q_s32(cRow + n + 4, accHi);
      }
      for (; n < nc; ++n) {
        std::int32_t acc = 0;
        for (int k = 0; k < kc; k += 4) {
          const int8_t* bPtr = bBuf + k * nBlock + 4 * n;
          const std::int16_t p01 =
              satInt16(aRow[k] * bPtr[0] + aRow[k + 1] * bPtr[1]);
          const std::int16_t p23 =
              satInt16(aRow[k + 2] * bPtr[2] + aRow[k + 3] * bPtr[3]);
          acc += p01 + p23;
        }
        cRow[n] = accum ? cRow[n] + acc : acc;
      }
    }
  }
}
#endif // FEAT_I8MM
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
    // No JIT micro-kernel exists for arm; compute this block with the NEON
    // kernel. `fn` bakes mc/nc/kc into the JIT'd code, so the NEON kernel
    // takes them as arguments instead.
    (void)bBuf_pf; // prefetch hint is unused on the NEON path
#if FEAT_I8MM
    computeBlockI8mm<typename packingAMatrix::accType>(
#else
    computeBlockNeon<typename packingAMatrix::accType>(
#endif
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
