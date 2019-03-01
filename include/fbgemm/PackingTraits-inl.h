/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

/*
 * This file configures the important cache blocking parameters and registers
 * blocking parameters for the matrix multiplication loops inside FBGEMM.
 *
 * ROW_INTERLEAVE: the number of interleaved rows to use vpmaddubsw instructions
 * for packing B matrix. For 32-bit accumulation, ROW_INTERLEAVE = 4; For 16-bit
 * accumulation, ROW_INTERLEAVE = 2.
 *
 * VLEN: the vector length of one SIMD register. For avx2, VLEN = 256; For
 * avx512, VLEN = 512.
 *
 * NR: the register blocking parameters for N dimension. NR columns of
 * interleaved rows of int8 (singed or unsigned) should fit into one SIMD
 * register. Basically, NR = VLEN / 8 / ROW_INTERLEAVE (8 is the bit length for
 * int8 (signed or unsigned).
 *
 * MR: the register blocking parameters for M dimension. MR is the total number
 * of SIMD registers used for M dimension of registers used for accumulation C.
 * This indicates the number of vpbroadcastw instructions for A.
 *
 * NCB: the cache blocking parameters for N dimension. NCB needs to be a
 * multiple of NR. The total register on N dimension of registers used for
 * accumulation C should be NCB/NR.
 *
 * (MR) * (NCB/NR): the number of registers used for accumulation C. (MR) *
 * (NCB/NR) should be less than the total register number (avx2 has 16 ymm
 * registers; avx512 has 32 zmm registers). (MR) * (NCB/NR) should be as large
 * as possible to increase the register utilization.
 *
 * KCB: the cache blocking parameters for K dimension.
 *
 * MCB: the cache blocking parameters for M dimension. MCB needs to be a
 * multiple of MR.
 *
 */

/**
 * @brief Packing parameter specialization for accumulation into 32-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx2.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int32_t,
    inst_set_t::avx2,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{12}; ///< Register block for M dimension.
  static constexpr int NR{8}; ///< Register block for N dimension.
                              ///< NR = VLEN/8/ROW_INTERLEAVE = 256 / 8 / 4 = 8.
                              ///< Total registers used for N dimension: NCB/NR.
                              ///< Here we use 12 x 1 ymm register blocking for
                              ///< the registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      120}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      8}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{512}; ///< Cache block for K dimension.
};

/**
 * @brief Packing parameter specialization for accumulation into 16-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx2.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int16_t,
    inst_set_t::avx2,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{3}; ///< Register block for M dimension.
  static constexpr int NR{
      16}; ///< Register block for N dimension;
           ///< NR = VLEN/8/ROW_INTERLEAVE = 256 / 8 / 2 = 16.
           ///< Total registers used for N dimension: NCB/NR.
           ///< Here we use 3 x 4 ymm register blocking for the
           ///< registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      2}; ///< 2 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      60}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      64}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.
};

/**
 * @brief Packing parameter specialization for float input and float
 * accumulation.
 *
 * This is picked when template paramtere T is of float type and instruction
 * set is avx2.
 */
template <>
struct PackingTraits<float, float, inst_set_t::avx2> {
  static constexpr int MR{3}; ///< Register block for M dimension
  static constexpr int NR{32}; ///< Register block for N dimension

  static constexpr int ROW_INTERLEAVE{1}; ///< No Row interleave.

  static constexpr int MCB{
      24}; ///< Cache block for M dimension (multiple of MR)
  static constexpr int NCB{
      64}; ///< Cache block for N dimension (multiple of NR)
  static constexpr int KCB{256}; ///< Cache block for K dimension
};

/**
 * @brief Packing parameter specialization for fp16 input and float
 * accumulation.
 *
 * This is picked when template parameter T is of float16 type and instruction
 * set is avx2.
 */
template <>
struct PackingTraits<float16, float, inst_set_t::avx2> {
  static constexpr int BCOL{8};
  static constexpr int ROW_INTERLEAVE{1};
};

/**
 * @brief Packing parameter specialization for accumulation into 32-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int32_t,
    inst_set_t::avx512,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{28}; ///< Register block for M dimension.
  static constexpr int NR{
      16}; ///< Register block for N dimension.
           ///< NR = VLEN/8/ROW_INTERLEAVE = 512 / 8 / 4 = 16.
           ///< Total registers used for N dimension: NCB/NR.
           ///< Here we use 28 x 1 zmm register blocking for
           ///< the registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      140}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      16}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{512}; ///< Cache block for K dimension.
};

/**
 * @brief Packing parameter specialization for accumulation into 16-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int16_t,
    inst_set_t::avx512,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{6}; ///< Register block for M dimension
  static constexpr int NR{
      32}; ///< Register block for N dimension;
           ///< NR = VLEN/8/ROW_INTERLEAVE = 512 / 8 / 2 = 32.
           ///< Total registers used for N dimension: NCB/NR.
           ///< Here we use 6 x 4 zmm register blocking for
           ///< the registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      2}; ///< 2 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      60}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      128}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.
};
