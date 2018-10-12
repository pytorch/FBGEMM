/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

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
  static constexpr int MR{12}; ///< Register block for M dimension
  static constexpr int NR{8}; ///< Register block for N dimension

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      120}; ///< cache block for M dimension (multiple of MR)
  static constexpr int NCB{8}; ///< cache block for N dimension (multiple of NR)
  static constexpr int KCB{512}; ///< cache block for K dimension
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
  static constexpr int MR{3}; ///< Register block for M dimension
  static constexpr int NR{16}; ///< Register block for N dimension; Total
                               ///< register used for N dimension: NCB/NR

  static constexpr int ROW_INTERLEAVE{
      2}; ///< 2 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      60}; ///< cache block for M dimension (multiple of MR)
  static constexpr int NCB{
      64}; ///< cache block for N dimension (multiple of NR)
  static constexpr int KCB{256}; ///< cache block for K dimension
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
      24}; ///< cache block for M dimension (multiple of MR)
  static constexpr int NCB{
      64}; ///< cache block for N dimension (multiple of NR)
  static constexpr int KCB{256}; ///< cache block for K dimension
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
  static constexpr int MR{28}; ///< Register block for M dimension
  static constexpr int NR{16}; ///< Register block for N dimension

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix

  static constexpr int MCB{
      140}; ///< cache block for M dimension (multiple of MR)
  static constexpr int NCB{
      16}; ///< cache block for N dimension (multiple of NR)
  static constexpr int KCB{512}; ///< cache block for K dimension
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
  static constexpr int NR{32}; ///< Register block for N dimension; Total
                               ///< register used for N dimension: NCB/NR

  static constexpr int ROW_INTERLEAVE{
      2}; ///< 2 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      60}; ///< cache block for M dimension (multiple of MR)
  static constexpr int NCB{
      128}; ///< cache block for N dimension (multiple of NR)
  static constexpr int KCB{256}; ///< cache block for K dimension
};
