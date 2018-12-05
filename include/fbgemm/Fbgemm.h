/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

/**
 * Top level include file for FBGEMM.
 */
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include "ConvUtils.h"
#include "FbgemmI8Spmdm.h"
#include "Types.h"
#include "Utils.h"
#include "FbgemmBuild.h"
#include "QuantUtilsAvx2.h"

// Turning on this option will print out time breakdown of each stage (e.g.,
// input packing, the main GEMM kernel, each output processing pipeline).
// Please note that currently this option won't report accurate timing if
// multiple threads are used.
// #define FBGEMM_MEASURE_TIME_BREAKDOWN

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
#include <chrono>
#include <iostream>
extern double packing_time;
extern double computing_time;
extern double kernel_time;
extern double postprocessing_time;
extern double run_time;
#endif

namespace fbgemm {

/**
 * @brief Templatized struct for packing parameters for A and B matrices.
 *
 * @tparam T input type
 * @tparam accT the type used for accumulation
 * @tparam instSet anyarch/avx2/avx512
 * @tparam int8Type an auxiliary template parameter to specialize for 8-bit
 *                  input types.
 */
template <
    typename T,
    typename accT,
    inst_set_t instSet,
    typename int8Type = void>
struct PackingTraits;

// type specialized implementation in an include file
#include "PackingTraits-inl.h"

/**
 * @brief Base class for packing matrices for higher GEMM performance.
 *
 * Matrix is tiled into blockRows() * blockCols() blocks.
 * Each block is with size blockRowSize() * blockColSize().
 * This class is designed using CRTP
 * (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
 *
 * @tparam PT actual packing type, e.g., PackAWithRowOffset
 */
template <typename PT, typename inpType, typename accType = std::int32_t>
class PackMatrix {
 public:
  PackMatrix() = delete; // no default constructor

  /**
   * @param rows total number of rows in the matrix
   *             (packed rows can be less than rows).
   * @param cols total number of columns in the matrix
   * @param pmat A buffer to contain the packed matrix.
   *             If nullptr, a buffer owned by PackMatrix will be allocated
   *             internally to contain the packed matrix.
   *             For non-constant matrices like activation matrices, the client
   *             code may want to pass a pre-allocated pmat to avoid the
   *             overhead of internal memory allocation everytime a PackMatrix
   *             is constructed. The client code can query how big patm should
   *             be with packedBufferSize function.
   * @param groups when groups > 1, we compute groups number of GEMMs each
   *               multiplies A.rows by A.cols/A.groups matrix with
   *               B.rows/B.groups by B.cols matrix (in conventional BLAS
   *               terminology, this is a batched GEMM but we use the name group
   *               to follow deep learning terminology). The result matrix has
   *               dimension A.rows by B.cols*B.groups .
   *               A.groups must be same as B.groups, A.groups must divide
   *               A.cols, and B.groups must divide B.rows and C.cols.
   */
  PackMatrix(
      std::int32_t rows,
      std::int32_t cols,
      inpType* pmat,
      int groups = 1);

  /**
   * @return true usually when the matrix is constant matrix (e.g., weight
   *         matrices) that can be prepacked
   */
  bool isPrePacked() const {
    return static_cast<const PT*>(this)->isPrePacked();
  }

  /**
   * @return true if this is the first input matrix in GEMM (i.e., A in C = A *
   *         B)
   */
  static constexpr bool isA() {
    return PT::isA();
  }

  /**
   * @brief The size of the buffer used for packing (The size is in number of
   *        elements).
   *
   * rows and cols are only used for fully packing, i.e., for B matrix.  The
   * client code can use this function to query how big the buffer used for
   * packing should be.
   */
  static int packedBufferSize(int rows = 0, int cols = 0);

  /**
   * @return Pointer to a buffer containing row offset results. Some packing
   *         objects fuse row offset computation for later requantization step.
   */
  std::int32_t* getRowOffsetBuffer() const {
    return static_cast<const PT*>(this)->getRowOffsetBuffer();
  }

  /**
   * @brief When k loop is also tiled/blocked, this function is used to check if
   * have executed computations for the last k block so that we can perform
   *        post-GEMM operations.
   */
  bool isThisLastKBlock(int block_id) const {
    return static_cast<const PT*>(this)->isThisLastKBlock(block_id);
  }

  /**
   * @brief Actual packing of a block of the source matrix in pmat buffer.
   */
  void pack(const block_type_t& block) {
    static_cast<PT*>(this)->pack(block);
  }

  std::int32_t numRows() const {
    return nrows_;
  }

  std::int32_t numCols() const {
    return ncols_;
  }

  /**
   * @return The number of rows in each block
   */
  std::int32_t blockRowSize() const {
    return brow_;
  }

  /**
   * @return The number of columns in each block
   */
  std::int32_t blockColSize() const {
    return bcol_;
  }

  /**
   * @return The number of blocks along rows
   */
  std::int32_t blockRows() const {
    return nbrow_;
  }

  /**
   * @return The number of blocks along columns
   */
  std::int32_t blockCols() const {
    return nbcol_;
  }

  /**
   * @return The number of the rows in the currently packed block of a matrix.
   *         For pre-packed (i.e., fully-packed), it's equal to the total number
   * of rows.
   */
  std::int32_t numPackedRows() const {
    return packedBlock_.row_size;
  }

  /**
   * @return The number of columns in the currently packed block of a matrix.
   *         For pre-packed (i.e., fully-packed), it's equal to the number of
   * columns.
   */
  std::int32_t numPackedCols() const {
    return packedBlock_.col_size;
  }

  /**
   * @return The first row of the block we're working on.
   */
  std::int32_t packedRowStart() const {
    return packedBlock_.row_start;
  }

  /**
   * @return The beginning of (rowBlockNum, colBlockNum)th block
   */
  inpType* getBuf(std::int32_t rowBlockNum = 0, std::int32_t colBlockNum = 0) {
    return buf_ + blockRowSize() * blockColSize() * rowBlockNum +
        blockRowSize() * blockColSize() * blockCols() * colBlockNum;
  }

  /**
   * @brief Print the packed block.
   */
  void printPackedMatrix(std::string name) {
    static_cast<PT*>(this)->printPackedMatrix(name);
  }

  /**
   * @return The number of rows in the last row block.
   */
  std::int32_t lastBrow() const {
    return last_brow_;
  }

  /**
   * @return The number of columns in the last column block.
   */
  std::int32_t lastBcol() const {
    return last_bcol_;
  }

  int numGroups() const {
    return G_;
  }

  /**
   * @return True if the last column block has fewer columns than the block
   *         size.
   */
  bool isThereColRemainder() const {
    return last_bcol_ != blockColSize();
  }

  virtual ~PackMatrix() {
    if (bufAllocatedHere_) {
      free(buf_);
    }
  }

 protected:
  /**
   * Set which block we're packing
   */
  void packedBlock(const block_type_t& block) {
    packedBlock_ = block;
    nbrow_ = (numPackedRows() + blockRowSize() - 1) / blockRowSize();
    nbcol_ = (numPackedCols() + blockColSize() - 1) / blockColSize();

    last_brow_ = ((numPackedRows() % blockRowSize()) == 0)
        ? blockRowSize()
        : (numPackedRows() % blockRowSize());
    last_bcol_ = ((numPackedCols() % blockColSize()) == 0)
        ? blockColSize()
        : (numPackedCols() % blockColSize());
  }

  inpType* buf_;
  std::int32_t brow_; ///< the number of rows in each block
  std::int32_t bcol_; ///< the number of columns in each block
  std::int32_t nbrow_; ///< the number of blocks along rows
  std::int32_t nbcol_; ///< the number of blocks along columns
  bool bufAllocatedHere_;

 private:
  std::int32_t nrows_, ncols_;
  int G_;
  block_type_t packedBlock_; ///< The block in the source matrix just packed
  std::int32_t last_brow_, last_bcol_;
};

/**
 * @brief Matrix packed for the first input matrix in GEMM (usually
 *        activation).  The source matrix is already quantized. Default
 * accumulation type is int32.
 */
template <typename T, typename accT = std::int32_t>
class FBGEMM_API PackAMatrix final
    : public PackMatrix<PackAMatrix<T, accT>, T, accT> {
 public:
  using This = PackAMatrix<T, accT>;
  using BaseType = PackMatrix<This, T, accT>;
  using inpType = T;
  using accType = accT;

  PackAMatrix() = delete; // no default constructor

  PackAMatrix(
      matrix_op_t trans,
      std::int32_t nRow,
      std::int32_t nCol,
      const inpType* smat,
      std::int32_t ld,
      inpType* pmat = nullptr,
      int groups = 1);

  /**
   * Activation matrices are not constant so cannot amortize the cost of
   * pre-packing.
   */
  bool isPrePacked() const {
    return false;
  }

  /**
   * @return True if this is used as A matrix.
   */
  static constexpr bool isA() {
    return true;
  }

  /**
   * @return A pointer to the row offset buffer. There is no row offset buffer
   *         calculations with this packing class, hence, it returns nullptr.
   */
  std::int32_t* getRowOffsetBuffer() const {
    return nullptr;
  }

  /**
   * @return Offset of the element in the packed matrix that was at (i, j) in
   *         the source matrix.
   */
  std::int32_t addr(std::int32_t i, std::int32_t j) const;

  /**
   * @brief Packs a block of source matrix into pmat buffer.
   */
  void pack(const block_type_t& block);

  /**
   * @brief Print the packed block.
   */
  void printPackedMatrix(std::string name);

 private:
  matrix_op_t trans_;
  const T* smat_;
  std::int32_t ld_;
  std::int32_t row_interleave_B_;
};

/**
 * @brief Matrix packed for the second input matrix in GEMM (usually weight).
 *        The source matrix is already quantized. Default accumulation
 *        type is int32.
 */
template <typename T, typename accT = std::int32_t>
class FBGEMM_API PackBMatrix final
    : public PackMatrix<PackBMatrix<T, accT>, T, accT> {
 public:
  using This = PackBMatrix<T, accT>;
  using BaseType = PackMatrix<This, T, accT>;
  using inpType = T;
  using accType = accT;

  PackBMatrix() = delete; // no default constructor

  /**
   * @params groups if > 1 and trans == NoTranspose, smat is nRow x nCol with
   *                groups are vertically concatenated: each group is
   *                (nRow / groups) x nCol .
   *                if > 1 and trans == Transpose, smat is (nCol * groups) x
   *                (nRow / groups) with groups are horizontally concatenated:
   *                each group is nCol x (nRow / groups) . Each group is
   *                transposed and vertically concatenated to match with the
   *                NoTranspose case.
   */
  PackBMatrix(
      matrix_op_t trans,
      std::int32_t nRow,
      std::int32_t nCol,
      const inpType* smat,
      std::int32_t ld,
      inpType* pmat = nullptr,
      int groups = 1);

  /**
   * Weight matrices are usually constant so worth pre-packing.
   */
  bool isPrePacked() const {
    return true;
  }

  /**
   * @return True if to be used as A matrix, False otherwise.
   */
  static constexpr bool isA() {
    return false;
  }

  /**
   * @brief When k loop is also tiled/blocked, this function is used to check if
   * have executed computations for the last k block so that we can perform
   *        post-GEMM operations.
   */
  bool isThisLastKBlock(int block_id) const {
    return (BaseType::blockRows() - 1) == block_id;
  }

  /**
   * @return Offset of the element in the packed matrix that was at (i, j) in
   *         the source matrix.
   */
  std::int32_t addr(std::int32_t i, std::int32_t j) const;

  /**
   * @brief Packs a block of source matrix into pmat buffer.
   */
  void pack(const block_type_t& block);

  /**
   * @brief Print the packed block.
   */
  void printPackedMatrix(std::string name);

  /**
   * @return true if meta information like matrix shape is the same.
   */
  bool metaEquals(const PackBMatrix<T, accT>& that) const;
  /**
   * @return true if matrices are the same.
   */
  bool equals(const PackBMatrix<T, accT>& that) const;

  ~PackBMatrix() {}

 private:
  matrix_op_t trans_;
  const T* smat_;
  std::int32_t ld_;
  std::int32_t row_interleave_;
};

/**
 * @brief Matrix packed for the first input matrix in GEMM (usually activation),
 *        and row offsets used for requantization is computed during packing.
 *        Im2col is fused with packing here. The source matrix is already
 * quantized.
 */
template <typename T, typename accT = std::int32_t, int SPATIAL_DIM = 2>
class FBGEMM_API PackAWithIm2Col
    : public PackMatrix<PackAWithIm2Col<T, accT, SPATIAL_DIM>, T, accT> {
 public:
  using This = PackAWithIm2Col<T, accT, SPATIAL_DIM>;
  using BaseType = PackMatrix<This, T, accT>;
  using inpType = T;
  using accType = accT;

  PackAWithIm2Col() = delete; // no default constructor
  /**
   * @param zero_pt the quantized value that maps to 0.0f floating-point number.
   */
  PackAWithIm2Col(
      const conv_param_t<SPATIAL_DIM>& conv_param,
      const T* sdata,
      inpType* pmat = nullptr,
      std::int32_t zero_pt = 0,
      std::int32_t* row_offset = nullptr);

  /**
   * Activation matrices are not constant so cannot amortize the cost of
   * pre-packing.
   */
  bool isPrePacked() const {
    return false;
  }

  /**
   * @return True if this is used as A matrix.
   */
  static constexpr bool isA() {
    return true;
  }

  /**
   * @brief Packs a block of source matrix into pmat buffer.
   */
  void pack(const block_type_t& block);

  /**
   * @return A pointer to the row offset buffer.
   */
  std::int32_t* getRowOffsetBuffer() const {
    return row_offset_;
  }

  /**
   * @brief Print the packed block.
   */
  void printPackedMatrix(std::string name);

  /**
   * @return Size of row offset buffer in number of elements
   */
  static int rowOffsetBufferSize();

  ~PackAWithIm2Col() {
    if (rowOffsetAllocatedHere) {
      free(row_offset_);
    }
  }

 private:
  const conv_param_t<SPATIAL_DIM>& conv_p_;
  const T* sdata_;
  std::int32_t zero_pt_;
  std::int32_t* row_offset_;
  bool rowOffsetAllocatedHere;
  std::int32_t row_interleave_B_;
};

/**
 * @brief Matrix packed for the first input matrix in GEMM (usually activation),
 *        and row offsets used for requantization is computed during packing.
 *        The source matrix is already quantized.
 */
template <typename T, typename accT = std::int32_t>
class FBGEMM_API PackAWithRowOffset final
    : public PackMatrix<PackAWithRowOffset<T, accT>, T, accT> {
 public:
  using This = PackAWithRowOffset<T, accT>;
  using BaseType = PackMatrix<This, T, accT>;
  using inpType = T;
  using accType = accT;

  PackAWithRowOffset() = delete; // no default constructor
  PackAWithRowOffset(
      matrix_op_t trans,
      std::uint32_t nRow,
      std::uint32_t nCol,
      const T* smat,
      std::uint32_t ld,
      inpType* pmat = nullptr,
      int groups = 1,
      std::int32_t* row_offset = nullptr);

  /**
   * Activation matrices are not constant so cannot amortize the cost of
   * pre-packing.
   */
  bool isPrePacked() const {
    return false;
  }

  /**
   * @return True if this is used as A matrix.
   */
  static constexpr bool isA() {
    return true;
  }

  /**
   * @return Offset of the element in the packed matrix that was at (i, j) in
   *         the source matrix
   */
  std::int32_t addr(std::int32_t i, std::int32_t j) const;

  /**
   * @brief Packs a block of source matrix into pmat buffer.
   */
  void pack(const block_type_t& block);

  /**
   * @return A pointer to the row offset buffer.
   */
  std::int32_t* getRowOffsetBuffer() const {
    return row_offset_;
  }

  /**
   * @brief Print the packed block.
   */
  void printPackedMatrix(std::string name);

  /**
   * @return size of row offset buffer in number of elements
   */
  static int rowOffsetBufferSize();

  ~PackAWithRowOffset() {
    if (rowOffsetAllocatedHere) {
      free(row_offset_);
    }
  }

 private:
  matrix_op_t trans_;
  const T* smat_;
  std::uint32_t ld_;
  std::int32_t* row_offset_;
  bool rowOffsetAllocatedHere;
  std::int32_t row_interleave_B_;
};

/**
 * @brief Matrix packed for the first input matrix in GEMM (usually activation),
 *        and row offsets used for requantization is computed during packing.
 *        The source matrix is in fp32 and quantized during packing.
 */
template <typename T, typename accT = std::int32_t>
class FBGEMM_API PackAWithQuantRowOffset final
    : public PackMatrix<PackAWithQuantRowOffset<T, accT>, T, accT> {
 public:
  using This = PackAWithQuantRowOffset<T, accT>;
  using BaseType = PackMatrix<This, T, accT>;
  using inpType = T;
  using accType = accT;

  PackAWithQuantRowOffset() = delete; // no default constructor
  PackAWithQuantRowOffset(
      matrix_op_t trans,
      std::int32_t nRow,
      std::int32_t nCol,
      const float* smat,
      std::int32_t ld,
      inpType* pmat = nullptr,
      float scale = 1.0f,
      std::int32_t zero_pt = 0,
      int groups = 1,
      std::int32_t* row_offset = nullptr);

  /**
   * Activation matrices are not constant so cannot amortize the cost of
   * pre-packing.
   */
  bool isPrePacked() const {
    return false;
  }

  /**
   * @return True if this is used as A matrix.
   */
  static constexpr bool isA() {
    return true;
  }

  /**
   * @return offset of the element in the packed matrix that was at (i, j) in
   *         the source matrix
   */
  std::int32_t addr(std::int32_t i, std::int32_t j) const;

  /**
   * @brief Packs a block of source matrix into pmat buffer.
   */
  void pack(const block_type_t& block);

  /**
   * @return A pointer to the row offset buffer.
   */
  std::int32_t* getRowOffsetBuffer() const {
    return row_offset_;
  }

  /**
   * @brief Print the packed block.
   */
  void printPackedMatrix(std::string name);

  /**
   * @return Size of row offset buffer in number of elements
   */
  static int rowOffsetBufferSize();

  ~PackAWithQuantRowOffset() {
    if (rowOffsetAllocatedHere) {
      free(row_offset_);
    }
  }

 private:
  matrix_op_t trans_;
  const float* smat_;
  std::int32_t ld_;
  float scale_;
  std::int32_t zero_pt_;
  std::int32_t* row_offset_;
  bool rowOffsetAllocatedHere;
  std::int32_t row_interleave_B_;
};

/*
 *
 * Post Processing of outputs
 *
 */

/**
 * @brief Does nothing. NoOp. Used as the last operation in the output
 *        processing pipeline.
 *
 */
template <typename outT = std::uint8_t, typename inT = std::uint8_t>
class FBGEMM_API DoNothing {
 public:
  using outType = outT;
  using inpType = inT;
  DoNothing() {}
  template <inst_set_t instSet>
  int f(
      outType* /* unused */,
      inpType* /* unused */,
      const block_type_t& /* unused */,
      int /* unused */,
      int /* unused */) const {
    return 0;
  }
};

/**
 * @brief Copy data pointed by inp ptr to out ptr when
 *        inp ptr and out ptr are not the same.
 *        inp buffer: row and column start points: (0, 0)
 *        output buffer: row and column start points:
 *        (block.row_start, block.col_start)
 *
 * This is the output processing stage that should passed when there is no
 * requantization and output is required in the same format as internal buffer
 * used for accumulation.
 */
template <
    typename outT = std::int32_t,
    typename inT = std::int32_t,
    typename nextOPType = DoNothing<outT, outT>>
class FBGEMM_API memCopy {
 public:
  using outType = outT;
  using inpType = inT;
  explicit memCopy(nextOPType& nextop) : nextop_(nextop) {}
  template <inst_set_t instSet>
  inline int f(
      outType* out,
      inpType* inp,
      const block_type_t& block,
      int ld_out,
      int ld_in) const;

 private:
  nextOPType& nextop_;
};

/**
 * @brief Perform scaling on accumulated data.
 */
template <
    typename outT = std::int32_t,
    typename inT = std::int32_t,
    typename nextOPType = DoNothing<outT, outT>>
class ScaleOP {
 public:
  using outType = outT;
  using inpType = inT;
  explicit ScaleOP(inpType scalingFactor) : scalingFactor_(scalingFactor) {}

  template <inst_set_t instSet>
  inline int f(
      outType* out,
      inpType* inp,
      const block_type_t& block,
      int ld_out,
      int ld_in) const;

 private:
  inpType scalingFactor_;
};

/**
 * @brief Perform Relu on accumulated data.
 */
template <
    typename outT = std::int32_t,
    typename inT = std::int32_t,
    typename nextOPType = DoNothing<outT, outT>>
class ReluOutput {
 public:
  using outType = outT;
  using inpType = inT;
  explicit ReluOutput(inpType zero_pt) : zero_pt_(zero_pt) {}

  template <inst_set_t instSet>
  inline int f(
      outType* out,
      inpType* inp,
      const block_type_t& block,
      int ld_out,
      int ld_in) const;

 private:
  inpType zero_pt_;
};

/**
 * @brief Perform Dense-Matrix * Sparse-Matrix as a part the of output
 * processing pipeline.
 *
 * SPMDM (SParse Matrix times Dense Matrix) inplace on the 32-bit input buffer
 * (inp). After modifying the input buffer, pass it to the next op.
 * When groups > 1, each group is numRows() x (numCols()/groups) matrix.
 */
template <
    typename outT = std::int32_t,
    typename inT = std::int32_t,
    typename nextOPType = DoNothing<inT, inT>>
class FBGEMM_API DoSpmdmOnInpBuffer {
 public:
  using outType = outT;
  using inpType = inT;
  DoSpmdmOnInpBuffer(
      nextOPType& nextop,
      const std::uint8_t* A,
      int lda,
      const CompressedSparseColumn& B_csc,
      int groups = 1)
      : nextop_(nextop), A_(A), lda_(lda), B_csc_(B_csc), groups_(groups) {}

  template <inst_set_t instSet>
  inline int f(
      outT* out,
      inT* inp,
      const block_type_t& block,
      int ld_out,
      int ld_in) const;

 private:
  nextOPType& nextop_;
  const std::uint8_t* A_;
  const int lda_;
  const CompressedSparseColumn& B_csc_;
  const int groups_;
};

/**
 * @brief Perform Dense-Matrix * Sparse-Matrix as a part the of output
 * processing pipeline.
 *
 * SPMDM (SParse Matrix times Dense Matrix) inplace on the 32-bit input buffer
 * (inp). After modifying the input buffer, pass it to the next op.
 * When groups > 1, each group is numRows() x (numCols()/groups) matrix.
 */
template <
    typename outT = std::int32_t,
    typename inT = std::int32_t,
    typename nextOPType = DoNothing<inT, inT>>
class FBGEMM_API DoSConvOnInpBuffer {
 public:
  using outType = outT;
  using inpType = inT;
  DoSConvOnInpBuffer(
      nextOPType& nextop,
      const std::uint8_t* A,
      const conv_param_t<>& conv_p,
      std::int32_t A_zero_point,
      const CompressedSparseColumn& B_csc,
      int groups = 1)
      : nextop_(nextop),
        A_(A),
        conv_p_(conv_p),
        A_zero_point_(A_zero_point),
        B_csc_(B_csc) {}

  template <inst_set_t instSet>
  inline int f(
      outT* out,
      inT* inp,
      const block_type_t& block,
      int ld_out,
      int ld_in) const;

 private:
  nextOPType& nextop_;
  const std::uint8_t* A_;
  const conv_param_t<>& conv_p_;
  const std::int32_t A_zero_point_;
  const CompressedSparseColumn& B_csc_;
};

/**
 * @brief Requantize values in inp buffer and write to out buffer.
 *        pass the out buffer to next op for further processing.
 */
template <
    bool FUSE_RELU,
    QuantizationGranularity Q_GRAN = QuantizationGranularity::TENSOR,
    typename outT = std::uint8_t,
    typename inT = std::int32_t,
    typename nextOPType = DoNothing<outT, outT>>
class FBGEMM_API ReQuantizeOutput {
 public:
  using outType = outT;
  using inpType = inT;
  ReQuantizeOutput(
      nextOPType& nextop,
      const float* C_multiplier,
      std::int32_t C_zero_point,
      std::int32_t Aq_zero_point,
      const std::int32_t* Bq_zero_point,
      const std::int32_t* row_offsets,
      const std::int32_t* col_offsets,
      const std::int32_t* bias,
      std::uint32_t nCol,
      int groups = 1)
      : nextop_(nextop),
        C_multiplier_(C_multiplier),
        C_zero_point_(C_zero_point),
        Aq_zero_point_(Aq_zero_point),
        Bq_zero_point_(Bq_zero_point),
        q_row_offsets_(row_offsets),
        q_col_offsets_(col_offsets),
        bias_(bias),
        ncols_(nCol),
        groups_(groups) {}

  template <inst_set_t instSet>
  inline int f(
      outT* out,
      const inT* inp,
      const block_type_t& block,
      int ld_out,
      int ld_in) const;

 private:
  nextOPType& nextop_;
  const float* C_multiplier_;
  std::int32_t C_zero_point_;
  std::int32_t Aq_zero_point_;
  const std::int32_t* Bq_zero_point_;
  const std::int32_t* q_row_offsets_;
  const std::int32_t* q_col_offsets_;
  const std::int32_t* bias_;
  std::uint32_t ncols_;
  int groups_;
};

/**
 * @brief Requantize to convert accumulated data to be used as float, i.e., the
 *        output would be used as float.
 */
template <
    bool FUSE_RELU,
    QuantizationGranularity Q_GRAN = QuantizationGranularity::TENSOR,
    typename outT = float,
    typename inT = std::int32_t,
    typename nextOPType = DoNothing<outT, outT>>
class FBGEMM_API ReQuantizeForFloat {
 public:
  using outType = outT;
  using inpType = inT;
  ReQuantizeForFloat(
      nextOPType& nextop,
      float Aq_scale,
      const float* Bq_scale,
      std::int32_t Aq_zero_point,
      const std::int32_t* Bq_zero_point,
      const std::int32_t* row_offsets,
      const std::int32_t* col_offsets,
      const float* bias,
      std::uint32_t nCol,
      int groups = 1)
      : nextop_(nextop),
        Aq_scale_(Aq_scale),
        Bq_scale_(Bq_scale),
        Aq_zero_point_(Aq_zero_point),
        Bq_zero_point_(Bq_zero_point),
        q_row_offsets_(row_offsets),
        q_col_offsets_(col_offsets),
        bias_(bias),
        ncols_(nCol),
        groups_(groups) {}

  template <inst_set_t instSet>
  inline int f(
      outT* out,
      inT* inp,
      const block_type_t& block,
      int ld_out,
      int ld_in) const;

 private:
  nextOPType& nextop_;
  float Aq_scale_;
  const float* Bq_scale_;
  std::int32_t Aq_zero_point_;
  const std::int32_t* Bq_zero_point_;
  const std::int32_t* q_row_offsets_;
  const std::int32_t* q_col_offsets_;
  const float* bias_;
  std::uint32_t ncols_;
  int groups_;
};

// type specialized implementation in an include file
#include "OutputProcessing-inl.h"

/*
 *
 * ####### GEMM related functions #######
 *
 */

/**
 * Matrix B must be prepacked. For matrix A, packA.pack function is called to
 * pack it.
 *
 * @tparam packingAMatrix processing of A matrix while packing,
 *                        e.g., PackAWithQuantRowOffset
 *
 * @tparam packingBMatrix processing of B matrix while packing,
 *                        e.g.,  pre-multiply by alpha
 * @tparam cT data type of C matrix
 * @tparam processOutputType further processing of outputs, e.g., Relu
 */
template <
    typename packingAMatrix,
    typename packingBMatrix,
    typename cT,
    typename processOutputType>
FBGEMM_API void fbgemmPacked(
    PackMatrix<
        packingAMatrix,
        typename packingAMatrix::inpType,
        typename packingAMatrix::accType>& packA,
    PackMatrix<
        packingBMatrix,
        typename packingBMatrix::inpType,
        typename packingBMatrix::accType>& packB,
    cT* C,
    std::int32_t* C_buffer,
    std::uint32_t ldc,
    const processOutputType& outProcess,
    int thread_id,
    int num_threads);

/**
 * @brief Perform depthwise separable convolution
 */
template <
    typename packingAMatrix,
    typename packingBMatrix,
    typename outT,
    typename processOutputType>
void convDepthwiseSeparable(
    const conv_param_t<>& conv_param_dw,
    const conv_param_t<>& conv_param_1x1,
    packingAMatrix& packdw,
    packingBMatrix& packed_1x1,
    outT* out,
    const processOutputType& output);

/**
 * @brief Allocate __size bytes of uninitialized storage whose alignment is
 * specified by __align.
 */
static void* fbgemmAlignedAlloc(size_t __align, size_t __size) {
  void* aligned_mem;
  if (posix_memalign(&aligned_mem, __align, __size))
    return 0;
  return aligned_mem;
}

/**
 * @brief Are we running on a fbgemm supported cpu?
 */
FBGEMM_API bool fbgemmSupportedCPU();

/*
 * @brief Partition the workload between 0 and m into num_threads segments. Each
 * thread gets a multiple of mr, except that the last one might receive the
 * fringe case. Return the start and end index of each thread.
 * Example: mr = 8
 *    m  mRegBlocks  mRegRemainder  num_thread_left _right  th0  th1  th2
 *  120          15              0                0      3   40   40   40
 *  123          15              3                0      3   40   40   43
 *  133          16              5                1      2   48   40   45
 *  140          17              4                2      1   48   48   44
 *  146          18              2                0      3   48   48   50
 *  144          18              0                0      3   48   48   48
 *
 * ToDo: Make this routine more general: partition the workload between any
 * intervals. We can then reuse this routine for the nested parallel workload
 * distribution.
 */
static void fbgemmGetRange(
    int num_threads,
    int thread_id,
    int m,
    int mr,
    int& start,
    int& end) {
  int mRegBlocks = m / mr;
  int mRegRemainder = m % mr;

  int m_blk_per_thread = mRegBlocks / num_threads;

  int num_thread_left = mRegBlocks % num_threads;
  // int num_thread_right = num_threads - num_thread_left;

  int m_blk_left, m_blk_right;
  if (num_thread_left == 0) {
    m_blk_left = m_blk_per_thread;
    m_blk_right = m_blk_per_thread;
  } else {
    m_blk_left = m_blk_per_thread + 1;
    m_blk_right = m_blk_per_thread;
  }

  int size_left = m_blk_left * mr;
  int size_right = m_blk_right * mr;

  if (thread_id < num_thread_left) {
    start = 0 + thread_id * size_left;
    end = 0 + (thread_id + 1) * size_left;
  } else { // thread_id >= num_thread_left
    start = num_thread_left * size_left +
        (thread_id - num_thread_left) * size_right;
    end = num_thread_left * size_left +
        (thread_id - num_thread_left + 1) * size_right;
    if (thread_id == num_threads - 1)
      end += mRegRemainder;
  }
}

} // namespace fbgemm
