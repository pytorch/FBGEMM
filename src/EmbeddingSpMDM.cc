/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS

#include "fbgemm/FbgemmEmbedding.h"

#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include "./CodeCache.h"
#include "./MaskAvx2.h"
#include "./RefImplementations.h"
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/SimdUtils.h"

namespace fbgemm {

namespace {

namespace x86 = asmjit::x86;

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    bool ROWWISE_SPARSE>
class ReturnFunctionSignature {};

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType>
class ReturnFunctionSignature<inType, indxType, offsetType, outType, false> {
 public:
  using jit_embedding_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t data_size,
      const inType* input,
      const indxType* indices,
      const offsetType* offsets_or_lengths,
      const float* weights,
      outType* out,
      const int* mask);
};

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType>
class ReturnFunctionSignature<inType, indxType, offsetType, outType, true> {
 public:
  using jit_embedding_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t uncompressed_data_size,
      // int64_t compressed_data_size,
      const inType* input,
      const indxType* indices,
      const offsetType* offsets_or_lengths,
      const float* weights,
      outType* out,
      const int32_t* compressed_indices_table,
      const int* mask);
};

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE = false,
    bool THREAD_LOCAL = false>
class GenEmbeddingSpMDMLookup {
 public:
  GenEmbeddingSpMDMLookup() {}
  typename ReturnFunctionSignature<
      inType,
      indxType,
      offsetType,
      outType,
      ROWWISE_SPARSE>::jit_embedding_kernel
  getOrCreate(
      int block_size,
      bool has_weight,
      bool is_weight_positional,
      bool normalize_by_lengths,
      int prefetch,
      bool use_offsets,
      int output_stride,
      int input_stride,
      bool scale_bias_last,
      bool isbf16);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                  // depents on other static
                                  // variables.  Required to prevent
                                  // initialization order fiasco
    return rt;
  }

  static std::mutex rtMutex_; ///< Controll access to runtime;

  // The hash depends on embedding dimension (block size), weighted sls,
  // positional weights, normalize by lenths, prefetch distance, use_offsets,
  // output_stride, input_stride, and scale_bias_last
  static CodeCache<
      std::tuple<int, bool, bool, bool, int, bool, int, int, bool, bool>,
      typename ReturnFunctionSignature<
          inType,
          indxType,
          offsetType,
          outType,
          ROWWISE_SPARSE>::jit_embedding_kernel,
      THREAD_LOCAL>
      codeCache_; ///< JIT Code Cache for reuse.
}; // GenEmbeddingSpmDMLookup

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE,
    bool THREAD_LOCAL>
std::mutex GenEmbeddingSpMDMLookup<
    inType,
    indxType,
    offsetType,
    outType,
    instSet,
    ROWWISE_SPARSE,
    THREAD_LOCAL>::rtMutex_;

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE,
    bool THREAD_LOCAL>
CodeCache<
    std::tuple<int, bool, bool, bool, int, bool, int, int, bool, bool>,
    typename ReturnFunctionSignature<
        inType,
        indxType,
        offsetType,
        outType,
        ROWWISE_SPARSE>::jit_embedding_kernel,
    THREAD_LOCAL>
    GenEmbeddingSpMDMLookup<
        inType,
        indxType,
        offsetType,
        outType,
        instSet,
        ROWWISE_SPARSE,
        THREAD_LOCAL>::codeCache_;

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    inst_set_t instSet,
    bool ROWWISE_SPARSE,
    bool THREAD_LOCAL>
typename ReturnFunctionSignature<
    inType,
    indxType,
    offsetType,
    outType,
    ROWWISE_SPARSE>::jit_embedding_kernel
GenEmbeddingSpMDMLookup<
    inType,
    indxType,
    offsetType,
    outType,
    instSet,
    ROWWISE_SPARSE,
    THREAD_LOCAL>::
    getOrCreate(
        int block_size,
        bool has_weight,
        bool is_weight_positional,
        bool normalize_by_lengths,
        int prefetch,
        bool use_offsets,
        int output_stride,
        int input_stride,
        bool scale_bias_last,
        bool isbf16) {
  std::tuple<int, bool, bool, bool, int, bool, int, int, bool, bool> kernelSig =
      std::make_tuple(
          block_size,
          has_weight,
          is_weight_positional,
          normalize_by_lengths,
          prefetch,
          use_offsets,
          output_stride,
          input_stride,
          scale_bias_last,
          isbf16);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() -> typename ReturnFunctionSignature<
                inType,
                indxType,
                offsetType,
                outType,
                ROWWISE_SPARSE>::jit_embedding_kernel {
        bool is8bit = std::is_same<inType, uint8_t>::value;
        bool is16bit = std::is_same<inType, uint16_t>::value;
        bool is16bitout = std::is_same<outType, uint16_t>::value;
        bool isbf16out = isbf16;
        bool isfp16 = is16bit && !isbf16;
        bool isfp16out = is16bitout && !isbf16out;

        // TODO: Make this tunable
        int pref_dist = prefetch;
        bool areIndices64b = std::is_same<indxType, int64_t>::value;

        asmjit::CodeHolder code;
        code.init(runtime().environment());
        x86::Assembler assembler(&code);
        x86::Emitter* a = assembler.as<x86::Emitter>();
#if defined(FBGEMM_LOG_CODE)
        std::string filename = "embeddinglookup";
        if (is8bit) {
          filename += "_8bit";
        } else if (isfp16) {
          filename += "_fp16";
        } else if (isbf16) {
          filename += "_bf16";
        }
        if (isbf16out) {
          filename += "_bf16_out";
        } else if (isfp16out) {
          filename += "_fp16_out";
        }
        filename += "_emd_dim_" + std::to_string(block_size);
        filename += areIndices64b ? "_64bit" : "_32bit";
        filename += instSet == inst_set_t::avx512 ? "_avx512" : "_avx2";
        if (prefetch) {
          filename += "_prefetch";
        }
        if (has_weight) {
          filename += "_hasweight";
        }
        if (normalize_by_lengths) {
          filename += "_normalize_by_lengths";
        }
        if (!use_offsets) {
          filename += "_use_lengths";
        }
        if (ROWWISE_SPARSE) {
          filename += "_rowwise_sparse";
        }
        filename += "_out_stride_" + std::to_string(output_stride);
        if (!scale_bias_last) {
          filename += "_scale_bias_first";
        }
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif
        // arguments to the function created
        x86::Gp output_size = a->zdi();
        // index_size will be overwritten to hold the end address of indices
        x86::Gp index_size = a->zsi();
        x86::Gp data_size = a->zdx();
        x86::Gp input = a->zcx();
        int reg_id = 8;
        x86::Gp indices = a->gpz(reg_id); // 8
        ++reg_id;
        x86::Gp lengths = a->gpz(reg_id); // 9
        ++reg_id;
        x86::Gp weights = a->gpz(reg_id); // 10
        ++reg_id;
        x86::Gp out = a->gpz(reg_id); // 11

        x86::Gp compressed_indices_table;
        if (ROWWISE_SPARSE) {
          ++reg_id;
          compressed_indices_table = a->gpz(reg_id); // 12
        }
        ++reg_id;
        x86::Gp scratchReg1_ = a->gpz(reg_id); // 12 or 13, also for mask

        ++reg_id;
        x86::Gpd lengths_R_ = a->gpz(reg_id).r32(); // 13 or 14
        ++reg_id;
        x86::Gp scratchReg2_ = a->gpz(reg_id); // 14 or 15

        asmjit::FuncDetail func;

        if (ROWWISE_SPARSE) {
          func.init(
              asmjit::FuncSignatureT<
                  bool,
                  int64_t, // output_size
                  int64_t, // index_size
                  int64_t, // uncompressed_data_size
                  const inType*, // input uint8_t or float
                  const indxType*, // indices
                  const offsetType*, // offsets or lengths
                  const float*, // weights
                  outType*, // out
                  const int32_t*, // compressed_indices_table and then mask
                  const int*>(asmjit::CallConvId::kHost),
              a->environment());
        } else {
          func.init(
              asmjit::FuncSignatureT<
                  bool,
                  int64_t, // output_size
                  int64_t, // index_size
                  int64_t, // data_size
                  const inType*, // input uint8_t or float
                  const indxType*, // indices
                  const offsetType*, // offsets or lengths
                  const float*, // weights
                  outType*, // out and then mask
                  const int*>(asmjit::CallConvId::kHost),
              a->environment());
        }

        asmjit::FuncFrame frame;
        frame.init(func);

        if (instSet == inst_set_t::avx2) {
          frame.setDirtyRegs(
              asmjit::RegGroup::kVec,
              asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                  asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));
        } else {
          frame.setDirtyRegs(
              asmjit::RegGroup::kVec,
              asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                  asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15) |
                  asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
                  asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31));
        }

        frame.setDirtyRegs(
            asmjit::RegGroup::kGp,
            reg_id == 15
                ? asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15)
                : asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14));

        asmjit::FuncArgsAssignment args(&func);
        if (ROWWISE_SPARSE) {
          args.assignAll(
              output_size,
              index_size,
              data_size,
              input,
              indices,
              lengths,
              weights,
              out,
              compressed_indices_table,
              scratchReg1_);
        } else {
          args.assignAll(
              output_size,
              index_size,
              data_size,
              input,
              indices,
              lengths,
              weights,
              out,
              scratchReg1_);
        }

        args.updateFuncFrame(frame);
        frame.finalize();

        a->emitProlog(frame);
        a->emitArgsAssignment(frame, args);

        constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
        constexpr int NUM_VEC_REG = simd_info<instSet>::NUM_VEC_REGS;
        int unroll_factor = NUM_VEC_REG;

        typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

        int num_vec_regs_per_block = (block_size + vlen - 1) / vlen;
        int remainder = block_size % vlen;

        vec_reg_t scale_vreg; // holds scale
        vec_reg_t bias_vreg; // holds bias
        vec_reg_t w_vreg; // for weighted sls -- weights
        vec_reg_t
            vlen_inv_vreg; // used for normalize by lengths -- 1/ lengths[i]
        vec_reg_t src_vreg; // for holding embedding value temporarily
        x86::Ymm mask_vreg; // mask for avx2
        x86::Xmm mask_fp16_vreg; // mask for loading fp16 in avx2
        vec_reg_t ones_vreg; // 2^15 for bf16_2_fp32_rn

        if (is8bit) {
          // We need 2 vec registers for 1. scale 2. bias
          --unroll_factor;
          scale_vreg = vec_reg_t(unroll_factor);
          --unroll_factor;
          bias_vreg = vec_reg_t(unroll_factor);
        }

        if (isbf16out) {
          --unroll_factor;
          ones_vreg = vec_reg_t(unroll_factor);
          a->mov(scratchReg2_, 1 << 15);
          a->vpinsrd(ones_vreg.xmm(), ones_vreg.xmm(), scratchReg2_, 0);
          a->vpbroadcastd(ones_vreg, ones_vreg.xmm());
        }

        if (is8bit || is16bit || (remainder && instSet == inst_set_t::avx2)) {
          --unroll_factor;
          src_vreg = vec_reg_t(unroll_factor);
        }

        if (has_weight) {
          --unroll_factor;
          w_vreg = vec_reg_t(unroll_factor);
        }

        if (remainder && instSet == inst_set_t::avx2) {
          // AVX512 doesn't need to use vector register for masking
          --unroll_factor;
          mask_vreg = x86::ymm(unroll_factor);
          if (remainder > 1 && (is16bit || isbf16out || isfp16out)) {
            --unroll_factor;
            mask_fp16_vreg = x86::xmm(unroll_factor);
          }
        }

        if (normalize_by_lengths) {
          --unroll_factor;
          vlen_inv_vreg = vec_reg_t(unroll_factor);
        }

        if (remainder) {
          if (instSet == inst_set_t::avx2) {
            a->vmovups(
                mask_vreg,
                x86::ymmword_ptr(
                    scratchReg1_, (vlen - remainder) % vlen * sizeof(int32_t)));
            if (is16bit || isbf16out || isfp16out) {
              if (remainder > 1) {
                a->vmovups(
                    mask_fp16_vreg,
                    x86::xmmword_ptr(
                        scratchReg1_,
                        (vlen - remainder / 2) * sizeof(int32_t)));
              }
              // We need to keep using the stack during the main loop
              a->lea(
                  x86::rsp,
                  x86::dword_ptr(
                      x86::rsp, static_cast<int32_t>(-vlen * sizeof(int32_t))));
            }
          } else {
            a->mov(scratchReg1_, (1 << remainder) - 1);
            a->kmovw(x86::k(1), scratchReg1_);
          }
        }

        // Compute the end address of indices
        a->lea(
            index_size, x86::ptr(indices, index_size, areIndices64b ? 3 : 2));

        asmjit::Label exit = a->newLabel();
        asmjit::Label error = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        // rangeIndex loop begins (iterate output_size times)
        a->bind(LoopRangeIndexBegin);
        a->dec(output_size);
        a->jl(LoopRangeIndexEnd);

        if (normalize_by_lengths) {
          asmjit::Label IfLengthsBegin = a->newLabel();
          asmjit::Label IfLengthsEnd = a->newLabel();
          a->bind(IfLengthsBegin);
          if (use_offsets) {
            a->mov(lengths_R_, x86::dword_ptr(lengths, sizeof(offsetType)));
            a->sub(lengths_R_, x86::dword_ptr(lengths));
          } else {
            a->mov(lengths_R_, x86::dword_ptr(lengths));
          }
          a->cmp(lengths_R_, 1);
          // Initialize vlen_inv as 0 in case lengths is 0
          a->vxorps(vlen_inv_vreg, vlen_inv_vreg, vlen_inv_vreg);
          a->jl(IfLengthsEnd);

          // OK to use vreg0 because it's for out_vreg used in the main loop
          vec_reg_t temp_vreg(0);
          if (instSet == inst_set_t::avx2) {
            a->mov(scratchReg1_, 1);
            a->cvtsi2ss(vlen_inv_vreg.xmm(), scratchReg1_);
            a->cvtsi2ss(temp_vreg.xmm(), lengths_R_);
            a->divss(vlen_inv_vreg.xmm(), temp_vreg.xmm());
            a->vpbroadcastd(vlen_inv_vreg, vlen_inv_vreg.xmm());
          } else { // avx512
            a->mov(scratchReg1_, 1);
            a->cvtsi2ss(temp_vreg.xmm(), scratchReg1_);
            a->vpbroadcastd(vlen_inv_vreg, temp_vreg.xmm());
            a->vpbroadcastd(temp_vreg, lengths_R_);
            a->vcvtdq2ps(temp_vreg, temp_vreg);
            a->vdivps(vlen_inv_vreg, vlen_inv_vreg, temp_vreg);
          }
          a->bind(IfLengthsEnd);
        }

        for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
             vec_idx += unroll_factor) {
          int cur_unroll_factor =
              std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

          // Initialize output regs
          for (int v = 0; v < cur_unroll_factor; ++v) {
            vec_reg_t out_vreg = vec_reg_t(v);
            a->vxorps(out_vreg, out_vreg, out_vreg);
          }

          if (use_offsets) {
            a->mov(lengths_R_, x86::dword_ptr(lengths, sizeof(offsetType)));
            a->sub(lengths_R_, x86::dword_ptr(lengths));
          } else {
            a->mov(lengths_R_, x86::dword_ptr(lengths));
          }

          // Array out of bound check
          a->lea(
              scratchReg1_,
              x86::ptr(indices, lengths_R_, areIndices64b ? 3 : 2));
          a->cmp(scratchReg1_, index_size);
          a->jg(error);

          asmjit::Label LoopDataIndexBegin = a->newLabel();
          asmjit::Label LoopDataIndexEnd = a->newLabel();
          asmjit::Label ValidIndexLabel = a->newLabel();

          // dataIndex loop begins (iterate lengths_R_ times)
          a->bind(LoopDataIndexBegin);
          a->dec(lengths_R_);
          a->jl(LoopDataIndexEnd);

          // Array out of bound check
          if (areIndices64b) {
            a->mov(scratchReg1_, x86::qword_ptr(indices));
          } else {
            a->mov(scratchReg1_.r32(), x86::dword_ptr(indices));
          }
          if (!scale_bias_last) {
            // When scale_bias_last == false, assume this is for table batched
            // embedding (TBE) that can get -1 for pruned rows.
            if (areIndices64b) {
              a->cmp(scratchReg1_, static_cast<asmjit::Imm>(-1));
            } else {
              a->cmp(scratchReg1_.r32(), static_cast<asmjit::Imm>(-1));
            }
            a->jne(ValidIndexLabel);
            a->add(indices, static_cast<asmjit::Imm>(sizeof(indxType)));
            a->jmp(LoopDataIndexBegin);
            a->bind(ValidIndexLabel);
          }
          // A trick to check x >= data_size or x < 0 in one shot by treating
          // scratchReg1_ as if it has unsigned value
          // (https://stackoverflow.com/a/34072155).
          a->cmp(scratchReg1_, data_size);
          a->jae(error);

          if (ROWWISE_SPARSE) {
            a->mov(
                scratchReg1_.r32(),
                x86::dword_ptr(
                    compressed_indices_table,
                    scratchReg1_,
                    2)); // use of 2 is to multiply by 4
          }

          int fused_block_size = input_stride * sizeof(inType);

          if (pref_dist) {
            asmjit::Label pref_dist_reset_start = a->newLabel();
            asmjit::Label pref_dist_reset_end = a->newLabel();
            // out of bound handling for prefetch
            a->lea(
                scratchReg2_, x86::ptr(indices, pref_dist * sizeof(indxType)));
            a->cmp(scratchReg2_, index_size);
            a->jge(pref_dist_reset_start);

            if (areIndices64b) {
              a->mov(
                  scratchReg2_,
                  x86::qword_ptr(indices, pref_dist * sizeof(indxType)));
            } else {
              a->mov(
                  scratchReg2_.r32(),
                  x86::dword_ptr(indices, pref_dist * sizeof(indxType)));
            }

            a->jmp(pref_dist_reset_end);

            a->bind(pref_dist_reset_start);
            // things are not okay just get the current row
            // this can be improved to getting the max dist row.
            if (areIndices64b) {
              a->mov(scratchReg2_, x86::qword_ptr(indices));
            } else {
              a->mov(scratchReg2_.r32(), x86::dword_ptr(indices));
            }

            a->bind(pref_dist_reset_end);
            if (ROWWISE_SPARSE) {
              asmjit::Label rowwise_sparse_pref_corner_case_begin =
                  a->newLabel();
              asmjit::Label rowwise_sparse_pref_corner_case_end = a->newLabel();
              a->cmp(scratchReg2_, data_size);
              a->jae(rowwise_sparse_pref_corner_case_begin);

              a->mov(
                  scratchReg2_.r32(),
                  x86::dword_ptr(
                      compressed_indices_table,
                      scratchReg2_,
                      2)); // use of 2 is to multiply by 4
              a->test(scratchReg2_.r32(), scratchReg2_.r32());
              // Check negative
              a->jns(rowwise_sparse_pref_corner_case_end);

              a->bind(rowwise_sparse_pref_corner_case_begin);
              // For corner case, just set prefetch row id to 0.
              a->xor_(scratchReg2_.r32(), scratchReg2_.r32());
              a->bind(rowwise_sparse_pref_corner_case_end);
            }
            a->imul(scratchReg2_, static_cast<asmjit::Imm>(fused_block_size));
          }

          a->add(indices, static_cast<asmjit::Imm>(sizeof(indxType)));

          if (has_weight) {
            a->vbroadcastss(w_vreg, x86::dword_ptr(weights));
            a->add(weights, static_cast<asmjit::Imm>(sizeof(float)));
          }

          if (ROWWISE_SPARSE) {
            a->cmp(scratchReg1_.r32(), static_cast<asmjit::Imm>(-1));
            a->je(LoopDataIndexBegin);
          }

          a->imul(scratchReg1_, static_cast<asmjit::Imm>(fused_block_size));

          // broadcast the scale
          x86::Mem scale_src, bias_src;
          constexpr unsigned int CACHE_LINE_LEN = 64;
          if (is8bit) {
            if (scale_bias_last) {
              scale_src = x86::dword_ptr(
                  input, scratchReg1_, 0, block_size * sizeof(uint8_t));
              bias_src = x86::dword_ptr(
                  input,
                  scratchReg1_,
                  0,
                  block_size * sizeof(uint8_t) + sizeof(float));
              a->vbroadcastss(scale_vreg, scale_src);
              a->vbroadcastss(bias_vreg, bias_src);
            } else {
              scale_src = x86::word_ptr(input, scratchReg1_);
              bias_src =
                  x86::word_ptr(input, scratchReg1_, 0, sizeof(uint16_t));
              a->vpbroadcastw(scale_vreg.half(), scale_src);
              a->vpbroadcastw(bias_vreg.half(), bias_src);
              a->vcvtph2ps(scale_vreg, scale_vreg.half());
              a->vcvtph2ps(bias_vreg, bias_vreg.half());
            }

            if (pref_dist && fused_block_size % CACHE_LINE_LEN > 0 &&
                fused_block_size % CACHE_LINE_LEN <= 2 * sizeof(float)) {
              a->prefetcht0(x86::dword_ptr(
                  input,
                  scratchReg2_,
                  0,
                  fused_block_size / CACHE_LINE_LEN * CACHE_LINE_LEN));
            }
          }

          if (has_weight && is8bit) {
            a->vmulps(scale_vreg, scale_vreg, w_vreg);
            a->vmulps(bias_vreg, bias_vreg, w_vreg);
          }

          // The main computation
          int src_addr_offset =
              is8bit && !scale_bias_last ? 2 * sizeof(uint16_t) : 0;
          for (int v = 0; v < cur_unroll_factor; ++v) {
            constexpr int BYTES_PER_VLOAD = vlen * sizeof(inType);
            auto src_addr = x86::dword_ptr(
                input,
                scratchReg1_,
                0,
                src_addr_offset + (vec_idx + v) * BYTES_PER_VLOAD);
            vec_reg_t out_vreg = vec_reg_t(v);

            // For 8bit SLS convert usigned 8-bit to 32bit int, then to float
            // multiply with scale and then add with bias
            if (is8bit) {
              if (remainder && vec_idx + v == num_vec_regs_per_block - 1 &&
                  instSet == inst_set_t::avx512) {
                a->k(x86::k(1)).z().vpmovzxbd(src_vreg, src_addr);
              } else {
                // We don't use a mask for AVX2 since we can use the extra
                // "padding" of the 2 floats (= 8 chars) scale and bias
                // this ensures we never access out of bound data
                a->vpmovzxbd(src_vreg, src_addr);
              }
              a->vcvtdq2ps(src_vreg, src_vreg);
              a->vaddps(out_vreg, out_vreg, bias_vreg);
              a->vfmadd231ps(out_vreg, src_vreg, scale_vreg);
            } else if (is16bit) {
              if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                if (instSet == inst_set_t::avx2) {
                  if (remainder % 2 == 0) {
                    a->vmaskmovps(src_vreg.xmm(), mask_fp16_vreg, src_addr);
                  } else {
                    a->vpbroadcastw(
                        src_vreg.xmm(),
                        x86::word_ptr(
                            input,
                            scratchReg1_,
                            0,
                            src_addr_offset + (vec_idx + v) * BYTES_PER_VLOAD +
                                (remainder - 1) * sizeof(inType)));
                    if (remainder > 1) {
                      // AVX2 can't do masking for the last 16-bit so we store
                      // them to a stack and reload.
                      // First put broadcasted last 16-bit element
                      a->vmovups(x86::xmmword_ptr(x86::rsp), src_vreg.xmm());
                      // Mask store the remaining 16-bit elements
                      a->vmaskmovps(src_vreg.xmm(), mask_fp16_vreg, src_addr);
                      a->vmaskmovps(
                          x86::xmmword_ptr(x86::rsp),
                          mask_fp16_vreg,
                          src_vreg.xmm());
                      // Load combined 16-bit elements
                      a->vmovups(src_vreg.xmm(), x86::xmmword_ptr(x86::rsp));
                    } // remainder > 1
                  } // remainder % 2
                  if (isfp16) {
                    a->vcvtph2ps(src_vreg.ymm(), src_vreg.xmm());
                  } else if (isbf16) {
                    // bf16
                    a->vpmovzxwd(src_vreg.ymm(), src_vreg.xmm());
                    a->vpslld(src_vreg.ymm(), src_vreg.ymm(), 16);
                  }
                } else {
                  // avx512
                  if (isfp16) {
                    a->k(x86::k(1)).z().vcvtph2ps(src_vreg, src_addr);
                  } else if (isbf16) {
                    // bf16
                    a->k(x86::k(1)).z().vpmovzxwd(src_vreg, src_addr);
                    a->k(x86::k(1)).z().vpslld(src_vreg, src_vreg, 16);
                  }
                }
              } else {
                // no remainder
                if (isfp16) {
                  a->vcvtph2ps(src_vreg, src_addr);
                } else if (isbf16) {
                  // bf16
                  a->vpmovzxwd(src_vreg, src_addr);
                  a->vpslld(src_vreg, src_vreg, 16);
                }
              }
              if (has_weight) {
                a->vfmadd231ps(out_vreg, w_vreg, src_vreg);
              } else {
                a->vaddps(out_vreg, out_vreg, src_vreg);
              }
            } else {
              // This part for FP32 SLS
              if (remainder && vec_idx + v == num_vec_regs_per_block - 1 &&
                  instSet == inst_set_t::avx2) {
                a->vmaskmovps(src_vreg.ymm(), mask_vreg.ymm(), src_addr);
              }
              if (has_weight) {
                if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                  if (instSet == inst_set_t::avx2) {
                    a->vfmadd231ps(out_vreg, w_vreg, src_vreg);
                  } else {
                    a->k(x86::k(1)).vfmadd231ps(out_vreg, w_vreg, src_addr);
                  }
                } else {
                  a->vfmadd231ps(out_vreg, w_vreg, src_addr);
                }
              } else {
                if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                  if (instSet == inst_set_t::avx2) {
                    a->vaddps(out_vreg, out_vreg, src_vreg);
                  } else {
                    a->k(x86::k(1)).vaddps(out_vreg, out_vreg, src_addr);
                  }
                } else {
                  a->vaddps(out_vreg, out_vreg, src_addr);
                }
              }
            }

            constexpr int VLOAD_PER_CACHE_LINE =
                CACHE_LINE_LEN / BYTES_PER_VLOAD;
            if (pref_dist && (vec_idx + v) % VLOAD_PER_CACHE_LINE == 0) {
              a->prefetcht0(x86::dword_ptr(
                  input, scratchReg2_, 0, (vec_idx + v) * BYTES_PER_VLOAD));
            }
          }

          a->jmp(LoopDataIndexBegin);
          a->bind(LoopDataIndexEnd);

          // This loop is for writing back out_vreg (results)
          // back to memory
          for (int v = 0; v < cur_unroll_factor; ++v) {
            auto dst_addr =
                x86::dword_ptr(out, (vec_idx + v) * vlen * sizeof(outType));
            vec_reg_t out_vreg = vec_reg_t(v);

            if (normalize_by_lengths) {
              a->vmulps(out_vreg, out_vreg, vlen_inv_vreg);
            }

            if (std::is_same<outType, float>::value) {
              if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                if (instSet == inst_set_t::avx2) {
                  a->vmaskmovps(dst_addr, mask_vreg, out_vreg.ymm());
                } else {
                  a->k(x86::k(1)).vmovups(dst_addr, out_vreg);
                }
              } else {
                a->vmovups(dst_addr, out_vreg);
              }
            } else {
              // fp16/bf16 output
              if (instSet == inst_set_t::avx2) {
                // round nearest with no exception
                if (isfp16out) {
                  a->vcvtps2ph(out_vreg.xmm(), out_vreg, 8);
                } else if (isbf16out) {
                  a->vpaddd(out_vreg, out_vreg, ones_vreg);
                  a->vpsrld(out_vreg, out_vreg, 16);
                  a->vpackusdw(out_vreg, out_vreg, out_vreg);
                  a->vpermq(out_vreg, out_vreg, 0xd8);
                }
                if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                  if (remainder > 1) {
                    a->vmaskmovps(dst_addr, mask_fp16_vreg, out_vreg.xmm());
                  }
                  if (remainder % 2 != 0) {
                    a->vmovups(x86::xmmword_ptr(x86::rsp), out_vreg.xmm());
                    a->mov(
                        scratchReg1_.r16(),
                        x86::word_ptr(
                            x86::rsp, (remainder - 1) * sizeof(outType)));
                    a->mov(
                        x86::word_ptr(
                            out,
                            ((vec_idx + v) * vlen + (remainder - 1)) *
                                sizeof(outType)),
                        scratchReg1_.r16());
                  }
                } else {
                  a->vmovups(dst_addr, out_vreg.xmm());
                }
              } else {
                if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
                  if (isfp16out) {
                    a->k(x86::k(1)).vcvtps2ph(dst_addr, out_vreg, 8);
                  } else if (isbf16out) {
                    // bf16
                    a->k(x86::k(1)).vpaddd(out_vreg, out_vreg, ones_vreg);
                    a->k(x86::k(1)).vpsrld(out_vreg, out_vreg, 16);
                    a->k(x86::k(1)).vpmovdw(dst_addr, out_vreg);
                  }
                } else {
                  if (isfp16out) {
                    a->vcvtps2ph(dst_addr, out_vreg, 8);
                  } else if (isbf16out) {
                    // bf16
                    a->vpaddd(out_vreg, out_vreg, ones_vreg);
                    a->vpsrld(out_vreg, out_vreg, 16);
                    a->vpmovdw(dst_addr, out_vreg);
                  }
                }
              }
            }
          }

          if (vec_idx + unroll_factor < num_vec_regs_per_block ||
              (has_weight && is_weight_positional)) {
            // Reset lengths_R_, indices, weights to run the dataIndex loop
            // again
            if (use_offsets) {
              a->mov(lengths_R_, x86::dword_ptr(lengths, sizeof(offsetType)));
              a->sub(lengths_R_, x86::dword_ptr(lengths));
            } else {
              a->mov(lengths_R_, x86::dword_ptr(lengths));
            }

            if (has_weight) {
              a->imul(
                  scratchReg1_,
                  lengths_R_,
                  static_cast<asmjit::Imm>(sizeof(float)));
              a->sub(weights, scratchReg1_);

              if (vec_idx + unroll_factor < num_vec_regs_per_block) {
                a->imul(
                    scratchReg1_,
                    static_cast<asmjit::Imm>(sizeof(indxType) / sizeof(float)));
                a->sub(indices, scratchReg1_);
              }
            } else {
              a->imul(
                  scratchReg1_,
                  lengths_R_,
                  static_cast<asmjit::Imm>(sizeof(indxType)));
              a->sub(indices, scratchReg1_);
            }
          }
        }

        a->add(lengths, static_cast<asmjit::Imm>(sizeof(offsetType)));
        a->add(out, static_cast<asmjit::Imm>(output_stride * sizeof(outType)));

        a->jmp(LoopRangeIndexBegin);
        a->bind(LoopRangeIndexEnd);

        a->cmp(indices, index_size);
        a->jne(error);
        a->mov(x86::eax, true);
        a->jmp(exit);
        a->bind(error);
        a->mov(x86::eax, false);
        a->bind(exit);

        if (remainder && instSet == inst_set_t::avx2 &&
            (is16bit || isbf16out || isfp16out)) {
          a->lea(x86::rsp, x86::ymmword_ptr(x86::rsp, vlen * sizeof(int32_t)));
        }

        a->emitEpilog(frame);

        // jit_fused8bitembedding_kernel fn;
        typename ReturnFunctionSignature<
            inType,
            indxType,
            offsetType,
            outType,
            ROWWISE_SPARSE>::jit_embedding_kernel fn;
        asmjit::Error err;
        {
          std::unique_lock<std::mutex> lock(rtMutex_);
          err = runtime().add(&fn, &code);
        }
        if (err) {
          std::cout << "Error: in fn add" << std::endl;
          return nullptr;
        }

#if defined(FBGEMM_LOG_CODE)
        fclose(codeLogFile);
        delete codeLogger;
#endif
        return fn;
      });
}

} // namespace

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    bool THREAD_LOCAL>
typename EmbeddingSpMDMKernelSignature<inType, indxType, offsetType, outType>::
    Type
    GenerateEmbeddingSpMDMWithStrides(
        const int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch,
        bool is_weight_positional,
        bool use_offsets,
        int64_t output_stride /*=-1*/,
        int64_t input_stride /*=-1*/,
        bool scale_bias_last /*=true*/,
        bool no_bag /*=false*/,
        bool isbf16 /*=false*/) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if (output_stride == -1) {
    output_stride = block_size;
  }
  if (input_stride == -1) {
    if (std::is_same<inType, uint8_t>::value) {
      const auto scale_bias_offset =
          2 * (scale_bias_last ? sizeof(float) : sizeof(uint16_t));
      input_stride = block_size + scale_bias_offset;
    } else {
      input_stride = block_size;
    }
  }
  const inst_set_t isa = fbgemmInstructionSet();
  if (no_bag == true) {
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               const inType* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               outType* out) {
      return EmbeddingSpMDM_ref(
          block_size,
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          normalize_by_lengths,
          out,
          is_weight_positional,
          use_offsets,
          output_stride,
          input_stride,
          scale_bias_last,
          no_bag,
          isbf16);
    };
  }

  if ((std::is_same<inType, float>::value ||
       std::is_same<inType, uint16_t>::value) &&
      block_size == 1 && isYmm(isa) && output_stride == block_size &&
      input_stride == block_size && std::is_same<outType, float>::value) {
    return
        [=](int64_t output_size,
            int64_t index_size,
            int64_t data_size,
            const inType* input,
            const indxType* indices,
            const offsetType* offsets_or_lengths,
            const float* weights, // optional, can be null for non-weighted sum
            outType* out) {
          return internal::EmbeddingSpMDMBlockSize1_(
              output_size,
              index_size,
              data_size,
              input,
              indices,
              offsets_or_lengths,
              weights,
              normalize_by_lengths,
              reinterpret_cast<float*>(out),
              is_weight_positional,
              use_offsets,
              isbf16);
        };
  } else if (isZmm(isa)) {
    static GenEmbeddingSpMDMLookup<
        inType,
        indxType,
        offsetType,
        outType,
        inst_set_t::avx512,
        /*ROWWISE_SPARSE=*/false,
        THREAD_LOCAL>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets,
        output_stride,
        input_stride,
        scale_bias_last,
        isbf16);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               const inType* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               outType* out) {
      return original_func(
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          nullptr /* mask not used in avx512 */);
    };
  } else if (isYmm(isa)) {
    static GenEmbeddingSpMDMLookup<
        inType,
        indxType,
        offsetType,
        outType,
        inst_set_t::avx2,
        /*ROWWISE_SPARSE=*/false,
        THREAD_LOCAL>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets,
        output_stride,
        input_stride,
        scale_bias_last,
        isbf16);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               const inType* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               outType* out) {
      return original_func(
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          internal::avx2_ps_or_epi32_combined_mask);
    };
  } else {
#ifdef VLOG
    VLOG(0) << "AVX2 or AVX512 not found, taking the slow path";
#endif
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               const inType* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               outType* out) {
      return EmbeddingSpMDM_ref(
          block_size,
          output_size,
          index_size,
          data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          normalize_by_lengths,
          out,
          is_weight_positional,
          use_offsets,
          output_stride,
          input_stride,
          scale_bias_last,
          no_bag,
          isbf16);
    };
  }
}

template <
    typename inType,
    typename indxType,
    typename offsetType,
    typename outType,
    bool THREAD_LOCAL>
typename EmbeddingSpMDMKernelSignature<inType, indxType, offsetType, outType>::
    Type
    GenerateEmbeddingSpMDM(
        const int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch,
        bool is_weight_positional,
        bool use_offsets,
        bool isbf16) {
  return GenerateEmbeddingSpMDMWithStrides<
      inType,
      indxType,
      offsetType,
      outType,
      THREAD_LOCAL>(
      block_size,
      has_weight,
      normalize_by_lengths,
      prefetch,
      is_weight_positional,
      use_offsets,
      /*output_stride=*/-1,
      /*input_stride=*/-1,
      /*scale_bias_last=*/true,
      /*no_bag=*/false,
      isbf16);
}

template <typename indxType, typename offsetType, typename outType>
typename EmbeddingSpMDMKernelSignature<uint8_t, indxType, offsetType, outType>::
    Type
    GenerateEmbeddingSpMDMFP8WithStrides(
        const int64_t block_size,
        bool normalize_by_lengths,
        bool is_weight_positional,
        bool use_offsets,
        int64_t output_stride /*=-1*/,
        int64_t input_stride /*=-1*/,
        int exponent_bits,
        int exponent_bias) {
  if (output_stride == -1) {
    output_stride = block_size;
  }
  if (input_stride == -1) {
    input_stride = block_size;
  }
  // There is only the reference implementation for FP8 embedding
  return [=](int64_t output_size,
             int64_t index_size,
             int64_t data_size,
             const uint8_t* input,
             const indxType* indices,
             const offsetType* offsets_or_lengths,
             const float* weights,
             outType* out) {
    return EmbeddingSpMDMFP8_ref(
        block_size,
        output_size,
        index_size,
        data_size,
        input,
        indices,
        offsets_or_lengths,
        weights,
        normalize_by_lengths,
        out,
        is_weight_positional,
        use_offsets,
        output_stride,
        input_stride,
        exponent_bits,
        exponent_bias);
  };
}

template <typename inType, typename indxType, typename offsetType>
typename EmbeddingSpMDMRowWiseSparseKernelSignature<
    inType,
    indxType,
    offsetType>::Type
GenerateEmbeddingSpMDMRowWiseSparse(
    const int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional,
    bool use_offsets) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  int64_t input_stride = block_size;
  if (std::is_same<inType, uint8_t>::value) {
    const auto scale_bias_offset = 2 * sizeof(float);
    input_stride = block_size + scale_bias_offset;
  }
  inst_set_t isa = fbgemmInstructionSet();
  if (isZmm(isa)) {
    static GenEmbeddingSpMDMLookup<
        inType,
        indxType,
        offsetType,
        /*outType=*/float,
        inst_set_t::avx512,
        /*rowwise_sparse=*/true>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets,
        /*output_stride=*/block_size,
        input_stride,
        /*scale_bias_last=*/true,
        /*isbf16=*/false);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t uncompressed_data_size,
               const inType* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out,
               const int32_t* compressed_indices_table) {
      return original_func(
          output_size,
          index_size,
          uncompressed_data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          compressed_indices_table,
          nullptr /* mask not used in avx512 */);
    };
  } else if (isYmm(isa)) {
    static GenEmbeddingSpMDMLookup<
        inType,
        indxType,
        offsetType,
        /*outType=*/float,
        inst_set_t::avx2,
        /*rowwise_sparse=*/true>
        kernel_generator;
    const auto original_func = kernel_generator.getOrCreate(
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch,
        use_offsets,
        /*output_stride=*/block_size,
        input_stride,
        /*scale_bias_last=*/true,
        /*isbf16=*/false);
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t uncompressed_data_size,
               const inType* input,
               const indxType* indices,
               const offsetType* offsets_or_lengths,
               const float* weights,
               float* out,
               const int32_t* compressed_indices_table) {
      return original_func(
          output_size,
          index_size,
          uncompressed_data_size,
          input,
          indices,
          offsets_or_lengths,
          weights,
          out,
          compressed_indices_table,
          internal::avx2_ps_or_epi32_combined_mask);
    };
  } else {
#ifdef VLOG
    VLOG(0) << "AVX2 or AVX512 not found, taking the slow path";
#endif
    return
        [=](int64_t output_size,
            int64_t index_size,
            int64_t uncompressed_data_size,
            const inType* input,
            const indxType* indices,
            const offsetType* offsets_or_lengths,
            const float* weights, // optional, can be null for non-weighted sum
            float* out,
            const int32_t* compressed_indices_table) {
          return EmbeddingSpMDMRowWiseSparse_ref(
              block_size,
              output_size,
              index_size,
              uncompressed_data_size,
              // compressed_data_size,
              input,
              indices,
              compressed_indices_table,
              offsets_or_lengths,
              weights,
              normalize_by_lengths,
              out,
              is_weight_positional,
              use_offsets);
        };
  }
}

#define INSTANTIATE_SPMDM_BASE(                               \
    IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE, THREAD_LOCAL) \
  template FBGEMM_API typename EmbeddingSpMDMKernelSignature< \
      IN_TYPE,                                                \
      INDEX_TYPE,                                             \
      OFFSET_TYPE,                                            \
      OUT_TYPE>::Type                                         \
  GenerateEmbeddingSpMDMWithStrides<                          \
      IN_TYPE,                                                \
      INDEX_TYPE,                                             \
      OFFSET_TYPE,                                            \
      OUT_TYPE,                                               \
      THREAD_LOCAL>(                                          \
      const int64_t block_size,                               \
      bool has_weight,                                        \
      bool normalize_by_lengths,                              \
      int prefetch,                                           \
      bool is_weight_positional,                              \
      bool use_offsets,                                       \
      int64_t output_stride,                                  \
      int64_t input_stride,                                   \
      bool scale_bias_last,                                   \
      bool no_bag,                                            \
      bool isbf16);

#define INSTANTIATE_SPMDMFP8_BASE(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE)       \
  template FBGEMM_API typename EmbeddingSpMDMKernelSignature<              \
      uint8_t,                                                             \
      INDEX_TYPE,                                                          \
      OFFSET_TYPE,                                                         \
      OUT_TYPE>::Type                                                      \
  GenerateEmbeddingSpMDMFP8WithStrides<INDEX_TYPE, OFFSET_TYPE, OUT_TYPE>( \
      const int64_t block_size,                                            \
      bool normalize_by_lengths,                                           \
      bool is_weight_positional,                                           \
      bool use_offsets,                                                    \
      int64_t output_stride,                                               \
      int64_t input_stride,                                                \
      int exponent_bits,                                                   \
      int exponent_bias);

#define INSTANTIATE_SPMDM_NOSTRIDE_BASE(                      \
    IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE, THREAD_LOCAL) \
  template FBGEMM_API typename EmbeddingSpMDMKernelSignature< \
      IN_TYPE,                                                \
      INDEX_TYPE,                                             \
      OFFSET_TYPE,                                            \
      OUT_TYPE>::Type                                         \
  GenerateEmbeddingSpMDM<                                     \
      IN_TYPE,                                                \
      INDEX_TYPE,                                             \
      OFFSET_TYPE,                                            \
      OUT_TYPE,                                               \
      THREAD_LOCAL>(                                          \
      const int64_t block_size,                               \
      bool has_weight,                                        \
      bool normalize_by_lengths,                              \
      int prefetch,                                           \
      bool is_weight_positional,                              \
      bool use_offsets,                                       \
      bool isbf16);

#define INSTANTIATE_SPMDM_ROWWISE_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE)   \
  template FBGEMM_API typename EmbeddingSpMDMRowWiseSparseKernelSignature< \
      IN_TYPE,                                                             \
      INDEX_TYPE,                                                          \
      OFFSET_TYPE>::Type                                                   \
  GenerateEmbeddingSpMDMRowWiseSparse<IN_TYPE, INDEX_TYPE, OFFSET_TYPE>(   \
      const int64_t block_size,                                            \
      bool has_weight,                                                     \
      bool normalize_by_lengths,                                           \
      int prefetch,                                                        \
      bool is_weight_positional,                                           \
      bool use_offsets);

#define INSTANTIATE_SPMDMFP8_BASE_uint8_t(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE) \
  INSTANTIATE_SPMDMFP8_BASE(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE)
#define INSTANTIATE_SPMDMFP8_BASE_float(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE)
#define INSTANTIATE_SPMDMFP8_BASE_uint16_t(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE)

#define INSTANTIATE_SPMDM_THREAD_LOCAL(                                     \
    IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE)                             \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE, true)  \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE, false) \
  INSTANTIATE_SPMDM_NOSTRIDE_BASE(                                          \
      IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE, true)                     \
  INSTANTIATE_SPMDM_NOSTRIDE_BASE(                                          \
      IN_TYPE, INDEX_TYPE, OFFSET_TYPE, OUT_TYPE, false)                    \
  INSTANTIATE_SPMDMFP8_BASE_##IN_TYPE(INDEX_TYPE, OFFSET_TYPE, OUT_TYPE)

#define INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, OFFSET_TYPE)            \
  INSTANTIATE_SPMDM_THREAD_LOCAL(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, float)    \
  INSTANTIATE_SPMDM_THREAD_LOCAL(IN_TYPE, INDEX_TYPE, OFFSET_TYPE, uint16_t) \
  INSTANTIATE_SPMDM_ROWWISE_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE)

#define INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, INDEX_TYPE) \
  INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, int32_t) \
  INSTANTIATE_SPMDM_OUT_T(IN_TYPE, INDEX_TYPE, int64_t)

#define INSTANTIATE_SPMDM_INDEX_T(IN_TYPE)     \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, int32_t) \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, int64_t)

INSTANTIATE_SPMDM_INDEX_T(float)
INSTANTIATE_SPMDM_INDEX_T(uint16_t)
INSTANTIATE_SPMDM_INDEX_T(uint8_t)

#undef INSTANTIATE_SPMDM_INDEX_T
#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_OUT_T
#undef INSTANTIATE_SPMDM_THREAD_LOCAL
#undef INSTANTIATE_SPMDM_BASE
#undef INSTANTIATE_SPMDMFP8_BASE
#undef INSTANTIATE_SPMDM_NOSTRIDE_BASE
#undef INSTANTIATE_SPMDM_ROWWISE_BASE

template <typename IndexType>
void compressed_indices_remap(
    std::int32_t offsets_len,
    const IndexType* indices,
    const int32_t* compressed_indices_mapping,
    const IndexType* offsets,
    const float* weights, // optional, can be null,
    IndexType* out_indices,
    IndexType* out_offsets,
    float* out_weights) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  const inst_set_t isa = fbgemmInstructionSet();
  if (isZmm(isa)) {
#ifndef __HIP_PLATFORM_HCC__
    if (weights == nullptr) {
      internal::compressed_indices_remap_avx512<IndexType, false>(
          offsets_len,
          indices,
          compressed_indices_mapping,
          offsets,
          weights,
          out_indices,
          out_offsets,
          out_weights);
    } else {
      internal::compressed_indices_remap_avx512<IndexType, true>(
          offsets_len,
          indices,
          compressed_indices_mapping,
          offsets,
          weights,
          out_indices,
          out_offsets,
          out_weights);
    }
#endif
  } else {
    compressed_indices_remap_ref<IndexType>(
        offsets_len,
        indices,
        compressed_indices_mapping,
        offsets,
        weights,
        out_indices,
        out_offsets,
        out_weights);
  }
}

#define INSTANTIATE_REMAP_BASE(INDEX_TYPE)           \
  template FBGEMM_API void compressed_indices_remap( \
      std::int32_t offsets_numel,                    \
      const INDEX_TYPE* indices,                     \
      const int32_t* compressed_indices_mapping,     \
      const INDEX_TYPE* offsets,                     \
      const float* weights,                          \
      INDEX_TYPE* out_indices,                       \
      INDEX_TYPE* out_offsets,                       \
      float* out_weights);

INSTANTIATE_REMAP_BASE(int32_t)
INSTANTIATE_REMAP_BASE(int64_t)

#undef INSTANTIATE_REMAP_BASE

} // namespace fbgemm
