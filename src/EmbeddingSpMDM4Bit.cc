/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS

#include "fbgemm/FbgemmEmbedding.h"

#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include "./CodeCache.h"
#include "./RefImplementations.h"
#include "fbgemm/Types.h"

namespace fbgemm {

namespace {

template <typename T>
T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

namespace x86 = asmjit::x86;

template <typename indxType = std::int64_t>
class ReturnFunctionSignature {
 public:
  using jit_embedding_kernel = bool (*)(
      std::int64_t output_size,
      std::int64_t index_size,
      std::int64_t data_size,
      const std::uint8_t* input,
      const indxType* indices,
      const int* lengths,
      const float* weights,
      float* out);
};

template <typename indxType = std::int64_t>
class GenEmbeddingSpMDM4BitLookup {
 public:
  GenEmbeddingSpMDM4BitLookup() {}
  typename ReturnFunctionSignature<indxType>::jit_embedding_kernel getOrCreate(
      int block_size,
      bool has_weight,
      bool is_weight_positional,
      bool normalize_by_lengths,
      int prefetch);

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
  // positional weights, normalize by lenths, and prefetch distance, is8bit
  static CodeCache<
      std::tuple<int, bool, bool, bool, int>,
      typename ReturnFunctionSignature<indxType>::jit_embedding_kernel>
      codeCache_; ///< JIT Code Cache for reuse.

  // These are the registers shared
  // between uint8 and fp32 implementations
  x86::Gp output_size;
  x86::Gp index_size;
  x86::Gp data_size;
  x86::Gp input;
  x86::Gp indices;
  x86::Gp lengths;
  x86::Gp weights;
  x86::Gp out;
  x86::Gp scratchReg1_;
  x86::Gp scratchReg1D_;
  x86::Gp scratchReg2_;
  x86::Gp scratchReg3_;
  x86::Gpd lengths_R_;
}; // GenEmbeddingSpmDMLookup

template <typename indxType>
std::mutex GenEmbeddingSpMDM4BitLookup<indxType>::rtMutex_;

template <typename indxType>
CodeCache<
    std::tuple<int, bool, bool, bool, int>,
    typename ReturnFunctionSignature<indxType>::jit_embedding_kernel>
    GenEmbeddingSpMDM4BitLookup<indxType>::codeCache_;

template <typename indxType>
typename ReturnFunctionSignature<indxType>::jit_embedding_kernel
GenEmbeddingSpMDM4BitLookup<indxType>::getOrCreate(
    int block_size,
    bool has_weight,
    bool is_weight_positional,
    bool normalize_by_lengths,
    int prefetch) {
  std::tuple<int, bool, bool, bool, int> kernelSig = std::make_tuple(
      block_size,
      has_weight,
      is_weight_positional,
      normalize_by_lengths,
      prefetch);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() ->
      typename ReturnFunctionSignature<indxType>::jit_embedding_kernel {
        constexpr inst_set_t instSet = inst_set_t::avx512;

        // TODO: Make this tunable
        int pref_dist = prefetch;
        bool areIndices64b = std::is_same<indxType, std::int64_t>::value;

        asmjit::CodeHolder code;
        code.init(runtime().codeInfo());
        x86::Assembler assembler(&code);
        x86::Emitter* a = assembler.as<x86::Emitter>();
#if defined(FBGEMM_LOG_CODE)
        std::string filename = "embeddinglookup_4bit_";
        filename += "_emd_dim_" + std::to_string(block_size);
        filename += areIndices64b ? "_64bit" : "_32bit";
        filename += "_avx512";
        if (prefetch) {
          filename += "_prefetch";
        }
        if (has_weight) {
          filename += "_hasweight";
        }
        if (normalize_by_lengths) {
          filename += "_normalize_by_lengths";
        }
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif
        // arguments to the function created
        output_size = a->zdi();
        // index_size will be overwritten to hold the end address of indices
        index_size = a->zsi();
        data_size = a->zdx();
        input = a->zcx();
        indices = a->gpz(8);
        lengths = a->gpz(9);
        weights = a->gpz(10);
        out = a->gpz(11);
        lengths_R_ = a->gpz(12).r32();

        scratchReg1_ = a->gpz(13);
        scratchReg1D_ = a->gpz(13).r32();
        scratchReg2_ = a->gpz(14);
        scratchReg3_ = a->gpz(15);

        asmjit::FuncDetail func;

        func.init(asmjit::FuncSignatureT<
                  bool,
                  std::int64_t, // output_size
                  std::int64_t, // index_size
                  std::int64_t, // data_size
                  const uint8_t*, // input uint8_t or float
                  const indxType*, // indices
                  const int*, // lengths
                  const float*, // weights
                  float*>(asmjit::CallConv::kIdHost));

        asmjit::FuncFrame frame;
        frame.init(func);

        frame.setDirtyRegs(
            x86::Reg::kGroupVec,
            asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15) |
                asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
                asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31));

        frame.setDirtyRegs(
            x86::Reg::kGroupGp,
            asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));

        asmjit::FuncArgsAssignment args(&func);
        args.assignAll(
            output_size,
            index_size,
            data_size,
            input,
            indices,
            lengths,
            weights,
            out);

        args.updateFuncFrame(frame);
        frame.finalize();

        a->emitProlog(frame);
        a->emitArgsAssignment(frame, args);

        constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
        constexpr int NUM_VEC_REG = simd_info<instSet>::NUM_VEC_REGS;
        int unroll_factor = NUM_VEC_REG;

        typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

        int num_vec_regs_per_block = ceil_div(block_size, vlen);
        int remainder = block_size % vlen;

        vec_reg_t scale_vreg; // holds scale
        vec_reg_t bias_vreg; // holds bias
        vec_reg_t w_vreg; // for weighted sls -- weights
        vec_reg_t
            vlen_inv_vreg; // used for normalize by lengths -- 1/ lengths[i]
        vec_reg_t src_vreg; // for holding embedding value temporarily

        // We need 2 vec registers for 1. scale 2. bias
        --unroll_factor;
        scale_vreg = vec_reg_t(unroll_factor);
        --unroll_factor;
        bias_vreg = vec_reg_t(unroll_factor);

        --unroll_factor;
        src_vreg = vec_reg_t(unroll_factor);
        // temporary register for bit manipulation instructions
        --unroll_factor;
        vec_reg_t temp_vreg = vec_reg_t(unroll_factor);

        // Create a mask that extracts lower 4 bits from each 8-bit block
        --unroll_factor;
        vec_reg_t mask_vreg = vec_reg_t(unroll_factor);
        a->lea(
            x86::rsp,
            x86::dword_ptr(x86::rsp, -1 * static_cast<int>(sizeof(int32_t))));
        a->mov(x86::word_ptr(x86::rsp), 0x0f0f);
        a->vpbroadcastw(mask_vreg, x86::word_ptr(x86::rsp));
        a->lea(x86::rsp, x86::dword_ptr(x86::rsp, sizeof(int32_t)));

        if (has_weight) {
          --unroll_factor;
          w_vreg = vec_reg_t(unroll_factor);
        }

        if (normalize_by_lengths) {
          --unroll_factor;
          vlen_inv_vreg = vec_reg_t(unroll_factor);
        }

        // Make unroll_factor a multiple of 4
        unroll_factor = unroll_factor / 4 * 4;

        if (remainder) {
          a->mov(scratchReg1_, (1 << remainder) - 1);
          a->kmovw(x86::k(1), scratchReg1_);
        }
        // Creating a mask for vector load
        // Since every row is followed by 2 fp16 (scale and bias), luckily
        // we don't need mask at 4-bit granularity but just at 32-bit
        // granularity.
        int num_elem_per_32bit = 32 / 4;
        // multiply by 4 because we're handling 4 vlen per iteration
        int num_of_32bit_per_vload = vlen * 4 / num_elem_per_32bit;
        int remainder_32bit_granularity =
            ceil_div(block_size, num_elem_per_32bit) % num_of_32bit_per_vload;
        if (remainder_32bit_granularity) {
          a->mov(scratchReg1_, (1 << remainder_32bit_granularity) - 1);
          a->kmovw(x86::k(2), scratchReg1_);
        }

        // Compute the end address of indices
        a->imul(
            scratchReg1_,
            index_size,
            static_cast<asmjit::Imm>(sizeof(indxType)));
        a->add(scratchReg1_, indices);
        a->mov(index_size, scratchReg1_);

        asmjit::Label exit = a->newLabel();
        asmjit::Label error = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        if (has_weight && is_weight_positional) {
          a->mov(scratchReg3_, weights);
        }

        // rangeIndex loop begins (iterate output_size times)
        a->bind(LoopRangeIndexBegin);
        a->dec(output_size);
        a->jl(LoopRangeIndexEnd);

        if (normalize_by_lengths) {
          asmjit::Label IfLengthsBegin = a->newLabel();
          asmjit::Label IfLengthsEnd = a->newLabel();
          a->bind(IfLengthsBegin);
          a->cmp(x86::dword_ptr(lengths), 1);
          // Initialize vlen_inv as 0 in case lengths is 0
          a->vxorps(vlen_inv_vreg, vlen_inv_vreg, vlen_inv_vreg);
          a->jl(IfLengthsEnd);

          vec_reg_t temp_zmm = vec_reg_t(0);
          a->mov(lengths_R_, 1);
          a->cvtsi2ss(x86::xmm(temp_zmm.id()), lengths_R_);
          a->vpbroadcastd(vlen_inv_vreg, x86::xmm(temp_zmm.id()));
          a->vpbroadcastd(temp_zmm, x86::dword_ptr(lengths));
          a->vcvtdq2ps(temp_zmm, temp_zmm);
          a->vdivps(vlen_inv_vreg, vlen_inv_vreg, temp_zmm);
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

          a->mov(lengths_R_, x86::dword_ptr(lengths));

          // Array out of bound check
          a->imul(
              scratchReg1_,
              lengths_R_,
              static_cast<asmjit::Imm>(sizeof(indxType)));

          a->add(scratchReg1_, indices);
          a->cmp(scratchReg1_, index_size);
          a->jg(error);

          asmjit::Label LoopDataIndexBegin = a->newLabel();
          asmjit::Label LoopDataIndexEnd = a->newLabel();

          if (has_weight && is_weight_positional) {
            a->mov(weights, scratchReg3_);
          }
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
          a->cmp(scratchReg1_, 0);
          a->jl(error);
          a->cmp(scratchReg1_, data_size);
          a->jge(error);

          int fused_block_size = ceil_div(block_size, 2) + 2 * sizeof(float16);
          a->imul(scratchReg1_, static_cast<asmjit::Imm>(fused_block_size));

          if (pref_dist) {
            asmjit::Label pref_dist_reset_start = a->newLabel();
            asmjit::Label pref_dist_reset_end = a->newLabel();
            // out of bound handling for prefetch
            a->mov(scratchReg2_, indices);
            a->add(
                scratchReg2_,
                static_cast<asmjit::Imm>(pref_dist * sizeof(indxType)));
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

            a->cmp(scratchReg2_, 0);
            a->jl(pref_dist_reset_start);
            a->cmp(scratchReg2_, data_size);
            a->jge(pref_dist_reset_start);

            // everything is okay, prefetch a few rows ahead
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
            // This has to be fused_block_size
            a->imul(scratchReg2_, static_cast<asmjit::Imm>(fused_block_size));
          }

          a->add(indices, static_cast<asmjit::Imm>(sizeof(indxType)));

          // broadcast the scale
          x86::Mem scale_src, bias_src;
          scale_src =
              x86::word_ptr(input, scratchReg1_, 0, ceil_div(block_size, 2));
          bias_src = x86::word_ptr(
              input,
              scratchReg1_,
              0,
              ceil_div(block_size, 2) + sizeof(float16));
          a->vpbroadcastw(x86::Ymm(scale_vreg.id()), scale_src);
          a->vpbroadcastw(x86::Ymm(bias_vreg.id()), bias_src);
          a->vcvtph2ps(scale_vreg, x86::Ymm(scale_vreg.id()));
          a->vcvtph2ps(bias_vreg, x86::Ymm(bias_vreg.id()));

          if (has_weight) {
            a->vbroadcastss(w_vreg, x86::dword_ptr(weights));
            a->vmulps(scale_vreg, scale_vreg, w_vreg);
            a->vmulps(bias_vreg, bias_vreg, w_vreg);
            a->add(weights, static_cast<asmjit::Imm>(sizeof(float)));
          }

          // The main computation
          // Handling 4 vector registers per iteration because we get zmm from
          // ymm load via vpmovzxbw (epu8->epi16), and then get 4 zmms from
          // each 128-bit portion of zmm via vpmovsxbd (epi8->epi32).
          for (int v = 0; v < cur_unroll_factor; v += 4) {
            // Divide by 2 because we're doing ymm load rather than zmm
            constexpr int BYTES_PER_VLOAD = (vlen / 2) * sizeof(uint8_t);
            auto src_addr = x86::dword_ptr(
                input, scratchReg1_, 0, (vec_idx + v) * BYTES_PER_VLOAD);

            if (num_vec_regs_per_block - (vec_idx + v) < 4 &&
                remainder_32bit_granularity) {
              a->k(x86::k(2)).vmovups(x86::Ymm(src_vreg.id()), src_addr);
              a->vpmovzxbw(src_vreg, x86::Ymm(src_vreg.id()));
            } else {
              a->vpmovzxbw(src_vreg, src_addr);
            }
            a->vpslld(temp_vreg, src_vreg, asmjit::Imm(4));
            a->vpord(src_vreg, src_vreg, temp_vreg);
            a->vpandd(src_vreg, src_vreg, mask_vreg);

            for (int i = 0;
                 i < std::min(4, num_vec_regs_per_block - (vec_idx + v));
                 ++i) {
              vec_reg_t out_vreg = vec_reg_t(v + i);
              if (i == 0) {
                a->vpmovsxbd(temp_vreg, x86::Xmm(src_vreg.id()));
              } else {
                // We could've used avx512_ymm for clock frequency advantage,
                // if there's an instruction to extract a 64-bit portion from
                // a YMM as an XMM register.
                a->vextracti32x4(
                    x86::Xmm(temp_vreg.id()), src_vreg, asmjit::Imm(i));
                a->vpmovsxbd(temp_vreg, x86::Xmm(temp_vreg.id()));
              }
              a->vcvtdq2ps(temp_vreg, temp_vreg);
              a->vaddps(out_vreg, out_vreg, bias_vreg);
              a->vfmadd231ps(out_vreg, temp_vreg, scale_vreg);
            }

            constexpr int CACHE_LINE_LEN = 64;
            constexpr int VLOAD_PER_CACHE_LINE =
                CACHE_LINE_LEN / BYTES_PER_VLOAD;
            int v_aligned = ceil_div(vec_idx + v, 4) * 4;
            if (pref_dist && v_aligned * 4 % VLOAD_PER_CACHE_LINE == 0) {
              a->prefetcht0(x86::dword_ptr(
                  input, scratchReg2_, 0, v_aligned * BYTES_PER_VLOAD));
            }
          }

          a->jmp(LoopDataIndexBegin);
          a->bind(LoopDataIndexEnd);

          // This loop is for writing back out_vreg (results)
          // back to memory
          for (int v = 0; v < cur_unroll_factor; ++v) {
            auto dst_addr =
                x86::dword_ptr(out, (vec_idx + v) * vlen * sizeof(float));
            vec_reg_t out_vreg = vec_reg_t(v);

            if (normalize_by_lengths) {
              a->vmulps(out_vreg, out_vreg, vlen_inv_vreg);
            }

            if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
              a->k(x86::k(1)).vmovups(dst_addr, out_vreg);
            } else {
              a->vmovups(dst_addr, out_vreg);
            }
          }

          if (vec_idx + unroll_factor < num_vec_regs_per_block) {
            // Reset lengths_R_, indices, weights to run the dataIndex loop
            // again
            a->mov(lengths_R_, x86::dword_ptr(lengths));

            if (has_weight) {
              a->imul(
                  scratchReg1_,
                  lengths_R_,
                  static_cast<asmjit::Imm>(sizeof(float)));
              a->sub(weights, scratchReg1_);
              a->imul(
                  scratchReg1_,
                  static_cast<asmjit::Imm>(sizeof(indxType) / sizeof(float)));
              a->sub(indices, scratchReg1_);
            } else {
              a->imul(
                  scratchReg1_,
                  lengths_R_,
                  static_cast<asmjit::Imm>(sizeof(indxType)));
              a->sub(indices, scratchReg1_);
            }
          }
        }

        a->add(lengths, static_cast<asmjit::Imm>(sizeof(int)));
        a->add(out, static_cast<asmjit::Imm>(block_size * sizeof(float)));

        a->jmp(LoopRangeIndexBegin);
        a->bind(LoopRangeIndexEnd);

        a->cmp(indices, index_size);
        a->jne(error);
        a->mov(x86::eax, true);
        a->jmp(exit);
        a->bind(error);
        a->mov(x86::eax, false);
        a->bind(exit);

        a->emitEpilog(frame);

        // jit_fused8bitembedding_kernel fn;
        typename ReturnFunctionSignature<indxType>::jit_embedding_kernel fn;
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

template <typename indxType>
typename EmbeddingSpMDMKernelSignature<std::uint8_t, indxType>::Type
GenerateEmbeddingSpMDM4Bit(
    const std::int64_t block_size,
    bool has_weight,
    bool normalize_by_lengths,
    int prefetch,
    bool is_weight_positional) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  if (fbgemmHasAvx512Support()) {
    static GenEmbeddingSpMDM4BitLookup<indxType> kernel_generator;
    return kernel_generator.getOrCreate(
        block_size,
        has_weight,
        is_weight_positional,
        normalize_by_lengths,
        prefetch);
  } else {
#ifdef VLOG
    VLOG(0) << "AVX2 or AVX512 not found, taking the slow path";
#endif
    return
        [=](std::int64_t output_size,
            std::int64_t index_size,
            std::int64_t data_size,
            const std::uint8_t* input,
            const indxType* indices,
            const int* lengths,
            const float* weights, // optional, can be null for non-weighted sum
            float* out) {
          return EmbeddingSpMDM4Bit_ref(
              block_size,
              output_size,
              index_size,
              data_size,
              input,
              indices,
              lengths,
              weights,
              normalize_by_lengths,
              out,
              is_weight_positional);
        };
  }
}

template <typename indxType>
bool EmbeddingSpMDM4Bit(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const std::uint8_t* input,
    const indxType* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    int prefetch,
    bool is_weight_positional) {
  auto fn = GenerateEmbeddingSpMDM4Bit<indxType>(
      block_size,
      weights != nullptr,
      normalize_by_lengths,
      prefetch,
      is_weight_positional);
  return fn(
      output_size,
      index_size,
      data_size,
      input,
      indices,
      lengths,
      weights,
      out);
}

template
    typename EmbeddingSpMDMKernelSignature<std::uint8_t, std::int64_t>::Type
    GenerateEmbeddingSpMDM4Bit<std::int64_t>(
        const std::int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch,
        bool is_weight_positional);

template
    typename EmbeddingSpMDMKernelSignature<std::uint8_t, std::int32_t>::Type
    GenerateEmbeddingSpMDM4Bit<std::int32_t>(
        const std::int64_t block_size,
        bool has_weight,
        bool normalize_by_lengths,
        int prefetch,
        bool is_weight_positional);

template bool EmbeddingSpMDM4Bit(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const std::uint8_t* input,
    const std::int64_t* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    int prefetch,
    bool is_weight_positional);

template bool EmbeddingSpMDM4Bit(
    const std::int64_t block_size,
    const std::int64_t output_size,
    const std::int64_t index_size,
    const std::int64_t data_size,
    const std::uint8_t* input,
    const std::int32_t* indices,
    const int* lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    int prefetch,
    bool is_weight_positional);

} // namespace fbgemm
