/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include "./CodeGenHelpers.h"
#include "./DirectConv.h"

namespace fbgemm {

namespace x86 = asmjit::x86;
/**
 * Generate AVX256 instructions for computing block in the rank-k update of
 * 32-bit Accumulation kernel.
 *
 * this compute block implements the following register blocking
        // register blocking:
        // leverage vpmaddubsw instructions
        for (int _icb = icb; _icb < icb + row_interleave; _icb ++ ) {
          for (int _oc = oc; _oc < oc + mRegBLockSize; _oc ++) {
            for (int _ow = ow; _ow < std::min(ow + 12, OUT_DIM[1]); _ow ++) {
              out[_oc + _ow * OC] +=
              input[_ich + (_ow + s * stride[1]) * IC + r * IC * IN_DIM[1]]
              *
              weights[(((((_oc/8) * (IC/4) + icb/4) * K[0] + r) * K[1] + s)
                    *8 + (_oc % 8)) * 4 + (_icb % 4)];
            }
          }
        }
 *
 */

/**
 * Generate AVX256 instructions for storing the C registers back to the memory
 * in 32-bit Accumulation kernel.
 */
template <>
template <inst_set_t instSet>
void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::storeCRegs(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_Offset,
    x86::Gp ldcReg,
    bool accum) {
  using VecT = typename simd_info<instSet>::vec_reg_t;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  for (int i = 0; i < rowRegs; ++i) {
    if (i != 0) {
      a->add(C_Offset, ldcReg);
    } else {
      a->xor_(C_Offset.r32(), C_Offset.r32());
    }
    for (int j = 0; j < colRegs; ++j) {
      if (accum) {
        a->vpaddd(
            VecT(i * colRegs + j),
            VecT(i * colRegs + j),
            x86::dword_ptr(
                a->zcx(), C_Offset, 0, j * vectorLen * sizeof(int8_t)));
      }
      a->vmovups(
          x86::dword_ptr(a->zcx(), C_Offset, 0, j * vectorLen * sizeof(int8_t)),
          VecT(i * colRegs + j));
    }
  }
}

template <>
template <inst_set_t instSet>
void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    genComputeBlockDirectConv(
        x86::Emitter* a,
        x86::Gp buffer_A,
        x86::Gp buffer_B,
        x86::Gp /*B_pf*/,
        int rowRegs,
        int colRegs,
        int strideXich) {
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;

  // used for matrix A
  VecRegT AReg(numRegs - 1);

  // used for matrix B
  VecRegT BReg(numRegs - 2);

  // Contains 16-bit 1s
  VecRegT oneReg(numRegs - 3);

  // temporary register
  VecRegT res1(numRegs - 4);

  for (int j = 0; j < colRegs; ++j) {
    // load B
    emitLoadDWord<instSet, VecRegT>(
        a, BReg, x86::dword_ptr(buffer_B, j * vectorLen * sizeof(int8_t)));
    // load A, broadcast and fmas
    for (int i = 0; i < rowRegs; ++i) {
      a->vpbroadcastd(
          AReg, x86::dword_ptr(buffer_A, (i * strideXich) * sizeof(uint8_t)));
      a->vpmaddubsw(res1, AReg, BReg);
      a->vpmaddwd(res1, oneReg, res1);
      a->vpaddd(VecRegT(i * colRegs + j), res1, VecRegT(i * colRegs + j));
    }
    // a->prefetcht0(x86::dword_ptr(B_pf, j * vectorLen * sizeof(int8_t)));
  }
}

/**
 * Get or Create the AVX256 instructions for 32-bit Accumulation macro-kernel.
 *
 * This function implements a direct convolution kernel that is specialized
 * for kernel size (2, 6) and input_height (IN_DIM[0]) = 2.
 *
 * More specifically the implementation has the following requirements:
 * * Weights has layout {OC/8, KH, KW, IC/4, 8, 4}
 * * kernel size (2, 6), IN_DIM[0] = 2, therefore: OUT_DIM[0] = 1
 * * Features are in channel last format
 *
 * mRegBlockSize = 12: the number of avx2 registers for output
 * nRegBlockSize = 8: the # of output elements in one avx2 register
 * row_interleave = 4: the horizontal reduction size for vpmaddubsw instruction
 * O1: output_width: OUT_DIM[1]
 * i1Xich: input_width multiply input_channel: IN_DIM[1] x IC
 * strideXich: stride multiply input_channel: stride[1] x input_channel
 *
 *
 * The kernel implements the following algorithm:

for (int ow = 0; ow < OUT_DIM[1]; ow+=12) {
   L1 blocking: following weights are in L1 cache
     for (int s = 0; s < K[1]; ++s) {
       for (int r = 0; r < K[0]; ++r) {
        for (int icb = 0; icb < IC; icb+=row_interleave) {

        // register blocking:
        // leverage vpmaddubsw instructions
        for (int _icb = icb; _icb < icb + row_interleave; _icb ++ ) {
          for (int _oc = oc; _oc < oc + mRegBLockSize; _oc ++) {
            for (int _ow = ow; _ow < std::min(ow + 12, OUT_DIM[1]); _ow ++) {
              out[_oc + _ow * OC] +=
              input[_ich + (_ow + s * stride[1]) * IC + r * IC * IN_DIM[1]]
              *
              weights[(((((_oc/8) * (IC/4) + icb/4) * K[0] + r) * K[1] + s)
                    *8 + (_oc % 8)) * 4 + (_icb % 4)];

              // If we get rid of the brackets, and substitute corrresponding
variables
              //
              // input[_ich + _ow * IC + s * strideXich + r * i1Xich]
              // *
              // weights[(((((_oc/8) * (IC/4) + icb/4) * K[0] + r) * K[1] + s)
              //       *8 + (_oc % 8)) * 4 + (_icb % 4)];
            }
          }
        }

      }
    }
  }
 *
 */
template <>
template <inst_set_t instSet>
DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::jit_micro_kernel_fp
DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::getOrCreateDirectConv(
    bool accum,
    int32_t O1,
    int32_t i1Xich,
    int32_t strideXich) {
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  std::tuple<bool, int, int, int, int, int, int> kernelSig;
  // int ichSize = 32;
  int mRegBlockSize = 12;
  int nRegBlockSize = 8;
  // int nRegBlockSizeMin;
  int row_interleave = 4;

  kernelSig = std::make_tuple(
      accum, O1, i1Xich, strideXich, i1Xich, mRegBlockSize, nRegBlockSize);

  return codeCache_.getOrCreate(kernelSig, [&]() -> jit_micro_kernel_fp {
    asmjit::CodeHolder code;
    code.init(runtime().environment());
    x86::Assembler assembler(&code);
    x86::Emitter* a = assembler.as<x86::Emitter>();
#if defined(FBGEMM_LOG_CODE)
    // generated code logging
    FILE* codeLogfile = fopen(
        getCodeLoggingFile<instSet>(
            accum, O1, i1Xich, strideXich, i1Xich, mRegBlockSize, nRegBlockSize)
            .c_str(),
        "w");
    asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
    if (codeLogger) {
      code.setLogger(codeLogger);
    }
#endif

    const int maxMRegs = mRegBlockSize;
    (void)maxMRegs; // Suppress unused variable warning
    const int maxNRegs = nRegBlockSize * row_interleave / vectorLen;
    assert(
        maxMRegs * maxNRegs <= numRegs - 4 &&
        "MRegs x NRegs is above available registers (MAX_REGS - 4)");

    int O1RegBlocks = O1 / mRegBlockSize;
    int O1RegBlocksRem = O1 % mRegBlockSize;

    // arguments to the function created
    x86::Gp buffer_A = a->zdi();
    x86::Gp buffer_B = a->zsi();
    x86::Gp B_pf = a->zdx();
    x86::Gp CBase = a->zcx();
    x86::Gp ichXk1 = a->gpz(8);
    x86::Gp ldcReg = a->gpz(9);

    asmjit::FuncDetail func;
    func.init(
        asmjit::FuncSignatureT<
            void,
            uint8_t*,
            int8_t*,
            int8_t*,
            int32_t*,
            int,
            int>(asmjit::CallConvId::kHost),
        a->environment());

    asmjit::FuncFrame frame;
    frame.init(func);

    auto dirtyVecRegs = asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
        asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15);
    if (numRegs >= 16) {
      dirtyVecRegs |= asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
          asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31);
    }

    frame.setDirtyRegs(asmjit::RegGroup::kVec, dirtyVecRegs);
    frame.setDirtyRegs(
        asmjit::RegGroup::kGp,
        asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(buffer_A, buffer_B, B_pf, CBase, ichXk1, ldcReg);

    args.updateFuncFrame(frame);
    frame.finalize();

    a->emitProlog(frame);
    a->emitArgsAssignment(frame, args);

    asmjit::Label LoopMBlocks = a->newLabel();
    // asmjit::Label LoopOBlocks = a->newLabel();
    // asmjit::Label LoopNBlocks = a->newLabel();

    x86::Gp buffer_B_saved = a->gpz(10);
    x86::Gp C_Offset = a->gpz(11);
    // x86::Gp B_pf_saved = a->gpz(12);
    x86::Gp iIdx = a->gpz(13);
    // x86::Gp jIdx = a->gpz(14);
    x86::Gp kIdx = a->gpz(15);
    // x86::Gp B_pf = a->gpz(8);

    VecRegT oneReg(numRegs - 3);

    gen16BitVectorOne<instSet, VecRegT>(a, oneReg);
    a->imul(ldcReg, ldcReg, static_cast<asmjit::Imm>(sizeof(int32_t)));
    // a->xor_(C_Offset.r32(), C_Offset.r32());

    // a->mov(B_pf_saved, B_pf);

    int colRegs = maxNRegs;

    auto issueLoopOverK = [&](int rowRegs) {
      // loopKLabel: corresponds to loop "r" where r = 0
      // loopK0Label: corresponds to loop "r" where r = 1
      asmjit::Label LoopKLabel = a->newLabel();
      asmjit::Label LoopK0Label = a->newLabel();

      // Init C (result) vector registers
      initCRegs(a, rowRegs, colRegs);

      // Loops over K: input channel
      // a.k.a this issueLoopOverK code block generates code
      // corresponding to the "ich" loop of the psedo-code
      a->xor_(kIdx.r32(), kIdx.r32());
      a->bind(LoopKLabel);

      // k is incremented by row_interleave
      a->add(kIdx, static_cast<asmjit::Imm>(row_interleave));

      // this ComputeBlock generates code correspondent to
      // the above psedu-code since the kernel_height loop (loop "r").
      // And because K[0] == 2 and IN_DIM[2] (requirement #2),
      // we can unroll loop "r" here. Thus this following
      // genComputeBlockDirectConv generates code for loop "r" = 0
      genComputeBlockDirectConv<instSet>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, strideXich);

      // update buffer_A address for next k iteration
      a->add(
          buffer_A, static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

      // update buffer_B address for next k iteration
      a->add(buffer_B, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
      a->add(B_pf, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

      a->cmp(kIdx, ichXk1);
      a->jl(LoopKLabel);

      a->sub(buffer_A, ichXk1);

      a->add(buffer_A, static_cast<asmjit::Imm>(i1Xich));

      a->xor_(kIdx.r32(), kIdx.r32());
      a->bind(LoopK0Label);

      // k is incremented by row_interleave
      a->add(kIdx, static_cast<asmjit::Imm>(row_interleave));

      // this ComputeBlock generates code that corresponds
      // to the kernel_height loop (loop "r") in the psedu-code above.
      // And the following genComputeBlockDirectConv
      // generates code for loop "r" where "r" = 1
      genComputeBlockDirectConv<instSet>(
          a, buffer_A, buffer_B, B_pf, rowRegs, colRegs, strideXich);

      // update buffer_A address for next k iteration
      a->add(
          buffer_A, static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

      // update buffer_B address for next k iteration
      a->add(buffer_B, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));
      a->add(B_pf, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

      a->cmp(kIdx, ichXk1);
      a->jl(LoopK0Label);

      a->sub(buffer_A, ichXk1);

      // store C matrix
      storeCRegs<instSet>(a, rowRegs, colRegs, C_Offset, ldcReg, accum);
    };

    if (O1RegBlocks > 0) {
      // move 0 to iteration variables
      a->xor_(iIdx.r32(), iIdx.r32());

      // iIdex loop corresponds to kernel_width loop (loop "s")
      // in the direct conv loops
      a->bind(LoopMBlocks);
      a->inc(iIdx);

      // save B_buffer address
      a->mov(buffer_B_saved, buffer_B);

      issueLoopOverK(mRegBlockSize);

      int rowRegs = mRegBlockSize;

      // reset A
      a->sub(buffer_A, static_cast<asmjit::Imm>(i1Xich));

      // increment A for next block
      a->add(
          buffer_A,
          static_cast<asmjit::Imm>(rowRegs * strideXich * sizeof(uint8_t)));

      // B for next block
      a->mov(buffer_B, buffer_B_saved);

      // increment C for next B block
      // ldcReg already multiplied with 4 (sizeof(int32_t))
      a->imul(
          C_Offset, ldcReg, static_cast<asmjit::Imm>(rowRegs * sizeof(int8_t)));
      a->add(CBase, C_Offset);

      // a->add(CBase, static_cast<asmjit::Imm>(12*16*4));
      // storeCRegs<instSet>(a, 12, 1, C_Offset, ldcReg, accum);

      a->cmp(iIdx, O1RegBlocks);
      a->jl(LoopMBlocks);
    }

    // generate code for remainder
    if (O1RegBlocksRem > 0) {
      issueLoopOverK(O1RegBlocksRem);
    }

    a->emitEpilog(frame);

    jit_micro_kernel_fp fn;
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
    fclose(codeLogfile);
    delete codeLogger;
#endif

    return fn;
  });
}

/**
 * Generate AVX256 instructions for storing the C registers back to the memory
 * in 32-bit Accumulation kernel.
 */
template <>
template <inst_set_t instSet>
void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::storeCRegsTrans(
    x86::Emitter* a,
    int rowRegs,
    int colRegs,
    x86::Gp C_offset,
    x86::Gp o1XocReg,
    x86::Gp ldcReg,
    bool accum) {
  using VecT = typename simd_info<instSet>::vec_reg_t;
  // static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  a->xor_(C_offset.r32(), C_offset.r32());
  for (int i = 0; i < rowRegs; ++i) {
    for (int j = 0; j < colRegs; ++j) {
      if (accum) {
        a->vpaddd(
            VecT(i * colRegs + j),
            VecT(i * colRegs + j),
            x86::dword_ptr(a->zcx(), C_offset));
      }
      a->vmovups(x86::dword_ptr(a->zcx(), C_offset), VecT(i * colRegs + j));
      a->add(C_offset, ldcReg);
    }
    a->add(C_offset, o1XocReg);
  }
}

/**
 * Generate AVX256 instructions for computing block in the rank-k update of
 * 32-bit Accumulation kernel.

The function generates the register blocking code for transposed
direct convolution
          // register blocking for transposed direct convolution:
          // K[0] x K[1] = 12, the corresponding 12 x 8 output elements will
          // be kept in the twelve avx2 registers, which is:
          // out[ih + 0..r][iw + 0..s][8]
          for (int r = 0; r < K[0]; ++r) {
            for (int s = 0; s < K[1]; ++s) {
              int oh = ih * conv_p.stride[0] + r;
              int ow = iw * conv_p.stride[1] + s;

              int a = input[((ih)*IN_DIM[1] + iw) * IC + icb];
              int b = weight
                  [(((((oc / 8) * K[0] + r) * K[1] + s) * (IC / 4) + icb / 4) *
                        8 +
                    (oc % 8)) *
                       4 +
                   (icb % 4)];
              out[((oh)*OUT_DIM[1] + ow) * OC + oc] += a * b;
            }
          }
        }
 *
 */
template <>
template <inst_set_t instSet>
void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    genComputeBlockDirectConvTrans(
        x86::Emitter* a,
        x86::Gp buffer_A,
        x86::Gp buffer_B,
        x86::Gp icReg,
        x86::Gp C_offset,
        int rowRegs,
        int colRegs) {
  // static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;

  // used for matrix A
  VecRegT AReg(numRegs - 1);

  // used for matrix B
  VecRegT BReg(numRegs - 2);

  // Contains 16-bit 1s
  VecRegT oneReg(numRegs - 3);

  // temporary register
  VecRegT res1(numRegs - 4);

  // load A
  a->vpbroadcastd(AReg, x86::dword_ptr(buffer_A));

  a->xor_(C_offset.r32(), C_offset.r32());
  for (int i = 0; i < rowRegs; ++i) {
    for (int j = 0; j < colRegs; ++j) {
      // load B, broadcast and fmas
      emitLoadDWord<instSet, VecRegT>(
          a, BReg, x86::dword_ptr(buffer_B, C_offset, 3, 0));
      a->vpmaddubsw(res1, AReg, BReg);
      a->vpmaddwd(res1, oneReg, res1);
      a->vpaddd(VecRegT(i * colRegs + j), res1, VecRegT(i * colRegs + j));
      a->add(C_offset, icReg);
    }
    // a->prefetcht0(x86::dword_ptr(B_pf, j * vectorLen * sizeof(int8_t)));
  }
}

/**
 * Get or Create the AVX256 instructions for 32-bit Accumulation macro-kernel.
 *
 * This function implements a direct convolution kernel that is specialized
 * for kernel size (2, 6)
 *
 * More specifically the implementation has the following prerequisites:
 * * Weights has layout {OC/8, KH, KW, IC/4, 8, 4}
 * * kernel size (2, 6)
 * * Features are in channel last format
 * * Stride[0] = 1, Stride[1] = 1 or 2
 * * Padding = 0
 *
 * mRegBlockSize = 12: the number of avx2 registers for output
 * nRegBlockSize = 8: the # of output elements in one avx2 register
 * row_interleave = 4: the horizontal reduction size for vpmaddubsw instruction
 * stride: stride[1], 1 or 2. we stride[0] = 1
 * ic: input channel
 * i1: input_width: IN_DIM[1]
 * ldcReg: leading dimension of output, a.k.a OC
 * o1Xoc: output width multiply output channel: OUT_DIM[1] x OC
 *
 * The kernel implements the following algorithm:

  for (int oc = 0; oc < OC; oc++) {
    for (int ih = 0; ih < IN_DIM[0]; ++ih) {
      for (int iw = 0; iw < IN_DIM[1]; iw++) {
        // L1 blocking
        for (int icb = 0; icb < IC; icb+=4) {
          // register blocking:
          // K[0] x K[1] = 12, the corresponding 12 x 8 output elements will
          // be kept in the twelve avx2 registers, which is:
          // out[ih + 0..r][iw + 0..s][8]
          for (int r = 0; r < K[0]; ++r) {
            for (int s = 0; s < K[1]; ++s) {
              for (int _icb = icb ; _icb < icb + 4; _icb ++) {
              int oh = ih * conv_p.stride[0] + r;
              int ow = iw * conv_p.stride[1] + s;

              int a = input[((ih)*IN_DIM[1] + iw) * IC + icb];
              int b = weight
                  [(((((oc / 8) * K[0] + r) * K[1] + s) * (IC / 4) + icb / 4) *
                        8 +
                    (oc % 8)) *
                       4 +
                   (icb % 4)];
              out[((oh)*OUT_DIM[1] + ow) * OC + oc] += a * b;

              // if we get rid of the brackets, and substitude corresponding
 variables:
              // out[ih * stride0 * o1Xoc + r * o1Xoc + iw * ldcReg + oc]
              // input[ih * i1 * ic + iw * ic + icb]
              }
            }
          }
        }
      } // for each ic
    } // for each s
  } // for each r

 *
 */

/**
 * Get or Create the AVX256 instructions for 32-bit Accumulation macro-kernel.
 *
 */
template <>
template <inst_set_t instSet>
DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    jit_micro_kernel_fp_convT
    DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
        getOrCreateDirectConvTrans(
            bool accum,
            int32_t stride,
            int32_t numColRegs) {
  using VecRegT = typename simd_info<instSet>::vec_reg_t;
  constexpr int numRegs = simd_info<instSet>::NUM_VEC_REGS;
  static constexpr int vectorLen = simd_info<instSet>::WIDTH_BYTES;

  std::tuple<bool, int, int, int> kernelSig;
  // int ichSize = 32;
  int mRowRegBlockSize = 2;
  int mColRegBlockSize = numColRegs;
  int mRegBlockSize = mRowRegBlockSize * mColRegBlockSize;
  int nRegBlockSize = 8;
  // int nRegBlockSizeMin;
  int row_interleave = 4;

  kernelSig = std::make_tuple(accum, stride, mRegBlockSize, nRegBlockSize);

  return codeCacheT_.getOrCreate(kernelSig, [&]() -> jit_micro_kernel_fp_convT {
    asmjit::CodeHolder code;
    code.init(runtime().environment());
    x86::Assembler assembler(&code);
    x86::Emitter* a = assembler.as<x86::Emitter>();
#if defined(FBGEMM_LOG_CODE)
    // generated code logging
    FILE* codeLogfile = fopen(
        getCodeLoggingFile<instSet>(accum, stride, mRegBlockSize, nRegBlockSize)
            .c_str(),
        "w");
    asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
    if (codeLogger) {
      code.setLogger(codeLogger);
    }
#endif

    const int maxMRegs = mRegBlockSize;
    (void)maxMRegs; // Suppress unused variable warning
    const int maxNRegs = nRegBlockSize * row_interleave / vectorLen;
    assert(
        maxMRegs * maxNRegs <= numRegs - 4 &&
        "MRegs x NRegs is above available registers (MAX_REGS - 4)");

    // arguments to the function created
    x86::Gp buffer_A = a->zdi();
    x86::Gp buffer_B = a->zsi();
    x86::Gp CBase = a->zcx();
    x86::Gp ic = a->gpz(8);
    x86::Gp ldcReg = a->gpz(9);
    x86::Gp o1Xoc = a->gpz(10);
    x86::Gp i1 = a->gpz(11);

    asmjit::FuncDetail func;
    func.init(
        asmjit::FuncSignatureT<
            void,
            uint8_t*,
            int8_t*,
            int32_t*,
            int,
            int,
            int,
            int>(asmjit::CallConvId::kHost),
        a->environment());

    asmjit::FuncFrame frame;
    frame.init(func);

    auto dirtyVecRegs = asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
        asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15);
    if (numRegs >= 16) {
      dirtyVecRegs |= asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
          asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31);
    }

    frame.setDirtyRegs(asmjit::RegGroup::kVec, dirtyVecRegs);
    frame.setDirtyRegs(
        asmjit::RegGroup::kGp,
        asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(buffer_A, buffer_B, CBase, ic, ldcReg, o1Xoc, i1);

    args.updateFuncFrame(frame);
    frame.finalize();

    a->emitProlog(frame);
    a->emitArgsAssignment(frame, args);

    asmjit::Label LoopMBlocks = a->newLabel();

    x86::Gp C_offset = a->gpz(12);
    x86::Gp buffer_B_saved = a->gpz(13);
    x86::Gp iIdx = a->gpz(14);
    x86::Gp kIdx = a->gpz(15);

    VecRegT oneReg(numRegs - 3);

    gen16BitVectorOne<instSet, VecRegT>(a, oneReg);
    a->imul(ldcReg, ldcReg, static_cast<asmjit::Imm>(sizeof(int32_t)));

    int colRegs = maxNRegs;

    auto issueLoopOverK = [&](int rowRegs) {
      asmjit::Label LoopKLabel = a->newLabel();

      // Init C (result) vector registers
      initCRegs(a, rowRegs, colRegs);

      // Loops over K: input channel
      a->xor_(kIdx.r32(), kIdx.r32());
      a->bind(LoopKLabel);

      // k is incremented by row_interleave
      a->add(kIdx, 4);
      genComputeBlockDirectConvTrans<instSet>(
          a,
          buffer_A,
          buffer_B,
          ic,
          C_offset,
          mRowRegBlockSize,
          mColRegBlockSize);

      // update buffer_A address for next k iteration
      a->add(
          buffer_A, static_cast<asmjit::Imm>(row_interleave * sizeof(uint8_t)));

      // update buffer_B address for next k iteration
      a->add(buffer_B, static_cast<asmjit::Imm>(8 * sizeof(int32_t)));

      a->cmp(kIdx, ic);
      a->jl(LoopKLabel);

      // store C matrix
      storeCRegsTrans<instSet>(
          a,
          mRowRegBlockSize,
          mColRegBlockSize,
          C_offset,
          o1Xoc,
          ldcReg,
          accum);
    };

    {
      // move 0 to iteration variables
      a->xor_(iIdx.r32(), iIdx.r32());

      a->bind(LoopMBlocks);
      a->inc(iIdx);

      // save B_buffer address
      a->mov(buffer_B_saved, buffer_B);

      issueLoopOverK(mRegBlockSize);

      // B for next block
      a->mov(buffer_B, buffer_B_saved);
      // increment C for next B block
      a->imul(
          C_offset,
          ldcReg,
          static_cast<asmjit::Imm>(stride)); // ldcReg already multiplied by 4
      a->add(CBase, C_offset);

      a->cmp(iIdx, i1);
      a->jl(LoopMBlocks);
    }

    a->emitEpilog(frame);

    jit_micro_kernel_fp_convT fn;
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
    fclose(codeLogfile);
    delete codeLogger;
#endif

    return fn;
  });
}

/**
 * Instantiate the inst_set_t::avx512 instructions for store kernel.
 *
 */
template void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    storeCRegs<inst_set_t::avx512>(
        x86::Emitter* a,
        int rowRegs,
        int colRegs,
        x86::Gp C_Offset,
        x86::Gp ldcReg,
        bool accum);

/**
 * Instantiate the inst_set_t::avx512_ymm instructions for store kernel.
 *
 */
template void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    storeCRegs<inst_set_t::avx512_ymm>(
        x86::Emitter* a,
        int rowRegs,
        int colRegs,
        x86::Gp C_Offset,
        x86::Gp ldcReg,
        bool accum);

/**
 * Instantiate the inst_set_t::avx2 instructions for store kernel.
 *
 */
template void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    storeCRegs<inst_set_t::avx2>(
        x86::Emitter* a,
        int rowRegs,
        int colRegs,
        x86::Gp C_Offset,
        x86::Gp ldcReg,
        bool accum);

/**
 * Instantiate the inst_set_t::avx2 instructions for store kernel.
 *
 */
template void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    storeCRegsTrans<inst_set_t::avx512>(
        x86::Emitter* a,
        int rowRegs,
        int colRegs,
        x86::Gp C_offset,
        x86::Gp o1XocReg,
        x86::Gp ldcReg,
        bool accum);

/**
 * Instantiate the inst_set_t::avx2 instructions for store kernel.
 *
 */
template void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    storeCRegsTrans<inst_set_t::avx512_ymm>(
        x86::Emitter* a,
        int rowRegs,
        int colRegs,
        x86::Gp C_offset,
        x86::Gp o1XocReg,
        x86::Gp ldcReg,
        bool accum);

/**
 * Instantiate the inst_set_t::avx2 instructions for store kernel.
 *
 */
template void DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    storeCRegsTrans<inst_set_t::avx2>(
        x86::Emitter* a,
        int rowRegs,
        int colRegs,
        x86::Gp C_offset,
        x86::Gp o1XocReg,
        x86::Gp ldcReg,
        bool accum);

/**
 * Instantiate the AVX2 instructions for 32-bit Accumulation macro-kernel.
 *
 */
template DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    jit_micro_kernel_fp
    DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
        getOrCreateDirectConv<inst_set_t::avx2>(
            bool accum,
            int32_t O1,
            int32_t i1Xich,
            int32_t strideXich);

/**
 * Instantiate the AVX2 instructions for 32-bit Accumulation macro-kernel.
 *
 */
template DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
    jit_micro_kernel_fp_convT
    DirectConvCodeGenBase<uint8_t, int8_t, int32_t, int32_t>::
        getOrCreateDirectConvTrans<inst_set_t::avx2>(
            bool accum,
            int32_t stride,
            int32_t numColRegs);

} // namespace fbgemm
