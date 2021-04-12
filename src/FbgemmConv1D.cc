#define FBGEMM_EXPORTS

#include "./FbgemmConv1D.h"
#include <asmjit/asmjit.h>
#include <cpuinfo.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept> // for logic_error
#include <tuple>
#include <type_traits>
#include <vector>
#include "./CodeGenHelpers.h"
#include "./GroupwiseConv.h"
#include "fbgemm/Fbgemm.h"

// #define FBGEMM_MEASURE_TIME_BREAKDOWN

//#define FBGEMM_LOG_CODE 1
namespace fbgemm {

using namespace std;
namespace x86 = asmjit::x86;

void kernel_compute_1d(
    const conv_param_t<1>& conv_p,
    const uint8_t* in_acts,
    int8_t* wghts,
    int32_t* out_acts,
    int32_t a_zero_pt,
    int32_t ow_start,
    int32_t ow_end,
    int32_t* rowOffset,
    int32_t vsize,
    int32_t nreg,
    bool accum = 0) {
  // int IW = conv_p.IN_DIM[0];
  int IC = conv_p.IC;
  int OC = conv_p.OC;
  int G = conv_p.G;
  int S = conv_p.K[0];
  int IC_per_G = conv_p.IC / G;
  int OC_per_G = conv_p.OC / G;

  int paddedICPerG = (IC_per_G + 3) / 4 * 4;
  // int nreg = getRegNumForConv1D();
  int paddedOCPerG = (OC_per_G + vsize - 1) / vsize * vsize;
  int bsize = nreg * vsize;
  int nblock = paddedOCPerG / bsize;

  for (int ow = ow_start; ow < ow_end; ++ow) {
    for (int k = 0; k < OC_per_G; ++k) {
      int sum = 0;
      int rowSum = 0;
      for (int s = 0; s < S; ++s) {
        int w_in = -conv_p.pad[0] + ow * conv_p.stride[0] + s;
        for (int c = 0; c < IC_per_G; ++c) {
          bool out_of_range = w_in < 0 || w_in >= conv_p.IN_DIM[0];
          int w_index = w_in;
          if (ow_start > 0) {
            w_index = (ow - ow_start) * conv_p.stride[0] + s;
          }
          int a = out_of_range ? a_zero_pt : in_acts[w_index * IC + c];

          int block = k / bsize;
          int usedbsize = bsize;
          if (block == nblock) {
            usedbsize = paddedOCPerG % bsize;
          }
          int idx = s * (paddedICPerG * paddedOCPerG) +
              paddedICPerG * bsize * block + (c / 4) * usedbsize * 4 +
              (k % bsize) * 4 + (c % 4);

          int b = wghts[idx];
          sum += a * b;
          rowSum += a;
        }
        if (k == 0) printf("AAA ow=%4d  s=%4d, rowsum =  %d\n", ow -
         ow_start, s, rowSum);
      }
      if (accum) {
        out_acts[(ow - ow_start) * OC + k] += sum;
        if (k == 0) {
          // only accumulate for k == 0
          rowOffset[ow - ow_start] += rowSum;
        }
      } else {
        out_acts[(ow - ow_start) * OC + k] = sum;
        rowOffset[ow - ow_start] = rowSum;
      }
    }
  }
}

kernel1d_sig_t getKernel1DSig(
    const conv_param_t<1>& conv_param,
    bool isAZeroPointZero,
    bool needRowOffset,
    bool accum) {
  // kernel is specialized on number of input channels per group, number of
  // output channels per group, whether stride is 1 or stride is 2, whether or
  // not zero point for activations is 0 or not, whether or not row offset
  // calculations are needed, whether or not top edge is included and whether or
  // not bottom edge is included.
  // use_padding_: If false, the right padding on the width side and bottom
  // padding on height side are not used for the case of stride = 2
  // accum: accumulate results for output and rowoffset
  int C_per_G = conv_param.IC / conv_param.G;
  int K_per_G = conv_param.OC / conv_param.G;
  auto kernelSig = make_tuple(
      isAZeroPointZero,
      needRowOffset,
      accum,
      conv_param.G,
      conv_param.stride[0],
      C_per_G,
      K_per_G,
      conv_param.IN_DIM[0]);
  return kernelSig;
}

jit_conv1d_kernel_fp getOrCreateConv1DKernel(
    const conv_param_t<1>& conv_param,
    int nreg,
    int a_zero_point,
    bool needRowOffset,
    bool accum = 0) {
  // Note: Wrong code is generated if it's not one of the supported convolution
  assert(take1DFastPath<1>(conv_param));
  auto kernelSig =
      getKernel1DSig(conv_param, a_zero_point == 0, needRowOffset, accum);
  if (cpuinfo_initialize()) {
    if (fbgemmHasAvx512VnniSupport()) {
      return GenConv1DKernel<inst_set_t::avx512_vnni>::codeCache_.getOrCreate(
          kernelSig, [&]() {
            auto genObj = GenConv1DKernel<inst_set_t::avx512_vnni>(
                conv_param, nreg, a_zero_point, needRowOffset, accum);
            return genObj.getOrCreate();
          });
    } else if (fbgemmHasAvx512Support()) {
      return GenConv1DKernel<inst_set_t::avx512>::codeCache_.getOrCreate(
          kernelSig, [&]() {
            auto genObj = GenConv1DKernel<inst_set_t::avx512>(
                conv_param, nreg, a_zero_point, needRowOffset, accum);
            return genObj.getOrCreate();
          });
    } else if (fbgemmHasAvx2Support()) {
      return GenConv1DKernel<inst_set_t::avx2>::codeCache_.getOrCreate(
          kernelSig, [&]() {
            auto genObj = GenConv1DKernel<inst_set_t::avx2>(
                conv_param, nreg, a_zero_point, needRowOffset, accum);
            return genObj.getOrCreate();
          });
    } else {
      // TODO: Have default slower path
      assert(0 && "unsupported architecture");
    }
  } else {
    throw runtime_error("Failed to initialize cpuinfo!");
  }
  return nullptr;
}

template <inst_set_t INST_SET>
jit_conv1d_kernel_fp GenConv1DKernel<INST_SET>::getOrCreate() {
  asmjit::CodeHolder code;
  code.init(this->runtime().codeInfo());
  x86::Assembler assembler(&code);
  x86::Emitter* a = assembler.as<x86::Emitter>();

  typedef typename simd_info<INST_SET>::vec_reg_t vec_reg_t;
#if defined(FBGEMM_LOG_CODE)
  auto kernelSig = make_tuple(
      this->isAZeroPointZero_,
      this->needRowOffset_,
      this->accum_,
      this->G_,
      this->STRIDE_,
      this->C_per_G_,
      this->K_per_G_,
      this->IW_);
  // log code to a file
  FILE* codeLogfile = fopen(this->getCodeLoggingFile(kernelSig).c_str(), "w");
  asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogfile);
  if (codeLogger) {
    code.setLogger(codeLogger);
  }
#endif

  // arguments to the function created
  in_acts_R_ = a->zdi();
  wghts_R_ = a->zsi();
  out_acts_R_ = a->zdx();
  a_zero_pt_R_ = a->zcx();
  W_start_R_ = a->gpz(8);
  W_end_R_ = a->gpz(9);
  W_R_ = a->gpz(10); // shan : r10 is free now
  row_offset_R_ = a->gpz(11);

  // register for temporary use
  scratchReg1_ = a->gpz(12);
  scratchReg2_ = a->gpz(13);

  func_.init(
      asmjit::FuncSignatureT<
          void,
          uint8_t*,
          int8_t*,
          int32_t*,
          int32_t,
          int32_t,
          int32_t,
          int32_t,
          int32_t*>(asmjit::CallConv::kIdHost),
      a->environment());

  frame_.init(func_);

  frame_.setDirtyRegs(
      x86::Reg::kGroupVec,
      asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
          asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15) |
          asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
          asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31));
  frame_.setDirtyRegs(
      x86::Reg::kGroupGp,
      asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));

  asmjit::FuncArgsAssignment args(&func_);
  args.assignAll(
      in_acts_R_,
      wghts_R_,
      out_acts_R_,
      a_zero_pt_R_,
      W_start_R_,
      W_end_R_,
      W_R_,
      row_offset_R_);

  args.updateFuncFrame(frame_);
  frame_.finalize();

  a->emitProlog(frame_);
  a->emitArgsAssignment(frame_, args);

  // We have run out of register so can't keep
  // this in a register. It's generated again at
  // each use. Only used for the case of C_per_G == 2 or 4
  // gen8BitVectorOne(a, oneReg8Bit_V_);
  gen16BitVectorOne<INST_SET, vec_reg_t>(a, oneReg16Bit_V_);

  loopR1_ = a->gpz(14);
  loopR2_ = a->gpz(15);

  if (!this->isAZeroPointZero_) {
    broadcast8Bit<vec_reg_t>(a, a_zero_pt_R_, zeroPTReg_V_);
  } else {
    a->vpxor(zeroPTReg_V_, zeroPTReg_V_, zeroPTReg_V_);
  }

  // shan no need to handle W_R
  // no need to handle top/bottom

  genCoreInsts(a);

  a->emitEpilog(frame_);

  jit_conv1d_kernel_fp fn;
  asmjit::Error err;
  {
    unique_lock<mutex> lock(this->rtMutex_);
    err = this->runtime().add(&fn, &code);
  }

  if (err) {
    cout << "Error: in fn add" << endl;
    return nullptr;
  }

#if defined(FBGEMM_LOG_CODE)
  fclose(codeLogfile);
  delete codeLogger;
#endif

  return fn;
}

template <inst_set_t INST_SET>
void GenConv1DKernel<INST_SET>::genForSingleOutput(x86::Emitter* a, int nw) {
  // init result regs
  // initResultRegs(a);

  asmjit::Label Over1 = a->newLabel();
  asmjit::Label LoopS = a->newLabel();
  asmjit::Label OutImage0 = a->newLabel();
  asmjit::Label OutImage1 = a->newLabel();
  asmjit::Label OutImage2 = a->newLabel();
  asmjit::Label NoChange = a->newLabel();

  // ToDo merge s into Filter
  a->push(in_acts_R_);

  a->mov(scratchReg2_, static_cast<asmjit::Imm>(nw));
  a->mov(scratchReg2_, W_start_R_);
  a->imul(scratchReg2_, static_cast<asmjit::Imm>(this->STRIDE_));
  a->sub(scratchReg2_, static_cast<asmjit::Imm>(this->W_PAD_));

  a->cmp(scratchReg2_, 0);
  a->jl(NoChange);
  a->mov(loopR2_, scratchReg2_);
  a->imul(loopR2_, static_cast<asmjit::Imm>(this->C_));
  a->add(in_acts_R_, loopR2_);

  a->bind(NoChange);
  a->mov(loopR2_, static_cast<asmjit::Imm>(this->S_));
  a->bind(LoopS);

  a->cmp(scratchReg2_, 0);
  a->jl(OutImage0);
  a->cmp(scratchReg2_, static_cast<asmjit::Imm>(this->IW_));
  a->jge(OutImage0);

  // consider right side
  // a->add(scratchReg2_, static_cast<asmjit::Imm>(this->STRIDE_ * nw));
  a->cmp(
      scratchReg2_, static_cast<asmjit::Imm>(this->STRIDE_ * (nw - 1) * (-1)));
  a->jl(OutImage1);
  a->cmp(
      scratchReg2_,
      static_cast<asmjit::Imm>(this->IW_ - this->STRIDE_ * (nw - 1)));
  a->jge(OutImage1);

  genForSingleFilterPoint(a, nw, 0x0);
  a->add(in_acts_R_, static_cast<asmjit::Imm>(this->C_));
  a->jmp(Over1);

  a->bind(OutImage1);
  genForSingleFilterPoint(a, nw, 0x2);
  a->add(in_acts_R_, static_cast<asmjit::Imm>(this->C_)); // matter ?
  a->jmp(Over1);

  a->bind(OutImage0);
  // consider right side
  // a->add(scratchReg2_, static_cast<asmjit::Imm>(this->STRIDE_ * (nw - 1)));
  a->cmp(
      scratchReg2_, static_cast<asmjit::Imm>(this->STRIDE_ * (nw - 1) * (-1)));
  a->jl(OutImage2);
  a->cmp(
      scratchReg2_,
      static_cast<asmjit::Imm>(this->IW_ - this->STRIDE_ * (nw - 1)));
  a->jge(OutImage2);

  genForSingleFilterPoint(a, nw, 0x1);
  // a->add(in_acts_R_, static_cast<asmjit::Imm>(this->C_ * (nw - 1)));
  a->jmp(Over1);

  a->bind(OutImage2);
  genForSingleFilterPoint(a, nw, 0x3);
  // a->add(in_acts_R_, static_cast<asmjit::Imm>(this->C_ * (nw - 1)));

  a->bind(Over1);

  // a->sub(scratchReg2_, static_cast<asmjit::Imm>(this->STRIDE_ * (nw -1 )));
  a->add(scratchReg2_, 1);
  a->add(
      wghts_R_,
      static_cast<asmjit::Imm>(this->paddedICPerG_ * this->paddedOCPerG_));

  a->dec(loopR2_);
  a->jnz(LoopS);

  // advance output pointer
  a->add(
      out_acts_R_, static_cast<asmjit::Imm>(nw * this->K_ * sizeof(int32_t)));

  a->sub(
      wghts_R_,
      static_cast<asmjit::Imm>(
          this->paddedICPerG_ * this->paddedOCPerG_ * (this->S_)));
  a->pop(in_acts_R_);
}

template <inst_set_t INST_SET>
void GenConv1DKernel<INST_SET>::genCoreInsts(x86::Emitter* a) {
  // main compute
  asmjit::Label LoopWStart = a->newLabel();
  asmjit::Label LoopWEnd = a->newLabel();
  asmjit::Label Loop2 = a->newLabel();
  asmjit::Label Loop1 = a->newLabel();

  // handle offsets
  if (this->needRowOffset_) {

      asmjit::Label OffsetLoop = a->newLabel();
      asmjit::Label OffsetLoopOVER = a->newLabel();
      asmjit::Label NoZERO = a->newLabel();
      asmjit::Label ZERO = a->newLabel();

      a->push(row_offset_R_);

      //rename to temp
      a->vxorps(rowOffsetReg_V1_, rowOffsetReg_V1_, rowOffsetReg_V1_);

      a->mov(loopR1_, W_start_R_);
      a->mov(loopR2_, W_end_R_);
      a->add(loopR2_, this->S_ - 1);
      a->bind(OffsetLoop);
      a->cmp(loopR1_, loopR2_);
      a->jge(OffsetLoopOVER);

      a->vxorps(rowOffsetReg_V_, rowOffsetReg_V_, rowOffsetReg_V_);
      a->mov(
          scratchReg1_,
          static_cast<asmjit::Imm>(((int64_t)1 << this->cinLoopRemainder_) - 1));
      a->kmovq(x86::k(3), scratchReg1_);

      a->imul(scratchReg2_, loopR1_, static_cast<asmjit::Imm>(this->STRIDE_));
      a->sub(scratchReg2_, static_cast<asmjit::Imm>(this->W_PAD_));
      a->cmp(scratchReg2_, 0);
      a->jl(ZERO);
      a->cmp(scratchReg2_, static_cast<asmjit::Imm>(this->IW_));
      a->jl(NoZERO);

      a->bind(ZERO);

      a->vmovdqu8(actReg_V_, zeroPTReg_V_);
      a->vpsadbw(actReg_V_, actReg_V_, rowOffsetReg_V1_);
      for (int j = 0; j < this->cinLoopIters_ - 1; j++) {
          a->vpaddd(rowOffsetReg_V_, actReg_V_, rowOffsetReg_V_);
      }
      if (this->cinLoopRemainder_ > 0) {
          a->z().k(x86::k(3)).vmovdqu8(actReg_V_, zeroPTReg_V_);
          a->vpsadbw(actReg_V_, actReg_V_, rowOffsetReg_V1_);
      }
      a->vpaddd(rowOffsetReg_V_, actReg_V_, rowOffsetReg_V_);
      storeOffset(a);
      a->add(row_offset_R_, static_cast<asmjit::Imm>(sizeof(int32_t)));
      a->inc(loopR1_);
      a->jmp(OffsetLoop);

      a->bind(NoZERO);

      a->imul(scratchReg1_, scratchReg2_, static_cast<asmjit::Imm>(this->C_));
      for (int j = 0; j < this->cinLoopIters_; j++) {
        if (this->cinLoopRemainder_ > 0 && j == this->cinLoopIters_ - 1) {
          a->z().k(x86::k(3)).vmovdqu8(
              actReg_V_,
              x86::zmmword_ptr(
                  in_acts_R_,
                  scratchReg1_,
                  0,
                  (j * this->vsize_) * sizeof(uint8_t)));
        } else {
          a->vmovupd(
              actReg_V_,
              x86::zmmword_ptr(
                  in_acts_R_,
                  scratchReg1_,
                  0,
                  (j * this->vsize_) * sizeof(uint8_t)));
        }
        a->vpsadbw(actReg_V_, actReg_V_, rowOffsetReg_V1_);
        a->vpaddd(rowOffsetReg_V_, actReg_V_, rowOffsetReg_V_);
      }
      storeOffset(a);
      a->add(row_offset_R_, static_cast<asmjit::Imm>(sizeof(int32_t)));

      a->inc(loopR1_);
      a->jmp(OffsetLoop);

      a->bind(OffsetLoopOVER);
      a->pop(row_offset_R_);
  }

  // push
  a->push(W_start_R_);

  a->mov(loopR1_, W_end_R_);
  a->sub(loopR1_, W_start_R_);

  a->bind(LoopWStart);
  a->cmp(loopR1_, 3);
  a->jl(Loop2);

  genForSingleOutput(a, 3);
  a->sub(loopR1_, 3);
  a->add(W_start_R_, 3);
  a->jmp(LoopWStart);

  a->bind(Loop2);
  a->cmp(loopR1_, 2);
  a->jl(Loop1);
  genForSingleOutput(a, 2);
  a->jmp(LoopWEnd);

  a->bind(Loop1);
  a->cmp(loopR1_, 1);
  a->jl(LoopWEnd);
  genForSingleOutput(a, 1);
  a->bind(LoopWEnd);

  // pop
  a->pop(W_start_R_);
}

template <typename processOutputType>
int fbgemmFast1DConv(
    const conv_param_t<1>& conv_p,
    const std::uint8_t* activations,
    PackWeightMatrixFor1DConv<std::int8_t, std::int32_t>& packed_weights,
    std::uint8_t* out,
    std::int32_t* outBuffer,
    processOutputType& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params) {
#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end,
      t_very_start;
  double dt;
  t_start = std::chrono::high_resolution_clock::now();
  t_very_start = std::chrono::high_resolution_clock::now();
#endif

  if (!cpuinfo_initialize()) {
    throw runtime_error("Failed to initialize cpuinfo!");
  }
  constexpr QuantizationGranularity Q_GRAN = processOutputType::QGRANType;

  if (blocking_params) {
    throw runtime_error("Blocking_params is not used in fbgemmFast1DConv!");
  }

  int MB = conv_p.MB;
  int OW = conv_p.OUT_DIM[0];
  int S = conv_p.K[0];
  int G = conv_p.G;
  int OC = conv_p.OC;
  int IC = conv_p.IC;
  int OC_per_G = conv_p.OC / G;
  int IC_per_G = conv_p.IC / G;
  int IW = conv_p.IN_DIM[0];
  int paddedICPerG = (IC_per_G + 3) / 4 * 4;

  // use a var later for performance
  const inst_set_t isa = fbgemmInstructionSet();
  int vsize4;
  if (isa == inst_set_t::avx2) {
    vsize4 = simd_info<inst_set_t::avx2>::WIDTH_BYTES / 4;
  } else {
    vsize4 = simd_info<inst_set_t::avx512>::WIDTH_BYTES / 4;
  }
  int paddedOCPerG = (OC_per_G + vsize4 - 1) / vsize4 * vsize4;

  int32_t* rowOffsetBuf = const_cast<int32_t*>(outProcess.getRowOffsets());
  int32_t a_zero_point = outProcess.getAZeroPoint();

  bool b_symmetric = (Q_GRAN == QuantizationGranularity::TENSOR &&
                      outProcess.getBZeroPoint()[0] == 0) ||
      rowOffsetBuf == nullptr;

  bool needAccum = true;

  // Parallelization:
  int batch_start = 0;
  int batch_end = MB;
  int ow_start = 0;
  int ow_end = OW;
  if (MB >= num_threads) {
    fbgemmPartition1D(thread_id, num_threads, MB, batch_start, batch_end);
  } else {
    fbgemmPartition1D(thread_id, num_threads, OW, ow_start, ow_end);
  }

  if (batch_start >= batch_end || ow_start >= ow_end) {
    // There is no work for this thread
    return 0;
  }

  // generate convolution  + rowOffset kernel
  bool calculateRowOffset = !b_symmetric;

  int nreg = packed_weights.getRegNumForConv1D();
  jit_conv1d_kernel_fp fpConv = getOrCreateConv1DKernel(
      conv_p, nreg, a_zero_point, calculateRowOffset, needAccum);

  // need to clear output buffer ?
  memset(out, 0, OW * OC * sizeof(std::uint8_t));
  memset(outBuffer, 0, OW * OC * sizeof(std::int32_t));

  int ow_dummy = 1;
  int iw_start = 0;
  if (ow_start > 0) {
    iw_start = -conv_p.pad[0] + ow_start * conv_p.stride[0];
  }
  int32_t* out_start = outBuffer + ow_start * OC;
  const uint8_t* in_start = activations + iw_start * IC;
  int32_t* rowOffsetBuf_start = rowOffsetBuf + OW;
  for (int i = batch_start; i < batch_end; ++i) {
    const uint8_t* in_start_batch = in_start + i * IW * IC;
    int32_t* out_start_batch = out_start + i * OW * OC;
    int32_t* rowOffsetBuf_start_batch = rowOffsetBuf_start + i * OW * G;
    for (int g = 0; g < G; g++) {
      const uint8_t* in_start_group = in_start_batch + g * IC_per_G;
      int8_t* weight_start =
          packed_weights.getBuf() + g * S * paddedOCPerG * paddedICPerG;
      // reuse the buffer ?
      int32_t* out_start_group = out_start_batch + g * OC_per_G;
      int32_t* rowOffsetBuf_start_group = rowOffsetBuf_start_batch + g * OW;

      // exactly the same compute as the JIT'ed below
      /**
      kernel_compute_1d(
          conv_p,
          in_start_group,
          weight_start,
          out_start_group,
          a_zero_point,
          ow_start,
          ow_end,
          rowOffsetBuf_start_group,
          vsize4,
          nreg);
      **/

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      t_start = std::chrono::high_resolution_clock::now();
#endif

      fpConv(
          in_start_group,
          weight_start,
          out_start_group,
          a_zero_point,
          ow_start,
          ow_end,
          ow_dummy,
          rowOffsetBuf_start_group);

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      t_end = std::chrono::high_resolution_clock::now();
      dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
               .count();
      kernel_time += (dt);
      t_start = std::chrono::high_resolution_clock::now();
#endif

      int32_t *temps = rowOffsetBuf_start_group + conv_p.pad[0];
      int32_t *tempd = rowOffsetBuf_start_group - OW + ow_start;

      for (int off = 0; off < ow_end - ow_start; off++) {
        int pos = (off + ow_start) * conv_p.stride[0] - conv_p.pad[0];
        tempd[off] = temps[pos];
        for (int s = 1; s < S; s++) {
          tempd[off] += temps[pos+s];
        }
      }

      // int32_t *tempd = rowOffsetBuf_start_group;

      const int32_t* inp = out_start_group;
      block_type_t block{
          i * OW + ow_start, ow_end - ow_start, g * OC_per_G, OC_per_G};

      int ld_out = G * OC_per_G;
      int ld_in = G * OC_per_G;

      outProcess.setRowOffsets(tempd);
      if (isa == inst_set_t::avx2) {
        outProcess.template f<inst_set_t::avx2>(out, inp, block, ld_out, ld_in);
      } else {
        outProcess.template f<inst_set_t::avx512>(
            out, inp, block, ld_out, ld_in);
      }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
      t_end = std::chrono::high_resolution_clock::now();
      dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
               .count();
      postprocessing_time += (dt);
#endif
    }

    outProcess.setRowOffsets(rowOffsetBuf);
  }

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
  t_end = std::chrono::high_resolution_clock::now();
  dt =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_very_start)
          .count();
  run_time += (dt);
#endif

  return 0;
}

int rowOffsetBufferSize1DConv(const conv_param_t<1>& conv_param) {
  if (cpuinfo_initialize()) {
    int bufferSize = conv_param.OUT_DIM[0] + conv_param.pad[0] + conv_param.IN_DIM[0];
    if (fbgemmHasAvx512Support()) {
      return conv_param.MB * bufferSize *
          conv_param.G; // shan G should be (G+3)/4*4 ?
    } else if (fbgemmHasAvx2Support()) {
      return conv_param.MB * bufferSize * conv_param.G;
    } else {
      // TODO: Have default slower path
      assert(0 && "unsupported architecture");
      return -1;
    }
  } else {
    throw runtime_error("Failed to initialize cpuinfo!");
  }
}

#define INSTANTIATE_BASE(RELU, Q_GRAN, BIAS_TYPE)                           \
  template FBGEMM_API int fbgemmFast1DConv(                                 \
      const conv_param_t<1>& conv_p,                                        \
      const std::uint8_t* activations,                                      \
      PackWeightMatrixFor1DConv<std::int8_t, std::int32_t>& packed_weights, \
      std::uint8_t* out,                                                    \
      std::int32_t* outBuffer,                                              \
      ReQuantizeOutput<RELU, Q_GRAN, BIAS_TYPE>& outProcess,                \
      int thread_id,                                                        \
      int num_threads,                                                      \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_BIAS_T(RELU, Q_GRAN) \
  INSTANTIATE_BASE(RELU, Q_GRAN, float); \
  INSTANTIATE_BASE(RELU, Q_GRAN, std::int32_t);

#define INSTANTIATE_GRANS(RELU)                              \
  INSTANTIATE_BIAS_T(RELU, QuantizationGranularity::TENSOR); \
  INSTANTIATE_BIAS_T(RELU, QuantizationGranularity::GROUP);  \
  INSTANTIATE_BIAS_T(RELU, QuantizationGranularity::OUT_CHANNEL);

INSTANTIATE_GRANS(false);
INSTANTIATE_GRANS(true);

#undef INSTANTIATE_GRANS
#undef INSTANTIATE_BIAS_T
#undef INSTANTIATE_BASE

} // namespace fbgemm
