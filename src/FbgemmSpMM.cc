/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmSpMM.h"
#include "./FbgemmSpMM-inl.h"

#include <cpuinfo.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <sstream>

#include <fbgemm/Utils.h>

using namespace std;

namespace fbgemm {

using internal::SpMMTypeTrait;

namespace {

template <typename ACC_T, inst_set_t instSet>
class SpMM_JitterASMJIT
    : public internal::SpMM_JitterASMJIT_base<ACC_T, instSet> {
 private:
  int N_;

  x86::Gp NLoopReg_ = x86::r8;
  x86::Gpd tmpReg_ = x86::r9.r32();

  // For AVX2
  asmjit::Label fullMaskLabel_;
  asmjit::Label partialMaskLabel_;

  using BaseType = internal::SpMM_JitterASMJIT_base<ACC_T, instSet>;
  using BaseType::a;
  using BaseType::Breg_;
  using BaseType::Creg_;
  using BaseType::firstAvailableVecRegister_;
  using BaseType::useAvx512_;
  using BaseType::WIDTH_32BIT_ELEMS;
  using BaseType::WIDTH_BYTES;
  using typename BaseType::vec_reg_t;

  // Multiply rowRegBlock rows of A with B
  void loopOverN(int AOffset, int rowRegBlock, bool restorePointers) override {
    int fullIterations = N_ / WIDTH_32BIT_ELEMS;
    int rest = N_ % WIDTH_32BIT_ELEMS;

    asmjit::Label fnLabel =
        this->registerBlockedLoop(AOffset, rowRegBlock, rest);

    bool updatesPointerRegisters =
        ((fullIterations > 1) || (fullIterations == 1 && rest)) &&
        restorePointers;

    if (fullIterations > 0 && rest) {
      if (useAvx512_) {
        a.mov(tmpReg_, 0xffff);
        a.kmovw(this->maskReg_, tmpReg_);
      } else {
        a.vmovups(this->maskVReg_, x86::ptr(fullMaskLabel_));
      }
    }

    if (updatesPointerRegisters) {
      a.push(Creg_);
      a.push(Breg_);
    }

    if (fullIterations == 1) {
      a.call(fnLabel);
      if (rest) {
        a.add(Breg_, WIDTH_BYTES);
        a.add(Creg_, WIDTH_BYTES);
      }
    } else if (fullIterations > 1) {
      a.mov(NLoopReg_, static_cast<asmjit::Imm>(0));

      asmjit::Label loopLabel = a.newLabel();
      a.bind(loopLabel);

      a.add(NLoopReg_, 1);

      a.call(fnLabel);

      a.add(Breg_, WIDTH_BYTES);
      a.add(Creg_, WIDTH_BYTES);

      a.cmp(NLoopReg_, static_cast<asmjit::Imm>(fullIterations));
      a.jl(loopLabel);
    }

    if (rest) {
      if (useAvx512_) {
        a.mov(tmpReg_, (1 << rest) - 1);
        a.kmovw(this->maskReg_, tmpReg_);
      } else {
        a.vmovups(this->maskVReg_, x86::ptr(partialMaskLabel_));
      }

      a.call(fnLabel);
    }

    if (updatesPointerRegisters) {
      a.pop(Breg_);
      a.pop(Creg_);
    }
  }

 public:
  SpMM_JitterASMJIT(
      asmjit::CodeHolder& code,
      int M,
      int N,
      int K,
      int LDA,
      int LDB,
      int LDC,
      const typename SpMMTypeTrait<ACC_T>::a_type* aData,
      bool canUseVNNI = false)
      : BaseType(code, M, K, LDA, LDB, LDC, aData, canUseVNNI),
        N_(N),
        fullMaskLabel_(a.newLabel()),
        partialMaskLabel_(a.newLabel()) {
    this->flagsReg_ = x86::rdx;
    this->dataReg_ = x86::rcx;

    int rest = N % WIDTH_32BIT_ELEMS;
    asmjit::Label onesLabel = this->assignAuxVecRegisters(rest);

#ifdef FBGEMM_LOG_CODE
    std::ostringstream oss;
    oss << "SpMM_";
    if (is_same<ACC_T, float>::value) {
      oss << "float_";
    } else {
      oss << "int8_";
    }
    if (useAvx512_) {
      if (canUseVNNI) {
        oss << "avx512vnni";
      } else {
        oss << "avx512";
      }
    } else {
      oss << "avx2";
    }
    oss << "_M-" << M;
    oss << "_N-" << N;
    oss << "_K-" << K;
    oss << "_LDA-" << LDA;
    oss << "_LDB-" << LDB;
    oss << "_LDC-" << LDC;
    oss << ".txt";

    FILE* codeLogFile = fopen(oss.str().c_str(), "w");
    asmjit::FileLogger codeLogger(codeLogFile);
    code.setLogger(&codeLogger);
#endif

    asmjit::FuncDetail func;
    func.init(asmjit::FuncSignatureT<
              void,
              typename internal::SpMMTypeTrait<ACC_T>::b_type const*,
              ACC_T*,
              uint64_t>(asmjit::CallConv::kIdHost));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(Breg_, Creg_, this->flagsReg_);

    asmjit::FuncFrame frame;
    frame.init(func);
    frame.setDirtyRegs(x86::Reg::kGroupGp, asmjit::Support::bitMask(8, 9));

    this->prologue(args, frame);
    this->main(frame, code, onesLabel);

    if (rest && !useAvx512_) {
      // alignTo(32);
      if (N > WIDTH_32BIT_ELEMS) {
        a.bind(fullMaskLabel_);
        for (int i = 0; i < WIDTH_32BIT_ELEMS; ++i) {
          a.dd(0xffffffffu);
        }
      }
      a.bind(partialMaskLabel_);
      for (int i = 0; i < rest; ++i) {
        a.dd(0xffffffffu);
      }
      for (int i = rest; i < WIDTH_32BIT_ELEMS; ++i) {
        a.dd(0);
      }
    }

    a.finalize();

#ifdef FBGEMM_LOG_CODE
    fclose(codeLogFile);
#endif
  }
};

template <typename ACC_T>
class MicroKernelFunctionTrait {};

template <>
class MicroKernelFunctionTrait<float> {
 public:
  using type = void (*)(const SpMMTypeTrait<float>::b_type*, float*, uint64_t);
};

template <>
class MicroKernelFunctionTrait<int32_t> {
 public:
  using type =
      void (*)(const SpMMTypeTrait<int32_t>::b_type*, int32_t*, uint64_t);
};

template <typename ACC_T>
typename MicroKernelFunctionTrait<ACC_T>::type generateSpMMfp32_microkernel(
    int M,
    int N,
    int K,
    int LDA,
    int LDB,
    int LDC,
    const typename SpMMTypeTrait<ACC_T>::a_type* AData,
    bool canUseVNNI) {
  static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                // depents on other static
                                // variables.  Required to prevent
                                // initialization order fiasco

  asmjit::CodeHolder code;
  code.init(rt.codeInfo());

  if (fbgemmHasAvx512Support()) {
    SpMM_JitterASMJIT<ACC_T, inst_set_t::avx512> JITSpMM(
        code, M, N, K, LDA, LDB, LDC, AData, canUseVNNI);
  } else {
    SpMM_JitterASMJIT<ACC_T, inst_set_t::avx2> JITSpMM(
        code, M, N, K, LDA, LDB, LDC, AData, canUseVNNI);
  }

  code.flatten();
  code.resolveUnresolvedLinks();

  typename MicroKernelFunctionTrait<ACC_T>::type ret;
  rt.add(&ret, &code);

  return ret;
}

} // anonymous namespace

template <typename ACC_T>
function<void(
    const typename SpMMTypeTrait<ACC_T>::b_type* BData,
    ACC_T* CData,
    uint64_t flags)>
generateSpMM(
    int m,
    int n,
    int k,
    const typename SpMMTypeTrait<ACC_T>::a_type* AData,
    int lda,
    int ldb,
    int ldc) {
  cpuinfo_initialize();
  bool canUseVNNI = fbgemmHasAvx512VnniSupport();
  constexpr int TYPE_SCALE_FACTOR = is_same<ACC_T, int32_t>::value ? 4 : 1;
  assert((k % TYPE_SCALE_FACTOR) == 0);

  // Block K so that each block in B has 192K numbers.
  // TODO: tune based on cache size
  int effK = (192) * 1024 / 4 / n;

  effK = min(effK, k / TYPE_SCALE_FACTOR);
  auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

  // divide as evenly as possible but not larger than effK
  effK =
      ceil_div(k / TYPE_SCALE_FACTOR, ceil_div(k / TYPE_SCALE_FACTOR, effK)) *
      TYPE_SCALE_FACTOR;

  int step = effK * ldb;

  int full = k / effK;
  int rest = k % effK;

  vector<typename MicroKernelFunctionTrait<ACC_T>::type> fns;

  fns.resize(full + (rest ? 1 : 0));

  for (int i = 0; i < full; ++i) {
    fns[i] = generateSpMMfp32_microkernel<ACC_T>(
        m, n, effK, lda, ldb, ldc, AData, canUseVNNI);

    AData += effK;
  }

  if (rest) {
    fns[full] = generateSpMMfp32_microkernel<ACC_T>(
        m, n, rest, lda, ldb, ldc, AData, canUseVNNI);
  }

  return [=](typename SpMMTypeTrait<ACC_T>::b_type const* b,
             ACC_T* c,
             uint64_t flags) {
    fns[0](b, c, flags);
    b += step;

    for (int i = 1; i < fns.size(); ++i) {
      fns[i](b, c, 1);
      b += step;
    }
  };
}

template FBGEMM_API
    function<void(const float* BData, float* CData, uint64_t flags)>
    generateSpMM(
        int m,
        int n,
        int k,
        const float* AData,
        int lda,
        int ldb,
        int ldc);

template FBGEMM_API
    function<void(const uint8_t* BData, int32_t* CData, uint64_t flags)>
    generateSpMM(
        int m,
        int n,
        int k,
        const int8_t* AData,
        int lda,
        int ldb,
        int ldc);

namespace {

/**
 * A version of SpMM jitter that supports variable N
 */
template <typename ACC_T, inst_set_t instSet>
class SpMM_JitterASMJIT_VaryingN {
 private:
  using a_temp_type = typename SpMMTypeTrait<ACC_T>::a_temp_type;
  a_temp_type const* AData_;

  int M_;
  // int N_;
  int K_;
  int LDA_;

  x86::Gp Breg_ = x86::rdi;
  x86::Gp Creg_ = x86::rsi;
  x86::Gpd Nreg_ = x86::rdx.r32();
  x86::Gpd LDBreg_ = x86::rcx.r32();
  x86::Gp LDCreg_ = x86::r8;
  x86::Gp flagsReg_ = x86::r9;

  x86::Gp NLoopReg_ = x86::r10;
  x86::Gpd tmpReg_ = x86::r11.r32();
  x86::Gp COffsetReg_ = x86::r12;
  x86::Gp dataReg_ = x86::r13;

  std::vector<a_temp_type> compressedAData_;

  x86::Builder a;
  asmjit::Label dataLabel_;

  // For AVX2
  asmjit::Label maskLabel_;

  using vec_reg_t = typename simd_info<instSet>::vec_reg_t;
  vec_reg_t onesVReg_;

  // For AVX512
  x86::KReg maskReg_ = x86::k(1);
  // For AVX2
  x86::Ymm maskVReg_;
  x86::Ymm broadcastVReg_;

  std::vector<std::function<void()>> toAppend_;

  bool canUseVNNI_{false};
  bool useAvx512_{false};

  // First available register to store weights and activations after
  // aux registers for storing masks, ones, broadcasting, and so on.
  int firstAvailableVecRegister_ = 0;

  static inline bool is_zero(a_temp_type val) {
    return val == 0;
  }

  static bool has_any_non_zeros(a_temp_type const* data, int N, int stride) {
    for (int n = 0; n < N; ++n) {
      if (!is_zero(data[n * stride])) {
        return true;
      }
    }
    return false;
  }

  int WIDTH_BYTES;
  int WIDTH_32BIT_ELEMS;

  // Multiply rowRegBlock rows of A with <=16 B columns (by default 16 columns
  // that span 1 cache line)
  void emitRegisterBlockedLoop(
      int AOffset,
      int rowRegBlock,
      asmjit::Label myLabel,
      bool hasMask) {
    a.bind(myLabel);

    constexpr int TYPE_SCALE_FACTOR =
        std::is_same<ACC_T, int32_t>::value ? 4 : 1;
    a_temp_type const* Adata = AData_ + AOffset / TYPE_SCALE_FACTOR;

    std::vector<vec_reg_t> Cvalues(rowRegBlock);

    bool needAuxRegisters = std::is_same<ACC_T, int32_t>::value && !canUseVNNI_;
    int nextRegister = firstAvailableVecRegister_;

    for (int r = 0; r < rowRegBlock; ++r) {
      Cvalues[r] = vec_reg_t(nextRegister++);
    }

    asmjit::Label doneInitializing = a.newLabel();
    asmjit::Label initZeros = a.newLabel();

    a.cmp(flagsReg_, 0);
    a.je(initZeros);

    a.mov(COffsetReg_, 0);
    for (int r = 0; r < rowRegBlock; ++r) {
      if (r > 0) {
        a.add(COffsetReg_, LDCreg_);
      }
      auto src_ptr = x86::ptr(Creg_, COffsetReg_);
      if (hasMask) {
        if (useAvx512_) {
          a.k(maskReg_).vmovups(Cvalues[r], src_ptr);
        } else {
          a.vmaskmovps(x86::Ymm(Cvalues[r].id()), maskVReg_, src_ptr);
        }
      } else {
        a.vmovups(Cvalues[r], src_ptr);
      }
    }

    a.jmp(doneInitializing);

    a.bind(initZeros);

    // Initialize outputs to zero
    for (int r = 0; r < rowRegBlock; ++r) {
      if (useAvx512_) {
        // vxorpd requires avx512dq so use vpxord to use avx512f
        a.vpxord(Cvalues[r], Cvalues[r], Cvalues[r]);
      } else {
        a.vxorpd(Cvalues[r], Cvalues[r], Cvalues[r]);
      }
    }

    a.bind(doneInitializing);

    constexpr int NUM_VEC_REGISTERS = simd_info<instSet>::NUM_VEC_REGS;
    assert(nextRegister < NUM_VEC_REGISTERS);

    int numActivationRegisters = NUM_VEC_REGISTERS - nextRegister;

    if (needAuxRegisters) {
      numActivationRegisters /= 2;
    }

    std::vector<vec_reg_t> activationRegisters(numActivationRegisters);
    std::vector<vec_reg_t> auxRegisters(
        needAuxRegisters ? numActivationRegisters : 0);

    for (auto& activationRegister : activationRegisters) {
      activationRegister = vec_reg_t(nextRegister++);
    }

    for (auto& auxRegister : auxRegisters) {
      auxRegister = vec_reg_t(nextRegister++);
    }

    assert(nextRegister <= NUM_VEC_REGISTERS);

    std::vector<std::function<void()>> delayedFMAs(numActivationRegisters);

    for (auto& fn : delayedFMAs) {
      fn = []() {};
    }

    int delayedIndex = 0;

    a.lea(
        dataReg_,
        x86::ptr(
            dataLabel_,
            static_cast<int>(compressedAData_.size() * sizeof(ACC_T))));

    int dataOffset = 0;
    constexpr int MAX_DATA_OFFSET = 256;
    a.mov(COffsetReg_, 0);
    for (int k = 0; k < K_; k += TYPE_SCALE_FACTOR) {
      if (has_any_non_zeros(
              Adata + k / TYPE_SCALE_FACTOR,
              rowRegBlock,
              LDA_ / TYPE_SCALE_FACTOR)) {
        delayedFMAs[delayedIndex]();

        a.imul(
            COffsetReg_,
            LDBreg_,
            static_cast<asmjit::Imm>(
                k * sizeof(typename SpMMTypeTrait<ACC_T>::b_type)));
        auto src_ptr = x86::ptr(Breg_, COffsetReg_);
        if (hasMask) {
          if (useAvx512_) {
            a.k(maskReg_).vmovups(activationRegisters[delayedIndex], src_ptr);
          } else {
            a.vmaskmovps(
                x86::Ymm(activationRegisters[delayedIndex].id()),
                maskVReg_,
                src_ptr);
          }
        } else {
          a.vmovups(activationRegisters[delayedIndex], src_ptr);
        }

        delayedFMAs[delayedIndex] = [&, k, delayedIndex]() {
          for (int r = 0; r < rowRegBlock; ++r) {
            if (!is_zero(Adata
                             [k / TYPE_SCALE_FACTOR +
                              r * (LDA_ / TYPE_SCALE_FACTOR)])) {
              compressedAData_.push_back(
                  Adata
                      [k / TYPE_SCALE_FACTOR + r * (LDA_ / TYPE_SCALE_FACTOR)]);

              if (dataOffset * sizeof(a_temp_type) == MAX_DATA_OFFSET) {
                a.add(dataReg_, MAX_DATA_OFFSET);
                dataOffset = 0;
              }

              auto ptr = x86::ptr(dataReg_, dataOffset * sizeof(4));
              if (std::is_same<float, ACC_T>::value) {
                if (useAvx512_) {
                  a.vfmadd231ps(
                      Cvalues[r],
                      activationRegisters[delayedIndex],
                      ptr._1to16());
                } else {
                  a.vbroadcastss(broadcastVReg_, ptr);
                  a.vfmadd231ps(
                      Cvalues[r],
                      activationRegisters[delayedIndex],
                      vec_reg_t(broadcastVReg_.id()));
                }
              } else if (canUseVNNI_) {
                a.vpdpbusd(
                    Cvalues[r],
                    activationRegisters[delayedIndex],
                    ptr._1to16());
              } else {
                a.vbroadcastss(auxRegisters[delayedIndex], ptr);

                a.vpmaddubsw(
                    auxRegisters[delayedIndex],
                    activationRegisters[delayedIndex],
                    auxRegisters[delayedIndex]);

                a.vpmaddwd(
                    auxRegisters[delayedIndex],
                    onesVReg_,
                    auxRegisters[delayedIndex]);

                a.vpaddd(Cvalues[r], Cvalues[r], auxRegisters[delayedIndex]);
              }

              ++dataOffset;
            }
          }
        };

        delayedIndex = (delayedIndex + 1) % numActivationRegisters;
      }
    }

    for (int i = 0; i < numActivationRegisters; ++i) {
      delayedFMAs[delayedIndex]();
      delayedIndex = (delayedIndex + 1) % numActivationRegisters;
    }

    // Store results
    a.mov(COffsetReg_, static_cast<asmjit::Imm>(0));
    for (int r = 0; r < rowRegBlock; ++r) {
      if (r > 0) {
        a.add(COffsetReg_, LDCreg_);
      }
      auto dst_ptr = x86::ptr(Creg_, COffsetReg_);
      if (hasMask) {
        if (useAvx512_) {
          a.k(maskReg_).vmovups(dst_ptr, Cvalues[r]);
        } else {
          a.vmaskmovps(dst_ptr, maskVReg_, x86::Ymm(Cvalues[r].id()));
        }
      } else {
        a.vmovups(dst_ptr, Cvalues[r]);
      }
    }

    a.ret();
  }

  asmjit::Label
  registerBlockedLoop(int AOffset, int rowRegBlock, bool hasMask) {
    asmjit::Label rbLoopLabel = a.newLabel();

    toAppend_.push_back([=]() {
      emitRegisterBlockedLoop(AOffset, rowRegBlock, rbLoopLabel, hasMask);
    });

    return rbLoopLabel;
  }

  // Multiply rowRegBlock rows of A with B
  void loopOverN(int AOffset, int rowRegBlock, bool restorePointers) {
    asmjit::Label fnLabel =
        this->registerBlockedLoop(AOffset, rowRegBlock, true);

    if (restorePointers) {
      a.push(Creg_);
      a.push(Breg_);
    }

    asmjit::Label loopLabel = a.newLabel();
    asmjit::Label loopEndLabel = a.newLabel();

    a.mov(NLoopReg_, Nreg_);
    a.cmp(NLoopReg_, WIDTH_32BIT_ELEMS);
    a.jl(loopEndLabel);

    if (useAvx512_) {
      a.mov(tmpReg_, 0xffff);
      a.kmovw(maskReg_, tmpReg_);
    } else {
      a.vmovups(maskVReg_, x86::ptr(maskLabel_));
    }

    a.bind(loopLabel);

    a.sub(NLoopReg_, WIDTH_32BIT_ELEMS);

    a.call(fnLabel);

    a.add(Breg_, WIDTH_BYTES);
    a.add(Creg_, WIDTH_BYTES);

    a.cmp(NLoopReg_, WIDTH_32BIT_ELEMS);
    a.jge(loopLabel);

    a.bind(loopEndLabel);

    asmjit::Label restLabel = a.newLabel();
    a.cmp(NLoopReg_, 0);
    a.jle(restLabel);

    if (useAvx512_) {
      // Use COffsetReg_ as a temporary register
      a.mov(tmpReg_, 1);
      a.shlx(tmpReg_, tmpReg_, NLoopReg_);
      a.dec(tmpReg_);
      a.kmovw(maskReg_, tmpReg_);
    } else {
      a.mov(tmpReg_, static_cast<asmjit::Imm>(WIDTH_32BIT_ELEMS));
      a.sub(tmpReg_.r64(), NLoopReg_);
      a.lea(COffsetReg_, x86::ptr(maskLabel_));
      a.vmovups(maskVReg_, x86::ptr(COffsetReg_, tmpReg_.r64(), 2 /* shift */));
    }
    a.call(fnLabel);

    a.bind(restLabel);

    if (restorePointers) {
      a.pop(Breg_);
      a.pop(Creg_);
    }
  }

  void loopOverM(int rowRegBlock) {
    int AOffset = 0;
    int numFullIterations = M_ / rowRegBlock;
    int restSize = M_ % rowRegBlock;

    for (int n = 0; n < numFullIterations; ++n) {
      bool restorePointers = n < numFullIterations - 1 || restSize;
      loopOverN(AOffset, rowRegBlock, restorePointers);

      if (restorePointers) {
        AOffset += LDA_ * rowRegBlock;
        a.mov(COffsetReg_, LDCreg_);
        a.imul(COffsetReg_, static_cast<asmjit::Imm>(rowRegBlock));
        a.add(Creg_, COffsetReg_);
      }
    }

    if (restSize) {
      loopOverN(AOffset, restSize, false);
    }
  }

  /**
   * @return onesLabel
   */
  asmjit::Label assignAuxVecRegisters(bool useMask) {
    asmjit::Label onesLabel;
    firstAvailableVecRegister_ = 0;
    if (std::is_same<ACC_T, int32_t>::value) {
      assert(K_ % 4 == 0);

      if (!canUseVNNI_) {
        onesLabel = a.newLabel();
        onesVReg_ = vec_reg_t(firstAvailableVecRegister_);
        ++firstAvailableVecRegister_;
        a.vbroadcastss(onesVReg_, x86::ptr(onesLabel));
      }
    }
    if (!useAvx512_) {
      if (useMask) {
        maskVReg_ = x86::Ymm(firstAvailableVecRegister_);
        ++firstAvailableVecRegister_;
      }
      if (std::is_same<ACC_T, float>::value) {
        broadcastVReg_ = x86::Ymm(firstAvailableVecRegister_);
        ++firstAvailableVecRegister_;
      }
    }
    return onesLabel;
  }

  int getRowRegBlock() {
    // TODO: determine best blockings
    if (useAvx512_) {
      if (std::is_same<ACC_T, int32_t>::value && !canUseVNNI_) {
        // remaining registers are used in the following way:
        // 1: all ones
        // 7: activation registers
        // 7: auxiliary registers
        return 16;
      } else {
        // remaining 8 used for activation registers
        return 24;
      }
    } else {
      if (std::is_same<ACC_T, int32_t>::value) {
        // remaining registers are used in the following way:
        // 1: all ones
        // 0 or 1: mask
        // 4: activation registers
        // 4: auxiliary registers
        return 6;
      } else {
        // remaining registers are used in the following way:
        // 1: broadcasting
        // 0 or 1: mask
        // 3 or 4: activation registers
        return 11;
      }
    }
  }

  void prologue(asmjit::FuncArgsAssignment& args, asmjit::FuncFrame& frame) {
    if (useAvx512_) {
      frame.setDirtyRegs(
          x86::Reg::kGroupVec,
          asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
              asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15) |
              asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
              asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31));
    } else {
      frame.setDirtyRegs(
          x86::Reg::kGroupVec,
          asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
              asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));
    }

    args.updateFuncFrame(frame);
    frame.finalize();

    a.emitProlog(frame);
    a.emitArgsAssignment(frame, args);
  }

  void main(
      asmjit::FuncFrame& frame,
      asmjit::CodeHolder& code,
      asmjit::Label onesLabel) {
    int rowRegBlock = getRowRegBlock();
    loopOverM(rowRegBlock);
    a.emitEpilog(frame);

    for (auto const& f : toAppend_) {
      f();
    }

    asmjit::Section* data;

    code.newSection(
        &data,
        ".data", // Section name
        SIZE_MAX, // Name length if the name is not
                  // null terminated (or SIZE_MAX).
        0, // Section flags, see Section::Flags.
        64);

    // alignTo(4);
    a.section(data); // Switches to the end of .data section.

    if (std::is_same<ACC_T, int32_t>::value && !canUseVNNI_) {
      a.bind(onesLabel);
      a.dw(1);
      a.dw(1);
    }

    // alignTo(4);
    a.bind(dataLabel_);
    for (auto f : compressedAData_) {
      a.dd(internal::bit_cast<uint32_t>(f));
    }
  }

 public:
  SpMM_JitterASMJIT_VaryingN(
      asmjit::CodeHolder& code,
      int M,
      // int N,
      int K,
      int LDA,
      const typename SpMMTypeTrait<ACC_T>::a_type* aData,
      bool canUseVNNI = false)
      : AData_(reinterpret_cast<const a_temp_type*>(aData)),
        M_(M),
        // N_(N),
        K_(K),
        LDA_(LDA),
        a(&code),
        dataLabel_(a.newLabel()),
        maskLabel_(a.newLabel()),
        canUseVNNI_(canUseVNNI) {
    useAvx512_ = instSet == inst_set_t::avx512;
    if (useAvx512_) {
      WIDTH_BYTES = simd_info<inst_set_t::avx512>::WIDTH_BYTES;
      WIDTH_32BIT_ELEMS = simd_info<inst_set_t::avx512>::WIDTH_32BIT_ELEMS;
    } else {
      WIDTH_BYTES = simd_info<inst_set_t::avx2>::WIDTH_BYTES;
      WIDTH_32BIT_ELEMS = simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
    }

    asmjit::Label onesLabel = this->assignAuxVecRegisters(true);

#ifdef FBGEMM_LOG_CODE
    std::ostringstream oss;
    oss << "SpMM_VaryingN_";
    if (is_same<ACC_T, float>::value) {
      oss << "float_";
    } else {
      oss << "int8_";
    }
    if (useAvx512_) {
      if (canUseVNNI) {
        oss << "avx512vnni";
      } else {
        oss << "avx512";
      }
    } else {
      oss << "avx2";
    }
    oss << "_M-" << M;
    oss << "_K-" << K;
    oss << "_LDA-" << LDA;
    oss << ".txt";

    FILE* codeLogFile = fopen(oss.str().c_str(), "w");
    asmjit::FileLogger codeLogger(codeLogFile);
    code.setLogger(&codeLogger);
#endif

    asmjit::FuncDetail func;
    func.init(asmjit::FuncSignatureT<
              void,
              typename internal::SpMMTypeTrait<ACC_T>::b_type const*,
              ACC_T*,
              int,
              int,
              int,
              uint64_t>(asmjit::CallConv::kIdHost));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(
        Breg_, Creg_, Nreg_, LDBreg_, LDCreg_.r32(), this->flagsReg_);

    asmjit::FuncFrame frame;
    frame.init(func);
    frame.setDirtyRegs(
        x86::Reg::kGroupGp, asmjit::Support::bitMask(8, 9, 10, 11, 12, 13));

    this->prologue(args, frame);

    a.imul(LDCreg_, static_cast<asmjit::Imm>(sizeof(a_temp_type)));

    this->main(frame, code, onesLabel);

    if (!useAvx512_) {
      a.bind(maskLabel_);
      for (int i = 0; i < WIDTH_32BIT_ELEMS; ++i) {
        a.dd(0xffffffffu);
      }
      for (int i = 0; i < WIDTH_32BIT_ELEMS - 1; ++i) {
        a.dd(0);
      }
    }

    a.finalize();

#ifdef FBGEMM_LOG_CODE
    fclose(codeLogFile);
#endif
  }

  virtual ~SpMM_JitterASMJIT_VaryingN() {}
}; // SpMM_JitterASMJIT_VaryingN

template <typename ACC_T>
class MicroKernelFunctionVaryingNTrait {};

template <>
class MicroKernelFunctionVaryingNTrait<float> {
 public:
  using type = void (*)(
      const SpMMTypeTrait<float>::b_type*,
      float*,
      int,
      int,
      int,
      uint64_t);
};

template <>
class MicroKernelFunctionVaryingNTrait<int32_t> {
 public:
  using type = void (*)(
      const SpMMTypeTrait<int32_t>::b_type*,
      int32_t*,
      int,
      int,
      int,
      uint64_t);
};

} // anonymous namespace

template <typename ACC_T>
typename MicroKernelFunctionVaryingNTrait<ACC_T>::type
generateSpMMfp32VaryingN_microkernel(
    int M,
    int K,
    int LDA,
    const typename SpMMTypeTrait<ACC_T>::a_type* AData,
    bool canUseVNNI) {
  static asmjit::JitRuntime rt; //< JIT Runtime for asmjit,
                                // depents on other static
                                // variables.  Required to prevent
                                // initialization order fiasco

  asmjit::CodeHolder code;
  code.init(rt.codeInfo());

  if (fbgemmHasAvx512Support()) {
    SpMM_JitterASMJIT_VaryingN<ACC_T, inst_set_t::avx512> JITSpMM(
        code, M, K, LDA, AData, canUseVNNI);
  } else {
    SpMM_JitterASMJIT_VaryingN<ACC_T, inst_set_t::avx2> JITSpMM(
        code, M, K, LDA, AData, canUseVNNI);
  }

  code.flatten();
  code.resolveUnresolvedLinks();

  typename MicroKernelFunctionVaryingNTrait<ACC_T>::type ret;
  rt.add(&ret, &code);

  return ret;
}

template <typename ACC_T>
function<void(
    const typename SpMMTypeTrait<ACC_T>::b_type* BData,
    ACC_T* CData,
    int N,
    int LDB,
    int LDC,
    uint64_t flags)>
generateSpMM(
    int m,
    int k,
    const typename SpMMTypeTrait<ACC_T>::a_type* AData,
    int lda) {
  cpuinfo_initialize();
  bool canUseVNNI = fbgemmHasAvx512VnniSupport();
  constexpr int TYPE_SCALE_FACTOR = is_same<ACC_T, int32_t>::value ? 4 : 1;
  assert((k % TYPE_SCALE_FACTOR) == 0);

  // Block K so that each block in B has 192K numbers.
  // TODO: tune based on cache size
  int n_guessed = 1024;
  int effK = (192) * 1024 / 4 / n_guessed;

  effK = min(effK, k / TYPE_SCALE_FACTOR);
  auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

  // divide as evenly as possible but not larger than effK
  effK =
      ceil_div(k / TYPE_SCALE_FACTOR, ceil_div(k / TYPE_SCALE_FACTOR, effK)) *
      TYPE_SCALE_FACTOR;

  int full = k / effK;
  int rest = k % effK;

  vector<typename MicroKernelFunctionVaryingNTrait<ACC_T>::type> fns;

  fns.resize(full + (rest ? 1 : 0));

  for (int i = 0; i < full; ++i) {
    fns[i] = generateSpMMfp32VaryingN_microkernel<ACC_T>(
        m, effK, lda, AData, canUseVNNI);

    AData += effK;
  }

  if (rest) {
    fns[full] = generateSpMMfp32VaryingN_microkernel<ACC_T>(
        m, rest, lda, AData, canUseVNNI);
  }

  return [=](typename SpMMTypeTrait<ACC_T>::b_type const* b,
             ACC_T* c,
             int N,
             int LDB,
             int LDC,
             uint64_t flags) {
    fns[0](b, c, N, LDB, LDC, flags);
    b += effK * LDB;

    for (int i = 1; i < fns.size(); ++i) {
      fns[i](b, c, N, LDB, LDC, 1);
      b += effK * LDB;
    }
  };
}

template FBGEMM_API function<void(
    const float* BData,
    float* CData,
    int N,
    int LDB,
    int LDC,
    uint64_t flags)>
generateSpMM(int m, int k, const float* AData, int lda);

template FBGEMM_API function<void(
    const uint8_t* BData,
    int32_t* CData,
    int N,
    int LDB,
    int LDC,
    uint64_t flags)>
generateSpMM(int m, int k, const int8_t* AData, int lda);

} // namespace fbgemm
