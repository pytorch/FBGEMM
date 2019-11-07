#include "fbgemm/FbgemmSpMM.h"

#include <cpuinfo.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <fbgemm/Utils.h>

using namespace std;

namespace fbgemm {

namespace x86 = asmjit::x86;

using internal::SpMMTypeTrait;

namespace {

template <class To, class From>
typename enable_if<
    (sizeof(To) == sizeof(From)) && is_trivially_copyable<From>::value &&
        is_trivial<To>::value,
    To>::type
bit_cast(const From& src) noexcept {
  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename ACC_T, inst_set_t instSet>
class SpMM_JitterASMJIT {
 private:
  using a_temp_type = typename SpMMTypeTrait<ACC_T>::a_temp_type;
  a_temp_type const* AData_;

  int M_;
  int N_;
  int K_;
  int LDA_, LDB_, LDC_;

  x86::Gp Breg_ = x86::rdi;
  x86::Gp Creg_ = x86::rsi;
  x86::Gp flagsReg_ = x86::rdx;
  x86::Gp dataReg_ = x86::rcx;
  x86::Gp NLoopReg_ = x86::r8;
  x86::Gpd tmpReg_ = x86::r9.r32();

  vector<a_temp_type> compressedAData_;

  x86::Builder a;
  asmjit::Label dataLabel_;
  // For AVX2
  asmjit::Label fullMaskLabel_;
  asmjit::Label partialMaskLabel_;

  using vec_reg_t = typename simd_info<instSet>::vec_reg_t;
  vec_reg_t onesVReg_;

  // For AVX512
  x86::KReg maskReg_ = x86::k(1);
  // For AVX2
  x86::Ymm maskVReg_;
  x86::Ymm broadcastVReg_;

  vector<function<void()>> toAppend_;

  bool canUseVNNI_{false};
  bool useAvx512_{false};

  // First available register to store weights and activations after
  // aux registers for storing masks, ones, broadcasting, and so on.
  int firstAvailableVecRegister_ = 0;

  static bool is_zero(a_temp_type val) {
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

    constexpr int TYPE_SCALE_FACTOR = is_same<ACC_T, int32_t>::value ? 4 : 1;
    a_temp_type const* Adata = AData_ + AOffset / TYPE_SCALE_FACTOR;

    vector<vec_reg_t> Cvalues(rowRegBlock);

    bool needAuxRegisters = is_same<ACC_T, int32_t>::value && !canUseVNNI_;
    int nextRegister = firstAvailableVecRegister_;

    for (int r = 0; r < rowRegBlock; ++r) {
      Cvalues[r] = vec_reg_t(nextRegister++);
    }

    asmjit::Label doneInitializing = a.newLabel();
    asmjit::Label initZeros = a.newLabel();

    a.cmp(flagsReg_, 0);
    a.je(initZeros);

    for (int r = 0; r < rowRegBlock; ++r) {
      auto src_ptr = x86::ptr(Creg_, r * LDC_ * sizeof(a_temp_type));
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
        // vxorpd requires avx512dq
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

    vector<vec_reg_t> activationRegisters(numActivationRegisters);
    vector<vec_reg_t> auxRegisters(
        needAuxRegisters ? numActivationRegisters : 0);

    for (auto& activationRegister : activationRegisters) {
      activationRegister = vec_reg_t(nextRegister++);
    }

    for (auto& auxRegister : auxRegisters) {
      auxRegister = vec_reg_t(nextRegister++);
    }

    assert(nextRegister <= NUM_VEC_REGISTERS);

    vector<function<void()>> delayedFMAs(numActivationRegisters);

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
    for (int k = 0; k < K_; k += TYPE_SCALE_FACTOR) {
      if (has_any_non_zeros(
              Adata + k / TYPE_SCALE_FACTOR,
              rowRegBlock,
              LDA_ / TYPE_SCALE_FACTOR)) {
        delayedFMAs[delayedIndex]();

        auto src_ptr = x86::ptr(
            Breg_, k * LDB_ * sizeof(typename SpMMTypeTrait<ACC_T>::b_type));
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
              if (is_same<float, ACC_T>::value) {
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
    for (int r = 0; r < rowRegBlock; ++r) {
      auto dst_ptr = x86::ptr(Creg_, r * LDC_ * sizeof(a_temp_type));
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
    int fullIterations = N_ / WIDTH_32BIT_ELEMS;
    int rest = N_ % WIDTH_32BIT_ELEMS;

    asmjit::Label fnLabel = registerBlockedLoop(AOffset, rowRegBlock, rest);

    bool updatesPointerRegisters =
        ((fullIterations > 1) || (fullIterations == 1 && rest)) &&
        restorePointers;

    if (fullIterations > 0 && rest) {
      if (useAvx512_) {
        a.mov(tmpReg_, 0xffff);
        a.kmovw(maskReg_, tmpReg_);
      } else {
        a.vmovups(maskVReg_, x86::ptr(fullMaskLabel_));
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
        a.kmovw(maskReg_, tmpReg_);
      } else {
        a.vmovups(maskVReg_, x86::ptr(partialMaskLabel_));
      }

      a.call(fnLabel);
    }

    if (updatesPointerRegisters) {
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
        a.add(
            Creg_,
            static_cast<asmjit::Imm>(LDC_ * sizeof(int32_t) * rowRegBlock));
      }
    }

    if (restSize) {
      loopOverN(AOffset, restSize, false);
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
      : AData_(reinterpret_cast<const a_temp_type*>(aData)),
        M_(M),
        N_(N),
        K_(K),
        LDA_(LDA),
        LDB_(LDB),
        LDC_(LDC),
        a(&code),
        dataLabel_(a.newLabel()),
        fullMaskLabel_(a.newLabel()),
        partialMaskLabel_(a.newLabel()),
        canUseVNNI_(canUseVNNI) {
    useAvx512_ = instSet == inst_set_t::avx512;
    if (useAvx512_) {
      WIDTH_BYTES = simd_info<inst_set_t::avx512>::WIDTH_BYTES;
      WIDTH_32BIT_ELEMS = simd_info<inst_set_t::avx512>::WIDTH_32BIT_ELEMS;
    } else {
      WIDTH_BYTES = simd_info<inst_set_t::avx2>::WIDTH_BYTES;
      WIDTH_32BIT_ELEMS = simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
    }

    int rest = N % WIDTH_32BIT_ELEMS;

    asmjit::Label onesLabel;
    firstAvailableVecRegister_ = 0;
    if (is_same<ACC_T, int32_t>::value) {
      assert(K % 4 == 0);

      if (!canUseVNNI) {
        onesLabel = a.newLabel();
        onesVReg_ = vec_reg_t(firstAvailableVecRegister_);
        ++firstAvailableVecRegister_;
        a.vbroadcastss(onesVReg_, x86::ptr(onesLabel));
      }
    }
    if (!useAvx512_) {
      if (rest) {
        maskVReg_ = x86::Ymm(firstAvailableVecRegister_);
        ++firstAvailableVecRegister_;
      }
      if (is_same<ACC_T, float>::value) {
        broadcastVReg_ = x86::Ymm(firstAvailableVecRegister_);
        ++firstAvailableVecRegister_;
      }
    }

#ifdef FBGEMM_LOG_CODE
    std::ostringstream oss;
    oss << "spmm_";
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

    FILE* codeLogFile = fopen(ost.str(), "w");
    asmjit::FileLogger codeLogger(codeLogFile);
    code.setLogger(&codeLogger);
#endif

    asmjit::FuncDetail func;
    func.init(asmjit::FuncSignatureT<
              void,
              typename internal::SpMMTypeTrait<ACC_T>::b_type const*,
              ACC_T*,
              uint64_t>(asmjit::CallConv::kIdHost));

    asmjit::FuncFrame frame;
    frame.init(func);
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
    frame.setDirtyRegs(x86::Reg::kGroupGp, asmjit::Support::bitMask(8, 9));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(Breg_, Creg_, flagsReg_);

    args.updateFuncFrame(frame);
    frame.finalize();

    a.emitProlog(frame);
    a.emitArgsAssignment(frame, args);

    // TODO: determine best blockings
    int rowRegBlock;
    if (useAvx512_) {
      if (is_same<ACC_T, int32_t>::value && !canUseVNNI) {
        // remaining registers are used in the following way:
        // 1: all ones
        // 7: activation registers
        // 7: auxiliary registers
        rowRegBlock = 16;
      } else {
        // remaining 8 used for activation registers
        rowRegBlock = 24;
      }
    } else {
      if (is_same<ACC_T, int32_t>::value) {
        // remaining registers are used in the following way:
        // 1: all ones
        // 0 or 1: mask
        // 4: activation registers
        // 4: auxiliary registers
        rowRegBlock = 6;
      } else {
        // remaining registers are used in the following way:
        // 1: broadcasting
        // 0 or 1: mask
        // 3 or 4: activation registers
        rowRegBlock = 11;
      }
    }
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
        4);

    // alignTo(4);
    a.section(data); // Switches to the end of .data section.

    if (is_same<ACC_T, int32_t>::value && !canUseVNNI) {
      a.bind(onesLabel);
      a.dw(1);
      a.dw(1);
    }

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

    // alignTo(4);
    a.bind(dataLabel_);
    for (auto f : compressedAData_) {
      a.dd(bit_cast<uint32_t>(f));
    }

    a.finalize();

#ifdef FBGEMM_LOG_CODE
    fclose(codeLogFile);
#endif
  }
};

namespace {
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
}; // anonymous namespace

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
        m, rest, n, lda, ldb, ldc, AData, canUseVNNI);
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

template function<void(const float* BData, float* CData, uint64_t flags)>
generateSpMM(
    int m,
    int n,
    int k,
    const float* AData,
    int lda,
    int ldb,
    int ldc);

template function<void(const uint8_t* BData, int32_t* CData, uint64_t flags)>
generateSpMM(
    int m,
    int n,
    int k,
    const int8_t* AData,
    int lda,
    int ldb,
    int ldc);

} // namespace fbgemm
