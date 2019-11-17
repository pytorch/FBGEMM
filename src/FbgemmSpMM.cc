#include "fbgemm/FbgemmSpMM.h"
#include "./FbgemmSpMM-inl.h"

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
