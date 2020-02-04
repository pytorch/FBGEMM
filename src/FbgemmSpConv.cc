/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmSpConv.h"
#include "./FbgemmSpMM-inl.h"
#include "fbgemm/Utils.h"

#include <cpuinfo.h>
#include <algorithm>
#include <cassert>
#include <cstring>

using namespace std;

namespace fbgemm {

using internal::SpMMTypeTrait;

namespace {

template <typename ACC_T, inst_set_t instSet>
class SpConv_JitterASMJIT
    : public internal::SpMM_JitterASMJIT_base<ACC_T, instSet> {
 private:
  x86::Gp numGroupsOf16Reg_ = x86::rdx;
  x86::Gp masksPtrReg_ = x86::rcx;
  x86::Gp NLoopReg_;
  x86::Gp NLoopEndReg_;

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
    asmjit::Label fnLabel =
        this->registerBlockedLoop(AOffset, rowRegBlock, true /*hasMask*/);

    a.mov(NLoopReg_, masksPtrReg_);

    if (restorePointers) {
      a.push(Creg_);
      a.push(Breg_);
    }

    asmjit::Label loopLabel = a.newLabel();

    a.bind(loopLabel);
    if (useAvx512_) {
      a.kmovw(this->maskReg_, x86::word_ptr(NLoopReg_));
      a.add(NLoopReg_, 2); // each mask has 2 Bytes
    } else {
      a.vmovups(this->maskVReg_, x86::word_ptr(NLoopReg_));
      a.add(NLoopReg_, WIDTH_BYTES); // each mask has 32 Bytes
    }

    a.call(fnLabel);

    a.add(Breg_, WIDTH_BYTES);
    a.add(Creg_, WIDTH_BYTES);

    a.cmp(NLoopReg_, NLoopEndReg_);
    a.jl(loopLabel);

    if (restorePointers) {
      a.pop(Breg_);
      a.pop(Creg_);
    }
  }

 public:
  SpConv_JitterASMJIT(
      asmjit::CodeHolder& code,
      int M,
      int K,
      int LDA,
      int LDB,
      int LDC,
      const typename SpMMTypeTrait<ACC_T>::a_type* aData,
      bool canUseVNNI = false)
      : BaseType(code, M, K, LDA, LDB, LDC, aData, canUseVNNI) {
    this->flagsReg_ = x86::r8;
    NLoopReg_ = x86::r9;
    NLoopEndReg_ = x86::r10;
    this->dataReg_ = x86::r11;

    asmjit::Label onesLabel = this->assignAuxVecRegisters(true /* useMask */);

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
              uint64_t,
              uint16_t const*,
              uint64_t>(asmjit::CallConv::kIdHost));

    asmjit::FuncArgsAssignment args(&func);
    args.assignAll(
        Breg_, Creg_, numGroupsOf16Reg_, masksPtrReg_, this->flagsReg_);

    asmjit::FuncFrame frame;
    frame.init(func);
    frame.setDirtyRegs(
        x86::Reg::kGroupGp, asmjit::Support::bitMask(8, 9, 10, 11));

    this->prologue(args, frame);

    a.imul(this->dataReg_, numGroupsOf16Reg_, 2);
    a.lea(NLoopEndReg_, x86::ptr(masksPtrReg_, this->dataReg_));

    this->main(frame, code, onesLabel);

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
  using type = void (*)(
      const SpMMTypeTrait<float>::b_type*,
      float*,
      uint64_t,
      const uint16_t*,
      uint64_t);
};

template <>
class MicroKernelFunctionTrait<int32_t> {
 public:
  using type = void (*)(
      const SpMMTypeTrait<int32_t>::b_type*,
      int32_t*,
      uint64_t,
      const uint16_t*,
      uint64_t);
};
}; // anonymous namespace

template <typename ACC_T>
typename MicroKernelFunctionTrait<ACC_T>::type generateConv_microkernel(
    int M,
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
    SpConv_JitterASMJIT<ACC_T, inst_set_t::avx512> JITSpMM(
        code, M, K, LDA, LDB, LDC, AData, canUseVNNI);
  } else {
    SpConv_JitterASMJIT<ACC_T, inst_set_t::avx2> JITSpMM(
        code, M, K, LDA, LDB, LDC, AData, canUseVNNI);
  }

  code.flatten();
  code.resolveUnresolvedLinks();

  typename MicroKernelFunctionTrait<ACC_T>::type ret;
  rt.add(&ret, &code);

  return ret;
}

} // anonymous namespace

template <typename ACC_T>
function<void(const typename SpMMTypeTrait<ACC_T>::b_type* BData, ACC_T* CData)>
generateSpConv(
    int Cin,
    int Cout,
    int IY,
    int IX,
    const typename SpMMTypeTrait<ACC_T>::a_type* KData) {
  cpuinfo_initialize();
  bool canUseVNNI = fbgemmHasAvx512VnniSupport();
  bool useAvx512 = fbgemmHasAvx512Support();
  array<array<typename MicroKernelFunctionTrait<ACC_T>::type, 3>, 3> sgemms;

  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
      sgemms[ky][kx] = generateConv_microkernel<ACC_T>(
          Cout,
          Cin,
          Cin,
          IX * IY,
          IX * IY,
          KData + (ky * 3 + kx) * (Cin * Cout),
          canUseVNNI);
    }
  }

  auto const setMaskToZero = [&](uint16_t* base, int loc) {
    int element = loc / 16;
    int bit = loc % 16;

    base[element] &= (~static_cast<uint16_t>(1 << bit));
  };

  // By default, we assume we're using avx512 where each mask is 16-bit.
  // In avx2, each mask is 256-bit so we need to adjust accordingly.
  vector<uint16_t> allMasks;

  array<array<int, 2>, 2> maskOffsets;
  array<array<int, 2>, 2> groupsOf16;

  for (int ky = 0; ky < 2; ++ky) {
    for (int kx = 0; kx < 2; ++kx) {
      int N = IX * IY - ky * IX - kx;
      if (useAvx512) {
        // Each groupsOf16 has a 16-bit mask used for avx512 instructions
        // with vector length 16.
        groupsOf16[ky][kx] = (N + 15) / 16;
      } else {
        // Compute N aligned to WIDTH_32BIT_ELEMS and then multiply extra
        // factor 2 because each mask for avx2 is 32-bit rather than 16-bit.
        constexpr int WIDTH_32BIT_ELEMS =
            simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
        groupsOf16[ky][kx] = (N + WIDTH_32BIT_ELEMS - 1) / WIDTH_32BIT_ELEMS *
            WIDTH_32BIT_ELEMS * 2;
      }
      maskOffsets[ky][kx] = allMasks.size();

      allMasks.resize(maskOffsets[ky][kx] + groupsOf16[ky][kx], 0xffff);

      uint16_t* base = allMasks.data() + maskOffsets[ky][kx];

      if (useAvx512) {
        for (int i = N; i < groupsOf16[ky][kx] * 16; ++i) {
          setMaskToZero(base, i);
        }

        if (kx) {
          for (int i = IX - 1; i < groupsOf16[ky][kx] * 16; i += IX) {
            setMaskToZero(base, i);
          }
        }
      } else {
        for (int i = N; i < groupsOf16[ky][kx] / 2; ++i) {
          reinterpret_cast<int32_t*>(base)[i] = 0;
        }

        if (kx) {
          for (int i = IX - 1; i < groupsOf16[ky][kx] / 2; i += IX) {
            reinterpret_cast<int32_t*>(base)[i] = 0;
          }
        }
      }
    } // kx
  } // ky

  vector<
      tuple<typename MicroKernelFunctionTrait<ACC_T>::type, int, int, int, int>>
      to_exec;
  to_exec.reserve(9);

  to_exec.push_back(make_tuple(sgemms[1][1], 0, 0, 0, groupsOf16[0][0]));

  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
      if ((ky != 1) || (kx != 1)) {
        int del_in = 0;
        int del_out = 0;

        if (ky == 0) {
          del_out += IX;
        }
        if (ky == 2) {
          del_in += IX;
        }
        if (kx == 0) {
          del_out += 1;
        }
        if (kx == 2) {
          del_in += 1;
        }

        to_exec.push_back(make_tuple(
            sgemms[ky][kx],
            // When ACC_T == int32_t, 4 rows are interleaved in B
            del_in * (is_same<ACC_T, int32_t>::value ? 4 : 1),
            del_out,
            maskOffsets[ky != 1][kx != 1],
            groupsOf16[ky != 1][kx != 1]));
      }
    }
  }

  return [=](const typename SpMMTypeTrait<ACC_T>::b_type* in, ACC_T* out) {
    for (int k = 0; k < to_exec.size(); ++k) {
      auto const& e = to_exec[k];
      auto const& fn = std::get<0>(e);

      const typename SpMMTypeTrait<ACC_T>::b_type* ein = in + std::get<1>(e);
      ACC_T* eout = out + std::get<2>(e);

      fn(ein,
         eout,
         std::get<4>(e),
         allMasks.data() + std::get<3>(e),
         (k > 0) ? 1 : 0);
    }
  };
}

template FBGEMM_API function<void(const float* BData, float* CData)>
generateSpConv(int Cin, int Cout, int IY, int IX, const float* KData);

template FBGEMM_API function<void(const uint8_t* BData, int32_t* CData)>
generateSpConv(int Cin, int Cout, int IY, int IX, const int8_t* KData);

} // namespace fbgemm
