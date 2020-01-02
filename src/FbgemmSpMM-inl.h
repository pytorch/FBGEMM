/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "fbgemm/FbgemmSpMM.h"
#include "fbgemm/Utils.h"

#include <cassert>
#include <vector>

namespace fbgemm {

namespace x86 = asmjit::x86;

namespace internal {

template <class To, class From>
typename std::enable_if<
    (sizeof(To) == sizeof(From)) && std::is_trivially_copyable<From>::value &&
        std::is_trivial<To>::value,
    To>::type
bit_cast(const From& src) noexcept {
  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename ACC_T, inst_set_t instSet>
class SpMM_JitterASMJIT_base {
 protected:
  using a_temp_type = typename SpMMTypeTrait<ACC_T>::a_temp_type;
  a_temp_type const* AData_;

  int M_;
  int K_;
  int LDA_, LDB_, LDC_;

  x86::Gp Breg_ = x86::rdi;
  x86::Gp Creg_ = x86::rsi;
  x86::Gp flagsReg_;
  x86::Gp dataReg_;

  std::vector<a_temp_type> compressedAData_;

  x86::Builder a;
  asmjit::Label dataLabel_;
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

  virtual void
  loopOverN(int AOffset, int rowRegBlock, bool restorePointers) = 0;

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
        4);

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

  SpMM_JitterASMJIT_base(
      asmjit::CodeHolder& code,
      int M,
      int K,
      int LDA,
      int LDB,
      int LDC,
      const typename SpMMTypeTrait<ACC_T>::a_type* aData,
      bool canUseVNNI = false)
      : AData_(reinterpret_cast<const a_temp_type*>(aData)),
        M_(M),
        K_(K),
        LDA_(LDA),
        LDB_(LDB),
        LDC_(LDC),
        a(&code),
        dataLabel_(a.newLabel()),
        canUseVNNI_(canUseVNNI) {
    useAvx512_ = instSet == inst_set_t::avx512;
    if (useAvx512_) {
      WIDTH_BYTES = simd_info<inst_set_t::avx512>::WIDTH_BYTES;
      WIDTH_32BIT_ELEMS = simd_info<inst_set_t::avx512>::WIDTH_32BIT_ELEMS;
    } else {
      WIDTH_BYTES = simd_info<inst_set_t::avx2>::WIDTH_BYTES;
      WIDTH_32BIT_ELEMS = simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
    }
  }

  virtual ~SpMM_JitterASMJIT_base() {}
}; // SpMM_JitterASMJIT_base

} // namespace internal
} // namespace fbgemm
