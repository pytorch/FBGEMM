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
#include <cassert>
#include <iostream>
#include <mutex>
#include "./CodeCache.h"
#include "./MaskAvx2.h"
#include "./RefImplementations.h"
#include "fbgemm/Utils.h"

using namespace std;

namespace fbgemm {
namespace {
namespace x86 = asmjit::x86;

template <typename indxType = int64_t>
class ReturnFunctionSignature {
 public:
  using jit_sparse_adagrad_kernel = bool (*)(
      int64_t output_size,
      int64_t index_size,
      int64_t data_size, // number of rows in w
      float* w, // input/output parameters
      const float* g, // input gradients
      float* h, // input/output momentums
      const indxType* indices, // indices of each row
      const int* lengths,
      float epsilon,
      float lr,
      const int* mask_avx2);
};

template <typename indxType = int64_t, inst_set_t instSet = inst_set_t::avx2>
class GenRowWiseSparseAdagradFused {
 public:
  GenRowWiseSparseAdagradFused() {}

  typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
  getOrCreate(int block_size, int prefetch);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; // JIT Runtime for asmjit
    return rt;
  }

  static mutex rtMutex_; /// Controll access to runtime;

  // The hash depends on embedding dimension (block size), and prefetch distance
  static CodeCache<
      tuple<int, int>,
      typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel>
      codeCache_; ///< JIT Code Cache for reuse.
}; // class GenRowWiseSparseAdagradFused

template <typename indxType, inst_set_t instSet>
mutex GenRowWiseSparseAdagradFused<indxType, instSet>::rtMutex_;

template <typename indxType, inst_set_t instSet>
CodeCache<
    tuple<int, int>,
    typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel>
    GenRowWiseSparseAdagradFused<indxType, instSet>::codeCache_;

template <typename indxType, inst_set_t instSet>
typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
GenRowWiseSparseAdagradFused<indxType, instSet>::getOrCreate(
    int block_size,
    int prefetch) {
  tuple<int, int> kernelSig = make_tuple(block_size, prefetch);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() ->
      typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel {
        asmjit::CodeHolder code;
        code.init(runtime().codeInfo());
        x86::Assembler assembler(&code);
        x86::Emitter* a = assembler.as<x86::Emitter>();
        bool areIndices64b = is_same<indxType, int64_t>::value;
#if defined(FBGEMM_LOG_CODE)
        string filename = "RowWiseSparseAdagradFused";
        filename += "_emd_dim_" + to_string(block_size);
        filename += areIndices64b ? "_64bit" : "_32bit";
        filename += instSet == inst_set_t::avx512 ? "_avx512" : "_avx2";
        if (prefetch) {
          filename += "_prefetch";
        }
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif

        x86::Gp output_size = a->zdi();
        x86::Gp index_size = a->zsi();
        x86::Gp data_size = a->zdx();
        x86::Gp w = a->zcx();
        x86::Gp g = a->gpz(8);
        x86::Gp h = a->gpz(9);
        x86::Gp indices = a->gpz(10);
        x86::Gp lengths = a->gpz(11);
        x86::Xmm epsilon = x86::xmm0;
        x86::Xmm lr = x86::xmm1;
        x86::Gp mask_avx2 = a->gpz(12);

        // reuse mask_avx2 because mask_avx2 is used only at the beginning
        x86::Gpd lengths_R = a->gpz(12).r32();
        x86::Gp scratchReg1 = a->gpz(13);
        x86::Gp scratchReg2 = a->gpz(14); // for prefetching

        asmjit::FuncDetail func;
        func.init(asmjit::FuncSignatureT<
                  bool, // return type
                  int64_t, // output_size
                  int64_t, // index_size
                  int64_t, // data_size
                  float*, // w
                  const float*, // g
                  float*, // h
                  const indxType*, // indices
                  const int*, // lengths
                  float, // epsilon
                  float, // lr then mask_avx2
                  const int*>(asmjit::CallConv::kIdHost));

        asmjit::FuncFrame frame;
        frame.init(func);

        if (instSet == inst_set_t::avx2) {
          frame.setDirtyRegs(
              x86::Reg::kGroupVec,
              asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                  asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15));
        } else {
          frame.setDirtyRegs(
              x86::Reg::kGroupVec,
              asmjit::Support::bitMask(0, 1, 2, 3, 4, 5, 6, 7) |
                  asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14, 15) |
                  asmjit::Support::bitMask(16, 17, 18, 19, 20, 21, 22, 23) |
                  asmjit::Support::bitMask(24, 25, 26, 27, 28, 29, 30, 31));
        }

        // TODO
        frame.setDirtyRegs(
            x86::Reg::kGroupGp,
            asmjit::Support::bitMask(8, 9, 10, 11, 12, 13, 14));

        asmjit::FuncArgsAssignment args(&func);
        args.assignAll(
            output_size,
            index_size,
            data_size,
            w,
            g,
            h,
            indices,
            lengths,
            epsilon,
            lr,
            mask_avx2);

        args.updateFuncFrame(frame);
        frame.finalize();
        a->emitProlog(frame);
        a->emitArgsAssignment(frame, args);

        constexpr int vlen = simd_info<instSet>::WIDTH_32BIT_ELEMS;
        constexpr int NUM_VEC_REG = simd_info<instSet>::NUM_VEC_REGS;

        typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

        int num_vec_regs_per_block = (block_size + vlen - 1) / vlen;
        int remainder = block_size % vlen;

        vec_reg_t src_vreg; // for holding embedding value temporarily
        x86::Ymm mask_vreg;

        // Reserve registers with small ids first because some of them need to
        // be used with an instruction not supported in avx512 for which a big
        // register id won't work.
        int first_available_vec_reg_id = 0;
        x86::Ymm partial_sum_vreg = x86::Ymm(first_available_vec_reg_id);
        ++first_available_vec_reg_id;
        vec_reg_t float_step_vreg = vec_reg_t(first_available_vec_reg_id);
        ++first_available_vec_reg_id;
        vec_reg_t epsilon_vreg = vec_reg_t(first_available_vec_reg_id);
        ++first_available_vec_reg_id;
        vec_reg_t lr_vreg = vec_reg_t(first_available_vec_reg_id);
        ++first_available_vec_reg_id;

        if (remainder) {
          if (instSet == inst_set_t::avx2) {
            src_vreg = vec_reg_t(first_available_vec_reg_id);
            ++first_available_vec_reg_id;

            mask_vreg = x86::Ymm(first_available_vec_reg_id);
            ++first_available_vec_reg_id;

            a->vmovups(
                mask_vreg,
                x86::ymmword_ptr(
                    mask_avx2, (vlen - remainder) % vlen * sizeof(int32_t)));
          } else {
            a->mov(scratchReg1, (1 << remainder) - 1);
            a->kmovw(x86::k(1), scratchReg1);
          }
        }
        // Need an extra mask for computing sum of gradients
        int remainder_avx2 =
            block_size % simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
        x86::KReg reduce_mask_avx512;
        if (remainder_avx2 && instSet == inst_set_t::avx512) {
          reduce_mask_avx512 = x86::k(2);
          a->mov(scratchReg1, (1 << remainder_avx2) - 1);
          a->kmovw(reduce_mask_avx512, scratchReg1);
        }

        int unroll_factor = NUM_VEC_REG - first_available_vec_reg_id;

        a->vpbroadcastd(epsilon_vreg, epsilon);
        a->vpbroadcastd(lr_vreg, lr);

        // Compute the end address of indices
        a->imul(
            scratchReg1,
            index_size,
            static_cast<asmjit::Imm>(sizeof(indxType)));
        a->add(scratchReg1, indices);
        a->mov(index_size, scratchReg1);

        asmjit::Label exit = a->newLabel();
        asmjit::Label error = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        // rangeIndex loop begin (iterate output_size times)
        a->bind(LoopRangeIndexBegin);
        a->dec(output_size);
        a->jl(LoopRangeIndexEnd);

        // Compute sq avg of gradients
        // Even with avx512, we only need to use avx2 registers when computing
        // partial_sum because some instructions we're using like vhaddps
        // are only in avx2.
        constexpr int vlen_avx2 =
            simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS;
        int num_vec_regs_per_block_avx2 =
            (block_size + vlen_avx2 - 1) / vlen_avx2;

        a->vxorps(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);

        // TODO: need to do a tree-reduction to fully take advantage of
        // unrolling
        for (int vec_idx = 0; vec_idx < num_vec_regs_per_block_avx2;
             vec_idx += unroll_factor) {
          int cur_unroll_factor =
              std::min(unroll_factor, num_vec_regs_per_block_avx2 - vec_idx);
          for (int v = 0; v < cur_unroll_factor; ++v) {
            x86::Ymm out_vreg = x86::Ymm(v + first_available_vec_reg_id);

            auto g_ptr =
                x86::dword_ptr(g, (vec_idx + v) * vlen_avx2 * sizeof(float));
            if (block_size % simd_info<inst_set_t::avx2>::WIDTH_32BIT_ELEMS &&
                vec_idx + v == num_vec_regs_per_block_avx2 - 1) {
              if (instSet == inst_set_t::avx2) {
                a->vmaskmovps(out_vreg, mask_vreg, g_ptr);
              } else {
                a->k(reduce_mask_avx512).z().vmovups(out_vreg, g_ptr);
              }
            } else {
              a->vmovups(out_vreg, g_ptr);
            }
            a->vmulps(out_vreg, out_vreg, out_vreg);
            a->vaddps(partial_sum_vreg, partial_sum_vreg, out_vreg);
          }
        }
        // Reduce sum to 1 value
        // __m256 partial_sum_2 = _mm256_hadd_ps(partial_sum, partial_sum);
        // __m256 partial_sum_3 = _mm256_hadd_ps(partial_sum_2, partial_sum_2);
        // Use YMM/XMMs with smaller ids for AVX2 specific instructions like
        // vhaddps
        x86::Xmm partial_sum_xmm = x86::Xmm(partial_sum_vreg.id());
        x86::Xmm float_step_xmm = x86::Xmm(float_step_vreg.id());
        // a->vmovups(partial_sum_temp0_ymm, partial_sum_vreg);
        a->vhaddps(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);
        a->vhaddps(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);

        //_mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3))
        a->movss(float_step_xmm, partial_sum_xmm);
        //_mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1))
        a->vextractf128(partial_sum_xmm, partial_sum_vreg, 1);

        // final_sum = _mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3)) +
        //    _mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1));
        a->addss(partial_sum_xmm, float_step_xmm);

        // This fragment moves block size (N) to stack and bcasts it to xmm reg
        a->lea(
            x86::rsp,
            x86::dword_ptr(x86::rsp, -1 * static_cast<int>(sizeof(int32_t))));
        a->mov(x86::dword_ptr(x86::rsp), block_size);
        a->vbroadcastss(
            float_step_xmm,
            x86::dword_ptr(x86::rsp)); // N is partial_sum_xmm1
        a->vcvtdq2ps(float_step_xmm, float_step_xmm);
        a->lea(x86::rsp, x86::dword_ptr(x86::rsp, sizeof(int32_t)));

        // final_sum /= N
        a->divss(partial_sum_xmm, float_step_xmm);

        a->mov(lengths_R, x86::dword_ptr(lengths));

        // Array out of bound check
        a->imul(
            scratchReg1, lengths_R, static_cast<asmjit::Imm>(sizeof(indxType)));

        a->add(scratchReg1, indices);
        a->cmp(scratchReg1, index_size);
        a->jg(error);

        asmjit::Label LoopDataIndexBegin = a->newLabel();
        asmjit::Label LoopDataIndexEnd = a->newLabel();

        // dataIndex loop begins (iterate lengths_R_ times)
        a->bind(LoopDataIndexBegin);
        a->dec(lengths_R);
        a->jl(LoopDataIndexEnd);

        // Array out of bound check
        if (areIndices64b) {
          a->mov(scratchReg1, x86::qword_ptr(indices));
        } else {
          a->mov(scratchReg1.r32(), x86::dword_ptr(indices));
        }
        // A trick to check x >= data_size or x < 0 in one shot by treating
        // scratchReg1_ as if it has unsigned value
        // (https://stackoverflow.com/a/34072155).
        a->cmp(scratchReg1, data_size);
        a->jae(error);

        if (prefetch) {
          asmjit::Label pref_dist_reset_start = a->newLabel();
          asmjit::Label pref_dist_reset_end = a->newLabel();
          // out of bound handling for prefetch
          a->mov(scratchReg2, indices);
          a->add(
              scratchReg2,
              static_cast<asmjit::Imm>(prefetch * sizeof(indxType)));
          a->cmp(scratchReg2, index_size);
          a->jge(pref_dist_reset_start);

          if (areIndices64b) {
            a->mov(
                scratchReg2,
                x86::qword_ptr(indices, prefetch * sizeof(indxType)));
          } else {
            a->mov(
                scratchReg2.r32(),
                x86::dword_ptr(indices, prefetch * sizeof(indxType)));
          }

          a->cmp(scratchReg2, data_size);
          a->jb(pref_dist_reset_end);

          a->bind(pref_dist_reset_start);
          // things are not okay just get the current row
          // this can be improved to getting the max dist row.
          if (areIndices64b) {
            a->mov(scratchReg2, x86::qword_ptr(indices));
          } else {
            a->mov(scratchReg2.r32(), x86::dword_ptr(indices));
          }

          a->bind(pref_dist_reset_end);
          a->imul(scratchReg2, static_cast<asmjit::Imm>(sizeof(float)));
        }

        a->add(indices, static_cast<asmjit::Imm>(sizeof(indxType)));

        a->imul(scratchReg1, static_cast<asmjit::Imm>(sizeof(float)));

        if (prefetch) {
          a->prefetchw(x86::dword_ptr(h, scratchReg2));
        }
        // load h
        a->movss(float_step_xmm, x86::dword_ptr(h, scratchReg1));
        // *h + final_sum
        a->addss(float_step_xmm, partial_sum_xmm);
        // store h
        a->movss(x86::dword_ptr(h, scratchReg1), float_step_xmm);
        // sqrt(hi)
        a->sqrtss(float_step_xmm, float_step_xmm);
        // bcast partial to all of ymm/zmm reg
        a->vpbroadcastd(float_step_vreg, float_step_xmm);
        // lr / sqrt(hi) + epsilon
        a->vaddps(float_step_vreg, float_step_vreg, epsilon_vreg);
        a->vdivps(float_step_vreg, lr_vreg, float_step_vreg);

        a->imul(scratchReg1, static_cast<asmjit::Imm>(block_size));
        if (prefetch) {
          a->imul(scratchReg2, static_cast<asmjit::Imm>(block_size));
        }

        for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
             vec_idx += unroll_factor) {
          int cur_unroll_factor =
              std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

          // The main computation
          for (int v = 0; v < cur_unroll_factor; ++v) {
            vec_reg_t out_vreg = vec_reg_t(v + first_available_vec_reg_id);

            auto g_ptr =
                x86::dword_ptr(g, (vec_idx + v) * vlen * sizeof(float));
            auto w_ptr = x86::dword_ptr(
                w, scratchReg1, 0, (vec_idx + v) * vlen * sizeof(float));
            if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
              if (instSet == inst_set_t::avx2) {
                a->vmaskmovps(x86::ymm(src_vreg.id()), mask_vreg, g_ptr);
                a->vmulps(src_vreg, float_step_vreg, src_vreg);

                a->vmaskmovps(x86::ymm(out_vreg.id()), mask_vreg, w_ptr);
                a->vaddps(out_vreg, src_vreg, out_vreg);

                a->vmaskmovps(w_ptr, mask_vreg, x86::ymm(out_vreg.id()));
              } else {
                a->k(x86::k(1)).vmulps(out_vreg, float_step_vreg, g_ptr);
                a->k(x86::k(1)).vaddps(out_vreg, out_vreg, w_ptr);
                a->k(x86::k(1)).vmovups(w_ptr, out_vreg);
              }
            } else {
              a->vmulps(out_vreg, float_step_vreg, g_ptr);
              a->vaddps(out_vreg, out_vreg, w_ptr);
              a->vmovups(w_ptr, out_vreg);
            }

            constexpr int CACHE_LINE_LEN = 64;
            constexpr int BYTES_PER_VLOAD = vlen * sizeof(float);
            constexpr int VLOAD_PER_CACHE_LINE =
                CACHE_LINE_LEN / BYTES_PER_VLOAD;
            if (prefetch && (vec_idx + v) % VLOAD_PER_CACHE_LINE == 0) {
              a->prefetchw(x86::dword_ptr(
                  w, scratchReg2, 0, (vec_idx + v) * BYTES_PER_VLOAD));
            }
          }
        }

        a->jmp(LoopDataIndexBegin);
        a->bind(LoopDataIndexEnd);

        a->add(lengths, static_cast<asmjit::Imm>(sizeof(int)));
        a->add(g, static_cast<asmjit::Imm>(block_size * sizeof(float)));

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
        typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
            fn;
        asmjit::Error err;
        {
          unique_lock<mutex> lock(rtMutex_);
          err = runtime().add(&fn, &code);
        }
        if (err) {
          cout << "Error: in fn add" << endl;
          return nullptr;
        }

#if defined(FBGEMM_LOG_CODE)
        fclose(codeLogFile);
        delete codeLogger;
#endif
        return fn;
      });
} // getOrCreate

} // namespace

template <typename IndexType>
FBGEMM_API typename RowWiseSparseAdaGradFusedSignature<IndexType>::Type
GenerateRowWiseSparseAdaGradFused(
    int block_size, // number of parameters per row
    int prefetch) {
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }

  // Always use avx2 because avx512 doesn't provide speedups
  if (fbgemmHasAvx512Support() || fbgemmHasAvx2Support()) {
    static GenRowWiseSparseAdagradFused<IndexType, inst_set_t::avx2>
        kernel_generator;
    const auto original_func =
        kernel_generator.getOrCreate(block_size, prefetch);
    const auto lambda_func = [=](int64_t output_size,
                                 int64_t index_size,
                                 int64_t data_size,
                                 float* w,
                                 const float* g,
                                 float* h,
                                 const IndexType* indices,
                                 const int* lengths,
                                 float epsilon,
                                 float lr) {
      return original_func(
          output_size,
          index_size,
          data_size,
          w, // input/output parameters
          g, // input gradients
          h, // input/output momentums
          indices, // indices of each row
          lengths,
          epsilon,
          lr,
          internal::avx2_ps_or_epi32_combined_mask);
    };
    return lambda_func;
  } else {
    return [=](int64_t output_size,
               int64_t index_size,
               int64_t data_size,
               float* w,
               const float* g,
               float* h,
               const IndexType* indices,
               const int* lengths,
               float epsilon,
               float lr) {
      return rowwise_sparse_adagrad_fused_ref(
          block_size,
          output_size,
          index_size,
          data_size,
          w,
          g,
          h,
          indices,
          lengths,
          epsilon,
          lr);
    };
  }
}

template FBGEMM_API typename RowWiseSparseAdaGradFusedSignature<int64_t>::Type
GenerateRowWiseSparseAdaGradFused<int64_t>(
    int block_size, // number of parameters per row
    int prefetch);

template FBGEMM_API typename RowWiseSparseAdaGradFusedSignature<int32_t>::Type
GenerateRowWiseSparseAdaGradFused<int32_t>(
    int block_size, // number of parameters per row
    int prefetch);

} // namespace fbgemm
