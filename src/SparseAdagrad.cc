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
#include "fbgemm/Fbgemm.h"
#include "fbgemm/Utils.h"

namespace fbgemm {

namespace {
namespace x86 = asmjit::x86;

template <typename indxType = std::int64_t>
class ReturnFunctionSignature {
 public:
  using jit_sparse_adagrad_kernel = int (*)(
      int num_rows, // number of rows reading
      std::uint64_t param_size, // total number of parameters
      const float* w, // input parameters
      const float* g, // input gradients
      const float* h, // input momentums
      const indxType* indices, // indices of each row
      float epsilon,
      float lr);
};

template <
    typename indxType = std::int64_t,
    inst_set_t instSet = inst_set_t::avx2>
class GenSparseAdagrad {
 public:
  GenSparseAdagrad() {}
  void genSparseAdagrad(
      x86::Emitter* a,
      int vlen,
      int unroll_factor,
      int num_vec_regs_per_block,
      int remainder,
      int prefetch,
      typename simd_info<instSet>::vec_reg_t epsilon_vreg,
      typename simd_info<instSet>::vec_reg_t lr_vreg,
      typename simd_info<instSet>::vec_reg_t mask_vreg,
      typename simd_info<instSet>::vec_reg_t temp_vreg);

  void genRowwiseSparseAdagrad(
      x86::Emitter* a,
      int block_size,
      int vlen,
      int unroll_factor,
      int num_vec_regs_per_block,
      int remainder,
      int prefetch,
      typename simd_info<instSet>::vec_reg_t epsilon_vreg,
      typename simd_info<instSet>::vec_reg_t lr_vreg,
      typename simd_info<instSet>::vec_reg_t mask_vreg);

  typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
  getOrCreate(int block_size, int prefetch, bool rowwise);

 private:
  static asmjit::JitRuntime& runtime() {
    static asmjit::JitRuntime rt; // JIT Runtime for asmjit
    return rt;
  }

  static std::mutex rtMutex_; /// Controll access to runtime;

  // The hash depends on embedding dimension (block size), and prefetch distance
  static CodeCache<
      std::tuple<int, int, bool>,
      typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel>
      codeCache_; ///< JIT Code Cache for reuse.

  // These are register we share accross SparseAdagrad and RowwiseSparseAdagrad
  x86::Gpd num_rows;
  x86::Gp param_size;
  x86::Gp w;
  x86::Gp g;
  x86::Gp h;
  x86::Gp indices;
  x86::Xmm epsilon;
  x86::Xmm lr;
  x86::Gp base_offset_g;
  x86::Gp base_offset;
  x86::Gp temp1_;
  x86::Gp temp2_;
  x86::Gp temp3_;

}; // GenEmbeddingLookup

template <typename indxType, inst_set_t instSet>
std::mutex GenSparseAdagrad<indxType, instSet>::rtMutex_;

template <typename indxType, inst_set_t instSet>
CodeCache<
    std::tuple<int, int, bool>,
    typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel>
    GenSparseAdagrad<indxType, instSet>::codeCache_;

template <typename indxType, inst_set_t instSet>
void GenSparseAdagrad<indxType, instSet>::genSparseAdagrad(
    x86::Emitter* a,
    int vlen,
    int unroll_factor,
    int num_vec_regs_per_block,
    int remainder,
    int prefetch,
    typename simd_info<instSet>::vec_reg_t epsilon_vreg,
    typename simd_info<instSet>::vec_reg_t lr_vreg,
    typename simd_info<instSet>::vec_reg_t mask_vreg,
    typename simd_info<instSet>::vec_reg_t temp_vreg) {
  typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;
  for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
       vec_idx += unroll_factor) {
    int cur_unroll_factor =
        std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

    for (int v = 0; v < cur_unroll_factor; ++v) {
      vec_reg_t out_vreg = vec_reg_t(v);
      vec_reg_t nw_vreg = vec_reg_t(v + cur_unroll_factor);

      if (prefetch && (v % (64 / (vlen * sizeof(float))) == 0)) {
        a->prefetchwt1(
            x86::dword_ptr(h, temp2_, 0, (vec_idx + v) * vlen * sizeof(float)));

        a->prefetchwt1(
            x86::dword_ptr(w, temp2_, 0, (vec_idx + v) * vlen * sizeof(float)));
      }

      if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
        if (instSet == inst_set_t::avx2) {
          a->vmaskmovps(
              x86::ymm(out_vreg.id()),
              x86::ymm(mask_vreg.id()),
              x86::dword_ptr(
                  g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->vmulps(out_vreg, out_vreg, out_vreg);
          a->vmaskmovps(
              x86::ymm(temp_vreg.id()),
              x86::ymm(mask_vreg.id()),
              x86::dword_ptr(
                  h, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->vaddps(out_vreg, out_vreg, temp_vreg);

          a->vmaskmovps(
              x86::dword_ptr(
                  h, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)),
              x86::ymm(mask_vreg.id()),
              x86::ymm(out_vreg.id()));

          a->vsqrtps(out_vreg, out_vreg);
          a->vaddps(out_vreg, out_vreg, epsilon_vreg);

          a->vmaskmovps(
              x86::ymm(nw_vreg.id()),
              x86::ymm(mask_vreg.id()),
              x86::dword_ptr(
                  g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->vdivps(out_vreg, nw_vreg, out_vreg);

          a->vmulps(out_vreg, out_vreg, lr_vreg);
          a->vmaskmovps(
              x86::ymm(temp_vreg.id()),
              x86::ymm(mask_vreg.id()),
              x86::dword_ptr(
                  w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->vaddps(out_vreg, out_vreg, temp_vreg);

          a->vmaskmovps(
              x86::dword_ptr(
                  w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)),
              x86::ymm(mask_vreg.id()),
              x86::ymm(out_vreg.id()));
        } else if (instSet == inst_set_t::avx512) {
          a->k(x86::k(1)).vmovups(
              out_vreg,
              x86::dword_ptr(
                  g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->k(x86::k(1)).vmulps(out_vreg, out_vreg, out_vreg);

          a->k(x86::k(1)).vaddps(
              out_vreg,
              out_vreg,
              x86::dword_ptr(
                  h, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->k(x86::k(1)).vmovups(
              x86::dword_ptr(
                  h, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)),
              out_vreg);

          a->k(x86::k(1)).vsqrtps(out_vreg, out_vreg);
          a->k(x86::k(1)).vaddps(out_vreg, out_vreg, epsilon_vreg);

          a->k(x86::k(1)).vmovups(
              nw_vreg,
              x86::dword_ptr(
                  g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->k(x86::k(1)).vdivps(out_vreg, nw_vreg, out_vreg);
          a->k(x86::k(1)).vmulps(out_vreg, out_vreg, lr_vreg);

          a->k(x86::k(1)).vaddps(
              out_vreg,
              out_vreg,
              x86::dword_ptr(
                  w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->k(x86::k(1)).vmovups(
              x86::dword_ptr(
                  w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)),
              out_vreg);
        }
      } else {
        a->vmovups(
            out_vreg,
            x86::dword_ptr(
                g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

        a->vmulps(out_vreg, out_vreg, out_vreg);

        a->vaddps(
            out_vreg,
            out_vreg,
            x86::dword_ptr(
                h, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)));

        a->vmovups(
            x86::dword_ptr(
                h, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)),
            out_vreg);

        a->vsqrtps(out_vreg, out_vreg);
        a->vaddps(out_vreg, out_vreg, epsilon_vreg);

        a->vmovups(
            nw_vreg,
            x86::dword_ptr(
                g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

        a->vdivps(out_vreg, nw_vreg, out_vreg);
        a->vmulps(out_vreg, out_vreg, lr_vreg);

        a->vaddps(
            out_vreg,
            out_vreg,
            x86::dword_ptr(
                w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)));

        a->vmovups(
            x86::dword_ptr(
                w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)),
            out_vreg);
      }
    }
  }
}

template <typename indxType, inst_set_t instSet>
void GenSparseAdagrad<indxType, instSet>::genRowwiseSparseAdagrad(
    x86::Emitter* a,
    int block_size,
    int vlen,
    int unroll_factor,
    int num_vec_regs_per_block,
    int remainder,
    int prefetch,
    typename simd_info<instSet>::vec_reg_t epsilon_vreg,
    typename simd_info<instSet>::vec_reg_t lr_vreg,
    typename simd_info<instSet>::vec_reg_t mask_vreg) {
  typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

  // Reduce the unroll factor by 2
  // one for partial sum
  // one for temporary xmm register

  // Multiply unroll factor by 2 as we had halved it earlier
  vec_reg_t partial_sum_vreg = vec_reg_t(2 * unroll_factor - 1);
  x86::Xmm partial_sum_xmm0 = x86::Xmm(2 * unroll_factor - 1);
  x86::Xmm partial_sum_xmm1 = x86::Xmm(2 * unroll_factor - 2);
  --unroll_factor;

  a->vxorps(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);

  if (prefetch) {
    a->prefetchwt1(x86::dword_ptr(h, temp3_));
  }

  for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
       vec_idx += unroll_factor) {
    int cur_unroll_factor =
        std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

    for (int v = 0; v < cur_unroll_factor; ++v) {
      vec_reg_t out_vreg = vec_reg_t(v);

      if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
        if (instSet == inst_set_t::avx2) {
          a->vmaskmovps(
              x86::ymm(out_vreg.id()),
              x86::ymm(mask_vreg.id()),
              x86::dword_ptr(
                  g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->vmulps(out_vreg, out_vreg, out_vreg);
          a->vaddps(partial_sum_vreg, partial_sum_vreg, out_vreg);
        } else {
          a->k(x86::k(1)).vmovups(
              out_vreg,
              x86::dword_ptr(
                  g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->k(x86::k(1)).vmulps(out_vreg, out_vreg, out_vreg);
          a->k(x86::k(1)).vaddps(partial_sum_vreg, partial_sum_vreg, out_vreg);
        }
      } else {
        a->vmovups(
            out_vreg,
            x86::dword_ptr(
                g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

        a->vmulps(out_vreg, out_vreg, out_vreg);
        a->vaddps(partial_sum_vreg, partial_sum_vreg, out_vreg);
      }
    }
  }
  // Reduce sum to 1 value
  // __m256 partial_sum_2 = _mm256_hadd_ps(partial_sum, partial_sum);
  // __m256 partial_sum_3 = _mm256_hadd_ps(partial_sum_2, partial_sum_2);
  a->vhaddps(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);
  a->vhaddps(partial_sum_vreg, partial_sum_vreg, partial_sum_vreg);

  //_mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3))
  a->movss(partial_sum_xmm1, partial_sum_xmm0);
  //_mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1))
  a->vextractf128(partial_sum_xmm0, partial_sum_vreg, 1);

  // final_sum = _mm_cvtss_f32(_mm256_castps256_ps128(partial_sum_3)) +
  //    _mm_cvtss_f32(_mm256_extractf128_ps(partial_sum_3, 1));
  a->addss(partial_sum_xmm0, partial_sum_xmm1);

  // This fragment moves block size (N) to stack and bcasts it to xmm reg
  a->lea(
      x86::rsp,
      x86::dword_ptr(x86::rsp, -1 * static_cast<int>(sizeof(int32_t))));
  a->mov(x86::dword_ptr(x86::rsp), block_size);
  a->vbroadcastss(
      partial_sum_xmm1, x86::dword_ptr(x86::rsp)); // N is partial_sum_xmm1
  a->vcvtdq2ps(partial_sum_xmm1, partial_sum_xmm1);
  a->lea(x86::rsp, x86::dword_ptr(x86::rsp, sizeof(int32_t)));

  // final_sum =/N
  a->divss(partial_sum_xmm0, partial_sum_xmm1);
  // load h
  a->movss(partial_sum_xmm1, x86::dword_ptr(h, base_offset));
  //*h + final_sum
  a->addss(partial_sum_xmm0, partial_sum_xmm1);
  // store h
  a->movss(x86::dword_ptr(h, base_offset), partial_sum_xmm0);
  // sqrt(hi)
  a->sqrtss(partial_sum_xmm0, partial_sum_xmm0);
  // bcast partial to all of ymm/zmm reg
  a->vpbroadcastd(partial_sum_vreg, partial_sum_xmm0);
  // lr / sqrt(hi) +epsilon
  a->vaddps(partial_sum_vreg, partial_sum_vreg, epsilon_vreg);
  a->vdivps(partial_sum_vreg, lr_vreg, partial_sum_vreg);
  // partial_sum_vreg now has float_step

  bool areIndices64b = std::is_same<indxType, std::int64_t>::value;
  if (areIndices64b) {
    a->imul(
        base_offset,
        x86::qword_ptr(
            indices,
            temp1_,
            3), // use of 3 is to muliply by 8 (int64_t)
        static_cast<asmjit::Imm>(block_size * sizeof(float)));
  } else {
    a->imul(
        base_offset.r32(),
        x86::dword_ptr(
            indices,
            temp1_,
            2), // use of 2 is to muliply by 4 (int32_t)
        static_cast<asmjit::Imm>(block_size * sizeof(float)));
  }

  for (int vec_idx = 0; vec_idx < num_vec_regs_per_block;
       vec_idx += unroll_factor) {
    int cur_unroll_factor =
        std::min(unroll_factor, num_vec_regs_per_block - vec_idx);

    for (int v = 0; v < cur_unroll_factor; ++v) {
      vec_reg_t out_vreg = vec_reg_t(v);
      vec_reg_t w_vreg = vec_reg_t(v + cur_unroll_factor);

      if (prefetch && (v % (64 / (vlen * sizeof(float))) == 0)) {
        a->prefetchwt1(
            x86::dword_ptr(w, temp2_, 0, (vec_idx + v) * vlen * sizeof(float)));
      }

      if (remainder && vec_idx + v == num_vec_regs_per_block - 1) {
        if (instSet == inst_set_t::avx2) {
          a->vmaskmovps(
              x86::ymm(out_vreg.id()),
              x86::ymm(mask_vreg.id()),
              x86::dword_ptr(
                  g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->vmaskmovps(
              x86::ymm(w_vreg.id()),
              x86::ymm(mask_vreg.id()),
              x86::dword_ptr(
                  w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->vmulps(out_vreg, out_vreg, partial_sum_vreg);
          a->vaddps(w_vreg, w_vreg, out_vreg);
          a->vmaskmovps(
              x86::dword_ptr(
                  w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)),
              x86::ymm(mask_vreg.id()),
              x86::ymm(w_vreg.id()));

        } else {
          a->k(x86::k(1)).vmovups(
              out_vreg,
              x86::dword_ptr(
                  g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->k(x86::k(1)).vmovups(
              w_vreg,
              x86::dword_ptr(
                  w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)));

          a->k(x86::k(1)).vmulps(out_vreg, out_vreg, partial_sum_vreg);
          a->k(x86::k(1)).vaddps(w_vreg, w_vreg, out_vreg);
          a->k(x86::k(1)).vmovups(
              x86::dword_ptr(
                  w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)),
              w_vreg);
        }
      } else {
        a->vmovups(
            out_vreg,
            x86::dword_ptr(
                g, base_offset_g, 0, (vec_idx + v) * vlen * sizeof(float)));

        a->vmovups(
            w_vreg,
            x86::dword_ptr(
                w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)));

        a->vmulps(out_vreg, out_vreg, partial_sum_vreg);
        a->vaddps(w_vreg, w_vreg, out_vreg);
        a->vmovups(
            x86::dword_ptr(
                w, base_offset, 0, (vec_idx + v) * vlen * sizeof(float)),
            w_vreg);
      }
    }
  }
}

template <typename indxType, inst_set_t instSet>
typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
GenSparseAdagrad<indxType, instSet>::getOrCreate(
    int block_size,
    int prefetch,
    bool rowwise) {
  std::tuple<int, int, bool> kernelSig =
      std::make_tuple(block_size, prefetch, rowwise);

  return codeCache_.getOrCreate(
      kernelSig,
      [&]() ->
      typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel {
        asmjit::CodeHolder code;
        code.init(runtime().codeInfo());
        x86::Assembler assembler(&code);
        x86::Emitter* a = assembler.as<x86::Emitter>();
        bool areIndices64b = std::is_same<indxType, std::int64_t>::value;
#if defined(FBGEMM_LOG_CODE)
        std::string filename = "SparseAdagrad";
        filename += "_emd_dim_" + std::to_string(block_size);
        if (rowwise)
          filename += "_rowwise_";
        if (!areIndices64b)
          filename += "_32bit";
        if (areIndices64b)
          filename += "_64bit";
        if (instSet == inst_set_t::avx2)
          filename += "_avx2";
        if (instSet == inst_set_t::avx512)
          filename += "_avx512";
        if (prefetch)
          filename += "_prefetch";
        filename += ".txt";
        FILE* codeLogFile = fopen(filename.c_str(), "w");
        asmjit::FileLogger* codeLogger = new asmjit::FileLogger(codeLogFile);
        code.setLogger(codeLogger);
#endif

        num_rows = a->zdi().r32();
        param_size = a->zsi();
        w = a->zdx();
        g = a->zcx();
        h = a->gpz(8);
        indices = a->gpz(9);
        epsilon = x86::xmm0;
        lr = x86::xmm1;

        base_offset_g = a->gpz(10);
        base_offset = a->gpz(11);
        temp1_ = a->gpz(12);
        temp2_ = a->gpz(13);
        temp3_ = a->gpz(14);

        asmjit::FuncDetail func;
        func.init(asmjit::FuncSignatureT<
                  int, // return type
                  int, // num rows
                  std::uint64_t, // param_size
                  const float*, // w
                  const float*, // g
                  const float*, // h
                  const indxType*, // indices
                  float, // epsilon then lr
                  float>(asmjit::CallConv::kIdHost));

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

        frame.setDirtyRegs(
            x86::Reg::kGroupGp, asmjit::Support::bitMask(8, 9, 10, 11, 12, 13));

        asmjit::FuncArgsAssignment args(&func);
        args.assignAll(num_rows, param_size, w, g, h, indices, epsilon, lr);

        args.updateFuncFrame(frame);
        frame.finalize();
        a->emitProlog(frame);
        a->emitArgsAssignment(frame, args);

        simd_info<instSet> inst_trait;
        constexpr int vlen = inst_trait.WIDTH_32BIT_ELEMS;
        constexpr int NUM_VEC_REG = inst_trait.NUM_VEC_REGS;
        int unroll_factor = NUM_VEC_REG;

        typedef typename simd_info<instSet>::vec_reg_t vec_reg_t;

        int num_vec_regs_per_block = (block_size + vlen - 1) / vlen;
        int remainder = block_size % vlen;

        vec_reg_t epsilon_vreg;
        vec_reg_t lr_vreg;
        vec_reg_t mask_vreg; // mask for avx2
        vec_reg_t
            temp_vreg; // temp vreg for avx2 to handle remainder computation

        --unroll_factor;
        epsilon_vreg = vec_reg_t(unroll_factor);
        --unroll_factor;
        lr_vreg = vec_reg_t(unroll_factor);

        if (remainder) {
          // AVX512 doesn't need to use vector register for masking
          unroll_factor -= (instSet == inst_set_t::avx2 ? 2 : 0);
          if (instSet == inst_set_t::avx2) {
            mask_vreg = vec_reg_t(unroll_factor);
            temp_vreg = vec_reg_t(unroll_factor + 1);
          }
        }
        // Creating masks for non multiples of vlen iterations
        if (remainder) {
          if (instSet == inst_set_t::avx2) {
            a->lea(
                x86::rsp,
                x86::dword_ptr(x86::rsp, (int32_t)(-vlen * sizeof(int32_t))));
            for (int i = 0; i < remainder; i++) {
              a->mov(x86::dword_ptr(x86::rsp, i * sizeof(int32_t)), -1);
            }
            for (int i = remainder; i < vlen; i++) {
              a->mov(x86::dword_ptr(x86::rsp, i * sizeof(int32_t)), 0);
            }
            a->vmovups(mask_vreg, x86::dword_ptr(x86::rsp));
            a->lea(
                x86::rsp,
                x86::dword_ptr(x86::rsp, (int32_t)(vlen * sizeof(int32_t))));

          } else {
            a->mov(temp1_, (1 << remainder) - 1);
            a->kmovw(x86::k(1), temp1_);
          }
        }

        unroll_factor = unroll_factor / 2; // accont for nw

        asmjit::Label exit = a->newLabel();
        asmjit::Label LoopRangeIndexBegin = a->newLabel();
        asmjit::Label LoopRangeIndexEnd = a->newLabel();

        a->vpbroadcastd(epsilon_vreg, epsilon);
        a->vpbroadcastd(lr_vreg, lr);

        a->xor_(temp1_, temp1_);

        a->bind(LoopRangeIndexBegin);
        a->cmp(temp1_.r32(), num_rows); // temp1_ is the loop trip counter
        a->jge(LoopRangeIndexEnd);

        // set offset to zero
        a->xor_(base_offset_g, base_offset_g);

        if (rowwise) {
          if (areIndices64b) {
            a->imul(
                base_offset,
                x86::qword_ptr(
                    indices,
                    temp1_,
                    3), // use of 3 is to muliply by 8 (int64_t)
                static_cast<asmjit::Imm>(sizeof(float)));
          } else {
            a->imul(
                base_offset.r32(),
                x86::dword_ptr(
                    indices,
                    temp1_,
                    2), // use of 2 is to muliply by 4 (int32_t)
                static_cast<asmjit::Imm>(sizeof(float)));
          }

        } else { // sparse adagrad

          if (areIndices64b) {
            a->imul(
                base_offset,
                x86::qword_ptr(
                    indices,
                    temp1_,
                    3), // use of 3 is to muliply by 8 (int64_t)
                static_cast<asmjit::Imm>(block_size * sizeof(float)));
          } else {
            a->imul(
                base_offset.r32(),
                x86::dword_ptr(
                    indices,
                    temp1_,
                    2), // use of 2 is to muliply by 4 (int32_t)
                static_cast<asmjit::Imm>(block_size * sizeof(float)));
          }
        }

        // Perform this check
        // if (block_size + offsetIdx > param_size) {
        //   return i;
        // }
        if (areIndices64b) {
          a->mov(temp2_, x86::qword_ptr(indices, temp1_, 3));
        } else {
          a->mov(temp2_.r32(), x86::dword_ptr(indices, temp1_, 2));
        }
        a->inc(temp2_);
        a->imul(
            temp2_,
            static_cast<asmjit::Imm>(block_size)); //(offsetIdx+1)*blocksize
        a->cmp(temp2_, param_size);
        a->jg(exit);

        if (prefetch) {
          asmjit::Label pref_dist_reset_start = a->newLabel();
          asmjit::Label pref_dist_reset_end = a->newLabel();

          a->mov(temp2_, temp1_);
          a->add(temp2_, prefetch);
          a->cmp(temp2_.r32(), num_rows);
          a->jge(pref_dist_reset_start);

          if (areIndices64b) {
            if (rowwise) {
              a->imul(
                  temp3_,
                  x86::qword_ptr(indices, temp2_, 3),
                  static_cast<asmjit::Imm>(sizeof(float)));
            }
            a->imul(
                temp2_,
                x86::qword_ptr(indices, temp2_, 3),
                static_cast<asmjit::Imm>(block_size * sizeof(float)));
          } else {
            if (rowwise) {
              a->imul(
                  temp3_.r32(),
                  x86::dword_ptr(indices, temp2_, 2),
                  static_cast<asmjit::Imm>(sizeof(float)));
            }
            a->imul(
                temp2_.r32(),
                x86::dword_ptr(indices, temp2_, 2),
                static_cast<asmjit::Imm>(block_size * sizeof(float)));
          }

          a->jmp(pref_dist_reset_end);

          a->bind(pref_dist_reset_start);
          if (areIndices64b) {
            a->imul(
                temp2_,
                x86::qword_ptr(indices, temp1_, 3),
                static_cast<asmjit::Imm>(block_size * sizeof(float)));
            if (rowwise) {
              a->imul(
                  temp3_,
                  x86::qword_ptr(indices, temp1_, 3),
                  static_cast<asmjit::Imm>(sizeof(float)));
            }
          } else {
            a->imul(
                temp2_.r32(),
                x86::dword_ptr(indices, temp1_, 2),
                static_cast<asmjit::Imm>(block_size * sizeof(float)));
            if (rowwise) {
              a->imul(
                  temp3_.r32(),
                  x86::dword_ptr(indices, temp1_, 2),
                  static_cast<asmjit::Imm>(sizeof(float)));
            }
          }

          a->bind(pref_dist_reset_end);
        } // prefetch

        if (rowwise) {
          genRowwiseSparseAdagrad(
              a,
              block_size,
              vlen,
              unroll_factor,
              num_vec_regs_per_block,
              remainder,
              prefetch,
              epsilon_vreg,
              lr_vreg,
              mask_vreg);
        } else {
          genSparseAdagrad(
              a,
              vlen,
              unroll_factor,
              num_vec_regs_per_block,
              remainder,
              prefetch,
              epsilon_vreg,
              lr_vreg,
              mask_vreg,
              temp_vreg);
        }

        a->add(g, static_cast<asmjit::Imm>(block_size * sizeof(float)));
        a->inc(temp1_);
        a->jmp(LoopRangeIndexBegin);
        a->bind(LoopRangeIndexEnd);

        a->bind(exit);
        a->mov(x86::eax, temp1_.r32());
        a->emitEpilog(frame);

        typename ReturnFunctionSignature<indxType>::jit_sparse_adagrad_kernel
            fn;
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
} // namespace

} // namespace

template <typename IndexType>
int SparseAdaGrad(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const IndexType* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise,
    int prefetch) {
  static GenSparseAdagrad<IndexType, inst_set_t::avx2> kernel_generator;
  if (!cpuinfo_initialize()) {
    throw std::runtime_error("Failed to initialize cpuinfo!");
  }
  typename ReturnFunctionSignature<IndexType>::jit_sparse_adagrad_kernel fn;
  // There is a AVX512 implementation, but for perf reasons we only call AVX2
  if (fbgemmHasAvx512Support() || fbgemmHasAvx2Support()) {
    fn = kernel_generator.getOrCreate(block_size, prefetch, rowwise);
  } else {
#ifdef VLOG
    VLOG(0) << "AVX2 or AVX512 not found, taking the slow path";
#endif
    auto success = sparse_adagrad_ref(
        num_rows, // number of rows reading
        block_size, // number of parameters per rows
        param_size, // total number of parameters
        w, // input parameters
        g, // input gradients
        h, // input momentums
        indices,
        epsilon,
        lr);

    return success;
  }
  auto success =
      fn(num_rows,
         param_size, // total number of parameters
         w, // input parameters
         g, // input gradients
         h, // input momentums
         indices,
         epsilon,
         lr);
  return success;
}

template int SparseAdaGrad(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int64_t* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise,
    int prefetch);

template int SparseAdaGrad(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int32_t* indices, // indices of each row
    float epsilon,
    float lr,
    bool rowwise,
    int prefetch);

} // namespace fbgemm
