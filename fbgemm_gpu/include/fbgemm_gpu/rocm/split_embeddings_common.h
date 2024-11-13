/*******************************************************************************
 * Copyright (c) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 ******************************************************************************/
#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/******************************************************************************/
typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));
typedef float floatx2_t __attribute__((ext_vector_type(2)));
#define AMDGCN_BUFFER_RES_3 0x00027000
#define AMDGCN_WAVE_SIZE 64
#define THREADS_PER_ROW 64
#define BLOCK_SIZE 256

namespace fbgemm_gpu::rocm {
template <typename T> union amdgcn_buffer_resource {
  // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
  int32x4_t content;
  struct {
    T *address;
    int32_t range;
    int32_t config;
  };
};

template <typename T>
__device__ int32x4_t amdgcn_make_buffer_resource(const T *addr) {
  amdgcn_buffer_resource<T> buffer_resource;
  buffer_resource.address = const_cast<T *>(addr);
  buffer_resource.range = 0xffffffff;
  buffer_resource.config = AMDGCN_BUFFER_RES_3; // for gfx9

  return buffer_resource.content;
}

// buffer load fp32
__device__ half llvm_amdgcn_raw_buffer_load_fp16(
    int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16");

__device__ float llvm_amdgcn_raw_buffer_load_fp32(
    int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ half2 llvm_amdgcn_raw_buffer_load_fp16x2(
    int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f16");

__device__ void llvm_amdgcn_raw_buffer_store_fp32(
    float vdata, int32x4_t rsrc, int32_t voffset, int32_t soffset,
    int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");

__device__ void llvm_amdgcn_raw_buffer_store_fp32x2(
    floatx2_t vdata, int32x4_t rsrc, int32_t voffset, int32_t soffset,
    int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

/******************************************************************************/

template <typename emb_t, int32_t embedding_dim, typename index_t>
struct load_row_per_warp {
  static __device__ void run(emb_t *emb_data, index_t row_index,
                             const emb_t *p_emb_table, int lane_id) {}
};

template <int32_t embedding_dim, typename index_t>
struct load_row_per_warp<float, embedding_dim, index_t> {
  static constexpr int dword_per_row =
      (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
  static __device__ void run(float *emb_data, index_t row_index,
                             const float *p_emb_table, int lane_id) {
    int32x4_t emb_res =
        amdgcn_make_buffer_resource(p_emb_table + row_index * embedding_dim);
#pragma unroll
    for (int i = 0; i < dword_per_row; i++) {
      if constexpr (embedding_dim == 160) {
        if ((lane_id + i * THREADS_PER_ROW) % 192 < 160) {
          emb_data[i] = llvm_amdgcn_raw_buffer_load_fp32(
              emb_res, (lane_id + i * THREADS_PER_ROW) * sizeof(float), 0, 0);
        } else {
          emb_data[i] = 0.f;
        }
      } else {
        emb_data[i] = llvm_amdgcn_raw_buffer_load_fp32(
            emb_res, (lane_id + i * THREADS_PER_ROW) * sizeof(float), 0, 0);
      }
    }
  }
};

template <typename index_t> struct load_row_per_warp<half, 64, index_t> {
  static __device__ void run(half *emb_data, index_t row_index,
                             const half *p_emb_table, int lane_id) {
    int32x4_t emb_res =
        amdgcn_make_buffer_resource(p_emb_table + row_index * 64);
    emb_data[0] =
        llvm_amdgcn_raw_buffer_load_fp16(emb_res, lane_id * sizeof(half), 0, 0);
  }
};

template <typename index_t> struct load_row_per_warp<half, 128, index_t> {
  static __device__ void run(half *emb_data, index_t row_index,
                             const half *p_emb_table, int lane_id) {
    int32x4_t emb_res =
        amdgcn_make_buffer_resource(p_emb_table + row_index * 128);
    *reinterpret_cast<half2 *>(emb_data) = llvm_amdgcn_raw_buffer_load_fp16x2(
        emb_res, lane_id * sizeof(half2), 0, 0);
  }
};

template <typename index_t> struct load_row_per_warp<half, 160, index_t> {
  static __device__ void run(half *emb_data, index_t row_index,
                             const half *p_emb_table, int lane_id) {
    int32x4_t emb_res =
        amdgcn_make_buffer_resource(p_emb_table + row_index * 192);
    *reinterpret_cast<half2 *>(emb_data) = llvm_amdgcn_raw_buffer_load_fp16x2(
        emb_res, lane_id * sizeof(half2), 0, 0);
    if ((lane_id + 128) % 192 < 160) {
      emb_data[2] = llvm_amdgcn_raw_buffer_load_fp16(
          emb_res, (lane_id + 128) * sizeof(half), 0, 0);
    } else {
      emb_data[2] = __float2half(0.0);
    }
  }
};

template <typename index_t> struct load_row_per_warp<half, 192, index_t> {
  static __device__ void run(half *emb_data, index_t row_index,
                             const half *p_emb_table, int lane_id) {
    int32x4_t emb_res =
        amdgcn_make_buffer_resource(p_emb_table + row_index * 192);
    *reinterpret_cast<half2 *>(emb_data) = llvm_amdgcn_raw_buffer_load_fp16x2(
        emb_res, lane_id * sizeof(half2), 0, 0);
    emb_data[2] = llvm_amdgcn_raw_buffer_load_fp16(
        emb_res, (lane_id + 128) * sizeof(half), 0, 0);
  }
};

template <typename index_t> struct load_row_per_warp<half, 256, index_t> {
  static __device__ void run(half *emb_data, index_t row_index,
                             const half *p_emb_table, int lane_id) {
    int32x4_t emb_res =
        amdgcn_make_buffer_resource(p_emb_table + row_index * 256);
    *reinterpret_cast<half2 *>(&emb_data[0]) =
        llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, lane_id * sizeof(half2), 0,
                                           0);
    *reinterpret_cast<half2 *>(&emb_data[2]) =
        llvm_amdgcn_raw_buffer_load_fp16x2(
            emb_res, (lane_id + 64) * sizeof(half2), 0, 0);
  }
};

template <typename index_t> struct load_row_per_warp<half, 512, index_t> {
  static __device__ void run(half *emb_data, index_t row_index,
                             const half *p_emb_table, int lane_id) {
    int32x4_t emb_res =
        amdgcn_make_buffer_resource(p_emb_table + row_index * 512);
    *reinterpret_cast<half2 *>(&emb_data[0]) =
        llvm_amdgcn_raw_buffer_load_fp16x2(emb_res, lane_id * sizeof(half2), 0,
                                           0);
    *reinterpret_cast<half2 *>(&emb_data[2]) =
        llvm_amdgcn_raw_buffer_load_fp16x2(
            emb_res, (lane_id + 64) * sizeof(half2), 0, 0);
    *reinterpret_cast<half2 *>(&emb_data[4]) =
        llvm_amdgcn_raw_buffer_load_fp16x2(
            emb_res, (lane_id + 64 * 2) * sizeof(half2), 0, 0);
    *reinterpret_cast<half2 *>(&emb_data[6]) =
        llvm_amdgcn_raw_buffer_load_fp16x2(
            emb_res, (lane_id + 64 * 3) * sizeof(half2), 0, 0);
  }
};

template <typename emb_t, int32_t embedding_dim, typename output_t,
          bool weighted>
struct accumulate_row_per_warp {
  static constexpr int dword_per_row =
      (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
  static __device__ void run(output_t *acc, emb_t *emb_data, int lane_id,
                             float row_weight = 1.0) {
    if constexpr (!weighted) {
#pragma unroll
      for (int i = 0; i < dword_per_row; i++) {
        acc[i] += static_cast<output_t>(emb_data[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < dword_per_row; i++) {
        acc[i] += static_cast<output_t>((float)emb_data[i] * row_weight);
      }
    }
  }
};

template <typename emb_t, int32_t embedding_dim, typename output_t>
struct store_row_per_warp {
  static constexpr int dword_per_row =
      (embedding_dim + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
  static __device__ void run(output_t *acc, output_t *p_output, int lane_id) {
    if constexpr (embedding_dim == 160) {
      for (int i = 0; i < dword_per_row; i++) {
        if ((lane_id + i * THREADS_PER_ROW) % 192 < 160) {
          p_output[lane_id + i * THREADS_PER_ROW] = acc[i];
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < dword_per_row; i++) {
        p_output[lane_id + i * THREADS_PER_ROW] = acc[i];
      }
    }
  }
};

template <> struct store_row_per_warp<half, 128, float> {
  static __device__ void run(float *acc, float *p_output, int lane_id) {
    int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
    llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t *>(acc),
                                        out_res, lane_id * sizeof(floatx2_t), 0,
                                        0);
  }
};

template <> struct store_row_per_warp<half, 160, float> {
  static __device__ void run(float *acc, float *p_output, int lane_id) {
    int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
    llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t *>(acc),
                                        out_res, lane_id * sizeof(floatx2_t), 0,
                                        0);
    if ((lane_id + 128) % 192 < 160) {
      llvm_amdgcn_raw_buffer_store_fp32(acc[2], out_res,
                                        (lane_id + 128) * sizeof(float), 0, 0);
    }
  }
};

template <> struct store_row_per_warp<half, 192, float> {
  static __device__ void run(float *acc, float *p_output, int lane_id) {
    int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
    llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t *>(acc),
                                        out_res, lane_id * sizeof(floatx2_t), 0,
                                        0);
    llvm_amdgcn_raw_buffer_store_fp32(acc[2], out_res,
                                      (lane_id + 128) * sizeof(float), 0, 0);
  }
};

template <> struct store_row_per_warp<half, 256, float> {
  static __device__ void run(float *acc, float *p_output, int lane_id) {
    int32x4_t out_res = amdgcn_make_buffer_resource(p_output);
    llvm_amdgcn_raw_buffer_store_fp32x2(*reinterpret_cast<floatx2_t *>(acc),
                                        out_res, lane_id * sizeof(floatx2_t), 0,
                                        0);
    llvm_amdgcn_raw_buffer_store_fp32x2(
        *reinterpret_cast<floatx2_t *>(&acc[2]), out_res,
        (lane_id + 64) * sizeof(floatx2_t), 0, 0);
  }
};

// Helper function to pack fp16 and fp32 into int to further pass
// into mov_dpp and readfirstlane()
template <typename to_t, typename from_t>
  requires((sizeof(to_t) == 4 || sizeof(to_t) == 2) &&
           (sizeof(from_t) == 4 || sizeof(from_t) == 2))
__device__ to_t pack(const from_t &v) {
  to_t result = 0;
  if constexpr (sizeof(to_t) == sizeof(from_t)) {
    result = __builtin_bit_cast(to_t, v);
    return result;
  }

  memcpy(&result, &v, 2);

  return result;
}

namespace reduce_op {
struct sum {};
struct sub {};
struct mul {};
struct div {};
} // namespace reduce_op

template <typename data_t> struct reduce_op_sum_t {
  __device__ data_t operator()(const data_t &a, const data_t &b) {
    return a + b;
  }
};

#define DPP_REDUCE(OP, TYPE)                                                   \
  __asm__ volatile("v_nop\n"                                                   \
                   "v_nop\n"                                                   \
                   "v_nop\n"                                                   \
                   "v_nop\n"                                                   \
                   "v_" #OP "_" #TYPE "_dpp %0 %0 %0 quad_perm:[1,0,3,2]\n"    \
                   "v_nop\n"                                                   \
                   "v_nop\n"                                                   \
                   "v_" #OP "_" #TYPE "_dpp %0 %0 %0 quad_perm:[2,3,0,1]\n"    \
                   "v_nop\n"                                                   \
                   "v_nop\n"                                                   \
                   "v_" #OP "_" #TYPE "_dpp %0 %0 %0 row_shr:4\n"              \
                   "v_nop\n"                                                   \
                   "v_nop\n"                                                   \
                   "v_" #OP "_" #TYPE "_dpp %0 %0 %0 row_shr:8\n"              \
                   "v_nop\n"                                                   \
                   "v_nop\n"                                                   \
                   "v_" #OP "_" #TYPE "_dpp %0 %0 %0 row_bcast:15\n"           \
                   "v_nop\n"                                                   \
                   "v_nop\n"                                                   \
                   "v_" #OP "_" #TYPE "_dpp %0 %0 %0 row_bcast:31\n"           \
                   "v_nop\n"                                                   \
                   "v_nop\n"                                                   \
                   : "=v"(result)                                              \
                   : "0"(result))

#define DPP_REDUCE_F16_F32(OP)                                                 \
  if constexpr (std::is_same_v<data_t, float>) {                               \
    DPP_REDUCE(OP, f32);                                                       \
  }                                                                            \
                                                                               \
  if constexpr (std::is_same_v<data_t, c10::Half>) {                           \
    DPP_REDUCE(OP, f16);                                                       \
  }

template <typename data_t, typename reduce_op_t, int wave_size>
__device__ __forceinline__ void generic_dpp_reduction(data_t &result) {
  constexpr int row_mask = 0xf;
  constexpr int bank_mask = 0xf;
  constexpr bool bound_ctrl = false;

  reduce_op_t reduce_op;

  if constexpr (wave_size > 1) {
    result = reduce_op(result,
                       pack<data_t, int>(__builtin_amdgcn_mov_dpp(
                           pack<int, data_t>(result), 0xb1, row_mask, bank_mask,
                           bound_ctrl))); // quad_perm:[1,0,3,2]
  }
  if constexpr (wave_size > 2) {
    result = reduce_op(result,
                       pack<data_t, int>(__builtin_amdgcn_mov_dpp(
                           pack<int, data_t>(result), 0x4e, row_mask, bank_mask,
                           bound_ctrl))); // quad_perm:[2,3,0,1]
  }
  if constexpr (wave_size > 4) {
    result = reduce_op(
        result, pack<data_t, int>(__builtin_amdgcn_mov_dpp(
                    pack<int, data_t>(result), 0x114, row_mask, bank_mask,
                    bound_ctrl))); // row_shr:4
  }
  if constexpr (wave_size > 8) {
    result = reduce_op(
        result, pack<data_t, int>(__builtin_amdgcn_mov_dpp(
                    pack<int, data_t>(result), 0x118, row_mask, bank_mask,
                    bound_ctrl))); // row_shr:8
  }
  if constexpr (wave_size > 16) {
    result = reduce_op(
        result, pack<data_t, int>(__builtin_amdgcn_mov_dpp(
                    pack<int, data_t>(result), 0x142, row_mask, bank_mask,
                    bound_ctrl))); // row_bcast:15
  }
  if constexpr (wave_size > 32) {
    result = reduce_op(
        result, pack<data_t, int>(__builtin_amdgcn_mov_dpp(
                    pack<int, data_t>(result), 0x143, row_mask, bank_mask,
                    bound_ctrl))); // row_bcast:31
  }
}

// Use corresponding assebly instruction for dpp reduction in case
// of trivial operation with an option to use custom operation
template <typename data_t, typename reduce_op_t, int wave_size = 64>
__device__ __forceinline__ void dpp_reduction(data_t &result) {
#if defined(__gfx942__) || defined(__gfx90a__)
  if constexpr (std::is_same_v<reduce_op_t, reduce_op::sum>) {
    DPP_REDUCE_F16_F32(add);
    return;
  } else if constexpr (std::is_same_v<reduce_op_t, reduce_op::sub>) {
    DPP_REDUCE_F16_F32(sub);
    return;
  } else if constexpr (std::is_same_v<reduce_op_t, reduce_op::mul>) {
    DPP_REDUCE_F16_F32(mul);
    return;
  } else if constexpr (std::is_same_v<reduce_op_t, reduce_op::div>) {
    DPP_REDUCE_F16_F32(div);
    return;
  } else {
    generic_dpp_reduction<data_t, reduce_op_t, wave_size>(result);
  }
#endif
}

template <typename reduce_op_t, typename data_t, int wave_size>
__device__ inline data_t wave_reduce(const data_t &thread_data) {
  data_t result = thread_data;

  // now the reduced value is in the last lane of wave
  dpp_reduction<data_t, reduce_op::sum, wave_size>(result);
  return pack<data_t, int>(
      __builtin_amdgcn_readlane(pack<int, data_t>(result), wave_size - 1));
}

struct rowwise_adagrad_kernel_arg_t {
  void *p_momentum;
  float eps;
  float learning_rate;
  float weight_decay;
  int64_t weight_decay_mode;
};

typedef struct {
  uint32_t magic;
  uint32_t shift; // actually 8 bit is enough
} magic_div_u32_t;

static inline magic_div_u32_t magic_div_u32_gen(uint32_t d) {
  assert(d >= 1 && d <= INT32_MAX);
  uint8_t shift;
  for (shift = 0; shift < 32; shift++)
    if ((1U << shift) >= d)
      break;

  uint64_t one = 1;
  uint64_t magic = ((one << 32) * ((one << shift) - d)) / d + 1;
  assert(magic <= 0xffffffffUL);

  magic_div_u32_t result;
  result.magic = magic;
  result.shift = shift;
  return result;
}

// numer / denom = quotient, reminder
__device__ inline uint32_t magic_div_u32_run(const magic_div_u32_t &mdiv,
                                             const uint32_t &n) {
  uint32_t tmp = __umulhi(n, mdiv.magic);
  return (tmp + n) >> mdiv.shift;
}

__device__ inline void
magic_div_u32_run_with_mod(const magic_div_u32_t &mdiv, const uint32_t &n,
                           const uint32_t d, uint32_t &quo, uint32_t &rem) {
  quo = magic_div_u32_run(mdiv, n);
  rem = n - quo * d;
}
} // namespace fbgemm_gpu::rocm