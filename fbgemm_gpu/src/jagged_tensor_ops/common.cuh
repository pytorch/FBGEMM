/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <ATen/cuda/Atomic.cuh>
#include <cub/cub.cuh>

// clang-format off
#include "fbgemm_gpu/utils/cub_namespace_prefix.cuh"
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/utils/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"
#include "fbgemm_gpu/utils/binary_search_range.cuh"
#include "fbgemm_gpu/utils/cuda_block_count.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"
#include "fbgemm_gpu/utils/fixed_divisor.cuh"
#include "fbgemm_gpu/utils/inclusive_sum_scan.cuh"
#include "fbgemm_gpu/utils/ops_utils.h"
#include "fbgemm_gpu/utils/shared_memory.cuh"
#include "fbgemm_gpu/utils/tensor_accessor.h"
#include "fbgemm_gpu/utils/tensor_utils.h"
#include "fbgemm_gpu/utils/vec4.cuh"

namespace fbgemm_gpu {

using Tensor = at::Tensor;

// A wrapper class for passing dynamically sized dimension information (e.g.
// tensor.dims()) from the host to device.
constexpr size_t kStackArrayMaxDims = 5;

template <typename T>
struct StackArray {
  T vals[kStackArrayMaxDims];
  size_t ndim;
};

namespace {

// template <typename T>
// struct SharedMemory;

// template <>
// struct SharedMemory<int64_t> {
//   __device__ int64_t* getPointer() {
//     extern __shared__ int64_t s_int64_t[];
//     return s_int64_t;
//   }
// };

// template <>
// struct SharedMemory<int32_t> {
//   __device__ int32_t* getPointer() {
//     extern __shared__ int32_t s_int32_t[];
//     return s_int32_t;
//   }
// };

/// @defgroup jagged-tensor-ops-cuda Jagged Tensor CUDA Operators
/// The following are Jagged Tensor CUDA Operators
///

/**
 * Ref. http://tensor-compiler.org/kjolstad-oopsla17-tensor-compiler.pdf
 * @param offset the input value points to the offset in the first jagged dim
 *               and output is the final offset to access the value tensor.
 *               It would've been better if we return a pair including this
 *               offset but CUDA doesn't seem to have comprehensive support
 *               on std::pair like std::tie.
 * @returns true if the flattend jagged idx points to zero'ed (masked out)
 *               portion of the jagged tensor
 */
template <int NUM_JAGGED_DIM, typename index_t>
DEVICE_INLINE bool walk_down_tensor_storage_tree_(
    int& offset,
    const int flattened_jagged_idx,
    const StackArray<int64_t>& jagged_dims,
    const StackArray<index_t*>& x_offsets) {
  // compute coorindates
  int jagged_coords[NUM_JAGGED_DIM];
  int j_temp = flattened_jagged_idx;
#pragma unroll
  for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
    const int jagged_size = jagged_dims.vals[d];
    jagged_coords[d] = j_temp % jagged_size;
    j_temp /= jagged_size;
  }

  // walk down the tree
  bool is_zero = false;
#pragma unroll
  for (int d = 0; d < NUM_JAGGED_DIM; ++d) {
    const int begin = x_offsets.vals[d][offset];
    const int end = x_offsets.vals[d][offset + 1];
    if (jagged_coords[d] >= end - begin) {
      is_zero = true;
      break;
    }
    offset = begin + jagged_coords[d];
  }
  return is_zero;
}

// output = f(x, y) where x is jagged, y is dense, and output is dense.
// A generic elementwise operation between a jagged tensor and a dense tensor
// This kernel assumes jagged dims are clustered together, preceded by outer
// dense dimensions and followed by inner dense dimensions.
// The outer/inner dense dimensions, and jagged dimensions in between are
// assumed to be folded so physically the dense tensor is 3D and the value of
// jagged tensor is 2D.
// To support arbitrary number of jagged dimensions, we pass a vector of
// pointers to offset tensors (this is ugly and probably we can use nested
// tensor here).
// This kernel parallelizes the (folded) inner dense dimension across
// blockDim.x so the inner dense dimension should be similar to or bigger than
// warp size.
// We rely on compiler unrolling the compiler time constant NUM_JAGGED_DIM.
template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
__global__
__launch_bounds__(kMaxThreads) void jagged_dense_elementwise_dense_output_kernel_(
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    StackArray<index_t*> x_offsets,
    const pta::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y,
    pta::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> output,
    StackArray<int64_t> jagged_dims,
    F f,
    const scalar_t padding_value) {
  const int outer_dense_size = y.size(0);
  const int jagged_folded_size = y.size(1);
  const int inner_dense_size = y.size(2);

  const int outer_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int outer_stride = gridDim.x * blockDim.y;
  for (int outer = outer_begin; outer < outer_dense_size * jagged_folded_size;
       outer += outer_stride) {
    const int oidx = outer / jagged_folded_size;
    const int jidx = outer % jagged_folded_size;

    int offset = oidx;
    const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
        offset, jidx, jagged_dims, x_offsets);

    if (is_zero) {
      int iidx;
      for (iidx = threadIdx.x; iidx * 2 + 1 < inner_dense_size;
           iidx += blockDim.x) {
        output[oidx][jidx][2 * iidx] =
            f(padding_value, y[oidx][jidx][2 * iidx]);
        output[oidx][jidx][2 * iidx + 1] =
            f(padding_value, y[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output[oidx][jidx][2 * iidx] =
            f(padding_value, y[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = threadIdx.x; iidx * 2 + 1 < inner_dense_size;
           iidx += blockDim.x) {
        output[oidx][jidx][2 * iidx] =
            f(x_values[offset][2 * iidx], y[oidx][jidx][2 * iidx]);
        output[oidx][jidx][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], y[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output[oidx][jidx][2 * iidx] =
            f(x_values[offset][2 * iidx], y[oidx][jidx][2 * iidx]);
      }
    }
  }
}

inline std::tuple<dim3, dim3, StackArray<int64_t>> check_shape_and_partition_(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense_tensor) {
  const int outer_dense_size = dense_tensor.size(0);
  TORCH_CHECK(
      outer_dense_size == offsets[0].numel() - 1,
      "outer_dense_size, ",
      outer_dense_size,
      " != offsets[0].numel() - 1, ",
      offsets[0].numel() - 1);
  const int inner_dense_size = dense_tensor.size(-1);
  TORCH_CHECK(
      inner_dense_size == values.size(-1),
      "inner_dense_size, ",
      inner_dense_size,
      " != values.size(-1), ",
      values.size(-1));
  const int jagged_folded_size =
      dense_tensor.numel() / (outer_dense_size * inner_dense_size);

  const int threads_x =
      inner_dense_size >= kWarpSize / 2 ? kWarpSize : inner_dense_size;
  const int threads_y = kMaxThreads / kWarpSize;
  const dim3 blocks(
      div_round_up(outer_dense_size * jagged_folded_size, threads_y));

  StackArray<int64_t> jagged_dims_tensor;
  const int num_jagged_dim = dense_tensor.dim() - 2;
  TORCH_CHECK(num_jagged_dim <= kStackArrayMaxDims);
  jagged_dims_tensor.ndim = num_jagged_dim;
  std::memcpy(
      &(jagged_dims_tensor.vals[0]),
      dense_tensor.sizes().data() + 1,
      num_jagged_dim * sizeof(int64_t));
  return {dim3(threads_x, threads_y), blocks, jagged_dims_tensor};
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim ",
      num_jagged_dim);

  if (y.numel() == 0) {
    return;
  }

  dim3 threads, blocks;
  StackArray<int64_t> jagged_dims_tensor;
  std::tie(threads, blocks, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, y);

  // Canonicalize y and output to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  Tensor output_reshaped = output.view(y_reshaped.sizes());

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                               \
  {                                                                          \
    std::vector<Tensor> x_offsets_contig;                                    \
    x_offsets_contig.resize(num_jagged_dim);                                 \
    StackArray<index_t*> x_offset_ptrs;                                      \
    x_offset_ptrs.ndim = num_jagged_dim;                                     \
    for (int d = 0; d < num_jagged_dim; ++d) {                               \
      x_offsets_contig[d] = x_offsets[d].contiguous();                       \
      x_offset_ptrs.vals[d] =                                                \
          x_offsets_contig[d].template data_ptr<index_t>();                  \
    }                                                                        \
    [[maybe_unused]] const auto func_name =                                  \
        "jagged_dense_elementwise_dense_output_kernel_";                     \
    jagged_dense_elementwise_dense_output_kernel_<NUM_JAGGED_DIM, index_t>   \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(          \
            MAKE_PTA_WITH_NAME(func_name, x_values, scalar_t, 2, 32),        \
            x_offset_ptrs,                                                   \
            MAKE_PTA_WITH_NAME(func_name, y_reshaped, scalar_t, 3, 32),      \
            MAKE_PTA_WITH_NAME(func_name, output_reshaped, scalar_t, 3, 32), \
            jagged_dims_tensor,                                              \
            f,                                                               \
            padding_value);                                                  \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#undef INVOKE_KERNEL_WITH_DIM
}

template <typename scalar_t, typename F>
Tensor jagged_dense_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
  Tensor output = at::empty_like(y);
  jagged_dense_elementwise_dense_output_(
      x_values, x_offsets, y, output, f, padding_value);
  return output;
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
__global__
__launch_bounds__(kMaxThreads) void jagged_dense_dense_elementwise_jagged_output_kernel_(
    const pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    StackArray<index_t*> x_offsets,
    StackArray<int64_t> x_offsets_sizes,
    const pta::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y_0,
    const pta::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y_1,
    pta::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        output_values,
    StackArray<int64_t> jagged_dims,
    F f) {
  const int outer_dense_size = y_0.size(0);
  const int inner_dense_size = y_0.size(2);
  const int nnz = x_values.size(0);

  const int offset_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int offset_stride = gridDim.x * blockDim.y;
  for (int offset = offset_begin; offset < nnz; offset += offset_stride) {
    int offset_temp = offset;
    int jidx = 0;
    bool truncated = false;
    int dim_prod = 1;
#pragma unroll
    for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
      // Binary search the first that is bigger than offset
      int count = x_offsets_sizes.vals[d] - 1;
      int first = 1;
      while (count > 0) {
        int idx = first;
        int step = count / 2;
        idx += step;
        if (x_offsets.vals[d][idx] <= offset_temp) {
          first = ++idx;
          count -= step + 1;
        } else {
          count = step;
        }
      }

      --first;
      int coord = offset_temp - x_offsets.vals[d][first];
      if (coord >= jagged_dims.vals[d]) {
        truncated = true;
        break;
      }
      jidx += coord * dim_prod;
      dim_prod *= jagged_dims.vals[d];
      offset_temp = first;
    }

    if (offset_temp >= outer_dense_size) {
      // This can happen when values have more elements than the last element of
      // offset
      truncated = true;
    }
    if (!truncated) {
      const int oidx = offset_temp;
      int iidx;
      for (iidx = threadIdx.x; iidx * 2 + 1 < inner_dense_size;
           iidx += blockDim.x) {
        output_values[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
        output_values[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1],
              y_0[oidx][jidx][2 * iidx + 1],
              y_1[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = threadIdx.x; iidx * 2 + 1 < inner_dense_size;
           iidx += blockDim.x) {
        output_values[offset][2 * iidx] = f(x_values[offset][2 * iidx], 0, 0);
        output_values[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], 0, 0);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values[offset][2 * iidx] = f(x_values[offset][2 * iidx], 0, 0);
      }
    }
  }
}

template <typename index_t>
__global__ void jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_(
    const pta::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets,
    pta::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> rows,
    pta::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> cols,
    int nnz,
    int B) {
  struct SharedMemory<index_t> smem;
  index_t* offsets_sh = smem.getPointer();

  for (int i = threadIdx.x; i < B + 1; i += blockDim.x) {
    offsets_sh[i] = offsets[i];
  }
  __syncthreads();
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= nnz)
    return;
  int first = -1;
  int count = B - 1;
  first = 1;
  while (count > 0) {
    int idx = first;
    int step = count / 2;
    idx += step;
    if (offsets_sh[idx] <= row) {
      first = ++idx;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  --first;

  int dense_row = first;
  int offset = offsets_sh[dense_row];
  int dense_col = row - offset;
  rows[row] = dense_row;
  cols[row] = dense_col;
}

struct VecType128 {
  typedef float4 TType; // Transaction Type
  typedef struct __align__(16) {
    __half a, b, c, d, w, x, y, z;
  }
  half8;

  union Data {
    half8 val;
    TType mask;
  } data;

  __device__ VecType128() {
    data.mask = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
};

struct VecType64 {
  typedef float2 TType; // Transaction Type
  typedef struct __align__(8) {
    __half a, b, c, d;
  }
  half4;

  union Data {
    half4 val;
    TType mask;
  } data;

  __device__ VecType64() {
    data.mask = make_float2(0.0f, 0.0f);
  }
};

struct VecType32 {
  typedef float TType; // Transaction Type

  union Data {
    __half2 val;
    TType mask;
  } data;

  __device__ VecType32() {
    data.mask = 0.0f;
  }
};

template <typename F>
__device__ void f128(
    VecType128& v_out,
    const VecType128& x,
    const VecType128& y0,
    const VecType128& y1,
    F f) {
  v_out.data.val.a = f(x.data.val.a, y0.data.val.a, y1.data.val.a);
  v_out.data.val.b = f(x.data.val.b, y0.data.val.b, y1.data.val.b);
  v_out.data.val.c = f(x.data.val.c, y0.data.val.c, y1.data.val.c);
  v_out.data.val.d = f(x.data.val.d, y0.data.val.d, y1.data.val.d);
  v_out.data.val.w = f(x.data.val.w, y0.data.val.w, y1.data.val.w);
  v_out.data.val.x = f(x.data.val.x, y0.data.val.x, y1.data.val.x);
  v_out.data.val.y = f(x.data.val.y, y0.data.val.y, y1.data.val.y);
  v_out.data.val.z = f(x.data.val.z, y0.data.val.z, y1.data.val.z);
}

template <typename F>
__device__ void f64(
    VecType64& v_out,
    const VecType64& x,
    const VecType64& y0,
    const VecType64& y1,
    F f) {
  v_out.data.val.a = f(x.data.val.a, y0.data.val.a, y1.data.val.a);
  v_out.data.val.b = f(x.data.val.b, y0.data.val.b, y1.data.val.b);
  v_out.data.val.c = f(x.data.val.c, y0.data.val.c, y1.data.val.c);
  v_out.data.val.d = f(x.data.val.d, y0.data.val.d, y1.data.val.d);
}

template <typename F>
__device__ void f32(
    VecType32& v_out,
    const VecType32& x,
    const VecType32& y0,
    const VecType32& y1,
    F f) {
  v_out.data.val = __halves2half2(
      f(__low2half(x.data.val),
        __low2half(y0.data.val),
        __low2half(y1.data.val)),
      f(__high2half(x.data.val),
        __high2half(y0.data.val),
        __high2half(y1.data.val)));
}

template <typename F>
__device__ void
fh(__half& v_out, const __half& x, const __half& y0, const __half& y1, F f) {
  v_out = f(x, y0, y1);
}

template <typename index_t, typename F>
__global__ void jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_(
    pta::PackedTensorAccessor32<c10::Half, 2, at::RestrictPtrTraits> values,
    const pta::PackedTensorAccessor32<c10::Half, 2, at::RestrictPtrTraits>
        x_values,
    const pta::PackedTensorAccessor32<c10::Half, 3, at::RestrictPtrTraits> y0,
    const pta::PackedTensorAccessor32<c10::Half, 3, at::RestrictPtrTraits> y1,
    const pta::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> rows,
    const pta::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> cols,
    const int nnz,
    const int E,
    F f) {
  int values_row = threadIdx.y + blockIdx.y * blockDim.y;
  if (values_row >= nnz)
    return;
  for (int real_row = values_row; real_row < nnz;
       real_row += blockDim.y * gridDim.y) {
    int dense_row = rows[real_row];
    int dense_col = cols[real_row];
    __half* values_ptr = reinterpret_cast<__half*>(&values[real_row][0]);
    const __half* x_ptr =
        reinterpret_cast<const __half*>(&x_values[real_row][0]);
    const __half* y0_ptr =
        reinterpret_cast<const __half*>(&y0[dense_row][dense_col][0]);
    const __half* y1_ptr =
        reinterpret_cast<const __half*>(&y1[dense_row][dense_col][0]);
    if ((dense_col < y0.size(1)) && (dense_row < y0.size(0)) &&
        (dense_col < y1.size(1)) && (dense_row < y1.size(0)) &&
        (dense_col >= 0) && (dense_row >= 0)) {
      for (int tid = threadIdx.x; tid < E / 8; tid += blockDim.x) {
        VecType128 v_x, v_out, v_y0, v_y1;
        v_x.data.mask =
            (reinterpret_cast<const VecType128::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType128::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType128::TType*>(y1_ptr))[tid];
        f128(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType128::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 8) * 8; tid < E / 4;
           tid += blockDim.x) {
        VecType64 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType64::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType64::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType64::TType*>(y1_ptr))[tid];
        f64(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType64::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 4) * 4; tid < E / 2;
           tid += blockDim.x) {
        VecType32 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType32::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType32::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType32::TType*>(y1_ptr))[tid];
        f32(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType32::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 2) * 2; tid < E; tid += blockDim.x) {
        __half v_x, v_out, v_y0, v_y1;
        v_x = static_cast<__half>(x_ptr[tid]);
        v_y0 = static_cast<__half>(y0_ptr[tid]);
        v_y1 = static_cast<__half>(y1_ptr[tid]);
        fh(v_out, v_x, v_y0, v_y1, f);
        values_ptr[tid] = v_out;
      }
    } else {
      for (int tid = threadIdx.x; tid < E / 8; tid += blockDim.x) {
        VecType128 v_x, v_out, v_y0, v_y1;
        v_x.data.mask =
            (reinterpret_cast<const VecType128::TType*>(x_ptr))[tid];
        f128(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType128::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 8) * 8; tid < E / 4;
           tid += blockDim.x) {
        VecType64 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType64::TType*>(x_ptr))[tid];
        f64(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType64::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 4) * 4; tid < E / 2;
           tid += blockDim.x) {
        VecType32 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType32::TType*>(x_ptr))[tid];
        f32(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType32::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (int tid = threadIdx.x + (E / 2) * 2; tid < E; tid += blockDim.x) {
        __half v_x, v_out, v_y0, v_y1;
        v_x = static_cast<__half>(x_ptr[tid]);
        fh(v_out, v_x, v_y0, v_y1, f);
        values_ptr[tid] = v_out;
      }
    }
  }
}

} // namespace

// Check to see if the inputs to the op are amenable to the fast path
inline bool jagged_dense_dense_elementwise_jagged_output_matches_opt(
    const int& num_jagged_dim,
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0_reshaped,
    const Tensor& y_1_reshaped,
    const Tensor& output_values) {
  bool matches = true;
  matches &= (num_jagged_dim == 1);

  // Unit stride embedding dim
  matches &= (x_values.stride(-1) == 1);
  matches &= (output_values.stride(-1) == 1);
  matches &= (y_0_reshaped.stride(-1) == 1);
  matches &= (y_1_reshaped.stride(-1) == 1);

  // Each row is aligned to 128-bit
  matches &= (x_values.stride(-2) % 8 == 0);
  matches &= (output_values.stride(-2) % 8 == 0);
  matches &= (y_0_reshaped.stride(-2) % 8 == 0);
  matches &= (y_1_reshaped.stride(-2) % 8 == 0);

  // Base addresses aligned to 128-bit
  matches &= (reinterpret_cast<uint64_t>(x_values.data_ptr()) % 16 == 0);
  matches &= (reinterpret_cast<uint64_t>(output_values.data_ptr()) % 16 == 0);
  matches &= (reinterpret_cast<uint64_t>(y_0_reshaped.data_ptr()) % 16 == 0);
  matches &= (reinterpret_cast<uint64_t>(y_1_reshaped.data_ptr()) % 16 == 0);

  // Rows and col fit into int32_t
  matches &= (y_0_reshaped.size(0) < INT_MAX);
  matches &= (y_0_reshaped.size(1) < INT_MAX);

  int max_shared_bytes;
#ifndef USE_ROCM
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared_bytes,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      y_0_reshaped.get_device()));
#else
  // MI100 has 64 KB local memory (shared memory) per workgroup
  max_shared_bytes = 64 << 10;
#endif
  int shared_kb = max_shared_bytes >> 10;
#ifndef USE_ROCM
  // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
  int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
  TORCH_CHECK(used_shared_kb > 0);
#else
  // MI100 has independent shared mem and L1
  int used_shared_kb = shared_kb;
#endif
  int used_shared_bytes = used_shared_kb << 10;
  AT_DISPATCH_INDEX_TYPES(
      x_offsets[0].scalar_type(), "check_shared_memory", [&] {
        auto B = y_0_reshaped.size(0);
        // the default shared memory on V100/A100/H100 is 48 KB from
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
        if ((B + 1) * sizeof(index_t) >= used_shared_bytes) {
          matches = false;
        }
      });
  return matches;
}

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                              \
  {                                                                         \
    dim3 threads, blocks;                                                   \
    StackArray<int64_t> jagged_dims_tensor;                                 \
    std::tie(threads, blocks, jagged_dims_tensor) =                         \
        check_shape_and_partition_(x_values, x_offsets, y);                 \
    blocks.x = div_round_up(x_values.size(0), threads.y);                   \
    std::vector<Tensor> x_offsets_contig;                                   \
    x_offsets_contig.resize(num_jagged_dim);                                \
    StackArray<index_t*> x_offset_ptrs;                                     \
    x_offset_ptrs.ndim = num_jagged_dim;                                    \
    StackArray<int64_t> x_offset_sizes;                                     \
    x_offset_sizes.ndim = num_jagged_dim;                                   \
    for (int d = 0; d < num_jagged_dim; ++d) {                              \
      x_offsets_contig[d] = x_offsets[d].contiguous();                      \
      x_offset_ptrs.vals[d] =                                               \
          x_offsets_contig[d].template data_ptr<index_t>();                 \
      x_offset_sizes.vals[d] = x_offsets[d].numel();                        \
    }                                                                       \
    [[maybe_unused]] const auto func_name =                                 \
        "jagged_dense_dense_elementwise_jagged_output_kernel_";             \
    jagged_dense_dense_elementwise_jagged_output_kernel_<                   \
        NUM_JAGGED_DIM,                                                     \
        index_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
        MAKE_PTA_WITH_NAME(func_name, x_values, scalar_t, 2, 32),           \
        x_offset_ptrs,                                                      \
        x_offset_sizes,                                                     \
        MAKE_PTA_WITH_NAME(func_name, y_reshaped, scalar_t, 3, 32),         \
        MAKE_PTA_WITH_NAME(func_name, y_reshaped, scalar_t, 3, 32),         \
        MAKE_PTA_WITH_NAME(func_name, output_values, scalar_t, 2, 32),      \
        jagged_dims_tensor,                                                 \
        [f] __device__(scalar_t x, scalar_t y, scalar_t /*unused*/)         \
            -> scalar_t { return f(x, y); });                               \
  }

///@addtogroup jagged-tensor-ops-cuda
template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_opt_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (y.numel() == 0 || x_values.numel() == 0) {
    return;
  }

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  if (jagged_dense_dense_elementwise_jagged_output_matches_opt(
          num_jagged_dim,
          x_values,
          x_offsets,
          y_reshaped,
          y_reshaped,
          output_values)) {
    AT_DISPATCH_INDEX_TYPES(
        x_offsets[0].scalar_type(), "jagged_indices_fast_path", [=] {
          auto nnz = output_values.size(0);
          auto B = y_reshaped.size(0);
          auto E = y_reshaped.size(2);
          Tensor t_rows_after_bs = at::empty(
              {nnz},
              at::TensorOptions().dtype(at::kInt).device(
                  at::kCUDA, at::cuda::current_device()));
          Tensor t_cols_after_bs = at::empty(
              {nnz},
              at::TensorOptions().dtype(at::kInt).device(
                  at::kCUDA, at::cuda::current_device()));

          // Binary search
          size_t dynamic_smem_size = (B + 1) * sizeof(index_t);
          auto cur_max_shared_bytes =
              at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
          if (dynamic_smem_size > cur_max_shared_bytes) {
            int max_shared_bytes;
#ifndef USE_ROCM
            C10_CUDA_CHECK(cudaDeviceGetAttribute(
                &max_shared_bytes,
                cudaDevAttrMaxSharedMemoryPerBlockOptin,
                y_reshaped.get_device()));
#else
            // MI100 has 64 KB local memory (shared memory) per workgroup
            max_shared_bytes = 64 << 10;
#endif
            int shared_kb = max_shared_bytes >> 10;
#ifndef USE_ROCM
            // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
            int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
            TORCH_CHECK(used_shared_kb > 0);
#else
            // MI100 has independent shared mem and L1
            int used_shared_kb = shared_kb;
#endif
            int used_shared_bytes = used_shared_kb << 10;
#ifndef USE_ROCM
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
                    index_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes)); // V100: 64 KB; A100: 96 KB; H100: 144 KB
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            TORCH_CHECK(dynamic_smem_size <= used_shared_bytes);
          }
          dim3 threads_bs = dim3(1024, 1, 1);
          dim3 blocks_bs = dim3(div_round_up(nnz, threads_bs.x), 1, 1);
#ifdef FBGEMM_GPU_MEMCHECK
          const auto func_name1 =
              "jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_";
#endif
          jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
              index_t>
              <<<blocks_bs,
                 threads_bs,
                 dynamic_smem_size,
                 at::cuda::getCurrentCUDAStream()>>>(
                  MAKE_PTA_WITH_NAME(func_name1, x_offsets[0], index_t, 1, 32),
                  MAKE_PTA_WITH_NAME(func_name1, t_rows_after_bs, int, 1, 32),
                  MAKE_PTA_WITH_NAME(func_name1, t_cols_after_bs, int, 1, 32),
                  nnz,
                  B);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          // Gather kernel
          dim3 threads = dim3(16, 16, 1);
          dim3 blocks = dim3(1, div_round_up(nnz, threads.y), 1);
          if (blocks.y > 65535) {
            blocks.y = 65535;
          }
#ifdef FBGEMM_GPU_MEMCHECK
          const auto func_name2 =
              "jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_";
#endif
          jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_<
              index_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  MAKE_PTA_WITH_NAME(
                      func_name2, output_values, c10::Half, 2, 32),
                  MAKE_PTA_WITH_NAME(func_name2, x_values, c10::Half, 2, 32),
                  MAKE_PTA_WITH_NAME(func_name2, y_reshaped, c10::Half, 3, 32),
                  MAKE_PTA_WITH_NAME(func_name2, y_reshaped, c10::Half, 3, 32),
                  MAKE_PTA_WITH_NAME(func_name2, t_rows_after_bs, int, 1, 32),
                  MAKE_PTA_WITH_NAME(func_name2, t_cols_after_bs, int, 1, 32),
                  nnz,
                  E,
                  [f] __device__(__half x, __half y0, __half) -> __half {
                    return f(x, y0);
                  });
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }); // AT_DISPATCH
  } else {
    JAGGED_TENSOR_DISPATCH_DIMS();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

///@addtogroup jagged-tensor-ops-cuda
template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (y.numel() == 0 || x_values.numel() == 0) {
    return;
  }

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef INVOKE_KERNEL_WITH_DIM

} // namespace fbgemm_gpu
