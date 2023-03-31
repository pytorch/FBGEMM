/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
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
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include <cub/device/device_scan.cuh>
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
#include "fbgemm_gpu/sparse_ops.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {

template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<int64_t> {
  __device__ int64_t* getPointer() {
    extern __shared__ int64_t s_int64_t[];
    return s_int64_t;
  }
};

template <>
struct SharedMemory<int32_t> {
  __device__ int32_t* getPointer() {
    extern __shared__ int32_t s_int32_t[];
    return s_int32_t;
  }
};

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
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    StackArray<index_t*> x_offsets,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y,
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> output,
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

std::tuple<dim3, dim3, StackArray<int64_t>> check_shape_and_partition_(
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

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                \
  {                                                                           \
    std::vector<Tensor> x_offsets_contig;                                     \
    x_offsets_contig.resize(num_jagged_dim);                                  \
    StackArray<index_t*> x_offset_ptrs;                                       \
    x_offset_ptrs.ndim = num_jagged_dim;                                      \
    for (int d = 0; d < num_jagged_dim; ++d) {                                \
      x_offsets_contig[d] = x_offsets[d].contiguous();                        \
      x_offset_ptrs.vals[d] =                                                 \
          x_offsets_contig[d].template data_ptr<index_t>();                   \
    }                                                                         \
    jagged_dense_elementwise_dense_output_kernel_<NUM_JAGGED_DIM, index_t>    \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(           \
            x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
            x_offset_ptrs,                                                    \
            y_reshaped                                                        \
                .packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),     \
            output_reshaped                                                   \
                .packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),     \
            jagged_dims_tensor,                                               \
            f,                                                                \
            padding_value);                                                   \
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
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    StackArray<index_t*> x_offsets,
    StackArray<int64_t> x_offsets_sizes,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y_0,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> y_1,
    at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
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
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> rows,
    at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> cols,
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
    at::PackedTensorAccessor32<c10::Half, 2, at::RestrictPtrTraits> values,
    const at::PackedTensorAccessor32<c10::Half, 2, at::RestrictPtrTraits>
        x_values,
    const at::PackedTensorAccessor32<c10::Half, 3, at::RestrictPtrTraits> y0,
    const at::PackedTensorAccessor32<c10::Half, 3, at::RestrictPtrTraits> y1,
    const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> rows,
    const at::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> cols,
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

// Check to see if the inputs to the op are amenable to the fast path
bool jagged_dense_dense_elementwise_jagged_output_matches_opt(
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
#ifndef __HIP_PLATFORM_HCC__
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared_bytes,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,
      y_0_reshaped.get_device()));
#else
  // MI100 has 64 KB local memory (shared memory) per workgroup
  max_shared_bytes = 64 << 10;
#endif
  int shared_kb = max_shared_bytes >> 10;
#ifndef __HIP_PLATFORM_HCC__
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
        // the default shared memory on V100/A100 is 48 KB from
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
        if ((B + 1) * sizeof(index_t) >= used_shared_bytes) {
          matches = false;
        }
      });
  return matches;
}

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                 \
  {                                                                            \
    dim3 threads, blocks;                                                      \
    StackArray<int64_t> jagged_dims_tensor;                                    \
    std::tie(threads, blocks, jagged_dims_tensor) =                            \
        check_shape_and_partition_(x_values, x_offsets, y);                    \
    blocks.x = div_round_up(x_values.size(0), threads.y);                      \
    std::vector<Tensor> x_offsets_contig;                                      \
    x_offsets_contig.resize(num_jagged_dim);                                   \
    StackArray<index_t*> x_offset_ptrs;                                        \
    x_offset_ptrs.ndim = num_jagged_dim;                                       \
    StackArray<int64_t> x_offset_sizes;                                        \
    x_offset_sizes.ndim = num_jagged_dim;                                      \
    for (int d = 0; d < num_jagged_dim; ++d) {                                 \
      x_offsets_contig[d] = x_offsets[d].contiguous();                         \
      x_offset_ptrs.vals[d] =                                                  \
          x_offsets_contig[d].template data_ptr<index_t>();                    \
      x_offset_sizes.vals[d] = x_offsets[d].numel();                           \
    }                                                                          \
    jagged_dense_dense_elementwise_jagged_output_kernel_<                      \
        NUM_JAGGED_DIM,                                                        \
        index_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(    \
        x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),      \
        x_offset_ptrs,                                                         \
        x_offset_sizes,                                                        \
        y_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),    \
        y_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),    \
        output_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
        jagged_dims_tensor,                                                    \
        [f] __device__(scalar_t x, scalar_t y, scalar_t /*unused*/)            \
            -> scalar_t { return f(x, y); });                                  \
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
#ifndef __HIP_PLATFORM_HCC__
            C10_CUDA_CHECK(cudaDeviceGetAttribute(
                &max_shared_bytes,
                cudaDevAttrMaxSharedMemoryPerBlockOptin,
                y_reshaped.get_device()));
#else
            // MI100 has 64 KB local memory (shared memory) per workgroup
            max_shared_bytes = 64 << 10;
#endif
            int shared_kb = max_shared_bytes >> 10;
#ifndef __HIP_PLATFORM_HCC__
            // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
            int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
            TORCH_CHECK(used_shared_kb > 0);
#else
            // MI100 has independent shared mem and L1
            int used_shared_kb = shared_kb;
#endif
            int used_shared_bytes = used_shared_kb << 10;
#ifndef __HIP_PLATFORM_HCC__
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
                    index_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes)); // V100: 64 KB; A100: 96 KB.
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            TORCH_CHECK(dynamic_smem_size <= used_shared_bytes);
          }
          dim3 threads_bs = dim3(1024, 1, 1);
          dim3 blocks_bs = dim3(div_round_up(nnz, threads_bs.x), 1, 1);
          jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
              index_t>
              <<<blocks_bs,
                 threads_bs,
                 dynamic_smem_size,
                 at::cuda::getCurrentCUDAStream()>>>(
                  x_offsets[0]
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  t_rows_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  t_cols_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  nnz,
                  B);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          // Gather kernel
          dim3 threads = dim3(16, 16, 1);
          dim3 blocks = dim3(1, div_round_up(nnz, threads.y), 1);
          if (blocks.y > 65535) {
            blocks.y = 65535;
          }
          jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_<
              index_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  output_values
                      .packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
                  x_values
                      .packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
                  y_reshaped
                      .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
                  y_reshaped
                      .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
                  t_rows_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  t_cols_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
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

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                 \
  {                                                                            \
    dim3 threads, blocks;                                                      \
    StackArray<int64_t> jagged_dims_tensor;                                    \
    std::tie(threads, blocks, jagged_dims_tensor) =                            \
        check_shape_and_partition_(x_values, x_offsets, y_0);                  \
    blocks.x = div_round_up(x_values.size(0), threads.y);                      \
    std::vector<Tensor> x_offsets_contig;                                      \
    x_offsets_contig.resize(num_jagged_dim);                                   \
    StackArray<index_t*> x_offset_ptrs;                                        \
    x_offset_ptrs.ndim = num_jagged_dim;                                       \
    StackArray<int64_t> x_offset_sizes;                                        \
    x_offset_sizes.ndim = num_jagged_dim;                                      \
    for (int d = 0; d < num_jagged_dim; ++d) {                                 \
      x_offsets_contig[d] = x_offsets[d].contiguous();                         \
      x_offset_ptrs.vals[d] =                                                  \
          x_offsets_contig[d].template data_ptr<index_t>();                    \
      x_offset_sizes.vals[d] = x_offsets[d].numel();                           \
    }                                                                          \
    jagged_dense_dense_elementwise_jagged_output_kernel_<                      \
        NUM_JAGGED_DIM,                                                        \
        index_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(    \
        x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),      \
        x_offset_ptrs,                                                         \
        x_offset_sizes,                                                        \
        y_0_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),  \
        y_1_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),  \
        output_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
        jagged_dims_tensor,                                                    \
        f);                                                                    \
  }

template <typename scalar_t, typename F>
void jagged_dense_dense_elementwise_jagged_output_opt_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0,
    const Tensor& y_1,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y_0.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (y_0.numel() == 0 || x_values.numel() == 0) {
    return;
  }

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_0_reshaped = y_0.view({y_0.size(0), -1, y_0.size(-1)});
  const Tensor y_1_reshaped = y_1.view({y_1.size(0), -1, y_1.size(-1)});

  if (jagged_dense_dense_elementwise_jagged_output_matches_opt(
          num_jagged_dim,
          x_values,
          x_offsets,
          y_0_reshaped,
          y_1_reshaped,
          output_values)) {
    AT_DISPATCH_INDEX_TYPES(
        x_offsets[0].scalar_type(),
        "jagged_dense_dense_indices_fast_path",
        [=] {
          auto nnz = output_values.size(0);
          auto B = y_0_reshaped.size(0);
          auto E = y_0_reshaped.size(2);
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
#ifndef __HIP_PLATFORM_HCC__
            C10_CUDA_CHECK(cudaDeviceGetAttribute(
                &max_shared_bytes,
                cudaDevAttrMaxSharedMemoryPerBlockOptin,
                y_0_reshaped.get_device()));
#else
            // MI100 has 64 KB local memory (shared memory) per workgroup
            max_shared_bytes = 64 << 10;
#endif
            int shared_kb = max_shared_bytes >> 10;
#ifndef __HIP_PLATFORM_HCC__
            // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
            int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
            TORCH_CHECK(used_shared_kb > 0);
#else
            // MI100 has independent shared mem and L1
            int used_shared_kb = shared_kb;
#endif
            int used_shared_bytes = used_shared_kb << 10;
#ifndef __HIP_PLATFORM_HCC__
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
                    index_t>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes)); // V100: 64 KB; A100: 96 KB.
#endif
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            TORCH_CHECK(dynamic_smem_size <= used_shared_bytes);
          }
          dim3 threads_bs = dim3(1024, 1, 1);
          dim3 blocks_bs = dim3(div_round_up(nnz, threads_bs.x), 1, 1);
          jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
              index_t>
              <<<blocks_bs,
                 threads_bs,
                 dynamic_smem_size,
                 at::cuda::getCurrentCUDAStream()>>>(
                  x_offsets[0]
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  t_rows_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  t_cols_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  nnz,
                  B);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          // Gather kernel
          dim3 threads = dim3(16, 16, 1);
          dim3 blocks = dim3(1, div_round_up(nnz, threads.y), 1);
          if (blocks.y > 65535) {
            blocks.y = 65535;
          }
          jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_<
              index_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  output_values
                      .packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
                  x_values
                      .packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
                  y_0_reshaped
                      .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
                  y_1_reshaped
                      .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
                  t_rows_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  t_cols_after_bs
                      .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                  nnz,
                  E,
                  [f] __device__(__half x, __half y0, __half y1) -> __half {
                    return f(x, y0, y1);
                  });
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }); // AT_DISPATCH
  } else {
    JAGGED_TENSOR_DISPATCH_DIMS();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename scalar_t, typename F>
void jagged_dense_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0,
    const Tensor& y_1,
    const Tensor& output_values,
    F f) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = y_0.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (y_0.numel() == 0 || x_values.numel() == 0) {
    return;
  }

  // Canonicalize y to 3D, collapsing jagged dimensions.
  const Tensor y_0_reshaped = y_0.view({y_0.size(0), -1, y_0.size(-1)});
  const Tensor y_1_reshaped = y_1.view({y_1.size(0), -1, y_1.size(-1)});

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef INVOKE_KERNEL_WITH_DIM

///@ingroup jagged-tensor-ops-cuda
at::Tensor jagged_to_padded_dense_forward(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const std::vector<int64_t>& max_lengths,
    const double padding_value) {
  const size_t num_jagged_dim = offsets.size();
  TORCH_CHECK(
      max_lengths.size() == num_jagged_dim,
      "max_lengths.size(), ",
      max_lengths.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const Tensor values_canonicalized = values.view(
      {values.size(0),
       std::accumulate(
           values.sizes().begin() + 1,
           values.sizes().end(),
           1,
           std::multiplies<size_t>())});
  at::DimVector padded_values_shape({offsets[0].size(0) - 1});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = values.dim() == 1;
  if (!D_folded) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values = at::empty(padded_values_shape, values.options());
  Tensor padded_values_view =
      D_folded ? padded_values.unsqueeze(-1) : padded_values;

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "jagged_to_padded_dense",
      [&] {
        jagged_dense_elementwise_dense_output_<scalar_t>(
            values_canonicalized,
            offsets,
            padded_values_view, // dummy not used in the lambda function
            padded_values_view,
            [] __device__(scalar_t x, scalar_t /*unused*/) -> scalar_t {
              return x;
            },
            static_cast<scalar_t>(padding_value));
      });

  return padded_values;
}

at::Tensor jagged_to_padded_dense_backward(
    const Tensor& grad_output,
    const std::vector<Tensor>& offsets,
    const int64_t total_L) {
  auto grad_padded_values = grad_output;
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_padded_values.get_device());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = grad_padded_values.dim() == offsets.size() + 1;
  Tensor grad_padded_values_view =
      D_folded ? grad_padded_values.unsqueeze(-1) : grad_padded_values;
  int32_t D = grad_padded_values_view.size(-1);

  // Initialize with zeros so output will be zero for the portion truncated
  // in forward.
  auto grad_values = at::zeros({total_L, D}, grad_padded_values.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_padded_values.scalar_type(),
      "jagged_to_dense_backward_kernel",
      [&] {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            grad_values, // dummy not used in the lambda function
            {offsets},
            grad_padded_values_view,
            grad_values,
            [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
              return y;
            });
      });

  return D_folded ? grad_values.squeeze(-1) : grad_values;
}

Tensor dense_to_jagged_forward(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    const c10::optional<int64_t>& total_L) {
  // D is the embedding dimension
  auto D = dense.size(-1);

  // If total_L is not given then compute it
  int64_t total_L_computed;
  if (total_L.has_value()) {
    total_L_computed = total_L.value();
  } else {
    total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
  }
  auto values = at::empty({total_L_computed, D}, dense.options());
  auto output = at::empty_like(values);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(dense.get_device());

#define DISPATCH_DENSE_TO_JAGGED_CASE(TYPE)                          \
  AT_DISPATCH_CASE(TYPE, [&] {                                       \
    jagged_dense_elementwise_jagged_output_opt_<scalar_t>(           \
        values,                                                      \
        offsets,                                                     \
        dense,                                                       \
        output,                                                      \
        [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t { \
          return y;                                                  \
        });                                                          \
  })

  // clang-format off
  AT_DISPATCH_SWITCH(
      values.scalar_type(),
      "dense_to_jagged_gpu_op_forward",
      DISPATCH_DENSE_TO_JAGGED_CASE(at::ScalarType::Half)
      DISPATCH_DENSE_TO_JAGGED_CASE(at::ScalarType::Int)
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(
          at::ScalarType::Long,
          at::ScalarType::BFloat16,
          [&] {
            jagged_dense_elementwise_jagged_output_<scalar_t>(
                values,
                offsets,
                dense,
                output,
                [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
                  return y;
                }); // device lambda
          } // lambda
          ) // CASE_FLOATING_TYPES_AND
  ); // SWITCH
  // clang-format on

#undef DISPATCH_DENSE_TO_JAGGED_CASE

  return output;
}

Tensor jagged_dense_dense_elementwise_add_jagged_output_forward(
    const Tensor& x_values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense_0,
    const Tensor& dense_1) {
  TORCH_CHECK(dense_0.sizes() == dense_1.sizes());
  auto output = at::empty_like(x_values);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(dense_0.get_device());

  if (x_values.scalar_type() == at::ScalarType::BFloat16 &&
      dense_0.scalar_type() == at::ScalarType::BFloat16 &&
      dense_1.scalar_type() == at::ScalarType::Float) {
    AT_DISPATCH_SWITCH(
        x_values.scalar_type(),
        "jagged_dense_dense_elementwise_jagged_output_forward",
        AT_DISPATCH_CASE_FLOATING_TYPES_AND(
            at::ScalarType::BFloat16,
            [&] {
              jagged_dense_dense_elementwise_jagged_output_<scalar_t>(
                  x_values,
                  offsets,
                  dense_0,
                  dense_1.to(at::ScalarType::BFloat16),
                  output,
                  [] __device__(scalar_t x, scalar_t y_0, scalar_t y_1)
                      -> scalar_t { return x + y_0 + y_1; });
            } // lambda
            ) // AT_DISPATCH_CASE_FLOATING_TYPES_AND
    ); // SWITCH
  } else {
    AT_DISPATCH_SWITCH(
        x_values.scalar_type(),
        "jagged_dense_dense_elementwise_jagged_output_forward",
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&] {
              jagged_dense_dense_elementwise_jagged_output_opt_<scalar_t>(
                  x_values,
                  offsets,
                  dense_0,
                  dense_1,
                  output,
                  [] __device__(scalar_t x, scalar_t y_0, scalar_t y_1)
                      -> scalar_t { return x + y_0 + y_1; });
            } // lambda
            ) // CASE
        AT_DISPATCH_CASE_FLOATING_TYPES_AND(
            at::ScalarType::BFloat16,
            [&] {
              jagged_dense_dense_elementwise_jagged_output_<scalar_t>(
                  x_values,
                  offsets,
                  dense_0,
                  dense_1,
                  output,
                  [] __device__(scalar_t x, scalar_t y_0, scalar_t y_1)
                      -> scalar_t { return x + y_0 + y_1; });
            } // lambda
            ) // CASE_FLOATING_TYPES_AND
    ); // SWITCH
  }

  return output;
}

class JaggedDenseAddJaggedOutputGPUOp
    : public torch::autograd::Function<JaggedDenseAddJaggedOutputGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& x_values,
      const std::vector<Tensor>& offsets,
      const Tensor& dense) {
    ctx->save_for_backward(offsets);
    ctx->saved_data["dense_shape"] = dense.sizes();

    auto output = at::empty_like(x_values);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dense.get_device());

    AT_DISPATCH_SWITCH(
        x_values.scalar_type(),
        "jagged_dense_elementwise_jagged_output_forward",
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&] {
              jagged_dense_elementwise_jagged_output_opt_<scalar_t>(
                  x_values,
                  offsets,
                  dense,
                  output,
                  [] __device__(scalar_t x, scalar_t y) -> scalar_t {
                    return x + y;
                  }); // device lambda
            } // lambda
            ) // CASE
        AT_DISPATCH_CASE_FLOATING_TYPES_AND(
            at::ScalarType::BFloat16,
            [&] {
              jagged_dense_elementwise_jagged_output_<scalar_t>(
                  x_values,
                  offsets,
                  dense,
                  output,
                  [] __device__(scalar_t x, scalar_t y) -> scalar_t {
                    return x + y;
                  }); // device lambda
            } // lambda
            ) // CASE_FLOATING_TYPES_AND
    ); // SWITCH

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    auto dense_shape = ctx->saved_data["dense_shape"].toIntVector();
    TORCH_CHECK(grad_outputs.size() == 1);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(grad_outputs[0].get_device());

    Tensor dense_values_grad = jagged_to_padded_dense_forward(
        grad_outputs[0],
        offsets,
        std::vector<int64_t>(dense_shape.begin() + 1, dense_shape.end() - 1),
        /*padding_value=*/0);
    TORCH_CHECK(dense_values_grad.sizes() == dense_shape);

    return {
        grad_outputs[0],
        torch::autograd::Variable(), // offsets
        dense_values_grad};
  }
};

///@ingroup jagged-tensor-ops-cuda
/// output = x + y where x is jagged, y is dense, and output is jagged
std::tuple<Tensor, std::vector<Tensor>>
jagged_dense_elementwise_add_jagged_output(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  auto sum_values =
      JaggedDenseAddJaggedOutputGPUOp::apply(x_values, x_offsets, y)[0];

  return {sum_values, x_offsets};
}

/**
 * output = f(x, y) where x and y are jagged (and share x_offsets), and output
 * is dense.
 *
 * @param padding_value padding_value for the output, not for inputs
 */
template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
__global__
__launch_bounds__(kMaxThreads) void jagged_jagged_elementwise_dense_output_kernel_(
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        x_values,
    StackArray<index_t*> x_offsets,
    const at::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits>
        y_values,
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> output,
    StackArray<int64_t> jagged_dims,
    F f,
    const scalar_t padding_value) {
  const int outer_dense_size = output.size(0);
  const int jagged_folded_size = output.size(1);
  const int inner_dense_size = output.size(2);

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
      for (int iidx = threadIdx.x; iidx < inner_dense_size;
           iidx += blockDim.x) {
        output[oidx][jidx][iidx] = padding_value;
      }
    } else {
      for (int iidx = threadIdx.x; iidx < inner_dense_size;
           iidx += blockDim.x) {
        output[oidx][jidx][iidx] =
            f(x_values[offset][iidx], y_values[offset][iidx]);
      }
    }
  }
}

template <typename scalar_t, typename F>
void jagged_jagged_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_values,
    const Tensor& output,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
  TENSOR_ON_CUDA_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_CUDA_GPU(x_offset);
  }

  const int num_jagged_dim = output.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  if (output.numel() == 0) {
    return;
  }

  dim3 threads, blocks;
  StackArray<int64_t> jagged_dims_tensor;
  std::tie(threads, blocks, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, output);

  // Canonicalize output to 3D, collapsing jagged dimensions.
  Tensor output_reshaped = output.view({output.size(0), -1, output.size(-1)});

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                \
  {                                                                           \
    std::vector<Tensor> x_offsets_contig;                                     \
    x_offsets_contig.resize(num_jagged_dim);                                  \
    StackArray<index_t*> x_offset_ptrs;                                       \
    x_offset_ptrs.ndim = num_jagged_dim;                                      \
    for (int d = 0; d < num_jagged_dim; ++d) {                                \
      x_offsets_contig[d] = x_offsets[d].contiguous();                        \
      x_offset_ptrs.vals[d] =                                                 \
          x_offsets_contig[d].template data_ptr<index_t>();                   \
    }                                                                         \
    jagged_jagged_elementwise_dense_output_kernel_<NUM_JAGGED_DIM, index_t>   \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(           \
            x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
            x_offset_ptrs,                                                    \
            y_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
            output_reshaped                                                   \
                .packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),     \
            jagged_dims_tensor,                                               \
            f,                                                                \
            padding_value);                                                   \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#undef INVOKE_KERNEL_WITH_DIM
}

Tensor jagged_dense_elementwise_mul_forward(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(x_values.get_device());

  Tensor output = at::empty_like(x_values);

  AT_DISPATCH_SWITCH(
      x_values.scalar_type(),
      "jagged_dense_elementwise_mul_jagged_output_forward",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&] {
            jagged_dense_elementwise_jagged_output_opt_<scalar_t>(
                x_values,
                x_offsets,
                y,
                output,
                [] __device__(scalar_t x, scalar_t y) -> scalar_t {
                  return x * y;
                });
          } // lambda
          ) // CASE
      AT_DISPATCH_CASE_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          [&] {
            jagged_dense_elementwise_jagged_output_<scalar_t>(
                x_values,
                x_offsets,
                y,
                output,
                [] __device__(scalar_t x, scalar_t y) -> scalar_t {
                  return x * y;
                });
          } // lambda
          ) // CASE_FLOATING_TYPES_AND

  ); // SWITCH

  return output;
}

std::tuple<Tensor, Tensor> jagged_dense_elementwise_mul_backward(
    const Tensor& grad_output,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& x_values) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  Tensor x_values_grad = at::empty_like(grad_output);
  Tensor y_grad = at::empty_like(y);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x_values.scalar_type(),
      "jagged_scalars",
      [&] {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            grad_output,
            x_offsets,
            y,
            x_values_grad,
            [] __device__(scalar_t x, scalar_t y) -> scalar_t {
              return x * y;
            });

        jagged_jagged_elementwise_dense_output_<scalar_t>(
            grad_output,
            x_offsets,
            x_values,
            y_grad,
            [] __device__(scalar_t x, scalar_t y) -> scalar_t {
              return x * y;
            });
      });

  return {x_values_grad, y_grad};
}

template <typename index_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void dense_vec_jagged_2d_bmm(
    const at::PackedTensorAccessor32<scalar_t, 2> v,
    const at::PackedTensorAccessor32<scalar_t, 2> a_values,
    const at::PackedTensorAccessor32<index_t, 1> a_offsets,
    at::PackedTensorAccessor32<scalar_t, 2> output) {
  const int B = a_offsets.size(0) - 1;
  const int H = v.size(0) / B;
  const int max_L = v.size(1);
  const int D = output.size(1);

  const int b_h_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int b_h_step = gridDim.x * blockDim.y;
  for (int b_h = b_h_begin; b_h < B * H; b_h += b_h_step) {
    const int b = b_h / H;
    const int h = b_h % H;

    const int row_start = a_offsets[b];
    const int row_end = a_offsets[b + 1];
    const int length = std::min(row_end - row_start, max_L);
    if (length == 0) {
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        output[b_h][d] = 0;
      }
    } else {
      // TODO: use shared memory
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        at::acc_type<scalar_t, true> acc =
            v[b_h][0] * a_values[row_start][h * D + d];
        for (int l = 1; l < length; ++l) {
          acc += v[b_h][l] * a_values[row_start + l][h * D + d];
        }
        output[b_h][d] = acc;
      }
    }
  }
}

template <typename index_t, typename scalar_t>
__global__
__launch_bounds__(kMaxThreads) void dense_vec_jagged_2d_transposed_bmm(
    const at::PackedTensorAccessor32<scalar_t, 2> v,
    const at::PackedTensorAccessor32<scalar_t, 2> a_values,
    const at::PackedTensorAccessor32<index_t, 1> a_offsets,
    at::PackedTensorAccessor32<scalar_t, 2> output) {
  const int B = a_offsets.size(0) - 1;
  const int H = v.size(0) / B;
  const int max_L = output.size(1);
  const int D = v.size(1);

  const int b_h_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int b_h_step = gridDim.x * blockDim.y;
  for (int b_h = b_h_begin; b_h < B * H; b_h += b_h_step) {
    const int b = b_h / H;
    const int h = b_h % H;

    const int row_start = a_offsets[b];
    const int row_end = a_offsets[b + 1];
    const int length = std::min(row_end - row_start, max_L);
    if (D == 0) {
      for (int l = threadIdx.x; l < max_L; ++l) {
        output[b_h][l] = 0;
      }
    } else {
      int l;
      for (l = threadIdx.x; l < length; l += blockDim.x) {
        at::acc_type<scalar_t, true> acc =
            v[b_h][0] * a_values[row_start + l][h * D];
        for (int d = 1; d < D; ++d) {
          acc += v[b_h][d] * a_values[row_start + l][h * D + d];
        }
        output[b_h][l] = acc;
      }
      for (; l < max_L; l += blockDim.x) {
        output[b_h][l] = 0;
      }
    }
  }
}

template <typename index_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void outer_prod_jagged_2d_output(
    const at::PackedTensorAccessor32<scalar_t, 2> x,
    const at::PackedTensorAccessor32<scalar_t, 2> y,
    const at::PackedTensorAccessor32<index_t, 1> offsets,
    at::PackedTensorAccessor32<scalar_t, 2> output_values) {
  const int B = offsets.size(0) - 1;
  const int H = x.size(0) / B;
  const int max_L = x.size(1);
  const int D = y.size(1);

  const int b_h_l_begin = blockIdx.x * blockDim.y + threadIdx.y;
  const int b_h_l_step = gridDim.x * blockDim.y;
  for (int b_h_l = b_h_l_begin; b_h_l < B * H * max_L; b_h_l += b_h_l_step) {
    const int b_h = b_h_l / max_L;
    const int b = b_h / H;
    const int h = b_h % H;
    const int l = b_h_l % max_L;

    const int row_start = offsets[b];
    const int row_end = offsets[b + 1];
    const int length = row_end - row_start;
    if (l < length) {
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        output_values[row_start + l][h * D + d] = x[b_h][l] * y[b_h][d];
      }
    }
  }
}

Tensor batched_dense_vec_jagged_2d_mul_forward(
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  TENSOR_ON_CUDA_GPU(v);
  TENSOR_ON_CUDA_GPU(a_values);
  TENSOR_ON_CUDA_GPU(a_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(v.get_device());

  const int B = a_offsets.numel() - 1;
  TORCH_CHECK(
      B == 0 || v.size(0) % B == 0,
      "B, ",
      B,
      " doesn't divide v.size(0), ",
      v.size(0));
  const int H = (B == 0) ? 1 : v.size(0) / B;
  const int D = a_values.size(-1) / H;
  auto output = at::empty({B * H, D}, v.options());

  if (B > 0 && D > 0) {
    const int block_dim_x =
        std::min(div_round_up(D, kWarpSize) * kWarpSize, kMaxThreads);
    const int block_dim_y = kMaxThreads / block_dim_x;

    AT_DISPATCH_INDEX_TYPES(
        a_offsets.scalar_type(), "dense_vec_jagged_2d_bmm_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              a_values.scalar_type(),
              "dense_vec_jagged_2d_bmm_kernel_2",
              [&] {
                dense_vec_jagged_2d_bmm<index_t, scalar_t>
                    <<<div_round_up(B * H, block_dim_y),
                       dim3(block_dim_x, block_dim_y),
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        v.packed_accessor32<scalar_t, 2>(),
                        a_values.packed_accessor32<scalar_t, 2>(),
                        a_offsets.packed_accessor32<index_t, 1>(),
                        output.packed_accessor32<scalar_t, 2>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }

  return output;
}

std::tuple<Tensor, Tensor> batched_dense_vec_jagged_2d_mul_backward(
    const Tensor& grad_output,
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  TENSOR_ON_CUDA_GPU(grad_output);
  TENSOR_ON_CUDA_GPU(a_values);
  TENSOR_ON_CUDA_GPU(a_offsets);
  TENSOR_ON_CUDA_GPU(v);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const int B = a_offsets.numel() - 1;
  const int D = grad_output.size(-1);

  Tensor a_values_grad = at::zeros_like(a_values);
  Tensor v_grad = at::empty_like(v);

  if (B > 0 && D > 0) {
    TORCH_CHECK(
        v.size(0) % B == 0, "B, ", B, " doesn't divide v.size(0), ", v.size(0));
    const int H = v.size(0) / B;
    const int max_L = v.size(-1);

    AT_DISPATCH_INDEX_TYPES(
        a_offsets.scalar_type(),
        "dense_vec_jagged_2d_bmm_backward_kernel_1",
        [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              grad_output.scalar_type(),
              "dense_vec_jagged_2d_bmm_backward_kernel_2",
              [&] {
                int block_dim_x = std::min(
                    div_round_up(max_L, kWarpSize) * kWarpSize, kMaxThreads);
                int block_dim_y = kMaxThreads / block_dim_x;

                dense_vec_jagged_2d_transposed_bmm<index_t, scalar_t>
                    <<<div_round_up(B * H, block_dim_y),
                       dim3(block_dim_x, block_dim_y),
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        grad_output.packed_accessor32<scalar_t, 2>(),
                        a_values.packed_accessor32<scalar_t, 2>(),
                        a_offsets.packed_accessor32<index_t, 1>(),
                        v_grad.packed_accessor32<scalar_t, 2>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();

                block_dim_x = std::min(
                    div_round_up(D, kWarpSize) * kWarpSize, kMaxThreads);
                block_dim_y = kMaxThreads / block_dim_x;

                outer_prod_jagged_2d_output<index_t, scalar_t>
                    <<<div_round_up(B * H * max_L, block_dim_y),
                       dim3(block_dim_x, block_dim_y),
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        v.packed_accessor32<scalar_t, 2>(),
                        grad_output.packed_accessor32<scalar_t, 2>(),
                        a_offsets.packed_accessor32<index_t, 1>(),
                        a_values_grad.packed_accessor32<scalar_t, 2>());
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  } else {
    v_grad.zero_();
  }

  return {v_grad, a_values_grad};
}

template <const int THREADS_PER_BLOCK, typename index_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void jagged_softmax_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2> values,
    const at::PackedTensorAccessor32<index_t, 1> offsets,
    at::PackedTensorAccessor32<scalar_t, 2> output,
    const int max_L) {
  const auto B = offsets.size(0) - 1;
  const auto D = output.size(1);

  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<scalar_t, THREADS_PER_BLOCK> BlockReduceT;

  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  __shared__ scalar_t max_value;
  __shared__ scalar_t exp_sum;

  const auto tid = threadIdx.x;
  for (uint32_t b = blockIdx.y; b < B; b += gridDim.y) {
    const index_t row_start = offsets[b];
    const index_t row_end = offsets[b + 1];
    const auto length = min(row_end - row_start, (index_t)max_L);

    if (length > 0) {
      const auto num_l_blocks =
          (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

      for (uint32_t d = blockIdx.x; d < D; d += gridDim.x) {
        if (tid == 0) {
          max_value = values[row_start][d];
          exp_sum = 0;
        }

        // Loop through all blocks to calculate the max value
        // Each block has its own max value block_max_value, and
        // max_value is the max value across all blocks
        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          const auto l = bk_l * blockDim.x + tid;
          scalar_t thread_val = values[row_start][d];
          if (l < length) {
            thread_val = values[row_start + l][d];
          }

          // Collectively compute the block-wide max reduction
          scalar_t block_max_value =
              BlockReduceT(temp_storage).Reduce(thread_val, cub::Max());
          __syncthreads();

          if (tid == 0) {
            max_value = max(max_value, block_max_value);
          }
        }

        // The max_value was updated by thread 0 in the last loop, sync here to
        // make sure the next loop uses the updated max_value
        __syncthreads();

        // Loop through all blocks to calculate the sum of exp
        // Each block has its own sum block_exp_acc, and
        // exp_sum is the sum across all blocks
        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          auto l = bk_l * blockDim.x + tid;

          scalar_t thread_exp = 0;
          if (l < length) {
            thread_exp = std::exp(values[row_start + l][d] - max_value);
          }

          // Collectively compute the block-wide sum reduction
          scalar_t block_exp_sum = BlockReduceT(temp_storage).Sum(thread_exp);
          __syncthreads();

          if (tid == 0) {
            exp_sum += block_exp_sum;
          }
        }

        // The exp_sum was updated by thread 0 in the last loop, sync here to
        // make sure the next loop uses the updated exp_sum
        __syncthreads();

        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          auto l = bk_l * blockDim.x + tid;
          scalar_t thread_exp = 0;
          if (l < length) {
            thread_exp = std::exp(values[row_start + l][d] - max_value);
            output[row_start + l][d] = thread_exp / exp_sum;
          }
        }

        // The max_value and exp_sum will be reinitialized by thread 0 in the
        // next d iteration, sync here to make sure the last loop still uses the
        // reduced values before reinitialization
        __syncthreads();
      }
    }
  }
}

Tensor jagged_softmax_forward(
    const Tensor& values,
    const Tensor& offsets,
    const int64_t max_L) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const auto B = offsets.numel() - 1;
  const auto D = values.size(1);
  auto output = at::empty_like(values);

  if (B > 0 && D > 0) {
    constexpr int THREADS_PER_BLOCK = 128;
    const dim3 grid(D, std::min((int32_t)B, (int32_t)kMaxBlockYDim), 1);

    AT_DISPATCH_INDEX_TYPES(
        offsets.scalar_type(), "jagged_softmax_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              values.scalar_type(),
              "jagged_softmax_kernel_2",
              [&] {
                jagged_softmax_kernel<THREADS_PER_BLOCK, index_t, scalar_t>
                    <<<grid,
                       THREADS_PER_BLOCK,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        values.packed_accessor32<scalar_t, 2>(),
                        offsets.packed_accessor32<index_t, 1>(),
                        output.packed_accessor32<scalar_t, 2>(),
                        (int)max_L);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }

  return output;
}

template <const int THREADS_PER_BLOCK, typename index_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void jagged_softmax_backward_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2> grad_output,
    const at::PackedTensorAccessor32<scalar_t, 2> output,
    const at::PackedTensorAccessor32<index_t, 1> offsets,
    at::PackedTensorAccessor32<scalar_t, 2> grad_input,
    const int max_L) {
  const auto B = offsets.size(0) - 1;
  const auto D = grad_output.size(1);

  // Specialize BlockReduce type for our thread block
  typedef cub::BlockReduce<scalar_t, THREADS_PER_BLOCK> BlockReduceT;

  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  __shared__ scalar_t sum_value;

  const auto tid = threadIdx.x;
  for (uint32_t b = blockIdx.y; b < B; b += gridDim.y) {
    const index_t row_start = offsets[b];
    const index_t row_end = offsets[b + 1];
    const auto length = min(row_end - row_start, (index_t)max_L);

    if (length > 0) {
      const auto num_l_blocks =
          (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

      for (uint32_t d = blockIdx.x; d < D; d += gridDim.x) {
        if (tid == 0) {
          sum_value = 0;
        }

        // Loop through all blocks to calculate the sum value
        // Each block has its own sum, and sum_value is the sum value across all
        // blocks
        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          const auto l = bk_l * blockDim.x + tid;
          scalar_t thread_val = 0;
          if (l < length) {
            thread_val =
                grad_output[row_start + l][d] * output[row_start + l][d];
          }

          // Collectively compute the block-wide sum reduction
          scalar_t block_sum_value = BlockReduceT(temp_storage).Sum(thread_val);
          __syncthreads();

          if (tid == 0) {
            sum_value += block_sum_value;
          }
        }

        // The sum_value was updated by thread 0 in the last loop, sync here to
        // make sure the next loop uses the updated sum_value
        __syncthreads();

        for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
          const auto l = bk_l * blockDim.x + tid;
          if (l < length) {
            grad_input[row_start + l][d] =
                (grad_output[row_start + l][d] - sum_value) *
                output[row_start + l][d];
          }
        }

        // The sum_value will be reinitialized by thread 0 in the
        // next d iteration, sync here to make sure the last loop still uses the
        // reduced value before reinitialization
        __syncthreads();
      }
    }
  }
}

Tensor jagged_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& offsets,
    const int64_t max_L) {
  TENSOR_ON_CUDA_GPU(grad_output);
  TENSOR_ON_CUDA_GPU(output);
  TENSOR_ON_CUDA_GPU(offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_output.get_device());

  const auto B = offsets.numel() - 1;
  const auto D = grad_output.size(1);
  auto grad_input = at::empty_like(grad_output);

  if (B > 0 && D > 0) {
    constexpr int THREADS_PER_BLOCK = 128;
    const dim3 grid(D, std::min((int32_t)B, (int32_t)kMaxBlockYDim), 1);

    AT_DISPATCH_INDEX_TYPES(
        offsets.scalar_type(), "jagged_softmax_backward_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              grad_output.scalar_type(),
              "jagged_softmax_backward_kernel_2",
              [&] {
                jagged_softmax_backward_kernel<
                    THREADS_PER_BLOCK,
                    index_t,
                    scalar_t>
                    <<<grid,
                       THREADS_PER_BLOCK,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        grad_output.packed_accessor32<scalar_t, 2>(),
                        output.packed_accessor32<scalar_t, 2>(),
                        offsets.packed_accessor32<index_t, 1>(),
                        grad_input.packed_accessor32<scalar_t, 2>(),
                        (int)max_L);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }
  return grad_input;
}

template <const int BLOCK_SIZE, typename index_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void jagged_jagged_bmm_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2> x_values,
    const at::PackedTensorAccessor32<scalar_t, 2> y_values,
    const at::PackedTensorAccessor32<index_t, 1> offsets,
    at::PackedTensorAccessor32<scalar_t, 3> output,
    const int max_L) {
  const int B = offsets.size(0) - 1;
  const int M = x_values.size(1);
  const int N = y_values.size(1);

  const auto block_row = blockIdx.y;
  const auto block_col = blockIdx.x;
  const auto row = threadIdx.y;
  const auto col = threadIdx.x;
  __shared__ scalar_t Xs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_t Ys[BLOCK_SIZE][BLOCK_SIZE];

  for (uint32_t b = blockIdx.z; b < B; b += gridDim.z) {
    const index_t row_start = offsets[b];
    const index_t row_end = offsets[b + 1];
    const auto length = min(row_end - row_start, (index_t)max_L);
    auto num_l_blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;

    at::acc_type<scalar_t, true> acc = 0;

    const auto row_offset = block_row * BLOCK_SIZE + row;
    const auto col_offset = block_col * BLOCK_SIZE + col;

    // for loop block tile in length dimension
    for (auto bk_l = 0; bk_l < num_l_blocks; bk_l++) {
      Xs[row][col] = 0;
      Ys[row][col] = 0;
      const auto bk_offset = bk_l * BLOCK_SIZE;

      // load data from global memory to shared memory
      const auto l_x = bk_offset + col;
      if (row_offset < M && l_x < length) {
        Xs[row][col] = x_values[row_start + l_x][row_offset];
      }

      const auto l_y = bk_offset + row;
      if (l_y < length && col_offset < N) {
        Ys[row][col] = y_values[row_start + l_y][col_offset];
      }

      __syncthreads();

#pragma unroll
      for (auto e = 0; e < BLOCK_SIZE; e++) {
        acc += Xs[row][e] * Ys[e][col];
      }
      __syncthreads();
    }

    // write the result to the output
    if ((row_offset < M) && (col_offset < N))
      output[b][row_offset][col_offset] = acc;
  }
}

Tensor jagged_jagged_bmm_forward(
    const Tensor& x_values,
    const Tensor& y_values,
    const Tensor& offsets,
    const int64_t max_L) {
  TENSOR_ON_CUDA_GPU(x_values);
  TENSOR_ON_CUDA_GPU(y_values);
  TENSOR_ON_CUDA_GPU(offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(x_values.get_device());

  const int B = offsets.numel() - 1;
  const int M = x_values.size(-1);
  const int N = y_values.size(-1);
  auto output = at::zeros({B, M, N}, x_values.options());

  if (B > 0 && M > 0 && N > 0) {
    constexpr int BLOCK_SIZE = 16;
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const auto grid_dim_x = div_round_up(N, BLOCK_SIZE);
    const auto grid_dim_y = div_round_up(M, BLOCK_SIZE);
    TORCH_CHECK(
        grid_dim_y <= kMaxBlockYDim,
        "M cannot be larger than",
        grid_dim_y * BLOCK_SIZE + 1 - BLOCK_SIZE);
    const auto grid_dim_z = std::min(B, kMaxBlockZDim);
    const dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);

    AT_DISPATCH_INDEX_TYPES(
        offsets.scalar_type(), "jagged_jagged_bmm_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              x_values.scalar_type(),
              "jagged_jagged_bmm_kernel_2",
              [&] {
                jagged_jagged_bmm_kernel<BLOCK_SIZE, index_t, scalar_t>
                    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                        x_values.packed_accessor32<scalar_t, 2>(),
                        y_values.packed_accessor32<scalar_t, 2>(),
                        offsets.packed_accessor32<index_t, 1>(),
                        output.packed_accessor32<scalar_t, 3>(),
                        (int)max_L);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }
  return output;
}

template <
    const int BLOCK_TILE_M, // tile height of C that each thread block
                            // calculates
    const int BLOCK_TILE_N, // tile width of C that each thread block
                            // calculates
    const int BLOCK_TILE_K, // tile width of A that each thread block calculates
    const int THREAD_TILE_M, // tile height of C that each thread
                             // calculates
    const int THREAD_TILE_N, // tile width of C that each thread calcualtes
    typename index_t,
    typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void jagged_dense_bmm_kernel(
    const at::PackedTensorAccessor32<scalar_t, 2> x_values,
    const at::PackedTensorAccessor32<index_t, 1> x_offsets,
    const at::PackedTensorAccessor32<scalar_t, 3> y,
    at::PackedTensorAccessor32<scalar_t, 2> output,
    const int max_L) {
  const int B = x_offsets.size(0) - 1;
  const int K = x_values.size(1);
  const int N = y.size(2);

  const auto block_row = blockIdx.y;
  const auto block_col = blockIdx.x;

  const int THREADS_X_PER_BLOCK = BLOCK_TILE_N / THREAD_TILE_N;
  const int THREADS_Y_PER_BLOCK = BLOCK_TILE_M / THREAD_TILE_M;
  const int THREADS_PER_BLOCK = THREADS_X_PER_BLOCK * THREADS_Y_PER_BLOCK;
  const auto thread_row = threadIdx.x / THREADS_X_PER_BLOCK;
  const auto thread_col = threadIdx.x % THREADS_X_PER_BLOCK;
  const auto NUM_K_BLOCKS = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

  __shared__ scalar_t As[BLOCK_TILE_M][BLOCK_TILE_K];
  __shared__ scalar_t Bs[BLOCK_TILE_K][BLOCK_TILE_N];

  // Once we remove ROCm<=5.3 support, we should replace uint32_t with auto.
  // See #1655
  for (uint32_t b = blockIdx.z; b < B; b += gridDim.z) {
    const index_t row_start = x_offsets[b];
    const index_t row_end = x_offsets[b + 1];
    const auto length = min(row_end - row_start, (index_t)max_L);

    // the indices that this current will load into shared mem
    const auto inner_row_a = threadIdx.x / BLOCK_TILE_K;
    const auto inner_col_a = threadIdx.x % BLOCK_TILE_K;
    // the number of rows of As that will be loaded per step by a thread block
    const auto A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / BLOCK_TILE_K;

    const auto inner_row_b = threadIdx.x / BLOCK_TILE_N;
    const auto inner_col_b = threadIdx.x % BLOCK_TILE_N;
    const auto B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / BLOCK_TILE_N;

    // registers for C
    scalar_t accum[THREAD_TILE_M][THREAD_TILE_N] = {0};

    // registers for As and Bs
    scalar_t fragment_a[THREAD_TILE_M] = {0};
    scalar_t fragment_b[THREAD_TILE_N] = {0};

    // loop for block tiles in K dimension
    for (auto block = 0; block < NUM_K_BLOCKS; block++) {
// load a block of x_values from global memory to shared memory
// apply tiling for threads in a block
#pragma unroll
      for (auto offset = 0; offset < BLOCK_TILE_M;
           offset += A_TILE_ROW_STRIDE) {
        auto x_row_offset = block_row * BLOCK_TILE_M + inner_row_a + offset;
        auto x_col_offset = block * BLOCK_TILE_K + inner_col_a;
        if ((x_row_offset < length) && (x_col_offset < K)) {
          As[inner_row_a + offset][inner_col_a] =
              x_values[row_start + x_row_offset][x_col_offset];
        } else {
          As[inner_row_a + offset][inner_col_a] = 0;
        }
      }

// load a block of y from global memory to shared memory
// apply tiling for threads in a block
#pragma unroll
      for (auto offset = 0; offset < BLOCK_TILE_K;
           offset += B_TILE_ROW_STRIDE) {
        auto y_row_offset = block * BLOCK_TILE_K + inner_row_b + offset;
        auto y_col_offset = block_col * BLOCK_TILE_N + inner_col_b;
        if ((y_row_offset < K) && (y_col_offset < N)) {
          Bs[inner_row_b + offset][inner_col_b] =
              y[b][y_row_offset][y_col_offset];
        } else {
          Bs[inner_row_b + offset][inner_col_b] = 0;
        }
      }

      __syncthreads();

// calculate the results per thread
#pragma unroll
      for (auto k = 0; k < BLOCK_TILE_K; k++) {
        // load values from shared memory to registers for x_values
        for (auto row = 0; row < THREAD_TILE_M; row++) {
          fragment_a[row] = As[thread_row * THREAD_TILE_M + row][k];
        }

// load values from shared memory to registers for y
#pragma unroll
        for (auto col = 0; col < THREAD_TILE_N; col++) {
          fragment_b[col] = Bs[k][thread_col * THREAD_TILE_N + col];
        }

// each thread calcualtes THREAD_TILE_M * THREAD_TILE_N elements
#pragma unroll
        for (auto row = 0; row < THREAD_TILE_M; row++) {
#pragma unroll
          for (auto col = 0; col < THREAD_TILE_N; col++) {
            accum[row][col] += fragment_a[row] * fragment_b[col];
          }
        }
      }

      __syncthreads();
    }

// write the result to the output
#pragma unroll
    for (auto row = 0; row < THREAD_TILE_M; row++) {
#pragma unroll
      for (auto col = 0; col < THREAD_TILE_N; col++) {
        auto out_row_offset =
            block_row * BLOCK_TILE_M + thread_row * THREAD_TILE_M + row;
        auto out_col_offset =
            block_col * BLOCK_TILE_N + thread_col * THREAD_TILE_N + col;
        if ((out_row_offset < length) && (out_col_offset < N)) {
          output[row_start + out_row_offset][out_col_offset] = accum[row][col];
        }
      }
    }
  }
}

Tensor jagged_dense_bmm_forward(
    const Tensor& x_values,
    const Tensor& x_offsets,
    const Tensor& y,
    const int64_t max_L) {
  TENSOR_ON_CUDA_GPU(x_values);
  TENSOR_ON_CUDA_GPU(x_offsets);
  TENSOR_ON_CUDA_GPU(y);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(x_values.get_device());

  const int B = x_offsets.numel() - 1;
  const int M = x_values.size(-1);
  const int N = y.size(-1);
  const int total_L = x_values.size(0);
  auto output = at::zeros({total_L, N}, x_values.options());
  if (B > 0 && M > 0 && N > 0) {
    // The shared memory size is (BLOCK_TILE_M + BLOCK_TILE_N) * BLOCK_TILE_K
    // BLOCK_TILE_M needs to be multiple of THREAD_TILE_M, and
    // BLOCK_TILE_N needs to be multiple of THREAD_TILE_N
    // The setting of these parameters needs to balance the hardware's shared
    // memory size limit and occupancy
    // TODO: autotune these parameters based on max_L and input and output
    // tensor sizes
    constexpr int BLOCK_TILE_M = 64;
    constexpr int BLOCK_TILE_N = 8;
    constexpr int BLOCK_TILE_K = 8;
    constexpr int THREAD_TILE_M = 4;
    constexpr int THREAD_TILE_N = 4;

    const dim3 block(
        (BLOCK_TILE_M * BLOCK_TILE_N) / (THREAD_TILE_M * THREAD_TILE_N));
    const auto grid_dim_x = div_round_up(N, BLOCK_TILE_N);
    const auto grid_dim_y = div_round_up(max_L, BLOCK_TILE_M);
    TORCH_CHECK(
        grid_dim_y <= kMaxBlockYDim,
        "max_L cannot be larger than",
        grid_dim_y * BLOCK_TILE_M + 1 - BLOCK_TILE_M);
    const auto grid_dim_z = std::min(B, kMaxBlockZDim);
    const dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);

    AT_DISPATCH_INDEX_TYPES(
        x_offsets.scalar_type(), "jagged_dense_bmm_kernel_1", [&] {
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::ScalarType::Half,
              at::ScalarType::BFloat16,
              x_values.scalar_type(),
              "jagged_dense_bmm_kernel_2",
              [&] {
                jagged_dense_bmm_kernel<
                    BLOCK_TILE_M,
                    BLOCK_TILE_N,
                    BLOCK_TILE_K,
                    THREAD_TILE_M,
                    THREAD_TILE_N,
                    index_t,
                    scalar_t>
                    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                        x_values.packed_accessor32<scalar_t, 2>(),
                        x_offsets.packed_accessor32<index_t, 1>(),
                        y.packed_accessor32<scalar_t, 3>(),
                        output.packed_accessor32<scalar_t, 2>(),
                        (int)max_L);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }

  return output;
}

} // namespace

Tensor jagged_2d_to_dense_gpu_forward(
    Tensor values,
    Tensor offsets,
    int64_t max_sequence_length) {
  return jagged_to_padded_dense_forward(
      values, {offsets}, {max_sequence_length}, /*padding_value=*/0);
}

Tensor jagged_2d_to_dense_gpu_backward(
    Tensor grad_output,
    at::Tensor offsets,
    int64_t max_lengths) {
  return jagged_to_padded_dense_backward(grad_output, {offsets}, max_lengths);
}

// stacked ops
std::tuple<std::vector<Tensor>, std::vector<Tensor>>
stacked_jagged_2d_to_dense_forward_cuda(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(lengths.dim() == 2);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const auto lengths_contig = lengths.contiguous();
  int32_t D = values.size(1);
  int32_t B = lengths.size(1);
  int32_t T = lengths.size(0);
  std::vector<Tensor> padded_values_per_key;
  std::vector<Tensor> offsets_tensor_per_key;
  for (int32_t t = 0; t < T; t++) {
    int64_t max_L = max_lengths_per_key[t];
    size_t temp_storage_bytes = 0;
    auto offsets = at::empty({B + 1}, lengths.options());
    offsets[0].zero_();
    AT_DISPATCH_INDEX_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              nullptr,
              temp_storage_bytes,
              &(lengths_contig.data_ptr<index_t>()[t * B]),
              offsets.data_ptr<index_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        });
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        lengths.options().dtype(at::kByte));
    AT_DISPATCH_INDEX_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper2", [&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              temp_storage.data_ptr(),
              temp_storage_bytes,
              &(lengths_contig.data_ptr<index_t>()[t * B]),
              offsets.data_ptr<index_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        });
    offsets_tensor_per_key.push_back(offsets);

    padded_values_per_key.push_back(jagged_to_padded_dense_forward(
        values.slice(0, offset_per_key[t], offset_per_key[t + 1]),
        {offsets},
        {max_L},
        padding_value));
  }

  return std::make_tuple(padded_values_per_key, offsets_tensor_per_key);
}

Tensor stacked_jagged_2d_to_dense_backward_cuda(
    int64_t B,
    int64_t D,
    int64_t total_L,
    const std::vector<Tensor>& grad_padded_values_per_key,
    const std::vector<Tensor>& offsets_tensor_per_key,
    const std::vector<int64_t>& offset_per_key) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_padded_values_per_key[0].get_device());

  auto grad_values =
      at::zeros({total_L, D}, grad_padded_values_per_key[0].options());
  int32_t T = grad_padded_values_per_key.size();
  for (int32_t t = 0; t < T; t++) {
    TORCH_CHECK(grad_padded_values_per_key[t].dim() == 3);
    TORCH_CHECK(grad_padded_values_per_key[t].size(0) == B);
    TORCH_CHECK(grad_padded_values_per_key[t].size(2) == D);

    Tensor grad_values_slice =
        grad_values.slice(0, offset_per_key[t], offset_per_key[t + 1]);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        grad_values.scalar_type(),
        "jagged_2d_to_dense_backward_kernel",
        [&] {
          jagged_dense_elementwise_jagged_output_<scalar_t>(
              grad_values_slice, // dummy not used in the lambda function
              {offsets_tensor_per_key[t]},
              grad_padded_values_per_key[t],
              grad_values_slice,
              [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
                return y;
              });
        });
  }

  return grad_values;
}

std::vector<Tensor> stacked_jagged_1d_to_dense_gpu(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(lengths.dim() == 2);
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  const auto lengths_contig = lengths.contiguous();
  int32_t B = lengths.size(1);
  int32_t T = lengths.size(0);
  auto offsets = at::empty({B + 1}, lengths.options());
  offsets[0].zero_();
  std::vector<Tensor> padded_values_per_key;
  for (int32_t t = 0; t < T; t++) {
    int64_t max_L = max_lengths_per_key[t];
    size_t temp_storage_bytes = 0;
    AT_DISPATCH_INDEX_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper1", [&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              nullptr,
              temp_storage_bytes,
              &(lengths_contig.data_ptr<index_t>()[t * B]),
              offsets.data_ptr<index_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        });
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        lengths.options().dtype(at::kByte));
    AT_DISPATCH_INDEX_TYPES(
        lengths_contig.scalar_type(), "cub_inclusive_sum_wrapper2", [&] {
          AT_CUDA_CHECK(FBGEMM_GPU_CUB_NS_PREFIX cub::DeviceScan::InclusiveSum(
              temp_storage.data_ptr(),
              temp_storage_bytes,
              &(lengths_contig.data_ptr<index_t>()[t * B]),
              offsets.data_ptr<index_t>() + 1,
              B,
              at::cuda::getCurrentCUDAStream()));
        });

    padded_values_per_key.push_back(jagged_to_padded_dense_forward(
        values.slice(0, offset_per_key[t], offset_per_key[t + 1]),
        {offsets},
        {max_L},
        padding_value));
  }
  return padded_values_per_key;
}

template <typename index_t, typename offset_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void jagged_index_select_2d_kernel(
    scalar_t* output,
    const scalar_t* input,
    const offset_t* input_offsets,
    const index_t* indices,
    const offset_t* output_offsets,
    const int64_t num_output_rows,
    const int64_t num_dense_output_rows,
    const int64_t num_cols) {
  __shared__ int smem[1];
  for (offset_t dense_output_offset = blockIdx.x;
       dense_output_offset < num_dense_output_rows;
       dense_output_offset += gridDim.x) {
    // Binary search
    // TODO: use multiple threads to do bin search to reduce number of steps
    if (threadIdx.x == 0) {
      binary_search_range(
          smem, output_offsets, dense_output_offset, num_output_rows);
    }
    __syncthreads();

    // All threads load index_pos from shared memory and return if the index_pos
    // is invalid
    int index_pos = smem[0];

    // TODO: Can also be obtained during the binary search
    // Relative index position
    const offset_t rel_index = dense_output_offset -
        (index_pos == 0 ? 0 : output_offsets[index_pos - 1]);
    const index_t index = indices[index_pos];
    const offset_t input_offset =
        (index == 0 ? 0 : input_offsets[index - 1]) + rel_index;

    // Shift buffers
    scalar_t* output_ = output + dense_output_offset * num_cols;
    const scalar_t* input_ = input + input_offset * num_cols;

    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
      output_[i] = input_[i];
    }
  }
}

/// Copy sequences from input jagged tensor based on indices specified in the
/// indices tensor to an output jagged tensor (host function for dispatching
/// jagged_index_select_2d_kernel to GPU)
/// @param values                2D dense value tensor of input jagged tensor
/// @param indices               1D tensor that contains indices to be selected
///                              from output jagged tensor
/// @param input_offsets         1D tensor that contains offsets of input
///                              jagged tensor
/// @param output_offsets        1D tensor that contains offsets of output
///                              jagged tensor
/// @param num_dense_output_rows The total number of rows in the 2D dense value
///                              tensor of output jagged tensor
Tensor jagged_index_select_2d_forward_cuda(
    const Tensor& values,
    const Tensor& indices,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    const int64_t num_dense_output_rows) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(input_offsets);
  TENSOR_ON_CUDA_GPU(output_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  auto num_cols = values.size(1);
  const int64_t num_output_rows = indices.numel();

  const int64_t max_num_blocks = 1024; // Arbitrarily set to this number of now
  const int64_t max_num_threads = kMaxThreads;
  const int64_t num_blocks = std::min(max_num_blocks, num_dense_output_rows);
  const int64_t num_threads = std::min(max_num_threads, num_cols);
  Tensor output =
      at::empty({num_dense_output_rows, num_cols}, values.options());

  if (num_blocks > 0) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        values.scalar_type(),
        "jagged_index_select_2d_kernel_wrapper_1",
        [&] {
          AT_DISPATCH_INDEX_TYPES(
              indices.scalar_type(),
              "jagged_index_select_2d_kernel_wrapper_2",
              [&] {
                jagged_index_select_2d_kernel<<<
                    dim3(num_blocks),
                    dim3(num_cols),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    output.data_ptr<scalar_t>(),
                    values.data_ptr<scalar_t>(),
                    input_offsets.data_ptr<int64_t>(),
                    indices.data_ptr<index_t>(),
                    output_offsets.data_ptr<int64_t>(),
                    num_output_rows,
                    num_dense_output_rows,
                    num_cols);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }

  return output;
}

template <typename index_t, typename offset_t, typename scalar_t>
__global__ __launch_bounds__(kMaxThreads) void jagged_index_add_2d_kernel(
    scalar_t* output,
    const scalar_t* values,
    const offset_t* input_offsets,
    const index_t* indices,
    const offset_t* output_offsets,
    const int64_t num_input_rows,
    const int64_t num_dense_input_rows,
    const int64_t num_cols) {
  __shared__ int smem[1];
  for (offset_t dense_input_offset = blockIdx.x;
       dense_input_offset < num_dense_input_rows;
       dense_input_offset += gridDim.x) {
    // Binary search
    // TODO: use multiple threads to do bin search to reduce number of steps
    if (threadIdx.x == 0) {
      binary_search_range(
          smem, input_offsets, dense_input_offset, num_input_rows);
    }
    __syncthreads();

    // All threads load index_pos from shared memory and return if the index_pos
    // is invalid
    int index_pos = smem[0];

    // TODO: Can also be obtained during the binary search
    // Relative index position
    const offset_t rel_index = dense_input_offset -
        (index_pos == 0 ? 0 : input_offsets[index_pos - 1]);
    const index_t index = indices[index_pos];
    const offset_t output_offset =
        (index == 0 ? 0 : output_offsets[index - 1]) + rel_index;

    // Shift buffers
    const scalar_t* values_ = values + dense_input_offset * num_cols;
    scalar_t* output_ = output + output_offset * num_cols;

    // TODO: Avoid using atoimcAdd (because it could lead to the numerical
    // indeterminism issue)
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
      gpuAtomicAdd(&output_[i], values_[i]);
    }
  }
}

/// Add sequences from input jagged tensor to output jagged tensor based on
/// indices specified in the indices tensor (host function for dispatching
/// jagged_index_add_2d_kernel to GPU)
/// @param values               2D dense value tensor of input jagged tensor
/// @param indices              1D tensor that contains indices to be added in
///                             output jagged tensor
/// @param input_offsets        1D tensor that contains offsets of input
///                             jagged tensor
/// @param output_offsets       1D tensor that contains offsets of output
///                             jagged tensor
/// @param num_dense_input_rows The total number of rows in the 2D dense value
///                             tensor of input jagged tensor
/// @param num_output_rows      The number of sequences in jagged output tensor
Tensor jagged_index_add_2d_forward_cuda(
    const Tensor& values,
    const Tensor& indices,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    const int64_t num_dense_input_rows,
    const int64_t num_output_rows) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(indices);
  TENSOR_ON_CUDA_GPU(input_offsets);
  TENSOR_ON_CUDA_GPU(output_offsets);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  auto num_cols = values.size(1);
  const int64_t num_input_rows = indices.numel();

  const int64_t max_num_blocks = 1024; // Arbitrarily set to this number of now
  const int64_t max_num_threads = kMaxThreads;
  const int64_t num_blocks = std::min(max_num_blocks, num_dense_input_rows);
  const int64_t num_threads = std::min(max_num_threads, num_cols);
  Tensor output = at::zeros({num_output_rows, num_cols}, values.options());

  if (num_blocks > 0) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        values.scalar_type(),
        "jagged_index_add_2d_kernel_wrapper_1",
        [&] {
          AT_DISPATCH_INDEX_TYPES(
              indices.scalar_type(),
              "jagged_index_add_2d_kernel_wrapper_2",
              [&] {
                jagged_index_add_2d_kernel<<<
                    dim3(num_blocks),
                    dim3(num_cols),
                    0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    output.data_ptr<scalar_t>(),
                    values.data_ptr<scalar_t>(),
                    input_offsets.data_ptr<int64_t>(),
                    indices.data_ptr<index_t>(),
                    output_offsets.data_ptr<int64_t>(),
                    num_input_rows,
                    num_dense_input_rows,
                    num_cols);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }

  return output;
}

class StackedJagged2DToDenseGPUOp
    : public torch::autograd::Function<StackedJagged2DToDenseGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      Tensor values,
      Tensor lengths,
      const std::vector<int64_t>& offset_per_key,
      const std::vector<int64_t>& max_lengths_per_key,
      int64_t padding_value) {
    int64_t total_L = values.size(0);
    ctx->saved_data["B"] = lengths.size(1);
    ctx->saved_data["D"] = values.size(1);
    ctx->saved_data["total_L"] = total_L;
    ctx->saved_data["offset_per_key"] = offset_per_key;

    auto [padded_values_per_key, offsets_tensor_per_key] =
        stacked_jagged_2d_to_dense_forward_cuda(
            values,
            lengths,
            offset_per_key,
            max_lengths_per_key,
            padding_value);
    ctx->saved_data["offsets_tensor_per_key"] = offsets_tensor_per_key;

    return padded_values_per_key;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto B = ctx->saved_data["B"].toInt();
    auto D = ctx->saved_data["D"].toInt();
    auto total_L = ctx->saved_data["total_L"].toInt();
    auto offset_per_key = ctx->saved_data["offset_per_key"].toIntVector();
    auto offsets_tensor_per_key =
        ctx->saved_data["offsets_tensor_per_key"].toTensorVector();

    using torch::autograd::Variable;
    auto grad_values = stacked_jagged_2d_to_dense_backward_cuda(
        B, D, total_L, grad_outputs, offsets_tensor_per_key, offset_per_key);
    return {
        grad_values,
        Variable(), // lengths
        Variable(), // offset_per_key
        Variable(), // max_lengths_per_key
        Variable(), // padding_value
    };
  }
};

std::vector<Tensor> stacked_jagged_2d_to_dense_gpu(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value) {
  TENSOR_ON_CUDA_GPU(values);
  TENSOR_ON_CUDA_GPU(lengths);
  TENSORS_ON_SAME_DEVICE(values, lengths);
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(lengths.dim() == 2);
  return StackedJagged2DToDenseGPUOp::apply(
      values, lengths, offset_per_key, max_lengths_per_key, padding_value);
}

template <
    typename scalar_t,
    typename index_t,
    typename offset_t,
    typename weight_t,
    bool has_weights>
__global__ void keyed_jagged_index_select_dim1_kernel(
    scalar_t* output,
    weight_t* output_weights,
    const scalar_t* input,
    const weight_t* weights,
    const offset_t* input_offsets,
    const index_t* indices,
    const offset_t* output_offsets,
    const int num_batches,
    const int input_batch_size,
    const int output_batch_size,
    const int64_t num_outputs) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_outputs) {
    // Each thread searches index position
    int index_pos;
    binary_search_range(
        &index_pos,
        output_offsets,
        (offset_t)tid,
        num_batches * output_batch_size);

    const offset_t rel_index =
        tid - (index_pos == 0 ? 0 : output_offsets[index_pos - 1]);

    // indices are the same for all batches
    const index_t index = indices[index_pos % output_batch_size];
    const int bid = index_pos / output_batch_size;
    const offset_t input_offset =
        (index == 0 && bid == 0
             ? 0
             : input_offsets[bid * input_batch_size + index - 1]) +
        rel_index;

    // Store data
    output[tid] = input[input_offset];
    if (has_weights) {
      output_weights[tid] = weights[input_offset];
    }
  }
}

template <typename scalar_t, typename index_t, typename offset_t>
__global__ void keyed_jagged_index_add_dim1_kernel(
    scalar_t* output,
    const scalar_t* input,
    const offset_t* input_offsets,
    const index_t* indices,
    const offset_t* output_offsets,
    const int num_batches,
    const int input_batch_size,
    const int output_batch_size,
    const int64_t num_inputs) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_inputs) {
    // Each thread searches index position
    int index_pos;
    binary_search_range(
        &index_pos,
        input_offsets,
        (offset_t)tid,
        num_batches * input_batch_size);

    const offset_t rel_index =
        tid - (index_pos == 0 ? 0 : input_offsets[index_pos - 1]);

    // indices are the same for all batches
    const index_t index = indices[index_pos % input_batch_size];
    const int bid = index_pos / input_batch_size;
    const offset_t output_offset =
        (index == 0 && bid == 0
             ? 0
             : output_offsets[bid * output_batch_size + index - 1]) +
        rel_index;

    // Store data
    gpuAtomicAdd(&output[output_offset], input[tid]);
  }
}

template <
    typename scalar_t,
    typename index_t,
    typename acc_t,
    int NUM_THREADS_PER_BLOCK,
    int MAX_ENTRIES_PER_BLOCK>
__global__ void index_select_scalar_cumsum_kernel(
    scalar_t* output,
    acc_t* output_cumsum,
    const scalar_t* __restrict__ input,
    const index_t* __restrict__ indices,
    const int num_batches,
    const int input_batch_size,
    const int output_batch_size,
    const int last_block_num_entries,
    int* block_flags,
    acc_t* block_sums) {
  typedef cub::BlockScan<acc_t, NUM_THREADS_PER_BLOCK> BlockScan;
  __shared__ typename BlockScan::TempStorage bs_temp_storage;
  __shared__ acc_t smem[MAX_ENTRIES_PER_BLOCK];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = tid / output_batch_size;
  const int num_entries_per_block = blockIdx.x == gridDim.x - 1
      ? last_block_num_entries
      : MAX_ENTRIES_PER_BLOCK;

  // Load data
  acc_t local_data[1];
  if (tid < num_batches * output_batch_size) {
    *local_data =
        input[bid * input_batch_size + indices[tid % output_batch_size]];
    output[tid] = *local_data;
  } else {
    *local_data = 0;
  }

  // Cumsum
  inclusive_sum_scan_kernel<acc_t, 1, NUM_THREADS_PER_BLOCK>(
      local_data,
      bs_temp_storage,
      block_flags,
      block_sums,
      &smem[0],
      num_entries_per_block,
      blockIdx.x,
      gridDim.x > 1,
      1);

  // Store data
  if (tid < num_batches * output_batch_size) {
    output_cumsum[tid] = *local_data;
  }
}

class KeyedJaggedIndexSelectDim1GPUOp
    : public torch::autograd::Function<KeyedJaggedIndexSelectDim1GPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const Tensor& lengths,
      const Tensor& offsets,
      const Tensor& indices, // select same indices for all batches
      const int batch_size,
      const c10::optional<Tensor>& weights) {
    // TODO: Add weights support
    TENSOR_ON_CUDA_GPU(lengths);
    TENSOR_ON_CUDA_GPU(offsets);
    TENSOR_ON_CUDA_GPU(values);
    TENSOR_ON_CUDA_GPU(indices);
    TENSORS_ON_SAME_DEVICE(lengths, indices);
    TENSORS_ON_SAME_DEVICE(offsets, indices);
    TENSORS_ON_SAME_DEVICE(values, indices);
    TORCH_CHECK(values.dim() == 1, "values must be a 1D tensor");
    TORCH_CHECK(lengths.dim() == 1, "lengths must be a 1D tensor");
    TORCH_CHECK(offsets.dim() == 1, "offsets must be a 1D tensor");
    TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK(
        lengths.numel() + 1 == offsets.numel(),
        "offsets size must be lengths size + 1");
    TORCH_CHECK(lengths.numel() % batch_size == 0, "lengths");

    if (weights.has_value()) {
      const Tensor& pos_weights = weights.value();
      TENSOR_ON_CUDA_GPU(pos_weights);
      TENSORS_ON_SAME_DEVICE(pos_weights, indices);
      TORCH_CHECK(pos_weights.dim() == 1, "weights must be a 1D tensor");
      TORCH_CHECK(
          pos_weights.numel() == values.numel(),
          "weights size and values size must be the same");
    }

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(values.get_device());

    const int num_batches = lengths.numel() / batch_size;
    const int num_output_lengths = num_batches * indices.numel();
    const int MAX_CUMSUM_ENTRIES_PER_BLOCK = 256;
    auto grid_size = cuda_calc_xblock_count(
        num_output_lengths, MAX_CUMSUM_ENTRIES_PER_BLOCK);

    Tensor output_offsets =
        at::empty({num_batches * indices.numel()}, offsets.options());
    Tensor output_lengths =
        at::empty({num_batches * indices.numel()}, lengths.options());

    Tensor block_flags, block_sums;
    if (grid_size > 1) {
      block_flags = at::zeros({grid_size}, lengths.options().dtype(at::kInt));
      block_sums = at::empty({grid_size}, output_offsets.options());
    }
    // Do index select and cumsum
    AT_DISPATCH_INDEX_TYPES(
        lengths.scalar_type(), "index_select_scalar_cumsum_wrapper_1", [&] {
          using length_t = index_t;
          AT_DISPATCH_INDEX_TYPES(
              offsets.scalar_type(),
              "index_select_scalar_cumsum_wrapper_2",
              [&] {
                using offset_t = index_t;
                AT_DISPATCH_INDEX_TYPES(
                    indices.scalar_type(),
                    "index_select_scalar_cumsum_wrapper_3",
                    [&] {
                      index_select_scalar_cumsum_kernel<
                          length_t,
                          index_t,
                          offset_t,
                          MAX_CUMSUM_ENTRIES_PER_BLOCK,
                          MAX_CUMSUM_ENTRIES_PER_BLOCK>
                          <<<grid_size,
                             MAX_CUMSUM_ENTRIES_PER_BLOCK,
                             0,
                             at::cuda::getCurrentCUDAStream()>>>(
                              output_lengths.data_ptr<length_t>(),
                              output_offsets.data_ptr<offset_t>(),
                              lengths.data_ptr<length_t>(),
                              indices.data_ptr<index_t>(),
                              num_batches,
                              batch_size,
                              indices.numel(),
                              num_output_lengths -
                                  MAX_CUMSUM_ENTRIES_PER_BLOCK *
                                      (grid_size - 1),
                              grid_size > 1 ? block_flags.data_ptr<int>()
                                            : nullptr,
                              grid_size > 1 ? block_sums.data_ptr<offset_t>()
                                            : nullptr);
                      C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
              });
        });

    // TODO: Try to not do D->H transfer
    const int64_t num_outputs =
        output_offsets[output_offsets.numel() - 1].item<int64_t>();
    Tensor output = at::empty({num_outputs}, values.options());
    Tensor output_weights;
    if (weights.has_value()) {
      output_weights = at::empty({num_outputs}, weights.value().options());
    }
    grid_size = cuda_calc_xblock_count(num_outputs, kMaxThreads);

    if (grid_size != 0) {
#define LAUNCH_KERNEL(WEIGHTED, WEIGHT_TYPE, OUTPUT_WEIGHTS, WEIGHTS)      \
  {                                                                        \
    keyed_jagged_index_select_dim1_kernel<                                 \
        value_t,                                                           \
        index_t,                                                           \
        offset_t,                                                          \
        WEIGHT_TYPE,                                                       \
        WEIGHTED>                                                          \
        <<<grid_size, kMaxThreads, 0, at::cuda::getCurrentCUDAStream()>>>( \
            output.data_ptr<value_t>(),                                    \
            OUTPUT_WEIGHTS,                                                \
            values.data_ptr<value_t>(),                                    \
            WEIGHTS,                                                       \
            offsets.data_ptr<offset_t>() + 1,                              \
            indices.data_ptr<index_t>(),                                   \
            output_offsets.data_ptr<offset_t>(),                           \
            num_batches,                                                   \
            batch_size,                                                    \
            indices.numel(),                                               \
            num_outputs);                                                  \
  }
      AT_DISPATCH_ALL_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          values.scalar_type(),
          "keyed_jagged_index_select_dim1_warpper_1",
          [&] {
            using value_t = scalar_t;
            AT_DISPATCH_INDEX_TYPES(
                offsets.scalar_type(),
                "keyed_jagged_index_select_dim1_warpper_2",
                [&] {
                  using offset_t = index_t;
                  AT_DISPATCH_INDEX_TYPES(
                      indices.scalar_type(),
                      "keyed_jagged_index_select_dim1_warpper_3",
                      [&] {
                        if (weights.has_value()) {
                          AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                              weights.value().scalar_type(),
                              "keyed_jagged_index_select_dim1_warpper_4",
                              [&] {
                                using weight_t = scalar_t;
                                LAUNCH_KERNEL(
                                    true,
                                    weight_t,
                                    output_weights.data_ptr<weight_t>(),
                                    weights.value().data_ptr<weight_t>())
                              });
                        } else {
                          LAUNCH_KERNEL(false, scalar_t, nullptr, nullptr)
                        }
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      });
                });
          });
    }

#undef LAUNCH_KERNEL

    ctx->save_for_backward({indices, output_offsets, offsets});
    ctx->saved_data["num_outputs"] = num_outputs;
    ctx->saved_data["num_inputs"] = values.numel();
    ctx->saved_data["batch_size"] = batch_size;
    ctx->saved_data["num_batches"] = num_batches;
    ctx->saved_data["has_weights"] = weights.has_value();

    if (weights.has_value()) {
      return {output, output_lengths, output_weights};
    }
    return {output, output_lengths};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    bool has_weights = ctx->saved_data["has_weights"].toBool();
    TORCH_CHECK(
        (has_weights && grad_outputs.size() == 3) || grad_outputs.size() == 2);

    const Tensor& grad = grad_outputs[0];
    TENSOR_ON_CUDA_GPU(grad_outputs[0]);

    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    const Tensor& indices = *savedItr++;
    const Tensor& grad_offsets = *savedItr++;
    const Tensor& output_offsets = *savedItr++;

    TENSORS_ON_SAME_DEVICE(grad, indices);

    int64_t num_grads = ctx->saved_data["num_outputs"].toInt();
    int64_t num_outputs = ctx->saved_data["num_inputs"].toInt();
    int64_t output_batch_size = ctx->saved_data["batch_size"].toInt();
    int64_t num_batches = ctx->saved_data["num_batches"].toInt();

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(grad.get_device());

    Tensor grad_input = at::zeros({num_outputs}, grad.options());
    auto grid_size = cuda_calc_xblock_count(grad.numel(), kMaxThreads);

    if (grid_size != 0) {
      AT_DISPATCH_ALL_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          grad.scalar_type(),
          "keyed_jagged_index_add_dim1_wrapper_1",
          [&] {
            AT_DISPATCH_INDEX_TYPES(
                grad_offsets.scalar_type(),
                "keyed_jagged_index_add_dim1_wrapper_2",
                [&] {
                  using offset_t = index_t;
                  AT_DISPATCH_INDEX_TYPES(
                      indices.scalar_type(),
                      "keyed_jagged_index_add_dim1_wrapper_3",
                      [&] {
                        keyed_jagged_index_add_dim1_kernel<<<
                            grid_size,
                            kMaxThreads,
                            0,
                            at::cuda::getCurrentCUDAStream()>>>(
                            grad_input.data_ptr<scalar_t>(),
                            grad.data_ptr<scalar_t>(),
                            grad_offsets.data_ptr<offset_t>(),
                            indices.data_ptr<index_t>(),
                            output_offsets.data_ptr<offset_t>() +
                                1, // shift it to make it inclusive cumsum
                            num_batches,
                            indices.numel(),
                            output_batch_size,
                            grad.numel());
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                      });
                });
          });
    }

    return {
        grad_input,
        torch::autograd::Variable(), // lengths
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable(), // indices
        torch::autograd::Variable(), // batch_size
        torch::autograd::Variable() // weights
    };
  }
};

std::vector<Tensor> keyed_jagged_index_select_dim_1_gpu(
    const Tensor& values,
    const Tensor& lengths,
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t batch_size,
    const c10::optional<Tensor>& weights) {
  return KeyedJaggedIndexSelectDim1GPUOp::apply(
      values, lengths, offsets, indices, batch_size, weights);
}

} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA("dense_to_jagged", fbgemm_gpu::dense_to_jagged);
  DISPATCH_TO_CUDA(
      "dense_to_jagged_forward", fbgemm_gpu::dense_to_jagged_forward);
  DISPATCH_TO_CUDA(
      "jagged_to_padded_dense", fbgemm_gpu::jagged_to_padded_dense);
  DISPATCH_TO_CUDA(
      "jagged_to_padded_dense_forward",
      fbgemm_gpu::jagged_to_padded_dense_forward);
  DISPATCH_TO_CUDA(
      "jagged_to_padded_dense_backward",
      fbgemm_gpu::jagged_to_padded_dense_backward);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_add", fbgemm_gpu::jagged_dense_elementwise_add);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_add_jagged_output",
      fbgemm_gpu::jagged_dense_elementwise_add_jagged_output);
  DISPATCH_TO_CUDA(
      "jagged_dense_dense_elementwise_add_jagged_output_forward",
      fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output_forward);
  DISPATCH_TO_CUDA(
      "jagged_dense_dense_elementwise_add_jagged_output",
      fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_mul", fbgemm_gpu::jagged_dense_elementwise_mul);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_mul_forward",
      fbgemm_gpu::jagged_dense_elementwise_mul_forward);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_mul_backward",
      fbgemm_gpu::jagged_dense_elementwise_mul_backward);
  DISPATCH_TO_CUDA(
      "batched_dense_vec_jagged_2d_mul",
      fbgemm_gpu::batched_dense_vec_jagged_2d_mul);
  DISPATCH_TO_CUDA(
      "batched_dense_vec_jagged_2d_mul_forward",
      fbgemm_gpu::batched_dense_vec_jagged_2d_mul_forward);
  DISPATCH_TO_CUDA(
      "batched_dense_vec_jagged_2d_mul_backward",
      fbgemm_gpu::batched_dense_vec_jagged_2d_mul_backward);
  DISPATCH_TO_CUDA(
      "jagged_index_select_2d_forward",
      fbgemm_gpu::jagged_index_select_2d_forward_cuda);
  DISPATCH_TO_CUDA(
      "jagged_index_add_2d_forward",
      fbgemm_gpu::jagged_index_add_2d_forward_cuda);
  DISPATCH_TO_CUDA("jagged_1d_to_dense", fbgemm_gpu::jagged_1d_to_dense);
  DISPATCH_TO_CUDA("jagged_2d_to_dense", fbgemm_gpu::jagged_2d_to_dense);
  DISPATCH_TO_CUDA(
      "stacked_jagged_1d_to_dense", fbgemm_gpu::stacked_jagged_1d_to_dense_gpu);
  DISPATCH_TO_CUDA(
      "stacked_jagged_2d_to_dense", fbgemm_gpu::stacked_jagged_2d_to_dense_gpu);
  DISPATCH_TO_CUDA(
      "stacked_jagged_2d_to_dense_forward",
      fbgemm_gpu::stacked_jagged_2d_to_dense_forward_cuda);
  DISPATCH_TO_CUDA(
      "stacked_jagged_2d_to_dense_backward",
      fbgemm_gpu::stacked_jagged_2d_to_dense_backward_cuda);
  // TODO: combine the API with permute_2D_sparse_data and implement a CPU op
  DISPATCH_TO_CUDA(
      "keyed_jagged_index_select_dim1",
      fbgemm_gpu::keyed_jagged_index_select_dim_1_gpu);
  DISPATCH_TO_CUDA("jagged_softmax", fbgemm_gpu::jagged_softmax);
  DISPATCH_TO_CUDA(
      "jagged_softmax_forward", fbgemm_gpu::jagged_softmax_forward);
  DISPATCH_TO_CUDA(
      "jagged_softmax_backward", fbgemm_gpu::jagged_softmax_backward);
  DISPATCH_TO_CUDA("jagged_jagged_bmm", fbgemm_gpu::jagged_jagged_bmm);
  DISPATCH_TO_CUDA(
      "jagged_jagged_bmm_forward", fbgemm_gpu::jagged_jagged_bmm_forward);
  DISPATCH_TO_CUDA("jagged_dense_bmm", fbgemm_gpu::jagged_dense_bmm);
  DISPATCH_TO_CUDA(
      "jagged_dense_bmm_forward", fbgemm_gpu::jagged_dense_bmm_forward);
}
