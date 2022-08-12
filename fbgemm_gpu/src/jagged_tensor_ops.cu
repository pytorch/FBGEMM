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

// clang-format off
#include "fbgemm_gpu/cub_namespace_prefix.cuh"
#include "cub/device/device_scan.cuh"
#include "fbgemm_gpu/cub_namespace_postfix.cuh"
// clang-format on

#include "fbgemm_gpu/fbgemm_cuda_utils.cuh"
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
        [f_ = f] __device__(scalar_t x, scalar_t y, scalar_t /*unused*/)       \
            -> scalar_t { return f_(x, y); });                                 \
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
          num_jagged_dim, x_values, y_reshaped, y_reshaped, output_values)) {
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
                  [_f = f] __device__(__half x, __half y0, __half) -> __half {
                    return _f(x, y0);
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
                  [_f = f] __device__(__half x, __half y0, __half y1)
                      -> __half { return _f(x, y0, y1); });
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
  if (values.dim() > 1) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values = at::empty(padded_values_shape, values.options());
  Tensor padded_values_view =
      values.dim() == 1 ? padded_values.unsqueeze(-1) : padded_values;

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
    const std::vector<int64_t>& max_lengths) {
  auto grad_padded_values = grad_output;
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad_padded_values.get_device());

  int32_t D = grad_padded_values.size(-1);
  // Initialize with zeros so output will be zero for the portion truncated
  // in forward.
  auto grad_values =
      at::zeros({max_lengths[0], D}, grad_padded_values.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_padded_values.scalar_type(),
      "jagged_2d_to_dense_backward_kernel",
      [&] {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            grad_values, // dummy not used in the lambda function
            {offsets},
            grad_padded_values,
            grad_values,
            [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
              return y;
            });
      });

  return grad_values;
}
class JaggedToPaddedDenseGPUOp
    : public torch::autograd::Function<JaggedToPaddedDenseGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const std::vector<Tensor>& offsets,
      const std::vector<int64_t>& max_lengths,
      const double padding_value) {
    ctx->save_for_backward(offsets);
    ctx->saved_data["total_L"] = values.size(0);

    Tensor padded_values = jagged_to_padded_dense_forward(
        values, offsets, max_lengths, padding_value);

    return {padded_values};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    int32_t total_L = ctx->saved_data["total_L"].toInt();
    TORCH_CHECK(grad_outputs.size() == 1);

    TORCH_CHECK(total_L >= 0);
    auto grad_values =
        jagged_to_padded_dense_backward(grad_outputs[0], {offsets}, {total_L});

    return {
        grad_values,
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable(), // max_lengths
        torch::autograd::Variable(), // padding_value
    };
  }
};

///@ingroup jagged-tensor-ops-cuda
Tensor jagged_to_padded_dense(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const std::vector<int64_t>& max_lengths,
    const double padding_value) {
  return JaggedToPaddedDenseGPUOp::apply(
      values, offsets, max_lengths, padding_value)[0];
}

///@ingroup jagged-tensor-ops-cuda
/// output = x + y where x is jagged, y and output are dense
Tensor jagged_dense_elementwise_add(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  // Construct max_lengths from y
  std::vector<int64_t> max_lengths;
  max_lengths.reserve(x_offsets.size());
  for (int d = 1; d < y.dim() - 1; d++) {
    max_lengths.push_back(y.size(d));
  }
  TORCH_CHECK(max_lengths.size() == x_offsets.size());

  // Convert x to dense (assume padding is 0.0)
  auto xd = JaggedToPaddedDenseGPUOp::apply(
      x_values,
      x_offsets,
      max_lengths,
      /* padding_value */ 0.0)[0];

  auto dense_output = xd + y;
  return dense_output;
}

///@ingroup jagged-tensor-ops-cuda
class DenseToJaggedGPUOp
    : public torch::autograd::Function<DenseToJaggedGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& dense,
      const std::vector<Tensor>& offsets,
      const c10::optional<int64_t>& total_L) {
    ctx->save_for_backward(offsets);
    ctx->saved_data["dense_shape"] = dense.sizes();

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

    AT_DISPATCH_SWITCH(
        values.scalar_type(),
        "dense_to_jagged_gpu_op_forward",
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&] {
              jagged_dense_elementwise_jagged_output_opt_<scalar_t>(
                  values,
                  offsets,
                  dense,
                  output,
                  [] __device__(scalar_t /*unused*/, scalar_t y) -> scalar_t {
                    return y;
                  }); // device lambda
            } // lambda
            ) // CASE
        AT_DISPATCH_CASE_FLOATING_TYPES_AND(
            at::ScalarType::Long,
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

    Tensor dense_values_grad = jagged_to_padded_dense(
        grad_outputs[0],
        offsets,
        std::vector<int64_t>(dense_shape.begin() + 1, dense_shape.end() - 1),
        /*padding_value=*/0);
    TORCH_CHECK(dense_values_grad.sizes() == dense_shape);

    return {
        dense_values_grad,
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable()}; // total_L
  }
};

class JaggedDenseDenseAddJaggedOutputGPUOp
    : public torch::autograd::Function<JaggedDenseDenseAddJaggedOutputGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& x_values,
      const std::vector<Tensor>& offsets,
      const Tensor& dense_0,
      const Tensor& dense_1) {
    ctx->save_for_backward(offsets);
    ctx->saved_data["dense_shape"] = dense_0.sizes();

    TORCH_CHECK(dense_0.sizes() == dense_1.sizes());
    auto output = at::empty_like(x_values);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dense_0.get_device());

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
        AT_DISPATCH_CASE_FLOATING_TYPES([&] {
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

    Tensor dense_values_grad_0 = jagged_to_padded_dense(
        grad_outputs[0],
        offsets,
        std::vector<int64_t>(dense_shape.begin() + 1, dense_shape.end() - 1),
        /*padding_value=*/0);
    TORCH_CHECK(dense_values_grad_0.sizes() == dense_shape);

    Tensor dense_values_grad_1 = dense_values_grad_0;

    return {
        grad_outputs[0],
        torch::autograd::Variable(), // offsets
        dense_values_grad_0,
        dense_values_grad_1};
  }
};

///@ingroup jagged-tensor-ops-cuda
std::tuple<Tensor, std::vector<Tensor>> dense_to_jagged(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    const c10::optional<int64_t>& total_L) {
  return {DenseToJaggedGPUOp::apply(dense, offsets, total_L)[0], offsets};
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
        AT_DISPATCH_CASE_FLOATING_TYPES([&] {
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

    Tensor dense_values_grad = jagged_to_padded_dense(
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

// output = x + y_0 + y_1 where x is jagged, y_0 and y_1 are dense, and output
// is jagged
std::tuple<Tensor, std::vector<Tensor>>
jagged_dense_dense_elementwise_add_jagged_output(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0,
    const Tensor& y_1) {
  auto sum_values = JaggedDenseDenseAddJaggedOutputGPUOp::apply(
      x_values, x_offsets, y_0, y_1)[0];

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

class JaggedDenseMulGPUOp
    : public torch::autograd::Function<JaggedDenseMulGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& x_values,
      const Tensor& y,
      const std::vector<Tensor>& x_offsets) {
    std::vector<Tensor> tensors_to_save;
    tensors_to_save.push_back(x_values);
    tensors_to_save.insert(
        tensors_to_save.end(), x_offsets.begin(), x_offsets.end());
    tensors_to_save.push_back(y);
    ctx->save_for_backward(tensors_to_save);

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
        AT_DISPATCH_CASE_FLOATING_TYPES([&] {
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

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const Tensor x_values = ctx->get_saved_variables().front();
    std::vector<Tensor> x_offsets;
    for (int i = 1; i < ctx->get_saved_variables().size() - 1; ++i) {
      x_offsets.push_back(ctx->get_saved_variables()[i]);
    }
    Tensor y = ctx->get_saved_variables().back();
    TORCH_CHECK(grad_outputs.size() == 1);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(grad_outputs[0].get_device());

    Tensor x_values_grad = at::empty_like(grad_outputs[0]);
    Tensor y_grad = at::empty_like(y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x_values.scalar_type(), "jagged_scalars", [&] {
          jagged_dense_elementwise_jagged_output_<scalar_t>(
              grad_outputs[0],
              x_offsets,
              y,
              x_values_grad,
              [] __device__(scalar_t x, scalar_t y) -> scalar_t {
                return x * y;
              });

          jagged_jagged_elementwise_dense_output_<scalar_t>(
              grad_outputs[0],
              x_offsets,
              x_values,
              y_grad,
              [] __device__(scalar_t x, scalar_t y) -> scalar_t {
                return x * y;
              });
        });

    return {x_values_grad, y_grad, torch::autograd::Variable()};
  }
};

///@ingroup jagged-tensor-ops-cuda
std::tuple<Tensor, std::vector<Tensor>> jagged_dense_elementwise_mul(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  // Convert to jagged
  auto prod_values = JaggedDenseMulGPUOp::apply(x_values, y, x_offsets)[0];

  return {prod_values, x_offsets};
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

// batched dense vector x jagged 2D tensor multiplication
// dense vector [B H, N]
// jagged tensor [B, N, H D] where N is jagged
class BatchedDenseVecJagged2DMulGPUOp
    : public torch::autograd::Function<BatchedDenseVecJagged2DMulGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& v,
      const Tensor& a_values,
      const Tensor& a_offsets) {
    ctx->save_for_backward({v, a_values, a_offsets});

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

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    const Tensor v = *savedItr++;
    const Tensor a_values = *savedItr++;
    const Tensor a_offsets = *savedItr++;
    TORCH_CHECK(grad_outputs.size() == 1);

    TENSOR_ON_CUDA_GPU(grad_outputs[0]);
    TENSOR_ON_CUDA_GPU(a_values);
    TENSOR_ON_CUDA_GPU(a_offsets);
    TENSOR_ON_CUDA_GPU(v);

    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(grad_outputs[0].get_device());

    const int B = a_offsets.numel() - 1;
    const int D = grad_outputs[0].size(-1);

    Tensor a_values_grad = at::zeros_like(a_values);
    Tensor v_grad = at::empty_like(v);

    if (B > 0 && D > 0) {
      TORCH_CHECK(
          v.size(0) % B == 0,
          "B, ",
          B,
          " doesn't divide v.size(0), ",
          v.size(0));
      const int H = v.size(0) / B;
      const int max_L = v.size(-1);

      AT_DISPATCH_INDEX_TYPES(
          a_offsets.scalar_type(),
          "dense_vec_jagged_2d_bmm_baackward_kernel_1",
          [&] {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                grad_outputs[0].scalar_type(),
                "dense_vec_jagged_2d_bmm_baackward_kernel_2",
                [&] {
                  int block_dim_x = std::min(
                      div_round_up(max_L, kWarpSize) * kWarpSize, kMaxThreads);
                  int block_dim_y = kMaxThreads / block_dim_x;

                  dense_vec_jagged_2d_transposed_bmm<index_t, scalar_t>
                      <<<div_round_up(B * H, block_dim_y),
                         dim3(block_dim_x, block_dim_y),
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          grad_outputs[0].packed_accessor32<scalar_t, 2>(),
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
                          grad_outputs[0].packed_accessor32<scalar_t, 2>(),
                          a_offsets.packed_accessor32<index_t, 1>(),
                          a_values_grad.packed_accessor32<scalar_t, 2>());
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    } else {
      v_grad.zero_();
    }

    return {
        v_grad,
        a_values_grad,
        torch::autograd::Variable(), // a_offsets
    };
  }
};

///@ingroup jagged-tensor-ops-cuda
Tensor batched_dense_vec_jagged_2d_mul(
    const Tensor& v,
    const Tensor& a_values,
    const Tensor& a_offsets) {
  return BatchedDenseVecJagged2DMulGPUOp::apply(v, a_values, a_offsets)[0];
}

} // namespace

Tensor jagged_1d_to_dense_gpu(
    Tensor values,
    Tensor offsets,
    int64_t max_L,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  return jagged_to_padded_dense(values, {offsets}, {max_L}, padding_value);
}

Tensor jagged_2d_to_dense_gpu(
    Tensor values,
    Tensor offsets,
    int64_t max_sequence_length) {
  return jagged_to_padded_dense(
      values,
      {offsets},
      {max_sequence_length},
      /*padding_value=*/0);
}

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
  return jagged_to_padded_dense_backward(grad_output, {offsets}, {max_lengths});
}

// stacked ops
std::tuple<std::vector<Tensor>, std::vector<Tensor>>
stacked_jagged_2d_to_dense_forward_cuda(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key) {
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

    padded_values_per_key.push_back(jagged_to_padded_dense(
        values.slice(0, offset_per_key[t], offset_per_key[t + 1]),
        {offsets},
        {max_L},
        /*padding_value=*/0));
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_values.scalar_type(), "jagged_2d_to_dense_backward_kernel", [&] {
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

    padded_values_per_key.push_back(jagged_1d_to_dense_gpu(
        values.slice(0, offset_per_key[t], offset_per_key[t + 1]),
        offsets,
        max_L,
        padding_value));
  }

  return padded_values_per_key;
}

template <typename scalar_t>
__device__ __forceinline__ void binary_search_range(
    int* found,
    const scalar_t* arr,
    const scalar_t target,
    const int num_entries) {
  const int last_entry = num_entries - 1;
  int start = 0, end = last_entry;
  int found_ = -1;
  while (start <= end) {
    int mid = start + (end - start) / 2;
    scalar_t mid_offset = arr[mid];
    if (target == mid_offset) {
      if (mid != last_entry && target != arr[last_entry]) {
        // Do linear scan in case of duplicate data (We assume that the
        // number of duplicates is small.  This can we very bad if the
        // number of duplicates is large)
        for (int i = mid + 1; i < num_entries; i++) {
          if (target != arr[i]) {
            found_ = i;
            break;
          }
        }
      }
      break;
    } else if (target < mid_offset) {
      if (mid == 0) {
        found_ = 0;
        break;
      } else if (mid - 1 >= 0 && target > arr[mid - 1]) {
        found_ = mid;
        break;
      }
      end = mid - 1;
    } else {
      if (mid + 1 <= last_entry && target < arr[mid + 1]) {
        found_ = mid + 1;
        break;
      }
      start = mid + 1;
    }
  }
  *found = found_;
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

Tensor jagged_index_select_2d_cuda(
    const Tensor& values,
    const Tensor& indices,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    const int64_t num_dense_output_rows) {
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
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half,
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
    const scalar_t* grad,
    const offset_t* grad_offsets,
    const index_t* indices,
    const offset_t* output_offsets,
    const int64_t num_grad_rows,
    const int64_t num_dense_grad_rows,
    const int64_t num_cols) {
  __shared__ int smem[1];
  for (offset_t dense_grad_offset = blockIdx.x;
       dense_grad_offset < num_dense_grad_rows;
       dense_grad_offset += gridDim.x) {
    // Binary search
    // TODO: use multiple threads to do bin search to reduce number of steps
    if (threadIdx.x == 0) {
      binary_search_range(smem, grad_offsets, dense_grad_offset, num_grad_rows);
    }
    __syncthreads();

    // All threads load index_pos from shared memory and return if the index_pos
    // is invalid
    int index_pos = smem[0];

    // TODO: Can also be obtained during the binary search
    // Relative index position
    const offset_t rel_index =
        dense_grad_offset - (index_pos == 0 ? 0 : grad_offsets[index_pos - 1]);
    const index_t index = indices[index_pos];
    const offset_t output_offset =
        (index == 0 ? 0 : output_offsets[index - 1]) + rel_index;

    // Shift buffers
    const scalar_t* grad_ = grad + dense_grad_offset * num_cols;
    scalar_t* output_ = output + output_offset * num_cols;

    // TODO: Avoid using atoimcAdd (because it could lead to the numerical
    // indeterminism issue)
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
      gpuAtomicAdd(&output_[i], grad_[i]);
    }
  }
}

Tensor jagged_index_add_2d_cuda(
    const Tensor& grad,
    const Tensor& indices,
    const Tensor& grad_offsets,
    const Tensor& output_offsets,
    const int64_t num_dense_grad_rows,
    const int64_t num_output_rows) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(grad.get_device());

  auto num_cols = grad.size(1);
  const int64_t num_grad_rows = indices.numel();

  const int64_t max_num_blocks = 1024; // Arbitrarily set to this number of now
  const int64_t max_num_threads = kMaxThreads;
  const int64_t num_blocks = std::min(max_num_blocks, num_dense_grad_rows);
  const int64_t num_threads = std::min(max_num_threads, num_cols);
  Tensor output = at::zeros({num_output_rows, num_cols}, grad.options());

  if (num_blocks > 0) {
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half,
        grad.scalar_type(),
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
                    grad.data_ptr<scalar_t>(),
                    grad_offsets.data_ptr<int64_t>(),
                    indices.data_ptr<index_t>(),
                    output_offsets.data_ptr<int64_t>(),
                    num_grad_rows,
                    num_dense_grad_rows,
                    num_cols);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }

  return output;
}

class JaggedIndexSelect2dGPUOp
    : public torch::autograd::Function<JaggedIndexSelect2dGPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const Tensor& lengths,
      const Tensor& indices) {
    TENSOR_ON_CUDA_GPU(lengths);
    TENSOR_ON_CUDA_GPU(values);
    TENSOR_ON_CUDA_GPU(indices);
    TENSORS_ON_SAME_DEVICE(lengths, indices);
    TENSORS_ON_SAME_DEVICE(values, indices);

    Tensor output_lengths = at::index_select(lengths, 0, indices);
    Tensor output_offsets = output_lengths.cumsum(0);
    Tensor input_offsets = lengths.cumsum(0);

    // TODO: Try to not do D->H transfer
    // The challenge here is num_dense_output_rows is needed for allocating the
    // output buffer
    int64_t num_dense_output_rows =
        output_offsets[output_offsets.numel() - 1].item<int64_t>();

    ctx->save_for_backward({indices, output_offsets, input_offsets});
    ctx->saved_data["num_dense_grad_rows"] = num_dense_output_rows;
    ctx->saved_data["num_input_rows"] = values.size(0);

    return {
        jagged_index_select_2d_cuda(
            values,
            indices,
            input_offsets,
            output_offsets,
            num_dense_output_rows),
        output_lengths};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    TORCH_CHECK(grad_outputs.size() == 2);
    TENSOR_ON_CUDA_GPU(grad_outputs[0]);

    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    Tensor indices = *savedItr++;
    Tensor grad_offsets = *savedItr++;
    Tensor output_offsets = *savedItr++;

    Tensor grad = grad_outputs[0];
    TENSORS_ON_SAME_DEVICE(grad, indices);

    int64_t num_dense_grad_rows =
        ctx->saved_data["num_dense_grad_rows"].toInt();
    int64_t num_output_rows = ctx->saved_data["num_input_rows"].toInt();

    return {
        jagged_index_add_2d_cuda(
            grad,
            indices,
            grad_offsets,
            output_offsets,
            num_dense_grad_rows,
            num_output_rows),
        torch::autograd::Variable(), // lengths
        torch::autograd::Variable() // indices
    };
  }
};

std::vector<Tensor> jagged_index_select_2d_gpu(
    const Tensor& values,
    const Tensor& lengths,
    const Tensor& indices) {
  return JaggedIndexSelect2dGPUOp::apply(values, lengths, indices);
}
} // namespace fbgemm_gpu

TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA("dense_to_jagged", fbgemm_gpu::dense_to_jagged);
  DISPATCH_TO_CUDA(
      "jagged_to_padded_dense", fbgemm_gpu::jagged_to_padded_dense);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_add", fbgemm_gpu::jagged_dense_elementwise_add);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_add_jagged_output",
      fbgemm_gpu::jagged_dense_elementwise_add_jagged_output);
  DISPATCH_TO_CUDA(
      "jagged_dense_dense_elementwise_add_jagged_output",
      fbgemm_gpu::jagged_dense_dense_elementwise_add_jagged_output);
  DISPATCH_TO_CUDA(
      "jagged_dense_elementwise_mul", fbgemm_gpu::jagged_dense_elementwise_mul);
  DISPATCH_TO_CUDA(
      "batched_dense_vec_jagged_2d_mul",
      fbgemm_gpu::batched_dense_vec_jagged_2d_mul);
  DISPATCH_TO_CUDA(
      "jagged_index_select", fbgemm_gpu::jagged_index_select_2d_gpu);
}
