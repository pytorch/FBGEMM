/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

using Tensor = at::Tensor;

namespace fbgemm_gpu {

namespace {

__host__ __device__ inline float float_to_msfp(
    const float val_fp,
    const int shared_expo,
    const int mbits,
    const int bias,
    const float max_pos) {
  fbgemm_gpu::fint32 X, bouncer, scale, inv_scale;
  int32_t expo, emin, delta_E, nbits2round;

  X.F = val_fp;
  const uint32_t sign_bit = X.I & 0x80000000;
  X.I = X.I & 0x7FFFFFFF; // 31 bits

  emin = 1 - bias;

  // Because the input value can be of extreme magnitude
  // We scale them into less extreme to avoid potential exception during
  // manipulation
  const int32_t E = ((X.I & 0x7F800000) >> 23) - 127;
  if (E >= 0) {
    scale.I = 0X2F800000;
    inv_scale.I = 0X4F800000; // scale is 2^-32, inv_scale is 2^32
    delta_E = -32;
  } else {
    scale.I = 0x4F800000;
    inv_scale.I = 0x2F800000;
    delta_E = 32;
  }
  X.F *= scale.F; // at this point X is never close to over/underflow
  expo = ((X.I & 0x7F800000) >> 23) - 127 - delta_E;

  // If expo >= emin
  // We round to mbits explicit mantissa bits
  // That is, we want to round off 23-mbits of the trailing bits in X
  nbits2round = 23 - mbits;
  // However, if expo < emin, we need to round more bits off
  nbits2round += ::max(emin - expo, 0); // max(emin - expo, 0);
  // also need to right shift mantissa with the shared expoennt
  nbits2round += ::max(shared_expo - expo, 0);

  bouncer.I = (nbits2round << 23) + (X.I & 0x7F800000);
  X.F = X.F + bouncer.F; // Because bouncer is exactly 2^nbits2round bigger
                         // this addition forces the rounding off of nbits2round
  X.F = X.F - bouncer.F; // X.F is the original X with nbits2round rounded off

  // restore the true magnitude by undoing the previous scale
  X.F *= inv_scale.F;
  // clip on the large end of the domain
  X.F = ::min(X.F, max_pos);
  // restores the original sign
  X.I |= sign_bit;

  const float val_msfp = X.F;
  return val_msfp;
}

__global__ inline void _compute_msfp_shared_exponent_cuda_kernel(
    const float* __restrict__ input,
    const int nrows,
    const int ncols,
    const int bounding_box_size,
    int* __restrict__ shared_exponents) {
  const int tidy = blockIdx.y * blockDim.y +
      threadIdx.y; // to get the threadid-y dimension of this thread
  const int tidx = blockIdx.x * blockDim.x +
      threadIdx.x; // to get the threadid-x dimension of this thread

  const int row_incre = blockDim.y * gridDim.y;
  const int col_incre = blockDim.x * gridDim.x;

  for (int row = tidy; row < nrows; row += row_incre) {
    const float* input_row = input + row * ncols;
    int* shared_expo_row = shared_exponents + row * ncols;
    for (int col = tidx; col < ncols; col += col_incre) {
      const int boundingbox_start = col / bounding_box_size * bounding_box_size;
      const int boundingbox_end =
          ::min(boundingbox_start + bounding_box_size, ncols);

      int32_t max_exponent = 0;
      for (int i = boundingbox_start; i < boundingbox_end; i++) {
        // update the max_exponent
        fbgemm_gpu::fint32 org_data;
        org_data.F = input_row[i];
        org_data.I = org_data.I & 0x7FFFFFFF; // 31 bits
        const int32_t exponent = ((org_data.I & 0x7F800000) >> 23);
        max_exponent = ::max(max_exponent, exponent);
      }
      shared_expo_row[col] = static_cast<int>(max_exponent) - 127;
    }
  }
}

} // namespace

/// @ingroup quantize-ops-cuda
/// Converts a tensor of  `float` values into a tensor of Microsoft Floating
/// Point (`msfp`) values.
///
/// @param input A tensor of `float` values
/// @param bounding_box_size
/// @param ebits
/// @param mbits
/// @param bias
/// @param min_pos
/// @param max_pos
///
/// @return A new tensor with values from the input tensor converted to `msfp`.
DLL_PUBLIC at::Tensor _float_to_msfp_gpu(
    const at::Tensor& input,
    const int64_t bounding_box_size,
    const int64_t ebits,
    const int64_t mbits,
    const int64_t bias,
    const double min_pos,
    const double max_pos) {
  TENSOR_ON_CUDA_GPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  TORCH_CHECK(ebits <= 8);
  TORCH_CHECK(mbits <= 23);
  TORCH_CHECK(ebits > 0 && mbits > 0);
  TORCH_CHECK(min_pos > 0 && max_pos > 0 && max_pos > min_pos);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(input.get_device());

  const int nrows = input.size(0);
  const int ncols = input.size(1);

  auto output = at::empty({nrows, ncols}, input.options().dtype(at::kFloat));
  if (nrows == 0 || ncols == 0) {
    return output;
  }

  constexpr int threads_per_block = 256;

  const int blockDim_x = std::min(ncols, threads_per_block);
  const dim3 blockDim(blockDim_x, threads_per_block / blockDim_x);
  const int gridDim_x = (ncols + blockDim.x - 1) / blockDim.x;
  const int gridDim_y = std::min((nrows + blockDim.y - 1) / blockDim.y, 65535u);
  const dim3 gridDim(gridDim_x, gridDim_y);

  auto shared_exponents =
      at::empty({nrows, ncols}, input.options().dtype(at::kInt));

  _compute_msfp_shared_exponent_cuda_kernel<<<
      gridDim,
      blockDim,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      input.contiguous().data_ptr<float>(),
      nrows,
      ncols,
      bounding_box_size,
      shared_exponents.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(input)
                  .add_input(shared_exponents)
                  .build();

  at::native::gpu_kernel(
      iter, [=] GPU_LAMBDA(float in, int shared_expo) -> float {
        return float_to_msfp(in, shared_expo, mbits, bias, max_pos);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

/// @ingroup quantize-ops-cuda
/// Converts a tensor of Microsoft Floating Point (`msfp`) values into a tensor
/// of `float` values.
///
/// @param input A tensor of `msfp` values
/// @param ebits
/// @param mbits
/// @param bias
///
/// @return A new tensor with values from the input tensor converted to `float`.
DLL_PUBLIC at::Tensor _msfp_to_float_gpu(
    const at::Tensor& input,
    const int64_t ebits,
    const int64_t mbits,
    const int64_t bias) {
  TENSOR_ON_CUDA_GPU(input);

  // Because float_to_msfp is a fakequant operator,
  // the input msfp number is already a FP32 number
  // with limited precision.
  // Thus this msfp_to_float is really a no-op
  return input.clone();
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(
    CUDA,
    "FloatToMSFPQuantized",
    fbgemm_gpu::_float_to_msfp_gpu);
FBGEMM_OP_DISPATCH(
    CUDA,
    "MSFPQuantizedToFloat",
    fbgemm_gpu::_msfp_to_float_gpu);
