// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include <cuda_bf16.h>
#include <torch/types.h>

namespace fbgemm_gpu {
namespace gen_ai {
namespace fb {
namespace tensor_parallel {

__global__ void row_col_rescale_kernel(
    at::PackedTensorAccessor64<at::BFloat16, 2, at::RestrictPtrTraits> inputs,
    at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> row_scale,
    at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> col_scale,
    at::PackedTensorAccessor64<at::BFloat16, 2, at::RestrictPtrTraits> outputs,
    const int64_t num_rows,
    const int64_t num_cols) {
  for (int64_t row_idx = blockIdx.x; row_idx < num_rows; row_idx += gridDim.x) {
    auto rowScaleValue = row_scale[row_idx];
    for (int64_t col_idx = 2 * threadIdx.x; col_idx < num_cols;
         col_idx += 2 * blockDim.x) {
      __nv_bfloat162 inputValue =
          *reinterpret_cast<__nv_bfloat162*>(&inputs[row_idx][col_idx]);
      float2 colScaleValue = *reinterpret_cast<float2*>(&col_scale[col_idx]);

      __nv_bfloat162 outputValue;
      outputValue.x = float(inputValue.x) / (rowScaleValue * colScaleValue.x);
      outputValue.y = float(inputValue.y) / (rowScaleValue * colScaleValue.y);
      *reinterpret_cast<__nv_bfloat162*>(&outputs[row_idx][col_idx]) =
          outputValue;
    }
  }
  return;
}

at::Tensor row_col_rescale(
    at::Tensor inputs,
    at::Tensor row_scale,
    at::Tensor col_scale,
    std::optional<at::Tensor> outputs) {
  const int64_t num_rows = row_scale.size(0);
  const int64_t num_cols = col_scale.size(0);
  // Check input dimensions
  TORCH_CHECK(inputs.size(0) == num_rows);
  TORCH_CHECK(inputs.size(1) == num_cols);

  at::Tensor out_tensor;
  if (outputs.has_value()) {
    out_tensor = outputs.value();
  } else {
    out_tensor = at::empty({num_rows, num_cols}, inputs.options());
  }

  // Check output dimensions
  TORCH_CHECK(out_tensor.size(0) == num_rows);
  TORCH_CHECK(out_tensor.size(1) == num_cols);

  constexpr int64_t kThreadsPerBlock = 512;
  const int64_t kNumBlocks = std::min(num_rows, static_cast<int64_t>(1024));
  dim3 threads = kThreadsPerBlock;
  dim3 blocks = kNumBlocks;
  row_col_rescale_kernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      inputs.packed_accessor64<at::BFloat16, 2, at::RestrictPtrTraits>(),
      row_scale.packed_accessor64<float, 1, at::RestrictPtrTraits>(),
      col_scale.packed_accessor64<float, 1, at::RestrictPtrTraits>(),
      out_tensor.packed_accessor64<at::BFloat16, 2, at::RestrictPtrTraits>(),
      num_rows,
      num_cols);
  return out_tensor;
}

} // namespace tensor_parallel
} // namespace fb
} // namespace gen_ai
} // namespace fbgemm_gpu
