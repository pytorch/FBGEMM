/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.cuh"

#include "fbgemm_gpu/cub_namespace_postfix.cuh"
#include "fbgemm_gpu/cub_namespace_prefix.cuh"

#include <cub/device/device_scan.cuh>

using Tensor = at::Tensor;

namespace fbgemm_gpu {

///@ingroup jagged-tensor-ops-cuda
at::Tensor jagged_to_padded_dense_forward(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const at::ArrayRef<at::SymInt>& max_lengths,
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
  at::SymDimVector padded_values_shape({at::SymInt(offsets[0].size(0) - 1)});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = values.dim() == 1;
  if (!D_folded) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values =
      at::empty_symint(padded_values_shape, values.options());
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
        at::ArrayRef<at::SymInt>({max_L}),
        padding_value));
  }
  return padded_values_per_key;
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
        at::ArrayRef<at::SymInt>({max_L}),
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

namespace {

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
} // namespace

std::vector<Tensor> stacked_jagged_2d_to_dense_gpu(
    Tensor values,
    Tensor lengths,
    const std::vector<int64_t>& offset_per_key,
    const std::vector<int64_t>& max_lengths_per_key,
    int64_t padding_value) {
  TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(values, lengths);
  TORCH_CHECK(values.dim() == 2);
  TORCH_CHECK(lengths.dim() == 2);
  return StackedJagged2DToDenseGPUOp::apply(
      values, lengths, offset_per_key, max_lengths_per_key, padding_value);
}

Tensor jagged_2d_to_dense_gpu_forward(
    Tensor values,
    Tensor offsets,
    int64_t max_sequence_length) {
  return jagged_to_padded_dense_forward(
      values,
      {offsets},
      c10::ArrayRef<c10::SymInt>({max_sequence_length}),
      /*padding_value=*/0);
}

namespace {

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
        c10::fromIntArrayRefKnownNonNegative(std::vector<int64_t>(
            dense_shape.begin() + 1, dense_shape.end() - 1)),
        /*padding_value=*/0);
    TORCH_CHECK(dense_values_grad.sizes() == dense_shape);

    return {
        grad_outputs[0],
        torch::autograd::Variable(), // offsets
        dense_values_grad};
  }
};
} // namespace

///@ingroup jagged-tensor-ops-cuda
/// output = x + y where x is jagged, y is dense, and output is jagged
std::tuple<Tensor, std::vector<Tensor>>
jagged_dense_elementwise_add_jagged_output_cuda(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  auto sum_values =
      JaggedDenseAddJaggedOutputGPUOp::apply(x_values, x_offsets, y)[0];

  return {sum_values, x_offsets};
}

} // namespace fbgemm_gpu

JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_to_padded_dense_forward",
    fbgemm_gpu::jagged_to_padded_dense_forward);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "stacked_jagged_1d_to_dense",
    fbgemm_gpu::stacked_jagged_1d_to_dense_gpu);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "stacked_jagged_2d_to_dense",
    fbgemm_gpu::stacked_jagged_2d_to_dense_gpu);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "stacked_jagged_2d_to_dense_forward",
    fbgemm_gpu::stacked_jagged_2d_to_dense_forward_cuda);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "stacked_jagged_2d_to_dense_backward",
    fbgemm_gpu::stacked_jagged_2d_to_dense_backward_cuda);
JAGGED_TENSOR_OPS_CUDA_DISPATCH(
    "jagged_dense_elementwise_add_jagged_output",
    fbgemm_gpu::jagged_dense_elementwise_add_jagged_output_cuda);
