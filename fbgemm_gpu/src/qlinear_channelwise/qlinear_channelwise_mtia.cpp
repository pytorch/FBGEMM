/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include "torch/types.h"

static at::Tensor qlinear_channelwise(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor input_scale,
    at::Tensor weight_scale,
    at::Tensor weight_zero_point,
    at::Tensor relu) {
  // quantized linear function with
  // activation: per-tensor quantization,
  // weight: per-tensor quantization
  // X.sizes = M * K, W.sizes = N * K, Y.sizes = M * N
  // Input_scale.sizes = M, Weight_scale.sizes = 1, b.sizes = N
  at::Tensor X = x.contiguous();
  at::Tensor W = weight.contiguous();
  at::Tensor b = bias.contiguous();
  at::Tensor Weight_scale = weight_scale.contiguous();
  at::Tensor Weight_zero_point = weight_zero_point.contiguous();

  const auto x_dimensions = X.sizes();
  const int x_num_dim = x_dimensions.size();

  TORCH_CHECK(
      x_dimensions.back() == W.sizes().back(),
      "X's inner-most dimension must match W's inner-most dimension!");

  const int M = x_dimensions[0];
  const int K = x_dimensions[x_num_dim - 1];
  const int N = W.sizes()[0];

  const uint8_t* X_data = (const uint8_t*)X.contiguous().storage().data();
  const uint8_t* W_data = (const uint8_t*)W.contiguous().storage().data();
  const float* b_data = b.data_ptr<float>();
  const uint8_t* Weight_zero_point_data =
      (const uint8_t*)Weight_zero_point.contiguous().storage().data();

  // Matmul
  // X.sizes = M * K, W.sizes = N * K, Y.sizes = M * N
  std::vector<float> Y_fp_vec =
      std::vector<float>(M * (std::vector<float>::size_type)(N));
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int32_t matmul = 0;
      for (int k = 0; k < K; k++) {
        int x = X_data[m * K + k];
        int w = W_data[n * K + k];

        matmul += (x - 127) * (w - *Weight_zero_point_data);
      }
      Y_fp_vec[m * N + n] = matmul;
    }
  }

  // re-scale to fp & add bias
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float tmp_scale =
          input_scale[i].item().toFloat() * Weight_scale[0].item().toFloat();
      float Y_tmp = (Y_fp_vec[i * N + j] * tmp_scale) + b_data[j];
      if (Y_tmp > 65504.0f) {
        Y_tmp = 65504.0f;
      }
      Y_fp_vec[i * N + j] = Y_tmp;
    }
  }

  auto Y = at::from_blob(
      Y_fp_vec.data(), {M, N}, at::TensorOptions().dtype(torch::kFloat32));
  return Y;
}

static at::Tensor qlinear_quant(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor input_scale,
    at::Tensor weight_scale,
    at::Tensor weight_zero_point,
    at::Tensor relu) {
  at::Tensor X = x.contiguous();
  const float* x_data = X.data_ptr<float>();
  const int M = X.sizes()[0];
  const int K = X.sizes()[1];
  at::Tensor I_S = input_scale.contiguous();
  const float* input_scale_data = I_S.data_ptr<float>();

  std::vector<uint8_t> X_int8_vec =
      std::vector<uint8_t>(M * (std::vector<uint8_t>::size_type)(K));
  for (int m = 0; m < M; m++) {
    const float inv_scale = 1.0f / input_scale_data[m];
    for (int k = 0; k < K; k++) {
      int32_t val = int32_t(inv_scale * x_data[m * K + k]) + 127;
      X_int8_vec[m * K + k] = uint8_t(std::max(0, std::min(val, UINT8_MAX)));
    }
  }

  auto Y = at::from_blob(
      X_int8_vec.data(), {M, K}, at::TensorOptions().dtype(torch::kUInt8));
  return Y;
}

static at::Tensor qlinear_qparams(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor input_scale,
    at::Tensor weight_scale,
    at::Tensor weight_zero_point,
    at::Tensor relu) {
  at::Tensor X = x.contiguous();
  at::Tensor W = weight.contiguous();
  at::Tensor b = bias.contiguous();
  at::Tensor Input_scale = input_scale.contiguous();
  at::Tensor Weight_scale = weight_scale.contiguous();
  at::Tensor Weight_zero_point = weight_zero_point.contiguous();

  const auto x_dimensions = X.sizes();
  const int x_num_dim = x_dimensions.size();
  const float* X_data = (const float*)X.contiguous().storage().data();

  // X.sizes = M * K
  const int M = x_dimensions[0];
  const int K = x_dimensions[x_num_dim - 1];

  std::vector<float> Y_range_vec = std::vector<float>(M);
  for (int m = 0; m < M; m++) {
    float tmp_max = 0.0f;
    for (int k = 0; k < K; k++) {
      float x = X_data[m * K + k];
      float abs_x = std::abs(x);
      if (abs_x > tmp_max) {
        tmp_max = abs_x;
      }
    }
    Y_range_vec[m] = tmp_max / 127.0f;
  }

  auto Y = at::from_blob(
      Y_range_vec.data(), {M}, at::TensorOptions().dtype(torch::kFloat32));
  return Y;
}

static at::Tensor qlinear_dynamic(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor input_scale,
    at::Tensor weight_scale,
    at::Tensor weight_zero_point,
    at::Tensor relu) {
  input_scale = qlinear_qparams(
      x, weight, bias, input_scale, weight_scale, weight_zero_point, relu);
  x = qlinear_quant(
      x, weight, bias, input_scale, weight_scale, weight_zero_point, relu);
  return qlinear_channelwise(
      x, weight, bias, input_scale, weight_scale, weight_zero_point, relu);
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "qlinear_channelwise(Tensor x, Tensor weight, Tensor "
      "bias, Tensor input_scale, Tensor weight_scale, Tensor "
      "weight_zero_point, Tensor relu) -> Tensor");
  m.impl(
      "qlinear_channelwise",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(qlinear_channelwise)));

  m.def(
      "qlinear_quant(Tensor x, Tensor weight, Tensor "
      "bias, Tensor input_scale, Tensor weight_scale, Tensor "
      "weight_zero_point, Tensor relu) -> Tensor");

  m.impl(
      "qlinear_quant",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(qlinear_quant)));

  m.def(
      "qlinear_qparams(Tensor x, Tensor weight, Tensor "
      "bias, Tensor input_scale, Tensor weight_scale, Tensor "
      "weight_zero_point, Tensor relu) -> Tensor");

  m.impl(
      "qlinear_qparams",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(qlinear_qparams)));

  m.def(
      "qlinear_dynamic(Tensor x, Tensor weight, Tensor "
      "bias, Tensor input_scale, Tensor weight_scale, Tensor "
      "weight_zero_point, Tensor relu) -> Tensor");

  m.impl(
      "qlinear_dynamic",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(qlinear_dynamic)));
}
