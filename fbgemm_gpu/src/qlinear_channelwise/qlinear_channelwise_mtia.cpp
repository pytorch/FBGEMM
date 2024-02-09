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

static at::Tensor qlinear_channelwise(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor input_scale,
    at::Tensor weight_scale,
    at::Tensor weight_zero_point,
    at::Tensor relu) {
  // quantized linear function with
  // activation: per-tensor quantization
  // weight: per-tensor quantization
  return x;
}

static at::Tensor qlinear_quant(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor input_scale,
    at::Tensor weight_scale,
    at::Tensor weight_zero_point,
    at::Tensor relu) {
  assert(x.options().dtype() == at::kHalf);
  assert(weight.options().dtype() == at::kQInt8);
  assert(bias.options().dtype() == at::kFloat);
  assert(input_scale.options().dtype() == at::kFloat);
  assert(weight_scale.options().dtype() == at::kFloat);
  assert(weight_zero_point.options().dtype() == at::kQUInt8);
  return x;
}

static at::Tensor qlinear_qparams(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor input_scale,
    at::Tensor weight_scale,
    at::Tensor weight_zero_point,
    at::Tensor relu) {
  assert(x.options().dtype() == at::kHalf);
  assert(weight.options().dtype() == at::kQInt8);
  assert(bias.options().dtype() == at::kFloat);
  assert(input_scale.options().dtype() == at::kFloat);
  assert(weight_scale.options().dtype() == at::kFloat);
  assert(weight_zero_point.options().dtype() == at::kQUInt8);
  return x;
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
}
