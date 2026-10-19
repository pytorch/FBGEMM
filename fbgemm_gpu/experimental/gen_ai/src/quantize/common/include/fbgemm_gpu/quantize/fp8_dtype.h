/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/detail/CUDAHooksInterface.h>

namespace fbgemm_gpu {

inline void check_ck_fp8_inputs(const at::Tensor& XQ, const at::Tensor& WQ) {
  TORCH_CHECK(XQ.is_cuda() && WQ.is_cuda(), "CK FP8 inputs must be on GPU");
  TORCH_CHECK(XQ.device() == WQ.device(), "CK FP8 inputs must share a device");
  TORCH_CHECK(
      at::detail::getCUDAHooks().isGPUArch(
          {"gfx942", "gfx950"}, XQ.get_device()),
      "Unsupported GPU architecture for CK FP8");
  const auto dtype =
      at::detail::getCUDAHooks().isGPUArch({"gfx950"}, XQ.get_device())
      ? at::kFloat8_e4m3fn
      : at::kFloat8_e4m3fnuz;
  TORCH_CHECK(
      XQ.scalar_type() == dtype && WQ.scalar_type() == dtype,
      "CK FP8 inputs must match the device native dtype: ",
      dtype);
}

} // namespace fbgemm_gpu
