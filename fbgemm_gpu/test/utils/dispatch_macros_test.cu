/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>

#include <string>

#include "fbgemm_gpu/embedding_common.h"
#include "fbgemm_gpu/utils/dispatch_macros.h"

namespace fbgemm_gpu {

// Only the fnuz emb_t variant of the TBE kernels is instantiated, so on archs
// where getNFP8ScalarType() labels NFP8 weights "fn" (gfx950, CUDA) every host
// entry point must call relabel_nfp8_for_dispatch() before dispatching. See the
// "NFP8 dtype flow" block in embedding_common.h.
//
// If that call is missed, the tensor must be rejected here, by name, rather
// than routed into the fnuz instantiation to die further downstream inside
// TensorAccessorBuilder or data_ptr<emb_t>() with a bare scalar-type mismatch
// that points at the symptom instead of the omission.
//
// This is only meaningful on ROCm: on CUDA fp8_e4m3_t *is* Float8_e4m3fn, so
// the label is the dispatchable one and there is nothing to reject.
#ifdef USE_ROCM

namespace {

// Keep this independent from getNFP8ScalarType() so the validation tests catch
// regressions in that helper's architecture mapping.
at::ScalarType expectedNFP8TypeForDevice(const c10::DeviceIndex device_index) {
  const std::string arch =
      at::cuda::getDeviceProperties(device_index)->gcnArchName;
  return arch.starts_with("gfx94") || arch.starts_with("gfx90a")
      ? at::kFloat8_e4m3fnuz
      : at::kFloat8_e4m3fn;
}

} // namespace

TEST(DispatchMacrosTest, UnrelabeledNFP8IsRejectedByName) {
  bool invoked = false;
  const auto dispatch = [&]() {
    dispatch_emb_cache_types(
        at::kFloat8_e4m3fn,
        at::kFloat,
        "test_kernel",
        [&]<typename emb_t, typename cache_t>() { invoked = true; });
  };

  EXPECT_THROW(dispatch(), c10::Error);
  EXPECT_FALSE(invoked) << "an un-relabeled fn tensor must not reach a kernel";

  try {
    dispatch();
    FAIL() << "expected dispatch to throw";
  } catch (const c10::Error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("relabel_nfp8_for_dispatch"), std::string::npos)
        << "the error must name the helper the caller forgot; got: " << msg;
  }
}

// The label the kernels are actually instantiated for still dispatches.
TEST(DispatchMacrosTest, FnuzNFP8Dispatches) {
  bool invoked = false;
  dispatch_emb_cache_types(
      at::kFloat8_e4m3fnuz,
      at::kFloat,
      "test_kernel",
      [&]<typename emb_t, typename cache_t>() { invoked = true; });
  EXPECT_TRUE(invoked);
}

TEST(DispatchMacrosTest, CorrectNFP8DeviceLabelIsAccepted) {
  if (!at::detail::getCUDAHooks().hasCUDA()) {
    GTEST_SKIP() << "No GPU available";
  }
  const auto expected_type = expectedNFP8TypeForDevice(0);
  EXPECT_EQ(getNFP8ScalarType(0), expected_type);
  const auto tensor = at::empty(
      {1},
      at::TensorOptions()
          .device(at::Device(at::kCUDA, 0))
          .dtype(expected_type));

  const auto dispatch_tensor = relabel_nfp8_for_dispatch(tensor);

  EXPECT_EQ(dispatch_tensor.scalar_type(), at::kFloat8_e4m3fnuz);
  EXPECT_EQ(dispatch_tensor.data_ptr(), tensor.data_ptr());
  EXPECT_EQ(
      dispatch_tensor.is_same(tensor), expected_type == at::kFloat8_e4m3fnuz);
}

TEST(DispatchMacrosTest, WrongNFP8DeviceLabelIsRejected) {
  if (!at::detail::getCUDAHooks().hasCUDA()) {
    GTEST_SKIP() << "No GPU available";
  }
  const auto expected_type = expectedNFP8TypeForDevice(0);
  EXPECT_EQ(getNFP8ScalarType(0), expected_type);
  const auto wrong_type = expected_type == at::kFloat8_e4m3fn
      ? at::kFloat8_e4m3fnuz
      : at::kFloat8_e4m3fn;
  const auto tensor = at::empty(
      {1},
      at::TensorOptions().device(at::Device(at::kCUDA, 0)).dtype(wrong_type));

  try {
    relabel_nfp8_for_dispatch(tensor);
    FAIL() << "expected a mismatched NFP8 device label to throw";
  } catch (const c10::Error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("but this device requires"), std::string::npos)
        << "the error must come from the device-label check; got: " << msg;
  }
}

TEST(DispatchMacrosTest, CpuNFP8TensorIsRelabeledForDispatch) {
  const auto tensor = at::empty(
      {1}, at::TensorOptions().device(at::kCPU).dtype(at::kFloat8_e4m3fn));

  const auto dispatch_tensor = relabel_nfp8_for_dispatch(tensor);

  EXPECT_EQ(dispatch_tensor.scalar_type(), at::kFloat8_e4m3fnuz);
  EXPECT_EQ(dispatch_tensor.data_ptr(), tensor.data_ptr());
}

#endif // USE_ROCM

} // namespace fbgemm_gpu
