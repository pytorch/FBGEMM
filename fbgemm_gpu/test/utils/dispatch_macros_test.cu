/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
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

#endif // USE_ROCM

} // namespace fbgemm_gpu
