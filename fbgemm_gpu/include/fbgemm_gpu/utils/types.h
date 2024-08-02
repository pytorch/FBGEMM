/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace fbgemm_gpu {

using fint32 = union fint32 {
  uint32_t I;
  float F;
};

} // namespace fbgemm_gpu
