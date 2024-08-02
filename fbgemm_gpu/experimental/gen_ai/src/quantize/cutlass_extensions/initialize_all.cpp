/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cutlass_extensions/include/manifest.h>

namespace cutlass_extensions {
/////////////////////////////////////////////////////////////////////////
//                     Declarations
/////////////////////////////////////////////////////////////////////////

// Auto-generate all SM90 GEMM operations
void initialize_all_sm90__gemm_operations(Manifest& manifest);

// Manually add all tensorwise quantization operations
void initialize_all_sm90_tensorwise_f8f8bf16(Manifest& manifest);

/////////////////////////////////////////////////////////////////////////
//            Top-level manifest initializer call
/////////////////////////////////////////////////////////////////////////
void initialize_all(Manifest& manifest) {
  // Add auto-generated kernel instances to manifest
  initialize_all_sm90__gemm_operations(manifest);

  // Add manual kernel instances to manifest
  initialize_all_sm90_tensorwise_f8f8bf16(manifest);
}

} // namespace cutlass_extensions
