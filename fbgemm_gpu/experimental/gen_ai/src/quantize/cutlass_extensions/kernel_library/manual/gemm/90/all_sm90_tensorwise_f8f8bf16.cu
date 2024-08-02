/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "cutlass_extensions/include/manifest.h"

namespace cutlass_extensions {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnt_align16_warpspecialized_epi_nosmem(Manifest &manifest);


//
// Entry point to construct operations
//
void initialize_all_sm90_tensorwise_f8f8bf16(Manifest &manifest) {
    initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnt_align16_warpspecialized_epi_nosmem(manifest);
}


///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass_extensions
