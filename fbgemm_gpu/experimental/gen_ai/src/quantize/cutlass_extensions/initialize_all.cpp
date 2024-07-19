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

// Four f8f8bf16 instances used in legacy cutlass_extensions.cu (Row-Col-Col
// TensorWise Scaled Gemms)
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmem(
    Manifest& manifest);
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmem(
    Manifest& manifest);
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmem(
    Manifest& manifest);
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmem(
    Manifest& manifest);

// Additional f8f8bf16 instances (Row-Col-Row TensorWise Scaled Gemms)
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnt_align16_warpspecialized_epi_nosmem(
    Manifest& manifest);
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnt_align16_warpspecialized_fp8_fastaccum_epi_nosmem(
    Manifest& manifest);
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem(
    Manifest& manifest);
void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_fp8_fastaccum_epi_nosmem(
    Manifest& manifest);

/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
//            Initializers for different Gemm variants
/////////////////////////////////////////////////////////////////////////

/// Add Pure FP8 operations to the manifest
void initialize_fp8_gemm_operations(Manifest& manifest) {
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmem(
      manifest);
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_fp8_fastaccum_epi_nosmem(
      manifest);
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnn_align16_warpspecialized_epi_nosmem(
      manifest);
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnn_align16_warpspecialized_epi_nosmem(
      manifest);

  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnt_align16_warpspecialized_epi_nosmem(
      manifest);
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_64x128x128_2x1x1_0_tnt_align16_warpspecialized_fp8_fastaccum_epi_nosmem(
      manifest);
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem(
      manifest);
  initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_fp8_fastaccum_epi_nosmem(
      manifest);
}
/////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
//            Top-level manifest initializer call
/////////////////////////////////////////////////////////////////////////
void initialize_all(Manifest& manifest) {
  initialize_fp8_gemm_operations(manifest);
  // initialize_fp8_rowwise_gemm_operations(manifest);
}

} // namespace cutlass_extensions
