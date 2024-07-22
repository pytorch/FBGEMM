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

#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

// cutlass_extensions includes
#include <cutlass_extensions/include/manifest.h>
#include <cutlass_extensions/include/gemm_operation_wrapper_3x.h>

///////////////////////////////////////////////////////////////////////////////////////////////////


using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 1,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 1,
    cutlass::epilogue::NoSmemWarpSpecialized,

    cutlass::epilogue::fusion::LinearCombination<
      cutlass::bfloat16_t,
      float,
      cutlass::bfloat16_t,
      float
    >

  >::CollectiveOp;

using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    float,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_1, cute::_2, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem_epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem
using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem_mainloop,
    cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem :
  public cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem_base { };



///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_extensions {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem(Manifest &manifest) {



  {
    using GemmKernel = cutlass::gemm::device::GemmUniversalAdapter<cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem>;
    manifest.append(
      new cutlass_extensions::GemmOperationWrapper3x<GemmKernel>("cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_0_tnt_align16_warpspecialized_epi_nosmem"));
  }



}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass_extensions

///////////////////////////////////////////////////////////////////////////////////////////////////
