/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off

#pragma once
#include <stdexcept>
#include <cutlass/cutlass.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/library/descriptions.h>
#include <cutlass/library/library.h>
#include <cutlass/library/types.h>
#include <cutlass_extensions/include/utils.h>
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_extensions {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Description of all GEMM computations
struct GemmDescription : public cutlass::library::OperationDescription {

  /// Indicates the kind of GEMM performed
  cutlass::library::GemmKind gemm_kind;
  
  /// Describes the A operand
  cutlass::library::TensorDescription A;

  /// Describes the B operand
  cutlass::library::TensorDescription B;

  /// Describes the source matrix
  cutlass::library::TensorDescription C;

  /// Describes the destination matrix
  cutlass::library::TensorDescription D;

  /// Describes the data type of the scalars passed to the epilogue
  cutlass::library::NumericTypeID element_epilogue;

  /// Describes the fusion kind followed by the Gemm
  cutlass_extensions::FusionKind fusion_kind;

  /// Describes the accumulation kind (none, slow, fast)
  cutlass_extensions::AccumKind accum_kind;

  /// Describes the mainloop schedule
  cutlass_extensions::MainloopSchedule mainloop_schedule;

  /// Describes the epilogue schedule
  cutlass_extensions::EpilogueSchedule epilogue_schedule;

  //
  // Methods
  //

  GemmDescription(
    cutlass::library::GemmKind gemm_kind = cutlass::library::GemmKind::kGemm,
    cutlass::library::TensorDescription const& A = cutlass::library::TensorDescription(),
    cutlass::library::TensorDescription const& B = cutlass::library::TensorDescription(),
    cutlass::library::TensorDescription const& C = cutlass::library::TensorDescription(),
    cutlass::library::TensorDescription const& D = cutlass::library::TensorDescription(),
    cutlass::library::NumericTypeID element_epilogue = cutlass::library::NumericTypeID::kInvalid,
    cutlass_extensions::FusionKind fusion_kind = cutlass_extensions::FusionKind::kNone,
    cutlass_extensions::AccumKind accum_kind = cutlass_extensions::AccumKind::kDefault,
    cutlass_extensions::MainloopSchedule mainloop_schedule = cutlass_extensions::MainloopSchedule::kUnknown,
    cutlass_extensions::EpilogueSchedule epilogue_schedule = cutlass_extensions::EpilogueSchedule::kUnknown
  ):
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    fusion_kind(fusion_kind),
    accum_kind(accum_kind),
    mainloop_schedule(mainloop_schedule),
    epilogue_schedule(epilogue_schedule) {} 

  GemmDescription(
    cutlass::library::OperationDescription op_desc,
    cutlass::library::GemmKind gemm_kind,
    cutlass::library::TensorDescription const& A,
    cutlass::library::TensorDescription const& B,
    cutlass::library::TensorDescription const& C,
    cutlass::library::TensorDescription const& D,
    cutlass::library::NumericTypeID element_epilogue,
    cutlass_extensions::FusionKind fusion_kind,
    cutlass_extensions::AccumKind accum_kind,
    cutlass_extensions::MainloopSchedule mainloop_schedule,
    cutlass_extensions::EpilogueSchedule epilogue_schedule
  ):
    cutlass::library::OperationDescription(op_desc),
    gemm_kind(gemm_kind),
    A(A),
    B(B),
    C(C),
    D(D),
    element_epilogue(element_epilogue),
    fusion_kind(fusion_kind),
    accum_kind(accum_kind),
    mainloop_schedule(mainloop_schedule),
    epilogue_schedule(epilogue_schedule)  {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass_extensions

/////////////////////////////////////////////////////////////////////////////////////////////////
