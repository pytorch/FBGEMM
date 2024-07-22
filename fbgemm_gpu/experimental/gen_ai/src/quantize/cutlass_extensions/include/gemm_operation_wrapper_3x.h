/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// @lint-ignore-every LICENSELINT

// clang-format off
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/library/library.h"
#include "library_internal.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include <unordered_map>

#include <cutlass_extensions/include/gemm_description.h>
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass_extensions {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmOperationWrapper3xBase : public cutlass::library::Operation {
public:
  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  // assuming all tensors use same type for StrideIndex
  using StrideIndex = typename Operator::LayoutA::Index;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using KernelSchedule = typename Operator::GemmKernel::DispatchPolicy::Schedule;

private:
  cutlass_extensions::GemmDescription description_;

public:

  /// Constructor
  GemmOperationWrapper3xBase(char const *name = "unknown_gemm", 
                             cutlass::library::GemmKind gemm_kind_ = cutlass::library::GemmKind::kGemm) {

    description_.name = name;
    description_.provider = cutlass::library::Provider::kCUTLASS;
    description_.kind = cutlass::library::OperationKind::kGemm;
    description_.gemm_kind = gemm_kind_;

    description_.tile_description.threadblock_shape = cutlass::make_Coord(
      Operator::ThreadblockShape::kM,
      Operator::ThreadblockShape::kN,
      Operator::ThreadblockShape::kK);

    if constexpr (Operator::ArchTag::kMinComputeCapability >= 90) {
      description_.tile_description.cluster_shape = cutlass::make_Coord(
        Operator::ClusterShape::kM,
        Operator::ClusterShape::kN,
        Operator::ClusterShape::kK);
    }

    description_.tile_description.threadblock_stages = Operator::kStages;

    description_.tile_description.warp_count = cutlass::make_Coord(
      Operator::WarpCount::kM,
      Operator::WarpCount::kN,
      Operator::WarpCount::kK);

    description_.tile_description.math_instruction.instruction_shape = cutlass::make_Coord(
      Operator::InstructionShape::kM,
      Operator::InstructionShape::kN,
      Operator::InstructionShape::kK);

    description_.tile_description.math_instruction.element_accumulator =
      cutlass::library::NumericTypeMap<ElementAccumulator>::kId;

    description_.tile_description.math_instruction.opcode_class =
      cutlass::library::OpcodeClassMap<typename Operator::OperatorClass>::kId;

    description_.tile_description.math_instruction.math_operation =
      cutlass::library::MathOperationMap<typename Operator::MathOperator>::kId;

    description_.tile_description.minimum_compute_capability =
      cutlass::library::ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMin;

    description_.tile_description.maximum_compute_capability =
      cutlass::library::ArchMap<typename Operator::ArchTag, typename Operator::OperatorClass>::kMax;

    description_.A = cutlass::library::make_TensorDescription<ElementA, LayoutA>(Operator::kAlignmentA);
    description_.B = cutlass::library::make_TensorDescription<ElementB, LayoutB>(Operator::kAlignmentB);
    description_.C = cutlass::library::make_TensorDescription<ElementC, LayoutC>(Operator::kAlignmentC);
    description_.D = cutlass::library::make_TensorDescription<ElementD, LayoutD>(Operator::kAlignmentD);
    description_.element_epilogue = cutlass::library::NumericTypeMap<ElementCompute>::kId;


    description_.fusion_kind = cutlass_extensions::FusionKind::kNone;
    description_.accum_kind = cutlass_extensions::KernelScheduleMap<KernelSchedule>::kAccumKind;
    description_.mainloop_schedule = cutlass_extensions::KernelScheduleMap<KernelSchedule>::kMainloopSchedule;
    description_.epilogue_schedule = cutlass_extensions::KernelScheduleMap<KernelSchedule>::kEpilogueSchedule;

    if ((description_.A.element == cutlass::library::NumericTypeID::kFE4M3) || 
        (description_.A.element == cutlass::library::NumericTypeID::kFE5M2) ||
        (description_.B.element == cutlass::library::NumericTypeID::kFE4M3) || 
        (description_.B.element == cutlass::library::NumericTypeID::kFE5M2)) {
      
      // FP8 by default atleast needs tensor wise scaling factor.
      // For Rowwise and Blockwise scaling, `fusion_kind` field is set in 
      // the respective inherited classes.
      description_.fusion_kind = cutlass_extensions::FusionKind::kTensorwiseScaling;
    }
    
  }

  /// Returns the description of the GEMM operation
  virtual cutlass::library::OperationDescription const & description() const {
    return description_;
  }

  /// Returns the description of the GEMM operation
  cutlass_extensions::GemmDescription const& get_gemm_description() const {
    return description_;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Operator_>
class GemmOperationWrapper3x : public GemmOperationWrapper3xBase<Operator_> {
public:

  using Operator = Operator_;
  using OperatorArguments = typename Operator::Arguments;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = typename Operator::ElementD;
  using LayoutD = typename Operator::LayoutD;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using CollectiveMainloop = typename Operator::CollectiveMainloop;
  using CollectiveEpilogue = typename Operator::CollectiveEpilogue;
  using ThreadEpilogueOp = typename CollectiveEpilogue::ThreadEpilogueOp;

public:

  /// Constructor
  GemmOperationWrapper3x(char const *name = "unknown_gemm"):
    GemmOperationWrapper3xBase<Operator_>(name, cutlass::library::GemmKind::kUniversal) {}

protected:

  /// Constructs the arguments structure given the configuration and arguments
  static cutlass::Status construct_arguments_(
      OperatorArguments &operator_args, cutlass::library::GemmUniversalConfiguration const *configuration) {
    // NOTE: GemmUniversalConfiguration does not contain problem shapes or batch strides
    // Do nothing here and construct kernel arguments in update_arguments_ instead
    // We also cannot construct TMA descriptors without all the arguments available

    operator_args.mode = configuration->mode;
    return cutlass::Status::kSuccess;
  }

  template<class FusionArgs, class = void>
  struct UpdateFusionArgs {
    static cutlass::Status update_(FusionArgs const& fusion_args, cutlass::library::GemmUniversalArguments const &arguments) {
      // If a custom EVT is instantiated then it is the users's responsibility
      // to ensure alpha and beta are updated appropriately
      return cutlass::Status::kSuccess;
    }
  };

  template<class FusionArgs>
  struct UpdateFusionArgs<FusionArgs, cute::void_t<decltype(FusionArgs{}.alpha)>> {
    static cutlass::Status update_(FusionArgs& fusion_args, cutlass::library::GemmUniversalArguments const &arguments) {
      if (arguments.pointer_mode == cutlass::library::ScalarPointerMode::kHost) {
        fusion_args.alpha = *static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta = *static_cast<ElementCompute const *>(arguments.beta);
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;

        return cutlass::Status::kSuccess;
      }
      else if (arguments.pointer_mode == cutlass::library::ScalarPointerMode::kDevice) {
        fusion_args.alpha = 0;
        fusion_args.beta = 0;
        fusion_args.alpha_ptr = static_cast<ElementCompute const *>(arguments.alpha);
        fusion_args.beta_ptr = static_cast<ElementCompute const *>(arguments.beta);

        return cutlass::Status::kSuccess;
      }
      else {
        return cutlass::Status::kErrorInvalidProblem;
      }
    }
  };

  /// Constructs the arguments structure given the configuration and arguments
  static cutlass::Status update_arguments_(
      OperatorArguments &operator_args,
      cutlass::library::GemmUniversalArguments const *arguments) {
    cutlass::Status status = cutlass::Status::kSuccess;

    status = UpdateFusionArgs<decltype(operator_args.epilogue.thread)>::update_(
      operator_args.epilogue.thread, *arguments);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    // TODO: type erase Arguments structure in 3.0 GEMM
    operator_args.problem_shape = cute::make_shape(
      arguments->problem_size.m(),
      arguments->problem_size.n(),
      arguments->problem_size.k(),
      arguments->batch_count);

    // update arguments
    operator_args.mainloop.ptr_A = static_cast<ElementA const *>(arguments->A);
    operator_args.mainloop.ptr_B = static_cast<ElementB const *>(arguments->B);
    operator_args.epilogue.ptr_C = static_cast<ElementC const *>(arguments->C);
    operator_args.epilogue.ptr_D = static_cast<ElementD       *>(arguments->D);

    operator_args.mainloop.dA = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideA>(
        arguments->lda, arguments->batch_stride_A);
    operator_args.mainloop.dB = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideB>(
        arguments->ldb, arguments->batch_stride_B);
    operator_args.epilogue.dC = cute::make_int_tuple_from<typename Operator::GemmKernel::StrideC>(
        arguments->ldc, arguments->batch_stride_C);
    operator_args.epilogue.dD = operator_args.epilogue.dC;

    /* Query device SM count to pass onto the kernel as an argument, where needed */
    operator_args.hw_info.sm_count = arguments->sm_count;

    if constexpr (!std::is_const_v<decltype(operator_args.scheduler.raster_order)>) {
      using Enum_t = decltype(operator_args.scheduler.raster_order);
      switch (arguments->raster_order) {
        case cutlass::library::RasterOrder::kAlongN:
          operator_args.scheduler.raster_order = Enum_t::AlongN;
          break;
        case cutlass::library::RasterOrder::kAlongM:
          operator_args.scheduler.raster_order = Enum_t::AlongM;
          break;
        default:
          operator_args.scheduler.raster_order = Enum_t::Heuristic;
      }
    }

    return status;
  }

public:

  /// Returns success if the operation can proceed
  cutlass::Status can_implement(
      void const *configuration_ptr, void const *arguments_ptr) const override {
    cutlass::library::GemmUniversalConfiguration const *configuration =
      static_cast<cutlass::library::GemmUniversalConfiguration const *>(configuration_ptr);
    cutlass::library::GemmUniversalArguments const *arguments =
      static_cast<cutlass::library::GemmUniversalArguments const *>(arguments_ptr);

    OperatorArguments args;
    auto status = update_arguments_(args, arguments);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    // can_implement rules may need access to problem shape
    args.problem_shape = cute::make_shape(
      configuration->problem_size.m(),
      configuration->problem_size.n(),
      configuration->problem_size.k(),
      configuration->batch_count);
    return Operator::can_implement(args);
  }

  /// Gets the host-side workspace
  uint64_t get_host_workspace_size(void const *configuration) const override {
    return sizeof(Operator);
  }

  /// Gets the device-side workspace
  uint64_t get_device_workspace_size(
      void const *configuration_ptr,void const *arguments_ptr) const override {

    OperatorArguments args;
    auto status = update_arguments_(
      args, static_cast<cutlass::library::GemmUniversalArguments const *>(arguments_ptr));
    if (status != cutlass::Status::kSuccess) {
      return 0;
    }

    uint64_t size = Operator::get_workspace_size(args);
    return size;
  }

  /// Initializes the workspace
  cutlass::Status initialize(
      void const *configuration_ptr,
      void *host_workspace,
      void *device_workspace,
      cudaStream_t stream = nullptr) const override {
    Operator *op = new (host_workspace) Operator;
    return cutlass::Status::kSuccess;
  }

  /// Runs the kernel
  cutlass::Status run(
      void const *arguments_ptr,
      void *host_workspace,
      void *device_workspace = nullptr,
      cudaStream_t stream = nullptr) const override {

    OperatorArguments args;
    cutlass::Status status = update_arguments_(args, static_cast<cutlass::library::GemmUniversalArguments const *>(arguments_ptr));
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    Operator *op = static_cast<Operator *>(host_workspace);
    // We need to call initialize() since we have to rebuild TMA desc for every new set of args
    status = op->run(args, device_workspace, stream);
    return status;
  }
};
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass_extensions

///////////////////////////////////////////////////////////////////////////////////////////////////
