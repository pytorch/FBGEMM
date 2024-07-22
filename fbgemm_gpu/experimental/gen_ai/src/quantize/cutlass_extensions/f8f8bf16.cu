/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/library/library.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

#include <cutlass_extensions/include/gemm_description.h>
#include <cutlass_extensions/include/manifest.h>
#include <cutlass_extensions/include/operation_table.h>
#include <cutlass_extensions/include/singleton.h>
#include <cutlass_extensions/include/utils.h>
#include <iostream>

namespace fbgemm_gpu {

class Fp8GemmTensorWiseHeuristics {
  //
  // Data members
  //

 private:
  // GemmPerformanceKey -> [operations]
  cutlass_extensions::GemmOperationPerformanceMap operation_map;

  //
  // Methods
  //

 private:
  /// Returns an operation ptr given a performance key.
  cutlass::library::Operation const* _find_operation(
      cutlass_extensions::GemmPerformanceKey const& key) const {
    auto operation_it = operation_map.find(key);
    if (operation_it == operation_map.end()) {
      std::stringstream ss;
      ss << "cutlass_extensions::GemmPerformanceKey key = ";
      cutlass_extensions::operator<<(ss, key) << std::endl;
      ss << "Not found in the GemmOperationPerformanceMap.";
      throw std::runtime_error(ss.str());
    }
    return operation_it->second[0];
  }

  /// Returns an operation based on a strict problem_shape -> operation mapping.
  cutlass::library::Operation const* _strict_heuristics(
      cutlass::gemm::GemmCoord problem_shape) const {
    /*
      TODO: For every problem shape we dispatch to the top-performning
      operation*
    */
    return nullptr;
  }

  /// Returns an operation based on a relaxed generalized heuristcs.
  cutlass::library::Operation const* _relaxed_heuristics(
      cutlass::gemm::GemmCoord problem_shape) const {
    if (problem_shape.n() <= 128) {
      cutlass_extensions::GemmPerformanceKey perf_key(
          {64, 128, 32}, // instruction shape
          {64, 128, 128}, // threadblock shape
          {2, 1, 1}, // cluster shape
          cutlass_extensions::MainloopSchedule::kWarpspecialized,
          cutlass_extensions::EpilogueSchedule::kUnknown);
      return _find_operation(perf_key);
    } else {
      cutlass_extensions::GemmPerformanceKey perf_key(
          {64, 128, 32}, // instruction shape
          {128, 128, 128}, // threadblock shape
          {1, 2, 1}, // cluster shape
          cutlass_extensions::MainloopSchedule::kWarpspecialized,
          cutlass_extensions::EpilogueSchedule::kUnknown);
      return _find_operation(perf_key);
    }
  }

 public:
  /// ctor
  Fp8GemmTensorWiseHeuristics(
      cutlass_extensions::GemmOperationPerformanceMap operation_map)
      : operation_map(operation_map) {}

  /// Returns a operation give a gemm problem shape.
  cutlass::library::Operation const* operator()(
      cutlass::gemm::GemmCoord problem_shape) const {
    if (auto const* operation = _strict_heuristics(problem_shape))
      return operation;
    else
      return _relaxed_heuristics(problem_shape);
  }
};

/// FP8 Gemm with tensor wise scaling.
/// Input A   : MxK matrix in FP8 and RowMajor layout.
/// Input B   : KxN matrix in FP8 and ColumnMajor layout.
/// Returns Y  : MxN matrix in FP8 and ColumnMajor layout.
at::Tensor f8f8bf16_tnn(
    at::Tensor A, // FP8, RowMajor
    at::Tensor B, // FP8, ColumnMajor
    at::Tensor scale,
    bool use_fast_accum) {
  // Get the gemm_operation_with_tensorwise operation table.
  auto& operation_map = cutlass_extensions::Singleton::get()
                            .operation_table.gemm_operations_with_tensorwise;

  // Select a gemm operation with tensorwise scaling.
  cutlass_extensions::AccumKind accum_kind = (use_fast_accum)
      ? (cutlass_extensions::AccumKind::kFastAccum)
      : (cutlass_extensions::AccumKind::kDefault);

  cutlass_extensions::GemmFunctionalKey functional_key(
      cutlass::library::Provider::kCUTLASS, // Operator provider
      cutlass::library::GemmKind::kUniversal, // Gemm kind
      cutlass::library::NumericTypeID::kF32, // Accumulation type
      cutlass::library::NumericTypeID::kFE4M3, // ElementA
      cutlass::library::LayoutTypeID::kRowMajor, // LayoutA
      cutlass::library::NumericTypeID::kFE4M3, // ElementB
      cutlass::library::LayoutTypeID::kColumnMajor, // LayoutB
      cutlass::library::NumericTypeID::kBF16, // ElementC
      cutlass::library::LayoutTypeID::kColumnMajor, // LayoutC
      cutlass::library::NumericTypeID::kBF16, // ElementD
      cutlass::library::LayoutTypeID::kColumnMajor, // LayoutC
      cutlass_extensions::FusionKind::kTensorwiseScaling, // Scaling type
      accum_kind);

  auto operation_perf_it = operation_map.find(functional_key);

  if (operation_perf_it == operation_map.end()) {
    std::stringstream ss;
    ss << "GemmFunctionaKey key = ";
    cutlass_extensions::operator<<(ss, functional_key) << std::endl;
    ss << "Not found in the GemmOperationFunctionalMap.";
    throw std::runtime_error(ss.str());
  }

  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1);
  cutlass::gemm::GemmCoord problem_shape{M, N, K};

  // Query heuristics and select a gemm with tensorwise scaling operation.
  Fp8GemmTensorWiseHeuristics heur(operation_perf_it->second);
  auto const* operation = heur(problem_shape);
  if (!operation) {
    throw std::runtime_error("Heuristics returned null operation.");
  }

  // Output tensor
  auto Y = at::empty(
      {problem_shape.n(), problem_shape.m()}, A.options().dtype(at::kBFloat16));

  // Initialize configuration
  cutlass::library::GemmUniversalConfiguration configuration{
      cutlass::library::GemmUniversalMode::kGemm,
      problem_shape, // Gemm problem shape
      1, // Batch size
      problem_shape.k(), // lda (row-major)
      problem_shape.k(), // ldb (column-major)
      problem_shape.m(), // ldc (column-major)
      problem_shape.m() // ldd (column-major)
  };

  // Initialize arguments
  cutlass::library::GemmUniversalArguments arguments;
  arguments.problem_size = problem_shape;
  arguments.batch_count = 1;
  arguments.lda = configuration.lda;
  arguments.ldb = configuration.ldb;
  arguments.ldc = configuration.ldc;
  arguments.ldd = configuration.ldd;
  arguments.batch_stride_A = problem_shape.mk().product();
  arguments.batch_stride_B = problem_shape.nk().product();
  arguments.batch_stride_C = problem_shape.mn().product();
  arguments.batch_stride_D = problem_shape.mn().product();

  arguments.A = reinterpret_cast<void const*>(A.data_ptr());
  arguments.B = reinterpret_cast<void const*>(B.data_ptr());
  arguments.C = reinterpret_cast<void*>(Y.data_ptr());
  arguments.D = reinterpret_cast<void*>(Y.data_ptr());
  arguments.alpha = reinterpret_cast<void const*>(scale.data_ptr());
  arguments.beta = nullptr;
  arguments.pointer_mode = cutlass::library::ScalarPointerMode::kDevice;
  arguments.sm_count = 132;
  arguments.raster_order = cutlass::library::RasterOrder::kHeuristic;

  // Buffer used for the operation's host workspace
  std::vector<uint8_t> host_workspace;
  uint64_t host_workspace_size =
      operation->get_host_workspace_size(&configuration);
  host_workspace.resize(host_workspace_size, 0);

  // Device workspace size
  uint64_t device_workspace_size =
      operation->get_device_workspace_size(&configuration, &arguments);

  // Allocate device workspace memory
  cutlass::device_memory::allocation<uint8_t> device_workspace(
      device_workspace_size);

  // Check if we can call this kernel
  cutlass::Status status = cutlass::Status::kSuccess;
  status = operation->can_implement(&configuration, &arguments);

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass operation can_implement failed.");
  }

  // Initialize host and device workspaces
  status = operation->initialize(
      &configuration,
      host_workspace.data(),
      device_workspace.get(),
      at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass operation initialize failed.");
  }

  // Run the Gemm operator
  status = operation->run(
      &arguments,
      host_workspace.data(),
      device_workspace.get(),
      at::cuda::getCurrentCUDAStream());

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass operation run failed.");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return Y;
}

/// FP8 Gemm with tensor wise scaling.
/// Input XQ   : MxK matrix in FP8 and RowMajor layout.
/// Input WQ   : KxN matrix in FP8 and ColumnMajor layout.
/// Returns Y  : MxN matrix in FP8 and RowMajor layout.
at::Tensor f8f8bf16_v2(
    at::Tensor XQ, // FP8, RowMajor
    at::Tensor WQ, // FP8, ColumnMajor
    at::Tensor scale,
    bool use_fast_accum) {
  // Swap and run the kernel with ColumnMajor output
  return f8f8bf16_tnn(WQ, XQ, scale, use_fast_accum);
}

} // namespace fbgemm_gpu
