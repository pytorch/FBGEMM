/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

// Each block handles a single batch and head
// Each warp handles separate D dimension.

// Load Q into registers in all warps.
// Split T across warps in a block
// Compute S[MAX_T] = for i in range(T): S[t] = sum(Q[d] * K[t, d])
// Use shared reduction to compute max and compute softmax on shared memory.

// Split T across warps in a block
// Each warp compute sum(t_subset) P[t] * V[t_subset, d]
// outputs are of size float[D]

// Wrapper to allow passing alpha/beta scaling params
// as device pointers.
namespace cutlass::epilogue::thread {

template <
    typename ElementOutput_, ///< Data type used to load and store tensors
    int Count, ///< Number of elements computed per operation.
               ///< Usually it is 128/sizeof_bits<ElementOutput_>,
               ///< but we use 64 or 32 sometimes when there are not enough data
               ///< to store
    typename ElementAccumulator_ = ElementOutput_, ///< Accumulator data type
    typename ElementCompute_ =
        ElementOutput_, ///< Data type used to compute linear combination
    ScaleType::Kind Scale =
        ScaleType::Default, ///< Control Alpha and Beta scaling
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
class LinearCombinationOnDevice {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static int const kCount = Count;
  static const ScaleType::Kind kScale = Scale;
  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kCount>;
  using ComputeFragment = Array<ElementCompute, kCount>;

  using ParamsBase = LinearCombinationParams;

  static FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params : ParamsBase {
    ElementCompute alpha; ///< scales accumulators
    ElementCompute beta; ///< scales source tensor
    ElementCompute const* alpha_ptr; ///< pointer to accumulator scalar - if not
                                     ///< null, loads it from memory
    ElementCompute const* beta_ptr; ///< pointer to source scalar - if not null,
                                    ///< loads it from memory

    CUTLASS_HOST_DEVICE
    Params()
        : ParamsBase(ElementCompute(1), ElementCompute(0)),
          alpha(ElementCompute(1)),
          beta(ElementCompute(0)),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : ParamsBase(alpha, beta),
          alpha(alpha),
          beta(beta),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha)
        : ParamsBase(alpha, ElementCompute(0)),
          alpha(alpha),
          beta(0),
          alpha_ptr(nullptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr)
        : ParamsBase(*alpha_ptr, *beta_ptr),
          alpha(0),
          beta(0),
          alpha_ptr(alpha_ptr),
          beta_ptr(beta_ptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr)
        : ParamsBase(ElementCompute(1), ElementCompute(0)),
          alpha(0),
          beta(0),
          alpha_ptr(alpha_ptr),
          beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ParamsBase const& base)
        : ParamsBase(base), alpha_ptr(nullptr), beta_ptr(nullptr) {
#if defined(__CUDA_ARCH__)
      alpha = reinterpret_cast<ElementCompute const&>(base.alpha_data);
      beta = reinterpret_cast<ElementCompute const&>(base.beta_data);
#else
      memcpy(alpha, base.alpha_data, sizeof(ElementCompute));
      memcpy(beta, base.alpha_data, sizeof(ElementCompute));
#endif
    }
  };

 private:
  //
  // Data members
  //

  const ElementCompute* alpha_ptr_;
  ElementCompute beta_;

 public:
  /// Constructs the function object, possibly loading from pointers in host
  /// memory
  CUTLASS_HOST_DEVICE
  LinearCombinationOnDevice(Params const& params) {
    alpha_ptr_ = params.alpha_ptr;
    beta_ = ElementCompute(0);
    // beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    if (Scale == ScaleType::NoBetaScaling)
      return true;

    if (Scale == ScaleType::OnlyAlphaScaling)
      return false;

    if (Scale == ScaleType::Nothing)
      return false;

    return beta_ != ElementCompute(0);
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
  }

  /// Computes linear scaling: D = alpha * accumulator + beta * source
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& accumulator,
      FragmentOutput const& source) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
        source_converter;
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    ComputeFragment converted_source = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    if (Scale == ScaleType::Nothing)
      return destination_converter(converted_accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    multiplies<ComputeFragment> mul_add_source;
    multiply_add<ComputeFragment> mul_add_accumulator;

    if (Scale == ScaleType::NoBetaScaling)
      intermediate = converted_source;
    else
      intermediate =
          mul_add_source(beta_, converted_source); // X =  beta * C + uniform

    intermediate = mul_add_accumulator(
        *alpha_ptr_,
        converted_accumulator,
        intermediate); // D = alpha * Accum + X

    return destination_converter(intermediate);
  }

  /// Computes linear scaling: D = alpha * accumulator
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
    // Convert source to interal compute numeric type
    NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    // Convert to destination numeric type
    NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    if (Scale == ScaleType::Nothing)
      return destination_converter(converted_accumulator);

    // Perform binary operations
    ComputeFragment intermediate;
    multiplies<ComputeFragment> mul_accumulator;

    intermediate = mul_accumulator(
        *alpha_ptr_, converted_accumulator); // D = alpha * Accum

    return destination_converter(intermediate);
  }
};

} // namespace cutlass::epilogue::thread
