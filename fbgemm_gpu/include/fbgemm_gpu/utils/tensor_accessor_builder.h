/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include "fbgemm_gpu/utils/tensor_accessor.h"

namespace fbgemm_gpu::utils {

#ifdef FBGEMM_GPU_MEMCHECK
namespace pta = fbgemm_gpu::utils;
#else
namespace pta = at;
#endif

//////////////////////////////////////////////////////////////////////////////
// Type Utilities
//////////////////////////////////////////////////////////////////////////////

template <typename T>
inline at::ScalarType scalar_type_for() {
#define TYPE_CASE(U, name)              \
  if constexpr (std::is_same_v<T, U>) { \
    return at::ScalarType::name;        \
  }

  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(TYPE_CASE)
#undef TYPE_CASE

  return at::ScalarType::Undefined;
}

////////////////////////////////////////////////////////////////////////////////
/// Tensor Accessor Builder
///
/// This thin wrapper class is used to build out a pta::*TensorAccessor.
/// Depending on the template parameters, it will perform various checks on the
/// tensor before constructing a pta::TensorAccessor or
/// pta::PackedTensorAccessor with the appropriate template arguments.
///
/// Usage:
///
///   ```cpp
///   at::Tensor tensor = ...;
///
///   // Assemble the tensor accessor builder
///   const auto x = TensorAccessorBuilder<T, N, INB, false>("name", tensor);
///   // Build a packed tensor accessor
///   const auto y = x.build("context");
///
///   // Using the utility macros to construct the builder(recommended)
///   const auto x = TA_B(tensor, T, N, INB);
///   const auto x = PTA_B(tensor, T, N, INB);
///   ```
///
/// NOTE: When logging debug information and assertion errors, we would like to
/// log the name of the relevant kernel function as well as the tensor variable
/// name.  Because C++ does not support reflection, we have collect this
/// information through C macros in two locations - at tensor accessor creation
/// site, and at the kernel function call site.  The TensorAccessorBuilder
/// abstraction and the separation of tensor accessor creation into the
/// creation and execution of the builder is meant to facilitate this.
////////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    size_t N,
    size_t index_nbits = 64,
    bool packed = true,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
struct TensorAccessorBuilder {
  //////////////////////////////////////////////////////////////////////////////
  // Static Assertions
  //////////////////////////////////////////////////////////////////////////////

  static_assert(
      N > 0,
      "Accessor is used for indexing tensor; for scalars use *data_ptr<T>() instead!");

  static_assert(
      index_nbits == 32 || index_nbits == 64,
      "index_nbits must be 32 or 64!");

  //////////////////////////////////////////////////////////////////////////////
  // Attributes
  //////////////////////////////////////////////////////////////////////////////

  const std::string_view name;
  const at::Tensor& tensor;

  //////////////////////////////////////////////////////////////////////////////
  // Type Aliases
  //////////////////////////////////////////////////////////////////////////////

  using index_t = std::conditional_t<index_nbits == 64, int64_t, int32_t>;
  using accessor_t = std::conditional_t<
      packed,
      pta::PackedTensorAccessor<T, N, PtrTraits, index_t>,
      pta::TensorAccessor<T, N, PtrTraits, index_t>>;

  //////////////////////////////////////////////////////////////////////////////
  // Constructor that takes in reference to tensor
  //////////////////////////////////////////////////////////////////////////////

  constexpr inline TensorAccessorBuilder(
      const std::string_view& name_,
      const at::Tensor& tensor_) noexcept
      : name(name_), tensor(tensor_) {}

  //////////////////////////////////////////////////////////////////////////////
  // Validate Tensor Properties
  //////////////////////////////////////////////////////////////////////////////

  inline void validate_tensor(const std::string_view& context) const {
    // Check numel is not out of bounds
    if constexpr (std::is_same_v<index_t, int32_t>) {
      TORCH_CHECK(
          tensor.numel() <=
              static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
          context,
          ": Tensor '",
          name,
          "' ",
          "numel needs to be smaller than int32_t max; otherwise, please use [packed_]accessor64");
    }

    // Check tensor dimensions
    TORCH_CHECK(
        tensor.dim() == N,
        context,
        ": Expected tensor '",
        name,
        "' to have ",
        N,
        " dims, but found ",
        tensor.dim(),
        " instead!");

    // Check tensor's scalar type works with T
    const auto expected_type = scalar_type_for<T>();
    TORCH_CHECK(
        tensor.scalar_type() == expected_type ||
            (isQIntType(tensor.scalar_type()) &&
             toUnderlying(tensor.scalar_type()) == expected_type),
        context,
        ": Expected tensor '",
        name,
        "' to have scalar type ",
        expected_type,
        ", but found ",
        tensor.scalar_type(),
        " instead!");
  }

  //////////////////////////////////////////////////////////////////////////////
  // Build out pta::TensorAccessor
  //////////////////////////////////////////////////////////////////////////////

  C10_ALWAYS_INLINE pta::TensorAccessor<T, N, PtrTraits, index_t> build_ta(
      const std::string_view& context) const {
    if (tensor.defined()) {
      validate_tensor(context);

#ifdef FBGEMM_GPU_MEMCHECK
      return fbgemm_gpu::utils::TensorAccessor<T, N, PtrTraits, index_t>(
          static_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr<T>()),
          tensor.sizes().data(),
          tensor.strides().data(),
          name.data(),
          context.data());
#else
      return tensor.accessor<T, N>();
#endif

    } else {
#ifdef FBGEMM_GPU_MEMCHECK
      return fbgemm_gpu::utils::TensorAccessor<T, N, PtrTraits, index_t>(
          nullptr, nullptr, nullptr, name.data(), context.data());
#else
      return at::TensorAccessor<T, N, PtrTraits, index_t>(
          nullptr, nullptr, nullptr);
#endif
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Build out pta::PackedTensorAccessor
  //////////////////////////////////////////////////////////////////////////////

  C10_ALWAYS_INLINE pta::PackedTensorAccessor<T, N, PtrTraits, index_t>
  build_pta(const std::string_view& context) const {
    validate_tensor(context);

#ifdef FBGEMM_GPU_MEMCHECK
    return fbgemm_gpu::utils::PackedTensorAccessor<T, N, PtrTraits, index_t>(
        static_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr<T>()),
        tensor.sizes().data(),
        tensor.strides().data(),
        name.data(),
        context.data());
#else
    return tensor.generic_packed_accessor<T, N, PtrTraits, index_t>();
#endif
  }

  //////////////////////////////////////////////////////////////////////////////
  // Build out pta::*TensorAccessor depending on `packed` template  parameter
  //////////////////////////////////////////////////////////////////////////////

  C10_ALWAYS_INLINE accessor_t build(const std::string_view& context) const {
    if constexpr (packed) {
      return build_pta(context);
    } else {
      return build_ta(context);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Check Tensor values for NaN
  //////////////////////////////////////////////////////////////////////////////

  C10_ALWAYS_INLINE void checkValues(const std::string_view& context) const {
    TORCH_CHECK(
        !at::isnan(tensor).any().item<bool>(),
        context,
        ": Tensor '",
        name,
        "' contains NaN values!");

    TORCH_CHECK(
        !at::isinf(tensor).any().item<bool>(),
        context,
        ": Tensor '",
        name,
        "' contains (+/-) Inf values!");
  }
};

} // namespace fbgemm_gpu::utils

//////////////////////////////////////////////////////////////////////////////
// (Packed) Tensor Accessor Builder-Assembler Macros
//
// These macros are used to assemble the TensorAccessorBuilder.  The primary
// purpose for using the macros instead of calling the class constructors
// directly is to be able to automatically capture the source variable name of
// the tensor.
//
// Usage:
//
//  ```cpp
//  at::Tensor tensor = ...;
//
//  // Assemble a builder for TensorAccessor<T, N, INB, at::DefaultPtrTraits>
//  const auto x = TA_B(tensor, T, N, INB);
//
//  // Assemble a builder for PackedTensorAccessor<T, N, INB,
//  // at::RestrictPtrTraits>
//  const auto x = PTA_B(tensor, T, N, INB);
//  ```
//////////////////////////////////////////////////////////////////////////////

#define TA_B(TENSOR, T, N, INDEX_NBITS)                                      \
  fbgemm_gpu::utils::                                                        \
      TensorAccessorBuilder<T, N, INDEX_NBITS, false, at::DefaultPtrTraits>( \
          #TENSOR, TENSOR)

#define PTA_B(TENSOR, T, N, INDEX_NBITS)                                     \
  fbgemm_gpu::utils::                                                        \
      TensorAccessorBuilder<T, N, INDEX_NBITS, true, at::RestrictPtrTraits>( \
          #TENSOR, TENSOR)

#define PTA_ACC_B(TENSOR, T, N, INDEX_NBITS) \
  fbgemm_gpu::utils::TensorAccessorBuilder<  \
      at::acc_type<T, true>,                 \
      N,                                     \
      INDEX_NBITS,                           \
      true,                                  \
      at::RestrictPtrTraits>(#TENSOR, TENSOR)

//////////////////////////////////////////////////////////////////////////////
// [LEGACY] (Packed) Tensor Accessor Builder Macros
//
// These macros are used to build out *TensorAccessor.  They are LEGACY macros
// for backward compatibility and will be removed once all FBGEMM kernels are
// migrated over to use the FBGEMM_LAUNCH_KERNEL macro.
//////////////////////////////////////////////////////////////////////////////

#ifdef FBGEMM_GPU_MEMCHECK

#define MAKE_TA_WITH_NAME(FUNC_NAME, TENSOR, T, N) \
  TA_B(TENSOR, T, N, 64).build(FUNC_NAME)

#define MAKE_PTA_WITH_NAME(FUNC_NAME, TENSOR, T, N, INDEX_NBITS) \
  PTA_B(TENSOR, T, N, INDEX_NBITS).build(FUNC_NAME)

#define MAKE_PTA_ACC_WITH_NAME(FUNC_NAME, TENSOR, T, N, INDEX_NBITS) \
  PTA_ACC_B(TENSOR, T, N, INDEX_NBITS).build(FUNC_NAME)

#else

#define MAKE_TA_WITH_NAME(FUNC_NAME, TENSOR, T, N) \
  TA_B(TENSOR, T, N, 64).build(#TENSOR)

#define MAKE_PTA_WITH_NAME(FUNC_NAME, TENSOR, T, N, INDEX_NBITS) \
  PTA_B(TENSOR, T, N, INDEX_NBITS).build(#TENSOR)

#define MAKE_PTA_ACC_WITH_NAME(FUNC_NAME, TENSOR, T, N, INDEX_NBITS) \
  PTA_ACC_B(TENSOR, T, N, INDEX_NBITS).build(#TENSOR)

#endif
