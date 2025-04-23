/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>

////////////////////////////////////////////////////////////////////////////////
// Extended TensorAccessor
//
// This file contains TensorAccessor and PackedTensorAccessor implementations
// that are used in FBGEMM_GPU for additional bounds checks that are not
// available in the standard ATen implementation. Using the builder macro
// MAKE_TA_WITH_NAME and MAKE_PTA_WITH_NAME, bounds checks can be enabled using
// the FBGEMM_GPU_MEMCHECK flag.
//
//  https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/TensorAccessor.h
//  https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/TensorBase.h
////////////////////////////////////////////////////////////////////////////////

namespace fbgemm_gpu::utils {

////////////////////////////////////////////////////////////////////////////////
// Pointer Trait Aliases
//
// Map from at:: namespace
////////////////////////////////////////////////////////////////////////////////

template <typename T>
using DefaultPtrTraits = at::DefaultPtrTraits<T>;

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
using RestrictPtrTraits = at::RestrictPtrTraits<T>;
#endif

static constexpr size_t NAME_MAX_LEN = 32;
static constexpr size_t CONTEXT_MAX_LEN = 256;

////////////////////////////////////////////////////////////////////////////////
// String Copy
//
// Implemented bc std::copy does not exist on the device side
////////////////////////////////////////////////////////////////////////////////

namespace {

C10_HOST_DEVICE inline void
copy_str(char* dst, const char* src, const size_t max_len) {
  // If dst is nullptr, then skip.
  if (dst == nullptr) {
    return;
  }

  // If src is nullptr or max_len is zero, then mark empty string and skip.
  if (src == nullptr || max_len == 0) {
    dst[0] = '\0';
    return;
  }

  // Count src buffer length up to max_len
  size_t len = 0;
  for (len = 0; src[len] != 0 && len < max_len; len++) {
    // no action - calculating string length
  }
  len = len < (max_len - 1) ? len : (max_len - 1);

  // Copy src to dst
  for (size_t i = 0; i < len; i++) {
    dst[i] = src[i];
  }
  dst[len] = '\0';
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// TensorAccessor
//
// This is an extension of at::TensorAccessorBase that consolidates some methods
// defined in at::TensorAccessor.
////////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class TensorAccessor : public at::TensorAccessorBase<T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      const PtrType data_,
      const index_t* const sizes_,
      const index_t* const strides_,
      const char* const _name_,
      const char* const _context_)
      : at::TensorAccessorBase<T, N, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_) {
    if (sizes_ && strides_) {
      numel_ = 1;
      for (size_t d = 0; d < N; d++) {
        numel_ += (sizes_[d] - 1) * strides_[d];
      }
    }

    copy_str(name_, _name_, NAME_MAX_LEN);
    copy_str(context_, _context_, CONTEXT_MAX_LEN);
  }

  template <size_t M = N>
  C10_HOST_DEVICE inline auto operator[](const index_t i)
      -> std::
          enable_if_t<(M > 1), TensorAccessor<T, N - 1, PtrTraits, index_t>> {
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1,
        this->name_,
        this->context_);
  }

  template <size_t M = N>
  C10_HOST_DEVICE inline auto operator[](const index_t i) const
      -> std::enable_if_t<
          (M > 1),
          const TensorAccessor<T, N - 1, PtrTraits, index_t>> {
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1,
        this->name_,
        this->context_);
  }

  template <size_t M = N>
  C10_HOST_DEVICE inline auto operator[](const index_t i)
      -> std::enable_if_t<(M == 1), T&> {
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    return this->at(this->strides_[0] * i);
  }

  template <size_t M = N>
  C10_HOST_DEVICE inline auto operator[](const index_t i) const
      -> std::enable_if_t<(M == 1), const T&> {
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    return this->at(this->strides_[0] * i);
  }

  C10_HOST_DEVICE inline auto numel() const {
    return numel_;
  }

  C10_HOST_DEVICE inline T& at(const index_t idx) const {
    if (idx < 0) {
      printf(
          "[%s][Tensor %s] CUDA Kernel Assertion: index is outside of accessor bounds (idx=%ld) < 0\n",
          this->context_,
          this->name_,
          static_cast<int64_t>(idx));
      CUDA_KERNEL_ASSERT_MSG(
          idx >= 0, "index is outside of accessor bounds (idx >= 0)");

    } else if (idx >= numel_) {
      printf(
          "[%s][Tensor %s] CUDA Kernel Assertion: index is outside of accessor bounds (idx=%ld) >= (numel=%ld)\n",
          this->context_,
          this->name_,
          static_cast<int64_t>(idx),
          static_cast<int64_t>(numel_));
      CUDA_KERNEL_ASSERT_MSG(
          idx < numel_, "index is outside of accessor bounds (idx < numel_)");
    }

    return this->data_[idx];
  }

 protected:
  size_t numel_;
  char name_[NAME_MAX_LEN];
  char context_[CONTEXT_MAX_LEN];
};

////////////////////////////////////////////////////////////////////////////////
// PackedTensorAccessor
//
// This is an extension of at::GeneticPackedTensorAccessorBase that consolidates
// some methods defined in at::GeneticPackedTensorAccessor.
//
// `GenericPackedTensorAccessor`s are used on for CUDA `Tensor`s on the host and
// as In contrast to `TensorAccessor`s, they copy the strides and sizes on
// instantiation (on the host) in order to transfer them on the device when
// calling kernels.  On the device, indexing of multidimensional tensors gives
// to `TensorAccessor`s.
////////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class PackedTensorAccessor
    : public at::GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST PackedTensorAccessor(
      const PtrType data_,
      const index_t* const sizes_,
      const index_t* const strides_,
      const char* const _name_,
      const char* const _context_)
      : at::GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_) {
    set_numel_(sizes_, strides_);
    copy_str(name_, _name_, NAME_MAX_LEN);
    copy_str(context_, _context_, CONTEXT_MAX_LEN);
  }

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <
      typename source_index_t,
      class = std::enable_if_t<std::is_same_v<source_index_t, int64_t>>>
  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_,
      const char* const _name_,
      const char* const _context_)
      : at::GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_) {
    set_numel_(sizes_, strides_);
    copy_str(name_, _name_, NAME_MAX_LEN);
    copy_str(context_, _context_, CONTEXT_MAX_LEN);
  }

  template <size_t M = N>
  C10_HOST_DEVICE inline auto operator[](const index_t i)
      -> std::
          enable_if_t<(M > 1), TensorAccessor<T, N - 1, PtrTraits, index_t>> {
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1,
        this->name_,
        this->context_);
  }

  template <size_t M = N>
  C10_HOST_DEVICE inline auto operator[](const index_t i) const
      -> std::enable_if_t<
          (M > 1),
          const TensorAccessor<T, N - 1, PtrTraits, index_t>> {
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1,
        this->name_,
        this->context_);
  }

  template <size_t M = N>
  C10_HOST_DEVICE inline auto operator[](const index_t i)
      -> std::enable_if_t<(M == 1), T&> {
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    return this->at(this->strides_[0] * i);
  }

  template <size_t M = N>
  C10_HOST_DEVICE inline auto operator[](const index_t i) const
      -> std::enable_if_t<(M == 1), const T&> {
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    return this->at(this->strides_[0] * i);
  }

  C10_HOST_DEVICE inline auto numel() const {
    return numel_;
  }

  C10_HOST_DEVICE inline T& at(const index_t idx) const {
    if (idx < 0) {
      printf(
          "[%s][Tensor %s] CUDA Kernel Assertion: index is outside of accessor bounds (idx=%ld) < 0\n",
          this->context_,
          this->name_,
          static_cast<int64_t>(idx));
      CUDA_KERNEL_ASSERT_MSG(
          idx >= 0, "index is outside of accessor bounds (idx >= 0)");

    } else if (idx >= numel_) {
      printf(
          "[%s][Tensor %s] CUDA Kernel Assertion: index is outside of accessor bounds (idx=%ld) >= (numel=%ld)\n",
          this->context_,
          this->name_,
          static_cast<int64_t>(idx),
          static_cast<int64_t>(numel_));
      CUDA_KERNEL_ASSERT_MSG(
          idx < numel_, "index is outside of accessor bounds (idx < numel_)");
    }

    return this->data_[idx];
  }

  C10_HOST inline auto transpose(const index_t dim1, const index_t dim2) const {
    if constexpr (N > 1) {
      this->bounds_check_(dim1);
      this->bounds_check_(dim2);
      auto result = PackedTensorAccessor<T, N, PtrTraits, index_t>(
          this->data_, this->sizes_, this->strides_, name_, context_);
      std::swap(result.strides_[dim1], result.strides_[dim2]);
      std::swap(result.sizes_[dim1], result.sizes_[dim2]);
      return result;

    } else {
      this->bounds_check_(dim1);
      this->bounds_check_(dim2);
      return PackedTensorAccessor<T, 1, PtrTraits, index_t>(
          this->data_, this->sizes_, this->strides_, name_, context_);
    }
  }

 protected:
  size_t numel_;
  char name_[NAME_MAX_LEN];
  char context_[CONTEXT_MAX_LEN];

  template <typename source_index_t>
  C10_HOST_DEVICE inline void set_numel_(
      const source_index_t* sizes_,
      const source_index_t* strides_) {
    numel_ = 0;
    if (sizes_ && strides_) {
      numel_ = 1;
      for (size_t d = 0; d < N; d++) {
        numel_ += (sizes_[d] - 1) * strides_[d];
      }
    }
  }

  C10_HOST inline void bounds_check_(const index_t i) const {
    TORCH_CHECK_INDEX(
        0 <= i && i < index_t{N},
        "[",
        context_,
        "][",
        name_,
        "]: ",
        "Index ",
        i,
        " is not within bounds of a tensor of dimension ",
        N);
  }
};

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor32 = PackedTensorAccessor<T, N, PtrTraits, int32_t>;

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor64 = PackedTensorAccessor<T, N, PtrTraits, int64_t>;

////////////////////////////////////////////////////////////////////////////////
// NOTE: The following code will be removed once all FBGEMM kernels are migrated
// over to the FBGEMM_LAUNCH_KERNEL macro.
////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline at::ScalarType scalar_type_for_2() {
#define TYPE_CASE(U, name)              \
  if constexpr (std::is_same_v<T, U>) { \
    return at::ScalarType::name;        \
  }

  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(TYPE_CASE)

#undef TYPE_CASE

  return at::ScalarType::Undefined;
}

////////////////////////////////////////////////////////////////////////////////
// Tensor Checks with Descriptive Error Messages
////////////////////////////////////////////////////////////////////////////////

template <size_t N>
inline void check_tensor_dim(
    const at::TensorBase& tensor
#ifdef FBGEMM_GPU_MEMCHECK
    ,
    const char* const func_name,
    const char* const tensor_name
#endif
) {
  TORCH_CHECK(
      tensor.dim() == N,
#ifdef FBGEMM_GPU_MEMCHECK
      "[ ",
      func_name,
      " ]: ",
#endif
      "Expected tensor ",
#ifdef FBGEMM_GPU_MEMCHECK
      "'",
      tensor_name,
      "' ",
#endif
      "to have ",
      N,
      " dims, but found ",
      tensor.dim(),
      " instead!");
}

template <typename T>
inline void check_scalar_type(
    const at::TensorBase& tensor
#ifdef FBGEMM_GPU_MEMCHECK
    ,
    const char* const func_name,
    const char* const tensor_name
#endif
) {
  const auto expected_type = scalar_type_for_2<T>();

  TORCH_CHECK(
      tensor.scalar_type() == expected_type ||
          (isQIntType(tensor.scalar_type()) &&
           toUnderlying(tensor.scalar_type()) == expected_type),
#ifdef FBGEMM_GPU_MEMCHECK
      "[ ",
      func_name,
      " ]: ",
#endif
      "Expected tensor ",
#ifdef FBGEMM_GPU_MEMCHECK
      "'",
      tensor_name,
      "' ",
#endif
      "to have scalar type ",
      expected_type,
      ", but found ",
      tensor.scalar_type(),
      " instead!");
}

} // namespace fbgemm_gpu::utils

#ifdef FBGEMM_GPU_MEMCHECK
namespace pta = fbgemm_gpu::utils;
#else
namespace pta = at;
template <
    typename T,
    size_t N,
    template <typename U>
    class PtrTraits,
    typename index_t>
using PackedTensorAccessor =
    at::GenericPackedTensorAccessor<T, N, PtrTraits, index_t>;
#endif

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = at::DefaultPtrTraits,
    typename index_t = int64_t>
inline pta::TensorAccessor<T, N, PtrTraits, index_t> make_tensor_accessor(
#ifdef FBGEMM_GPU_MEMCHECK
    const at::Tensor& tensor,
    const char* const tensor_name,
    const char* const func_name) {
#else
    const at::Tensor& tensor) {
#endif

  static_assert(
      N > 0,
      "Accessor is used for indexing tensor, for scalars use *data_ptr<T>()");

  // If the tensor is defined, then check the tensor dimensions and scalar type
  // before building and returning the accessor.
  if (tensor.defined()) {
    fbgemm_gpu::utils::check_tensor_dim<N>(
        tensor
#ifdef FBGEMM_GPU_MEMCHECK
        ,
        func_name,
        tensor_name
#endif
    );

    fbgemm_gpu::utils::check_scalar_type<T>(
        tensor
#ifdef FBGEMM_GPU_MEMCHECK
        ,
        func_name,
        tensor_name
#endif
    );

#ifdef FBGEMM_GPU_MEMCHECK
    return fbgemm_gpu::utils::TensorAccessor<T, N, PtrTraits, index_t>(
        static_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr<T>()),
        tensor.sizes().data(),
        tensor.strides().data(),
        tensor_name,
        func_name);
#else
    return tensor.accessor<T, N>();
#endif

  } else {
    // Else, just return a null tensor accessor - this is useful for cases where
    // optionals are not used.

#ifdef FBGEMM_GPU_MEMCHECK
    return fbgemm_gpu::utils::TensorAccessor<T, N, PtrTraits, index_t>(
        nullptr, nullptr, nullptr, tensor_name, func_name);
#else
    return at::TensorAccessor<T, N, PtrTraits, index_t>(
        nullptr, nullptr, nullptr);
#endif
  }
}

////////////////////////////////////////////////////////////////////////////////
// *TensorAccessor Builder Functions
////////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = at::DefaultPtrTraits,
    typename index_t = int64_t>
inline pta::PackedTensorAccessor<T, N, PtrTraits, index_t>
make_generic_packed_tensor_accessor(
#ifdef FBGEMM_GPU_MEMCHECK
    const at::Tensor& tensor,
    const char* const tensor_name,
    const char* const func_name) {
#else
    const at::Tensor& tensor) {
#endif
  static_assert(
      N > 0,
      "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");

  fbgemm_gpu::utils::check_tensor_dim<N>(
      tensor
#ifdef FBGEMM_GPU_MEMCHECK
      ,
      func_name,
      tensor_name
#endif
  );

  fbgemm_gpu::utils::check_scalar_type<T>(
      tensor
#ifdef FBGEMM_GPU_MEMCHECK
      ,
      func_name,
      tensor_name
#endif
  );

#ifdef FBGEMM_GPU_MEMCHECK
  return fbgemm_gpu::utils::PackedTensorAccessor<T, N, PtrTraits, index_t>(
      static_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr<T>()),
      tensor.sizes().data(),
      tensor.strides().data(),
      tensor_name,
      func_name);
#else
  return tensor.generic_packed_accessor<T, N, PtrTraits, index_t>();
#endif
}

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
pta::PackedTensorAccessor32<T, N, PtrTraits> make_packed_tensor_accessor32(
#ifdef FBGEMM_GPU_MEMCHECK
    const at::Tensor& tensor,
    const char* const tensor_name,
    const char* const func_name) {
#else
    const at::Tensor& tensor) {
#endif

  TORCH_CHECK(
      tensor.numel() <=
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
#ifdef FBGEMM_GPU_MEMCHECK
      "[ ",
      func_name,
      " ]: Tensor ",
      tensor_name,
      " ",
#endif
      "numel needs to be smaller than int32_t max; otherwise, please use packed_accessor64");

#ifdef FBGEMM_GPU_MEMCHECK
  return make_generic_packed_tensor_accessor<T, N, PtrTraits, int32_t>(
      tensor, tensor_name, func_name);
#else
  return tensor.packed_accessor32<T, N, PtrTraits>();
#endif
}

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
pta::PackedTensorAccessor64<T, N, PtrTraits> make_packed_tensor_accessor64(
#ifdef FBGEMM_GPU_MEMCHECK
    const at::Tensor& tensor,
    const char* const tensor_name,
    const char* const func_name) {
#else
    const at::Tensor& tensor) {
#endif

#ifdef FBGEMM_GPU_MEMCHECK
  return make_generic_packed_tensor_accessor<T, N, PtrTraits, int64_t>(
      tensor, tensor_name, func_name);
#else
  return tensor.packed_accessor64<T, N, PtrTraits>();
#endif
}

////////////////////////////////////////////////////////////////////////////////
// *TensorAccessor Builder Macros
////////////////////////////////////////////////////////////////////////////////

#ifdef FBGEMM_GPU_MEMCHECK
#define MAKE_TA_WITH_NAME(FUNC_NAME, TENSOR, T, N) \
  make_tensor_accessor<T, N>(TENSOR, #TENSOR, FUNC_NAME)

#define MAKE_PACKED_TENSOR_ACCESSOR_BASE(                     \
    FUNC_NAME, TENSOR, T, N, PTR_TRAITS, INDEX_NBITS)         \
  make_packed_tensor_accessor##INDEX_NBITS<T, N, PTR_TRAITS>( \
      TENSOR, #TENSOR, FUNC_NAME)

#define MAKE_PACKED_TENSOR_ACCESSOR_ACC_TYPE_BASE(    \
    FUNC_NAME, TENSOR, T, N, PTR_TRAITS, INDEX_NBITS) \
  make_packed_tensor_accessor##INDEX_NBITS<           \
      at::acc_type<T, true>,                          \
      N,                                              \
      PTR_TRAITS>(TENSOR, #TENSOR, FUNC_NAME)

#else
#define MAKE_TA_WITH_NAME(FUNC_NAME, TENSOR, T, N) \
  make_tensor_accessor<T, N>(TENSOR)

#define MAKE_PACKED_TENSOR_ACCESSOR_BASE(             \
    FUNC_NAME, TENSOR, T, N, PTR_TRAITS, INDEX_NBITS) \
  make_packed_tensor_accessor##INDEX_NBITS<T, N, PTR_TRAITS>(TENSOR)

#define MAKE_PACKED_TENSOR_ACCESSOR_ACC_TYPE_BASE(    \
    FUNC_NAME, TENSOR, T, N, PTR_TRAITS, INDEX_NBITS) \
  make_packed_tensor_accessor##INDEX_NBITS<           \
      at::acc_type<T, true>,                          \
      N,                                              \
      PTR_TRAITS>(TENSOR)
#endif

#define MAKE_PTA_WITH_NAME(FUNC_NAME, TENSOR, T, N, INDEX_NBITS) \
  MAKE_PACKED_TENSOR_ACCESSOR_BASE(                              \
      FUNC_NAME, TENSOR, T, N, at::RestrictPtrTraits, INDEX_NBITS)

#define MAKE_PTA_ACC_WITH_NAME(FUNC_NAME, TENSOR, T, N, INDEX_NBITS) \
  MAKE_PACKED_TENSOR_ACCESSOR_ACC_TYPE_BASE(                         \
      FUNC_NAME, TENSOR, T, N, at::RestrictPtrTraits, INDEX_NBITS)

// !! Please do not modify the overflow_safe_int_t value unless you absolutely
// understand what you are doing !!
//
// An integer datatype for preventing the integer overflow problem
using overflow_safe_int_t = int64_t;
