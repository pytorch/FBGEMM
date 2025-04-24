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
// Extended *TensorAccessor
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
// This is an extension of at::TensorAccessorBase that consolidates template
// specializations of operator[] defined in at::TensorAccessor.
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
// template specializations of operator[] defined in
// at::GeneticPackedTensorAccessor.
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
        "] [",
        name_,
        "]: ",
        "Dimension ",
        i,
        " is not within bounds of tensor of ",
        N,
        "dimension(s).");
  }
};

////////////////////////////////////////////////////////////////////////////////
// PackedTensorAccessor Aliases
////////////////////////////////////////////////////////////////////////////////

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

} // namespace fbgemm_gpu::utils

////////////////////////////////////////////////////////////////////////////////
// *TensorAccessor Selector
//
// Select fbgemm_gpu::utils::*TensorAccessor or at::*TensorAccessor based on
// FBGEMM_GPU_MEMCHECK flag
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
// Integer datatype for preventing the integer overflow problem
//
// NOTE: !! Please do not modify the overflow_safe_int_t value unless you
// absolutely　understand what you are doing !!
////////////////////////////////////////////////////////////////////////////////

using overflow_safe_int_t = int64_t;
