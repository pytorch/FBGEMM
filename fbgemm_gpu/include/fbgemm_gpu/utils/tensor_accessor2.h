/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <cstddef>
#include <cstdint>

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

template <typename T>
using DefaultPtrTraits = at::DefaultPtrTraits<T>;

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
using RestrictPtrTraits = at::RestrictPtrTraits<T>;
#endif

static constexpr size_t NAME_MAX_LEN = 32;
static constexpr size_t CONTEXT_MAX_LEN = 256;

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

  C10_HOST_DEVICE T& at(const index_t idx) const {
    if (idx < 0) {
      printf(
          "[%s][Tensor %s] ERROR: (idx=%ld) < 0\n",
          this->context_,
          this->name_,
          static_cast<int64_t>(idx));
      CUDA_KERNEL_ASSERT(idx >= 0);

    } else if (idx >= numel_) {
      printf(
          "[%s][Tensor %s] ERROR: (idx=%ld) >= (numel=%ld)\n",
          this->context_,
          this->name_,
          static_cast<int64_t>(idx),
          static_cast<int64_t>(numel_));
      CUDA_KERNEL_ASSERT(idx < numel_);
    }

    return this->data_[idx];
  }

 protected:
  size_t numel_;
  char name_[NAME_MAX_LEN];
  char context_[CONTEXT_MAX_LEN];
};

} // namespace fbgemm_gpu::utils
